from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra.runtime import runtime_from_env_or_module
from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_models.registry import get_model_config, model_names
from pricing_pipeline.workbench.artifacts import CandidateBundle, load_candidate_bundle
from pricing_pipeline.workbench.airflow import AirflowClient


_FRIENDLY_COLUMNS = [
    "Package",
    "Fitted",
    "Data through",
    "Parent",
    "State",
    "Baseline pooled CV deviance",
    "Editor train delta",
    "Editor",
]
_ARTIFACT_FIELDS = (
    "candidate_artifact_path",
    "candidate_artifact_sha256",
    "candidate_artifact_format",
    "candidate_artifact_size_bytes",
    "candidate_python_version",
    "candidate_superglm_version",
    "model_source_sha256",
)


class CandidateLineageError(RuntimeError):
    """Raised when a package cannot resolve one trusted candidate run."""


def _reviewed_champion_evidence(revision_metadata_json: str | None) -> dict[str, Any]:
    if revision_metadata_json is None:
        raise CandidateLineageError("editor child package has no champion comparison metadata")
    try:
        revision_metadata = json.loads(revision_metadata_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CandidateLineageError("editor child package has invalid revision metadata") from exc
    if not isinstance(revision_metadata, dict):
        raise CandidateLineageError("editor child revision metadata must be a JSON object")
    comparison = revision_metadata.get("champion_comparison")
    if not isinstance(comparison, dict):
        raise CandidateLineageError("editor child package has no champion comparison metadata")

    status = comparison.get("status")
    if status not in {"COMPARED", "NO_CHAMPION", "UNAVAILABLE"}:
        raise CandidateLineageError("editor child package has invalid champion comparison status")
    raw_slot = comparison.get("deployment_slot")
    slot = str(raw_slot or "").strip().upper()
    if not slot:
        raise CandidateLineageError("editor child champion comparison has no deployment slot")

    raw_rate_package_id = comparison.get("rate_package_id")
    if status == "NO_CHAMPION":
        if raw_rate_package_id is not None:
            raise CandidateLineageError(
                "NO_CHAMPION comparison must not identify a rate package"
            )
        rate_package_id = None
    else:
        if isinstance(raw_rate_package_id, bool):
            raise CandidateLineageError("champion comparison rate package ID is invalid")
        try:
            rate_package_id = int(raw_rate_package_id)
        except (TypeError, ValueError) as exc:
            raise CandidateLineageError(
                "champion comparison rate package ID is invalid"
            ) from exc
        if rate_package_id <= 0:
            raise CandidateLineageError("champion comparison rate package ID is invalid")

    raw_reason = comparison.get("reason")
    reason = None if raw_reason is None else str(raw_reason).strip() or None
    if status == "UNAVAILABLE" and reason is None:
        raise CandidateLineageError("unavailable champion comparison must include a reason")
    expected_available = status == "COMPARED"
    if "available" in comparison and comparison["available"] is not expected_available:
        raise CandidateLineageError("champion comparison availability is inconsistent")
    return {
        "reviewed_champion_status": status,
        "reviewed_champion_rate_package_id": rate_package_id,
        "reviewed_deployment_slot": slot,
        "reviewed_champion_reason": reason,
    }


@dataclass
class Candidate:
    workbench: "Workbench"
    model_name: str
    package_version: int
    rate_package_id: int
    parent_rate_package_id: int | None
    model_run_id: int
    bundle: CandidateBundle
    technical: dict[str, Any]
    editor_session: Any | None = field(default=None, init=False, repr=False)
    editor_widget: Any | None = field(default=None, init=False, repr=False)

    def editor(self):
        """Open this candidate in one retained, live SuperGLM editor session."""
        if self.editor_session is None:
            self.editor_session = self.workbench.create_editor_session(self.bundle)
            self.editor_widget = self.editor_session.widget()
        return self.editor_widget

    def submit_edits(self, *, reason: str):
        cleaned_reason = str(reason).strip()
        if not cleaned_reason:
            raise ValueError("A non-empty reason is required to submit editor changes")
        if self.editor_session is None or self.editor_widget is None:
            raise RuntimeError("Open the candidate editor before submitting edits")
        from pricing_pipeline.workbench.submission import create_editor_submission

        return create_editor_submission(
            self,
            editor_session=self.editor_session,
            reason=cleaned_reason,
            airflow_client=self.workbench.airflow_client,
        )

    def close_editor(self) -> None:
        if self.editor_widget is not None:
            close = getattr(self.editor_widget, "close", None)
            if callable(close):
                close()
        self.editor_widget = None
        self.editor_session = None


class Workbench:
    def __init__(
        self,
        *,
        engine,
        settings: Settings,
        config_loader: Callable[[str], Any] = get_model_config,
        model_names_loader: Callable[[], tuple[str, ...]] = model_names,
        editor_session_factory: Callable[..., Any] | None = None,
        airflow_client: AirflowClient | Any | None = None,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self._config_loader = config_loader
        self._model_names_loader = model_names_loader
        self._editor_session_factory = editor_session_factory
        self._airflow_client = airflow_client

    @classmethod
    def from_runtime(cls, runtime_module: str | None = None) -> "Workbench":
        runtime = runtime_from_env_or_module(runtime_module)
        return cls(engine=runtime.get_engine(), settings=runtime.settings)

    @property
    def airflow_client(self) -> AirflowClient | Any:
        if self._airflow_client is None:
            self._airflow_client = AirflowClient(
                self.settings.airflow_api_url,
                token=self.settings.airflow_api_token,
                username=self.settings.airflow_api_username,
                password=self.settings.airflow_api_password,
            )
        return self._airflow_client

    def create_editor_session(self, bundle: CandidateBundle):
        factory = self._editor_session_factory
        if factory is None:
            from superglm.editor import EditorSession

            factory = EditorSession.from_model
        return factory(
            bundle.fitted_model,
            train_data=(bundle.X, bundle.y, bundle.sample_weight, bundle.offset),
            cv_report=bundle.cv_report,
        )

    def model_config(self, model_name: str):
        return self._config_loader(self._required_model_name(model_name))

    def models(self) -> list[str]:
        """Return the logical model names discovered by the model registry."""
        return list(self._model_names_loader())

    def current_champion_rate_package_id(
        self,
        model_name: str,
        *,
        deployment_slot: str | None = None,
    ) -> int | None:
        """Read the active champion at the moment an analyst requests deployment."""
        model_name = self._required_model_name(model_name)
        raw_slot = (
            self._config_loader(model_name).deployment_slot
            if deployment_slot is None
            else deployment_slot
        )
        slot = str(raw_slot).strip().upper()
        if not slot:
            raise ValueError("deployment_slot is required")
        schemas = schema_names_from_connectable(self.engine)
        query = text(
            f"""
            SELECT deployment.rate_package_id
            FROM {schemas.pricing}.PRICING_MODEL AS pm
            JOIN {schemas.pricing}.PRICING_MODEL_DEPLOYMENT AS deployment
              ON deployment.model_id = pm.model_id
            WHERE pm.model_name = :model_name
              AND deployment.deployment_slot = :deployment_slot
              AND deployment.effective_to_ts IS NULL
            """
        )
        with self.engine.begin() as connection:
            value = connection.execute(
                query,
                {
                    "model_name": model_name,
                    "deployment_slot": slot,
                },
            ).scalar_one_or_none()
        return None if value is None else int(value)

    def resolve_editor_publication(self, submission) -> dict[str, Any]:
        schemas = schema_names_from_connectable(self.engine)
        export_id = f"editor__{submission.submission_id.replace('-', '_')}"
        query = text(
            f"""
            SELECT
                pm.model_name,
                rp.rate_package_id,
                rp.package_version,
                rp.package_status,
                rp.parent_rate_package_id,
                rp.revision_metadata_json,
                mr.model_run_id,
                mr.run_status
            FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp
            JOIN {schemas.pricing}.PRICING_MODEL AS pm
              ON pm.model_id = rp.model_id
            JOIN {schemas.pricing}.MODEL_RUN AS mr
              ON mr.rate_package_id = rp.rate_package_id
            WHERE pm.model_name = :model_name
              AND rp.parent_rate_package_id = :parent_rate_package_id
              AND rp.source_export_id = :export_id
            """
        )
        with self.engine.begin() as connection:
            rows = list(
                connection.execute(
                    query,
                    {
                        "model_name": submission.model_name,
                        "parent_rate_package_id": submission.parent_rate_package_id,
                        "export_id": export_id,
                    },
                )
                .mappings()
                .all()
            )
        if len(rows) != 1:
            raise CandidateLineageError(
                "successful editor publication must resolve exactly one child package/run; "
                f"found {len(rows)}"
            )
        row = dict(rows[0])
        if row.get("package_status") != "PUBLISHED" or row.get("run_status") != "SUCCESS":
            raise CandidateLineageError("editor child package/run is not successfully published")
        row.update(_reviewed_champion_evidence(row.get("revision_metadata_json")))
        return row

    def candidates(
        self,
        model_name: str,
        *,
        deployment_slot: str | None = None,
        technical: bool = False,
    ) -> pd.DataFrame:
        model_name = self._required_model_name(model_name)
        slot = deployment_slot or self._config_loader(model_name).deployment_slot
        rows = [dict(row) for row in self._candidate_rows(model_name, slot)]
        if technical:
            return pd.DataFrame(rows)
        friendly = [self._friendly_row(row, deployment_slot=slot) for row in rows]
        return pd.DataFrame(friendly, columns=_FRIENDLY_COLUMNS)

    def open(self, model_name: str, *, package_version: int) -> Candidate:
        model_name = self._required_model_name(model_name)
        version = int(package_version)
        rows = [dict(row) for row in self._resolve_candidate_rows(model_name, version)]
        if len(rows) != 1:
            raise CandidateLineageError(
                f"{model_name} package {version} must resolve exactly one successful MODEL_RUN; "
                f"found {len(rows)}"
            )
        row = rows[0]
        if str(row.get("run_status") or "").upper() != "SUCCESS" or not self._editor_ready(row):
            raise CandidateLineageError(
                f"{model_name} package {version} has no verified candidate artifact"
            )
        bundle = load_candidate_bundle(
            row["candidate_artifact_path"],
            expected_sha256=row["candidate_artifact_sha256"],
            expected_size_bytes=int(row["candidate_artifact_size_bytes"]),
            expected_format=row["candidate_artifact_format"],
            expected_python_version=row["candidate_python_version"],
            expected_superglm_version=row["candidate_superglm_version"],
            allowed_root=Path(self.settings.workbench_artifact_root),
        )
        if bundle.manifest_id != row.get("manifest_id"):
            raise CandidateLineageError("candidate bundle manifest_id does not match SQL lineage")
        if bundle.split_set_id != row.get("split_set_id"):
            raise CandidateLineageError("candidate bundle split_set_id does not match SQL lineage")
        if bundle.model_source_sha256 != row.get("model_source_sha256"):
            raise CandidateLineageError(
                "candidate bundle model source hash does not match SQL lineage"
            )
        return Candidate(
            workbench=self,
            model_name=model_name,
            package_version=version,
            rate_package_id=int(row["rate_package_id"]),
            parent_rate_package_id=(
                None
                if row.get("parent_rate_package_id") is None
                else int(row["parent_rate_package_id"])
            ),
            model_run_id=int(row["model_run_id"]),
            bundle=bundle,
            technical=row,
        )

    def _resolve_candidate_rows(
        self,
        model_name: str,
        package_version: int,
    ) -> list[Mapping[str, Any]]:
        slot = self._config_loader(model_name).deployment_slot
        return [
            row
            for row in self._candidate_rows(model_name, slot)
            if int(row["package_version"]) == package_version
        ]

    def _candidate_rows(
        self,
        model_name: str,
        deployment_slot: str,
    ) -> list[Mapping[str, Any]]:
        schemas = schema_names_from_connectable(self.engine)
        query = text(
            f"""
            SELECT
                pm.model_name,
                rp.package_version,
                rp.rate_package_id,
                rp.parent_rate_package_id,
                parent_rp.package_version AS parent_package_version,
                mr.model_run_id,
                mr.run_status,
                mr.completed_ts,
                mr.manifest_id,
                split_link.split_set_id,
                mr.candidate_artifact_path,
                mr.candidate_artifact_sha256,
                mr.candidate_artifact_format,
                mr.candidate_artifact_size_bytes,
                mr.candidate_python_version,
                mr.candidate_superglm_version,
                mr.model_source_sha256,
                manifest.data_as_of_date,
                deployment.rate_package_id AS current_rate_package_id,
                COALESCE(cv.metric_value, parent_cv.metric_value) AS baseline_cv_deviance,
                cv.metric_scope AS baseline_metric_scope,
                CASE
                    WHEN rp.parent_rate_package_id IS NOT NULL
                     AND (
                         cv.metric_scope = 'inherited_cv'
                         OR (cv.metric_value IS NULL AND parent_cv.metric_value IS NOT NULL)
                     )
                    THEN 1 ELSE 0
                END AS baseline_is_parent,
                editor_delta.metric_value AS editor_training_delta
            FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp
            JOIN {schemas.pricing}.PRICING_MODEL AS pm
              ON pm.model_id = rp.model_id
            LEFT JOIN {schemas.pricing}.PRICING_RATE_PACKAGE AS parent_rp
              ON parent_rp.rate_package_id = rp.parent_rate_package_id
            LEFT JOIN {schemas.pricing}.MODEL_RUN AS mr
              ON mr.rate_package_id = rp.rate_package_id
            LEFT JOIN {schemas.pricing}.MODEL_RUN AS parent_mr
              ON parent_mr.rate_package_id = rp.parent_rate_package_id
            LEFT JOIN {schemas.pricing}.DATASET_MANIFEST AS manifest
              ON manifest.manifest_id = mr.manifest_id
            LEFT JOIN {schemas.mlops}.MODEL_RUN_SPLIT_SET AS split_link
              ON split_link.model_run_id = mr.model_run_id
             AND split_link.manifest_id = mr.manifest_id
             AND split_link.dataset_role = 'training'
             AND split_link.split_role = 'validation'
            LEFT JOIN {schemas.mlops}.MODEL_RUN_METRIC AS cv
              ON cv.model_run_id = mr.model_run_id
             AND cv.metric_name = 'cv_pooled_deviance'
            LEFT JOIN {schemas.mlops}.MODEL_RUN_METRIC AS parent_cv
              ON parent_cv.model_run_id = parent_mr.model_run_id
             AND parent_cv.metric_name = 'cv_pooled_deviance'
            LEFT JOIN {schemas.mlops}.MODEL_RUN_METRIC AS editor_delta
              ON editor_delta.model_run_id = mr.model_run_id
             AND editor_delta.metric_name = 'editor_training_deviance_delta'
            LEFT JOIN {schemas.pricing}.PRICING_MODEL_DEPLOYMENT AS deployment
              ON deployment.model_id = pm.model_id
             AND deployment.deployment_slot = :deployment_slot
             AND deployment.effective_to_ts IS NULL
            WHERE pm.model_name = :model_name
            ORDER BY rp.package_version DESC
            """
        )
        with self.engine.begin() as connection:
            return list(
                connection.execute(
                    query,
                    {
                        "model_name": model_name,
                        "deployment_slot": deployment_slot,
                    },
                )
                .mappings()
                .all()
            )

    @staticmethod
    def _required_model_name(model_name: str) -> str:
        cleaned = str(model_name).strip()
        if not cleaned:
            raise ValueError("model_name is required")
        return cleaned

    @staticmethod
    def _editor_ready(row: Mapping[str, Any]) -> bool:
        return all(row.get(field_name) is not None for field_name in _ARTIFACT_FIELDS)

    def _friendly_row(
        self,
        row: Mapping[str, Any],
        *,
        deployment_slot: str,
    ) -> dict[str, Any]:
        is_current = row.get("rate_package_id") == row.get("current_rate_package_id")
        is_edited = row.get("parent_rate_package_id") is not None
        if is_current:
            state = f"Champion in {deployment_slot}"
        elif is_edited:
            state = "Edited candidate"
        else:
            state = "Candidate"
        baseline = row.get("baseline_cv_deviance")
        if baseline is not None and bool(row.get("baseline_is_parent")):
            baseline = f"parent: {float(baseline):.3f}"
        return {
            "Package": int(row["package_version"]),
            "Fitted": row.get("completed_ts"),
            "Data through": row.get("data_as_of_date"),
            "Parent": row.get("parent_package_version"),
            "State": state,
            "Baseline pooled CV deviance": baseline,
            "Editor train delta": row.get("editor_training_delta"),
            "Editor": "Ready" if self._editor_ready(row) else "Unavailable",
        }
