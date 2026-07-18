from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.workbench.artifacts import CandidateBundle, load_candidate_bundle


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
    "model_version",
    "export_id",
    "candidate_artifact_path",
    "candidate_artifact_sha256",
    "candidate_artifact_format",
    "candidate_artifact_size_bytes",
    "candidate_python_version",
    "candidate_superglm_version",
    "candidate_superglm_git_sha",
    "model_source_sha256",
)


class CandidateLineageError(RuntimeError):
    """Raised when a package cannot resolve one trusted candidate run."""


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


class Workbench:
    def __init__(
        self,
        *,
        engine,
        settings: Settings,
        model_config: ModelBuildConfig,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.model_config = model_config

    def _required_model_name(self, model_name: str) -> str:
        cleaned = str(model_name).strip()
        if not cleaned:
            raise ValueError("model_name is required")
        if cleaned != self.model_config.model_name:
            raise ValueError(
                f"workbench is bound to model {self.model_config.model_name!r}, not {cleaned!r}"
            )
        return cleaned

    def candidates(
        self,
        model_name: str,
        *,
        deployment_slot: str | None = None,
        technical: bool = False,
    ) -> pd.DataFrame:
        model_name = self._required_model_name(model_name)
        slot = deployment_slot or self.model_config.deployment_slot
        rows = [dict(row) for row in self._candidate_rows(model_name, slot)]
        if technical:
            return pd.DataFrame(rows)
        friendly = [self._friendly_row(row, deployment_slot=slot) for row in rows]
        return pd.DataFrame(friendly, columns=_FRIENDLY_COLUMNS)

    def open(self, model_name: str, *, package_version: int) -> Candidate:
        model_name = self._required_model_name(model_name)
        version = int(package_version)
        deployment_slot = self.model_config.deployment_slot
        rows = [
            dict(row)
            for row in self._candidate_rows(
                model_name,
                deployment_slot,
                package_version=version,
            )
        ]
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
            expected_superglm_git_sha=row["candidate_superglm_git_sha"],
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
        if bundle.model_frame_sha256 != row.get("model_frame_sha256"):
            raise CandidateLineageError(
                "candidate bundle model_frame_sha256 does not match SQL lineage"
            )
        expected_identity = {
            "model_name": model_name,
            "model_version": row.get("model_version"),
            "export_id": row.get("export_id"),
        }
        for field_name, expected_value in expected_identity.items():
            if getattr(bundle, field_name) != expected_value:
                raise CandidateLineageError(
                    f"candidate bundle {field_name} does not match SQL lineage"
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

    def _candidate_rows(
        self,
        model_name: str,
        deployment_slot: str,
        *,
        package_version: int | None = None,
    ) -> list[Mapping[str, Any]]:
        schemas = schema_names_from_connectable(self.engine)
        package_filter = (
            "\n              AND rp.package_version = :package_version"
            if package_version is not None
            else ""
        )
        query = text(
            f"""
            SELECT
                pm.model_id,
                pm.model_name,
                mr.model_version,
                mr.export_id,
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
                mr.candidate_superglm_git_sha,
                mr.model_source_sha256,
                manifest.model_frame_sha256,
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
            WHERE pm.model_name = :model_name{package_filter}
            ORDER BY rp.package_version DESC
            """
        )
        params: dict[str, Any] = {
            "model_name": model_name,
            "deployment_slot": deployment_slot,
        }
        if package_version is not None:
            params["package_version"] = package_version
        with self.engine.begin() as connection:
            return list(
                connection.execute(
                    query,
                    params,
                )
                .mappings()
                .all()
            )

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
