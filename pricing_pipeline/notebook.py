"""Small, synchronous entry points for pricing-model notebooks.

The notebook owns model and data decisions.  These helpers own generated SQL
identifiers, audit records, artifact locations, and publication plumbing.
"""

from __future__ import annotations

import getpass
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlalchemy import text

from pricing_pipeline.build_identity import create_build_identity, stable_build_export_id
from pricing_pipeline.data.manifest import (
    ModelFrameManifestSpec,
    validation_split_indices,
)
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra.offline_sqlite import open_offline_sqlite
from pricing_pipeline.infra.runtime import runtime_from_env_or_module
from pricing_pipeline.modeling.standard_superglm import (
    ModelInputs,
    canonical_row_identity_index,
    run_standard_superglm_build,
)
from pricing_pipeline.models.config import ModelBuildConfig, ValidationSplitConfig
from pricing_pipeline.models.spec import ApprovedModelBuild
from pricing_pipeline.orchestration.publish_completed_build import (
    CompletedModelPublishResult,
    publish_completed_model_build,
)
from pricing_pipeline.publishing.model_registry import (
    register_pricing_model,
)
from pricing_pipeline.publishing.model_versions import resolve_model_version_for_export
from pricing_pipeline.publishing.naming import clean_identifier
from pricing_pipeline.publishing.deployment import deploy_rate_package
from pricing_pipeline.publishing.editor_candidate import publish_editor_submission
from pricing_pipeline.publishing.sqlite_notebook import (
    publish_sqlite_candidate,
    register_sqlite_model,
    resolve_sqlite_model_version,
)
from pricing_pipeline.publishing.superglm_publication_receipt import OffsetExportContract
from pricing_pipeline.workbench.core import Candidate, Workbench
from pricing_pipeline.workbench.submission import save_editor_submission


@dataclass(frozen=True)
class NotebookContext:
    engine: Any
    settings: Settings
    mode: str
    write_allowed: bool
    destination: str
    database_paths: Mapping[str, Path] = field(default_factory=dict)

    def require_write(self, operation: str) -> None:
        """Stop remote mutations until the notebook explicitly enables them."""
        if not self.write_allowed:
            raise PermissionError(
                f"Remote writes are disabled for {operation}. Confirm "
                "EXPECTED_REMOTE_DATABASE and set ALLOW_REMOTE_WRITES = True."
            )


@dataclass(frozen=True)
class PricingModelSpec:
    name: str
    label: str
    target: str
    model_type: str
    deployment_slot: str
    features: tuple[str, ...]
    dataset_name: str
    source_system: str
    pk_columns: tuple[str, ...]
    validation: ValidationSplitConfig = ValidationSplitConfig.kfold()
    offset_column: str | None = None
    offset_source_column: str | None = None
    offset_label: str | None = None
    sample_weight_column: str | None = None
    export_weight_column: str | None = None
    data_as_of_column: str | None = None
    scoring: tuple[str, ...] = ("deviance", "nll", "gini")
    fit_mode: str = "fit_reml"

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "label",
            "target",
            "model_type",
            "dataset_name",
            "source_system",
            "fit_mode",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "deployment_slot",
            _required_text(self.deployment_slot, "deployment_slot").upper(),
        )
        if self.fit_mode not in {"fit", "fit_reml"}:
            raise ValueError("fit_mode must be 'fit' or 'fit_reml'")
        object.__setattr__(
            self,
            "features",
            tuple(_required_text(value, "features") for value in self.features),
        )
        object.__setattr__(
            self,
            "pk_columns",
            tuple(_required_text(value, "pk_columns") for value in self.pk_columns),
        )
        if not isinstance(self.scoring, tuple | list) or any(
            not isinstance(value, str) for value in self.scoring
        ):
            raise ValueError("scoring must contain named metric strings")
        object.__setattr__(
            self,
            "scoring",
            tuple(_required_text(value, "scoring") for value in self.scoring),
        )
        for field_name in (
            "offset_column",
            "offset_source_column",
            "offset_label",
            "sample_weight_column",
            "export_weight_column",
            "data_as_of_column",
        ):
            value = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                None if value is None else _required_text(value, field_name),
            )
        offset_fields = (
            self.offset_column,
            self.offset_source_column,
            self.offset_label,
        )
        if any(value is not None for value in offset_fields) and not all(
            value is not None for value in offset_fields
        ):
            raise ValueError(
                "offset_column, offset_source_column, and offset_label must be configured together"
            )
        if not self.features:
            raise ValueError("features must contain at least one column")
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must not contain duplicates")
        if not self.pk_columns:
            raise ValueError("pk_columns must contain at least one column")
        if len(set(self.pk_columns)) != len(self.pk_columns):
            raise ValueError("pk_columns must not contain duplicates")
        if not self.scoring:
            raise ValueError("scoring must contain at least one metric")
        if len(set(self.scoring)) != len(self.scoring):
            raise ValueError("scoring must not contain duplicates")
        if self.validation.method not in {
            "kfold",
            "train_test_split",
            "column_kfold",
            "column_holdout",
        }:
            raise ValueError(
                f"validation method {self.validation.method!r} is not supported by "
                "the notebook workflow; use a generated or column-based split"
            )
        if not self.validation.materialize:
            object.__setattr__(
                self,
                "validation",
                replace(self.validation, materialize=True),
            )

        roles: dict[str, list[str]] = {}
        role_values = {
            "target": (self.target,),
            "primary key": self.pk_columns,
            "feature": self.features,
            "split": (self.validation.column,),
            "offset": (self.offset_column,),
            "offset source": (self.offset_source_column,),
            "sample weight": (self.sample_weight_column,),
            "export weight": (self.export_weight_column,),
            "data as of": (self.data_as_of_column,),
        }
        for role, columns in role_values.items():
            for column in columns:
                if column is not None:
                    roles.setdefault(column, []).append(role)
        structural_roles = {
            "target",
            "primary key",
            "feature",
            "split",
            "data as of",
        }
        overlaps = {
            column: assigned_roles
            for column, assigned_roles in roles.items()
            if len(assigned_roles) > 1 and any(role in structural_roles for role in assigned_roles)
        }
        if overlaps:
            detail = "; ".join(
                f"{column}={','.join(assigned_roles)}"
                for column, assigned_roles in sorted(overlaps.items())
            )
            raise ValueError(f"model column roles overlap: {detail}")


@dataclass(frozen=True)
class RegisteredModel:
    model_id: int
    config: ModelBuildConfig
    source_root: Path
    spec: PricingModelSpec

    @property
    def name(self) -> str:
        return self.config.model_name


@dataclass(frozen=True)
class BuiltCandidate:
    model: RegisteredModel
    completed_build: ApprovedModelBuild

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.completed_build.metrics)

    @property
    def validation_metrics(self) -> pd.DataFrame:
        base_columns = ["validation_split_no", "n_train", "n_validation"]
        splits = self.completed_build.validation_splits
        if not splits:
            return pd.DataFrame(columns=base_columns)
        metric_names = list(splits[0].metrics)
        rows = [
            {
                "validation_split_no": split.validation_split_no,
                "n_train": split.n_train,
                "n_validation": split.n_validation,
                **split.metrics,
            }
            for split in splits
        ]
        return pd.DataFrame.from_records(rows, columns=[*base_columns, *metric_names])


def _required_text(value: Any, field_name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def _created_by(value: str | None) -> str:
    identity = str(value or getpass.getuser()).strip()
    if not identity:
        raise ValueError("created_by is required")
    return identity


def _local_notebook_settings(root: Path) -> Settings:
    return replace(
        Settings(),
        pricing_database="local_sqlite",
        mlflow_enabled=False,
        skip_database_create=True,
        rating_export_root=root / "rating_exports",
        validation_split_artifact_root=root / "validation_splits",
        workbench_artifact_root=root / "workbench_artifacts",
    )


def _connect_local(local_root: str | Path | None) -> NotebookContext:
    if local_root is None or not str(local_root).strip():
        raise ValueError("local_root is required when mode='local'")
    root = Path(local_root).expanduser().resolve()
    engine, database_paths = open_offline_sqlite(root)
    return NotebookContext(
        engine=engine,
        settings=_local_notebook_settings(root),
        mode="local",
        write_allowed=True,
        destination=f"local SQLite: {root}",
        database_paths=database_paths,
    )


def _connect_remote(
    runtime_module: str | None,
    *,
    expected_database: str | None,
    allow_writes: bool,
) -> NotebookContext:
    expected = str(expected_database or "").strip()
    if not expected:
        raise ValueError("expected_remote_database is required when mode='remote'")

    runtime = runtime_from_env_or_module(runtime_module)
    engine = runtime.get_engine()
    with engine.connect() as connection:
        actual_value = connection.execute(text("SELECT DB_NAME()")).scalar_one()
    actual = str(actual_value or "").strip()
    if not actual:
        raise RuntimeError("Remote connection did not report a database name")
    if actual.casefold() != expected.casefold():
        raise RuntimeError(
            "Remote database mismatch: "
            f"expected {expected!r}, connected to {actual!r}. Writes remain disabled."
        )
    return NotebookContext(
        engine=engine,
        settings=runtime.settings,
        mode="remote",
        write_allowed=bool(allow_writes),
        destination=f"remote SQL database: {actual}",
    )


def connect(
    *,
    mode: str,
    runtime_module: str | None = None,
    local_root: str | Path | None = None,
    expected_remote_database: str | None = None,
    allow_remote_writes: bool = False,
) -> NotebookContext:
    """Connect locally or through a governed private runtime without Airflow."""
    selected_mode = str(mode).strip().lower()
    if selected_mode == "local":
        return _connect_local(local_root)
    if selected_mode == "remote":
        return _connect_remote(
            runtime_module,
            expected_database=expected_remote_database,
            allow_writes=allow_remote_writes,
        )
    raise ValueError("mode must be 'local' or 'remote'")


def register_model(
    pricing: NotebookContext,
    spec: PricingModelSpec,
    *,
    source_root: str | Path,
    created_by: str | None = None,
) -> RegisteredModel:
    """Create a model once, then strictly validate its stable SQL identity."""
    pricing.require_write("register_model")
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"source_root does not exist: {root}")
    config = ModelBuildConfig(
        model_name=spec.name,
        model_label=spec.label,
        target_name=spec.target,
        model_type=spec.model_type,
        deployment_slot=spec.deployment_slot,
        validation_split=spec.validation,
    )
    identity = _created_by(created_by)
    if pricing.mode == "local":
        record = register_sqlite_model(
            pricing.engine,
            config,
            created_by=identity,
        )
        return RegisteredModel(
            model_id=int(record.model_id),
            config=config,
            source_root=root,
            spec=spec,
        )

    with pricing.engine.begin() as connection:
        record = register_pricing_model(
            connection,
            config,
            created_by=identity,
        )
    return RegisteredModel(
        model_id=int(record.model_id),
        config=config,
        source_root=root,
        spec=spec,
    )


def _normalise_notebook_date(value: Any, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a date, datetime, or ISO date string")
    cleaned = value.strip()
    try:
        return datetime.fromisoformat(cleaned).date()
    except ValueError:
        try:
            return date.fromisoformat(cleaned)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a date, datetime, or ISO date string") from exc


def _resolve_data_as_of(
    frame: pd.DataFrame,
    *,
    explicit: date | datetime | str | None,
    column: str | None,
) -> date:
    column_value: date | None = None
    if column is not None:
        if column not in frame.columns:
            raise ValueError(f"data-as-of column is missing from model frame: {column}")
        if frame[column].isna().any():
            raise ValueError(f"data-as-of column {column!r} contains null values")
        values = {_normalise_notebook_date(value, "data_as_of") for value in frame[column]}
        if len(values) != 1:
            raise ValueError(f"data-as-of column {column!r} must contain exactly one date")
        column_value = values.pop()

    explicit_value = None if explicit is None else _normalise_notebook_date(explicit, "data_as_of")
    if explicit_value is not None and column_value is not None and explicit_value != column_value:
        raise ValueError("explicit data_as_of does not match the configured data-as-of column")
    resolved = explicit_value or column_value
    if resolved is None:
        raise ValueError("provide data_as_of or configure PricingModelSpec.data_as_of_column")
    return resolved


def _new_build_attempt_directory(
    artifact_root: str | Path,
    *,
    model_id: int,
) -> Path:
    if isinstance(model_id, bool) or not isinstance(model_id, int) or model_id <= 0:
        raise ValueError("model_id must be a positive integer for artifact paths")
    root = Path(artifact_root).expanduser().resolve()
    model_root = (root / f"model_{model_id}").resolve()
    attempt_dir = (model_root / f"attempt_{uuid4().hex}").resolve()
    if not model_root.is_relative_to(root) or not attempt_dir.is_relative_to(root):
        raise ValueError(f"candidate artifact path is outside WORKBENCH_ARTIFACT_ROOT {root}")
    return attempt_dir


def build_candidate(
    pricing: NotebookContext,
    *,
    model: RegisteredModel,
    frame: pd.DataFrame,
    superglm_model: Any,
    data_as_of: date | datetime | str | None = None,
    created_by: str | None = None,
) -> BuiltCandidate:
    """Fit and export one candidate while deriving its audit evidence."""
    pricing.require_write("build_candidate")
    spec = model.spec
    required_columns = {
        *spec.features,
        *spec.pk_columns,
        spec.target,
        spec.offset_column,
        spec.offset_source_column,
        spec.sample_weight_column,
        spec.export_weight_column,
        spec.data_as_of_column,
        spec.validation.stratify_column,
    }
    required_columns.discard(None)
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError("model frame is missing declared columns: " + ", ".join(missing_columns))

    resolved_data_as_of = _resolve_data_as_of(
        frame,
        explicit=data_as_of,
        column=spec.data_as_of_column,
    )
    row_ids = frame.loc[:, list(spec.pk_columns)].copy()
    identity_index = canonical_row_identity_index(row_ids)
    aligned_frame = frame.copy()
    aligned_frame.index = identity_index
    X = aligned_frame.loc[:, list(spec.features)]
    y = aligned_frame[spec.target].astype(float)
    sample_weight = None
    if spec.sample_weight_column is not None:
        column = spec.sample_weight_column
        sample_weight = aligned_frame[column].astype(float)
    offset = None
    offset_source = None
    export_weight = None
    offset_contract = None
    if spec.offset_column is not None:
        offset = aligned_frame[spec.offset_column].astype(float)
        if not np.isfinite(offset.to_numpy()).all():
            raise ValueError(
                f"offset column {spec.offset_column!r} must contain finite numeric values"
            )
        source_column = spec.offset_source_column
        offset_source = aligned_frame[source_column].copy()
        offset_contract = OffsetExportContract(
            handling="EXPORTED_FACTOR",
            source_factor_name=source_column,
            published_factor_name=clean_identifier(source_column),
            source_name=source_column,
            label=spec.offset_label,
        )
    if spec.export_weight_column is not None:
        export_weight = aligned_frame[spec.export_weight_column].astype(float)

    inputs = ModelInputs(
        X=X,
        y=y,
        sample_weight=sample_weight,
        sample_weight_name=spec.sample_weight_column,
        offset=offset,
        offset_source=offset_source,
        offset_source_name=spec.offset_source_column,
        export_weight=export_weight,
        export_weight_name=spec.export_weight_column,
        row_ids=row_ids,
    )
    manifest_spec = ModelFrameManifestSpec(
        dataset_name=spec.dataset_name,
        source_system=spec.source_system,
        data_as_of_date=resolved_data_as_of,
        pk_columns=spec.pk_columns,
        target_column=spec.target,
        weight_column=spec.sample_weight_column,
        feature_columns=spec.features,
        offset_column=spec.offset_column,
        offset_source_column=spec.offset_source_column,
        offset_label=spec.offset_label,
        export_weight_column=spec.export_weight_column,
        data_as_of_column=spec.data_as_of_column,
    )
    resolved_split_indices = tuple(validation_split_indices(frame, spec.validation))
    build_identity = create_build_identity(
        frame=frame,
        model_config=model.config,
        manifest_spec=manifest_spec,
        superglm_model=superglm_model,
        split_indices=resolved_split_indices,
        fit_mode=spec.fit_mode,
        scoring=spec.scoring,
        offset_contract=offset_contract or OffsetExportContract(handling="NONE"),
        model_source_root=model.source_root,
    )
    export_id = stable_build_export_id(build_identity)
    output_dir = _new_build_attempt_directory(
        pricing.settings.workbench_artifact_root,
        model_id=model.model_id,
    )
    if pricing.mode == "local":
        model_version = resolve_sqlite_model_version(
            pricing.engine,
            model_name=model.name,
            export_id=export_id,
        )
    else:
        model_version = resolve_model_version_for_export(
            pricing.engine,
            model_name=model.name,
            export_id=export_id,
            build_fingerprint_sha256=build_identity.build_fingerprint_sha256,
        )
    completed_build = run_standard_superglm_build(
        pricing.engine,
        frame=frame,
        inputs=inputs,
        superglm_model=superglm_model,
        split_indices=resolved_split_indices,
        expected_build_identity=build_identity,
        fit_mode=spec.fit_mode,
        scoring=spec.scoring,
        output_dir=output_dir,
        model_id=model.model_id,
        model_config=model.config,
        model_version=model_version,
        export_id=export_id,
        effective_from=None,
        manifest_spec=manifest_spec,
        split_artifact_root=pricing.settings.validation_split_artifact_root,
        model_source_root=model.source_root,
        created_by=_created_by(created_by),
        offset_contract=offset_contract,
    )
    return BuiltCandidate(model=model, completed_build=completed_build)


def publish_candidate(
    pricing: NotebookContext,
    candidate: BuiltCandidate,
) -> CompletedModelPublishResult:
    """Publish a built candidate and its audit lineage to the selected store."""
    pricing.require_write("publish_candidate")
    if pricing.mode == "local":
        return publish_sqlite_candidate(
            pricing.engine,
            settings=pricing.settings,
            model_id=candidate.model.model_id,
            model_config=candidate.model.config,
            completed_build=candidate.completed_build,
            created_by=candidate.completed_build.created_by,
        )
    return publish_completed_model_build(
        pricing.engine,
        settings=pricing.settings,
        model_config=candidate.model.config,
        completed_build=candidate.completed_build,
    )


def open_candidate(
    pricing: NotebookContext,
    *,
    model: RegisteredModel,
    package_version: int,
):
    """Open one published package for an optional live editor review."""
    if pricing.mode == "local":
        raise RuntimeError(
            "Remote mode is required for the editor; local SQLite records "
            "candidate audit evidence but does not create editable rating tables."
        )
    return Workbench(
        engine=pricing.engine,
        settings=pricing.settings,
        model_config=model.config,
    ).open(
        model.name,
        package_version=int(package_version),
    )


def publish_edits(
    pricing: NotebookContext,
    *,
    candidate,
    reason: str,
    created_by: str | None = None,
):
    """Persist and synchronously publish one retained editor session."""
    pricing.require_write("publish_edits")
    if pricing.mode == "local":
        raise RuntimeError(
            "Remote mode is required for the editor; local SQLite records "
            "candidate audit evidence but does not publish editor revisions."
        )
    if candidate.editor_session is None or candidate.editor_widget is None:
        raise RuntimeError("Open the candidate editor before publishing edits")
    if candidate.workbench.engine is not pricing.engine:
        raise ValueError("candidate was opened with a different notebook context")
    identity = _created_by(created_by)
    submission = save_editor_submission(
        candidate,
        editor_session=candidate.editor_session,
        reason=_required_text(reason, "reason"),
        claimed_identity=identity,
    )
    return publish_editor_submission(
        pricing.engine,
        settings=pricing.settings,
        submission_path=submission.path,
        submission_sha256=submission.sha256,
        dag_id="notebook_publish_editor_candidate",
        airflow_run_id=f"notebook__{submission.submission_id}",
        created_by=identity,
        model_config=candidate.workbench.model_config,
    )


def deploy_package(
    pricing: NotebookContext,
    *,
    package: Candidate,
    reason: str,
    deployed_by: str | None = None,
):
    """Deploy a package using the champion snapshot the analyst actually reviewed."""
    pricing.require_write("deploy_package")
    if pricing.mode == "local":
        raise RuntimeError(
            "Remote mode is required for deployment; local SQLite is an audit "
            "workbench and cannot change a live package."
        )
    if not isinstance(package, Candidate):
        raise TypeError(
            "package must come from open_candidate(); deployment requires the "
            "champion snapshot that was visible during review"
        )
    if package.workbench.engine is not pricing.engine:
        raise ValueError("package was opened with a different notebook context")
    if "current_rate_package_id" not in package.technical:
        raise ValueError("reviewed package has no champion snapshot")
    current_rate_package_id = package.technical["current_rate_package_id"]
    model_config = package.workbench.model_config
    model_id = int(package.technical.get("model_id", -1))
    if model_id <= 0:
        raise ValueError("reviewed package has no valid SQL model_id")
    return deploy_rate_package(
        pricing.engine,
        model_config,
        rate_package_id=int(package.rate_package_id),
        expected_current_rate_package_id=(
            None if current_rate_package_id is None else int(current_rate_package_id)
        ),
        deployment_reason=_required_text(reason, "reason"),
        deployed_by=_created_by(deployed_by),
        model_id=model_id,
    )


__all__ = [
    "BuiltCandidate",
    "NotebookContext",
    "PricingModelSpec",
    "RegisteredModel",
    "build_candidate",
    "connect",
    "deploy_package",
    "open_candidate",
    "publish_candidate",
    "publish_edits",
    "register_model",
]
