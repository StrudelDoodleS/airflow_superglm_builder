"""Small, synchronous entry points for pricing-model notebooks.

The notebook owns model and data decisions.  These helpers own generated SQL
identifiers, audit records, artifact locations, and publication plumbing.
"""

from __future__ import annotations

import getpass
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from pricing_pipeline.data.manifest import (
    ModelFrameManifestSpec,
    validation_split_indices,
)
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra.runtime import runtime_from_env_or_module
from pricing_pipeline.modeling.standard_superglm import (
    ModelInputs,
    StandardBuildResult,
    canonical_row_identity_index,
    run_standard_superglm_build,
)
from pricing_pipeline.models.config import ModelBuildConfig, ValidationSplitConfig
from pricing_pipeline.orchestration.completed_build_helpers import effective_from_for_run
from pricing_pipeline.orchestration.publish_completed_build import (
    CompletedModelPublishResult,
    publish_completed_model_build,
)
from pricing_pipeline.publishing.model_registry import (
    register_pricing_model,
    validate_registered_model,
)
from pricing_pipeline.publishing.model_versions import resolve_model_version_for_export
from pricing_pipeline.publishing.deployment import deploy_rate_package
from pricing_pipeline.publishing.editor_candidate import publish_editor_submission
from pricing_pipeline.publishing.rating_export import build_export_id
from pricing_pipeline.publishing.superglm_publication_receipt import OffsetExportContract
from pricing_pipeline.workbench.core import Workbench
from pricing_pipeline.workbench.submission import create_editor_submission


@dataclass(frozen=True)
class NotebookContext:
    engine: Any
    settings: Settings


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
    exposure_column: str | None = None
    sample_weight_column: str | None = None
    data_as_of_column: str | None = None
    scoring: tuple[str, ...] = ("deviance",)
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
        object.__setattr__(
            self,
            "scoring",
            tuple(_required_text(value, "scoring") for value in self.scoring),
        )
        for field_name in (
            "exposure_column",
            "sample_weight_column",
            "data_as_of_column",
        ):
            value = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                None if value is None else _required_text(value, field_name),
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

        roles: dict[str, list[str]] = {}
        role_values = {
            "target": (self.target,),
            "primary key": self.pk_columns,
            "feature": self.features,
            "exposure": (self.exposure_column,),
            "sample weight": (self.sample_weight_column,),
            "data as of": (self.data_as_of_column,),
        }
        for role, columns in role_values.items():
            for column in columns:
                if column is not None:
                    roles.setdefault(column, []).append(role)
        overlaps = {
            column: assigned_roles
            for column, assigned_roles in roles.items()
            if len(assigned_roles) > 1
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
    spec: PricingModelSpec | None = None

    @property
    def name(self) -> str:
        return self.config.model_name


@dataclass(frozen=True)
class BuiltCandidate:
    model: RegisteredModel
    standard_build: StandardBuildResult | Any

    @property
    def completed_build(self) -> dict[str, Any]:
        return self.standard_build.completed_build

    @property
    def metrics(self) -> dict[str, float]:
        return dict(getattr(self.standard_build, "metrics", {}))


@dataclass
class _LocalSubmissionClient:
    """Satisfy submission serialization without making an Airflow request."""

    triggered: bool = False

    def trigger_dag(self, dag_id: str, *, run_id: str, conf: dict[str, Any]):
        del conf
        self.triggered = True
        return type(
            "LocalNotebookRun",
            (),
            {"dag_id": dag_id, "dag_run_id": run_id, "state": "saved"},
        )()


def _required_text(value: Any, field_name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def _created_by(value: str | None) -> str:
    return _required_text(value or getpass.getuser(), "created_by")


def _new_notebook_run_key() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"notebook_{timestamp}_{uuid4().hex[:8]}"


def _identity_aligned(
    value: pd.Series | pd.DataFrame | np.ndarray | None,
    *,
    field_name: str,
    frame_index: pd.Index,
    identity_index: pd.Index,
    default_name: str | None = None,
) -> pd.Series | pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.Series | pd.DataFrame):
        if not (
            value.index.identical(identity_index)
            or value.index.equals(frame_index)
        ):
            raise ValueError(
                f"{field_name} index/order does not match the model frame"
            )
        aligned = value.copy()
    else:
        array = np.asarray(value)
        if array.ndim == 1:
            aligned = pd.Series(array.copy(), name=default_name)
        elif array.ndim == 2 and field_name == "y":
            aligned = pd.DataFrame(array.copy())
        else:
            raise ValueError(f"{field_name} must be one-dimensional")
    if len(aligned) != len(identity_index):
        raise ValueError(
            f"{field_name} length {len(aligned)} does not match model frame "
            f"length {len(identity_index)}"
        )
    aligned.index = identity_index
    if isinstance(aligned, pd.Series) and aligned.name is None and default_name:
        aligned.name = default_name
    return aligned


def connect(runtime_module: str | None = None) -> NotebookContext:
    """Connect through the governed runtime module without starting Airflow."""
    runtime = runtime_from_env_or_module(runtime_module)
    return NotebookContext(engine=runtime.get_engine(), settings=runtime.settings)


def register_model(
    pricing: NotebookContext,
    spec: PricingModelSpec | None = None,
    *,
    name: str | None = None,
    label: str | None = None,
    target: str | None = None,
    model_type: str | None = None,
    deployment_slot: str | None = None,
    validation_split: ValidationSplitConfig | None = None,
    source_root: str | Path,
    package_status: str = "PUBLISHED",
    created_by: str | None = None,
) -> RegisteredModel:
    """Create a model once, then strictly validate its stable SQL identity."""
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"source_root does not exist: {root}")
    legacy_identity = {
        "name": name,
        "label": label,
        "target": target,
        "model_type": model_type,
        "deployment_slot": deployment_slot,
        "validation_split": validation_split,
    }
    if spec is not None:
        supplied = [field for field, value in legacy_identity.items() if value is not None]
        if supplied:
            raise ValueError(
                "PricingModelSpec cannot be combined with legacy registration fields: "
                + ", ".join(supplied)
            )
        name = spec.name
        label = spec.label
        target = spec.target
        model_type = spec.model_type
        deployment_slot = spec.deployment_slot
        validation_split = spec.validation
    config = ModelBuildConfig(
        model_name=_required_text(name, "name"),
        model_label=_required_text(label, "label"),
        target_name=_required_text(target, "target"),
        model_type=_required_text(model_type, "model_type"),
        deployment_slot=_required_text(
            deployment_slot or "PRODUCTION",
            "deployment_slot",
        ).upper(),
        default_package_status=_required_text(package_status, "package_status").upper(),
        validation_split=validation_split or ValidationSplitConfig.kfold(),
    )
    identity = _created_by(created_by)
    with pricing.engine.begin() as connection:
        inserted_model_id = register_pricing_model(
            connection,
            config,
            created_by=identity,
        )
        record = validate_registered_model(connection, config)
    if int(inserted_model_id) != int(record.model_id):
        raise RuntimeError("registered model_id changed while resolving model identity")
    return RegisteredModel(
        model_id=int(record.model_id),
        config=config,
        source_root=root,
        spec=spec,
    )


def build_candidate(
    pricing: NotebookContext,
    *,
    model: RegisteredModel,
    frame: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    model_factory: Callable[[], Any],
    scoring: str | Callable | Sequence[str | Callable],
    dataset_name: str,
    source_system: str,
    data_as_of: date | datetime | str,
    pk_columns: Iterable[str],
    effective_from: date | datetime | str,
    sample_weight: pd.Series | np.ndarray | None = None,
    weight_column: str | None = None,
    offset: pd.Series | np.ndarray | None = None,
    export_weight: pd.Series | np.ndarray | None = None,
    sample_weight_name: str | None = None,
    export_weight_name: str | None = None,
    split_indices: Iterable[tuple[Any, Any]] | None = None,
    fit_mode: str = "fit_reml",
    offset_contract: OffsetExportContract | None = None,
    offset_export_options: dict[str, Any] | None = None,
    review_workbook_hook: Callable[..., str | Path | None] | None = None,
    run_key: str | None = None,
    created_by: str | None = None,
) -> BuiltCandidate:
    """Fit and export one candidate while deriving its audit evidence."""
    resolved_pk_columns = tuple(_required_text(column, "pk_columns") for column in pk_columns)
    if not resolved_pk_columns:
        raise ValueError("pk_columns must contain at least one column")
    missing_pk = [column for column in resolved_pk_columns if column not in frame.columns]
    if missing_pk:
        raise ValueError(f"model frame is missing primary key columns: {', '.join(missing_pk)}")

    resolved_run_key = _required_text(run_key or _new_notebook_run_key(), "run_key")
    export_id = build_export_id(model.name, resolved_run_key)
    model_version = resolve_model_version_for_export(
        pricing.engine,
        model_name=model.name,
        export_id=export_id,
    )
    validation_split = model.config.validation_split
    resolved_split_indices = (
        validation_split_indices(frame, validation_split)
        if split_indices is None
        else list(split_indices)
    )
    row_ids = frame.loc[:, list(resolved_pk_columns)].copy()
    identity_index = canonical_row_identity_index(row_ids)
    aligned_X = _identity_aligned(
        X,
        field_name="X",
        frame_index=frame.index,
        identity_index=identity_index,
    )
    if not isinstance(aligned_X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    inputs = ModelInputs(
        X=aligned_X,
        y=_identity_aligned(
            y,
            field_name="y",
            frame_index=frame.index,
            identity_index=identity_index,
            default_name=model.config.target_name,
        ),
        sample_weight=_identity_aligned(
            sample_weight,
            field_name="sample_weight",
            frame_index=frame.index,
            identity_index=identity_index,
            default_name=sample_weight_name or weight_column,
        ),
        sample_weight_name=sample_weight_name,
        offset=_identity_aligned(
            offset,
            field_name="offset",
            frame_index=frame.index,
            identity_index=identity_index,
        ),
        export_weight=_identity_aligned(
            export_weight,
            field_name="export_weight",
            frame_index=frame.index,
            identity_index=identity_index,
            default_name=export_weight_name,
        ),
        export_weight_name=export_weight_name,
        row_ids=row_ids,
    )
    standard_build = run_standard_superglm_build(
        pricing.engine,
        frame=frame,
        inputs=inputs,
        model_factory=model_factory,
        split_indices=resolved_split_indices,
        fit_mode=fit_mode,
        scoring=scoring,
        output_dir=(
            Path(pricing.settings.workbench_artifact_root)
            / model.name
            / resolved_run_key
        ),
        model_name=model.name,
        model_version=model_version,
        export_id=export_id,
        effective_from=effective_from_for_run(effective_from),
        manifest_spec=ModelFrameManifestSpec(
            dataset_name=dataset_name,
            source_system=source_system,
            data_as_of_date=data_as_of,
            pk_columns=resolved_pk_columns,
            target_column=model.config.target_name,
            weight_column=weight_column,
        ),
        validation_split=validation_split,
        split_artifact_root=pricing.settings.validation_split_artifact_root,
        model_source_root=model.source_root,
        created_by=_created_by(created_by),
        offset_contract=offset_contract,
        offset_export_options=offset_export_options,
        review_workbook_hook=review_workbook_hook,
    )
    return BuiltCandidate(model=model, standard_build=standard_build)


def publish_candidate(
    pricing: NotebookContext,
    candidate: BuiltCandidate,
    *,
    created_by: str | None = None,
) -> CompletedModelPublishResult:
    """Publish a built candidate and its audit lineage directly to SQL Server."""
    return publish_completed_model_build(
        pricing.engine,
        settings=pricing.settings,
        model_config=candidate.model.config,
        dataset=None,
        completed_build=candidate.completed_build,
        created_by=_created_by(created_by),
    )


def _config_loader(model: RegisteredModel):
    def load(model_name: str) -> ModelBuildConfig:
        if _required_text(model_name, "model_name") != model.name:
            raise KeyError(f"unknown notebook model {model_name!r}")
        return model.config

    return load


def _workbench(pricing: NotebookContext, model: RegisteredModel) -> Workbench:
    return Workbench(
        engine=pricing.engine,
        settings=pricing.settings,
        config_loader=_config_loader(model),
        model_names_loader=lambda: (model.name,),
    )


def open_candidate(
    pricing: NotebookContext,
    *,
    model: RegisteredModel,
    package_version: int,
):
    """Open one published package for an optional live editor review."""
    return _workbench(pricing, model).open(
        model.name,
        package_version=int(package_version),
    )


def publish_edits(
    pricing: NotebookContext,
    *,
    model: RegisteredModel,
    candidate,
    reason: str,
    created_by: str | None = None,
):
    """Persist and synchronously publish one retained editor session."""
    if candidate.model_name != model.name:
        raise ValueError("candidate model_name does not match the registered notebook model")
    if candidate.editor_session is None or candidate.editor_widget is None:
        raise RuntimeError("Open the candidate editor before publishing edits")
    identity = _created_by(created_by)
    submission = create_editor_submission(
        candidate,
        editor_session=candidate.editor_session,
        reason=_required_text(reason, "reason"),
        airflow_client=_LocalSubmissionClient(),
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
        model_config=model.config,
    )


def deploy_package(
    pricing: NotebookContext,
    *,
    model: RegisteredModel,
    package,
    reason: str,
    deployed_by: str | None = None,
):
    """Deploy a published package with an immediately refreshed stale guard."""
    slot = model.config.deployment_slot.upper()
    current_rate_package_id = _workbench(
        pricing,
        model,
    ).current_champion_rate_package_id(
        model.name,
        deployment_slot=slot,
    )
    return deploy_rate_package(
        pricing.engine,
        model.config,
        rate_package_id=int(package.rate_package_id),
        expected_current_rate_package_id=current_rate_package_id,
        deployment_slot=slot,
        deployment_reason=_required_text(reason, "reason"),
        deployed_by=_created_by(deployed_by),
        model_id=model.model_id,
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
