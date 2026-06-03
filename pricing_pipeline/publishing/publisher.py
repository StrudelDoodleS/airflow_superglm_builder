from __future__ import annotations

from pathlib import Path

import pandas as pd

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import ModelExportResult
from pricing_pipeline.publishing.deployment import deploy_rate_package
from pricing_pipeline.publishing.lifecycle import (
    DeploymentResult,
    PredictionComparison,
    PublishResult,
    RatePackageRevisionResult,
    RatePackageSelector,
    RatePackageSnapshot,
)
from pricing_pipeline.publishing.manual_revision import (
    create_manual_revision,
    load_rate_package_snapshot,
)
from pricing_pipeline.publishing.model_registry import (
    ModelRegistryError,
    validate_registered_model,
)
from pricing_pipeline.publishing.package_writer import publish_rating_package
from pricing_pipeline.publishing.prediction_compare import compare_prediction_vectors
from pricing_pipeline.publishing.staging import stage_rating_export


def _validate_export_matches_config(
    export_result: ModelExportResult,
    config: ModelBuildConfig,
    *,
    model_id: int,
) -> None:
    mismatches: list[str] = []
    if int(export_result.model_id) != int(model_id):
        mismatches.append(
            f"model_id export={export_result.model_id!r} config={model_id!r}"
        )
    if export_result.model_key != config.model_key:
        mismatches.append(
            f"model_key export={export_result.model_key!r} config={config.model_key!r}"
        )
    if export_result.target_name != config.target_name:
        mismatches.append(
            f"target_name export={export_result.target_name!r} "
            f"config={config.target_name!r}"
        )
    if export_result.model_type != config.model_type:
        mismatches.append(
            f"model_type export={export_result.model_type!r} "
            f"config={config.model_type!r}"
        )
    if export_result.deployment_slot != config.deployment_slot:
        mismatches.append(
            f"deployment_slot export={export_result.deployment_slot!r} "
            f"config={config.deployment_slot!r}"
        )

    if mismatches:
        raise ModelRegistryError(
            "training export does not match model config: " + "; ".join(mismatches)
        )


def validate_model_on_engine(engine, config: ModelBuildConfig) -> int:
    with engine.begin() as con:
        return validate_registered_model(con, config).model_id


class ModelPublisher:
    def __init__(self, engine, config: ModelBuildConfig):
        self.engine = engine
        self.config = config

    def validate_registered_model(self) -> int:
        return validate_model_on_engine(self.engine, self.config)

    def compare_prediction_vectors(
        self,
        before: pd.Series,
        after: pd.Series,
        *,
        top_n: int = 25,
    ) -> PredictionComparison:
        return compare_prediction_vectors(before, after, top_n=top_n)

    def load_rate_package(self, selector: RatePackageSelector) -> RatePackageSnapshot:
        return load_rate_package_snapshot(self.engine, self.config, selector)

    def create_manual_revision(
        self,
        *,
        parent: RatePackageSnapshot,
        edited_rate_cells: pd.DataFrame,
        reason: str,
        created_by: str,
    ) -> RatePackageRevisionResult:
        self.validate_registered_model()
        return create_manual_revision(
            self.engine,
            self.config,
            parent=parent,
            edited_rate_cells=edited_rate_cells,
            reason=reason,
            created_by=created_by,
        )

    def deploy(
        self,
        *,
        rate_package_id: int | None = None,
        package_version: int | None = None,
        deployment_reason: str,
        deployed_by: str,
        deployment_slot: str | None = None,
    ) -> DeploymentResult:
        model_id = self.validate_registered_model()
        return deploy_rate_package(
            self.engine,
            self.config,
            rate_package_id=rate_package_id,
            package_version=package_version,
            deployment_slot=deployment_slot,
            deployment_reason=deployment_reason,
            deployed_by=deployed_by,
            model_id=model_id,
        )

    def publish_training_export(self, export: ModelExportResult | dict) -> PublishResult:
        export_result = ModelExportResult.from_mapping(export)
        model_id = self.validate_registered_model()
        _validate_export_matches_config(
            export_result,
            self.config,
            model_id=model_id,
        )
        stage_rating_export(
            self.engine,
            workbook_path=Path(export_result.rating_workbook_path),
            export_id=export_result.export_id,
            model_name=self.config.model_key,
            model_version=export_result.model_version,
            target_name=self.config.target_name,
            model_type=self.config.model_type,
            effective_from=export_result.effective_from,
            created_by=export_result.created_by,
            replace=True,
            model_id=model_id,
        )
        result = publish_rating_package(
            self.engine,
            export_id=export_result.export_id,
            created_by=export_result.created_by,
            package_status=self.config.default_package_status,
        )
        return PublishResult(
            mlflow_run_id=export_result.mlflow_run_id,
            export_id=result.export_id,
            rate_package_id=result.rate_package_id,
            package_version=result.package_version,
            rating_workbook_path=export_result.rating_workbook_path,
            was_existing=result.was_existing,
        )
