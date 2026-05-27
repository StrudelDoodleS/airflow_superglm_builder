from __future__ import annotations

from pathlib import Path

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import ModelExportResult
from pricing_pipeline.publishing.deployment import deploy_rate_package
from pricing_pipeline.publishing.lifecycle import DeploymentResult, PublishResult
from pricing_pipeline.publishing.model_registry import validate_registered_model
from pricing_pipeline.publishing.package_writer import publish_rating_package
from pricing_pipeline.publishing.staging import stage_rating_export


def validate_model_on_engine(engine, config: ModelBuildConfig) -> int:
    with engine.begin() as con:
        return validate_registered_model(con, config).model_id


class ModelPublisher:
    def __init__(self, engine, config: ModelBuildConfig):
        self.engine = engine
        self.config = config

    def validate_registered_model(self) -> int:
        return validate_model_on_engine(self.engine, self.config)

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
        )
