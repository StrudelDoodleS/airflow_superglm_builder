from __future__ import annotations

import pickle

import pandas as pd

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.publishing.lineage import record_model_run
from pricing_pipeline.infra.mlflow_tracking import configure_mlflow
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import (
    ModelExportResult,
    ModelSpec,
    coerce_training_frame,
)
from pricing_pipeline.publishing.rating_export import (
    build_export_id,
    build_rating_export_path,
    export_rating_tables,
)
from pricing_pipeline.publishing.publisher import ModelPublisher, validate_model_on_engine
from pricing_pipeline.models.superglm_diagnostics import fit_reml_with_diagnostics


def train_and_export_model(
    engine,
    *,
    settings: Settings,
    manifest_id: str,
    split_set_id: str | None = None,
    dag_id: str,
    airflow_run_id: str,
    logical_date: str,
    spec: ModelSpec,
    model_config: ModelBuildConfig,
    created_by: str = "airflow",
) -> ModelExportResult:
    mlflow_client = configure_mlflow(
        settings.mlflow_tracking_uri,
        enabled=settings.mlflow_enabled,
    )
    model_id = validate_model_on_engine(engine, model_config)
    model_version = logical_date.replace("-", "")
    export_id = build_export_id(spec.model_key, airflow_run_id)
    workbook_path = build_rating_export_path(
        settings.rating_export_root,
        model_name=spec.model_key,
        logical_date=logical_date,
        export_id=export_id,
    )

    raw = pd.read_sql_query(spec.training_sql, engine)
    training_frame = coerce_training_frame(spec.build_training_frame(raw))

    mlflow_client.set_experiment(spec.experiment_name)
    with mlflow_client.start_run() as run:
        model = spec.build_model()
        workbook_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow_client.log_param("model_name", spec.model_key)
        mlflow_client.log_param("model_id", model_id)
        mlflow_client.log_param("model_version", model_version)
        mlflow_client.log_param("manifest_id", manifest_id)
        mlflow_client.log_param("target", spec.target_name)
        mlflow_client.log_param("offset", spec.offset_label)
        mlflow_client.log_param("row_count", len(training_frame.X))
        mlflow_client.log_param("feature_columns", ",".join(spec.feature_columns))

        fitted_model = fit_reml_with_diagnostics(
            model,
            training_frame.X,
            training_frame.y,
            offset=training_frame.offset,
            diagnostics_path=workbook_path.parent / "superglm_fit.log",
            mlflow_client=mlflow_client,
        )

        deviance = getattr(getattr(fitted_model, "result", None), "deviance", None)
        if deviance is not None:
            mlflow_client.log_metric("deviance", float(deviance))

        model_path = workbook_path.parent / "superglm_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as handle:
            pickle.dump(fitted_model, handle)
        mlflow_client.log_artifact(str(model_path), artifact_path="model")

        export_rating_tables(
            fitted_model,
            training_frame.X,
            training_frame.y,
            training_frame.exposure,
            output_path=workbook_path,
            mlflow_client=mlflow_client,
        )

        return ModelExportResult(
            model_id=model_id,
            model_key=spec.model_key,
            model_version=model_version,
            model_type=spec.model_type,
            target_name=spec.target_name,
            deployment_slot=spec.deployment_slot,
            manifest_id=manifest_id,
            dag_id=dag_id,
            airflow_run_id=airflow_run_id,
            mlflow_run_id=str(getattr(run.info, "run_id", "")),
            split_set_id=split_set_id,
            export_id=export_id,
            rating_workbook_path=str(workbook_path),
            effective_from=logical_date,
            created_by=created_by,
            package_status=spec.package_status,
        )


def publish_model_export(
    engine,
    export: ModelExportResult | dict,
    *,
    model_config: ModelBuildConfig,
) -> dict[str, str | bool]:
    export_result = ModelExportResult.from_mapping(export)
    publisher = ModelPublisher(engine, model_config)
    publisher.validate_registered_model()
    publish_result = publisher.publish_training_export(export_result)

    record_model_run(
        engine,
        dag_id=export_result.dag_id,
        airflow_run_id=export_result.airflow_run_id,
        mlflow_run_id=export_result.mlflow_run_id,
        manifest_id=export_result.manifest_id,
        split_set_id=export_result.split_set_id,
        export_id=export_result.export_id,
        model_id=export_result.model_id,
        model_name=export_result.model_key,
        model_version=export_result.model_version,
        rate_package_id=publish_result.rate_package_id,
        rating_workbook_path=str(publish_result.rating_workbook_path),
        run_status="SUCCESS",
        created_by=export_result.created_by,
    )

    return {
        "mlflow_run_id": str(publish_result.mlflow_run_id),
        "export_id": str(publish_result.export_id),
        "rate_package_id": str(publish_result.rate_package_id),
        "package_version": str(publish_result.package_version),
        "package_status": str(publish_result.package_status),
        "rating_workbook_path": str(publish_result.rating_workbook_path),
        "was_existing": bool(getattr(publish_result, "was_existing", False)),
    }


def run_training_export_publish(
    engine,
    *,
    settings: Settings,
    manifest_id: str,
    split_set_id: str | None = None,
    dag_id: str,
    airflow_run_id: str,
    logical_date: str,
    spec: ModelSpec,
    model_config: ModelBuildConfig,
    created_by: str = "airflow",
) -> dict[str, str]:
    export = train_and_export_model(
        engine,
        settings=settings,
        manifest_id=manifest_id,
        split_set_id=split_set_id,
        dag_id=dag_id,
        airflow_run_id=airflow_run_id,
        logical_date=logical_date,
        spec=spec,
        model_config=model_config,
        created_by=created_by,
    )
    return publish_model_export(engine, export, model_config=model_config)
