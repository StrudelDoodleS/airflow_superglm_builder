from __future__ import annotations

import pickle

import pandas as pd

try:
    import mlflow
except ModuleNotFoundError:

    class _MissingMLflow:
        def set_experiment(self, experiment_name: str) -> None:
            raise ModuleNotFoundError("No module named 'mlflow'")

        def start_run(self):
            raise ModuleNotFoundError("No module named 'mlflow'")

        def log_param(self, key: str, value) -> None:
            raise ModuleNotFoundError("No module named 'mlflow'")

        def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
            raise ModuleNotFoundError("No module named 'mlflow'")

        def log_metric(self, key: str, value: float) -> None:
            raise ModuleNotFoundError("No module named 'mlflow'")

    mlflow = _MissingMLflow()

from pricing_pipeline.config import Settings
from pricing_pipeline.lineage import record_model_run
from pricing_pipeline.mlflow_tracking import configure_mlflow
from pricing_pipeline.rating_export import (
    build_export_id,
    build_rating_export_path,
    export_rating_tables,
)
from pricing_pipeline.rating_package import publish_rating_package, stage_rating_export
from pricing_pipeline.training import FEATURE_COLUMNS, TRAINING_SQL, build_model, build_training_frame


def run_training_export_publish(
    engine,
    *,
    settings: Settings,
    manifest_id: str,
    dag_id: str,
    airflow_run_id: str,
    logical_date: str,
    created_by: str = "airflow",
) -> dict[str, str]:
    configure_mlflow(settings.mlflow_tracking_uri)
    model_name = "MTPL_FREQ"
    model_version = logical_date.replace("-", "")
    export_id = build_export_id(model_name, airflow_run_id)
    workbook_path = build_rating_export_path(
        settings.rating_export_root,
        model_name=model_name,
        logical_date=logical_date,
        export_id=export_id,
    )

    raw = pd.read_sql_query(TRAINING_SQL, engine)
    X, y, exposure, offset = build_training_frame(raw)

    mlflow.set_experiment("pricing-mtpl-frequency")
    with mlflow.start_run() as run:
        model = build_model()
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("manifest_id", manifest_id)
        mlflow.log_param("target", "ClaimNb")
        mlflow.log_param("offset", "log(Exposure)")
        mlflow.log_param("row_count", len(X))
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))

        fitted_model = model.fit_reml(X, y, offset=offset)
        if fitted_model is None:
            fitted_model = model

        deviance = getattr(getattr(fitted_model, "result", None), "deviance", None)
        if deviance is not None:
            mlflow.log_metric("deviance", float(deviance))

        model_path = workbook_path.parent / "superglm_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as handle:
            pickle.dump(fitted_model, handle)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        export_rating_tables(fitted_model, X, y, exposure, output_path=workbook_path)

        stage_rating_export(
            engine,
            workbook_path=workbook_path,
            export_id=export_id,
            model_name=model_name,
            model_version=model_version,
            effective_from=logical_date,
            created_by=created_by,
            replace=True,
        )
        rate_package_id = publish_rating_package(
            engine,
            export_id=export_id,
            pointer_name="MTPL_FREQ_UAT",
            created_by=created_by,
            package_status="DRAFT",
        )
        record_model_run(
            engine,
            dag_id=dag_id,
            airflow_run_id=airflow_run_id,
            mlflow_run_id=run.info.run_id,
            manifest_id=manifest_id,
            export_id=export_id,
            model_name=model_name,
            model_version=model_version,
            rate_package_id=rate_package_id,
            rating_workbook_path=str(workbook_path),
            run_status="SUCCESS",
            created_by=created_by,
        )

        return {
            "mlflow_run_id": run.info.run_id,
            "export_id": export_id,
            "rate_package_id": str(rate_package_id),
            "rating_workbook_path": str(workbook_path),
        }
