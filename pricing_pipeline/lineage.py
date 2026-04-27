from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


def record_model_run(
    engine: Engine,
    *,
    dag_id: str,
    airflow_run_id: str,
    mlflow_run_id: str,
    manifest_id: str,
    export_id: str,
    model_name: str,
    model_version: str,
    rate_package_id: int | None,
    rating_workbook_path: str,
    run_status: str,
    created_by: str,
) -> None:
    with engine.begin() as con:
        con.execute(
            text(
                """
                INSERT INTO pricing.MODEL_RUN (
                    dag_id,
                    airflow_run_id,
                    mlflow_run_id,
                    manifest_id,
                    export_id,
                    model_name,
                    model_version,
                    rate_package_id,
                    rating_workbook_path,
                    run_status,
                    completed_ts,
                    created_by
                )
                VALUES (
                    :dag_id,
                    :airflow_run_id,
                    :mlflow_run_id,
                    :manifest_id,
                    :export_id,
                    :model_name,
                    :model_version,
                    :rate_package_id,
                    :rating_workbook_path,
                    :run_status,
                    SYSUTCDATETIME(),
                    :created_by
                )
                """
            ),
            {
                "dag_id": dag_id,
                "airflow_run_id": airflow_run_id,
                "mlflow_run_id": mlflow_run_id,
                "manifest_id": manifest_id,
                "export_id": export_id,
                "model_name": model_name,
                "model_version": model_version,
                "rate_package_id": rate_package_id,
                "rating_workbook_path": rating_workbook_path,
                "run_status": run_status,
                "created_by": created_by,
            },
        )
