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
    model_id: int,
    model_name: str,
    model_version: str,
    rate_package_id: int | None,
    rating_workbook_path: str,
    run_status: str,
    created_by: str,
) -> None:
    params = {
        "dag_id": dag_id,
        "airflow_run_id": airflow_run_id,
        "mlflow_run_id": mlflow_run_id,
        "manifest_id": manifest_id,
        "export_id": export_id,
        "model_id": model_id,
        "model_name": model_name,
        "model_version": model_version,
        "rate_package_id": rate_package_id,
        "rating_workbook_path": rating_workbook_path,
        "run_status": run_status,
        "created_by": created_by,
    }
    with engine.begin() as con:
        con.execute(
            text(
                """
                MERGE pricing.MODEL_RUN WITH (HOLDLOCK) AS tgt
                USING (
                    SELECT
                        :dag_id AS dag_id,
                        :airflow_run_id AS airflow_run_id,
                        :model_id AS model_id,
                        :model_name AS model_name
                ) AS src
                ON tgt.dag_id = src.dag_id
                   AND tgt.airflow_run_id = src.airflow_run_id
                   AND tgt.model_id = src.model_id
                WHEN MATCHED THEN
                    UPDATE SET
                        mlflow_run_id = :mlflow_run_id,
                        manifest_id = :manifest_id,
                        export_id = :export_id,
                        model_id = :model_id,
                        model_name = :model_name,
                        model_version = :model_version,
                        rate_package_id = :rate_package_id,
                        rating_workbook_path = :rating_workbook_path,
                        run_status = :run_status,
                        completed_ts = SYSUTCDATETIME(),
                        created_by = :created_by
                WHEN NOT MATCHED THEN
                    INSERT (
                        dag_id,
                        airflow_run_id,
                        mlflow_run_id,
                        manifest_id,
                        export_id,
                        model_id,
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
                        :model_id,
                        :model_name,
                        :model_version,
                        :rate_package_id,
                        :rating_workbook_path,
                        :run_status,
                        SYSUTCDATETIME(),
                        :created_by
                    );
                """
            ),
            params,
        )
