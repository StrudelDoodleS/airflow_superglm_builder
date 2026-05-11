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
    split_set_id: str | None = None,
    export_id: str,
    model_id: int,
    model_name: str,
    model_version: str,
    rate_package_id: int | None,
    rating_workbook_path: str,
    run_status: str,
    created_by: str,
    dataset_role: str = "training",
    split_role: str = "cross_validation",
) -> int:
    params = {
        "dag_id": dag_id,
        "airflow_run_id": airflow_run_id,
        "mlflow_run_id": mlflow_run_id,
        "manifest_id": manifest_id,
        "split_set_id": split_set_id,
        "export_id": export_id,
        "model_id": model_id,
        "model_name": model_name,
        "model_version": model_version,
        "rate_package_id": rate_package_id,
        "rating_workbook_path": rating_workbook_path,
        "run_status": run_status,
        "created_by": created_by,
        "dataset_role": dataset_role,
        "split_role": split_role,
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
        model_run_id = con.execute(
            text(
                """
                SELECT model_run_id
                FROM pricing.MODEL_RUN
                WHERE dag_id = :dag_id
                  AND airflow_run_id = :airflow_run_id
                  AND model_id = :model_id
                """
            ),
            params,
        ).scalar_one()
        con.execute(
            text(
                """
                MERGE mlops.MODEL_RUN_DATASET WITH (HOLDLOCK) AS tgt
                USING (
                    SELECT
                        :model_run_id AS model_run_id,
                        :manifest_id AS manifest_id,
                        :dataset_role AS dataset_role
                ) AS src
                ON tgt.model_run_id = src.model_run_id
                   AND tgt.manifest_id = src.manifest_id
                   AND tgt.dataset_role = src.dataset_role
                WHEN NOT MATCHED THEN
                    INSERT (
                        model_run_id,
                        manifest_id,
                        dataset_role
                    )
                    VALUES (
                        src.model_run_id,
                        src.manifest_id,
                        src.dataset_role
                    );
                """
            ),
            {
                "model_run_id": model_run_id,
                "manifest_id": manifest_id,
                "dataset_role": dataset_role,
            },
        )
        if split_set_id is not None:
            con.execute(
                text(
                    """
                    MERGE mlops.MODEL_RUN_SPLIT_SET WITH (HOLDLOCK) AS tgt
                    USING (
                        SELECT
                            :model_run_id AS model_run_id,
                            :manifest_id AS manifest_id,
                            :split_set_id AS split_set_id,
                            :dataset_role AS dataset_role,
                            :split_role AS split_role
                    ) AS src
                    ON tgt.model_run_id = src.model_run_id
                       AND tgt.split_set_id = src.split_set_id
                       AND tgt.split_role = src.split_role
                    WHEN NOT MATCHED THEN
                        INSERT (
                            model_run_id,
                            manifest_id,
                            split_set_id,
                            dataset_role,
                            split_role
                        )
                        VALUES (
                            src.model_run_id,
                            src.manifest_id,
                            src.split_set_id,
                            src.dataset_role,
                            src.split_role
                        );
                    """
                ),
                {
                    "model_run_id": model_run_id,
                    "manifest_id": manifest_id,
                    "split_set_id": split_set_id,
                    "dataset_role": dataset_role,
                    "split_role": split_role,
                },
            )
    return int(model_run_id)
