from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext

from sqlalchemy import text
from sqlalchemy.engine import Engine


def record_model_run(
    engine: Engine | None,
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
    publication_receipt_path: str | None = None,
    publication_receipt_sha256: str | None = None,
    candidate_artifact_path: str | None = None,
    candidate_artifact_sha256: str | None = None,
    candidate_artifact_format: str | None = None,
    candidate_artifact_size_bytes: int | None = None,
    candidate_python_version: str | None = None,
    candidate_superglm_version: str | None = None,
    model_source_sha256: str | None = None,
    metrics: Mapping[str, float] | None = None,
    metric_scopes: Mapping[str, str] | None = None,
    fold_metrics: Sequence[Mapping[str, int | str | float]] = (),
    dataset_role: str = "training",
    split_role: str = "validation",
    parent_model_run_id: int | None = None,
    connection=None,
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
        "publication_receipt_path": publication_receipt_path,
        "publication_receipt_sha256": publication_receipt_sha256,
        "candidate_artifact_path": candidate_artifact_path,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "candidate_artifact_format": candidate_artifact_format,
        "candidate_artifact_size_bytes": candidate_artifact_size_bytes,
        "candidate_python_version": candidate_python_version,
        "candidate_superglm_version": candidate_superglm_version,
        "model_source_sha256": model_source_sha256,
        "dataset_role": dataset_role,
        "split_role": split_role,
    }
    transaction = engine.begin() if connection is None else nullcontext(connection)
    with transaction as con:
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
                        publication_receipt_path = :publication_receipt_path,
                        publication_receipt_sha256 = :publication_receipt_sha256,
                        candidate_artifact_path = :candidate_artifact_path,
                        candidate_artifact_sha256 = :candidate_artifact_sha256,
                        candidate_artifact_format = :candidate_artifact_format,
                        candidate_artifact_size_bytes = :candidate_artifact_size_bytes,
                        candidate_python_version = :candidate_python_version,
                        candidate_superglm_version = :candidate_superglm_version,
                        model_source_sha256 = :model_source_sha256,
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
                        publication_receipt_path,
                        publication_receipt_sha256,
                        candidate_artifact_path,
                        candidate_artifact_sha256,
                        candidate_artifact_format,
                        candidate_artifact_size_bytes,
                        candidate_python_version,
                        candidate_superglm_version,
                        model_source_sha256,
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
                        :publication_receipt_path,
                        :publication_receipt_sha256,
                        :candidate_artifact_path,
                        :candidate_artifact_sha256,
                        :candidate_artifact_format,
                        :candidate_artifact_size_bytes,
                        :candidate_python_version,
                        :candidate_superglm_version,
                        :model_source_sha256,
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
        split_lineage_params = {
            "model_run_id": model_run_id,
            "manifest_id": manifest_id,
            "split_set_id": split_set_id,
            "dataset_role": dataset_role,
            "split_role": split_role,
        }
        con.execute(
            text(
                """
                DELETE fold_metric
                FROM pricing.CV_FOLD_METRIC AS fold_metric
                WHERE fold_metric.model_run_id = :model_run_id
                  AND EXISTS (
                      SELECT 1
                      FROM mlops.MODEL_RUN_SPLIT_SET AS split_link
                      WHERE split_link.model_run_id = fold_metric.model_run_id
                        AND split_link.split_set_id = fold_metric.split_set_id
                        AND split_link.dataset_role = :dataset_role
                        AND split_link.split_role = :split_role
                        AND (
                            :split_set_id IS NULL
                            OR split_link.manifest_id <> :manifest_id
                            OR split_link.split_set_id <> :split_set_id
                        )
                  );
                """
            ),
            split_lineage_params,
        )
        con.execute(
            text(
                """
                DELETE split_link
                FROM mlops.MODEL_RUN_SPLIT_SET AS split_link
                WHERE split_link.model_run_id = :model_run_id
                  AND split_link.dataset_role = :dataset_role
                  AND split_link.split_role = :split_role
                  AND (
                      :split_set_id IS NULL
                      OR split_link.manifest_id <> :manifest_id
                      OR split_link.split_set_id <> :split_set_id
                  );
                """
            ),
            split_lineage_params,
        )
        con.execute(
            text(
                """
                DELETE dataset_link
                FROM mlops.MODEL_RUN_DATASET AS dataset_link
                WHERE dataset_link.model_run_id = :model_run_id
                  AND dataset_link.dataset_role = :dataset_role
                  AND dataset_link.manifest_id <> :manifest_id;
                """
            ),
            {
                "model_run_id": model_run_id,
                "manifest_id": manifest_id,
                "dataset_role": dataset_role,
            },
        )
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
        if parent_model_run_id is not None:
            con.execute(
                text(
                    """
                    MERGE mlops.MODEL_RUN_DATASET WITH (HOLDLOCK) AS tgt
                    USING (
                        SELECT
                            :model_run_id AS model_run_id,
                            parent_link.manifest_id,
                            parent_link.dataset_role
                        FROM mlops.MODEL_RUN_DATASET AS parent_link
                        WHERE parent_link.model_run_id = :parent_model_run_id
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
                    "parent_model_run_id": parent_model_run_id,
                },
            )
            con.execute(
                text(
                    """
                    MERGE mlops.MODEL_RUN_SPLIT_SET WITH (HOLDLOCK) AS tgt
                    USING (
                        SELECT
                            :model_run_id AS model_run_id,
                            parent_link.manifest_id,
                            parent_link.split_set_id,
                            parent_link.dataset_role,
                            parent_link.split_role
                        FROM mlops.MODEL_RUN_SPLIT_SET AS parent_link
                        WHERE parent_link.model_run_id = :parent_model_run_id
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
                    "parent_model_run_id": parent_model_run_id,
                },
            )
        for metric_name in sorted(metrics or {}):
            con.execute(
                text(
                    """
                    MERGE mlops.MODEL_RUN_METRIC WITH (HOLDLOCK) AS tgt
                    USING (
                        SELECT
                            :model_run_id AS model_run_id,
                            :metric_name AS metric_name
                    ) AS src
                    ON tgt.model_run_id = src.model_run_id
                       AND tgt.metric_name = src.metric_name
                    WHEN MATCHED THEN
                        UPDATE SET
                            metric_value = :metric_value,
                            metric_scope = :metric_scope
                    WHEN NOT MATCHED THEN
                        INSERT (model_run_id, metric_name, metric_value, metric_scope)
                        VALUES (
                            :model_run_id,
                            :metric_name,
                            :metric_value,
                            :metric_scope
                        );
                    """
                ),
                {
                    "model_run_id": model_run_id,
                    "metric_name": metric_name,
                    "metric_value": float((metrics or {})[metric_name]),
                    "metric_scope": (metric_scopes or {}).get(metric_name),
                },
            )
        if fold_metrics and split_set_id is None:
            raise ValueError("fold_metrics require split_set_id")
        for metric in fold_metrics:
            con.execute(
                text(
                    """
                    MERGE pricing.CV_FOLD_METRIC WITH (HOLDLOCK) AS tgt
                    USING (
                        SELECT
                            :model_run_id AS model_run_id,
                            :split_set_id AS split_set_id,
                            :fold_no AS fold_no,
                            :metric_name AS metric_name
                    ) AS src
                    ON tgt.model_run_id = src.model_run_id
                       AND tgt.split_set_id = src.split_set_id
                       AND tgt.fold_no = src.fold_no
                       AND tgt.metric_name = src.metric_name
                    WHEN MATCHED THEN
                        UPDATE SET metric_value = :metric_value
                    WHEN NOT MATCHED THEN
                        INSERT (
                            model_run_id,
                            split_set_id,
                            fold_no,
                            metric_name,
                            metric_value
                        )
                        VALUES (
                            :model_run_id,
                            :split_set_id,
                            :fold_no,
                            :metric_name,
                            :metric_value
                        );
                    """
                ),
                {
                    "model_run_id": model_run_id,
                    "split_set_id": split_set_id,
                    "fold_no": int(metric["fold_no"]),
                    "metric_name": str(metric["metric_name"]),
                    "metric_value": float(metric["metric_value"]),
                },
            )
    return int(model_run_id)


def record_model_run_on_connection(connection, **kwargs) -> int:
    return record_model_run(None, connection=connection, **kwargs)
