from __future__ import annotations

from contextlib import nullcontext

from sqlalchemy import text
from sqlalchemy.engine import Engine

from pricing_pipeline.models.spec import ApprovedModelBuild


class ModelRunIdentityError(RuntimeError):
    """Raised when a successful model-run identity is reused inconsistently."""


_DATASET_ROLE = "training"
_SPLIT_ROLE = "validation"


_IMMUTABLE_MODEL_RUN_FIELDS = (
    "dag_id",
    "airflow_run_id",
    "mlflow_run_id",
    "manifest_id",
    "export_id",
    "model_id",
    "model_name",
    "model_version",
    "rate_package_id",
    "rating_workbook_path",
    "rating_workbook_sha256",
    "run_status",
    "created_by",
    "publication_receipt_path",
    "publication_receipt_sha256",
    "candidate_artifact_path",
    "candidate_artifact_sha256",
    "candidate_artifact_format",
    "candidate_artifact_size_bytes",
    "candidate_python_version",
    "candidate_superglm_version",
    "candidate_superglm_git_sha",
    "model_source_sha256",
    "builder_source_sha256",
    "materialized_split_sha256",
    "runtime_sha256",
    "candidate_superglm_sha256",
    "validation_curve_status",
    "validation_curve_reason",
    "parent_model_run_id",
)


def _identity_value(value):
    if value is None:
        return None
    return str(value)


def record_model_run(
    engine: Engine | None,
    *,
    build: ApprovedModelBuild,
    dag_id: str,
    airflow_run_id: str,
    rate_package_id: int | None,
    parent_model_run_id: int | None = None,
    connection=None,
) -> int:
    manifest_id = build.manifest_id
    split_set_id = build.split_set_id
    metrics = build.metrics
    metric_scopes = build.metric_scopes
    fold_metrics = build.fold_metrics
    if fold_metrics and split_set_id is None:
        raise ValueError("fold_metrics require split_set_id")
    params = {
        "dag_id": dag_id,
        "airflow_run_id": airflow_run_id,
        "mlflow_run_id": build.mlflow_run_id,
        "manifest_id": manifest_id,
        "split_set_id": split_set_id,
        "export_id": build.export_id,
        "model_id": build.model_id,
        "model_name": build.model_name,
        "model_version": build.model_version,
        "rate_package_id": rate_package_id,
        "rating_workbook_path": build.rating_workbook_path,
        "rating_workbook_sha256": build.rating_workbook_sha256,
        "run_status": "SUCCESS",
        "created_by": build.created_by,
        "publication_receipt_path": build.publication_receipt_path,
        "publication_receipt_sha256": build.publication_receipt_sha256,
        "candidate_artifact_path": build.candidate_artifact_path,
        "candidate_artifact_sha256": build.candidate_artifact_sha256,
        "candidate_artifact_format": build.candidate_artifact_format,
        "candidate_artifact_size_bytes": build.candidate_artifact_size_bytes,
        "candidate_python_version": build.candidate_python_version,
        "candidate_superglm_version": build.candidate_superglm_version,
        "candidate_superglm_git_sha": build.candidate_superglm_git_sha,
        "model_source_sha256": build.model_source_sha256,
        "builder_source_sha256": build.builder_source_sha256,
        "materialized_split_sha256": build.materialized_split_sha256,
        "runtime_sha256": build.runtime_sha256,
        "candidate_superglm_sha256": build.candidate_superglm_sha256,
        "validation_curve_status": build.validation_curve_status,
        "validation_curve_reason": build.validation_curve_reason,
        "dataset_role": _DATASET_ROLE,
        "split_role": _SPLIT_ROLE,
        "parent_model_run_id": parent_model_run_id,
        "validation_source_model_run_id": None,
    }
    transaction = engine.begin() if connection is None else nullcontext(connection)
    with transaction as con:
        if rate_package_id is not None:
            package_row = (
                con.execute(
                    text(
                        """
                        SELECT
                            package.model_id,
                            package.model_version,
                            package.source_export_id,
                            package.parent_rate_package_id,
                            package.build_fingerprint_sha256
                        FROM pricing.PRICING_RATE_PACKAGE AS package
                            WITH (UPDLOCK, HOLDLOCK)
                        WHERE package.rate_package_id = :rate_package_id
                        """
                    ),
                    {"rate_package_id": rate_package_id},
                )
                .mappings()
                .one_or_none()
            )
            if package_row is None:
                raise ModelRunIdentityError(
                    f"rate package ownership cannot be validated: {rate_package_id!r} not found"
                )
            expected_package = {
                "model_id": build.model_id,
                "model_version": build.model_version,
                "source_export_id": build.export_id,
            }
            package_mismatches = [
                field_name
                for field_name, expected_value in expected_package.items()
                if _identity_value(package_row[field_name]) != _identity_value(expected_value)
            ]
            if parent_model_run_id is None:
                if package_row["parent_rate_package_id"] is not None:
                    package_mismatches.append("parent_rate_package_id")
                if _identity_value(package_row["build_fingerprint_sha256"]) != _identity_value(
                    build.build_fingerprint_sha256
                ):
                    package_mismatches.append("build_fingerprint_sha256")
            if package_mismatches:
                raise ModelRunIdentityError(
                    "rate package ownership differs from the completed build: "
                    + ", ".join(package_mismatches)
                )

        if split_set_id is not None:
            stored_fold_rows = [
                dict(row)
                for row in con.execute(
                    text(
                        """
                        SELECT
                            split_set.manifest_id,
                            fold.fold_no,
                            fold.n_train,
                            fold.n_test
                        FROM pricing.CV_SPLIT_SET AS split_set
                            WITH (UPDLOCK, HOLDLOCK)
                        LEFT JOIN pricing.CV_FOLD AS fold WITH (UPDLOCK, HOLDLOCK)
                          ON fold.split_set_id = split_set.split_set_id
                        WHERE split_set.split_set_id = :split_set_id
                        ORDER BY fold.fold_no
                        """
                    ),
                    {"split_set_id": split_set_id},
                )
                .mappings()
                .all()
            ]
            if not stored_fold_rows:
                raise ModelRunIdentityError(f"split_set_id {split_set_id!r} does not exist")
            stored_manifests = {str(row["manifest_id"]) for row in stored_fold_rows}
            if stored_manifests != {manifest_id}:
                raise ModelRunIdentityError(
                    "split_set_id does not belong to completed-build manifest_id"
                )
            if build.validation_splits:
                expected_geometry = {
                    (
                        split.validation_split_no,
                        split.n_train,
                        split.n_validation,
                    )
                    for split in build.validation_splits
                }
                actual_geometry = {
                    (int(row["fold_no"]), int(row["n_train"]), int(row["n_test"]))
                    for row in stored_fold_rows
                    if row["fold_no"] is not None
                }
                if actual_geometry != expected_geometry:
                    raise ModelRunIdentityError(
                        "CV_FOLD geometry differs from completed-build validation splits: "
                        f"stored={sorted(actual_geometry)!r}, "
                        f"expected={sorted(expected_geometry)!r}"
                    )

        if parent_model_run_id is not None:
            derived_validation_source = con.execute(
                text(
                    """
                    SELECT TOP (1)
                        COALESCE(
                            parent_run.validation_source_model_run_id,
                            parent_run.model_run_id
                        )
                    FROM pricing.PRICING_RATE_PACKAGE AS child_package
                        WITH (UPDLOCK, HOLDLOCK)
                    JOIN pricing.MODEL_RUN AS parent_run
                        WITH (UPDLOCK, HOLDLOCK)
                      ON parent_run.model_run_id = :parent_model_run_id
                     AND parent_run.rate_package_id = child_package.parent_rate_package_id
                    WHERE child_package.rate_package_id = :rate_package_id
                      AND child_package.model_id = :model_id
                      AND parent_run.model_id = :model_id
                      AND parent_run.run_status = 'SUCCESS'
                    """
                ),
                params,
            ).scalar_one_or_none()
            if derived_validation_source is None:
                raise ModelRunIdentityError(
                    "parent_model_run_id does not match the package parent, model, "
                    "or a successful parent run"
                )
            derived_validation_source = int(derived_validation_source)
            params["validation_source_model_run_id"] = derived_validation_source

        existing_successful_run = (
            con.execute(
                text(
                    """
                    SELECT TOP (1)
                        mr.model_run_id,
                        mr.dag_id,
                        mr.airflow_run_id,
                        mr.mlflow_run_id,
                        mr.manifest_id,
                        mr.export_id,
                        mr.model_id,
                        mr.model_name,
                        mr.model_version,
                        mr.rate_package_id,
                        mr.rating_workbook_path,
                        mr.rating_workbook_sha256,
                        mr.run_status,
                        mr.created_by,
                        mr.publication_receipt_path,
                        mr.publication_receipt_sha256,
                        mr.candidate_artifact_path,
                        mr.candidate_artifact_sha256,
                        mr.candidate_artifact_format,
                        mr.candidate_artifact_size_bytes,
                        mr.candidate_python_version,
                        mr.candidate_superglm_version,
                        mr.candidate_superglm_git_sha,
                        mr.model_source_sha256,
                        mr.builder_source_sha256,
                        mr.materialized_split_sha256,
                        mr.runtime_sha256,
                        mr.candidate_superglm_sha256,
                        mr.validation_curve_status,
                        mr.validation_curve_reason,
                        mr.validation_source_model_run_id,
                        mr.parent_model_run_id
                    FROM pricing.MODEL_RUN AS mr WITH (UPDLOCK, HOLDLOCK)
                    WHERE mr.run_status = 'SUCCESS'
                      AND (
                          (
                              mr.dag_id = :dag_id
                              AND mr.airflow_run_id = :airflow_run_id
                              AND (
                                  mr.model_id = :model_id
                                  OR mr.model_name = :model_name
                              )
                          )
                          OR (
                              :rate_package_id IS NOT NULL
                              AND mr.rate_package_id = :rate_package_id
                          )
                      )
                    ORDER BY
                        CASE
                            WHEN mr.dag_id = :dag_id
                             AND mr.airflow_run_id = :airflow_run_id
                             AND mr.model_id = :model_id
                            THEN 0
                            WHEN mr.rate_package_id = :rate_package_id
                            THEN 1
                            ELSE 2
                        END,
                        mr.model_run_id
                    """
                ),
                params,
            )
            .mappings()
            .one_or_none()
        )
        if existing_successful_run is not None:
            mismatched_fields = [
                field_name
                for field_name in _IMMUTABLE_MODEL_RUN_FIELDS
                if _identity_value(existing_successful_run[field_name])
                != _identity_value(params[field_name])
            ]
            expected_validation_source = (
                existing_successful_run["model_run_id"]
                if parent_model_run_id is None
                else params["validation_source_model_run_id"]
            )
            if _identity_value(
                existing_successful_run["validation_source_model_run_id"]
            ) != _identity_value(expected_validation_source):
                mismatched_fields.append("validation_source_model_run_id")
            if mismatched_fields:
                raise ModelRunIdentityError(
                    "Existing successful model run has different immutable lineage: "
                    + ", ".join(mismatched_fields)
                )
            association_rows = (
                con.execute(
                    text(
                        """
                        SELECT
                            'actual_dataset' AS lineage_source,
                            dataset_link.manifest_id,
                            CAST(NULL AS NVARCHAR(128)) AS split_set_id,
                            dataset_link.dataset_role,
                            CAST(NULL AS NVARCHAR(64)) AS split_role
                        FROM mlops.MODEL_RUN_DATASET AS dataset_link
                            WITH (UPDLOCK, HOLDLOCK)
                        WHERE dataset_link.model_run_id = :model_run_id

                        UNION ALL

                        SELECT
                            'actual_split' AS lineage_source,
                            split_link.manifest_id,
                            split_link.split_set_id,
                            split_link.dataset_role,
                            split_link.split_role
                        FROM mlops.MODEL_RUN_SPLIT_SET AS split_link
                            WITH (UPDLOCK, HOLDLOCK)
                        WHERE split_link.model_run_id = :model_run_id

                        UNION ALL

                        SELECT
                            'parent_dataset' AS lineage_source,
                            parent_dataset.manifest_id,
                            CAST(NULL AS NVARCHAR(128)) AS split_set_id,
                            parent_dataset.dataset_role,
                            CAST(NULL AS NVARCHAR(64)) AS split_role
                        FROM mlops.MODEL_RUN_DATASET AS parent_dataset
                            WITH (UPDLOCK, HOLDLOCK)
                        WHERE parent_dataset.model_run_id = :parent_model_run_id

                        UNION ALL

                        SELECT
                            'parent_split' AS lineage_source,
                            parent_split.manifest_id,
                            parent_split.split_set_id,
                            parent_split.dataset_role,
                            parent_split.split_role
                        FROM mlops.MODEL_RUN_SPLIT_SET AS parent_split
                            WITH (UPDLOCK, HOLDLOCK)
                        WHERE parent_split.model_run_id = :parent_model_run_id
                        """
                    ),
                    {
                        "model_run_id": existing_successful_run["model_run_id"],
                        "parent_model_run_id": existing_successful_run["parent_model_run_id"],
                    },
                )
                .mappings()
                .all()
            )
            dataset_sets = {
                "actual_dataset": set(),
                "parent_dataset": set(),
            }
            split_sets = {
                "actual_split": set(),
                "parent_split": set(),
            }
            for row in association_rows:
                lineage_source = str(row["lineage_source"])
                if lineage_source in dataset_sets:
                    dataset_sets[lineage_source].add(
                        (str(row["manifest_id"]), str(row["dataset_role"]))
                    )
                elif lineage_source in split_sets:
                    split_sets[lineage_source].add(
                        (
                            str(row["manifest_id"]),
                            str(row["split_set_id"]),
                            str(row["dataset_role"]),
                            str(row["split_role"]),
                        )
                    )
                else:
                    raise ModelRunIdentityError(
                        f"Unknown model-run lineage source {lineage_source!r}"
                    )

            expected_datasets = set(dataset_sets["parent_dataset"])
            expected_datasets.add((manifest_id, _DATASET_ROLE))
            expected_splits = set(split_sets["parent_split"])
            if split_set_id is not None:
                expected_splits.add((manifest_id, split_set_id, _DATASET_ROLE, _SPLIT_ROLE))

            association_mismatches = []
            if dataset_sets["actual_dataset"] != expected_datasets:
                association_mismatches.append(
                    "dataset associations "
                    f"stored={sorted(dataset_sets['actual_dataset'])!r} "
                    f"expected={sorted(expected_datasets)!r}"
                )
            if split_sets["actual_split"] != expected_splits:
                association_mismatches.append(
                    "split associations "
                    f"stored={sorted(split_sets['actual_split'])!r} "
                    f"expected={sorted(expected_splits)!r}"
                )
            if association_mismatches:
                raise ModelRunIdentityError(
                    "Existing successful model run has different immutable lineage: "
                    + "; ".join(association_mismatches)
                )
            return int(existing_successful_run["model_run_id"])

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
                        rating_workbook_sha256 = :rating_workbook_sha256,
                        publication_receipt_path = :publication_receipt_path,
                        publication_receipt_sha256 = :publication_receipt_sha256,
                        candidate_artifact_path = :candidate_artifact_path,
                        candidate_artifact_sha256 = :candidate_artifact_sha256,
                        candidate_artifact_format = :candidate_artifact_format,
                        candidate_artifact_size_bytes = :candidate_artifact_size_bytes,
                        candidate_python_version = :candidate_python_version,
                        candidate_superglm_version = :candidate_superglm_version,
                        candidate_superglm_git_sha = :candidate_superglm_git_sha,
                        model_source_sha256 = :model_source_sha256,
                        builder_source_sha256 = :builder_source_sha256,
                        materialized_split_sha256 = :materialized_split_sha256,
                        runtime_sha256 = :runtime_sha256,
                        candidate_superglm_sha256 = :candidate_superglm_sha256,
                        validation_curve_status = :validation_curve_status,
                        validation_curve_reason = :validation_curve_reason,
                        parent_model_run_id = :parent_model_run_id,
                        validation_source_model_run_id = :validation_source_model_run_id,
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
                        rating_workbook_sha256,
                        publication_receipt_path,
                        publication_receipt_sha256,
                        candidate_artifact_path,
                        candidate_artifact_sha256,
                        candidate_artifact_format,
                        candidate_artifact_size_bytes,
                        candidate_python_version,
                        candidate_superglm_version,
                        candidate_superglm_git_sha,
                        model_source_sha256,
                        builder_source_sha256,
                        materialized_split_sha256,
                        runtime_sha256,
                        candidate_superglm_sha256,
                        validation_curve_status,
                        validation_curve_reason,
                        parent_model_run_id,
                        validation_source_model_run_id,
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
                        :rating_workbook_sha256,
                        :publication_receipt_path,
                        :publication_receipt_sha256,
                        :candidate_artifact_path,
                        :candidate_artifact_sha256,
                        :candidate_artifact_format,
                        :candidate_artifact_size_bytes,
                        :candidate_python_version,
                        :candidate_superglm_version,
                        :candidate_superglm_git_sha,
                        :model_source_sha256,
                        :builder_source_sha256,
                        :materialized_split_sha256,
                        :runtime_sha256,
                        :candidate_superglm_sha256,
                        :validation_curve_status,
                        :validation_curve_reason,
                        :parent_model_run_id,
                        :validation_source_model_run_id,
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
        if parent_model_run_id is None:
            con.execute(
                text(
                    """
                    UPDATE pricing.MODEL_RUN
                    SET validation_source_model_run_id = :model_run_id
                    WHERE model_run_id = :model_run_id
                    """
                ),
                {"model_run_id": model_run_id},
            )
        split_lineage_params = {
            "model_run_id": model_run_id,
            "manifest_id": manifest_id,
            "split_set_id": split_set_id,
            "dataset_role": _DATASET_ROLE,
            "split_role": _SPLIT_ROLE,
            "parent_model_run_id": parent_model_run_id,
        }
        con.execute(
            text(
                """
                DELETE split_link
                FROM mlops.MODEL_RUN_SPLIT_SET AS split_link
                WHERE split_link.model_run_id = :model_run_id
                  AND NOT (
                      (
                          :split_set_id IS NOT NULL
                          AND split_link.manifest_id = :manifest_id
                          AND split_link.split_set_id = :split_set_id
                          AND split_link.dataset_role = :dataset_role
                          AND split_link.split_role = :split_role
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM mlops.MODEL_RUN_SPLIT_SET AS parent_split
                          WHERE parent_split.model_run_id = :parent_model_run_id
                            AND parent_split.manifest_id = split_link.manifest_id
                            AND parent_split.split_set_id = split_link.split_set_id
                            AND parent_split.dataset_role = split_link.dataset_role
                            AND parent_split.split_role = split_link.split_role
                      )
                  );
                """
            ),
            split_lineage_params,
        )
        con.execute(
            text(
                """
                DELETE fold_metric
                FROM pricing.CV_FOLD_METRIC AS fold_metric
                WHERE fold_metric.model_run_id = :model_run_id
                  AND NOT EXISTS (
                      SELECT 1
                      FROM mlops.MODEL_RUN_SPLIT_SET AS split_reference
                      WHERE split_reference.model_run_id = fold_metric.model_run_id
                        AND split_reference.split_set_id = fold_metric.split_set_id
                  );
                """
            ),
            {"model_run_id": model_run_id},
        )
        con.execute(
            text(
                """
                DELETE dataset_link
                FROM mlops.MODEL_RUN_DATASET AS dataset_link
                WHERE dataset_link.model_run_id = :model_run_id
                  AND NOT (
                      (
                          dataset_link.manifest_id = :manifest_id
                          AND dataset_link.dataset_role = :dataset_role
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM mlops.MODEL_RUN_DATASET AS parent_dataset
                          WHERE parent_dataset.model_run_id = :parent_model_run_id
                            AND parent_dataset.manifest_id = dataset_link.manifest_id
                            AND parent_dataset.dataset_role = dataset_link.dataset_role
                      )
                  );
                """
            ),
            {
                "model_run_id": model_run_id,
                "manifest_id": manifest_id,
                "dataset_role": _DATASET_ROLE,
                "parent_model_run_id": parent_model_run_id,
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
                "dataset_role": _DATASET_ROLE,
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
                    "dataset_role": _DATASET_ROLE,
                    "split_role": _SPLIT_ROLE,
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
        con.execute(
            text(
                """
                DELETE FROM mlops.MODEL_RUN_METRIC
                WHERE model_run_id = :model_run_id;
                """
            ),
            {"model_run_id": model_run_id},
        )
        if split_set_id is not None:
            con.execute(
                text(
                    """
                    DELETE FROM pricing.CV_FOLD_METRIC
                    WHERE model_run_id = :model_run_id
                      AND split_set_id = :split_set_id;
                    """
                ),
                {
                    "model_run_id": model_run_id,
                    "split_set_id": split_set_id,
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
        con.execute(
            text(
                """
                DELETE FROM pricing.CV_SPLIT_CURVE_POINT
                WHERE model_run_id = :model_run_id;
                """
            ),
            {"model_run_id": model_run_id},
        )
        if build.validation_curve_status == "COMPLETE":
            curve_point_params = [
                {
                    "model_run_id": model_run_id,
                    "split_set_id": split_set_id,
                    "split_no": point.validation_split_no,
                    "term_name": point.term_name,
                    "point_no": point.point_no,
                    "point_kind": point.point_kind,
                    "x_numeric": point.x_numeric,
                    "level_text": point.level_text,
                    "eta_contribution": point.eta_contribution,
                    "relativity": point.relativity,
                    "support_value": point.support_value,
                    "reference_value": point.reference_value,
                    "reference_level": point.reference_level,
                }
                for point in build.validation_curve_points
            ]
            con.execute(
                text(
                    """
                    INSERT INTO pricing.CV_SPLIT_CURVE_POINT (
                        model_run_id,
                        split_set_id,
                        split_no,
                        term_name,
                        point_no,
                        point_kind,
                        x_numeric,
                        level_text,
                        eta_contribution,
                        relativity,
                        support_value,
                        reference_value,
                        reference_level
                    ) VALUES (
                        :model_run_id,
                        :split_set_id,
                        :split_no,
                        :term_name,
                        :point_no,
                        :point_kind,
                        :x_numeric,
                        :level_text,
                        :eta_contribution,
                        :relativity,
                        :support_value,
                        :reference_value,
                        :reference_level
                    );
                    """
                ),
                curve_point_params,
            )
    return int(model_run_id)
