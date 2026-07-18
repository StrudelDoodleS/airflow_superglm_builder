"""SQLite-specific writes used by the local analyst notebook context."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sqlalchemy import text

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra.offline_sqlite import local_publish_lock
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import ApprovedModelBuild, ApprovedModelBuildError
from pricing_pipeline.orchestration.publish_completed_build import (
    CompletedModelPublishResult,
    _verify_candidate_artifact,
    load_candidate_sql_lineage,
)
from pricing_pipeline.publishing.model_registry import (
    PricingModelRecord,
    validate_registered_model,
)
from pricing_pipeline.publishing.staging import stage_rating_export
from pricing_pipeline.workbench.submission import sha256_file


_VERSION_PATTERN = re.compile(r"^v([0-9]+)$")


def register_sqlite_model(
    engine,
    config: ModelBuildConfig,
    *,
    created_by: str,
) -> PricingModelRecord:
    """Insert once and validate the complete stable model identity."""
    params = {
        "model_name": config.model_name,
        "model_label": config.model_label,
        "target_name": config.target_name,
        "model_type": config.model_type,
        "created_by": created_by,
    }
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT OR IGNORE INTO pricing.PRICING_MODEL (
                    model_name,
                    model_label,
                    target_name,
                    model_type,
                    model_status,
                    created_by
                ) VALUES (
                    :model_name,
                    :model_label,
                    :target_name,
                    :model_type,
                    'ACTIVE',
                    :created_by
                )
                """
            ),
            params,
        )
        return validate_registered_model(connection, config)


def resolve_sqlite_model_version(
    engine,
    *,
    model_name: str,
    export_id: str,
) -> str:
    """Transactionally reserve one trained version for an export."""
    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            model_id = int(
                connection.execute(
                    text(
                        """
                        SELECT model_id
                        FROM pricing.PRICING_MODEL
                        WHERE model_name = :model_name
                        """
                    ),
                    {"model_name": model_name},
                ).scalar_one()
            )
            existing = connection.execute(
                text(
                    """
                    SELECT model_version
                    FROM pricing.PRICING_RATE_PACKAGE
                    WHERE model_id = :model_id
                      AND source_export_id = :export_id
                    UNION ALL
                    SELECT model_version
                    FROM pricing.PRICING_MODEL_VERSION_RESERVATION
                    WHERE model_id = :model_id
                      AND export_id = :export_id
                    LIMIT 1
                    """
                ),
                {"model_id": model_id, "export_id": export_id},
            ).scalar_one_or_none()
            if existing is not None:
                connection.commit()
                return str(existing)

            versions = connection.execute(
                text(
                    """
                    SELECT model_version
                    FROM pricing.PRICING_RATE_PACKAGE
                    WHERE model_id = :model_id
                      AND parent_rate_package_id IS NULL
                    UNION ALL
                    SELECT model_version
                    FROM pricing.PRICING_MODEL_VERSION_RESERVATION
                    WHERE model_id = :model_id
                    """
                ),
                {"model_id": model_id},
            ).scalars()
            numbers = [
                int(match.group(1))
                for value in versions
                if (match := _VERSION_PATTERN.match(str(value))) is not None
            ]
            reserved = f"v{max(numbers, default=0) + 1}"
            connection.execute(
                text(
                    """
                    INSERT INTO pricing.PRICING_MODEL_VERSION_RESERVATION (
                        model_id, export_id, model_version
                    ) VALUES (
                        :model_id, :export_id, :model_version
                    )
                    """
                ),
                {
                    "model_id": model_id,
                    "export_id": export_id,
                    "model_version": reserved,
                },
            )
            connection.commit()
            return reserved
        except BaseException:
            connection.rollback()
            raise


def publish_sqlite_candidate(
    engine,
    *,
    settings: Settings,
    model_id: int,
    model_config: ModelBuildConfig,
    completed_build: ApprovedModelBuild,
    created_by: str,
) -> CompletedModelPublishResult:
    """Publish notebook audit evidence into the persistent local SQLite store."""
    local_root = Path(settings.workbench_artifact_root).resolve().parent
    with local_publish_lock(local_root):
        return _publish_sqlite_candidate_locked(
            engine,
            model_id=model_id,
            model_config=model_config,
            completed_build=completed_build,
            created_by=created_by,
            artifact_root=settings.workbench_artifact_root,
        )


def _publish_sqlite_candidate_locked(
    engine,
    *,
    model_id: int,
    model_config: ModelBuildConfig,
    completed_build: ApprovedModelBuild,
    created_by: str,
    artifact_root: str | Path,
) -> CompletedModelPublishResult:
    if not isinstance(completed_build, ApprovedModelBuild):
        raise TypeError("completed_build must be an ApprovedModelBuild")
    build = completed_build
    mismatches = []
    for field_name, actual, expected in (
        ("model_id", build.model_id, model_id),
        ("model_name", build.model_name, model_config.model_name),
        ("model_type", build.model_type, model_config.model_type),
        ("target_name", build.target_name, model_config.target_name),
        ("deployment_slot", build.deployment_slot, model_config.deployment_slot),
    ):
        if actual != expected:
            mismatches.append(f"{field_name} build={actual!r} registered={expected!r}")
    if mismatches:
        raise ApprovedModelBuildError(
            "approved build does not match the registered model: " + "; ".join(mismatches)
        )
    workbook_path = Path(build.rating_workbook_path)
    if not workbook_path.is_file():
        raise ApprovedModelBuildError(
            f"rating_workbook_path does not exist: {workbook_path.as_posix()}"
        )
    actual_workbook_sha256 = sha256_file(workbook_path)
    if actual_workbook_sha256 != build.rating_workbook_sha256:
        raise ApprovedModelBuildError(
            "rating workbook SHA-256 does not match the completed build: "
            f"expected={build.rating_workbook_sha256!r}, actual={actual_workbook_sha256!r}"
        )
    manifest_id = str(build.manifest_id or "").strip()
    if not manifest_id:
        raise ApprovedModelBuildError("local notebook publication requires an existing manifest_id")
    split_set_id = None if build.split_set_id is None else str(build.split_set_id).strip()
    export_id = str(build.export_id or "").strip()
    if not export_id:
        raise ApprovedModelBuildError("local notebook publication requires an export_id")
    if build.candidate_artifact_path is not None:
        sql_lineage = load_candidate_sql_lineage(
            engine,
            manifest_id=manifest_id,
            split_set_id=split_set_id,
        )
        _verify_candidate_artifact(
            build,
            sql_lineage=sql_lineage,
            allowed_root=artifact_root,
        )

    stage_rating_export(
        engine,
        workbook_path=Path(build.rating_workbook_path),
        export_id=export_id,
        expected_database=None,
        model_name=model_config.model_name,
        model_version=build.model_version,
        effective_from=build.effective_from,
        target_name=model_config.target_name,
        model_type=model_config.model_type,
        created_by=created_by,
        replace=True,
        model_id=model_id,
        publication_receipt_path=build.publication_receipt_path,
        publication_receipt_sha256=build.publication_receipt_sha256,
    )
    staged_workbook_sha256 = sha256_file(workbook_path)
    if staged_workbook_sha256 != build.rating_workbook_sha256:
        raise ApprovedModelBuildError(
            "rating workbook changed during local staging: "
            f"expected={build.rating_workbook_sha256!r}, actual={staged_workbook_sha256!r}"
        )

    with engine.begin() as connection:
        staged = (
            connection.execute(
                text(
                    """
                    SELECT *
                    FROM pricing_stg.STG_RATING_EXPORT
                    WHERE export_id = :export_id
                    """
                ),
                {"export_id": export_id},
            )
            .mappings()
            .one()
        )
        existing = _existing_local_publication(
            connection,
            model_id=model_id,
            export_id=export_id,
        )
        staged_conflicts = _staged_export_conflicts(
            staged,
            model_id=model_id,
            model_config=model_config,
            build=build,
            export_id=export_id,
        )
        if staged_conflicts:
            raise ValueError(
                f"export_id {export_id!r} has incompatible staged evidence: "
                + "; ".join(staged_conflicts)
            )
        if existing is not None:
            conflicts = _local_publication_conflicts(
                existing,
                build=build,
                manifest_id=manifest_id,
                split_set_id=split_set_id,
                staging_content_sha256=staged["staging_content_sha256"],
            )
            if conflicts:
                raise ValueError(
                    f"export_id {export_id!r} has incompatible publication evidence: "
                    + "; ".join(conflicts)
                )
            run_conflicts = _model_run_evidence_conflicts(
                connection,
                existing,
                build=build,
                export_id=export_id,
                manifest_id=manifest_id,
                split_set_id=split_set_id,
            )
            if run_conflicts:
                raise ValueError(
                    f"export_id {export_id!r} has incompatible model-run evidence: "
                    + "; ".join(run_conflicts)
                )
            return _local_publish_result(
                model_id=model_id,
                model_config=model_config,
                package_row=existing,
                was_existing=True,
            )

        reserved_version = connection.execute(
            text(
                """
                SELECT model_version
                FROM pricing.PRICING_MODEL_VERSION_RESERVATION
                WHERE model_id = :model_id
                  AND export_id = :export_id
                """
            ),
            {"model_id": model_id, "export_id": export_id},
        ).scalar_one_or_none()
        if reserved_version is None:
            raise ApprovedModelBuildError(
                f"local export {export_id!r} has no reserved model version; "
                "build it through build_candidate before publication"
            )
        if str(reserved_version) != build.model_version:
            raise ApprovedModelBuildError(
                f"local export {export_id!r} reserved model version "
                f"{reserved_version!r}, not {build.model_version!r}"
            )

        manifest_exists = connection.execute(
            text("SELECT 1 FROM pricing.DATASET_MANIFEST WHERE manifest_id = :manifest_id"),
            {"manifest_id": manifest_id},
        ).scalar_one_or_none()
        if manifest_exists is None:
            raise ApprovedModelBuildError(f"local manifest_id {manifest_id!r} does not exist")
        if split_set_id is not None:
            split_exists = connection.execute(
                text(
                    "SELECT 1 FROM pricing.CV_SPLIT_SET "
                    "WHERE split_set_id = :split_set_id "
                    "AND manifest_id = :manifest_id"
                ),
                {"split_set_id": split_set_id, "manifest_id": manifest_id},
            ).scalar_one_or_none()
            if split_exists is None:
                raise ApprovedModelBuildError(
                    f"local split_set_id {split_set_id!r} does not match the manifest"
                )

        package_version = int(
            connection.execute(
                text(
                    """
                    SELECT COALESCE(MAX(package_version), 0) + 1
                    FROM pricing.PRICING_RATE_PACKAGE
                    WHERE model_id = :model_id
                    """
                ),
                {"model_id": model_id},
            ).scalar_one()
        )
        package_insert = connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_RATE_PACKAGE (
                    parent_rate_package_id,
                    model_id,
                    model_name,
                    model_version,
                    package_version,
                    base_rate,
                    effective_from_date,
                    effective_to_date,
                    package_status,
                    source_export_id,
                    source_file,
                    publication_receipt_json,
                    publication_receipt_sha256,
                    staging_content_sha256,
                    package_metadata_json,
                    offset_handling,
                    offset_factor_name,
                    offset_source_name,
                    offset_label,
                    metadata_origin,
                    manifest_id,
                    split_set_id,
                    rating_workbook_path,
                    model_artifact_path,
                    created_by
                ) VALUES (
                    NULL,
                    :model_id,
                    :model_name,
                    :model_version,
                    :package_version,
                    :base_rate,
                    :effective_from_date,
                    :effective_to_date,
                    :package_status,
                    :source_export_id,
                    :source_file,
                    :publication_receipt_json,
                    :publication_receipt_sha256,
                    :staging_content_sha256,
                    :package_metadata_json,
                    :offset_handling,
                    :offset_factor_name,
                    :offset_source_name,
                    :offset_label,
                    :metadata_origin,
                    :manifest_id,
                    :split_set_id,
                    :rating_workbook_path,
                    :model_artifact_path,
                    :created_by
                )
                """
            ),
            {
                "model_id": model_id,
                "model_name": model_config.model_name,
                "model_version": build.model_version,
                "package_version": package_version,
                "base_rate": staged["base_rate"],
                "effective_from_date": staged["effective_from_date"],
                "effective_to_date": staged["effective_to_date"],
                "package_status": "LOCAL_AUDIT",
                "source_export_id": export_id,
                "source_file": staged["source_file"],
                "publication_receipt_json": staged["publication_receipt_json"],
                "publication_receipt_sha256": staged["publication_receipt_sha256"],
                "staging_content_sha256": staged["staging_content_sha256"],
                "package_metadata_json": staged["package_metadata_json"],
                "offset_handling": staged["offset_handling"] or "UNKNOWN",
                "offset_factor_name": staged["offset_factor_name"],
                "offset_source_name": staged["offset_source_name"],
                "offset_label": staged["offset_label"],
                "metadata_origin": staged["metadata_origin"],
                "manifest_id": manifest_id,
                "split_set_id": split_set_id,
                "rating_workbook_path": build.rating_workbook_path,
                "model_artifact_path": None,
                "created_by": created_by,
            },
        )
        rate_package_id = int(package_insert.lastrowid)
        model_run_id = rate_package_id
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_RUN (
                    model_run_id,
                    model_id,
                    dag_id,
                    airflow_run_id,
                    mlflow_run_id,
                    model_version,
                    export_id,
                    manifest_id,
                    split_set_id,
                    rate_package_id,
                    model_name,
                    rating_workbook_path,
                    rating_workbook_sha256,
                    publication_receipt_path,
                    publication_receipt_sha256,
                    model_artifact_path,
                    candidate_artifact_path,
                    candidate_artifact_sha256,
                    candidate_artifact_format,
                    candidate_artifact_size_bytes,
                    candidate_python_version,
                    candidate_superglm_version,
                    candidate_superglm_git_sha,
                    model_source_sha256,
                    effective_from,
                    run_status,
                    completed_ts,
                    created_by
                ) VALUES (
                    :model_run_id,
                    :model_id,
                    'notebook_local',
                    :airflow_run_id,
                    :mlflow_run_id,
                    :model_version,
                    :export_id,
                    :manifest_id,
                    :split_set_id,
                    :rate_package_id,
                    :model_name,
                    :rating_workbook_path,
                    :rating_workbook_sha256,
                    :publication_receipt_path,
                    :publication_receipt_sha256,
                    :model_artifact_path,
                    :candidate_artifact_path,
                    :candidate_artifact_sha256,
                    :candidate_artifact_format,
                    :candidate_artifact_size_bytes,
                    :candidate_python_version,
                    :candidate_superglm_version,
                    :candidate_superglm_git_sha,
                    :model_source_sha256,
                    :effective_from,
                    'SUCCESS',
                    CURRENT_TIMESTAMP,
                    :created_by
                )
                """
            ),
            {
                "model_run_id": model_run_id,
                "model_id": model_id,
                "airflow_run_id": export_id,
                "mlflow_run_id": build.mlflow_run_id,
                "model_version": build.model_version,
                "export_id": export_id,
                "manifest_id": manifest_id,
                "split_set_id": split_set_id,
                "rate_package_id": rate_package_id,
                "model_name": model_config.model_name,
                "rating_workbook_path": build.rating_workbook_path,
                "rating_workbook_sha256": build.rating_workbook_sha256,
                "publication_receipt_path": build.publication_receipt_path,
                "publication_receipt_sha256": build.publication_receipt_sha256,
                "model_artifact_path": None,
                "candidate_artifact_path": build.candidate_artifact_path,
                "candidate_artifact_sha256": build.candidate_artifact_sha256,
                "candidate_artifact_format": build.candidate_artifact_format,
                "candidate_artifact_size_bytes": build.candidate_artifact_size_bytes,
                "candidate_python_version": build.candidate_python_version,
                "candidate_superglm_version": build.candidate_superglm_version,
                "candidate_superglm_git_sha": build.candidate_superglm_git_sha,
                "model_source_sha256": build.model_source_sha256,
                "effective_from": build.effective_from,
                "created_by": created_by,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO mlops.MODEL_RUN_DATASET (
                    model_run_id, manifest_id, dataset_role
                ) VALUES (
                    :model_run_id, :manifest_id, 'training'
                )
                """
            ),
            {"model_run_id": model_run_id, "manifest_id": manifest_id},
        )
        if split_set_id is not None:
            connection.execute(
                text(
                    """
                    INSERT INTO mlops.MODEL_RUN_SPLIT_SET (
                        model_run_id, manifest_id, split_set_id,
                        dataset_role, split_role
                    ) VALUES (
                        :model_run_id, :manifest_id, :split_set_id,
                        'training', 'validation'
                    )
                    """
                ),
                {
                    "model_run_id": model_run_id,
                    "manifest_id": manifest_id,
                    "split_set_id": split_set_id,
                },
            )
        for metric_name, metric_value in sorted(build.metrics.items()):
            connection.execute(
                text(
                    """
                    INSERT INTO mlops.MODEL_RUN_METRIC (
                        model_run_id, metric_name, metric_value, metric_scope
                    ) VALUES (
                        :model_run_id, :metric_name, :metric_value, :metric_scope
                    )
                    """
                ),
                {
                    "model_run_id": model_run_id,
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "metric_scope": build.metric_scopes.get(metric_name, "model_run"),
                },
            )
        if build.fold_metrics and split_set_id is None:
            raise ApprovedModelBuildError(
                "fold metrics require split_set_id in local notebook publication"
            )
        for metric in build.fold_metrics:
            connection.execute(
                text(
                    """
                    INSERT INTO pricing.CV_FOLD_METRIC (
                        model_run_id, split_set_id, fold_no,
                        metric_name, metric_value
                    ) VALUES (
                        :model_run_id, :split_set_id, :fold_no,
                        :metric_name, :metric_value
                    )
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

        created = _existing_local_publication(
            connection,
            model_id=model_id,
            export_id=export_id,
        )
        if created is None:
            raise RuntimeError("Local publication was not visible after insert")
        return _local_publish_result(
            model_id=model_id,
            model_config=model_config,
            package_row=created,
            was_existing=False,
        )


def _existing_local_publication(
    connection,
    *,
    model_id: int,
    export_id: str,
):
    return (
        connection.execute(
            text(
                """
                SELECT
                    rp.rate_package_id,
                    rp.package_version,
                    rp.package_status,
                    rp.model_version,
                    rp.manifest_id,
                    rp.split_set_id,
                    rp.rating_workbook_path,
                    rp.publication_receipt_sha256,
                    rp.staging_content_sha256,
                    rp.source_export_id,
                    mr.model_run_id,
                    mr.run_status,
                    mr.rating_workbook_sha256,
                    mr.airflow_run_id,
                    mr.mlflow_run_id,
                    mr.publication_receipt_path,
                    mr.model_artifact_path,
                    mr.candidate_artifact_path,
                    mr.candidate_artifact_sha256,
                    mr.candidate_artifact_format,
                    mr.candidate_artifact_size_bytes,
                    mr.candidate_python_version,
                    mr.candidate_superglm_version,
                    mr.candidate_superglm_git_sha,
                    mr.model_source_sha256,
                    mr.effective_from
                FROM pricing.PRICING_RATE_PACKAGE AS rp
                LEFT JOIN pricing.MODEL_RUN AS mr
                  ON mr.rate_package_id = rp.rate_package_id
                WHERE rp.model_id = :model_id
                  AND rp.source_export_id = :export_id
                """
            ),
            {"model_id": model_id, "export_id": export_id},
        )
        .mappings()
        .one_or_none()
    )


def _staged_export_conflicts(
    staged,
    *,
    model_id: int,
    model_config: ModelBuildConfig,
    build: ApprovedModelBuild,
    export_id: str,
) -> list[str]:
    expected = {
        "export_id": export_id,
        "model_id": model_id,
        "model_name": model_config.model_name,
        "model_version": build.model_version,
        "effective_from_date": build.effective_from,
        "source_file": str(Path(build.rating_workbook_path).resolve()),
        "publication_receipt_sha256": build.publication_receipt_sha256,
    }
    conflicts = []
    for field_name, expected_value in expected.items():
        if _identity(staged[field_name]) != _identity(expected_value):
            conflicts.append(
                f"{field_name} staged={staged[field_name]!r} requested={expected_value!r}"
            )
    return conflicts


def _local_publication_conflicts(
    existing,
    *,
    build: ApprovedModelBuild,
    manifest_id: str,
    split_set_id: str | None,
    staging_content_sha256: str | None,
) -> list[str]:
    expected = {
        "model_version": build.model_version,
        "manifest_id": manifest_id,
        "split_set_id": split_set_id,
        "rating_workbook_path": build.rating_workbook_path,
        "rating_workbook_sha256": build.rating_workbook_sha256,
        "publication_receipt_sha256": build.publication_receipt_sha256,
    }
    conflicts = []
    for field_name, expected_value in expected.items():
        existing_value = existing[field_name]
        if (None if existing_value is None else str(existing_value)) != (
            None if expected_value is None else str(expected_value)
        ):
            conflicts.append(
                f"{field_name} existing={existing_value!r} requested={expected_value!r}"
            )
    existing_staging_digest = _identity(existing["staging_content_sha256"])
    requested_staging_digest = _identity(staging_content_sha256)
    if existing_staging_digest is not None and existing_staging_digest != requested_staging_digest:
        conflicts.append(
            "staging_content_sha256 "
            f"existing={existing_staging_digest!r} "
            f"requested={requested_staging_digest!r}"
        )
    return conflicts


def _model_run_evidence_conflicts(
    connection,
    existing,
    *,
    build: ApprovedModelBuild,
    export_id: str,
    manifest_id: str,
    split_set_id: str | None,
) -> list[str]:
    model_run_id = existing["model_run_id"]
    if model_run_id is None:
        raise RuntimeError(
            f"incomplete local publication lineage: export {export_id!r} has no model run"
        )
    if str(existing["run_status"] or "").upper() != "SUCCESS":
        raise RuntimeError(
            f"incomplete local publication lineage: export {export_id!r} "
            "has no successful model run"
        )
    _validate_existing_lineage_links(
        connection,
        model_run_id=model_run_id,
        manifest_id=manifest_id,
        split_set_id=split_set_id,
    )

    expected_scalars = {
        "airflow_run_id": export_id,
        "mlflow_run_id": build.mlflow_run_id,
        "publication_receipt_path": build.publication_receipt_path,
        "publication_receipt_sha256": build.publication_receipt_sha256,
        "model_artifact_path": None,
        "candidate_artifact_path": build.candidate_artifact_path,
        "candidate_artifact_sha256": build.candidate_artifact_sha256,
        "candidate_artifact_format": build.candidate_artifact_format,
        "candidate_artifact_size_bytes": build.candidate_artifact_size_bytes,
        "candidate_python_version": build.candidate_python_version,
        "candidate_superglm_version": build.candidate_superglm_version,
        "candidate_superglm_git_sha": build.candidate_superglm_git_sha,
        "model_source_sha256": build.model_source_sha256,
        "effective_from": build.effective_from,
    }
    conflicts = []
    for field_name, expected_value in expected_scalars.items():
        if _identity(existing[field_name]) != _identity(expected_value):
            conflicts.append(
                f"{field_name} existing={existing[field_name]!r} requested={expected_value!r}"
            )

    stored_metrics = {
        str(row[0]): (float(row[1]), _identity(row[2]))
        for row in connection.execute(
            text(
                """
                SELECT metric_name, metric_value, metric_scope
                FROM mlops.MODEL_RUN_METRIC
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": model_run_id},
        ).all()
    }
    expected_metrics = {
        str(name): (
            float(value),
            _identity(build.metric_scopes.get(name, "model_run")),
        )
        for name, value in build.metrics.items()
    }
    if stored_metrics != expected_metrics:
        conflicts.append(f"metrics existing={stored_metrics!r} requested={expected_metrics!r}")

    stored_fold_metrics = {
        (int(row[0]), str(row[1]), float(row[2]))
        for row in connection.execute(
            text(
                """
                SELECT fold_no, metric_name, metric_value
                FROM pricing.CV_FOLD_METRIC
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": model_run_id},
        ).all()
    }
    expected_fold_metrics = {
        (
            int(metric["fold_no"]),
            str(metric["metric_name"]),
            float(metric["metric_value"]),
        )
        for metric in build.fold_metrics
    }
    if stored_fold_metrics != expected_fold_metrics:
        conflicts.append(
            f"fold_metrics existing={stored_fold_metrics!r} requested={expected_fold_metrics!r}"
        )
    return conflicts


def _validate_existing_lineage_links(
    connection,
    *,
    model_run_id: Any,
    manifest_id: str,
    split_set_id: str | None,
) -> None:
    dataset_links = set(
        connection.execute(
            text(
                """
                SELECT manifest_id, dataset_role
                FROM mlops.MODEL_RUN_DATASET
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": model_run_id},
        ).all()
    )
    expected_dataset_links = {(manifest_id, "training")}
    split_links = set(
        connection.execute(
            text(
                """
                SELECT manifest_id, split_set_id, dataset_role, split_role
                FROM mlops.MODEL_RUN_SPLIT_SET
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": model_run_id},
        ).all()
    )
    expected_split_links = (
        set() if split_set_id is None else {(manifest_id, split_set_id, "training", "validation")}
    )
    if dataset_links != expected_dataset_links or split_links != expected_split_links:
        raise RuntimeError(
            "incomplete local publication lineage: dataset/split links do not "
            "match the immutable model run"
        )


def _local_publish_result(
    *,
    model_id: int,
    model_config: ModelBuildConfig,
    package_row,
    was_existing: bool,
) -> CompletedModelPublishResult:
    model_run_id = package_row["model_run_id"]
    if model_run_id is None:
        raise RuntimeError(
            f"Local export {package_row['source_export_id']!r} has a package "
            "without model-run audit rows"
        )
    return CompletedModelPublishResult(
        model_id=model_id,
        model_name=model_config.model_name,
        model_version=str(package_row["model_version"]),
        manifest_id=str(package_row["manifest_id"]),
        split_set_id=(
            None if package_row["split_set_id"] is None else str(package_row["split_set_id"])
        ),
        export_id=str(package_row["source_export_id"]),
        rate_package_id=int(package_row["rate_package_id"]),
        package_version=int(package_row["package_version"]),
        package_status=str(package_row["package_status"]),
        rating_workbook_path=str(package_row["rating_workbook_path"]),
        model_run_id=int(model_run_id),
        mlflow_run_id=(
            None if package_row["mlflow_run_id"] is None else str(package_row["mlflow_run_id"])
        ),
        publication_receipt_path=(
            None
            if package_row["publication_receipt_path"] is None
            else str(package_row["publication_receipt_path"])
        ),
        publication_receipt_sha256=(
            None
            if package_row["publication_receipt_sha256"] is None
            else str(package_row["publication_receipt_sha256"])
        ),
        was_existing=was_existing,
    )


def _identity(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value)
    return cleaned or None
