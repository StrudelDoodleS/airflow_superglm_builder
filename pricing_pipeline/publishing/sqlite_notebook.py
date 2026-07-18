"""SQLite-specific writes used by the local analyst notebook context."""

from __future__ import annotations

from collections import Counter
import json
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
    _discard_redundant_completed_build_attempt,
    _verify_candidate_artifact,
    load_candidate_sql_lineage,
)
from pricing_pipeline.publishing.model_registry import (
    PricingModelRecord,
    validate_registered_model,
)
from pricing_pipeline.publishing.staging import stage_rating_export
from pricing_pipeline.workbench.submission import sha256_file, xlsx_semantic_sha256


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
    build_fingerprint_sha256: str,
) -> str:
    """Transactionally reserve one trained version for a stable root build."""
    fingerprint = str(build_fingerprint_sha256).strip()
    if (
        len(fingerprint) != 64
        or fingerprint.lower() != fingerprint
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError(
            "build_fingerprint_sha256 must be a 64-character lowercase hex SHA-256 digest"
        )
    reservation_export_id = f"build_{fingerprint}"
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
            canonical_version = connection.execute(
                text(
                    """
                    SELECT model_version
                    FROM pricing.PRICING_RATE_PACKAGE
                    WHERE model_id = :model_id
                      AND parent_rate_package_id IS NULL
                      AND build_fingerprint_sha256 = :build_fingerprint_sha256
                    """
                ),
                {
                    "model_id": model_id,
                    "build_fingerprint_sha256": fingerprint,
                },
            ).scalar_one_or_none()
            reserved_version = connection.execute(
                text(
                    """
                    SELECT model_version
                    FROM pricing.PRICING_MODEL_VERSION_RESERVATION
                    WHERE model_id = :model_id
                      AND export_id = :reservation_export_id
                    """
                ),
                {
                    "model_id": model_id,
                    "reservation_export_id": reservation_export_id,
                },
            ).scalar_one_or_none()
            if (
                canonical_version is not None
                and reserved_version is not None
                and str(canonical_version) != str(reserved_version)
            ):
                raise RuntimeError(
                    "canonical root model version disagrees with its fingerprint reservation: "
                    f"canonical={canonical_version!r}, reservation={reserved_version!r}"
                )
            existing_version = canonical_version or reserved_version
            if existing_version is not None:
                connection.commit()
                return str(existing_version)

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
                    "export_id": reservation_export_id,
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
        result = _publish_sqlite_candidate_locked(
            engine,
            model_id=model_id,
            model_config=model_config,
            completed_build=completed_build,
            created_by=created_by,
            artifact_root=settings.workbench_artifact_root,
        )
        _discard_redundant_completed_build_attempt(
            completed_build,
            publish_result=result,
            artifact_root=settings.workbench_artifact_root,
        )
        return result


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
    workbook_path = _verify_local_audit_artifact(
        build.rating_workbook_path,
        expected_sha256=build.rating_workbook_sha256,
        artifact_root=artifact_root,
        label="rating workbook",
    )
    receipt_path = _verify_local_audit_artifact(
        build.publication_receipt_path,
        expected_sha256=build.publication_receipt_sha256,
        artifact_root=artifact_root,
        label="publication receipt",
    )
    manifest_id = str(build.manifest_id or "").strip()
    if not manifest_id:
        raise ApprovedModelBuildError("local notebook publication requires an existing manifest_id")
    split_set_id = None if build.split_set_id is None else str(build.split_set_id).strip()
    export_id = str(build.export_id or "").strip()
    if not export_id:
        raise ApprovedModelBuildError("local notebook publication requires an export_id")
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
        workbook_path=workbook_path,
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
        publication_receipt_path=receipt_path,
        publication_receipt_sha256=build.publication_receipt_sha256,
    )
    staged_workbook_sha256 = sha256_file(workbook_path)
    if staged_workbook_sha256 != build.rating_workbook_sha256:
        raise ApprovedModelBuildError(
            "rating workbook changed during local staging: "
            f"expected={build.rating_workbook_sha256!r}, actual={staged_workbook_sha256!r}"
        )
    staged_receipt_sha256 = sha256_file(receipt_path)
    if staged_receipt_sha256 != build.publication_receipt_sha256:
        raise ApprovedModelBuildError(
            "publication receipt changed during local staging: "
            f"expected={build.publication_receipt_sha256!r}, "
            f"actual={staged_receipt_sha256!r}"
        )
    _verify_candidate_artifact(
        build,
        sql_lineage=sql_lineage,
        allowed_root=artifact_root,
    )

    with engine.begin() as connection:
        registered_model = validate_registered_model(connection, model_config)
        if registered_model.model_id != model_id:
            raise ApprovedModelBuildError(
                "registered model_id does not match the notebook model: "
                f"registered={registered_model.model_id}, notebook={model_id}"
            )
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
        requested_lineage = _local_material_lineage_contract(
            connection,
            manifest_id=manifest_id,
            split_set_id=split_set_id,
        )
        _validate_validation_split_geometry(
            connection,
            build=build,
            split_set_id=split_set_id,
        )
        reserved_version = connection.execute(
            text(
                """
                SELECT model_version
                FROM pricing.PRICING_MODEL_VERSION_RESERVATION
                WHERE model_id = :model_id
                  AND export_id = :reservation_export_id
                """
            ),
            {
                "model_id": model_id,
                "reservation_export_id": f"build_{build.build_fingerprint_sha256}",
            },
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
        existing = _existing_local_publication(
            connection,
            model_id=model_id,
            build_fingerprint_sha256=build.build_fingerprint_sha256,
        )
        if existing is not None:
            run_conflicts = _model_run_evidence_conflicts(
                connection,
                existing,
                build=build,
                requested_lineage=requested_lineage,
            )
            if run_conflicts:
                raise ValueError(
                    "build fingerprint has incompatible model-run evidence: "
                    + "; ".join(run_conflicts)
                )
            canonical_workbook_path = _verify_canonical_publication_artifacts(
                existing,
                artifact_root=artifact_root,
            )
            _verify_canonical_candidate_artifact(
                engine,
                existing=existing,
                incoming_build=build,
                artifact_root=artifact_root,
            )
            conflicts = _local_publication_conflicts(
                existing,
                staged=staged,
                build=build,
                canonical_workbook_path=canonical_workbook_path,
            )
            if conflicts:
                raise ValueError(
                    "build fingerprint has incompatible publication evidence: "
                    + "; ".join(conflicts)
                )
            return _local_publish_result(
                model_id=model_id,
                model_config=model_config,
                package_row=existing,
                was_existing=True,
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
                    build_fingerprint_sha256,
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
                    :build_fingerprint_sha256,
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
                "build_fingerprint_sha256": build.build_fingerprint_sha256,
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
                    builder_source_sha256,
                    materialized_split_sha256,
                    runtime_sha256,
                    candidate_superglm_sha256,
                    validation_curve_status,
                    validation_curve_reason,
                    validation_source_model_run_id,
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
                    :builder_source_sha256,
                    :materialized_split_sha256,
                    :runtime_sha256,
                    :candidate_superglm_sha256,
                    :validation_curve_status,
                    :validation_curve_reason,
                    :validation_source_model_run_id,
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
                "builder_source_sha256": build.builder_source_sha256,
                "materialized_split_sha256": build.materialized_split_sha256,
                "runtime_sha256": build.runtime_sha256,
                "candidate_superglm_sha256": build.candidate_superglm_sha256,
                "validation_curve_status": build.validation_curve_status,
                "validation_curve_reason": build.validation_curve_reason,
                "validation_source_model_run_id": model_run_id,
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
        metric_params = [
            {
                "model_run_id": model_run_id,
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "metric_scope": build.metric_scopes.get(metric_name, "model_run"),
            }
            for metric_name, metric_value in sorted(build.metrics.items())
        ]
        if metric_params:
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
                metric_params,
            )
        split_metric_params = [
            {
                "model_run_id": model_run_id,
                "split_set_id": split_set_id,
                "fold_no": split.validation_split_no,
                "metric_name": metric_name,
                "metric_value": metric_value,
            }
            for split in build.validation_splits
            for metric_name, metric_value in split.metrics.items()
        ]
        if split_metric_params:
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
                split_metric_params,
            )
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
        if curve_point_params:
            connection.execute(
                text(
                    """
                    INSERT INTO pricing.CV_SPLIT_CURVE_POINT (
                        model_run_id, split_set_id, split_no,
                        term_name, point_no, point_kind,
                        x_numeric, level_text, eta_contribution,
                        relativity, support_value, reference_value,
                        reference_level
                    ) VALUES (
                        :model_run_id, :split_set_id, :split_no,
                        :term_name, :point_no, :point_kind,
                        :x_numeric, :level_text, :eta_contribution,
                        :relativity, :support_value, :reference_value,
                        :reference_level
                    )
                    """
                ),
                curve_point_params,
            )

        created = _existing_local_publication(
            connection,
            model_id=model_id,
            build_fingerprint_sha256=build.build_fingerprint_sha256,
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
    build_fingerprint_sha256: str,
):
    return (
        connection.execute(
            text(
                """
                SELECT
                    rp.rate_package_id,
                    rp.parent_rate_package_id,
                    rp.model_id AS package_model_id,
                    rp.model_name AS package_model_name,
                    rp.package_version,
                    rp.package_status,
                    rp.model_version,
                    rp.manifest_id,
                    rp.split_set_id,
                    rp.rating_workbook_path,
                    rp.publication_receipt_sha256,
                    rp.source_export_id,
                    rp.build_fingerprint_sha256,
                    rp.base_rate,
                    rp.effective_from_date,
                    rp.effective_to_date,
                    rp.publication_receipt_json,
                    rp.package_metadata_json,
                    rp.offset_handling,
                    rp.offset_factor_name,
                    rp.offset_source_name,
                    rp.offset_label,
                    rp.metadata_origin,
                    mr.model_run_id,
                    mr.parent_model_run_id,
                    mr.model_id AS run_model_id,
                    mr.model_name AS run_model_name,
                    mr.model_version AS run_model_version,
                    mr.export_id AS run_export_id,
                    mr.airflow_run_id AS run_airflow_run_id,
                    mr.manifest_id AS run_manifest_id,
                    mr.split_set_id AS run_split_set_id,
                    mr.rate_package_id AS run_rate_package_id,
                    mr.run_status,
                    mr.rating_workbook_path AS run_rating_workbook_path,
                    mr.rating_workbook_sha256,
                    mr.mlflow_run_id,
                    mr.publication_receipt_path,
                    mr.publication_receipt_sha256 AS run_publication_receipt_sha256,
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
                    mr.effective_from
                FROM pricing.PRICING_RATE_PACKAGE AS rp
                LEFT JOIN pricing.MODEL_RUN AS mr
                  ON mr.rate_package_id = rp.rate_package_id
                WHERE rp.model_id = :model_id
                  AND rp.parent_rate_package_id IS NULL
                  AND rp.build_fingerprint_sha256 = :build_fingerprint_sha256
                """
            ),
            {
                "model_id": model_id,
                "build_fingerprint_sha256": build_fingerprint_sha256,
            },
        )
        .mappings()
        .one_or_none()
    )


def _validate_validation_split_geometry(
    connection,
    *,
    build: ApprovedModelBuild,
    split_set_id: str | None,
) -> None:
    if not build.validation_splits:
        return
    if split_set_id is None:
        raise ApprovedModelBuildError(
            "validation split evidence requires split_set_id in local notebook publication"
        )
    stored_geometry = [
        (int(row[0]), int(row[1]), int(row[2]))
        for row in connection.execute(
            text(
                """
                SELECT fold_no, n_train, n_test
                FROM pricing.CV_FOLD
                WHERE split_set_id = :split_set_id
                ORDER BY fold_no
                """
            ),
            {"split_set_id": split_set_id},
        ).all()
    ]
    expected_geometry = [
        (
            split.validation_split_no,
            split.n_train,
            split.n_validation,
        )
        for split in build.validation_splits
    ]
    if stored_geometry != expected_geometry:
        raise ApprovedModelBuildError(
            "validation split geometry does not match CV_FOLD: "
            f"stored={stored_geometry!r}, requested={expected_geometry!r}"
        )


def _verify_local_audit_artifact(
    path_value: str | Path | None,
    *,
    expected_sha256: str | None,
    artifact_root: str | Path,
    label: str,
) -> Path:
    if path_value is None or not str(path_value).strip():
        raise ApprovedModelBuildError(f"{label} path is missing from local audit evidence")
    if expected_sha256 is None or not str(expected_sha256).strip():
        raise ApprovedModelBuildError(f"{label} SHA-256 is missing from local audit evidence")
    root = Path(artifact_root).expanduser().resolve()
    artifact_path = Path(path_value).expanduser().resolve()
    if not artifact_path.is_relative_to(root):
        raise ApprovedModelBuildError(f"{label} is outside the configured artifact root")
    if not artifact_path.is_file():
        raise ApprovedModelBuildError(f"{label} is missing: {artifact_path.as_posix()}")
    actual_sha256 = sha256_file(artifact_path)
    if actual_sha256 != str(expected_sha256):
        raise ApprovedModelBuildError(
            f"{label} SHA-256 verification failed: "
            f"expected={expected_sha256!r}, actual={actual_sha256!r}"
        )
    return artifact_path


def _verify_canonical_publication_artifacts(
    existing,
    *,
    artifact_root: str | Path,
) -> Path:
    package_receipt_sha256 = existing["publication_receipt_sha256"]
    run_receipt_sha256 = existing["run_publication_receipt_sha256"]
    if _identity(package_receipt_sha256) != _identity(run_receipt_sha256):
        raise ApprovedModelBuildError(
            "canonical publication receipt SHA-256 differs between package and model run"
        )
    workbook_path = _verify_local_audit_artifact(
        existing["rating_workbook_path"],
        expected_sha256=existing["rating_workbook_sha256"],
        artifact_root=artifact_root,
        label="canonical rating workbook",
    )
    _verify_local_audit_artifact(
        existing["publication_receipt_path"],
        expected_sha256=package_receipt_sha256,
        artifact_root=artifact_root,
        label="canonical publication receipt",
    )
    return workbook_path


def _verify_canonical_candidate_artifact(
    engine,
    *,
    existing,
    incoming_build: ApprovedModelBuild,
    artifact_root: str | Path,
) -> None:
    manifest_id = str(existing["run_manifest_id"])
    split_set_id = (
        None if existing["run_split_set_id"] is None else str(existing["run_split_set_id"])
    )
    sql_lineage = load_candidate_sql_lineage(
        engine,
        manifest_id=manifest_id,
        split_set_id=split_set_id,
    )
    canonical_build = incoming_build.model_copy(
        update={
            "model_id": int(existing["run_model_id"]),
            "model_name": str(existing["run_model_name"]),
            "model_version": str(existing["run_model_version"]),
            "manifest_id": manifest_id,
            "split_set_id": split_set_id,
            "export_id": str(existing["run_export_id"]),
            "rating_workbook_path": str(existing["run_rating_workbook_path"]),
            "rating_workbook_sha256": str(existing["rating_workbook_sha256"]),
            "publication_receipt_path": str(existing["publication_receipt_path"]),
            "publication_receipt_sha256": str(existing["publication_receipt_sha256"]),
            "candidate_artifact_path": str(existing["candidate_artifact_path"]),
            "candidate_artifact_sha256": str(existing["candidate_artifact_sha256"]),
            "candidate_artifact_format": str(existing["candidate_artifact_format"]),
            "candidate_artifact_size_bytes": int(existing["candidate_artifact_size_bytes"]),
            "candidate_python_version": str(existing["candidate_python_version"]),
            "candidate_superglm_version": str(existing["candidate_superglm_version"]),
            "candidate_superglm_git_sha": str(existing["candidate_superglm_git_sha"]),
            "build_fingerprint_sha256": str(existing["build_fingerprint_sha256"]),
            "builder_source_sha256": str(existing["builder_source_sha256"]),
            "materialized_split_sha256": str(existing["materialized_split_sha256"]),
            "runtime_sha256": str(existing["runtime_sha256"]),
            "candidate_superglm_sha256": str(existing["candidate_superglm_sha256"]),
            "row_order_sha256": (
                incoming_build.row_order_sha256
                if sql_lineage.split_row_order_sha256 is None
                else sql_lineage.split_row_order_sha256
            ),
            "model_source_sha256": str(existing["model_source_sha256"]),
            "model_frame_sha256": sql_lineage.model_frame_sha256,
            "mlflow_run_id": (
                None if existing["mlflow_run_id"] is None else str(existing["mlflow_run_id"])
            ),
            "effective_from": existing["effective_from"],
            "validation_curve_status": existing["validation_curve_status"],
            "validation_curve_reason": existing["validation_curve_reason"],
        }
    )
    try:
        _verify_candidate_artifact(
            canonical_build,
            sql_lineage=sql_lineage,
            allowed_root=artifact_root,
        )
    except ApprovedModelBuildError as exc:
        raise ApprovedModelBuildError(
            f"canonical candidate artifact verification failed: {exc}"
        ) from exc


def _local_material_lineage_contract(
    connection,
    *,
    manifest_id: str,
    split_set_id: str | None,
):
    manifest = (
        connection.execute(
            text(
                """
                SELECT
                    dataset_name, source_system, data_as_of_date, row_count,
                    pk_columns_json, target_column, weight_column,
                    model_frame_sha256, frame_hash_metadata_json,
                    exposure_column, data_as_of_column, offset_column,
                    offset_source_column, offset_label, export_weight_column
                FROM pricing.DATASET_MANIFEST
                WHERE manifest_id = :manifest_id
                """
            ),
            {"manifest_id": manifest_id},
        )
        .mappings()
        .one_or_none()
    )
    if manifest is None:
        raise ApprovedModelBuildError(f"local manifest_id {manifest_id!r} does not exist")
    manifest_contract = tuple(
        _json_evidence(manifest[field_name], field_name=field_name)
        if field_name in {"pk_columns_json", "frame_hash_metadata_json"}
        else manifest[field_name]
        for field_name in manifest.keys()
    )
    columns = tuple(
        tuple(row)
        for row in connection.execute(
            text(
                """
                SELECT ordinal_no, column_name, column_role, pandas_dtype,
                       null_count, distinct_count
                FROM pricing.DATASET_COLUMN
                WHERE manifest_id = :manifest_id
                ORDER BY ordinal_no
                """
            ),
            {"manifest_id": manifest_id},
        ).all()
    )
    if split_set_id is None:
        return (manifest_contract, columns, None, ())

    split_set = (
        connection.execute(
            text(
                """
                SELECT
                    split_mode, splitter_class, splitter_params_json,
                    row_order_sha256, row_count, fold_count,
                    groups_column, stratify_column, runtime_metadata_json
                FROM pricing.CV_SPLIT_SET
                WHERE split_set_id = :split_set_id
                  AND manifest_id = :manifest_id
                """
            ),
            {"split_set_id": split_set_id, "manifest_id": manifest_id},
        )
        .mappings()
        .one_or_none()
    )
    if split_set is None:
        raise ApprovedModelBuildError(
            f"local split_set_id {split_set_id!r} does not match the manifest"
        )
    split_contract = tuple(
        _json_evidence(split_set[field_name], field_name=field_name)
        if field_name in {"splitter_params_json", "runtime_metadata_json"}
        else split_set[field_name]
        for field_name in split_set.keys()
    )
    folds = tuple(
        tuple(row)
        for row in connection.execute(
            text(
                """
                SELECT fold_no, n_train, n_test
                FROM pricing.CV_FOLD
                WHERE split_set_id = :split_set_id
                ORDER BY fold_no
                """
            ),
            {"split_set_id": split_set_id},
        ).all()
    )
    return (manifest_contract, columns, split_contract, folds)


def _json_evidence(value: Any, *, field_name: str):
    if value is None:
        return None
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ApprovedModelBuildError(f"local audit evidence has invalid {field_name}") from exc


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
    staged,
    build: ApprovedModelBuild,
    canonical_workbook_path: Path,
) -> list[str]:
    canonical_workbook_sha256 = str(existing["rating_workbook_sha256"])
    expected = {
        "model_version": build.model_version,
        "publication_receipt_sha256": build.publication_receipt_sha256,
        "build_fingerprint_sha256": build.build_fingerprint_sha256,
        "base_rate": staged["base_rate"],
        "effective_from_date": staged["effective_from_date"],
        "effective_to_date": staged["effective_to_date"],
        "offset_handling": staged["offset_handling"] or "UNKNOWN",
        "offset_factor_name": staged["offset_factor_name"],
        "offset_source_name": staged["offset_source_name"],
        "offset_label": staged["offset_label"],
        "metadata_origin": staged["metadata_origin"],
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
    if canonical_workbook_sha256 != build.rating_workbook_sha256:
        canonical_semantic_sha256 = xlsx_semantic_sha256(canonical_workbook_path)
        requested_semantic_sha256 = xlsx_semantic_sha256(Path(build.rating_workbook_path))
        if canonical_semantic_sha256 != requested_semantic_sha256:
            conflicts.append(
                "rating_workbook_semantic_sha256 "
                f"existing={canonical_semantic_sha256!r} "
                f"requested={requested_semantic_sha256!r}"
            )
    for field_name in ("publication_receipt_json", "package_metadata_json"):
        existing_value = _json_evidence(existing[field_name], field_name=field_name)
        requested_value = _json_evidence(staged[field_name], field_name=field_name)
        if existing_value != requested_value:
            conflicts.append(
                f"{field_name} existing={existing_value!r} requested={requested_value!r}"
            )
    return conflicts


def _model_run_evidence_conflicts(
    connection,
    existing,
    *,
    build: ApprovedModelBuild,
    requested_lineage,
) -> list[str]:
    model_run_id = existing["model_run_id"]
    if model_run_id is None:
        raise RuntimeError(
            "incomplete local publication lineage: canonical package has no model run"
        )
    if str(existing["run_status"] or "").upper() != "SUCCESS":
        raise RuntimeError(
            "incomplete local publication lineage: canonical package has no successful model run"
        )
    _validate_existing_lineage_links(
        connection,
        model_run_id=model_run_id,
        manifest_id=str(existing["manifest_id"]),
        split_set_id=(None if existing["split_set_id"] is None else str(existing["split_set_id"])),
    )

    expected_scalars = {
        "candidate_artifact_format": build.candidate_artifact_format,
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
        "effective_from": build.effective_from,
    }
    conflicts = []
    canonical_lineage = {
        "parent_rate_package_id": None,
        "package_status": "LOCAL_AUDIT",
        "package_model_id": build.model_id,
        "package_model_name": build.model_name,
        "parent_model_run_id": None,
        "run_model_id": existing["package_model_id"],
        "run_model_name": existing["package_model_name"],
        "run_model_version": existing["model_version"],
        "run_export_id": existing["source_export_id"],
        "run_airflow_run_id": existing["source_export_id"],
        "run_manifest_id": existing["manifest_id"],
        "run_split_set_id": existing["split_set_id"],
        "run_rate_package_id": existing["rate_package_id"],
        "run_rating_workbook_path": existing["rating_workbook_path"],
        "run_publication_receipt_sha256": existing["publication_receipt_sha256"],
    }
    for field_name, expected_value in canonical_lineage.items():
        if _identity(existing[field_name]) != _identity(expected_value):
            conflicts.append(
                f"{field_name} existing={existing[field_name]!r} expected={expected_value!r}"
            )
    if _identity(existing["validation_source_model_run_id"]) != _identity(model_run_id):
        conflicts.append(
            "validation_source_model_run_id does not self-link the canonical root model run"
        )
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

    canonical_split_set_id = _identity(existing["split_set_id"])
    stored_split_metrics = Counter(
        (_identity(row[0]), int(row[1]), str(row[2]), float(row[3]))
        for row in connection.execute(
            text(
                """
                SELECT split_set_id, fold_no, metric_name, metric_value
                FROM pricing.CV_FOLD_METRIC
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": model_run_id},
        ).all()
    )
    expected_split_metrics = Counter(
        (
            canonical_split_set_id,
            split.validation_split_no,
            metric_name,
            metric_value,
        )
        for split in build.validation_splits
        for metric_name, metric_value in split.metrics.items()
    )
    if stored_split_metrics != expected_split_metrics:
        conflicts.append(
            f"split metrics existing={stored_split_metrics!r} requested={expected_split_metrics!r}"
        )
    stored_curves = Counter(
        (
            _identity(row[0]),
            int(row[1]),
            str(row[2]),
            int(row[3]),
            str(row[4]),
            None if row[5] is None else float(row[5]),
            _identity(row[6]),
            None if row[7] is None else float(row[7]),
            None if row[8] is None else float(row[8]),
            None if row[9] is None else float(row[9]),
            None if row[10] is None else float(row[10]),
            _identity(row[11]),
        )
        for row in connection.execute(
            text(
                """
                SELECT split_set_id, split_no, term_name, point_no, point_kind,
                       x_numeric, level_text, eta_contribution,
                       relativity, support_value, reference_value,
                       reference_level
                FROM pricing.CV_SPLIT_CURVE_POINT
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": model_run_id},
        ).all()
    )
    expected_curves = Counter(
        (
            canonical_split_set_id,
            point.validation_split_no,
            point.term_name,
            point.point_no,
            point.point_kind,
            point.x_numeric,
            point.level_text,
            point.eta_contribution,
            point.relativity,
            point.support_value,
            point.reference_value,
            point.reference_level,
        )
        for point in build.validation_curve_points
    )
    if stored_curves != expected_curves:
        conflicts.append(
            f"validation_curve_points existing={stored_curves!r} requested={expected_curves!r}"
        )
    stored_lineage = _local_material_lineage_contract(
        connection,
        manifest_id=str(existing["manifest_id"]),
        split_set_id=(None if existing["split_set_id"] is None else str(existing["split_set_id"])),
    )
    if stored_lineage != requested_lineage:
        conflicts.append("material manifest/split contract differs from the canonical root")
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
