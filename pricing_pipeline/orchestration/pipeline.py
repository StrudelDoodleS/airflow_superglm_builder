from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sqlalchemy import text

from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.publishing.lineage import record_model_run
from pricing_pipeline.publishing.lifecycle import CompletedModelPublishResult
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import ApprovedModelBuild
from pricing_pipeline.publishing.model_registry import (
    ModelRegistryError,
    validate_registered_model,
)
from pricing_pipeline.publishing.package_writer import (
    ExpectedModelIdentity,
    publish_rating_package,
)
from pricing_pipeline.publishing.staging import stage_rating_export
from pricing_pipeline.workbench.submission import sha256_file, xlsx_semantic_sha256


class PublishedRunIntegrityError(RuntimeError):
    """Raised when an export ID resolves incomplete or ambiguous durable lineage."""


def _verify_incoming_publication_artifact(
    path_value: str | Path | None,
    *,
    expected_sha256: str | None,
    label: str,
    allowed_artifact_root: str | Path | None,
) -> Path:
    if path_value is None or not str(path_value).strip():
        raise PublishedRunIntegrityError(f"{label} path is missing")
    artifact_path = Path(path_value).expanduser().resolve()
    if allowed_artifact_root is not None:
        root = Path(allowed_artifact_root).expanduser().resolve()
        if not artifact_path.is_relative_to(root):
            raise PublishedRunIntegrityError(f"{label} is outside the configured artifact root")
    if not artifact_path.is_file():
        raise PublishedRunIntegrityError(f"{label} does not exist: {artifact_path.as_posix()}")
    expected = str(expected_sha256 or "").strip()
    if not expected:
        raise PublishedRunIntegrityError(f"{label} SHA-256 evidence is missing")
    actual = sha256_file(artifact_path)
    if actual != expected:
        raise PublishedRunIntegrityError(
            f"{label} SHA-256 does not match the export evidence: "
            f"expected={expected!r}, actual={actual!r}"
        )
    return artifact_path


def publish_model_export(
    engine,
    export: ApprovedModelBuild,
    *,
    model_config: ModelBuildConfig,
    expected_database: str,
    allowed_artifact_root: str | Path,
    validated_model_id: int | None = None,
) -> CompletedModelPublishResult:
    if allowed_artifact_root is None:
        raise PublishedRunIntegrityError("allowed_artifact_root is required for remote publication")
    workbook_path = _verify_incoming_publication_artifact(
        export.rating_workbook_path,
        expected_sha256=export.rating_workbook_sha256,
        label="rating workbook",
        allowed_artifact_root=allowed_artifact_root,
    )
    receipt_path = _verify_incoming_publication_artifact(
        export.publication_receipt_path,
        expected_sha256=export.publication_receipt_sha256,
        label="publication receipt",
        allowed_artifact_root=allowed_artifact_root,
    )
    if validated_model_id is None:
        with engine.begin() as connection:
            model_id = validate_registered_model(connection, model_config).model_id
    else:
        model_id = int(validated_model_id)
    _validate_export_matches_config(export, model_config, model_id=model_id)

    def write_package_lineage(connection, rate_package_id: int) -> int:
        return record_model_run(
            None,
            build=export,
            dag_id="notebook",
            airflow_run_id=export.export_id,
            rate_package_id=rate_package_id,
            connection=connection,
        )

    staging_kwargs = {
        "workbook_path": workbook_path,
        "export_id": export.export_id,
        "expected_database": expected_database,
        "model_name": model_config.model_name,
        "model_version": export.model_version,
        "target_name": model_config.target_name,
        "model_type": model_config.model_type,
        "effective_from": export.effective_from,
        "created_by": export.created_by,
        "replace": True,
        "model_id": model_id,
        "publication_receipt_path": receipt_path,
        "publication_receipt_sha256": export.publication_receipt_sha256,
    }
    content_sha256 = stage_rating_export(engine, **staging_kwargs)
    staged_workbook_sha256 = sha256_file(workbook_path)
    if staged_workbook_sha256 != export.rating_workbook_sha256:
        raise PublishedRunIntegrityError(
            "rating workbook changed during staging: "
            f"expected={export.rating_workbook_sha256!r}, actual={staged_workbook_sha256!r}"
        )
    publish_result = publish_rating_package(
        engine,
        export_id=export.export_id,
        expected_database=expected_database,
        expected_model_identity=ExpectedModelIdentity(
            model_id=model_id,
            model_name=model_config.model_name,
            target_name=model_config.target_name,
            model_type=model_config.model_type,
        ),
        created_by=export.created_by,
        build_fingerprint_sha256=export.build_fingerprint_sha256,
        package_lineage_writer=write_package_lineage,
        expected_staged_metadata={
            "export_id": export.export_id,
            "model_id": model_id,
            "model_name": model_config.model_name,
            "model_version": export.model_version,
            "effective_from_date": export.effective_from,
            "effective_to_date": None,
            "source_file": str(Path(export.rating_workbook_path).resolve()),
            "publication_receipt_sha256": export.publication_receipt_sha256,
            "staging_content_sha256": content_sha256,
        },
    )
    if publish_result.was_existing:
        existing = _resolve_existing_published_run(
            engine,
            export,
            rate_package_id=publish_result.rate_package_id,
            allowed_artifact_root=allowed_artifact_root,
        )
        if existing is None:
            raise PublishedRunIntegrityError(
                f"existing package for export_id {export.export_id!r} "
                "disappeared before lineage validation"
            )
        if existing.rate_package_id != publish_result.rate_package_id:
            raise PublishedRunIntegrityError(
                f"existing package identity changed for export_id {export.export_id!r}"
            )
        return existing
    if publish_result.model_run_id is None:
        raise RuntimeError("package publication did not record scheduled model lineage")

    return CompletedModelPublishResult(
        model_id=model_id,
        model_name=export.model_name,
        model_version=export.model_version,
        manifest_id=export.manifest_id,
        split_set_id=export.split_set_id,
        export_id=publish_result.export_id,
        rate_package_id=publish_result.rate_package_id,
        package_version=publish_result.package_version,
        package_status=publish_result.package_status,
        rating_workbook_path=export.rating_workbook_path,
        model_run_id=publish_result.model_run_id,
        mlflow_run_id=export.mlflow_run_id or None,
        publication_receipt_path=export.publication_receipt_path,
        publication_receipt_sha256=export.publication_receipt_sha256,
        candidate_artifact_path=export.candidate_artifact_path,
        was_existing=publish_result.was_existing,
    )


def _validate_export_matches_config(
    export: ApprovedModelBuild,
    config: ModelBuildConfig,
    *,
    model_id: int,
) -> None:
    mismatches = []
    for field_name, actual, expected in (
        ("model_id", int(export.model_id), int(model_id)),
        ("model_name", export.model_name, config.model_name),
        ("target_name", export.target_name, config.target_name),
        ("model_type", export.model_type, config.model_type),
        ("deployment_slot", export.deployment_slot, config.deployment_slot),
    ):
        if actual != expected:
            mismatches.append(f"{field_name} export={actual!r} config={expected!r}")
    if mismatches:
        raise ModelRegistryError(
            "training export does not match model config: " + "; ".join(mismatches)
        )


def _resolve_existing_published_run(
    engine,
    export: ApprovedModelBuild,
    *,
    rate_package_id: int | None = None,
    allowed_artifact_root: str | Path | None = None,
) -> CompletedModelPublishResult | None:
    incoming_workbook = _verify_incoming_publication_artifact(
        export.rating_workbook_path,
        expected_sha256=export.rating_workbook_sha256,
        label="rating workbook",
        allowed_artifact_root=allowed_artifact_root,
    )
    schemas = schema_names_from_connectable(engine)
    query = text(
        f"""
        SELECT
            rp.model_id,
            pm.model_name,
            rp.model_version,
            rp.source_export_id,
            rp.rate_package_id,
            rp.package_version,
            rp.package_status,
            rp.parent_rate_package_id,
            rp.effective_from_date,
            rp.effective_to_date,
            rp.source_file,
            rp.build_fingerprint_sha256,
            rp.publication_receipt_sha256 AS package_publication_receipt_sha256,
            mr.model_run_id,
            mr.run_status,
            mr.export_id AS run_export_id,
            mr.model_id AS run_model_id,
            mr.model_name AS run_model_name,
            mr.model_version AS run_model_version,
            mr.dag_id,
            mr.airflow_run_id,
            mr.manifest_id,
            mr.rating_workbook_path,
            mr.rating_workbook_sha256,
            mr.mlflow_run_id,
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
            mr.validation_source_model_run_id
        FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp WITH (UPDLOCK, HOLDLOCK)
        JOIN {schemas.pricing}.PRICING_MODEL AS pm
          ON pm.model_id = rp.model_id
        LEFT JOIN {schemas.pricing}.MODEL_RUN AS mr WITH (UPDLOCK, HOLDLOCK)
          ON mr.rate_package_id = rp.rate_package_id
        WHERE rp.model_id = :model_id
          AND (
              (:rate_package_id IS NOT NULL AND rp.rate_package_id = :rate_package_id)
              OR (:rate_package_id IS NULL AND rp.source_export_id = :export_id)
          )
        """
    )
    with engine.begin() as connection:
        rows = list(
            connection.execute(
                query,
                {
                    "model_id": export.model_id,
                    "export_id": export.export_id,
                    "rate_package_id": rate_package_id,
                },
            )
            .mappings()
            .all()
        )
        if not rows:
            return None
        if len(rows) != 1:
            raise PublishedRunIntegrityError(
                f"export_id {export.export_id!r} resolves {len(rows)} package/run rows"
            )
        row = dict(rows[0])
        if row.get("model_run_id") is None:
            raise PublishedRunIntegrityError(
                f"export_id {export.export_id!r} has a package without model-run lineage; "
                "manual repair is required"
            )
        if str(row.get("run_status") or "").upper() != "SUCCESS":
            raise PublishedRunIntegrityError(
                f"export_id {export.export_id!r} has no successful model run"
            )
        if str(row.get("package_status") or "").upper() != "PUBLISHED":
            raise PublishedRunIntegrityError(
                f"export_id {export.export_id!r} has unusable package status"
            )

        model_run_id = int(row["model_run_id"])
        evidence_params = {"model_run_id": model_run_id}
        dataset_rows = [
            dict(item)
            for item in connection.execute(
                text(
                    f"""
                    SELECT manifest_id, dataset_role
                    FROM {schemas.mlops}.MODEL_RUN_DATASET AS dataset_link
                        WITH (UPDLOCK, HOLDLOCK)
                    WHERE dataset_link.model_run_id = :model_run_id
                    """
                ),
                evidence_params,
            )
            .mappings()
            .all()
        ]
        split_rows = [
            dict(item)
            for item in connection.execute(
                text(
                    f"""
                    SELECT manifest_id, split_set_id, dataset_role, split_role
                    FROM {schemas.mlops}.MODEL_RUN_SPLIT_SET AS split_link
                        WITH (UPDLOCK, HOLDLOCK)
                    WHERE split_link.model_run_id = :model_run_id
                    """
                ),
                evidence_params,
            )
            .mappings()
            .all()
        ]
        metric_rows = [
            dict(item)
            for item in connection.execute(
                text(
                    f"""
                    SELECT metric_name, metric_value, metric_scope
                    FROM {schemas.mlops}.MODEL_RUN_METRIC AS metric
                        WITH (UPDLOCK, HOLDLOCK)
                    WHERE metric.model_run_id = :model_run_id
                    """
                ),
                evidence_params,
            )
            .mappings()
            .all()
        ]
        fold_rows = [
            dict(item)
            for item in connection.execute(
                text(
                    f"""
                    SELECT split_set_id, fold_no, metric_name, metric_value
                    FROM {schemas.pricing}.CV_FOLD_METRIC AS fold_metric
                        WITH (UPDLOCK, HOLDLOCK)
                    WHERE fold_metric.model_run_id = :model_run_id
                    """
                ),
                evidence_params,
            )
            .mappings()
            .all()
        ]
        curve_rows = [
            dict(item)
            for item in connection.execute(
                text(
                    f"""
                    SELECT
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
                    FROM {schemas.pricing}.CV_SPLIT_CURVE_POINT AS curve
                        WITH (UPDLOCK, HOLDLOCK)
                    WHERE curve.model_run_id = :model_run_id
                    ORDER BY curve.split_no, curve.term_name, curve.point_no
                    """
                ),
                evidence_params,
            )
            .mappings()
            .all()
        ]

        canonical_manifest_id = str(row["manifest_id"])
        expected_dataset_links = {(canonical_manifest_id, "training")}
        actual_dataset_links = {
            (str(item["manifest_id"]), str(item["dataset_role"])) for item in dataset_rows
        }
        if actual_dataset_links != expected_dataset_links:
            raise PublishedRunIntegrityError(
                "existing export has incompatible evidence: canonical dataset links "
                f"stored={sorted(actual_dataset_links)!r}, "
                f"expected={sorted(expected_dataset_links)!r}"
            )

        canonical_split_links = [
            item
            for item in split_rows
            if str(item["dataset_role"]) == "training"
            and str(item["split_role"]) == "validation"
            and str(item["manifest_id"]) == canonical_manifest_id
        ]
        if len(canonical_split_links) != len(split_rows) or len(canonical_split_links) > 1:
            raise PublishedRunIntegrityError(
                "existing export has incompatible evidence: canonical split links are "
                "ambiguous or have unsupported roles"
            )
        canonical_split_set_id = (
            None if not canonical_split_links else str(canonical_split_links[0]["split_set_id"])
        )
        if (canonical_split_set_id is None) != (export.split_set_id is None):
            raise PublishedRunIntegrityError(
                "existing export has incompatible evidence: validation split presence differs"
            )

        canonical_manifest, canonical_columns = _load_manifest_evidence(
            connection,
            schemas=schemas,
            manifest_id=canonical_manifest_id,
        )
        incoming_manifest, incoming_columns = _load_manifest_evidence(
            connection,
            schemas=schemas,
            manifest_id=export.manifest_id,
        )
        canonical_split, canonical_geometry = _load_split_evidence(
            connection,
            schemas=schemas,
            split_set_id=canonical_split_set_id,
        )
        incoming_split, incoming_geometry = _load_split_evidence(
            connection,
            schemas=schemas,
            split_set_id=export.split_set_id,
        )

    conflicts = _retry_evidence_conflicts(
        row=row,
        export=export,
        metric_rows=metric_rows,
        fold_rows=fold_rows,
        curve_rows=curve_rows,
        canonical_manifest=canonical_manifest,
        incoming_manifest=incoming_manifest,
        canonical_columns=canonical_columns,
        incoming_columns=incoming_columns,
        canonical_split=canonical_split,
        incoming_split=incoming_split,
        canonical_geometry=canonical_geometry,
        incoming_geometry=incoming_geometry,
    )
    if conflicts:
        raise PublishedRunIntegrityError(
            "existing export has incompatible evidence: " + "; ".join(conflicts)
        )

    committed_workbook = Path(str(row["rating_workbook_path"])).expanduser().resolve()
    package_source_value = row.get("source_file")
    if package_source_value is None or not str(package_source_value).strip():
        raise PublishedRunIntegrityError("existing package source_file is missing")
    package_source = Path(str(package_source_value)).expanduser().resolve()
    if package_source != committed_workbook:
        raise PublishedRunIntegrityError(
            "existing package source_file does not match model-run rating_workbook_path"
        )
    if allowed_artifact_root is not None and not committed_workbook.is_relative_to(
        Path(allowed_artifact_root).expanduser().resolve()
    ):
        raise PublishedRunIntegrityError(
            "existing rating workbook is outside the configured artifact root"
        )
    if not committed_workbook.is_file():
        raise PublishedRunIntegrityError("existing rating workbook is missing")
    committed_sha256 = str(row.get("rating_workbook_sha256") or "")
    if sha256_file(committed_workbook) != committed_sha256:
        raise PublishedRunIntegrityError("existing rating workbook SHA-256 verification failed")
    if committed_sha256 != export.rating_workbook_sha256:
        try:
            canonical_semantic_sha256 = xlsx_semantic_sha256(committed_workbook)
            incoming_semantic_sha256 = xlsx_semantic_sha256(incoming_workbook)
        except (OSError, ValueError) as exc:
            raise PublishedRunIntegrityError(
                "rating workbook semantic verification failed"
            ) from exc
        if canonical_semantic_sha256 != incoming_semantic_sha256:
            raise PublishedRunIntegrityError("rating workbook semantic content differs")

    resolved_manifest_id = canonical_manifest_id
    resolved_split_set_id = canonical_split_set_id

    committed_receipt_value = row.get("publication_receipt_path")
    if committed_receipt_value is None:
        raise PublishedRunIntegrityError("existing publication receipt path is missing")
    committed_receipt = Path(str(committed_receipt_value)).expanduser().resolve()
    if allowed_artifact_root is not None and not committed_receipt.is_relative_to(
        Path(allowed_artifact_root).expanduser().resolve()
    ):
        raise PublishedRunIntegrityError(
            "existing publication receipt is outside the configured artifact root"
        )
    if not committed_receipt.is_file():
        raise PublishedRunIntegrityError("existing publication receipt is missing")
    committed_receipt_sha256 = str(row.get("publication_receipt_sha256") or "")
    if sha256_file(committed_receipt) != committed_receipt_sha256:
        raise PublishedRunIntegrityError("existing publication receipt SHA-256 verification failed")

    artifact_fields = (
        "candidate_artifact_path",
        "candidate_artifact_sha256",
        "candidate_artifact_format",
        "candidate_artifact_size_bytes",
        "candidate_python_version",
        "candidate_superglm_version",
        "candidate_superglm_git_sha",
    )
    artifact_values = [row.get(field) for field in artifact_fields]
    if any(value is not None for value in artifact_values):
        if any(value is None for value in artifact_values):
            raise PublishedRunIntegrityError(
                "existing successful run has incomplete candidate artifact metadata"
            )
        if allowed_artifact_root is None:
            raise PublishedRunIntegrityError(
                "existing candidate artifact requires a configured verification root"
            )
        from pricing_pipeline.workbench.artifacts import (
            CandidateArtifactError,
            load_candidate_bundle,
        )

        try:
            bundle = load_candidate_bundle(
                row["candidate_artifact_path"],
                expected_sha256=row["candidate_artifact_sha256"],
                expected_size_bytes=int(row["candidate_artifact_size_bytes"]),
                expected_format=row["candidate_artifact_format"],
                expected_python_version=row["candidate_python_version"],
                expected_superglm_version=row["candidate_superglm_version"],
                expected_superglm_git_sha=row["candidate_superglm_git_sha"],
                allowed_root=allowed_artifact_root,
            )
        except CandidateArtifactError as exc:
            raise PublishedRunIntegrityError(
                f"existing candidate artifact failed verification: {exc}"
            ) from exc
        expected_identity = {
            "model_name": str(row["run_model_name"]),
            "model_version": str(row["run_model_version"]),
            "export_id": str(row["run_export_id"]),
            "manifest_id": resolved_manifest_id,
            "split_set_id": resolved_split_set_id,
            "build_fingerprint_sha256": str(row["build_fingerprint_sha256"]),
            "model_source_sha256": str(row["model_source_sha256"]),
            "builder_source_sha256": str(row["builder_source_sha256"]),
            "materialized_split_sha256": str(row["materialized_split_sha256"]),
            "runtime_sha256": str(row["runtime_sha256"]),
            "candidate_superglm_sha256": str(row["candidate_superglm_sha256"]),
            "model_frame_sha256": str(canonical_manifest["model_frame_sha256"]),
        }
        if canonical_split is not None:
            expected_identity["row_order_sha256"] = str(canonical_split["row_order_sha256"])
        for field_name, expected_value in expected_identity.items():
            if getattr(bundle, field_name) != expected_value:
                raise PublishedRunIntegrityError(
                    f"existing candidate artifact {field_name} does not match model-run lineage"
                )

    return CompletedModelPublishResult(
        model_id=int(row["model_id"]),
        model_name=str(row["model_name"]),
        model_version=str(row["model_version"]),
        export_id=str(row["source_export_id"]),
        rate_package_id=int(row["rate_package_id"]),
        package_version=int(row["package_version"]),
        package_status=str(row["package_status"]),
        model_run_id=int(row["model_run_id"]),
        manifest_id=resolved_manifest_id,
        split_set_id=resolved_split_set_id,
        rating_workbook_path=str(row["rating_workbook_path"]),
        mlflow_run_id=str(row.get("mlflow_run_id") or "") or None,
        publication_receipt_path=(
            None
            if row.get("publication_receipt_path") is None
            else str(row["publication_receipt_path"])
        ),
        publication_receipt_sha256=(
            None
            if row.get("publication_receipt_sha256") is None
            else str(row["publication_receipt_sha256"])
        ),
        candidate_artifact_path=(
            None
            if row.get("candidate_artifact_path") is None
            else str(row["candidate_artifact_path"])
        ),
        was_existing=True,
    )


def _load_manifest_evidence(connection, *, schemas, manifest_id: str) -> tuple[dict, list[dict]]:
    manifest = (
        connection.execute(
            text(
                f"""
                SELECT
                    manifest_id,
                    dataset_name,
                    source_system,
                    data_as_of_date,
                    row_count,
                    pk_columns_json,
                    target_column,
                    weight_column,
                    exposure_column,
                    data_as_of_column,
                    model_frame_sha256,
                    frame_hash_metadata_json,
                    offset_column,
                    offset_source_column,
                    offset_label,
                    export_weight_column
                FROM {schemas.pricing}.DATASET_MANIFEST WITH (UPDLOCK, HOLDLOCK)
                WHERE manifest_id = :manifest_id
                """
            ),
            {"manifest_id": manifest_id},
        )
        .mappings()
        .one_or_none()
    )
    if manifest is None:
        raise PublishedRunIntegrityError(
            f"material manifest {manifest_id!r} is missing during retry validation"
        )
    columns = [
        dict(item)
        for item in connection.execute(
            text(
                f"""
                SELECT
                    ordinal_no,
                    column_name,
                    column_role,
                    pandas_dtype,
                    null_count,
                    distinct_count
                FROM {schemas.pricing}.DATASET_COLUMN WITH (UPDLOCK, HOLDLOCK)
                WHERE manifest_id = :manifest_id
                ORDER BY ordinal_no
                """
            ),
            {"manifest_id": manifest_id},
        )
        .mappings()
        .all()
    ]
    return dict(manifest), columns


def _load_split_evidence(connection, *, schemas, split_set_id: str | None):
    if split_set_id is None:
        return None, []
    split = (
        connection.execute(
            text(
                f"""
                SELECT
                    split_set_id,
                    manifest_id,
                    split_mode,
                    splitter_class,
                    splitter_params_json,
                    row_order_sha256,
                    row_count,
                    fold_count,
                    groups_column,
                    stratify_column,
                    artifact_sha256,
                    runtime_metadata_json
                FROM {schemas.pricing}.CV_SPLIT_SET WITH (UPDLOCK, HOLDLOCK)
                WHERE split_set_id = :split_set_id
                """
            ),
            {"split_set_id": split_set_id},
        )
        .mappings()
        .one_or_none()
    )
    if split is None:
        raise PublishedRunIntegrityError(
            f"validation split {split_set_id!r} is missing during retry validation"
        )
    geometry = [
        dict(item)
        for item in connection.execute(
            text(
                f"""
                SELECT fold.fold_no, fold.n_train, fold.n_test
                FROM {schemas.pricing}.CV_FOLD AS fold WITH (UPDLOCK, HOLDLOCK)
                WHERE fold.split_set_id = :split_set_id
                ORDER BY fold.fold_no
                """
            ),
            {"split_set_id": split_set_id},
        )
        .mappings()
        .all()
    ]
    return dict(split), geometry


def _normalise_material_row(row: dict, *, ignored: set[str]) -> tuple[tuple[str, object], ...]:
    normalised: list[tuple[str, object]] = []
    for field_name in sorted(set(row) - ignored):
        value = row[field_name]
        if field_name.endswith("_json") and value is not None:
            try:
                value = json.dumps(
                    json.loads(str(value)),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise PublishedRunIntegrityError(f"stored {field_name} is not valid JSON") from exc
        else:
            isoformat = getattr(value, "isoformat", None)
            if callable(isoformat):
                value = isoformat()
        normalised.append((field_name, value))
    return tuple(normalised)


def _retry_evidence_conflicts(
    *,
    row: dict,
    export: ApprovedModelBuild,
    metric_rows: list[dict],
    fold_rows: list[dict],
    curve_rows: list[dict],
    canonical_manifest: dict,
    incoming_manifest: dict,
    canonical_columns: list[dict],
    incoming_columns: list[dict],
    canonical_split: dict | None,
    incoming_split: dict | None,
    canonical_geometry: list[dict],
    incoming_geometry: list[dict],
) -> list[str]:
    expected_scalars = {
        "model_id": export.model_id,
        "model_name": export.model_name,
        "model_version": export.model_version,
        "run_model_id": export.model_id,
        "parent_rate_package_id": None,
        "effective_from_date": export.effective_from,
        "effective_to_date": None,
        "build_fingerprint_sha256": export.build_fingerprint_sha256,
        "package_publication_receipt_sha256": export.publication_receipt_sha256,
        "run_model_name": export.model_name,
        "run_model_version": export.model_version,
        "publication_receipt_sha256": export.publication_receipt_sha256,
        "candidate_artifact_format": export.candidate_artifact_format,
        "candidate_python_version": export.candidate_python_version,
        "candidate_superglm_version": export.candidate_superglm_version,
        "candidate_superglm_git_sha": export.candidate_superglm_git_sha,
        "model_source_sha256": export.model_source_sha256,
        "builder_source_sha256": export.builder_source_sha256,
        "materialized_split_sha256": export.materialized_split_sha256,
        "runtime_sha256": export.runtime_sha256,
        "candidate_superglm_sha256": export.candidate_superglm_sha256,
        "validation_curve_status": export.validation_curve_status,
        "validation_curve_reason": export.validation_curve_reason,
    }
    conflicts: list[str] = []
    for field_name, expected_value in expected_scalars.items():
        actual_value = row.get(field_name)
        expected_identity = None if expected_value is None else str(expected_value)
        actual_identity = None if actual_value is None else str(actual_value)
        if actual_identity != expected_identity:
            conflicts.append(
                f"{field_name} expected={expected_identity!r} stored={actual_identity!r}"
            )
    if int(row.get("validation_source_model_run_id") or -1) != int(row["model_run_id"]):
        conflicts.append("validation source must self-reference the canonical root run")
    if str(row.get("source_export_id")) != str(row.get("run_export_id")):
        conflicts.append("package source_export_id differs from model-run export_id")

    canonical_manifest_contract = _normalise_material_row(
        canonical_manifest,
        ignored={"manifest_id"},
    )
    incoming_manifest_contract = _normalise_material_row(
        incoming_manifest,
        ignored={"manifest_id"},
    )
    if canonical_manifest_contract != incoming_manifest_contract:
        conflicts.append("material manifest contract differs")
    for label, manifest in (
        ("canonical", canonical_manifest),
        ("incoming", incoming_manifest),
    ):
        if str(manifest.get("model_frame_sha256") or "") != export.model_frame_sha256:
            conflicts.append(f"{label} manifest model_frame_sha256 differs")

    canonical_column_contract = tuple(
        _normalise_material_row(item, ignored=set()) for item in canonical_columns
    )
    incoming_column_contract = tuple(
        _normalise_material_row(item, ignored=set()) for item in incoming_columns
    )
    if canonical_column_contract != incoming_column_contract:
        conflicts.append("material manifest columns differ")

    if (canonical_split is None) != (incoming_split is None):
        conflicts.append("validation split contract presence differs")
    elif canonical_split is not None and incoming_split is not None:
        if str(canonical_split.get("manifest_id")) != str(canonical_manifest["manifest_id"]):
            conflicts.append("canonical split belongs to a different manifest")
        if str(incoming_split.get("manifest_id")) != export.manifest_id:
            conflicts.append("incoming split belongs to a different manifest")
        canonical_split_contract = _normalise_material_row(
            canonical_split,
            ignored={"split_set_id", "manifest_id"},
        )
        incoming_split_contract = _normalise_material_row(
            incoming_split,
            ignored={"split_set_id", "manifest_id"},
        )
        if canonical_split_contract != incoming_split_contract:
            conflicts.append("validation split contract differs")
        for label, split in (("canonical", canonical_split), ("incoming", incoming_split)):
            if str(split.get("row_order_sha256") or "") != export.row_order_sha256:
                conflicts.append(f"{label} split row_order_sha256 differs")

    canonical_geometry_contract = {
        (int(item["fold_no"]), int(item["n_train"]), int(item["n_test"]))
        for item in canonical_geometry
    }
    incoming_geometry_contract = {
        (int(item["fold_no"]), int(item["n_train"]), int(item["n_test"]))
        for item in incoming_geometry
    }
    if canonical_geometry_contract != incoming_geometry_contract:
        conflicts.append("validation split geometry differs")
    if export.validation_splits:
        expected_geometry = {
            (split.validation_split_no, split.n_train, split.n_validation)
            for split in export.validation_splits
        }
        if incoming_geometry_contract != expected_geometry:
            conflicts.append("incoming split geometry differs from completed build")

    expected_metrics = {
        str(name): (
            float(value),
            str(export.metric_scopes.get(name, "model_run")),
        )
        for name, value in export.metrics.items()
    }
    actual_metrics = {
        str(item["metric_name"]): (
            float(item["metric_value"]),
            None if item.get("metric_scope") is None else str(item["metric_scope"]),
        )
        for item in metric_rows
    }
    if actual_metrics != expected_metrics:
        conflicts.append(f"metrics expected={expected_metrics!r} stored={actual_metrics!r}")

    canonical_split_set_id = (
        None if canonical_split is None else str(canonical_split["split_set_id"])
    )
    if any(str(item.get("split_set_id")) != canonical_split_set_id for item in fold_rows):
        conflicts.append("split metrics reference a non-canonical split_set_id")
    expected_folds = Counter(
        (split.validation_split_no, metric_name, metric_value)
        for split in export.validation_splits
        for metric_name, metric_value in split.metrics.items()
    )
    actual_folds = Counter(
        (int(item["fold_no"]), str(item["metric_name"]), float(item["metric_value"]))
        for item in fold_rows
    )
    if actual_folds != expected_folds:
        conflicts.append(
            f"split metrics expected={dict(expected_folds)!r} stored={dict(actual_folds)!r}"
        )

    if any(str(item.get("split_set_id")) != canonical_split_set_id for item in curve_rows):
        conflicts.append("validation curve points reference a non-canonical split_set_id")
    expected_curves = Counter(
        (
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
        for point in export.validation_curve_points
    )
    actual_curves = Counter(
        (
            int(item["split_no"]),
            str(item["term_name"]),
            int(item["point_no"]),
            str(item["point_kind"]),
            None if item["x_numeric"] is None else float(item["x_numeric"]),
            None if item["level_text"] is None else str(item["level_text"]),
            float(item["eta_contribution"]),
            None if item["relativity"] is None else float(item["relativity"]),
            None if item["support_value"] is None else float(item["support_value"]),
            None if item["reference_value"] is None else float(item["reference_value"]),
            None if item["reference_level"] is None else str(item["reference_level"]),
        )
        for item in curve_rows
    )
    if actual_curves != expected_curves:
        conflicts.append("validation curve points differ")
    return conflicts
