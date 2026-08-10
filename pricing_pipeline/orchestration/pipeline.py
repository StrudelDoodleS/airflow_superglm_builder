from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import ApprovedModelBuild
from pricing_pipeline.publishing.equivalence import (
    ensure_model_equivalence,
    find_equivalent_publication,
    release_unused_model_version_reservation,
)
from pricing_pipeline.publishing.lifecycle import CompletedModelPublishResult
from pricing_pipeline.publishing.lineage import record_model_run
from pricing_pipeline.publishing.model_registry import (
    ModelRegistryError,
    validate_registered_model,
)
from pricing_pipeline.publishing.package_writer import publish_rating_package
from pricing_pipeline.publishing.staging import stage_rating_export
from pricing_pipeline.workbench.submission import sha256_file


class PublishedRunIntegrityError(RuntimeError):
    """Raised when an export ID resolves incomplete or ambiguous durable lineage."""


def publish_model_export(
    engine,
    export: ApprovedModelBuild,
    *,
    model_config: ModelBuildConfig,
    validated_model_id: int | None = None,
    allowed_artifact_root: str | Path | None = None,
) -> CompletedModelPublishResult:
    workbook_path = Path(export.rating_workbook_path)
    if not workbook_path.is_file():
        raise PublishedRunIntegrityError(
            f"rating workbook does not exist: {workbook_path.as_posix()}"
        )
    actual_workbook_sha256 = sha256_file(workbook_path)
    if actual_workbook_sha256 != export.rating_workbook_sha256:
        raise PublishedRunIntegrityError(
            "rating workbook SHA-256 does not match the export evidence: "
            f"expected={export.rating_workbook_sha256!r}, actual={actual_workbook_sha256!r}"
        )
    if validated_model_id is None:
        with engine.begin() as connection:
            model_id = validate_registered_model(connection, model_config).model_id
    else:
        model_id = int(validated_model_id)
    _validate_export_matches_config(export, model_config, model_id=model_id)
    export = ensure_model_equivalence(export)
    equivalent = find_equivalent_publication(engine, build=export)
    if equivalent is not None:
        if equivalent.package_status.upper() != "PUBLISHED":
            raise PublishedRunIntegrityError("equivalent model package is not PUBLISHED")
        release_unused_model_version_reservation(
            engine,
            model_id=export.model_id,
            export_id=export.export_id,
        )
        return CompletedModelPublishResult(
            model_id=equivalent.model_id,
            model_name=equivalent.model_name,
            model_version=equivalent.model_version,
            manifest_id=equivalent.manifest_id,
            split_set_id=equivalent.split_set_id,
            export_id=equivalent.export_id,
            rate_package_id=equivalent.rate_package_id,
            package_version=equivalent.package_version,
            package_status=equivalent.package_status,
            rating_workbook_path=equivalent.rating_workbook_path,
            model_run_id=equivalent.model_run_id,
            mlflow_run_id=equivalent.mlflow_run_id,
            publication_receipt_path=equivalent.publication_receipt_path,
            publication_receipt_sha256=equivalent.publication_receipt_sha256,
            was_existing=True,
            deduplicated=True,
            model_kind=equivalent.model_kind,
            model_equivalence_sha256=equivalent.model_equivalence_sha256,
        )

    staging_kwargs = {
        "workbook_path": Path(export.rating_workbook_path),
        "export_id": export.export_id,
        "model_name": model_config.model_name,
        "model_version": export.model_version,
        "target_name": model_config.target_name,
        "model_type": model_config.model_type,
        "effective_from": export.effective_from,
        "created_by": export.created_by,
        "replace": True,
        "model_id": model_id,
        "publication_receipt_path": export.publication_receipt_path,
        "publication_receipt_sha256": export.publication_receipt_sha256,
    }
    content_sha256 = stage_rating_export(engine, **staging_kwargs)
    staged_workbook_sha256 = sha256_file(workbook_path)
    if staged_workbook_sha256 != export.rating_workbook_sha256:
        raise PublishedRunIntegrityError(
            "rating workbook changed during staging: "
            f"expected={export.rating_workbook_sha256!r}, actual={staged_workbook_sha256!r}"
        )
    equivalence_sha256 = export.model_equivalence_sha256
    if equivalence_sha256 is None:
        raise PublishedRunIntegrityError(
            "Python equivalence fingerprint is missing before SQL staging"
        )

    def write_package_lineage(connection, rate_package_id: int) -> int:
        return record_model_run(
            None,
            build=export,
            dag_id="notebook",
            airflow_run_id=export.export_id,
            rate_package_id=rate_package_id,
            connection=connection,
        )

    publish_result = publish_rating_package(
        engine,
        export_id=export.export_id,
        created_by=export.created_by,
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
            "model_equivalence_sha256": equivalence_sha256,
        },
        equivalence_key={
            "manifest_id": export.manifest_id,
            "model_kind": export.model_kind,
            "model_equivalence_sha256": equivalence_sha256,
        },
    )
    if publish_result.deduplicated:
        return _resolve_equivalent_published_run(
            engine,
            export=export,
            rate_package_id=publish_result.rate_package_id,
        )
    if publish_result.was_existing:
        existing = _resolve_existing_published_run(
            engine,
            export,
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
        was_existing=publish_result.was_existing,
        deduplicated=publish_result.deduplicated,
        model_kind=export.model_kind,
        model_equivalence_sha256=export.model_equivalence_sha256,
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


def _resolve_equivalent_published_run(
    engine,
    *,
    export: ApprovedModelBuild,
    rate_package_id: int,
) -> CompletedModelPublishResult:
    schemas = schema_names_from_connectable(engine)
    with engine.connect() as connection:
        rows = (
            connection.execute(
                text(
                    f"""
                    SELECT
                        rp.model_id,
                        pm.model_name,
                        rp.model_version,
                        rp.source_export_id,
                        rp.rate_package_id,
                        rp.package_version,
                        rp.package_status,
                        mr.model_run_id,
                        mr.run_status,
                        mr.manifest_id,
                        split_link.split_set_id,
                        mr.model_kind,
                        mr.model_equivalence_sha256,
                        mr.rating_workbook_path,
                        mr.mlflow_run_id,
                        mr.publication_receipt_path,
                        mr.publication_receipt_sha256,
                        dataset_link.manifest_id AS linked_manifest_id
                    FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp
                    JOIN {schemas.pricing}.PRICING_MODEL AS pm
                      ON pm.model_id = rp.model_id
                    JOIN {schemas.pricing}.MODEL_RUN AS mr
                      ON mr.rate_package_id = rp.rate_package_id
                    LEFT JOIN {schemas.mlops}.MODEL_RUN_DATASET AS dataset_link
                      ON dataset_link.model_run_id = mr.model_run_id
                     AND dataset_link.dataset_role = 'training'
                     AND dataset_link.manifest_id = mr.manifest_id
                    LEFT JOIN {schemas.mlops}.MODEL_RUN_SPLIT_SET AS split_link
                      ON split_link.model_run_id = mr.model_run_id
                     AND split_link.manifest_id = mr.manifest_id
                     AND split_link.dataset_role = 'training'
                     AND split_link.split_role = 'validation'
                    WHERE rp.rate_package_id = :rate_package_id
                    """
                ),
                {"rate_package_id": rate_package_id},
            )
            .mappings()
            .all()
        )
    if len(rows) != 1:
        raise PublishedRunIntegrityError(
            "equivalent package must resolve exactly one successful model run"
        )
    row = rows[0]
    expected = {
        "model_id": export.model_id,
        "model_name": export.model_name,
        "manifest_id": export.manifest_id,
        "linked_manifest_id": export.manifest_id,
        "model_kind": export.model_kind,
        "model_equivalence_sha256": export.model_equivalence_sha256,
        "run_status": "SUCCESS",
        "package_status": "PUBLISHED",
    }
    mismatches = [
        field_name
        for field_name, expected_value in expected.items()
        if str(row[field_name]) != str(expected_value)
    ]
    if mismatches:
        raise PublishedRunIntegrityError(
            "equivalent package has incompatible durable lineage: " + ", ".join(mismatches)
        )
    return CompletedModelPublishResult(
        model_id=int(row["model_id"]),
        model_name=str(row["model_name"]),
        model_version=str(row["model_version"]),
        manifest_id=str(row["manifest_id"]),
        split_set_id=(None if row["split_set_id"] is None else str(row["split_set_id"])),
        export_id=str(row["source_export_id"]),
        rate_package_id=int(row["rate_package_id"]),
        package_version=int(row["package_version"]),
        package_status=str(row["package_status"]),
        rating_workbook_path=str(row["rating_workbook_path"]),
        model_run_id=int(row["model_run_id"]),
        mlflow_run_id=str(row["mlflow_run_id"] or "") or None,
        publication_receipt_path=(
            None
            if row["publication_receipt_path"] is None
            else str(row["publication_receipt_path"])
        ),
        publication_receipt_sha256=(
            None
            if row["publication_receipt_sha256"] is None
            else str(row["publication_receipt_sha256"])
        ),
        was_existing=True,
        deduplicated=True,
        model_kind=str(row["model_kind"]),
        model_equivalence_sha256=str(row["model_equivalence_sha256"]),
    )


def _resolve_existing_published_run(
    engine,
    export: ApprovedModelBuild,
    *,
    allowed_artifact_root: str | Path | None = None,
) -> CompletedModelPublishResult | None:
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
            rp.publication_receipt_sha256 AS package_publication_receipt_sha256,
            mr.model_run_id,
            mr.run_status,
            mr.export_id AS run_export_id,
            mr.model_id AS run_model_id,
            mr.model_name AS run_model_name,
            mr.model_version AS run_model_version,
            mr.model_kind,
            mr.model_equivalence_sha256,
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
            mr.model_source_sha256
        FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp WITH (UPDLOCK, HOLDLOCK)
        JOIN {schemas.pricing}.PRICING_MODEL AS pm
          ON pm.model_id = rp.model_id
        LEFT JOIN {schemas.pricing}.MODEL_RUN AS mr WITH (UPDLOCK, HOLDLOCK)
          ON mr.rate_package_id = rp.rate_package_id
        WHERE rp.model_id = :model_id
          AND rp.source_export_id = :export_id
        """
    )
    with engine.begin() as connection:
        rows = list(
            connection.execute(
                query,
                {"model_id": export.model_id, "export_id": export.export_id},
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
        if str(row.get("package_status") or "").upper() not in {"DRAFT", "PUBLISHED"}:
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

    conflicts = _retry_evidence_conflicts(
        row=row,
        export=export,
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        metric_rows=metric_rows,
        fold_rows=fold_rows,
    )
    if conflicts:
        raise PublishedRunIntegrityError(
            "existing export has incompatible evidence: " + "; ".join(conflicts)
        )

    committed_workbook = Path(str(row["rating_workbook_path"])).expanduser().resolve()
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

    resolved_manifest_id = export.manifest_id
    resolved_split_set_id = export.split_set_id

    artifact_fields = (
        "candidate_artifact_path",
        "candidate_artifact_sha256",
        "candidate_artifact_format",
        "candidate_artifact_size_bytes",
        "candidate_python_version",
        "candidate_superglm_version",
        "model_source_sha256",
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
        }
        for field_name, expected_value in expected_identity.items():
            if getattr(bundle, field_name) != expected_value:
                raise PublishedRunIntegrityError(
                    f"existing candidate artifact {field_name} does not match model-run lineage"
                )
        if bundle.model_source_sha256 != str(row["model_source_sha256"]):
            raise PublishedRunIntegrityError(
                "existing candidate artifact source hash does not match model-run lineage"
            )
        if bundle.manifest_id != str(row["manifest_id"]):
            raise PublishedRunIntegrityError(
                "existing candidate artifact manifest does not match model-run lineage"
            )
        if bundle.split_set_id != resolved_split_set_id:
            raise PublishedRunIntegrityError(
                "existing candidate artifact split set does not match model-run lineage"
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
        was_existing=True,
        model_kind=str(row.get("model_kind") or "RAW"),
        model_equivalence_sha256=row.get("model_equivalence_sha256"),
    )


def _retry_evidence_conflicts(
    *,
    row: dict,
    export: ApprovedModelBuild,
    dataset_rows: list[dict],
    split_rows: list[dict],
    metric_rows: list[dict],
    fold_rows: list[dict],
) -> list[str]:
    expected_manifest_id = export.manifest_id
    expected_split_set_id = export.split_set_id
    path_fields = {
        "source_file",
        "rating_workbook_path",
        "publication_receipt_path",
        "candidate_artifact_path",
    }
    date_fields = {"effective_from_date", "effective_to_date"}
    integer_fields = {
        "model_id",
        "run_model_id",
        "candidate_artifact_size_bytes",
    }
    expected_scalars = {
        "model_id": export.model_id,
        "model_name": export.model_name,
        "model_version": export.model_version,
        "source_export_id": export.export_id,
        "parent_rate_package_id": None,
        "effective_from_date": export.effective_from,
        "effective_to_date": None,
        "source_file": export.rating_workbook_path,
        "package_publication_receipt_sha256": export.publication_receipt_sha256,
        "run_export_id": export.export_id,
        "run_model_id": export.model_id,
        "run_model_name": export.model_name,
        "run_model_version": export.model_version,
        "model_kind": export.model_kind,
        "model_equivalence_sha256": export.model_equivalence_sha256,
        "dag_id": "notebook",
        "airflow_run_id": export.export_id,
        "mlflow_run_id": export.mlflow_run_id,
        "manifest_id": expected_manifest_id,
        "rating_workbook_path": export.rating_workbook_path,
        "rating_workbook_sha256": export.rating_workbook_sha256,
        "publication_receipt_path": export.publication_receipt_path,
        "publication_receipt_sha256": export.publication_receipt_sha256,
        "candidate_artifact_path": export.candidate_artifact_path,
        "candidate_artifact_sha256": export.candidate_artifact_sha256,
        "candidate_artifact_format": export.candidate_artifact_format,
        "candidate_artifact_size_bytes": export.candidate_artifact_size_bytes,
        "candidate_python_version": export.candidate_python_version,
        "candidate_superglm_version": export.candidate_superglm_version,
        "model_source_sha256": export.model_source_sha256,
    }
    conflicts: list[str] = []
    for field_name, expected_value in expected_scalars.items():
        actual_value = row.get(field_name)
        if field_name == "model_kind" and actual_value is None:
            actual_value = "RAW"
        if field_name in path_fields:
            expected_identity = (
                None
                if expected_value is None
                else str(Path(str(expected_value)).expanduser().resolve())
            )
            actual_identity = (
                None
                if actual_value is None
                else str(Path(str(actual_value)).expanduser().resolve())
            )
        elif field_name in date_fields:
            expected_isoformat = getattr(expected_value, "isoformat", None)
            actual_isoformat = getattr(actual_value, "isoformat", None)
            expected_identity = (
                None
                if expected_value is None
                else str(expected_isoformat() if callable(expected_isoformat) else expected_value)
            )
            actual_identity = (
                None
                if actual_value is None
                else str(actual_isoformat() if callable(actual_isoformat) else actual_value)
            )
        elif field_name in integer_fields:
            expected_identity = None if expected_value is None else int(expected_value)
            actual_identity = None if actual_value is None else int(actual_value)
        else:
            expected_identity = None if expected_value is None else str(expected_value)
            actual_identity = None if actual_value is None else str(actual_value)
        if actual_identity != expected_identity:
            conflicts.append(
                f"{field_name} expected={expected_identity!r} stored={actual_identity!r}"
            )

    expected_datasets = {(expected_manifest_id, "training")}
    actual_datasets = {
        (str(item["manifest_id"]), str(item["dataset_role"])) for item in dataset_rows
    }
    if actual_datasets != expected_datasets:
        conflicts.append(
            f"dataset links expected={sorted(expected_datasets)!r} "
            f"stored={sorted(actual_datasets)!r}"
        )

    expected_splits = (
        set()
        if expected_split_set_id is None
        else {
            (
                expected_manifest_id,
                expected_split_set_id,
                "training",
                "validation",
            )
        }
    )
    actual_splits = {
        (
            str(item["manifest_id"]),
            str(item["split_set_id"]),
            str(item["dataset_role"]),
            str(item["split_role"]),
        )
        for item in split_rows
    }
    if actual_splits != expected_splits:
        conflicts.append(
            f"split links expected={sorted(expected_splits)!r} stored={sorted(actual_splits)!r}"
        )

    expected_metrics = {
        str(name): (
            float(value),
            (None if export.metric_scopes.get(name) is None else str(export.metric_scopes[name])),
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

    expected_folds = {
        (
            None if expected_split_set_id is None else str(expected_split_set_id),
            int(item["fold_no"]),
            str(item["metric_name"]),
            float(item["metric_value"]),
        )
        for item in export.fold_metrics
    }
    actual_folds = {
        (
            None if item["split_set_id"] is None else str(item["split_set_id"]),
            int(item["fold_no"]),
            str(item["metric_name"]),
            float(item["metric_value"]),
        )
        for item in fold_rows
    }
    if actual_folds != expected_folds:
        conflicts.append(
            f"fold metrics expected={sorted(expected_folds)!r} stored={sorted(actual_folds)!r}"
        )
    return conflicts
