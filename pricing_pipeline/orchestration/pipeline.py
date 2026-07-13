from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.publishing.lineage import record_model_run_on_connection
from pricing_pipeline.infra.mlflow_tracking import configure_mlflow
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import (
    ModelExportResult,
    ModelSpec,
    coerce_training_frame,
)
from pricing_pipeline.publishing.rating_export import (
    build_export_id,
    build_rating_export_path,
    export_rating_tables,
)
from pricing_pipeline.publishing.publisher import (
    ModelPublisher,
    _validate_export_matches_config,
    validate_model_on_engine,
)
from pricing_pipeline.models.superglm_diagnostics import fit_reml_with_diagnostics


class PublishedRunIntegrityError(RuntimeError):
    """Raised when an export ID resolves incomplete or ambiguous durable lineage."""


@dataclass(frozen=True)
class ExistingPublishedRun:
    model_id: int
    model_name: str
    model_version: str
    export_id: str
    rate_package_id: int
    package_version: int
    package_status: str
    model_run_id: int
    manifest_id: str
    split_set_id: str | None
    rating_workbook_path: str
    mlflow_run_id: str
    publication_receipt_path: str | None
    publication_receipt_sha256: str | None

    def to_publish_result(self) -> dict[str, str | bool | None]:
        result: dict[str, str | bool | None] = {
            "mlflow_run_id": self.mlflow_run_id,
            "export_id": self.export_id,
            "rate_package_id": str(self.rate_package_id),
            "package_version": str(self.package_version),
            "package_status": self.package_status,
            "rating_workbook_path": self.rating_workbook_path,
            "model_run_id": str(self.model_run_id),
            "manifest_id": self.manifest_id,
            "split_set_id": self.split_set_id,
            "was_existing": True,
        }
        if self.publication_receipt_path is not None:
            result["publication_receipt_path"] = self.publication_receipt_path
        if self.publication_receipt_sha256 is not None:
            result["publication_receipt_sha256"] = self.publication_receipt_sha256
        return result


def _text_evidence(value) -> str | None:
    return None if value is None else str(value)


def _date_evidence(value) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return str(isoformat()) if callable(isoformat) else str(value)


def _path_evidence(value) -> str | None:
    if value is None:
        return None
    return str(Path(str(value)).expanduser().resolve())


def _evidence_rows(connection, query, *, model_run_id: int) -> list[dict]:
    return [
        dict(row)
        for row in connection.execute(
            text(query),
            {"model_run_id": model_run_id},
        )
        .mappings()
        .all()
    ]


def _retry_evidence_conflicts(
    *,
    row: dict,
    export: ModelExportResult,
    dataset_rows: list[dict],
    split_rows: list[dict],
    metric_rows: list[dict],
    fold_rows: list[dict],
) -> list[str]:
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
        "dag_id": export.dag_id,
        "airflow_run_id": export.airflow_run_id,
        "mlflow_run_id": export.mlflow_run_id,
        "manifest_id": export.manifest_id,
        "rating_workbook_path": export.rating_workbook_path,
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
        if field_name in path_fields:
            expected_identity = _path_evidence(expected_value)
            actual_identity = _path_evidence(actual_value)
        elif field_name in date_fields:
            expected_identity = _date_evidence(expected_value)
            actual_identity = _date_evidence(actual_value)
        elif field_name in integer_fields:
            expected_identity = None if expected_value is None else int(expected_value)
            actual_identity = None if actual_value is None else int(actual_value)
        else:
            expected_identity = _text_evidence(expected_value)
            actual_identity = _text_evidence(actual_value)
        if actual_identity != expected_identity:
            conflicts.append(
                f"{field_name} expected={expected_identity!r} stored={actual_identity!r}"
            )

    expected_datasets = {(export.manifest_id, "training")}
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
        if export.split_set_id is None
        else {(export.manifest_id, export.split_set_id, "training", "validation")}
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
        str(name): (float(value), _text_evidence(export.metric_scopes.get(name)))
        for name, value in export.metrics.items()
    }
    actual_metrics = {
        str(item["metric_name"]): (
            float(item["metric_value"]),
            _text_evidence(item.get("metric_scope")),
        )
        for item in metric_rows
    }
    if actual_metrics != expected_metrics:
        conflicts.append(f"metrics expected={expected_metrics!r} stored={actual_metrics!r}")

    expected_folds = {
        (
            _text_evidence(export.split_set_id),
            int(item["fold_no"]),
            str(item["metric_name"]),
            float(item["metric_value"]),
        )
        for item in export.fold_metrics
    }
    actual_folds = {
        (
            _text_evidence(item["split_set_id"]),
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


def _resolve_existing_published_run(
    engine,
    export: ModelExportResult,
    *,
    allowed_artifact_root: str | Path | None = None,
) -> ExistingPublishedRun | None:
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
            mr.dag_id,
            mr.airflow_run_id,
            mr.manifest_id,
            mr.rating_workbook_path,
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
        dataset_rows = _evidence_rows(
            connection,
            f"""
            SELECT manifest_id, dataset_role
            FROM {schemas.mlops}.MODEL_RUN_DATASET AS dataset_link WITH (UPDLOCK, HOLDLOCK)
            WHERE dataset_link.model_run_id = :model_run_id
            """,
            model_run_id=model_run_id,
        )
        split_rows = _evidence_rows(
            connection,
            f"""
            SELECT manifest_id, split_set_id, dataset_role, split_role
            FROM {schemas.mlops}.MODEL_RUN_SPLIT_SET AS split_link WITH (UPDLOCK, HOLDLOCK)
            WHERE split_link.model_run_id = :model_run_id
            """,
            model_run_id=model_run_id,
        )
        metric_rows = _evidence_rows(
            connection,
            f"""
            SELECT metric_name, metric_value, metric_scope
            FROM {schemas.mlops}.MODEL_RUN_METRIC AS metric WITH (UPDLOCK, HOLDLOCK)
            WHERE metric.model_run_id = :model_run_id
            """,
            model_run_id=model_run_id,
        )
        fold_rows = _evidence_rows(
            connection,
            f"""
            SELECT split_set_id, fold_no, metric_name, metric_value
            FROM {schemas.pricing}.CV_FOLD_METRIC AS fold_metric WITH (UPDLOCK, HOLDLOCK)
            WHERE fold_metric.model_run_id = :model_run_id
            """,
            model_run_id=model_run_id,
        )

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
        if bundle.manifest_id != str(row["manifest_id"]):
            raise PublishedRunIntegrityError(
                "existing candidate artifact manifest does not match model-run lineage"
            )
        row_split_set_id = export.split_set_id
        if bundle.split_set_id != row_split_set_id:
            raise PublishedRunIntegrityError(
                "existing candidate artifact split set does not match model-run lineage"
            )

    return ExistingPublishedRun(
        model_id=int(row["model_id"]),
        model_name=str(row["model_name"]),
        model_version=str(row["model_version"]),
        export_id=str(row["source_export_id"]),
        rate_package_id=int(row["rate_package_id"]),
        package_version=int(row["package_version"]),
        package_status=str(row["package_status"]),
        model_run_id=int(row["model_run_id"]),
        manifest_id=str(row["manifest_id"]),
        split_set_id=export.split_set_id,
        rating_workbook_path=str(row["rating_workbook_path"]),
        mlflow_run_id=str(row.get("mlflow_run_id") or ""),
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
    )


def train_and_export_model(
    engine,
    *,
    settings: Settings,
    manifest_id: str,
    split_set_id: str | None = None,
    dag_id: str,
    airflow_run_id: str,
    logical_date: str,
    spec: ModelSpec,
    model_config: ModelBuildConfig,
    created_by: str = "airflow",
) -> ModelExportResult:
    mlflow_client = configure_mlflow(
        settings.mlflow_tracking_uri,
        enabled=settings.mlflow_enabled,
    )
    model_id = validate_model_on_engine(engine, model_config)
    model_version = logical_date.replace("-", "")
    export_id = build_export_id(spec.model_name, airflow_run_id)
    workbook_path = build_rating_export_path(
        settings.rating_export_root,
        model_name=spec.model_name,
        logical_date=logical_date,
        export_id=export_id,
    )

    raw = pd.read_sql_query(spec.training_sql, engine)
    training_frame = coerce_training_frame(spec.build_training_frame(raw))

    mlflow_client.set_experiment(spec.experiment_name)
    with mlflow_client.start_run() as run:
        model = spec.build_model()
        workbook_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow_client.log_param("model_name", spec.model_name)
        mlflow_client.log_param("model_id", model_id)
        mlflow_client.log_param("model_version", model_version)
        mlflow_client.log_param("manifest_id", manifest_id)
        mlflow_client.log_param("target", spec.target_name)
        mlflow_client.log_param("offset", spec.offset_label)
        mlflow_client.log_param("row_count", len(training_frame.X))
        mlflow_client.log_param("feature_columns", ",".join(spec.feature_columns))

        fitted_model = fit_reml_with_diagnostics(
            model,
            training_frame.X,
            training_frame.y,
            offset=training_frame.offset,
            diagnostics_path=workbook_path.parent / "superglm_fit.log",
            mlflow_client=mlflow_client,
        )

        deviance = getattr(getattr(fitted_model, "result", None), "deviance", None)
        if deviance is not None:
            mlflow_client.log_metric("deviance", float(deviance))

        model_path = workbook_path.parent / "superglm_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as handle:
            pickle.dump(fitted_model, handle)
        mlflow_client.log_artifact(str(model_path), artifact_path="model")

        export_rating_tables(
            fitted_model,
            training_frame.X,
            training_frame.y,
            training_frame.exposure,
            output_path=workbook_path,
            mlflow_client=mlflow_client,
        )

        return ModelExportResult(
            model_id=model_id,
            model_name=spec.model_name,
            model_version=model_version,
            model_type=spec.model_type,
            target_name=spec.target_name,
            deployment_slot=spec.deployment_slot,
            manifest_id=manifest_id,
            dag_id=dag_id,
            airflow_run_id=airflow_run_id,
            mlflow_run_id=str(getattr(run.info, "run_id", "")),
            split_set_id=split_set_id,
            export_id=export_id,
            rating_workbook_path=str(workbook_path),
            effective_from=logical_date,
            created_by=created_by,
            package_status=spec.package_status,
        )


def publish_model_export(
    engine,
    export: ModelExportResult | dict,
    *,
    model_config: ModelBuildConfig,
    allowed_artifact_root: str | Path | None = None,
) -> dict[str, str | bool | None]:
    export_result = ModelExportResult.from_mapping(export)
    publisher = ModelPublisher(engine, model_config)
    model_id = publisher.validate_registered_model()
    _validate_export_matches_config(export_result, model_config, model_id=model_id)
    existing = _resolve_existing_published_run(
        engine,
        export_result,
        allowed_artifact_root=allowed_artifact_root,
    )
    if existing is not None:
        return existing.to_publish_result()

    lineage_kwargs = {
        "dag_id": export_result.dag_id,
        "airflow_run_id": export_result.airflow_run_id,
        "mlflow_run_id": export_result.mlflow_run_id,
        "manifest_id": export_result.manifest_id,
        "split_set_id": export_result.split_set_id,
        "export_id": export_result.export_id,
        "model_id": export_result.model_id,
        "model_name": export_result.model_name,
        "model_version": export_result.model_version,
        "rating_workbook_path": export_result.rating_workbook_path,
        "run_status": "SUCCESS",
        "created_by": export_result.created_by,
        "publication_receipt_path": export_result.publication_receipt_path,
        "publication_receipt_sha256": export_result.publication_receipt_sha256,
        "metrics": export_result.metrics,
        "metric_scopes": export_result.metric_scopes,
        "fold_metrics": export_result.fold_metrics,
    }
    if export_result.candidate_artifact_path is not None:
        lineage_kwargs.update(
            {
                "candidate_artifact_path": export_result.candidate_artifact_path,
                "candidate_artifact_sha256": export_result.candidate_artifact_sha256,
                "candidate_artifact_format": export_result.candidate_artifact_format,
                "candidate_artifact_size_bytes": export_result.candidate_artifact_size_bytes,
                "candidate_python_version": export_result.candidate_python_version,
                "candidate_superglm_version": export_result.candidate_superglm_version,
                "model_source_sha256": export_result.model_source_sha256,
            }
        )
    model_run_id: int | None = None

    def write_package_lineage(connection, rate_package_id: int) -> None:
        nonlocal model_run_id
        model_run_id = record_model_run_on_connection(
            connection,
            **lineage_kwargs,
            rate_package_id=rate_package_id,
        )

    publish_result = publisher.publish_training_export(
        export_result,
        package_lineage_writer=write_package_lineage,
    )
    if model_run_id is None:
        raise RuntimeError("package publication did not record scheduled model lineage")

    result: dict[str, str | bool | None] = {
        "mlflow_run_id": str(publish_result.mlflow_run_id),
        "export_id": str(publish_result.export_id),
        "rate_package_id": str(publish_result.rate_package_id),
        "package_version": str(publish_result.package_version),
        "package_status": str(publish_result.package_status),
        "rating_workbook_path": str(publish_result.rating_workbook_path),
        "model_run_id": str(model_run_id),
        "manifest_id": export_result.manifest_id,
        "split_set_id": export_result.split_set_id,
        "was_existing": bool(getattr(publish_result, "was_existing", False)),
    }
    if export_result.publication_receipt_path is not None:
        result["publication_receipt_path"] = export_result.publication_receipt_path
    if export_result.publication_receipt_sha256 is not None:
        result["publication_receipt_sha256"] = export_result.publication_receipt_sha256
    return result


def run_training_export_publish(
    engine,
    *,
    settings: Settings,
    manifest_id: str,
    split_set_id: str | None = None,
    dag_id: str,
    airflow_run_id: str,
    logical_date: str,
    spec: ModelSpec,
    model_config: ModelBuildConfig,
    created_by: str = "airflow",
) -> dict[str, str | bool | None]:
    export = train_and_export_model(
        engine,
        settings=settings,
        manifest_id=manifest_id,
        split_set_id=split_set_id,
        dag_id=dag_id,
        airflow_run_id=airflow_run_id,
        logical_date=logical_date,
        spec=spec,
        model_config=model_config,
        created_by=created_by,
    )
    return publish_model_export(engine, export, model_config=model_config)
