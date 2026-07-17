from __future__ import annotations

import hashlib
from inspect import signature
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.config import ModelBuildConfig, ValidationSplitConfig
from pricing_pipeline.models.spec import CompletedModelBuild, CompletedModelBuildError
from pricing_pipeline.orchestration import pipeline
from pricing_pipeline.orchestration.publish_completed_build import (
    CandidateSQLLineage,
    CompletedModelPublishResult,
    publish_completed_model_build,
)
from pricing_pipeline.publishing.lifecycle import PublishResult
from pricing_pipeline.workbench.artifacts import CandidateBundle, save_candidate_bundle


class _ValidationEngine:
    class _Begin:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    def begin(self):
        return self._Begin()


def _config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_name="CLAIM_FREQ",
        model_label="Claim frequency",
        target_name="claim_count",
        model_type="superglm_poisson",
        deployment_slot="CLAIM_FREQ_CURRENT",
        validation_split=ValidationSplitConfig(
            method="none",
            n_splits=None,
            random_state=None,
            shuffle=False,
        ),
    )


def _settings(tmp_path) -> Settings:
    return Settings(
        pricing_database="PricingLab",
        mlflow_tracking_uri="",
        mlflow_enabled=False,
        rating_export_root=tmp_path / "rating_exports",
        validation_split_artifact_root=tmp_path / "validation_splits",
        workbench_artifact_root=tmp_path / "workbench",
    )


def _completed_publish_result(export, **overrides):
    values = {
        "model_id": export.model_id,
        "model_name": export.model_name,
        "model_version": export.model_version,
        "manifest_id": export.manifest_id,
        "split_set_id": export.split_set_id,
        "export_id": export.export_id,
        "rate_package_id": 42,
        "package_version": 7,
        "package_status": "PUBLISHED",
        "rating_workbook_path": export.rating_workbook_path,
        "publication_receipt_path": export.publication_receipt_path,
        "publication_receipt_sha256": export.publication_receipt_sha256,
    }
    values.update(overrides)
    return CompletedModelPublishResult(**values)


def _candidate_metadata(
    artifact_root,
    *,
    model_name="CLAIM_FREQ",
    model_version="20260603",
    export_id="export-1",
    manifest_id="manifest-existing",
    split_set_id=None,
    model_source_sha256="c" * 64,
    pk_columns=("policy_id",),
    row_count=1,
    row_order_sha256="d" * 64,
    model_frame_sha256="f" * 64,
):
    metadata = save_candidate_bundle(
        CandidateBundle(
            fitted_model=SimpleNamespace(name="candidate"),
            X=pd.DataFrame({"age": np.arange(row_count, dtype=float)}),
            y=np.zeros(row_count),
            sample_weight=None,
            offset=None,
            export_weight=None,
            cv_report={},
            model_name=model_name,
            model_version=model_version,
            export_id=export_id,
            manifest_id=manifest_id,
            split_set_id=split_set_id,
            pk_columns=pk_columns,
            row_order_sha256=row_order_sha256,
            model_source_sha256=model_source_sha256,
            model_frame_sha256=model_frame_sha256,
            offset_contract={"handling": "NONE"},
        ),
        Path(artifact_root) / "candidate.joblib",
    )
    return {
        "candidate_artifact_path": metadata.path,
        "candidate_artifact_sha256": metadata.sha256,
        "candidate_artifact_format": metadata.format,
        "candidate_artifact_size_bytes": metadata.size_bytes,
        "candidate_python_version": metadata.python_version,
        "candidate_superglm_version": metadata.superglm_version,
        "candidate_superglm_git_sha": metadata.superglm_git_sha,
        "model_source_sha256": model_source_sha256,
        "model_frame_sha256": model_frame_sha256,
    }


def _approved_build(
    tmp_path,
    *,
    workbook=None,
    candidate_metadata=None,
    **overrides,
):
    if workbook is None:
        workbook = tmp_path / "rating_tables.xlsx"
        workbook.write_bytes(b"rating workbook")
    if candidate_metadata is None:
        candidate_metadata = _candidate_metadata(tmp_path)
    receipt = tmp_path / "publication_receipt.json"
    receipt.write_bytes(b"publication receipt")
    values = {
        "model_id": 17,
        "model_name": "CLAIM_FREQ",
        "model_version": "20260603",
        "model_type": "superglm_poisson",
        "target_name": "claim_count",
        "deployment_slot": "CLAIM_FREQ_CURRENT",
        "manifest_id": "manifest-existing",
        "split_set_id": None,
        "export_id": "export-1",
        "rating_workbook_path": str(workbook),
        "rating_workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
        "effective_from": None,
        "created_by": "airflow",
        "publication_receipt_path": str(receipt),
        "publication_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        **candidate_metadata,
    }
    values.update(overrides)
    return CompletedModelBuild(**values)


class _FakeMappingResult:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def one_or_none(self):
        return self.row

    def first(self):
        return self.row


class _FakeCandidateLineageEngine:
    def __init__(self, *, manifest_row, split_row):
        self.manifest_row = manifest_row
        self.split_row = split_row
        self.queries = []

    def begin(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement, params):
        sql = str(statement)
        self.queries.append((sql, params))
        if "DATASET_MANIFEST" in sql:
            return _FakeMappingResult(self.manifest_row)
        if "CV_SPLIT_SET" in sql:
            return _FakeMappingResult(self.split_row)
        raise AssertionError(f"unexpected SQL: {sql}")


def _patch_candidate_sql_lineage(monkeypatch):
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.load_candidate_sql_lineage",
        lambda engine, *, manifest_id, split_set_id: CandidateSQLLineage(
            manifest_id=manifest_id,
            row_count=1,
            pk_columns=("policy_id",),
            split_set_id=split_set_id,
            split_row_order_sha256=(None if split_set_id is None else "d" * 64),
            model_frame_sha256="f" * 64,
        ),
    )


def test_completed_build_publisher_rejects_untyped_payload(tmp_path, monkeypatch):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_registered_model",
        lambda connection, config: SimpleNamespace(model_id=17),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        lambda engine, export, **kwargs: _completed_publish_result(export),
    )

    with pytest.raises(TypeError, match="ApprovedModelBuild"):
        publish_completed_model_build(
            _ValidationEngine(),
            settings=_settings(tmp_path),
            model_config=_config(),
            completed_build={
                "rating_workbook_path": str(workbook),
                "model_version": "20260603",
                "export_id": "export-1",
                "manifest_id": "manifest-existing",
                "created_by": "analyst",
            },
        )


def test_model_export_publisher_returns_typed_result(monkeypatch, tmp_path):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_bytes(b"rating workbook")
    export = _approved_build(tmp_path, workbook=workbook, created_by="analyst")
    connection = object()
    monkeypatch.setattr(pipeline, "stage_rating_export", lambda *args, **kwargs: "a" * 64)
    monkeypatch.setattr(
        pipeline,
        "record_model_run",
        lambda engine, *, connection, **kwargs: 91,
    )

    def publish_package(engine, *, package_lineage_writer, **kwargs):
        model_run_id = package_lineage_writer(connection, 42)
        return PublishResult(
            mlflow_run_id="",
            export_id=export.export_id,
            rate_package_id=42,
            package_version=7,
            package_status="PUBLISHED",
            rating_workbook_path=export.rating_workbook_path,
            model_run_id=model_run_id,
        )

    monkeypatch.setattr(pipeline, "publish_rating_package", publish_package)

    result = pipeline.publish_model_export(
        object(),
        export,
        model_config=_config(),
        validated_model_id=17,
    )

    assert isinstance(result, CompletedModelPublishResult)
    assert result.model_run_id == 91
    assert result.rate_package_id == 42


def test_model_export_publisher_returns_existing_result_unchanged(
    monkeypatch,
    tmp_path,
):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_bytes(b"rating workbook")
    export = _approved_build(tmp_path, workbook=workbook, created_by="analyst")
    existing = _completed_publish_result(export, was_existing=True)
    monkeypatch.setattr(pipeline, "stage_rating_export", lambda *args, **kwargs: "a" * 64)
    monkeypatch.setattr(
        pipeline,
        "publish_rating_package",
        lambda *args, **kwargs: PublishResult(
            mlflow_run_id="",
            export_id=export.export_id,
            rate_package_id=42,
            package_version=7,
            package_status="PUBLISHED",
            rating_workbook_path=export.rating_workbook_path,
            was_existing=True,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_resolve_existing_published_run",
        lambda *args, **kwargs: existing,
    )

    result = pipeline.publish_model_export(
        object(),
        export,
        model_config=_config(),
        validated_model_id=17,
    )

    assert result is existing


def test_completed_publication_requires_prebuilt_manifest_evidence():
    assert "dataset" not in signature(publish_completed_model_build).parameters
    assert CompletedModelBuild.model_fields["manifest_id"].is_required()


@pytest.mark.parametrize(
    ("case", "manifest_overrides", "split_overrides", "split_missing", "match"),
    [
        (
            "pk-columns",
            {"pk_columns_json": json.dumps(["account_id"])},
            {},
            False,
            "pk_columns",
        ),
        ("row-count", {"row_count": 2}, {}, False, "row count"),
        (
            "model-frame",
            {"model_frame_sha256": "e" * 64},
            {},
            False,
            "model_frame_sha256",
        ),
        (
            "pre-v033-null-model-frame",
            {"model_frame_sha256": None},
            {},
            False,
            "model_frame_sha256",
        ),
        ("missing-split", {}, {}, True, "split_set_id.*not found"),
        (
            "split-owner",
            {},
            {"manifest_id": "manifest-other"},
            False,
            "does not belong",
        ),
        (
            "split-row-order",
            {},
            {"row_order_sha256": "f" * 64},
            False,
            "row_order_sha256",
        ),
    ],
)
def test_candidate_publication_rejects_untrusted_sql_lineage_before_publish(
    tmp_path,
    monkeypatch,
    case,
    manifest_overrides,
    split_overrides,
    split_missing,
    match,
):
    del case
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")
    settings = _settings(tmp_path)
    candidate_metadata = _candidate_metadata(
        settings.workbench_artifact_root,
        split_set_id="split-existing",
    )
    manifest_row = {
        "manifest_id": "manifest-existing",
        "row_count": 1,
        "pk_columns_json": json.dumps(["policy_id"]),
        "model_frame_sha256": "f" * 64,
        **manifest_overrides,
    }
    split_row = (
        None
        if split_missing
        else {
            "split_set_id": "split-existing",
            "manifest_id": "manifest-existing",
            "row_count": 1,
            "row_order_sha256": "d" * 64,
            **split_overrides,
        }
    )
    engine = _FakeCandidateLineageEngine(
        manifest_row=manifest_row,
        split_row=split_row,
    )
    publish_calls = []
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.configure_engine",
        lambda engine_arg, schemas: engine_arg,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.schema_names_from_connectable",
        lambda engine_arg: settings.schema_names,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_registered_model",
        lambda connection, config_arg: SimpleNamespace(model_id=17),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        lambda *args, **kwargs: publish_calls.append((args, kwargs)),
    )

    with pytest.raises(CompletedModelBuildError, match=match):
        publish_completed_model_build(
            engine,
            settings=settings,
            model_config=_config(),
            completed_build=_approved_build(
                tmp_path,
                workbook=workbook,
                candidate_metadata=candidate_metadata,
                effective_from="2026-06-03",
                split_set_id="split-existing",
            ),
        )

    assert publish_calls == []


@pytest.mark.parametrize(
    ("split_row", "should_reject"),
    [
        (
            {
                "split_set_id": "split-owned",
                "manifest_id": "manifest-existing",
                "row_count": 1,
                "row_order_sha256": "d" * 64,
            },
            True,
        ),
        (None, False),
    ],
    ids=("owned-split-omitted", "legitimate-no-split"),
)
def test_candidate_publication_resolves_omitted_split_against_sql_manifest(
    tmp_path,
    monkeypatch,
    split_row,
    should_reject,
):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")
    settings = _settings(tmp_path)
    candidate_metadata = _candidate_metadata(
        settings.workbench_artifact_root,
        split_set_id=None,
    )
    engine = _FakeCandidateLineageEngine(
        manifest_row={
            "manifest_id": "manifest-existing",
            "row_count": 1,
            "pk_columns_json": json.dumps(["policy_id"]),
            "model_frame_sha256": "f" * 64,
        },
        split_row=split_row,
    )
    publish_calls = []
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.configure_engine",
        lambda engine_arg, schemas: engine_arg,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.schema_names_from_connectable",
        lambda engine_arg: settings.schema_names,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_registered_model",
        lambda connection, config_arg: SimpleNamespace(model_id=17),
    )

    def fake_publish(
        engine_arg,
        export,
        *,
        model_config,
        validated_model_id,
        allowed_artifact_root=None,
    ):
        publish_calls.append(export)
        return _completed_publish_result(export)

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        fake_publish,
    )
    completed_build = _approved_build(
        tmp_path,
        workbook=workbook,
        candidate_metadata=candidate_metadata,
        effective_from="2026-06-03",
    )

    if should_reject:
        with pytest.raises(CompletedModelBuildError, match="omits split_set_id.*owns"):
            publish_completed_model_build(
                engine,
                settings=settings,
                model_config=_config(),
                completed_build=completed_build,
            )
        assert publish_calls == []
    else:
        result = publish_completed_model_build(
            engine,
            settings=settings,
            model_config=_config(),
            completed_build=completed_build,
        )
        assert result.split_set_id is None
        assert len(publish_calls) == 1


def test_publish_completed_model_build_carries_publication_receipt_fields(
    tmp_path,
    monkeypatch,
):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")
    receipt_path = tmp_path / "superglm_publication_receipt.json"
    receipt_sha256 = "b" * 64
    engine = _ValidationEngine()
    published_exports = []
    _patch_candidate_sql_lineage(monkeypatch)
    candidate_metadata = _candidate_metadata(
        _settings(tmp_path).workbench_artifact_root,
    )

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_registered_model",
        lambda connection, config_arg: SimpleNamespace(model_id=17),
    )

    def fake_publish(
        engine_arg,
        export,
        *,
        model_config,
        validated_model_id,
        allowed_artifact_root=None,
    ):
        published_exports.append(export)
        return _completed_publish_result(export)

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        fake_publish,
    )

    result = publish_completed_model_build(
        engine,
        settings=_settings(tmp_path),
        model_config=_config(),
        completed_build=_approved_build(
            tmp_path,
            workbook=workbook,
            candidate_metadata=candidate_metadata,
            effective_from="2026-06-03",
            publication_receipt_path=str(receipt_path),
            publication_receipt_sha256=receipt_sha256,
            metrics={"cv_pooled_deviance": 0.42},
            metric_scopes={"cv_pooled_deviance": "cv"},
            fold_metrics=({"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},),
        ),
    )

    assert published_exports[0].publication_receipt_path == str(receipt_path)
    assert published_exports[0].publication_receipt_sha256 == receipt_sha256
    assert (
        published_exports[0].candidate_artifact_path
        == candidate_metadata["candidate_artifact_path"]
    )
    assert published_exports[0].metrics == {"cv_pooled_deviance": 0.42}
    assert published_exports[0].fold_metrics[0]["metric_name"] == "deviance"
    assert result.publication_receipt_path == str(receipt_path)
    assert result.publication_receipt_sha256 == receipt_sha256


@pytest.mark.parametrize("artifact_state", ["missing", "tampered", "outside-root"])
def test_publish_completed_model_build_rejects_untrusted_candidate_before_publish(
    tmp_path,
    monkeypatch,
    artifact_state,
):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")
    settings = _settings(tmp_path)
    _patch_candidate_sql_lineage(monkeypatch)
    artifact_root = (
        tmp_path / "outside-workbench"
        if artifact_state == "outside-root"
        else settings.workbench_artifact_root
    )
    candidate_metadata = _candidate_metadata(artifact_root)
    artifact_path = Path(candidate_metadata["candidate_artifact_path"])
    if artifact_state == "missing":
        artifact_path.unlink()
    elif artifact_state == "tampered":
        artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")

    publish_calls = []
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_registered_model",
        lambda connection, config_arg: SimpleNamespace(model_id=17),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        lambda *args, **kwargs: publish_calls.append((args, kwargs)),
    )

    with pytest.raises(CompletedModelBuildError, match="candidate artifact"):
        publish_completed_model_build(
            _ValidationEngine(),
            settings=settings,
            model_config=_config(),
            completed_build=_approved_build(
                tmp_path,
                workbook=workbook,
                candidate_metadata=candidate_metadata,
                effective_from="2026-06-03",
            ),
        )

    assert publish_calls == []


@pytest.mark.parametrize(
    ("lineage_field", "published_value"),
    [
        ("manifest_id", "manifest-published"),
        ("split_set_id", "split-published"),
        ("model_source_sha256", "e" * 64),
        ("model_frame_sha256", "e" * 64),
    ],
)
def test_publish_completed_model_build_rejects_candidate_lineage_mismatch(
    tmp_path,
    monkeypatch,
    lineage_field,
    published_value,
):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")
    settings = _settings(tmp_path)
    _patch_candidate_sql_lineage(monkeypatch)
    candidate_metadata = _candidate_metadata(
        settings.workbench_artifact_root,
        manifest_id="manifest-existing",
        split_set_id="split-existing",
    )
    completed_build = _approved_build(
        tmp_path,
        workbook=workbook,
        candidate_metadata=candidate_metadata,
        effective_from="2026-06-03",
        split_set_id="split-existing",
    ).model_dump()
    completed_build[lineage_field] = published_value

    publish_calls = []
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_registered_model",
        lambda connection, config_arg: SimpleNamespace(model_id=17),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        lambda *args, **kwargs: publish_calls.append((args, kwargs)),
    )

    with pytest.raises(CompletedModelBuildError, match=lineage_field):
        publish_completed_model_build(
            _ValidationEngine(),
            settings=settings,
            model_config=_config(),
            completed_build=CompletedModelBuild(**completed_build),
        )

    assert publish_calls == []


def test_publish_completed_model_build_configures_engine_with_settings_schema_names(
    tmp_path,
    monkeypatch,
):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")
    raw_engine = object()
    configured_engine = _ValidationEngine()
    calls = []
    settings = _settings(tmp_path)
    candidate_metadata = _candidate_metadata(settings.workbench_artifact_root)
    _patch_candidate_sql_lineage(monkeypatch)

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.configure_engine",
        lambda engine_arg, schemas: (
            calls.append(("configure", engine_arg, schemas)) or configured_engine
        ),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_registered_model",
        lambda connection, config_arg: (
            calls.append(("validate", configured_engine, config_arg))
            or SimpleNamespace(model_id=17)
        ),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        lambda engine_arg, export, *, model_config, validated_model_id, allowed_artifact_root=None: (
            calls.append(("publish", engine_arg, export, model_config))
            or _completed_publish_result(export)
        ),
    )

    publish_completed_model_build(
        raw_engine,
        settings=settings,
        model_config=_config(),
        completed_build=_approved_build(
            tmp_path,
            workbook=workbook,
            candidate_metadata=candidate_metadata,
            effective_from="2026-06-03",
        ),
    )

    assert calls[0] == ("configure", raw_engine, settings.schema_names)
    assert calls[1][0] == "validate"
    assert calls[1][1] is configured_engine
    assert calls[2][0] == "publish"
    assert calls[2][1] is configured_engine
