from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.data.manifest import DatasetManifestResult
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.config import ModelBuildConfig, ValidationSplitConfig
from pricing_pipeline.models.spec import DatasetSpec, ModelExportResult
from pricing_pipeline.orchestration.publish_completed_build import (
    CandidateSQLLineage,
    CompletedModelBuild,
    CompletedModelBuildError,
    CompletedModelPublishResult,
    publish_completed_model_build,
)
from pricing_pipeline.workbench.artifacts import CandidateBundle, save_candidate_bundle


def _install_fake_airflow_taskflow(monkeypatch, context):
    airflow_module = SimpleNamespace()
    airflow_sdk_module = SimpleNamespace()

    def task(func=None, **task_kwargs):
        def decorator(inner):
            return SimpleNamespace(function=inner, task_id=task_kwargs.get("task_id"))

        if func is None:
            return decorator
        return decorator(func)

    airflow_sdk_module.get_current_context = lambda: context
    airflow_sdk_module.task = task
    airflow_module.sdk = airflow_sdk_module
    monkeypatch.setitem(sys.modules, "airflow", airflow_module)
    monkeypatch.setitem(sys.modules, "airflow.sdk", airflow_sdk_module)


def _config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_name="CLAIM_FREQ",
        model_label="Claim frequency",
        target_name="claim_count",
        model_type="superglm_poisson",
        deployment_slot="CLAIM_FREQ_CURRENT",
        default_package_status="PUBLISHED",
        validation_split=ValidationSplitConfig.none(),
    )


def _dataset() -> DatasetSpec:
    return DatasetSpec(
        dataset_name="claim_freq_training",
        source_system="azure_sql",
        manifest_sql="SELECT * FROM work.claim_freq_training",
        pk_columns=("policy_id",),
        target_column="claim_count",
        weight_column="earned_exposure",
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


def _candidate_metadata(
    artifact_root,
    *,
    manifest_id="manifest-existing",
    split_set_id=None,
    model_source_sha256="c" * 64,
    pk_columns=("policy_id",),
    row_count=1,
    row_order_sha256="d" * 64,
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
            manifest_id=manifest_id,
            split_set_id=split_set_id,
            pk_columns=pk_columns,
            row_order_sha256=row_order_sha256,
            model_source_sha256=model_source_sha256,
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
        "model_source_sha256": model_source_sha256,
    }


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
        ),
    )


def test_completed_model_build_round_trips_plain_dict(tmp_path):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")

    build = CompletedModelBuild(
        rating_workbook_path=str(workbook),
        model_version="20260603",
        effective_from="2026-06-03",
        mlflow_run_id=None,
        metrics={"deviance": 12.5},
    )

    payload = build.to_dict()
    assert payload["rating_workbook_path"] == str(workbook)
    assert payload["metrics"] == {"deviance": 12.5}
    assert CompletedModelBuild.from_mapping(payload) == build
    assert CompletedModelBuild.from_mapping(build) is build


def test_completed_model_build_accepts_complete_candidate_metadata():
    build = CompletedModelBuild(
        rating_workbook_path="rating.xlsx",
        model_version="v1",
        effective_from="2026-07-12",
        manifest_id="manifest-existing",
        candidate_artifact_path="candidate.joblib",
        candidate_artifact_sha256="a" * 64,
        candidate_artifact_format="superglm-candidate-joblib-v1",
        candidate_artifact_size_bytes=123,
        candidate_python_version="3.14.4",
        candidate_superglm_version="0.11.0",
        model_source_sha256="b" * 64,
        metrics={"cv_pooled_deviance": 0.42},
        metric_scopes={"cv_pooled_deviance": "cv"},
        fold_metrics=(
            {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},
        ),
    )

    assert build.candidate_artifact_size_bytes == 123
    assert build.metric_scopes["cv_pooled_deviance"] == "cv"
    assert build.fold_metrics[0]["metric_name"] == "deviance"


def test_completed_model_build_rejects_candidate_without_existing_manifest_id():
    with pytest.raises(CompletedModelBuildError, match="candidate.*existing manifest_id"):
        CompletedModelBuild(
            rating_workbook_path="rating.xlsx",
            model_version="v1",
            effective_from="2026-07-12",
            candidate_artifact_path="candidate.joblib",
            candidate_artifact_sha256="a" * 64,
            candidate_artifact_format="superglm-candidate-joblib-v1",
            candidate_artifact_size_bytes=123,
            candidate_python_version="3.14.4",
            candidate_superglm_version="0.11.0",
            model_source_sha256="b" * 64,
        )


def test_model_export_result_round_trips_candidate_metadata():
    export = ModelExportResult(
        model_id=17,
        model_name="CLAIM_FREQ",
        model_version="v1",
        model_type="superglm_poisson",
        target_name="claim_count",
        deployment_slot="CLAIM_FREQ_CURRENT",
        manifest_id="manifest-1",
        dag_id="pricing_claim_freq",
        airflow_run_id="scheduled__20260712",
        mlflow_run_id="",
        split_set_id="split-1",
        export_id="export-1",
        rating_workbook_path="rating.xlsx",
        effective_from="2026-07-12",
        created_by="airflow",
        candidate_artifact_path="candidate.joblib",
        candidate_artifact_sha256="a" * 64,
        candidate_artifact_format="superglm-candidate-joblib-v1",
        candidate_artifact_size_bytes=123,
        candidate_python_version="3.14.4",
        candidate_superglm_version="0.11.0",
        model_source_sha256="b" * 64,
        metrics={"cv_pooled_deviance": 0.42},
        metric_scopes={"cv_pooled_deviance": "cv"},
        fold_metrics=(
            {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},
        ),
    )

    restored = ModelExportResult.from_mapping(export.to_dict())

    assert restored == export
    assert restored.candidate_artifact_size_bytes == 123


def test_completed_model_build_rejects_partial_candidate_metadata():
    with pytest.raises(CompletedModelBuildError, match="candidate artifact fields"):
        CompletedModelBuild(
            rating_workbook_path="rating.xlsx",
            model_version="v1",
            effective_from="2026-07-12",
            candidate_artifact_path="candidate.joblib",
        )


def test_completed_model_build_to_dict_omits_unset_publication_receipt_fields():
    build = CompletedModelBuild(
        rating_workbook_path="/tmp/rating.xlsx",
        model_version="v1",
        effective_from="2026-06-19",
    )

    payload = build.to_dict()

    assert "publication_receipt_path" not in payload
    assert "publication_receipt_sha256" not in payload


def test_completed_model_build_accepts_publication_receipt_fields():
    build = CompletedModelBuild(
        rating_workbook_path="/tmp/rating.xlsx",
        model_version="v1",
        effective_from="2026-06-19",
        publication_receipt_path="/tmp/superglm_publication_receipt.json",
        publication_receipt_sha256="a" * 64,
    )

    assert build.publication_receipt_path == "/tmp/superglm_publication_receipt.json"
    assert build.publication_receipt_sha256 == "a" * 64
    assert build.to_dict()["publication_receipt_sha256"] == "a" * 64


def test_completed_model_build_rejects_bad_receipt_hash():
    with pytest.raises(CompletedModelBuildError, match="publication_receipt_sha256") as exc:
        CompletedModelBuild(
            rating_workbook_path="/tmp/rating.xlsx",
            model_version="v1",
            effective_from="2026-06-19",
            publication_receipt_sha256="not-a-hash",
        )

    assert "64-character lowercase hex SHA-256 digest" in str(exc.value)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "publication_receipt_path": "/tmp/superglm_publication_receipt.json",
        },
        {
            "publication_receipt_sha256": "a" * 64,
        },
    ],
)
def test_completed_model_build_requires_receipt_path_and_hash_together(payload):
    with pytest.raises(CompletedModelBuildError, match="publication_receipt_path.*sha256"):
        CompletedModelBuild(
            rating_workbook_path="/tmp/rating.xlsx",
            model_version="v1",
            effective_from="2026-06-19",
            **payload,
        )


def test_completed_model_build_rejects_unknown_mapping_keys():
    with pytest.raises(CompletedModelBuildError, match="unknown completed build field"):
        CompletedModelBuild.from_mapping(
            {
                "rating_workbook_path": "rating.xlsx",
                "model_version": "20260603",
                "effective_from": "2026-06-03",
                "unexpected": "value",
            }
        )


@pytest.mark.parametrize(
    ("field_name", "payload"),
    [
        (
            "rating_workbook_path",
            {"model_version": "20260603", "effective_from": "2026-06-03"},
        ),
        (
            "model_version",
            {"rating_workbook_path": "rating.xlsx", "effective_from": "2026-06-03"},
        ),
        (
            "effective_from",
            {"rating_workbook_path": "rating.xlsx", "model_version": "20260603"},
        ),
    ],
)
def test_completed_model_build_missing_required_mapping_fields_raise_domain_error(
    field_name,
    payload,
):
    with pytest.raises(CompletedModelBuildError, match=field_name):
        CompletedModelBuild.from_mapping(payload)


@pytest.mark.parametrize("payload", [None, "not-a-mapping", ["rating.xlsx"]])
def test_completed_model_build_rejects_non_mapping_payload(payload):
    with pytest.raises(CompletedModelBuildError, match="expected a mapping"):
        CompletedModelBuild.from_mapping(payload)


@pytest.mark.parametrize(
    ("field_name", "payload"),
    [
        (
            "rating_workbook_path",
            {
                "rating_workbook_path": "   ",
                "model_version": "v1",
                "effective_from": "2026-06-03",
            },
        ),
        (
            "model_version",
            {
                "rating_workbook_path": "rating.xlsx",
                "model_version": "   ",
                "effective_from": "2026-06-03",
            },
        ),
        (
            "effective_from",
            {
                "rating_workbook_path": "rating.xlsx",
                "model_version": "v1",
                "effective_from": "   ",
            },
        ),
    ],
)
def test_completed_model_build_rejects_blank_required_strings(field_name, payload):
    with pytest.raises(CompletedModelBuildError, match=field_name):
        CompletedModelBuild.from_mapping(payload)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (date(2026, 6, 3), "2026-06-03"),
        (datetime(2026, 6, 3, 14, 30), "2026-06-03"),
        ("2026-06-03T14:30:00", "2026-06-03"),
    ],
)
def test_completed_model_build_normalises_effective_from(raw_value, expected):
    build = CompletedModelBuild(
        rating_workbook_path="rating.xlsx",
        model_version="v1",
        effective_from=raw_value,
    )

    assert build.effective_from == expected
    assert build.to_dict()["effective_from"] == expected


def test_completed_model_build_rejects_numeric_effective_from():
    with pytest.raises(CompletedModelBuildError, match="effective_from"):
        CompletedModelBuild(
            rating_workbook_path="rating.xlsx",
            model_version="v1",
            effective_from=20260603,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "created_by",
        "export_id",
        "dag_id",
        "airflow_run_id",
        "mlflow_run_id",
        "manifest_id",
        "split_set_id",
        "model_artifact_path",
    ],
)
def test_completed_model_build_blank_optional_strings_normalise_to_none(field_name):
    build = CompletedModelBuild(
        rating_workbook_path="rating.xlsx",
        model_version="v1",
        effective_from="2026-06-03",
        **{field_name: "   "},
    )

    assert getattr(build, field_name) is None
    assert build.to_dict()[field_name] is None


def test_completed_model_build_rejects_string_metric():
    with pytest.raises(CompletedModelBuildError, match="deviance"):
        CompletedModelBuild(
            rating_workbook_path="rating.xlsx",
            model_version="v1",
            effective_from="2026-06-03",
            metrics={"deviance": "12.5"},
        )


@pytest.mark.parametrize("bad_metric", [True, [12.5], {"nested": 12.5}])
def test_completed_model_build_rejects_non_numeric_metric_values(bad_metric):
    with pytest.raises(CompletedModelBuildError, match="deviance"):
        CompletedModelBuild(
            rating_workbook_path="rating.xlsx",
            model_version="v1",
            effective_from="2026-06-03",
            metrics={"deviance": bad_metric},
        )


@pytest.mark.parametrize("bad_metric", [float("nan"), float("inf"), -float("inf")])
def test_completed_model_build_rejects_non_finite_metric(bad_metric):
    with pytest.raises(CompletedModelBuildError, match="finite"):
        CompletedModelBuild(
            rating_workbook_path="rating.xlsx",
            model_version="v1",
            effective_from="2026-06-03",
            metrics={"deviance": bad_metric},
        )


def test_completed_build_publish_api_does_not_import_dag_factory():
    source = Path("pricing_pipeline/orchestration/publish_completed_build.py").read_text(
        encoding="utf-8"
    )

    assert "orchestration.dag_factory" not in source


def test_completed_model_publish_result_to_dict():
    result = CompletedModelPublishResult(
        model_id=17,
        model_name="CLAIM_FREQ",
        model_version="20260603",
        manifest_id="manifest-1",
        split_set_id=None,
        export_id="export-1",
        rate_package_id=42,
        package_version=3,
        package_status="PUBLISHED",
        rating_workbook_path="/tmp/rating.xlsx",
        mlflow_run_id=None,
        was_existing=False,
    )

    assert result.to_dict()["package_status"] == "PUBLISHED"
    assert result.to_dict()["was_existing"] is False


def test_publish_completed_model_build_creates_manifest_and_delegates(
    tmp_path,
    monkeypatch,
):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")
    engine = object()
    calls = []

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_model_on_engine",
        lambda engine_arg, config_arg: calls.append(("validate", engine_arg, config_arg)) or 17,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.new_manifest_id",
        lambda dataset_name: f"{dataset_name}_manifest",
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.create_dataset_manifest_with_split",
        lambda engine_arg, **kwargs: (
            calls.append(("manifest", engine_arg, kwargs))
            or DatasetManifestResult(
                manifest_id="claim_freq_training_manifest",
                split_set_id="split-1",
                split_artifact_uri=None,
            )
        ),
    )

    def fake_publish(engine_arg, export, *, model_config, allowed_artifact_root=None):
        calls.append(("publish", engine_arg, export, model_config))
        assert isinstance(export, ModelExportResult)
        return {
            "mlflow_run_id": "mlflow-1",
            "export_id": "export-1",
            "rate_package_id": "42",
            "package_version": "7",
            "rating_workbook_path": str(workbook),
        }

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        fake_publish,
    )

    result = publish_completed_model_build(
        engine,
        settings=_settings(tmp_path),
        model_config=_config(),
        dataset=_dataset(),
        completed_build={
            "rating_workbook_path": str(workbook),
            "model_version": "20260603",
            "effective_from": "2026-06-03",
            "export_id": "export-1",
        },
        created_by="airflow",
    )

    assert result.model_id == 17
    assert result.manifest_id == "claim_freq_training_manifest"
    assert result.split_set_id == "split-1"
    assert result.package_status == "PUBLISHED"
    assert result.mlflow_run_id == "mlflow-1"
    assert calls[0] == ("validate", engine, _config())
    assert calls[1][0] == "manifest"
    assert calls[2][0] == "publish"


def test_publish_completed_model_build_returns_canonical_retry_lineage_and_discards_attempt(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    retry_dir = settings.workbench_artifact_root / "HOME_FREQ" / "export-1" / "manifest-retry"
    retry_dir.mkdir(parents=True)
    retry_workbook = retry_dir / "rating_tables.xlsx"
    retry_workbook.write_bytes(b"retry workbook")
    original_dir = settings.workbench_artifact_root / "HOME_FREQ" / "export-1" / "manifest-original"
    original_dir.mkdir(parents=True)
    original_workbook = original_dir / "rating_tables.xlsx"
    original_workbook.write_bytes(b"original workbook")

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_model_on_engine",
        lambda engine, config: 17,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.new_manifest_id",
        lambda dataset_name: "manifest-retry",
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.create_dataset_manifest_with_split",
        lambda *args, **kwargs: DatasetManifestResult(
            manifest_id="manifest-retry",
            split_set_id="split-retry",
            split_artifact_uri=None,
        ),
    )

    def fake_publish(engine, export, *, model_config, allowed_artifact_root=None):
        assert allowed_artifact_root == settings.workbench_artifact_root
        return {
            "mlflow_run_id": "mlflow-original",
            "export_id": "export-1",
            "rate_package_id": "42",
            "package_version": "7",
            "package_status": "PUBLISHED",
            "rating_workbook_path": str(original_workbook),
            "model_run_id": "901",
            "manifest_id": "manifest-original",
            "split_set_id": "split-original",
            "was_existing": True,
        }

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        fake_publish,
    )

    result = publish_completed_model_build(
        object(),
        settings=settings,
        model_config=_config(),
        dataset=_dataset(),
        completed_build={
            "rating_workbook_path": str(retry_workbook),
            "model_version": "20260603",
            "effective_from": "2026-06-03",
            "export_id": "export-1",
            "created_by": "airflow",
        },
    )

    assert result.manifest_id == "manifest-original"
    assert result.split_set_id == "split-original"
    assert result.rating_workbook_path == str(original_workbook)
    assert result.was_existing is True
    assert not retry_dir.exists()
    assert original_workbook.read_bytes() == b"original workbook"


def test_candidate_build_requires_existing_manifest_before_database_work(
    tmp_path,
    monkeypatch,
):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")
    settings = _settings(tmp_path)
    candidate_metadata = _candidate_metadata(settings.workbench_artifact_root)
    database_calls = []
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_model_on_engine",
        lambda *args, **kwargs: database_calls.append("validate_model"),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.create_dataset_manifest_with_split",
        lambda *args, **kwargs: database_calls.append("create_manifest"),
    )

    with pytest.raises(CompletedModelBuildError, match="candidate.*existing manifest_id"):
        publish_completed_model_build(
            object(),
            settings=settings,
            model_config=_config(),
            dataset=_dataset(),
            completed_build={
                "rating_workbook_path": str(workbook),
                "model_version": "20260603",
                "effective_from": "2026-06-03",
                "export_id": "export-1",
                "created_by": "airflow",
                **candidate_metadata,
            },
        )

    assert database_calls == []


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
        **manifest_overrides,
    }
    split_row = None if split_missing else {
        "split_set_id": "split-existing",
        "manifest_id": "manifest-existing",
        "row_count": 1,
        "row_order_sha256": "d" * 64,
        **split_overrides,
    }
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
        "pricing_pipeline.orchestration.publish_completed_build.validate_model_on_engine",
        lambda engine_arg, config_arg: 17,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_existing_manifest",
        lambda engine_arg, manifest_id: None,
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
            dataset=None,
            completed_build={
                "rating_workbook_path": str(workbook),
                "model_version": "20260603",
                "effective_from": "2026-06-03",
                "export_id": "export-1",
                "manifest_id": "manifest-existing",
                "split_set_id": "split-existing",
                "created_by": "airflow",
                **candidate_metadata,
            },
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
        "pricing_pipeline.orchestration.publish_completed_build.validate_model_on_engine",
        lambda engine_arg, config_arg: 17,
    )

    def fake_publish(engine_arg, export, *, model_config, allowed_artifact_root=None):
        publish_calls.append(export)
        return {
            "mlflow_run_id": "",
            "export_id": export.export_id,
            "rate_package_id": "42",
            "package_version": "7",
            "rating_workbook_path": export.rating_workbook_path,
        }

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        fake_publish,
    )
    completed_build = {
        "rating_workbook_path": str(workbook),
        "model_version": "20260603",
        "effective_from": "2026-06-03",
        "export_id": "export-1",
        "manifest_id": "manifest-existing",
        "created_by": "airflow",
        **candidate_metadata,
    }

    if should_reject:
        with pytest.raises(CompletedModelBuildError, match="omits split_set_id.*owns"):
            publish_completed_model_build(
                engine,
                settings=settings,
                model_config=_config(),
                dataset=None,
                completed_build=completed_build,
            )
        assert publish_calls == []
    else:
        result = publish_completed_model_build(
            engine,
            settings=settings,
            model_config=_config(),
            dataset=None,
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
    engine = object()
    published_exports = []
    _patch_candidate_sql_lineage(monkeypatch)
    candidate_metadata = _candidate_metadata(
        _settings(tmp_path).workbench_artifact_root,
    )

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_model_on_engine",
        lambda engine_arg, config_arg: 17,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_existing_manifest",
        lambda engine_arg, manifest_id: None,
    )

    def fake_publish(engine_arg, export, *, model_config, allowed_artifact_root=None):
        published_exports.append(export)
        return {
            "mlflow_run_id": "",
            "export_id": export.export_id,
            "rate_package_id": "42",
            "package_version": "7",
            "rating_workbook_path": export.rating_workbook_path,
            "publication_receipt_path": export.publication_receipt_path,
            "publication_receipt_sha256": export.publication_receipt_sha256,
        }

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        fake_publish,
    )

    result = publish_completed_model_build(
        engine,
        settings=_settings(tmp_path),
        model_config=_config(),
        dataset=None,
        completed_build={
            "rating_workbook_path": str(workbook),
            "model_version": "20260603",
            "effective_from": "2026-06-03",
            "export_id": "export-1",
            "manifest_id": "manifest-existing",
            "created_by": "airflow",
            "publication_receipt_path": str(receipt_path),
            "publication_receipt_sha256": receipt_sha256,
            **candidate_metadata,
            "metrics": {"cv_pooled_deviance": 0.42},
            "metric_scopes": {"cv_pooled_deviance": "cv"},
            "fold_metrics": (
                {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},
            ),
        },
    )

    assert published_exports[0].publication_receipt_path == str(receipt_path)
    assert published_exports[0].publication_receipt_sha256 == receipt_sha256
    assert published_exports[0].candidate_artifact_path == candidate_metadata[
        "candidate_artifact_path"
    ]
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
        "pricing_pipeline.orchestration.publish_completed_build.validate_model_on_engine",
        lambda engine_arg, config_arg: 17,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_existing_manifest",
        lambda engine_arg, manifest_id: None,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        lambda *args, **kwargs: publish_calls.append((args, kwargs)),
    )

    with pytest.raises(CompletedModelBuildError, match="candidate artifact"):
        publish_completed_model_build(
            object(),
            settings=settings,
            model_config=_config(),
            dataset=None,
            completed_build={
                "rating_workbook_path": str(workbook),
                "model_version": "20260603",
                "effective_from": "2026-06-03",
                "export_id": "export-1",
                "manifest_id": "manifest-existing",
                "created_by": "airflow",
                **candidate_metadata,
            },
        )

    assert publish_calls == []


@pytest.mark.parametrize(
    ("lineage_field", "published_value"),
    [
        ("manifest_id", "manifest-published"),
        ("split_set_id", "split-published"),
        ("model_source_sha256", "e" * 64),
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
    completed_build = {
        "rating_workbook_path": str(workbook),
        "model_version": "20260603",
        "effective_from": "2026-06-03",
        "export_id": "export-1",
        "manifest_id": "manifest-existing",
        "split_set_id": "split-existing",
        "created_by": "airflow",
        **candidate_metadata,
    }
    completed_build[lineage_field] = published_value

    publish_calls = []
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_model_on_engine",
        lambda engine_arg, config_arg: 17,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_existing_manifest",
        lambda engine_arg, manifest_id: None,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        lambda *args, **kwargs: publish_calls.append((args, kwargs)),
    )

    with pytest.raises(CompletedModelBuildError, match=lineage_field):
        publish_completed_model_build(
            object(),
            settings=settings,
            model_config=_config(),
            dataset=None,
            completed_build=completed_build,
        )

    assert publish_calls == []


def test_publish_completed_model_build_configures_engine_with_settings_schema_names(
    tmp_path,
    monkeypatch,
):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")
    raw_engine = object()
    configured_engine = object()
    calls = []

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.configure_engine",
        lambda engine_arg, schemas: (
            calls.append(("configure", engine_arg, schemas)) or configured_engine
        ),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_model_on_engine",
        lambda engine_arg, config_arg: calls.append(("validate", engine_arg, config_arg)) or 17,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_existing_manifest",
        lambda engine_arg, manifest_id: calls.append(
            ("validate_manifest", engine_arg, manifest_id)
        ),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        lambda engine_arg, export, *, model_config, allowed_artifact_root=None: (
            calls.append(("publish", engine_arg, export, model_config))
            or {
                "mlflow_run_id": "",
                "export_id": export.export_id,
                "rate_package_id": "42",
                "package_version": "7",
                "rating_workbook_path": export.rating_workbook_path,
            }
        ),
    )

    settings = _settings(tmp_path)
    publish_completed_model_build(
        raw_engine,
        settings=settings,
        model_config=_config(),
        dataset=None,
        completed_build={
            "rating_workbook_path": str(workbook),
            "model_version": "20260603",
            "effective_from": "2026-06-03",
            "export_id": "export-1",
            "manifest_id": "manifest-existing",
            "created_by": "airflow",
        },
    )

    assert calls[0] == ("configure", raw_engine, settings.schema_names)
    assert calls[1][0] == "validate"
    assert calls[1][1] is configured_engine
    assert calls[2] == ("validate_manifest", configured_engine, "manifest-existing")
    assert calls[3][0] == "publish"
    assert calls[3][1] is configured_engine


def test_publish_completed_model_build_reuses_and_validates_supplied_manifest(
    tmp_path,
    monkeypatch,
):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")
    engine = object()
    calls = []

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_model_on_engine",
        lambda engine_arg, config_arg: 17,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_existing_manifest",
        lambda engine_arg, manifest_id: calls.append(
            ("validate_manifest", engine_arg, manifest_id)
        ),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.create_dataset_manifest_with_split",
        lambda *args, **kwargs: calls.append(("create_manifest", args, kwargs)),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        lambda engine_arg, export, *, model_config, allowed_artifact_root=None: {
            "mlflow_run_id": "",
            "export_id": export.export_id,
            "rate_package_id": "42",
            "package_version": "7",
            "rating_workbook_path": export.rating_workbook_path,
        },
    )

    result = publish_completed_model_build(
        engine,
        settings=_settings(tmp_path),
        model_config=_config(),
        dataset=None,
        completed_build={
            "rating_workbook_path": str(workbook),
            "model_version": "20260603",
            "effective_from": "2026-06-03",
            "export_id": "export-1",
            "manifest_id": "manifest-existing",
            "split_set_id": None,
            "created_by": "airflow",
        },
    )

    assert result.manifest_id == "manifest-existing"
    assert calls == [("validate_manifest", engine, "manifest-existing")]


def test_publish_completed_model_build_requires_dataset_without_manifest(tmp_path):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")

    with pytest.raises(CompletedModelBuildError, match="dataset is required"):
        publish_completed_model_build(
            object(),
            settings=_settings(tmp_path),
            model_config=_config(),
            dataset=None,
            completed_build={
                "rating_workbook_path": str(workbook),
                "model_version": "20260603",
                "effective_from": "2026-06-03",
                "export_id": "export-1",
                "created_by": "airflow",
            },
        )


def test_publish_completed_model_build_task_fills_airflow_context(
    tmp_path,
    monkeypatch,
):
    calls = []
    config = _config()
    dataset = _dataset()

    @dataclass(frozen=True)
    class FakeDag:
        dag_id: str

    class FakeRuntime:
        settings = _settings(tmp_path)

        def get_engine(self):
            return "engine"

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.runtime_from_env_or_module",
        lambda runtime_module=None, *, env=None: FakeRuntime(),
    )
    _install_fake_airflow_taskflow(
        monkeypatch,
        {
            "dag": FakeDag("claim_freq_build"),
            "run_id": "manual__20260603",
        },
    )

    def fake_publish(engine, **kwargs):
        calls.append((engine, kwargs))
        return CompletedModelPublishResult(
            model_id=17,
            model_name="CLAIM_FREQ",
            model_version="20260603",
            manifest_id="manifest-1",
            split_set_id=None,
            export_id="claim_freq__manual__20260603",
            rate_package_id=42,
            package_version=7,
            package_status="PUBLISHED",
            rating_workbook_path=kwargs["completed_build"]["rating_workbook_path"],
        )

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_completed_model_build",
        fake_publish,
    )

    from pricing_pipeline.orchestration.publish_completed_build import (
        publish_completed_model_build_task,
    )

    task_callable = publish_completed_model_build_task(
        model_config=config,
        dataset=dataset,
        created_by="airflow",
    )
    result = task_callable.function(
        {
            "rating_workbook_path": str(tmp_path / "rating_tables.xlsx"),
            "model_version": "20260603",
            "effective_from": "2026-06-03",
        }
    )

    assert result["model_name"] == "CLAIM_FREQ"
    assert calls[0][0] == "engine"
    completed = calls[0][1]["completed_build"]
    assert completed["dag_id"] == "claim_freq_build"
    assert completed["airflow_run_id"] == "manual__20260603"
    assert completed["created_by"] == "airflow"


def test_publish_completed_model_build_task_allows_dataset_none_with_manifest(
    tmp_path,
    monkeypatch,
):
    calls = []
    config = _config()

    @dataclass(frozen=True)
    class FakeDag:
        dag_id: str

    class FakeRuntime:
        settings = _settings(tmp_path)

        def get_engine(self):
            return "engine"

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.runtime_from_env_or_module",
        lambda runtime_module=None, *, env=None: FakeRuntime(),
    )
    _install_fake_airflow_taskflow(
        monkeypatch,
        {
            "dag": FakeDag("claim_freq_build"),
            "run_id": "manual__20260603",
        },
    )

    def fake_publish(engine, **kwargs):
        calls.append((engine, kwargs))
        return CompletedModelPublishResult(
            model_id=17,
            model_name="CLAIM_FREQ",
            model_version="20260603",
            manifest_id=kwargs["completed_build"]["manifest_id"],
            split_set_id=None,
            export_id="claim_freq__manual__20260603",
            rate_package_id=42,
            package_version=7,
            package_status="PUBLISHED",
            rating_workbook_path=kwargs["completed_build"]["rating_workbook_path"],
        )

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_completed_model_build",
        fake_publish,
    )

    from pricing_pipeline.orchestration.publish_completed_build import (
        publish_completed_model_build_task,
    )

    task_callable = publish_completed_model_build_task(
        model_config=config,
        created_by="airflow",
    )
    result = task_callable.function(
        {
            "rating_workbook_path": str(tmp_path / "rating_tables.xlsx"),
            "model_version": "20260603",
            "effective_from": "2026-06-03",
            "manifest_id": "manifest-existing",
        }
    )

    assert result["manifest_id"] == "manifest-existing"
    assert calls[0][1]["dataset"] is None


def test_register_pricing_model_task_uses_global_registry_helper(monkeypatch):
    from pricing_pipeline.orchestration import model_registry_tasks

    calls = []
    engine = object()
    config = _config()

    monkeypatch.setattr(
        model_registry_tasks,
        "runtime_from_env_or_module",
        lambda runtime_module=None: SimpleNamespace(get_engine=lambda: engine),
    )
    monkeypatch.setattr(
        model_registry_tasks,
        "ensure_pricing_model",
        lambda engine_arg, **kwargs: (
            calls.append(("ensure_pricing_model", engine_arg, kwargs)) or 17
        ),
    )

    task_callable = model_registry_tasks.register_pricing_model_task(
        model_config=config,
        runtime_module="work_runtime.database",
        created_by="pytest",
        task_id="register_claim_freq",
    )

    assert task_callable.function() == 17
    assert calls == [
        (
            "ensure_pricing_model",
            engine,
            {
                "model_name": "CLAIM_FREQ",
                "model_label": "Claim frequency",
                "target_name": "claim_count",
                "model_type": "superglm_poisson",
                "created_by": "pytest",
            },
        )
    ]


def test_create_prepared_dataset_manifest_task_carries_payload(monkeypatch, tmp_path):
    from pricing_pipeline.orchestration import manifest_tasks

    calls = []
    engine = object()
    config = _config()

    monkeypatch.setattr(
        manifest_tasks,
        "runtime_from_env_or_module",
        lambda runtime_module=None: SimpleNamespace(
            get_engine=lambda: engine,
            settings=SimpleNamespace(validation_split_artifact_root=tmp_path / "splits"),
        ),
    )
    monkeypatch.setattr(
        manifest_tasks,
        "create_model_build_manifest",
        lambda engine_arg, **kwargs: (
            calls.append(("create_model_build_manifest", engine_arg, kwargs))
            or DatasetManifestResult(
                manifest_id="manifest-1",
                split_set_id="split-1",
                split_artifact_uri=None,
            )
        ),
    )

    def dataset_builder(payload):
        return DatasetSpec(
            dataset_name="claim_freq_training",
            source_system="azure_sql",
            manifest_sql=f"SELECT * FROM work.{payload['training_table']}",
            pk_columns=("policy_id",),
            target_column="claim_count",
            weight_column="earned_exposure",
        )

    task_callable = manifest_tasks.create_prepared_dataset_manifest_task(
        model_config=config,
        dataset_builder=dataset_builder,
        runtime_module="work_runtime.database",
        created_by="pytest",
        task_id="create_claim_freq_manifest",
    )

    result = task_callable.function(
        {
            "training_table": "CLAIM_FREQ_RUN_1",
            "training_frame_path": "/shared/claim_freq/training.csv",
        }
    )

    assert result == {
        "training_table": "CLAIM_FREQ_RUN_1",
        "training_frame_path": "/shared/claim_freq/training.csv",
        "manifest_id": "manifest-1",
        "split_set_id": "split-1",
    }
    manifest_call = calls[0]
    assert manifest_call[1] is engine
    assert manifest_call[2]["dataset"].manifest_sql == ("SELECT * FROM work.CLAIM_FREQ_RUN_1")
    assert manifest_call[2]["model_config"] == config
    assert manifest_call[2]["validation_split_artifact_root"] == tmp_path / "splits"
    assert manifest_call[2]["created_by"] == "pytest"
