from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from pricing_pipeline.data.manifest import DatasetManifestResult
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.config import ModelBuildConfig, ValidationSplitConfig
from pricing_pipeline.models.spec import DatasetSpec, ModelExportResult
from pricing_pipeline.orchestration.publish_completed_build import (
    CompletedModelBuild,
    CompletedModelBuildError,
    CompletedModelPublishResult,
    publish_completed_model_build,
)


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

    def fake_publish(engine_arg, export, *, model_config):
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

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_model_on_engine",
        lambda engine_arg, config_arg: 17,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_existing_manifest",
        lambda engine_arg, manifest_id: None,
    )

    def fake_publish(engine_arg, export, *, model_config):
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
        },
    )

    assert published_exports[0].publication_receipt_path == str(receipt_path)
    assert published_exports[0].publication_receipt_sha256 == receipt_sha256
    assert result.publication_receipt_path == str(receipt_path)
    assert result.publication_receipt_sha256 == receipt_sha256


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
        lambda engine_arg, export, *, model_config: (
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
        lambda engine_arg, export, *, model_config: {
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
