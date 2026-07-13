import pytest
import pandas as pd

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import ModelExportResult
from pricing_pipeline.publishing.lifecycle import (
    DeploymentResult,
    PublishResult,
    RatePackageSelector,
    RatePackageRevisionResult,
    RatePackageSnapshot,
)
from pricing_pipeline.publishing.model_registry import ModelRegistryError
from pricing_pipeline.publishing.publisher import ModelPublisher


def config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_name="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot="MTPL_FREQ_UAT",
        default_package_status="PUBLISHED",
    )


def test_model_publisher_stores_engine_and_config():
    engine = object()
    publisher = ModelPublisher(engine, config())

    assert publisher.engine is engine
    assert publisher.config.model_name == "MTPL_FREQ"


def test_rate_package_selector_requires_one_selector():
    assert RatePackageSelector(rate_package_id=123).rate_package_id == 123
    assert RatePackageSelector(package_version=7).package_version == 7
    with pytest.raises(ValueError):
        RatePackageSelector()
    with pytest.raises(ValueError):
        RatePackageSelector(rate_package_id=123, package_version=7)


def test_model_publisher_validate_registered_model_delegates(monkeypatch):
    calls = []
    engine = object()
    publisher = ModelPublisher(engine, config())

    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.validate_model_on_engine",
        lambda engine_arg, config_arg: calls.append((engine_arg, config_arg)) or 17,
    )

    assert publisher.validate_registered_model() == 17
    assert calls == [(engine, config())]


def test_model_publisher_deploy_validates_model_and_delegates(monkeypatch):
    calls = []
    engine = object()
    publisher = ModelPublisher(engine, config())
    expected = DeploymentResult(
        model_id=17,
        deployment_slot="MTPL_FREQ_UAT",
        previous_rate_package_id=101,
        rate_package_id=202,
        package_version=4,
        deployed_by="airflow",
        deployment_reason="approved",
    )

    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.validate_model_on_engine",
        lambda engine_arg, config_arg: calls.append(("validate", engine_arg, config_arg)) or 17,
    )

    def fake_deploy_rate_package(engine_arg, config_arg, **kwargs):
        calls.append(("deploy", engine_arg, config_arg, kwargs))
        return expected

    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.deploy_rate_package",
        fake_deploy_rate_package,
        raising=False,
    )

    result = publisher.deploy(
        package_version=4,
        expected_current_rate_package_id=101,
        deployment_reason="approved",
        deployed_by="airflow",
    )

    assert result == expected
    assert calls == [
        ("validate", engine, config()),
        (
            "deploy",
            engine,
            config(),
            {
                "rate_package_id": None,
                "package_version": 4,
                "expected_current_rate_package_id": 101,
                "deployment_slot": None,
                "deployment_reason": "approved",
                "deployed_by": "airflow",
                "model_id": 17,
            },
        ),
    ]


def test_model_publisher_load_rate_package_delegates(monkeypatch):
    calls = []
    engine = object()
    publisher = ModelPublisher(engine, config())
    selector = RatePackageSelector(rate_package_id=123)
    expected = RatePackageSnapshot(
        metadata={"rate_package_id": 123},
        terms=object(),
        rate_cells=object(),
        cell_levels=object(),
        compiled_rate_cells=object(),
        compiled_1d_bands=object(),
    )

    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.load_rate_package_snapshot",
        lambda engine_arg, config_arg, selector_arg: (
            calls.append(
                (engine_arg, config_arg, selector_arg),
            )
            or expected
        ),
    )

    assert publisher.load_rate_package(selector) == expected
    assert calls == [(engine, config(), selector)]


def test_model_publisher_create_manual_revision_validates_model_and_delegates(
    monkeypatch,
):
    calls = []
    engine = object()
    publisher = ModelPublisher(engine, config())
    parent = object()
    edited_rate_cells = object()
    expected = RatePackageRevisionResult(
        rate_package_id=303,
        package_version=5,
        parent_rate_package_id=202,
        changed_rate_cell_count=2,
        base_rate_changed=False,
        diff_summary=object(),
    )

    monkeypatch.setattr(
        publisher,
        "validate_registered_model",
        lambda: calls.append(("validate",)) or 17,
    )

    def fake_create_manual_revision(engine_arg, config_arg, **kwargs):
        calls.append(("create", engine_arg, config_arg, kwargs))
        return expected

    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.create_manual_revision",
        fake_create_manual_revision,
        raising=False,
    )

    result = publisher.create_manual_revision(
        parent=parent,
        edited_rate_cells=edited_rate_cells,
        reason="pricing correction",
        created_by="pricing-user",
    )

    assert result == expected
    assert calls == [
        ("validate",),
        (
            "create",
            engine,
            config(),
            {
                "parent": parent,
                "edited_rate_cells": edited_rate_cells,
                "reason": "pricing correction",
                "created_by": "pricing-user",
            },
        ),
    ]


def test_model_publisher_compare_prediction_vectors_delegates(monkeypatch):
    calls = []
    engine = object()
    publisher = ModelPublisher(engine, config())
    before = pd.Series([1.0])
    after = pd.Series([2.0])
    expected = object()

    def fake_compare_prediction_vectors(before_arg, after_arg, *, top_n):
        calls.append(
            {
                "before": before_arg,
                "after": after_arg,
                "top_n": top_n,
            },
        )
        return expected

    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.compare_prediction_vectors",
        fake_compare_prediction_vectors,
        raising=False,
    )

    result = publisher.compare_prediction_vectors(before, after, top_n=3)

    assert result is expected
    assert len(calls) == 1
    assert calls[0]["before"] is before
    assert calls[0]["after"] is after
    assert calls[0]["top_n"] == 3


def test_model_publisher_publish_training_export_rejects_mismatched_export_identity(
    tmp_path,
    monkeypatch,
):
    engine = object()
    export = ModelExportResult(
        model_id=17,
        model_name="OTHER_MODEL",
        model_version="20260527",
        model_type="superglm_poisson",
        target_name="ClaimNb",
        deployment_slot="MTPL_FREQ_UAT",
        manifest_id="manifest-1",
        dag_id="dag",
        airflow_run_id="scheduled__2026-05-27",
        mlflow_run_id="mlflow-1",
        split_set_id=None,
        export_id="export-1",
        rating_workbook_path=str(tmp_path / "rating_tables.xlsx"),
        effective_from="2026-05-27",
        created_by="airflow",
    )

    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.validate_model_on_engine",
        lambda engine_arg, config_arg: 17,
    )
    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.stage_rating_export",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.publish_rating_package",
        lambda *args, **kwargs: PublishResult(
            mlflow_run_id="",
            export_id="export-1",
            rate_package_id=42,
            package_version=6,
            rating_workbook_path="",
        ),
    )

    with pytest.raises(ModelRegistryError, match="model_name"):
        ModelPublisher(engine, config()).publish_training_export(export)


def test_model_publisher_publish_training_export_validates_and_delegates(
    tmp_path,
    monkeypatch,
):
    calls = []
    engine = object()
    export = ModelExportResult(
        model_id=17,
        model_name="MTPL_FREQ",
        model_version="20260527",
        model_type="superglm_poisson",
        target_name="ClaimNb",
        deployment_slot="MTPL_FREQ_UAT",
        manifest_id="manifest-1",
        dag_id="dag",
        airflow_run_id="scheduled__2026-05-27",
        mlflow_run_id="mlflow-1",
        split_set_id=None,
        export_id="export-1",
        rating_workbook_path=str(tmp_path / "rating_tables.xlsx"),
        effective_from="2026-05-27",
        created_by="airflow",
    )

    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.validate_model_on_engine",
        lambda engine_arg, config_arg: calls.append(("validate", engine_arg, config_arg)) or 17,
    )

    def fake_stage_rating_export(engine_arg, **kwargs):
        calls.append(("stage", engine_arg, kwargs))

    def fake_publish_rating_package(engine_arg, **kwargs):
        calls.append(("publish", engine_arg, kwargs))
        return PublishResult(
            mlflow_run_id="",
            export_id="export-1",
            rate_package_id=42,
            package_version=6,
            rating_workbook_path="",
        )

    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.stage_rating_export",
        fake_stage_rating_export,
    )
    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.publish_rating_package",
        fake_publish_rating_package,
    )

    def lineage_writer(connection, rate_package_id):
        return None

    result = ModelPublisher(engine, config()).publish_training_export(
        export,
        package_lineage_writer=lineage_writer,
    )

    assert result.rate_package_id == 42
    assert calls[0] == ("validate", engine, config())
    stage_call = calls[1]
    assert stage_call[0] == "stage"
    assert stage_call[2]["model_id"] == 17
    publish_call = calls[2]
    assert publish_call[0] == "publish"
    assert publish_call[2]["package_lineage_writer"] is lineage_writer
    assert publish_call[2]["expected_staged_metadata"] == {
        "export_id": "export-1",
        "model_id": 17,
        "model_name": "MTPL_FREQ",
        "model_version": "20260527",
        "effective_from_date": "2026-05-27",
        "effective_to_date": None,
        "source_file": str((tmp_path / "rating_tables.xlsx").resolve()),
        "publication_receipt_sha256": None,
    }
