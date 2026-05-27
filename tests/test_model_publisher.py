import pytest

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.lifecycle import (
    DeploymentResult,
    RatePackageSelector,
    RatePackageRevisionResult,
    RatePackageSnapshot,
)
from pricing_pipeline.publishing.publisher import ModelPublisher


def config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_key="MTPL_FREQ",
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
    assert publisher.config.model_key == "MTPL_FREQ"


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
        lambda engine_arg, config_arg, selector_arg: calls.append(
            (engine_arg, config_arg, selector_arg),
        )
        or expected,
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
