from __future__ import annotations

from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.config import ModelBuildConfig, ValidationSplitConfig
from pricing_pipeline.orchestration.publish_completed_build import (
    CompletedModelBuild,
    CompletedModelPublishResult,
)
from pricing_pipeline.publishing.model_registry import PricingModelRecord


def _context(api, tmp_path: Path):
    return api.NotebookContext(
        engine=object(),
        settings=Settings(
            workbench_artifact_root=tmp_path / "workbench",
            validation_split_artifact_root=tmp_path / "splits",
        ),
        mode="remote",
        write_allowed=True,
        destination="remote SQL database: PricingAudit",
    )


def _registered_model(api, tmp_path: Path):
    source_root = tmp_path / "pricing_models" / "claim_frequency"
    source_root.mkdir(parents=True)
    (source_root / "model.py").write_text("MODEL = 'claim_frequency'\n", encoding="utf-8")
    spec = api.PricingModelSpec(
        name="CLAIM_FREQUENCY",
        label="Claim frequency",
        target="claim_count",
        model_type="superglm_poisson",
        deployment_slot="PRODUCTION",
        features=("age", "region"),
        dataset_name="claim_frequency_frame",
        source_system="pricing_sql",
        pk_columns=("policy_id",),
        exposure_column="exposure",
        validation=ValidationSplitConfig.kfold(n_splits=2, random_state=7),
    )
    return api.RegisteredModel(
        model_id=17,
        config=ModelBuildConfig(
            model_name=spec.name,
            model_label=spec.label,
            target_name=spec.target,
            model_type=spec.model_type,
            deployment_slot=spec.deployment_slot,
            validation_split=spec.validation,
        ),
        source_root=source_root.resolve(),
        spec=spec,
    )


def _registered_spec_model(api, tmp_path: Path, **spec_overrides):
    source_root = tmp_path / "pricing_models" / "claim_frequency_spec"
    source_root.mkdir(parents=True)
    values = {
        "name": "CLAIM_FREQUENCY",
        "label": "Claim frequency",
        "target": "claim_count",
        "model_type": "superglm_poisson",
        "deployment_slot": "PRODUCTION",
        "features": ("age", "region"),
        "dataset_name": "claim_frequency_frame",
        "source_system": "pricing_sql",
        "pk_columns": ("policy_id",),
        "exposure_column": "exposure",
        "validation": ValidationSplitConfig.kfold(n_splits=2, random_state=7),
    }
    values.update(spec_overrides)
    spec = api.PricingModelSpec(**values)
    return api.RegisteredModel(
        model_id=17,
        config=ModelBuildConfig(
            model_name=spec.name,
            model_label=spec.label,
            target_name=spec.target,
            model_type=spec.model_type,
            deployment_slot=spec.deployment_slot,
            validation_split=spec.validation,
        ),
        source_root=source_root.resolve(),
        spec=spec,
    )


def _approved_build(tmp_path: Path, **overrides) -> CompletedModelBuild:
    values = {
        "model_id": 17,
        "model_name": "CLAIM_FREQUENCY",
        "model_version": "v7",
        "model_type": "superglm_poisson",
        "target_name": "claim_count",
        "deployment_slot": "PRODUCTION",
        "manifest_id": "manifest-1",
        "split_set_id": "split-1",
        "export_id": "claim-frequency__test",
        "rating_workbook_path": str(tmp_path / "rating.xlsx"),
        "rating_workbook_sha256": "a" * 64,
        "effective_from": None,
        "created_by": "analyst@example.test",
        "publication_receipt_path": str(tmp_path / "publication_receipt.json"),
        "publication_receipt_sha256": "b" * 64,
        "candidate_artifact_path": str(tmp_path / "candidate.joblib"),
        "candidate_artifact_sha256": "c" * 64,
        "candidate_artifact_format": "superglm-candidate-joblib-v1",
        "candidate_artifact_size_bytes": 123,
        "candidate_python_version": "3.14.4",
        "candidate_superglm_version": "0.11.0",
        "model_source_sha256": "d" * 64,
        "model_frame_sha256": "e" * 64,
    }
    values.update(overrides)
    return CompletedModelBuild(**values)


def test_pricing_model_spec_holds_analyst_decisions():
    from pricing_pipeline import notebook as api

    validation = ValidationSplitConfig.kfold(
        n_splits=2,
        random_state=7,
        materialize=True,
    )

    spec = api.PricingModelSpec(
        name="  CLAIM_FREQUENCY  ",
        label="  Claim frequency  ",
        target="  claim_count  ",
        model_type="  superglm_poisson  ",
        deployment_slot="  production  ",
        features=(" age ", "region"),
        dataset_name="  claim_frequency_frame  ",
        source_system="  pricing_sql  ",
        pk_columns=(" policy_id ",),
        exposure_column=" exposure ",
        validation=validation,
    )

    assert spec.name == "CLAIM_FREQUENCY"
    assert spec.label == "Claim frequency"
    assert spec.target == "claim_count"
    assert spec.model_type == "superglm_poisson"
    assert spec.deployment_slot == "PRODUCTION"
    assert spec.features == ("age", "region")
    assert spec.dataset_name == "claim_frequency_frame"
    assert spec.source_system == "pricing_sql"
    assert spec.pk_columns == ("policy_id",)
    assert spec.exposure_column == "exposure"
    assert spec.validation is validation
    assert spec.scoring == ("deviance",)
    assert spec.fit_mode == "fit_reml"


def test_pricing_model_spec_materializes_split_evidence_automatically():
    from pricing_pipeline import notebook as api

    spec = api.PricingModelSpec(
        name="CLAIM_FREQUENCY",
        label="Claim frequency",
        target="claim_count",
        model_type="superglm_poisson",
        deployment_slot="PRODUCTION",
        features=("age", "region"),
        dataset_name="claim_frequency_frame",
        source_system="pricing_sql",
        pk_columns=("policy_id",),
        validation=ValidationSplitConfig.kfold(n_splits=2),
    )

    assert spec.validation.materialize is True


def test_notebook_build_api_only_accepts_declared_model_inputs():
    from pricing_pipeline import notebook as api

    assert tuple(signature(api.register_model).parameters) == (
        "pricing",
        "spec",
        "source_root",
        "created_by",
    )
    assert tuple(signature(api.build_candidate).parameters) == (
        "pricing",
        "model",
        "frame",
        "model_factory",
        "data_as_of",
        "created_by",
    )
    assert tuple(signature(api.publish_edits).parameters) == (
        "pricing",
        "candidate",
        "reason",
        "created_by",
    )
    assert tuple(signature(api.publish_candidate).parameters) == (
        "pricing",
        "candidate",
    )
    assert tuple(signature(api.deploy_package).parameters) == (
        "pricing",
        "package",
        "reason",
        "deployed_by",
    )
    assert not hasattr(ValidationSplitConfig, "none")
    assert not hasattr(ValidationSplitConfig, "custom")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"features": ()}, "features must contain"),
        ({"features": ("age", "age")}, "features must not contain duplicates"),
        ({"pk_columns": ()}, "pk_columns must contain"),
        (
            {"pk_columns": ("policy_id", "policy_id")},
            "pk_columns must not contain duplicates",
        ),
        ({"features": ("claim_count",)}, "model column roles overlap"),
        ({"features": ("policy_id",)}, "model column roles overlap"),
        ({"features": ("exposure",)}, "model column roles overlap"),
    ],
)
def test_pricing_model_spec_rejects_ambiguous_column_roles(overrides, message):
    from pricing_pipeline import notebook as api

    values = {
        "name": "CLAIM_FREQUENCY",
        "label": "Claim frequency",
        "target": "claim_count",
        "model_type": "superglm_poisson",
        "deployment_slot": "PRODUCTION",
        "features": ("age", "region"),
        "dataset_name": "claim_frequency_frame",
        "source_system": "pricing_sql",
        "pk_columns": ("policy_id",),
        "exposure_column": "exposure",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        api.PricingModelSpec(**values)


@pytest.mark.parametrize(
    "validation",
    [
        ValidationSplitConfig(
            method="none",
            n_splits=None,
            random_state=None,
            shuffle=False,
        ),
        ValidationSplitConfig(
            method="custom",
            n_splits=None,
            random_state=None,
            shuffle=False,
            materialize=True,
        ),
    ],
)
def test_pricing_model_spec_rejects_validation_modes_the_notebook_cannot_build(
    validation,
):
    from pricing_pipeline import notebook as api

    with pytest.raises(ValueError, match="not supported by the notebook workflow"):
        api.PricingModelSpec(
            name="CLAIM_FREQUENCY",
            label="Claim frequency",
            target="claim_count",
            model_type="superglm_poisson",
            deployment_slot="PRODUCTION",
            features=("age", "region"),
            dataset_name="claim_frequency_frame",
            source_system="pricing_sql",
            pk_columns=("policy_id",),
            validation=validation,
        )


def test_register_model_accepts_python_spec(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    source_root = tmp_path / "pricing_models" / "claim_frequency"
    source_root.mkdir(parents=True)
    context = _context(api, tmp_path)
    connection = object()

    class Engine:
        @contextmanager
        def begin(self):
            yield connection

    context = api.NotebookContext(
        engine=Engine(),
        settings=context.settings,
        mode=context.mode,
        write_allowed=context.write_allowed,
        destination=context.destination,
    )
    validation = ValidationSplitConfig.kfold(n_splits=2, random_state=7)
    spec = api.PricingModelSpec(
        name="CLAIM_FREQUENCY",
        label="Claim frequency",
        target="claim_count",
        model_type="superglm_poisson",
        deployment_slot="PRODUCTION",
        features=("age", "region"),
        dataset_name="claim_frequency_frame",
        source_system="pricing_sql",
        pk_columns=("policy_id",),
        exposure_column="exposure",
        validation=validation,
    )
    captured = {}

    def register(con, config, *, created_by):
        captured["register"] = (con, config, created_by)
        return PricingModelRecord(
            model_id=41,
            model_name=config.model_name,
            model_label=config.model_label,
            target_name=config.target_name,
            model_type=config.model_type,
            model_status="ACTIVE",
        )

    monkeypatch.setattr(api, "register_pricing_model", register)

    model = api.register_model(
        context,
        spec,
        source_root=source_root,
        created_by="analyst@example.test",
    )

    assert model.spec is spec
    assert model.config == ModelBuildConfig(
        model_name="CLAIM_FREQUENCY",
        model_label="Claim frequency",
        target_name="claim_count",
        model_type="superglm_poisson",
        deployment_slot="PRODUCTION",
        validation_split=spec.validation,
    )
    assert model.source_root == source_root.resolve()
    assert captured["register"] == (
        connection,
        model.config,
        "analyst@example.test",
    )


@pytest.mark.parametrize(
    ("exposure_column", "published_factor_name"),
    [
        ("exposure", "exposure"),
        ("Earned Exposure", "Earned_Exposure"),
    ],
)
def test_build_candidate_derives_simple_spec_inputs(
    monkeypatch,
    tmp_path,
    exposure_column,
    published_factor_name,
):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_spec_model(
        api,
        tmp_path,
        exposure_column=exposure_column,
        data_as_of_column="snapshot_date",
    )
    frame = pd.DataFrame(
        {
            "policy_id": [10, 20, 30, 40],
            "claim_count": [0.0, 1.0, 0.0, 2.0],
            exposure_column: [1.0, 0.5, 1.5, 0.75],
            "age": [25.0, 45.0, 35.0, 52.0],
            "region": ["N", "S", "N", "S"],
            "snapshot_date": ["2026-06-30"] * 4,
        }
    )
    folds = [(np.array([0, 1]), np.array([2, 3]))]
    captured = {}

    monkeypatch.setattr(api, "validation_split_indices", lambda frame, split: folds)
    monkeypatch.setattr(
        api,
        "build_export_id",
        lambda model_name, run_key: f"{model_name}__{run_key}",
    )
    monkeypatch.setattr(
        api,
        "resolve_model_version_for_export",
        lambda engine, *, model_name, export_id: "v7",
    )

    standard_result = SimpleNamespace(
        completed_build=_approved_build(tmp_path),
        metrics={"cv_mean_deviance": 1.25},
    )

    def run_build(engine, **kwargs):
        captured["engine"] = engine
        captured.update(kwargs)
        return standard_result

    monkeypatch.setattr(api, "run_standard_superglm_build", run_build)

    candidate = api.build_candidate(
        context,
        model=model,
        frame=frame,
        model_factory=lambda: object(),
        created_by="analyst@example.test",
    )

    assert candidate.standard_build is standard_result
    inputs = captured["inputs"]
    assert list(inputs.X.columns) == ["age", "region"]
    assert inputs.y.name == "claim_count"
    assert np.allclose(inputs.offset.to_numpy(), np.log(frame[exposure_column]))
    assert inputs.export_weight.name == exposure_column
    assert np.allclose(inputs.export_weight.to_numpy(), frame[exposure_column])
    manifest_spec = captured["manifest_spec"]
    assert manifest_spec.dataset_name == "claim_frequency_frame"
    assert manifest_spec.source_system == "pricing_sql"
    assert manifest_spec.data_as_of_date.isoformat() == "2026-06-30"
    assert manifest_spec.pk_columns == ("policy_id",)
    assert manifest_spec.target_column == "claim_count"
    assert manifest_spec.weight_column is None
    assert manifest_spec.feature_columns == ("age", "region")
    assert manifest_spec.exposure_column == exposure_column
    assert manifest_spec.data_as_of_column == "snapshot_date"
    assert captured["effective_from"] is None
    assert captured["scoring"] == ("deviance",)
    assert captured["fit_mode"] == "fit_reml"
    contract = captured["offset_contract"]
    assert contract.handling == "EXPORTED_FACTOR"
    assert contract.source_factor_name == exposure_column
    assert contract.published_factor_name == published_factor_name
    assert contract.source_name == exposure_column
    assert "offset_export_options" not in captured


def test_publish_candidate_returns_generated_sql_ids(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    standard_build = SimpleNamespace(
        completed_build=_approved_build(
            tmp_path,
            manifest_id="manifest-9",
            split_set_id="split-9",
            export_id="claim-frequency__run-9",
        )
    )
    candidate = api.BuiltCandidate(model=model, standard_build=standard_build)
    expected = CompletedModelPublishResult(
        model_id=17,
        model_name="CLAIM_FREQUENCY",
        model_version="v7",
        manifest_id="manifest-9",
        split_set_id="split-9",
        export_id="claim-frequency__run-9",
        rate_package_id=71,
        package_version=4,
        package_status="PUBLISHED",
        rating_workbook_path=str(tmp_path / "rating.xlsx"),
        model_run_id=901,
    )
    captured = {}

    def publish(engine, **kwargs):
        captured["engine"] = engine
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(api, "publish_completed_model_build", publish)

    result = api.publish_candidate(context, candidate)

    assert result is expected
    assert result.model_id == 17
    assert result.model_run_id == 901
    assert result.rate_package_id == 71
    assert result.package_version == 4
    assert captured == {
        "engine": context.engine,
        "settings": context.settings,
        "model_config": model.config,
        "completed_build": standard_build.completed_build,
    }


def test_open_candidate_uses_registered_python_config(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    expected = object()
    captured = {}

    class FakeWorkbench:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def open(self, model_name, *, package_version):
            captured["open"] = (model_name, package_version)
            return expected

    monkeypatch.setattr(api, "Workbench", FakeWorkbench)

    result = api.open_candidate(
        context,
        model=model,
        package_version=4,
    )

    assert result is expected
    assert captured["engine"] is context.engine
    assert captured["settings"] is context.settings
    assert captured["model_config"] is model.config
    assert captured["open"] == ("CLAIM_FREQUENCY", 4)


def test_publish_edits_runs_editor_publisher_synchronously(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    session = object()
    candidate = SimpleNamespace(
        model_name=model.name,
        workbench=SimpleNamespace(model_config=model.config),
        editor_session=session,
        editor_widget=object(),
    )
    submission = SimpleNamespace(
        submission_id="submission-1",
        path=str(tmp_path / "submission.json"),
        sha256="a" * 64,
    )
    expected = SimpleNamespace(
        model_name=model.name,
        rate_package_id=72,
        package_version=5,
        model_run_id=902,
    )
    captured = {}

    def save(loaded_candidate, **kwargs):
        captured["candidate"] = loaded_candidate
        captured["save"] = kwargs
        return submission

    def publish(engine, **kwargs):
        captured["engine"] = engine
        captured["publish"] = kwargs
        return expected

    monkeypatch.setattr(api, "save_editor_submission", save)
    monkeypatch.setattr(api, "publish_editor_submission", publish)

    result = api.publish_edits(
        context,
        candidate=candidate,
        reason="Sparse age-band market adjustment",
        created_by="analyst@example.test",
    )

    assert result is expected
    assert captured["candidate"] is candidate
    assert captured["save"]["editor_session"] is session
    assert captured["save"]["reason"] == "Sparse age-band market adjustment"
    assert captured["save"]["claimed_identity"] == "analyst@example.test"
    assert captured["engine"] is context.engine
    assert captured["publish"] == {
        "settings": context.settings,
        "submission_path": submission.path,
        "submission_sha256": submission.sha256,
        "dag_id": "notebook_publish_editor_candidate",
        "airflow_run_id": "notebook__submission-1",
        "created_by": "analyst@example.test",
        "model_config": model.config,
    }


def test_publish_edits_requires_an_open_editor(tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    candidate = SimpleNamespace(
        model_name=model.name,
        workbench=SimpleNamespace(model_config=model.config),
        editor_session=None,
        editor_widget=None,
    )

    try:
        api.publish_edits(
            context,
            candidate=candidate,
            reason="Market adjustment",
        )
    except RuntimeError as exc:
        assert "Open the candidate editor" in str(exc)
    else:
        raise AssertionError("publish_edits accepted a candidate without an open editor")


def test_deploy_package_uses_the_champion_snapshot_seen_during_review(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    package = api.Candidate(
        workbench=SimpleNamespace(model_config=model.config),
        model_name=model.name,
        package_version=5,
        rate_package_id=72,
        parent_rate_package_id=None,
        model_run_id=902,
        bundle=object(),
        technical={"model_id": model.model_id, "current_rate_package_id": 61},
    )
    expected = SimpleNamespace(
        model_id=17,
        previous_rate_package_id=61,
        rate_package_id=72,
        package_version=5,
    )
    captured = {}

    def deploy(engine, config, **kwargs):
        captured["engine"] = engine
        captured["config"] = config
        captured["deploy"] = kwargs
        return expected

    monkeypatch.setattr(api, "deploy_rate_package", deploy)

    result = api.deploy_package(
        context,
        package=package,
        reason="Approved at August pricing meeting",
        deployed_by="pricing.manager@example.test",
    )

    assert result is expected
    assert captured["engine"] is context.engine
    assert captured["config"] is model.config
    assert captured["deploy"] == {
        "rate_package_id": 72,
        "expected_current_rate_package_id": 61,
        "deployment_reason": "Approved at August pricing meeting",
        "deployed_by": "pricing.manager@example.test",
        "model_id": 17,
    }


def test_deploy_package_rejects_a_package_that_was_not_opened_for_review(tmp_path):
    from pricing_pipeline import notebook as api

    with pytest.raises(TypeError, match="open_candidate"):
        api.deploy_package(
            _context(api, tmp_path),
            package=SimpleNamespace(rate_package_id=72),
            reason="Approved at August pricing meeting",
        )
