from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.build_identity import BuildIdentity, BuildIdentityError
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.config import ModelBuildConfig, ValidationSplitConfig
from pricing_pipeline.models.spec import ApprovedModelBuild
from pricing_pipeline.orchestration.publish_completed_build import (
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
        "offset_column": "term_offset",
        "offset_source_column": "term",
        "offset_label": "log(term / 12)",
        "sample_weight_column": "model_weight",
        "export_weight_column": "rating_weight",
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


def _approved_build(tmp_path: Path, **overrides) -> ApprovedModelBuild:
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
        "candidate_artifact_format": "superglm-candidate-joblib-v3",
        "candidate_artifact_size_bytes": 123,
        "candidate_python_version": "3.14.4",
        "candidate_superglm_version": "0.12.0",
        "candidate_superglm_git_sha": "f" * 40,
        "build_fingerprint_sha256": "1" * 64,
        "builder_source_sha256": "2" * 64,
        "materialized_split_sha256": "3" * 64,
        "runtime_sha256": "4" * 64,
        "candidate_superglm_sha256": "5" * 64,
        "row_order_sha256": "6" * 64,
        "model_source_sha256": "d" * 64,
        "model_frame_sha256": "e" * 64,
    }
    values.update(overrides)
    return ApprovedModelBuild(**values)


def _build_identity() -> BuildIdentity:
    return BuildIdentity(
        build_fingerprint_sha256="1" * 64,
        model_frame_sha256="2" * 64,
        row_order_sha256="3" * 64,
        model_source_sha256="4" * 64,
        builder_source_sha256="5" * 64,
        materialized_split_sha256="6" * 64,
        runtime_sha256="7" * 64,
        candidate_superglm_sha256="8" * 64,
        candidate_python_version="3.14.2",
        candidate_superglm_version="0.12.0",
        candidate_superglm_git_sha="9" * 40,
    )


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
        offset_column=" term_offset ",
        offset_source_column=" term ",
        offset_label=" log(term / 12) ",
        sample_weight_column=" model_weight ",
        export_weight_column=" rating_weight ",
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
    assert spec.offset_column == "term_offset"
    assert spec.offset_source_column == "term"
    assert spec.offset_label == "log(term / 12)"
    assert spec.sample_weight_column == "model_weight"
    assert spec.export_weight_column == "rating_weight"
    assert spec.validation is validation
    assert spec.scoring == ("deviance", "nll", "gini")
    assert spec.fit_mode == "fit_reml"


@pytest.mark.parametrize(
    "scoring",
    [
        (lambda y, mu: 0.0,),
        (partial(pow, 2),),
        ("value".upper,),
        (type("Scorer", (), {"__call__": lambda self, y, mu: 0.0})(),),
        "deviance",
    ],
)
def test_pricing_model_spec_rejects_unnamed_or_callable_scoring(scoring):
    from pricing_pipeline import notebook as api

    with pytest.raises(ValueError, match="scoring must contain named metric strings"):
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
            scoring=scoring,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"offset_source_column": None},
        {"offset_label": None},
        {"offset_column": None},
    ],
)
def test_pricing_model_spec_requires_complete_offset_contract(override):
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
        "offset_column": "term_offset",
        "offset_source_column": "term",
        "offset_label": "log(term / 12)",
    }
    values.update(override)

    with pytest.raises(ValueError, match="offset_column, offset_source_column, and offset_label"):
        api.PricingModelSpec(**values)


def test_pricing_model_spec_rejects_unsupported_fit_mode_before_reservation(
    monkeypatch,
):
    from pricing_pipeline import notebook as api

    reservation_calls = []
    monkeypatch.setattr(
        api,
        "resolve_model_version_for_export",
        lambda *args, **kwargs: reservation_calls.append((args, kwargs)),
    )
    values = {
        "name": "CLAIM_FREQUENCY",
        "label": "Claim frequency",
        "target": "claim_count",
        "model_type": "superglm_poisson",
        "deployment_slot": "PRODUCTION",
        "features": ("age",),
        "dataset_name": "claim_frequency_frame",
        "source_system": "pricing_sql",
        "pk_columns": ("policy_id",),
        "fit_mode": "predict",
    }

    with pytest.raises(ValueError, match="fit_mode.*fit_reml"):
        api.PricingModelSpec(**values)

    assert reservation_calls == []


def test_pricing_model_spec_allows_one_operational_column_for_multiple_roles():
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
        offset_column="weight",
        offset_source_column="weight",
        offset_label="identity(weight)",
        sample_weight_column="weight",
        export_weight_column="weight",
    )

    assert (
        spec.offset_column
        == spec.offset_source_column
        == spec.sample_weight_column
        == spec.export_weight_column
    )


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
    build_parameters = signature(api.build_candidate).parameters
    assert tuple(build_parameters) == (
        "pricing",
        "model",
        "frame",
        "superglm_model",
        "data_as_of",
        "created_by",
    )
    assert "superglm_model" in build_parameters
    assert "model_factory" not in build_parameters
    assert tuple(signature(api.publish_edits).parameters) == (
        "pricing",
        "candidate",
        "editor_session",
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
        ({"features": ("term",)}, "model column roles overlap"),
        ({"data_as_of_column": "term"}, "model column roles overlap"),
        (
            {"validation": ValidationSplitConfig.column_kfold(column="claim_count")},
            "model column roles overlap",
        ),
        (
            {
                "validation": ValidationSplitConfig.column_holdout(
                    column="region",
                    train_values=("A",),
                    test_values=("B",),
                )
            },
            "model column roles overlap",
        ),
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
        "offset_column": "term_offset",
        "offset_source_column": "term",
        "offset_label": "log(term / 12)",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        api.PricingModelSpec(**values)


@pytest.mark.parametrize("stratify_column", ["claim_count", "region"])
def test_pricing_model_spec_allows_stratifying_by_target_or_feature(stratify_column):
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
        validation=ValidationSplitConfig.train_test_split(
            stratify_column=stratify_column,
        ),
    )

    assert spec.validation.stratify_column == stratify_column


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


def test_build_candidate_rejects_invalid_model_identity_before_version_or_artifact_work(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline import notebook as api

    context = replace(
        _context(api, tmp_path),
        mode="local",
        destination="local SQLite database",
    )
    model = _registered_model(api, tmp_path)
    frame = pd.DataFrame(
        {
            "policy_id": [10, 20],
            "claim_count": [0.0, 1.0],
            "exposure": [1.0, 1.0],
            "age": [25.0, 45.0],
            "region": ["N", "S"],
        }
    )

    monkeypatch.setattr(
        api,
        "resolve_sqlite_model_version",
        lambda *args, **kwargs: pytest.fail("model version was reserved"),
    )
    monkeypatch.setattr(
        api,
        "run_standard_superglm_build",
        lambda *args, **kwargs: pytest.fail("candidate artifacts were built"),
    )

    with pytest.raises(
        BuildIdentityError,
        match="SuperGLM identity is invalid",
    ):
        api.build_candidate(
            context,
            model=model,
            frame=frame,
            superglm_model=SimpleNamespace(_result=object()),
            data_as_of="2026-06-30",
        )

    assert not context.settings.workbench_artifact_root.exists()
    assert not context.settings.validation_split_artifact_root.exists()


@pytest.mark.parametrize(
    ("stratify_column", "test_size", "match"),
    [
        (
            "validation_cohort",
            0.25,
            "model frame is missing declared columns: validation_cohort",
        ),
        ("policy_id", 0.5, "least populated class"),
    ],
)
def test_build_candidate_validates_stratifier_before_reserving_model_version(
    monkeypatch,
    tmp_path,
    stratify_column,
    test_size,
    match,
):
    from pricing_pipeline import notebook as api

    context = replace(
        _context(api, tmp_path),
        mode="local",
        destination="local SQLite database",
    )
    validation = ValidationSplitConfig.train_test_split(
        test_size=test_size,
        random_state=7,
        stratify_column=stratify_column,
    )
    model = _registered_model(api, tmp_path)
    model = replace(
        model,
        config=replace(model.config, validation_split=validation),
        spec=replace(model.spec, validation=validation),
    )
    frame = pd.DataFrame(
        {
            "policy_id": [10, 20, 30, 40],
            "claim_count": [0.0, 1.0, 0.0, 2.0],
            "age": [25.0, 45.0, 35.0, 52.0],
            "region": ["N", "S", "N", "S"],
        }
    )

    monkeypatch.setattr(
        api,
        "resolve_sqlite_model_version",
        lambda *args, **kwargs: pytest.fail("model version was reserved"),
    )

    with pytest.raises(ValueError, match=match):
        api.build_candidate(
            context,
            model=model,
            frame=frame,
            superglm_model=object(),
            data_as_of="2026-06-30",
        )


def test_build_candidate_keeps_offset_source_and_weights_independent(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_spec_model(
        api,
        tmp_path,
        data_as_of_column="snapshot_date",
    )
    frame = pd.DataFrame(
        {
            "policy_id": [10, 20, 30, 40],
            "claim_count": [0.0, 1.0, 0.0, 2.0],
            "term": [12.0, 36.0, 12.0, 36.0],
            "term_offset": np.log([1.0, 3.0, 1.0, 3.0]),
            "model_weight": [0.5, 0.75, 1.25, 1.5],
            "rating_weight": [10.0, 20.0, 30.0, 40.0],
            "age": [25.0, 45.0, 35.0, 52.0],
            "region": ["N", "S", "N", "S"],
            "snapshot_date": ["2026-06-30"] * 4,
        }
    )
    folds = [(np.array([0, 1]), np.array([2, 3]))]
    captured = {}

    monkeypatch.setattr(api, "validation_split_indices", lambda frame, split: folds)
    monkeypatch.setattr(api, "create_build_identity", lambda **kwargs: _build_identity())
    monkeypatch.setattr(
        api,
        "resolve_model_version_for_export",
        lambda engine, *, model_name, export_id, build_fingerprint_sha256: "v7",
    )

    completed_build = _approved_build(
        tmp_path,
        metrics={"cv_mean_deviance": 1.25},
    )

    def run_build(engine, **kwargs):
        captured["engine"] = engine
        captured.update(kwargs)
        return completed_build

    monkeypatch.setattr(api, "run_standard_superglm_build", run_build)

    superglm_model = object()
    candidate = api.build_candidate(
        context,
        model=model,
        frame=frame,
        superglm_model=superglm_model,
        created_by="analyst@example.test",
    )

    assert candidate.completed_build is completed_build
    assert candidate.metrics == {"cv_mean_deviance": 1.25}
    assert candidate.metrics is not candidate.completed_build.metrics
    inputs = captured["inputs"]
    assert list(inputs.X.columns) == ["age", "region"]
    assert inputs.y.name == "claim_count"
    pd.testing.assert_series_equal(inputs.offset, frame.set_index("policy_id")["term_offset"])
    pd.testing.assert_series_equal(inputs.offset_source, frame.set_index("policy_id")["term"])
    pd.testing.assert_series_equal(
        inputs.sample_weight, frame.set_index("policy_id")["model_weight"]
    )
    pd.testing.assert_series_equal(
        inputs.export_weight, frame.set_index("policy_id")["rating_weight"]
    )
    assert inputs.offset_source_name == "term"
    assert inputs.sample_weight_name == "model_weight"
    assert inputs.export_weight_name == "rating_weight"
    manifest_spec = captured["manifest_spec"]
    assert manifest_spec.dataset_name == "claim_frequency_frame"
    assert manifest_spec.source_system == "pricing_sql"
    assert manifest_spec.data_as_of_date.isoformat() == "2026-06-30"
    assert manifest_spec.pk_columns == ("policy_id",)
    assert manifest_spec.target_column == "claim_count"
    assert manifest_spec.weight_column == "model_weight"
    assert manifest_spec.feature_columns == ("age", "region")
    assert manifest_spec.offset_column == "term_offset"
    assert manifest_spec.offset_source_column == "term"
    assert manifest_spec.offset_label == "log(term / 12)"
    assert manifest_spec.export_weight_column == "rating_weight"
    assert manifest_spec.data_as_of_column == "snapshot_date"
    assert captured["effective_from"] is None
    assert captured["model_config"] is model.config
    assert captured["superglm_model"] is superglm_model
    assert "model_name" not in captured
    assert "model_type" not in captured
    assert "target_name" not in captured
    assert "deployment_slot" not in captured
    assert "validation_split" not in captured
    assert captured["scoring"] == ("deviance", "nll", "gini")
    assert captured["fit_mode"] == "fit_reml"
    assert captured["expected_build_identity"] == _build_identity()
    assert captured["export_id"] == "build_" + "1" * 64
    assert Path(captured["output_dir"]).name.startswith("attempt_")
    assert "1" * 64 not in str(captured["output_dir"])
    contract = captured["offset_contract"]
    assert contract.handling == "EXPORTED_FACTOR"
    assert contract.source_factor_name == "term"
    assert contract.published_factor_name == "term"
    assert contract.source_name == "term"
    assert contract.label == "log(term / 12)"
    assert "offset_export_options" not in captured


def test_build_candidate_computes_identity_before_reservation_and_separates_attempts(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    frame = pd.DataFrame(
        {
            "policy_id": [10, 20, 30, 40],
            "claim_count": [0.0, 1.0, 0.0, 2.0],
            "age": [25.0, 45.0, 35.0, 52.0],
            "region": ["N", "S", "N", "S"],
        }
    )
    events = []
    identity_calls = []
    reservations = []
    attempts = []

    monkeypatch.setattr(
        api,
        "validation_split_indices",
        lambda frame, split: [
            (np.array([0, 1]), np.array([2, 3])),
            (np.array([2, 3]), np.array([0, 1])),
        ],
    )

    def build_identity(**kwargs):
        events.append("identity")
        identity_calls.append(kwargs)
        return _build_identity()

    def reserve(engine, *, model_name, export_id, build_fingerprint_sha256):
        del engine
        events.append("reserve")
        reservations.append((model_name, export_id, build_fingerprint_sha256))
        return "v7"

    def run_build(engine, **kwargs):
        del engine
        events.append("run")
        attempts.append(Path(kwargs["output_dir"]))
        return _approved_build(tmp_path)

    monkeypatch.setattr(api, "create_build_identity", build_identity)
    monkeypatch.setattr(api, "resolve_model_version_for_export", reserve)
    monkeypatch.setattr(api, "run_standard_superglm_build", run_build)

    superglm_model = object()
    for _ in range(2):
        api.build_candidate(
            context,
            model=model,
            frame=frame,
            superglm_model=superglm_model,
            data_as_of="2026-06-30",
        )

    assert events == ["identity", "reserve", "run"] * 2
    assert reservations == [
        ("CLAIM_FREQUENCY", "build_" + "1" * 64, "1" * 64),
        ("CLAIM_FREQUENCY", "build_" + "1" * 64, "1" * 64),
    ]
    assert attempts[0] != attempts[1]
    assert all(path.name.startswith("attempt_") for path in attempts)
    assert all("1" * 64 not in str(path) for path in attempts)
    assert identity_calls[0]["frame"] is frame
    assert identity_calls[0]["model_config"] is model.config
    assert identity_calls[0]["superglm_model"] is identity_calls[1]["superglm_model"]
    assert identity_calls[0]["scoring"] == ("deviance", "nll", "gini")


@pytest.mark.parametrize("analyst_name", ["../escaped", "x" * 500])
def test_build_candidate_uses_bounded_model_id_artifact_paths(
    monkeypatch,
    tmp_path,
    analyst_name,
):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    model = replace(
        model,
        config=replace(model.config, model_name=analyst_name),
        spec=replace(model.spec, name=analyst_name),
    )
    frame = pd.DataFrame(
        {
            "policy_id": [10, 20, 30, 40],
            "claim_count": [0.0, 1.0, 0.0, 2.0],
            "age": [25.0, 45.0, 35.0, 52.0],
            "region": ["N", "S", "N", "S"],
        }
    )
    captured = {}
    monkeypatch.setattr(
        api,
        "validation_split_indices",
        lambda frame, split: [
            (np.array([0, 1]), np.array([2, 3])),
            (np.array([2, 3]), np.array([0, 1])),
        ],
    )
    monkeypatch.setattr(api, "create_build_identity", lambda **kwargs: _build_identity())
    monkeypatch.setattr(
        api,
        "resolve_model_version_for_export",
        lambda *args, **kwargs: "v7",
    )

    def run_build(engine, **kwargs):
        del engine
        captured.update(kwargs)
        return _approved_build(tmp_path)

    monkeypatch.setattr(api, "run_standard_superglm_build", run_build)

    api.build_candidate(
        context,
        model=model,
        frame=frame,
        superglm_model=object(),
        data_as_of="2026-06-30",
    )

    root = Path(context.settings.workbench_artifact_root).resolve()
    output_dir = Path(captured["output_dir"]).resolve()
    assert output_dir.is_relative_to(root)
    assert output_dir.relative_to(root).parts[0] == "model_17"
    assert analyst_name not in str(output_dir)
    assert max(map(len, output_dir.relative_to(root).parts)) < 80


def test_build_candidate_identity_failure_happens_before_version_reservation(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    frame = pd.DataFrame(
        {
            "policy_id": [10, 20, 30, 40],
            "claim_count": [0.0, 1.0, 0.0, 2.0],
            "age": [25.0, 45.0, 35.0, 52.0],
            "region": ["N", "S", "N", "S"],
        }
    )
    monkeypatch.setattr(
        api,
        "create_build_identity",
        lambda **kwargs: (_ for _ in ()).throw(BuildIdentityError("untrusted model")),
    )
    monkeypatch.setattr(
        api,
        "resolve_model_version_for_export",
        lambda *args, **kwargs: pytest.fail("version was reserved"),
    )

    with pytest.raises(BuildIdentityError, match="untrusted model"):
        api.build_candidate(
            context,
            model=model,
            frame=frame,
            superglm_model=object(),
            data_as_of="2026-06-30",
        )


def test_built_candidate_returns_fresh_wide_validation_metrics(tmp_path):
    from pricing_pipeline import notebook as api

    model = _registered_model(api, tmp_path)
    completed_build = _approved_build(
        tmp_path,
        fold_metrics=(
            {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},
            {"fold_no": 1, "metric_name": "nll", "metric_value": 0.2},
            {"fold_no": 1, "metric_name": "gini", "metric_value": 0.7},
            {"fold_no": 2, "metric_name": "deviance", "metric_value": 0.5},
            {"fold_no": 2, "metric_name": "nll", "metric_value": 0.3},
            {"fold_no": 2, "metric_name": "gini", "metric_value": 0.8},
        ),
        validation_splits=(
            {
                "validation_split_no": 1,
                "n_train": 80,
                "n_validation": 20,
                "metrics": {"deviance": 0.4, "nll": 0.2, "gini": 0.7},
            },
            {
                "validation_split_no": 2,
                "n_train": 80,
                "n_validation": 20,
                "metrics": {"deviance": 0.5, "nll": 0.3, "gini": 0.8},
            },
        ),
    )
    candidate = api.BuiltCandidate(model=model, completed_build=completed_build)

    validation_metrics = candidate.validation_metrics

    pd.testing.assert_frame_equal(
        validation_metrics,
        pd.DataFrame(
            {
                "validation_split_no": [1, 2],
                "n_train": [80, 80],
                "n_validation": [20, 20],
                "deviance": [0.4, 0.5],
                "nll": [0.2, 0.3],
                "gini": [0.7, 0.8],
            }
        ),
    )
    validation_metrics.loc[0, "deviance"] = 99.0
    assert candidate.validation_metrics.loc[0, "deviance"] == pytest.approx(0.4)
    assert candidate.validation_metrics is not validation_metrics


def test_built_candidate_returns_empty_validation_metrics_for_legacy_record(tmp_path):
    from pricing_pipeline import notebook as api

    candidate = api.BuiltCandidate(
        model=_registered_model(api, tmp_path),
        completed_build=_approved_build(tmp_path),
    )

    assert candidate.validation_metrics.empty
    assert list(candidate.validation_metrics) == [
        "validation_split_no",
        "n_train",
        "n_validation",
    ]


def test_build_candidate_aligns_composite_primary_key_inputs(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline import notebook as api
    from pricing_pipeline.modeling.standard_superglm import _validate_canonical_row_ids

    context = _context(api, tmp_path)
    model = _registered_spec_model(
        api,
        tmp_path,
        pk_columns=("policy_id", "risk_id"),
        sample_weight_column="credibility",
        data_as_of_column="snapshot_date",
    )
    frame = pd.DataFrame(
        {
            "policy_id": [10, 10, 20, 20],
            "risk_id": [1, 2, 1, 2],
            "claim_count": [0.0, 1.0, 0.0, 2.0],
            "term": [12.0, 36.0, 12.0, 36.0],
            "term_offset": np.log([1.0, 3.0, 1.0, 3.0]),
            "rating_weight": [1.0, 0.5, 1.5, 0.75],
            "credibility": [0.8, 0.9, 1.0, 0.7],
            "age": [25.0, 45.0, 35.0, 52.0],
            "region": ["N", "S", "N", "S"],
            "snapshot_date": ["2026-06-30"] * 4,
        },
        index=[8, 3, 5, 1],
    )
    captured = {}

    monkeypatch.setattr(
        api,
        "validation_split_indices",
        lambda frame, split: [(np.array([0, 1]), np.array([2, 3]))],
    )
    monkeypatch.setattr(api, "create_build_identity", lambda **kwargs: _build_identity())
    monkeypatch.setattr(
        api,
        "resolve_model_version_for_export",
        lambda engine, *, model_name, export_id, build_fingerprint_sha256: "v7",
    )

    def run_build(engine, **kwargs):
        del engine
        captured.update(kwargs)
        _validate_canonical_row_ids(
            kwargs["frame"],
            kwargs["inputs"],
            pk_columns=("policy_id", "risk_id"),
        )
        return _approved_build(tmp_path)

    monkeypatch.setattr(api, "run_standard_superglm_build", run_build)

    superglm_model = object()
    api.build_candidate(
        context,
        model=model,
        frame=frame,
        superglm_model=superglm_model,
    )

    assert captured["frame"] is frame
    assert captured["superglm_model"] is superglm_model
    assert captured["inputs"].row_ids.equals(frame[["policy_id", "risk_id"]])
    expected_identity = pd.MultiIndex.from_frame(
        frame[["policy_id", "risk_id"]],
        names=["policy_id", "risk_id"],
    )
    for values in (
        captured["inputs"].X,
        captured["inputs"].y,
        captured["inputs"].sample_weight,
        captured["inputs"].offset,
        captured["inputs"].offset_source,
        captured["inputs"].export_weight,
    ):
        assert values.index.identical(expected_identity)


def test_publish_candidate_returns_generated_sql_ids(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    completed_build = _approved_build(
        tmp_path,
        manifest_id="manifest-9",
        split_set_id="split-9",
        export_id="claim-frequency__run-9",
    )
    candidate = api.BuiltCandidate(model=model, completed_build=completed_build)
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
        "completed_build": completed_build,
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
        workbench=SimpleNamespace(engine=context.engine, model_config=model.config),
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
        editor_session=session,
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


def test_publish_edits_rejects_candidate_opened_with_different_context(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    reviewed_context = _context(api, tmp_path)
    publishing_context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    candidate = SimpleNamespace(
        model_name=model.name,
        workbench=SimpleNamespace(
            engine=reviewed_context.engine,
            model_config=model.config,
        ),
    )

    def unexpected_call(*args, **kwargs):
        pytest.fail("cross-context publish reached save or publish")

    monkeypatch.setattr(api, "save_editor_submission", unexpected_call)
    monkeypatch.setattr(api, "publish_editor_submission", unexpected_call)

    with pytest.raises(ValueError, match="different notebook context"):
        api.publish_edits(
            publishing_context,
            candidate=candidate,
            editor_session=object(),
            reason="Sparse age-band market adjustment",
            created_by="analyst@example.test",
        )


def test_publish_edits_requires_an_explicit_editor_session(tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    candidate = SimpleNamespace(
        model_name=model.name,
        workbench=SimpleNamespace(engine=context.engine, model_config=model.config),
    )

    with pytest.raises(TypeError, match="editor_session"):
        api.publish_edits(
            context,
            candidate=candidate,
            reason="Market adjustment",
        )


def test_deploy_package_uses_the_champion_snapshot_seen_during_review(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    package = api.Candidate(
        workbench=SimpleNamespace(engine=context.engine, model_config=model.config),
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


def test_deploy_package_rejects_package_opened_with_different_context(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    reviewed_context = _context(api, tmp_path)
    deployment_context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    package = api.Candidate(
        workbench=SimpleNamespace(
            engine=reviewed_context.engine,
            model_config=model.config,
        ),
        model_name=model.name,
        package_version=5,
        rate_package_id=72,
        parent_rate_package_id=None,
        model_run_id=902,
        bundle=object(),
        technical={"model_id": model.model_id, "current_rate_package_id": 61},
    )

    monkeypatch.setattr(
        api,
        "deploy_rate_package",
        lambda *args, **kwargs: pytest.fail("cross-context deployment reached deploy_rate_package"),
    )

    with pytest.raises(ValueError, match="different notebook context"):
        api.deploy_package(
            deployment_context,
            package=package,
            reason="Approved at August pricing meeting",
            deployed_by="pricing.manager@example.test",
        )


def test_deploy_package_rejects_a_package_that_was_not_opened_for_review(tmp_path):
    from pricing_pipeline import notebook as api

    with pytest.raises(TypeError, match="open_candidate"):
        api.deploy_package(
            _context(api, tmp_path),
            package=SimpleNamespace(rate_package_id=72),
            reason="Approved at August pricing meeting",
        )
