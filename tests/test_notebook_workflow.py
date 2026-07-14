from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.config import ModelBuildConfig, ValidationSplitConfig
from pricing_pipeline.orchestration.publish_completed_build import CompletedModelPublishResult
from pricing_pipeline.publishing.model_registry import PricingModelRecord


def _context(api, tmp_path: Path):
    return api.NotebookContext(
        engine=object(),
        settings=Settings(
            workbench_artifact_root=tmp_path / "workbench",
            validation_split_artifact_root=tmp_path / "splits",
        ),
    )


def _registered_model(api, tmp_path: Path):
    source_root = tmp_path / "pricing_models" / "claim_frequency"
    source_root.mkdir(parents=True)
    (source_root / "model.py").write_text("MODEL = 'claim_frequency'\n", encoding="utf-8")
    return api.RegisteredModel(
        model_id=17,
        config=ModelBuildConfig(
            model_name="CLAIM_FREQUENCY",
            model_label="Claim frequency",
            target_name="claim_count",
            model_type="Poisson",
            deployment_slot="PRODUCTION",
            validation_split=ValidationSplitConfig.kfold(n_splits=2, random_state=7),
        ),
        source_root=source_root.resolve(),
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


def test_connect_uses_runtime_module_without_airflow(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    engine = object()
    settings = Settings(workbench_artifact_root=tmp_path / "artifacts")
    runtime = SimpleNamespace(settings=settings, get_engine=lambda: engine)
    calls = []
    monkeypatch.setattr(
        api,
        "runtime_from_env_or_module",
        lambda runtime_module=None: calls.append(runtime_module) or runtime,
    )

    result = api.connect("work_runtime.database")

    assert result.engine is engine
    assert result.settings is settings
    assert calls == ["work_runtime.database"]


def test_register_model_creates_then_validates_sql_identity(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    source_root = tmp_path / "pricing_models" / "home_frequency"
    source_root.mkdir(parents=True)
    (source_root / "pricing_model.ipynb").write_text("{}", encoding="utf-8")
    connection = object()

    class Engine:
        @contextmanager
        def begin(self):
            yield connection

    context = api.NotebookContext(engine=Engine(), settings=context.settings)
    calls = []

    def register(con, config, *, created_by):
        calls.append(("register", con, config, created_by))
        return 41

    def validate(con, config):
        calls.append(("validate", con, config))
        return PricingModelRecord(
            model_id=41,
            model_name=config.model_name,
            model_label=config.model_label,
            target_name=config.target_name,
            model_type=config.model_type,
            model_status="ACTIVE",
        )

    monkeypatch.setattr(api, "register_pricing_model", register)
    monkeypatch.setattr(api, "validate_registered_model", validate)
    split = ValidationSplitConfig.kfold(n_splits=4, random_state=13)

    model = api.register_model(
        context,
        name="HOME_FREQUENCY",
        label="Home frequency",
        target="claim_count",
        model_type="Poisson",
        deployment_slot="production",
        validation_split=split,
        source_root=source_root,
        created_by="analyst@example.test",
    )

    assert model.model_id == 41
    assert model.config == ModelBuildConfig(
        model_name="HOME_FREQUENCY",
        model_label="Home frequency",
        target_name="claim_count",
        model_type="Poisson",
        deployment_slot="PRODUCTION",
        validation_split=split,
    )
    assert model.source_root == source_root.resolve()
    assert [call[0] for call in calls] == ["register", "validate"]
    assert calls[0][3] == "analyst@example.test"


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

    context = api.NotebookContext(engine=Engine(), settings=context.settings)
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
        return 41

    def validate(con, config):
        captured["validate"] = (con, config)
        return PricingModelRecord(
            model_id=41,
            model_name=config.model_name,
            model_label=config.model_label,
            target_name=config.target_name,
            model_type=config.model_type,
            model_status="ACTIVE",
        )

    monkeypatch.setattr(api, "register_pricing_model", register)
    monkeypatch.setattr(api, "validate_registered_model", validate)

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
        validation_split=validation,
    )
    assert model.source_root == source_root.resolve()
    assert captured["register"] == (
        connection,
        model.config,
        "analyst@example.test",
    )


def test_build_candidate_derives_audit_plumbing(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    frame = pd.DataFrame(
        {
            "policy_id": [10, 20, 30, 40],
            "claim_count": [0.0, 1.0, 0.0, 2.0],
            "exposure": [1.0, 0.5, 1.5, 0.75],
            "age": [25.0, 45.0, 35.0, 52.0],
        }
    )
    X = frame[["age"]].copy()
    y = frame["claim_count"]
    weight = frame["exposure"]
    offset = np.log(weight.to_numpy())
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
        completed_build={"manifest_id": "manifest-1", "split_set_id": "split-1"},
        metrics={"cv_mean_deviance": 1.25},
    )

    def run_build(engine, **kwargs):
        captured["engine"] = engine
        captured.update(kwargs)
        return standard_result

    monkeypatch.setattr(api, "run_standard_superglm_build", run_build)
    offset_contract = api.OffsetExportContract(
        handling="EXPORTED_FACTOR",
        source_factor_name="exposure",
        published_factor_name="exposure",
        source_name="exposure",
        label="log(exposure)",
    )
    offset_options = {
        "offset_source": weight,
        "offset_name": "exposure",
        "offset_kind": "discrete",
    }

    candidate = api.build_candidate(
        context,
        model=model,
        frame=frame,
        X=X,
        y=y,
        model_factory=lambda: object(),
        scoring=("deviance",),
        dataset_name="home_model_frame",
        source_system="pricing_sql",
        data_as_of="2026-06-30",
        pk_columns=("policy_id",),
        effective_from="2026-08-01",
        sample_weight=weight,
        weight_column="exposure",
        offset=offset,
        offset_contract=offset_contract,
        offset_export_options=offset_options,
        run_key="notebook-run-1",
        created_by="analyst@example.test",
    )

    assert candidate.model is model
    assert candidate.standard_build is standard_result
    assert captured["engine"] is context.engine
    assert captured["model_name"] == "CLAIM_FREQUENCY"
    assert captured["model_version"] == "v7"
    assert captured["export_id"] == "CLAIM_FREQUENCY__notebook-run-1"
    assert captured["effective_from"] == "2026-08-01"
    assert captured["split_indices"] is folds
    assert captured["validation_split"] == model.config.validation_split
    assert captured["manifest_spec"].dataset_name == "home_model_frame"
    assert captured["manifest_spec"].source_system == "pricing_sql"
    assert captured["manifest_spec"].pk_columns == ("policy_id",)
    assert captured["manifest_spec"].target_column == "claim_count"
    assert captured["manifest_spec"].weight_column == "exposure"
    identity_index = pd.Index(
        frame["policy_id"].to_numpy(copy=True),
        name="policy_id",
    )
    assert captured["inputs"].X.index.identical(identity_index)
    assert captured["inputs"].y.index.identical(identity_index)
    assert captured["inputs"].sample_weight.index.identical(identity_index)
    assert captured["inputs"].offset.index.identical(identity_index)
    assert np.array_equal(captured["inputs"].offset.to_numpy(), offset)
    assert captured["inputs"].row_ids.equals(frame[["policy_id"]])
    assert captured["output_dir"] == (
        context.settings.workbench_artifact_root
        / "CLAIM_FREQUENCY"
        / "notebook-run-1"
    )
    assert captured["model_source_root"] == model.source_root
    assert captured["split_artifact_root"] == context.settings.validation_split_artifact_root
    assert captured["created_by"] == "analyst@example.test"
    assert captured["offset_contract"] is offset_contract
    assert captured["offset_export_options"] is offset_options


def test_build_candidate_derives_simple_spec_inputs(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_spec_model(api, tmp_path)
    frame = pd.DataFrame(
        {
            "policy_id": [10, 20, 30, 40],
            "claim_count": [0.0, 1.0, 0.0, 2.0],
            "exposure": [1.0, 0.5, 1.5, 0.75],
            "age": [25.0, 45.0, 35.0, 52.0],
            "region": ["N", "S", "N", "S"],
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
        completed_build={"manifest_id": "manifest-1", "split_set_id": "split-1"},
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
        data_as_of="2026-06-30",
        run_key="notebook-run-1",
        created_by="analyst@example.test",
    )

    assert candidate.standard_build is standard_result
    inputs = captured["inputs"]
    assert list(inputs.X.columns) == ["age", "region"]
    assert inputs.y.name == "claim_count"
    assert np.allclose(inputs.offset.to_numpy(), np.log(frame["exposure"]))
    assert inputs.export_weight.name == "exposure"
    assert np.allclose(inputs.export_weight.to_numpy(), frame["exposure"])
    manifest_spec = captured["manifest_spec"]
    assert manifest_spec.dataset_name == "claim_frequency_frame"
    assert manifest_spec.source_system == "pricing_sql"
    assert manifest_spec.data_as_of_date.isoformat() == "2026-06-30"
    assert manifest_spec.pk_columns == ("policy_id",)
    assert manifest_spec.target_column == "claim_count"
    assert manifest_spec.weight_column == "exposure"
    assert captured["scoring"] == ("deviance",)
    assert captured["fit_mode"] == "fit_reml"
    assert captured["offset_contract"].handling == "EXPORTED_FACTOR"
    export_options = captured["offset_export_options"]
    assert export_options["offset_name"] == "exposure"
    assert export_options["offset_kind"] == "auto"
    assert export_options["offset_source"].equals(inputs.export_weight)


def test_build_candidate_requires_metadata_for_caller_supplied_spec_offset(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_spec_model(api, tmp_path)
    frame = pd.DataFrame(
        {
            "policy_id": [10, 20, 30, 40],
            "claim_count": [0.0, 1.0, 0.0, 2.0],
            "exposure": [1.0, 0.5, 1.5, 0.75],
            "age": [25.0, 45.0, 35.0, 52.0],
            "region": ["N", "S", "N", "S"],
        }
    )
    monkeypatch.setattr(
        api,
        "validation_split_indices",
        lambda frame, split: [(np.array([0, 1]), np.array([2, 3]))],
    )
    monkeypatch.setattr(api, "build_export_id", lambda *args: "export-1")
    monkeypatch.setattr(
        api,
        "resolve_model_version_for_export",
        lambda *args, **kwargs: "v1",
    )
    monkeypatch.setattr(
        api,
        "run_standard_superglm_build",
        lambda *args, **kwargs: SimpleNamespace(completed_build={}, metrics={}),
    )

    with pytest.raises(
        ValueError,
        match="caller-supplied offset.*offset_contract.*offset_export_options",
    ):
        api.build_candidate(
            context,
            model=model,
            frame=frame,
            model_factory=lambda: object(),
            offset=np.array([0.0, 0.2, -0.1, 0.4]),
            data_as_of="2026-06-30",
            run_key="custom-offset",
            created_by="analyst@example.test",
        )


def test_build_candidate_validates_spec_less_exported_offset_options(
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
        }
    )
    monkeypatch.setattr(api, "build_export_id", lambda *args: "export-1")
    monkeypatch.setattr(
        api,
        "resolve_model_version_for_export",
        lambda *args, **kwargs: "v1",
    )
    monkeypatch.setattr(
        api,
        "run_standard_superglm_build",
        lambda *args, **kwargs: SimpleNamespace(completed_build={}, metrics={}),
    )
    contract = api.OffsetExportContract(
        handling="EXPORTED_FACTOR",
        source_factor_name="exposure",
        published_factor_name="exposure",
        source_name="exposure",
        label="log(exposure)",
    )

    with pytest.raises(
        ValueError,
        match="offset_export_options.*offset_source.*offset_name",
    ):
        api.build_candidate(
            context,
            model=model,
            frame=frame,
            X=frame[["age"]],
            y=frame["claim_count"],
            model_factory=lambda: object(),
            scoring=("deviance",),
            dataset_name="claim_frequency_frame",
            source_system="pricing_sql",
            data_as_of="2026-06-30",
            pk_columns=("policy_id",),
            split_indices=[(np.array([0, 1]), np.array([2, 3]))],
            offset=np.array([0.0, 0.2, -0.1, 0.4]),
            offset_contract=contract,
            offset_export_options={},
            run_key="custom-offset",
            created_by="analyst@example.test",
        )


@pytest.mark.parametrize("invalid_options", ["missing", "misaligned"])
def test_build_candidate_validates_caller_supplied_exported_offset_options(
    monkeypatch,
    tmp_path,
    invalid_options,
):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_spec_model(api, tmp_path)
    frame = pd.DataFrame(
        {
            "policy_id": [10, 20, 30, 40],
            "claim_count": [0.0, 1.0, 0.0, 2.0],
            "exposure": [1.0, 0.5, 1.5, 0.75],
            "age": [25.0, 45.0, 35.0, 52.0],
            "region": ["N", "S", "N", "S"],
        }
    )
    monkeypatch.setattr(
        api,
        "validation_split_indices",
        lambda frame, split: [(np.array([0, 1]), np.array([2, 3]))],
    )
    monkeypatch.setattr(api, "build_export_id", lambda *args: "export-1")
    monkeypatch.setattr(
        api,
        "resolve_model_version_for_export",
        lambda *args, **kwargs: "v1",
    )
    monkeypatch.setattr(
        api,
        "run_standard_superglm_build",
        lambda *args, **kwargs: SimpleNamespace(completed_build={}, metrics={}),
    )
    contract = api.OffsetExportContract(
        handling="EXPORTED_FACTOR",
        source_factor_name="exposure",
        published_factor_name="exposure",
        source_name="exposure",
        label="log(exposure)",
    )
    options = (
        {}
        if invalid_options == "missing"
        else {
            "offset_source": frame["exposure"],
            "offset_name": "different_exposure",
        }
    )
    expected = (
        "offset_export_options.*offset_source.*offset_name"
        if invalid_options == "missing"
        else "offset_name.*published_factor_name"
    )

    with pytest.raises(ValueError, match=expected):
        api.build_candidate(
            context,
            model=model,
            frame=frame,
            model_factory=lambda: object(),
            offset=np.array([0.0, 0.2, -0.1, 0.4]),
            offset_contract=contract,
            offset_export_options=options,
            data_as_of="2026-06-30",
            run_key="custom-offset",
            created_by="analyst@example.test",
        )


def test_publish_candidate_returns_generated_sql_ids(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    standard_build = SimpleNamespace(
        completed_build={"manifest_id": "manifest-9", "split_set_id": "split-9"}
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

    result = api.publish_candidate(context, candidate, created_by="analyst@example.test")

    assert result is expected
    assert result.model_id == 17
    assert result.model_run_id == 901
    assert result.rate_package_id == 71
    assert result.package_version == 4
    assert captured == {
        "engine": context.engine,
        "settings": context.settings,
        "model_config": model.config,
        "dataset": None,
        "completed_build": standard_build.completed_build,
        "created_by": "analyst@example.test",
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
    assert captured["config_loader"]("CLAIM_FREQUENCY") is model.config
    assert captured["model_names_loader"]() == ("CLAIM_FREQUENCY",)
    assert captured["open"] == ("CLAIM_FREQUENCY", 4)


def test_publish_edits_runs_editor_publisher_synchronously(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    session = object()
    candidate = SimpleNamespace(
        model_name=model.name,
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

    def create(loaded_candidate, **kwargs):
        captured["candidate"] = loaded_candidate
        captured["create"] = kwargs
        return submission

    def publish(engine, **kwargs):
        captured["engine"] = engine
        captured["publish"] = kwargs
        return expected

    monkeypatch.setattr(api, "create_editor_submission", create)
    monkeypatch.setattr(api, "publish_editor_submission", publish)

    result = api.publish_edits(
        context,
        model=model,
        candidate=candidate,
        reason="Sparse age-band market adjustment",
        created_by="analyst@example.test",
    )

    assert result is expected
    assert captured["candidate"] is candidate
    assert captured["create"]["editor_session"] is session
    assert captured["create"]["reason"] == "Sparse age-band market adjustment"
    assert captured["create"]["claimed_identity"] == "analyst@example.test"
    assert captured["create"]["airflow_client"].triggered is False
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
        editor_session=None,
        editor_widget=None,
    )

    try:
        api.publish_edits(
            context,
            model=model,
            candidate=candidate,
            reason="Market adjustment",
        )
    except RuntimeError as exc:
        assert "Open the candidate editor" in str(exc)
    else:
        raise AssertionError("publish_edits accepted a candidate without an open editor")


def test_deploy_package_reads_current_champion_then_uses_stale_guard(monkeypatch, tmp_path):
    from pricing_pipeline import notebook as api

    context = _context(api, tmp_path)
    model = _registered_model(api, tmp_path)
    package = SimpleNamespace(rate_package_id=72)
    expected = SimpleNamespace(
        model_id=17,
        previous_rate_package_id=61,
        rate_package_id=72,
        package_version=5,
    )
    captured = {}

    class FakeWorkbench:
        def __init__(self, **kwargs):
            captured["workbench"] = kwargs

        def current_champion_rate_package_id(self, model_name, *, deployment_slot=None):
            captured["champion"] = (model_name, deployment_slot)
            return 61

    def deploy(engine, config, **kwargs):
        captured["engine"] = engine
        captured["config"] = config
        captured["deploy"] = kwargs
        return expected

    monkeypatch.setattr(api, "Workbench", FakeWorkbench)
    monkeypatch.setattr(api, "deploy_rate_package", deploy)

    result = api.deploy_package(
        context,
        model=model,
        package=package,
        reason="Approved at August pricing meeting",
        deployed_by="pricing.manager@example.test",
    )

    assert result is expected
    assert captured["champion"] == ("CLAIM_FREQUENCY", "PRODUCTION")
    assert captured["engine"] is context.engine
    assert captured["config"] is model.config
    assert captured["deploy"] == {
        "rate_package_id": 72,
        "expected_current_rate_package_id": 61,
        "deployment_slot": "PRODUCTION",
        "deployment_reason": "Approved at August pricing meeting",
        "deployed_by": "pricing.manager@example.test",
        "model_id": 17,
    }
