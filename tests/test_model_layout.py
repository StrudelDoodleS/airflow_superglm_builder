from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pricing_models.mtpl_frequency.spec import MODEL_CONFIG, MODEL_SPEC


def _write_model_toml(
    package_dir,
    *,
    model_name: str,
    model_label: str | None = None,
    target_name: str = "target",
) -> None:
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "model.toml").write_text(
        dedent(
            f"""\
            model_name = "{model_name}"
            model_label = "{model_label or model_name.title()}"
            target_name = "{target_name}"
            model_type = "superglm_poisson"
            deployment_slot = "{model_name}_UAT"
            default_package_status = "PUBLISHED"

            [validation_split]
            method = "train_test_split"
            test_size = 0.2
            random_state = 42
            shuffle = true
            materialize = false
            """
        ),
        encoding="utf-8",
    )


def test_mtpl_frequency_spec_lives_in_model_package():
    from pricing_models.mtpl_frequency.spec import MODEL_SPEC

    assert MODEL_SPEC.model_name == "MTPL_FREQ"
    assert MODEL_SPEC.target_name == "ClaimNb"
    assert MODEL_SPEC.model_type == "superglm_poisson"
    assert MODEL_SPEC.experiment_name == "pricing-mtpl-frequency"
    assert MODEL_SPEC.deployment_slot == "MTPL_FREQ_UAT"
    assert MODEL_SPEC.dataset.dataset_name == "freMTPL2freq"
    assert MODEL_SPEC.build_model.__module__ == "pricing_models.mtpl_frequency.modeling"
    assert MODEL_SPEC.build_training_frame.__module__ == "pricing_models.mtpl_frequency.modeling"


def test_model_specs_are_available_from_registry():
    from pricing_models.registry import get_model_spec, model_names

    assert "MTPL_FREQ" in model_names()
    assert get_model_spec("MTPL_FREQ").model_name == "MTPL_FREQ"


def test_model_configs_are_available_from_registry():
    from pricing_models.registry import get_model_config

    assert get_model_config("MTPL_FREQ") == MODEL_CONFIG
    assert MODEL_CONFIG.validation_split.method == "kfold"
    assert MODEL_CONFIG.validation_split.n_splits == 5
    assert MODEL_CONFIG.validation_split.random_state == 42
    assert MODEL_CONFIG.validation_split.materialize is True


def test_mtpl_frequency_custom_model_modules_define_explicit_workflow_hooks():
    from pricing_models.mtpl_frequency import data, modeling

    assert data.MODEL_FRAME.dataset_name == "freMTPL2freq_model_frame"
    assert data.MODEL_FRAME.source_system == "freMTPL_raw_sql"
    assert data.MODEL_FRAME.pk_columns == ("IDpol",)
    assert data.MODEL_FRAME.target_column == "ClaimNb"
    assert data.MODEL_FRAME.weight_column == "Exposure"
    assert data.DEFAULT_OUTPUT_ROOT.as_posix().endswith("state/mtpl_frequency")
    assert hasattr(data, "prepare_source_data")
    assert modeling.FEATURE_COLUMNS == [
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "LogDensity",
        "Area",
        "VehPower",
        "VehBrand",
        "VehGas",
        "Region",
    ]
    assert not hasattr(modeling, "FEATURE_SOURCE_COLUMNS")
    assert not hasattr(modeling, "REQUIRED_RAW_COLUMNS")
    assert not hasattr(modeling, "FINAL_MODEL_FRAME_COLUMNS")

    raw = pd.DataFrame(
        {
            "IDpol": [1, 2, 3],
            "ClaimNb": [0, 2, 1],
            "Exposure": [0.5, 1.0, 2.0],
            "Area": ["A", "B", "C"],
            "VehPower": [6, 7, 8],
            "VehAge": [3, 4, 5],
            "DrivAge": [45, 50, 55],
            "BonusMalus": [50, 60, 70],
            "VehBrand": ["B1", "B2", "B3"],
            "VehGas": ["Regular", "Diesel", "Regular"],
            "Density": [0.2, 50.0, 123.0],
            "Region": ["R1", "R2", "R3"],
        }
    )
    frame = modeling.build_final_model_frame(raw)

    assert list(frame.columns) == [
        "IDpol",
        "ClaimNb",
        "Exposure",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "LogDensity",
        "Area",
        "VehPower",
        "VehBrand",
        "VehGas",
        "Region",
    ]
    np.testing.assert_allclose(
        frame["LogDensity"].to_numpy(),
        np.log(np.array([1.0, 50.0, 123.0])),
    )
    assert "build_model" in dir(modeling)
    assert "train_validate_export_model" in dir(modeling)


def test_mtpl_frequency_modeling_imports_single_frame_contract_object():
    import pricing_models.mtpl_frequency.modeling as modeling

    source = Path(modeling.__file__).read_text(encoding="utf-8")
    assert "from pricing_models.mtpl_frequency.data import MODEL_FRAME" in source
    assert "MODEL_FRAME.final_columns" not in source
    assert "FEATURE_SOURCE_COLUMNS" not in source
    assert "REQUIRED_RAW_COLUMNS" not in source
    assert "FINAL_MODEL_FRAME_COLUMNS" not in source
    assert "DATASET_NAME" not in source
    assert "SOURCE_SYSTEM" not in source
    assert "TARGET_COLUMN" not in source
    assert "WEIGHT_COLUMN" not in source
    assert "fit_reml_with_diagnostics" not in source
    assert "mlflow_client.log_param" in source
    assert "mlflow_client.log_metric" in source
    assert "mlflow_client.log_artifact" in source


def test_mtpl_frequency_fit_export_logs_visible_superglm_mlflow_diagnostics(
    monkeypatch,
    tmp_path,
):
    from pricing_models.mtpl_frequency import modeling

    frame = pd.DataFrame(
        {
            "IDpol": [1, 2, 3, 4],
            "ClaimNb": [0, 1, 0, 2],
            "Exposure": [0.5, 1.0, 0.75, 1.25],
            "VehAge": [3, 4, 5, 6],
            "DrivAge": [45, 50, 55, 60],
            "BonusMalus": [50, 60, 70, 80],
            "LogDensity": [0.0, 2.0, 3.0, 4.0],
            "Area": ["A", "B", "C", "D"],
            "VehPower": [6, 7, 8, 9],
            "VehBrand": ["B1", "B2", "B3", "B4"],
            "VehGas": ["Regular", "Diesel", "Regular", "Diesel"],
            "Region": ["R1", "R2", "R3", "R4"],
        }
    )
    calls = []

    class FakeMetrics:
        deviance = 8.5
        null_deviance = 12.0
        aic = 4.0
        bic = 5.0
        effective_df = 3.25
        explained_deviance = 0.5
        log_likelihood = -2.0
        n_active_groups = 7
        n_obs = 4
        pearson_chi2 = 1.25
        phi = 1.0

    class FakeModel:
        family = "poisson"

        def __init__(self):
                self.result = SimpleNamespace(
                    deviance=8.5,
                    n_iter=3,
                    converged=True,
                    effective_df=3.25,
                    phi=1.0,
                )
                self._specs = {"VehAge": modeling.Numeric()}
                self._feature_order = ["VehAge"]

        def fit_reml(self, X, y, *, offset=None, verbose=False):
            calls.append(("fit_reml", list(X.columns), len(y), verbose))
            self._fit_used_offset = offset is not None
            print("POI iter 1  obj=10.5  |grad|=2.5  delta_obj=inf  [VehAge=0.1, DrivAge=0.2]")
            print("POI iter 2  obj=8.25  |grad|=1.25  delta_obj=2.25  [VehAge=0.3, DrivAge=0.4]")
            return self

        def summary(self, detail="compact"):
            return f"summary detail={detail}"

        def diagnostics(self):
            return {"_model": {"deviance": 8.5}, "VehAge": {"edf": 1.2}}

        def reml_diagnostics(self):
            return {
                "enabled": True,
                "lambdas": {"VehAge": 0.3, "DrivAge": 0.4},
                "lambda_history": [
                    {"VehAge": 0.1, "DrivAge": 0.2},
                    {"VehAge": 0.3, "DrivAge": 0.4},
                ],
            }

        def metrics(self, X, y, *, sample_weight=None, offset=None):
            calls.append(("metrics", len(X), len(y), sample_weight is not None))
            return FakeMetrics()

    class FakeMlflow:
        def log_param(self, key, value):
            calls.append(("log_param", key, value))

        def log_metric(self, key, value, **kwargs):
            calls.append(("log_metric", key, value, kwargs))

        def log_artifact(self, path, artifact_path=None):
            calls.append(("log_artifact", Path(path).name, artifact_path))

        def start_span(self, name, span_type=None, attributes=None):
            calls.append(("start_span", name, span_type, attributes))

            class FakeSpan:
                def set_inputs(self, value):
                    calls.append(("span_inputs", value))

                def set_outputs(self, value):
                    calls.append(("span_outputs", value))

                def set_attributes(self, value):
                    calls.append(("span_attributes", value))

            class FakeSpanContext:
                def __enter__(self):
                    calls.append(("span_enter",))
                    return FakeSpan()

                def __exit__(self, exc_type, exc, tb):
                    calls.append(("span_exit", exc_type))
                    return False

            return FakeSpanContext()

    def fake_export_rating_tables(
        model,
        X,
        y,
        exposure,
        *,
        output_path,
        offset,
        offset_source,
        offset_name,
        offset_kind,
        offset_max_exact_levels,
        mlflow_client,
    ):
        Path(output_path).write_text("workbook", encoding="utf-8")
        calls.append(
            (
                "export_rating_tables",
                Path(output_path).name,
                offset.copy(),
                offset_source.copy(),
                offset_name,
                offset_kind,
                offset_max_exact_levels,
                mlflow_client,
            )
        )
        return output_path

    monkeypatch.setattr(modeling, "build_model", FakeModel)
    monkeypatch.setattr(modeling, "export_rating_tables", fake_export_rating_tables)
    monkeypatch.setattr(
        modeling.pickle,
        "dump",
        lambda fitted, handle: handle.write(b"fake-model"),
    )

    workbook_path, model_path, metrics = modeling.fit_validate_export_rating_tables(
        frame,
        split_indices=[(np.array([0, 1]), np.array([2, 3]))],
        output_dir=tmp_path,
        model_version="v1",
        effective_from="2026-06-05",
        mlflow_client=FakeMlflow(),
    )

    assert Path(workbook_path).exists()
    assert Path(model_path).exists()
    assert (tmp_path / "v1_2026-06-05" / "superglm_fit.log").exists()
    assert (tmp_path / "v1_2026-06-05" / "superglm_diagnostics.json").exists()
    assert (tmp_path / "v1_2026-06-05" / "superglm_reml_diagnostics.json").exists()
    assert (tmp_path / "v1_2026-06-05" / "superglm_training_trace.csv").exists()
    assert metrics["deviance"] == 8.5
    assert "superglm_aic" not in metrics
    assert "first_fold_train_rows" not in metrics
    assert "first_fold_test_rows" not in metrics
    assert "effective_df" not in metrics
    assert "phi" not in metrics
    assert ("fit_reml", modeling.FEATURE_COLUMNS, 4, True) in calls
    assert not any(call[0] == "metrics" for call in calls)
    assert ("log_param", "family", "poisson") in calls
    assert ("log_param", "feature_columns", ",".join(modeling.FEATURE_COLUMNS)) in calls
    assert (
        "start_span",
        "superglm.fit_reml",
        "TRAINING",
        {"row_count": 4, "feature_count": len(modeling.FEATURE_COLUMNS)},
    ) in calls
    assert ("span_inputs", {"rows": 4, "features": modeling.FEATURE_COLUMNS}) in calls
    assert ("span_outputs", {"final_training_objective": 8.25, "iteration_count": 2}) in calls
    assert ("log_metric", "deviance", 8.5, {}) in calls
    assert ("log_metric", "superglm_training_objective", 10.5, {"step": 0}) in calls
    assert ("log_metric", "superglm_training_objective", 8.25, {"step": 1}) in calls
    assert ("log_metric", "superglm_reml_gradient_norm", 2.5, {"step": 0}) in calls
    logged_metric_names = [call[1] for call in calls if call and call[0] == "log_metric"]
    assert "loss" not in logged_metric_names
    assert "training_loss" not in logged_metric_names
    assert "superglm_reml_objective" not in logged_metric_names
    assert "superglm_reml_delta_objective" not in logged_metric_names
    assert "superglm_aic" not in logged_metric_names
    assert "effective_df" not in logged_metric_names
    assert "phi" not in logged_metric_names
    assert "superglm_phi" not in logged_metric_names
    assert "superglm_lambda_VehAge" not in logged_metric_names
    assert "superglm_lambda_DrivAge" not in logged_metric_names
    export_call = next(call for call in calls if call[0] == "export_rating_tables")
    np.testing.assert_allclose(export_call[2], np.log(frame["Exposure"].to_numpy(dtype=float)))
    pd.testing.assert_series_equal(export_call[3].reset_index(drop=True), frame["Exposure"])
    assert export_call[4] == "Exposure"
    assert export_call[5] == "discrete"
    assert export_call[6] == 4
    assert ("log_artifact", "superglm_model.pkl", "model") in calls
    assert ("log_artifact", "model_summary.txt", "model") in calls
    assert ("log_artifact", "superglm_fit.log", "training_diagnostics") in calls
    assert ("log_artifact", "superglm_diagnostics.json", "training_diagnostics") in calls
    assert ("log_artifact", "superglm_reml_diagnostics.json", "training_diagnostics") in calls
    assert ("log_artifact", "superglm_training_trace.csv", "training_diagnostics") in calls


def test_mtpl_frequency_training_is_compatibility_shim():
    from pricing_models.mtpl_frequency import modeling, training

    source = Path(training.__file__).read_text(encoding="utf-8")
    assert "compatibility" in source.lower()
    assert "from superglm import" not in source
    assert "def build_model" not in source
    assert "def build_training_frame" not in source
    assert training.FEATURE_COLUMNS is modeling.FEATURE_COLUMNS
    assert training.build_model is modeling.build_model
    assert training.build_training_frame is modeling.build_training_frame


def test_mtpl_frequency_prepare_source_data_uses_configured_schema(monkeypatch, tmp_path):
    from pricing_models.mtpl_frequency import data

    class DummyEngine:
        _execution_options = {"pricing_schema": "python_pricing"}

    monkeypatch.setattr(data, "load_fremtpl_raw", lambda engine, *, replace=False: 3)

    payload = data.prepare_source_data(
        DummyEngine(),
        run_key="manual",
        output_dir=tmp_path,
    )

    assert payload["source_row_count"] == 3
    assert "FROM python_pricing.FREMTPL_RAW" in payload["source_sql"]


def test_mtpl_frequency_airflow_wrappers_are_thin_task_factories():
    from pricing_models.mtpl_frequency import airflow_tasks

    assert hasattr(airflow_tasks, "prepare_source_data_task")
    assert hasattr(airflow_tasks, "train_validate_export_task")


def test_model_config_registry_discovers_toml_without_importing_specs(tmp_path, monkeypatch):
    models_root = tmp_path / "pricing_models"
    package_dir = models_root / "lazy_model"
    _write_model_toml(package_dir, model_name="LAZY_MODEL", model_label="Lazy model")
    (package_dir / "spec.py").write_text(
        "raise RuntimeError('spec import should not happen for config lookup')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(tmp_path)

    from pricing_models.registry import get_model_config, model_names

    assert model_names(models_root=models_root) == ("LAZY_MODEL",)
    config = get_model_config("LAZY_MODEL", models_root=models_root)
    assert config.model_name == "LAZY_MODEL"
    assert config.model_label == "Lazy model"


def test_model_spec_registry_lazy_imports_only_selected_package(tmp_path, monkeypatch):
    models_root = tmp_path / "pricing_models"
    selected_dir = models_root / "selected_model"
    poison_dir = models_root / "poison_model"
    _write_model_toml(selected_dir, model_name="SELECTED_MODEL")
    _write_model_toml(poison_dir, model_name="POISON_MODEL")
    (selected_dir / "spec.py").write_text(
        dedent(
            """\
            from pricing_pipeline.models.spec import DatasetSpec, ModelSpec

            def build_model():
                return object()

            def build_training_frame(raw):
                return raw

            MODEL_SPEC = ModelSpec(
                model_name="SELECTED_MODEL",
                model_label="Selected model",
                target_name="target",
                model_type="superglm_poisson",
                experiment_name="pricing-selected",
                deployment_slot="SELECTED_MODEL_UAT",
                dataset=DatasetSpec(
                    dataset_name="selected_training",
                    source_system="sql_server",
                    manifest_sql="SELECT 1",
                    pk_columns=("id",),
                    target_column="target",
                ),
                training_sql="SELECT 1",
                feature_columns=("rating_factor",),
                build_model=build_model,
                build_training_frame=build_training_frame,
                package_status="PUBLISHED",
            )
            """
        ),
        encoding="utf-8",
    )
    (poison_dir / "spec.py").write_text(
        "raise RuntimeError('unselected spec import should not happen')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(tmp_path)

    from pricing_models.registry import get_model_spec

    spec = get_model_spec(
        "SELECTED_MODEL",
        models_root=models_root,
        package_prefix="pricing_models",
    )

    assert spec.model_name == "SELECTED_MODEL"


def test_model_config_registry_rejects_duplicate_model_names(tmp_path):
    models_root = tmp_path / "pricing_models"
    _write_model_toml(models_root / "first_model", model_name="DUPLICATE_MODEL")
    _write_model_toml(models_root / "second_model", model_name="DUPLICATE_MODEL")

    from pricing_models.registry import model_names

    try:
        model_names(models_root=models_root)
    except ValueError as exc:
        assert "Duplicate model_name 'DUPLICATE_MODEL'" in str(exc)
    else:
        raise AssertionError("duplicate model names should fail registry discovery")


def test_mtpl_frequency_model_config_matches_spec_identity():
    assert MODEL_CONFIG.model_name == MODEL_SPEC.model_name
    assert MODEL_CONFIG.model_label == MODEL_SPEC.model_label
    assert MODEL_CONFIG.target_name == MODEL_SPEC.target_name
    assert MODEL_CONFIG.model_type == MODEL_SPEC.model_type
    assert MODEL_CONFIG.deployment_slot == MODEL_SPEC.deployment_slot
    assert MODEL_CONFIG.default_package_status == "PUBLISHED"
