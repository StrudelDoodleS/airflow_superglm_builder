from __future__ import annotations

import os
import subprocess
import sys

import pytest

from scripts.scaffold_pricing_model import ScaffoldOptions, scaffold_pricing_model


def test_scaffold_pricing_model_writes_model_package_and_dag(tmp_path):
    result = scaffold_pricing_model(
        ScaffoldOptions(
            model_key="MY_MODEL",
            model_label="My model",
            target_name="derived_target",
            root=tmp_path,
        )
    )

    package_dir = tmp_path / "pricing_models" / "my_model"
    dag_path = tmp_path / "dags" / "pricing_my_model.py"

    assert result.created_files == (
        package_dir / "__init__.py",
        package_dir / "model.toml",
        package_dir / "training.py",
        package_dir / "spec.py",
        dag_path,
    )
    assert not hasattr(result, "registry_instructions")

    model_toml = (package_dir / "model.toml").read_text(encoding="utf-8")
    assert 'model_key = "MY_MODEL"' in model_toml
    assert 'model_label = "My model"' in model_toml
    assert 'target_name = "derived_target"' in model_toml
    assert 'deployment_slot = "MY_MODEL_UAT"' in model_toml
    assert 'method = "train_test_split"' in model_toml
    assert "test_size = 0.20" in model_toml

    training = (package_dir / "training.py").read_text(encoding="utf-8")
    assert "TRAINING_SQL" in training
    assert "FEATURE_COLUMNS" in training
    assert "def build_training_frame" in training
    assert "def build_model" in training
    assert "df[\"derived_target\"]" in training

    spec = (package_dir / "spec.py").read_text(encoding="utf-8")
    assert "MODEL_CONFIG = load_model_build_config" in spec
    assert 'dataset_name="my_model_training"' in spec
    assert 'pk_columns=("REPLACE_ME_ID",)' in spec
    assert "target_column=MODEL_CONFIG.target_name" in spec
    assert "experiment_name=\"pricing-my-model\"" in spec

    dag = dag_path.read_text(encoding="utf-8")
    assert "from pricing_models.my_model.spec import MODEL_CONFIG, MODEL_SPEC" in dag
    assert 'dag_id="pricing_my_model"' in dag
    assert "pricing_my_model = build_pricing_model_dag" in dag


def test_scaffold_pricing_model_refuses_to_overwrite_existing_files(tmp_path):
    options = ScaffoldOptions(
        model_key="MY_MODEL",
        model_label="My model",
        target_name="target",
        root=tmp_path,
    )
    scaffold_pricing_model(options)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        scaffold_pricing_model(options)

    result = scaffold_pricing_model(
        ScaffoldOptions(
            model_key="MY_MODEL",
            model_label="My model",
            target_name="target",
            root=tmp_path,
            force=True,
        )
    )

    assert tmp_path / "pricing_models" / "my_model" / "model.toml" in result.created_files


def test_scaffold_pricing_model_accepts_explicit_names(tmp_path):
    result = scaffold_pricing_model(
        ScaffoldOptions(
            model_key="WORK_FREQ",
            model_label="Work frequency",
            target_name="claim_count",
            model_type="superglm_tweedie",
            deployment_slot="WORK_FREQ_PROD",
            package_name="work_frequency",
            dag_id="price_work_frequency",
            experiment_name="pricing-work-frequency-prod",
            root=tmp_path,
        )
    )

    assert result.package_name == "work_frequency"
    assert result.dag_id == "price_work_frequency"
    model_toml = (
        tmp_path / "pricing_models" / "work_frequency" / "model.toml"
    ).read_text(encoding="utf-8")
    assert 'model_type = "superglm_tweedie"' in model_toml
    assert 'deployment_slot = "WORK_FREQ_PROD"' in model_toml
    assert "pricing-work-frequency-prod" not in model_toml


def test_scaffold_pricing_model_script_help_runs_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/scaffold_pricing_model.py", "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "--model-key" in result.stdout
    assert "--target-name" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_scaffold_pricing_model_script_reports_toml_discovery(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/scaffold_pricing_model.py",
            "--model-key",
            "SCRIPT_MODEL",
            "--target-name",
            "target",
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "auto-discovered from pricing_models/script_model/model.toml" in result.stdout
    assert "add these lines to pricing_models/registry.py" not in result.stdout
