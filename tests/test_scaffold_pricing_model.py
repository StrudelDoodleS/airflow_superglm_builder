from __future__ import annotations

import os
import subprocess
import sys

from scripts.scaffold_pricing_model import ScaffoldOptions, scaffold_pricing_model


def test_scaffold_pricing_model_writes_model_package_and_dag(tmp_path):
    result = scaffold_pricing_model(
        ScaffoldOptions(
            model_name="MY_MODEL",
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
        package_dir / "sql" / "source_data.sql",
        package_dir / "spec.py",
        package_dir / "data.py",
        package_dir / "modeling.py",
        package_dir / "airflow_tasks.py",
        dag_path,
    )
    assert not hasattr(result, "registry_instructions")

    model_toml = (package_dir / "model.toml").read_text(encoding="utf-8")
    assert 'model_name = "MY_MODEL"' in model_toml
    assert 'model_label = "My model"' in model_toml
    assert 'target_name = "derived_target"' in model_toml
    assert 'deployment_slot = "MY_MODEL_UAT"' in model_toml
    assert 'method = "train_test_split"' in model_toml
    assert "test_size = 0.20" in model_toml

    spec = (package_dir / "spec.py").read_text(encoding="utf-8")
    assert '"""Load the TOML model configuration for this pricing model."""' in spec
    assert "MODEL_CONFIG = load_model_build_config" in spec
    assert "ModelSpec" not in spec
    assert "TRAINING_SQL" not in spec
    assert "FEATURE_COLUMNS" not in spec

    sql = (package_dir / "sql" / "source_data.sql").read_text(encoding="utf-8")
    assert "Source-data query placeholder for MY_MODEL." in sql
    assert "data.py" in sql
    assert "SELECT" in sql

    data = (package_dir / "data.py").read_text(encoding="utf-8")
    assert '"""Read or stage source data for this pricing model.' in data
    assert 'SQL_DIR = Path(__file__).with_name("sql")' in data
    assert 'SOURCE_DATA_SQL = SQL_DIR / "source_data.sql"' in data
    assert "def prepare_source_data" in data
    assert "output_dir: str | Path" in data
    assert "output_root" not in data
    assert "DatasetSpec" not in data
    assert "dataset_spec_from_prepared" not in data
    assert "manifest_sql" not in data

    modeling = (package_dir / "modeling.py").read_text(encoding="utf-8")
    assert '"""Model-owned build logic for this pricing model.' in modeling
    assert "Edit These Model-Specific Functions" in modeling
    assert "Standard Build Recipe - Usually Leave This Alone" in modeling
    assert "data.py decides the handoff shape" in modeling
    assert "Do not pass large DataFrames through Airflow/XCom" in modeling
    assert "rating_workbook_path, model_artifact_path, metrics" in modeling
    assert "Start by customizing the functions above" in modeling
    assert "The manifest and split artifacts use this frame order" in modeling
    assert "def train_validate_export_model" in modeling
    assert "def read_prepared_source" in modeling
    assert "def build_final_model_frame" in modeling
    assert "def fit_validate_export_rating_tables" in modeling
    assert "def validation_split_indices_for_model" in modeling
    assert 'method = "custom"' in modeling
    assert "SQL lookup, external mapping" in modeling
    assert "source fold/holdout column" not in modeling
    assert "return validation_split_indices(frame, MODEL_CONFIG.validation_split)" in modeling
    assert "completed_build_helpers import" in modeling
    assert "completed_model_build_payload(" in modeling
    assert "resolve_model_version_for_export" in modeling
    assert "ModelFrameManifestSpec" in modeling
    assert "create_model_frame_manifest_with_split" in modeling
    assert "split_indices=split_indices" in modeling
    assert "manifest_id=manifest.manifest_id" in modeling
    assert "split_set_id=manifest.split_set_id" in modeling
    assert "If validation_split uses a source split column" in modeling
    assert "column as a rating feature unless this is an intentional model decision" in modeling
    assert "CompletedModelBuild(" not in modeling
    assert "def effective_from_for_run" not in modeling
    assert "def existing_model_version_for_export" not in modeling
    assert "def next_model_version" not in modeling
    assert "def resolve_model_version" not in modeling
    assert "def completed_model_build_payload" not in modeling

    airflow_tasks = (package_dir / "airflow_tasks.py").read_text(encoding="utf-8")
    assert '"""Airflow TaskFlow wrappers for this model package.' in airflow_tasks
    assert "Keep business logic in data.py and modeling.py" in airflow_tasks
    assert "def prepare_source_data_task" in airflow_tasks
    assert "def train_validate_export_task" in airflow_tasks
    assert "airflow_run_metadata import" in airflow_tasks
    assert "task_run_metadata(" in airflow_tasks
    assert "merge_prepared_payload_metadata(" in airflow_tasks
    assert "def _context_logical_date" not in airflow_tasks

    dag = dag_path.read_text(encoding="utf-8")
    assert '"""Airflow DAG for the MY_MODEL pricing model build."""' in dag
    assert "from pricing_models.my_model.spec import MODEL_CONFIG" in dag
    assert "register_pricing_model_task" in dag
    assert "create_prepared_dataset_manifest_task" not in dag
    assert "dataset_spec_from_prepared" not in dag
    assert "publish_completed_model_build_task" in dag
    assert 'dag_id="pricing_my_model"' in dag
    assert "build_pricing_model_dag" not in dag
    assert "MODEL_SPEC" not in dag


def test_scaffold_pricing_model_can_write_factory_template(tmp_path):
    result = scaffold_pricing_model(
        ScaffoldOptions(
            model_name="MY_MODEL",
            model_label="My model",
            target_name="derived_target",
            root=tmp_path,
            template="factory",
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

    training = (package_dir / "training.py").read_text(encoding="utf-8")
    assert "TRAINING_SQL" in training
    assert "FEATURE_COLUMNS" in training
    assert "def build_training_frame" in training
    assert "def build_model" in training
    assert 'df["derived_target"]' in training

    spec = (package_dir / "spec.py").read_text(encoding="utf-8")
    assert "MODEL_CONFIG = load_model_build_config" in spec
    assert 'dataset_name="my_model_training"' in spec
    assert 'pk_columns=("REPLACE_ME_ID",)' in spec
    assert "target_column=MODEL_CONFIG.target_name" in spec
    assert 'experiment_name="pricing-my-model"' in spec

    dag = dag_path.read_text(encoding="utf-8")
    assert "from pricing_models.my_model.spec import MODEL_CONFIG, MODEL_SPEC" in dag
    assert 'dag_id="pricing_my_model"' in dag
    assert "pricing_my_model = build_pricing_model_dag" in dag


def test_custom_scaffold_is_config_discovered_but_not_spec_runnable(tmp_path):
    scaffold_pricing_model(
        ScaffoldOptions(
            model_name="CUSTOM_MODEL",
            model_label="Custom model",
            target_name="target",
            root=tmp_path,
        )
    )
    scaffold_pricing_model(
        ScaffoldOptions(
            model_name="FACTORY_MODEL",
            model_label="Factory model",
            target_name="target",
            root=tmp_path,
            template="factory",
        )
    )

    from pricing_models.registry import get_model_config, model_names, model_spec_names

    models_root = tmp_path / "pricing_models"

    assert model_names(models_root=models_root) == ("CUSTOM_MODEL", "FACTORY_MODEL")
    assert get_model_config("CUSTOM_MODEL", models_root=models_root).model_name == "CUSTOM_MODEL"
    assert model_spec_names(models_root=models_root) == ("FACTORY_MODEL",)


def test_scaffold_pricing_model_skips_existing_files_and_recreates_missing_files(tmp_path):
    options = ScaffoldOptions(
        model_name="MY_MODEL",
        model_label="My model",
        target_name="target",
        root=tmp_path,
    )
    scaffold_pricing_model(options)

    package_dir = tmp_path / "pricing_models" / "my_model"
    data_path = package_dir / "data.py"
    toml_path = package_dir / "model.toml"
    modeling_path = package_dir / "modeling.py"

    data_path.write_text(data_path.read_text(encoding="utf-8") + "\n# user edit\n")
    toml_path.write_text(toml_path.read_text(encoding="utf-8") + "\n# user edit\n")
    data_before = data_path.read_text(encoding="utf-8")
    toml_before = toml_path.read_text(encoding="utf-8")
    modeling_path.unlink()

    result = scaffold_pricing_model(options)

    assert result.created_files == (modeling_path,)
    assert modeling_path.exists()
    assert data_path.read_text(encoding="utf-8") == data_before
    assert toml_path.read_text(encoding="utf-8") == toml_before

    rerun = scaffold_pricing_model(options)

    assert rerun.created_files == ()


def test_scaffold_pricing_model_force_overwrites_existing_files(tmp_path):
    options = ScaffoldOptions(
        model_name="MY_MODEL",
        model_label="My model",
        target_name="target",
        root=tmp_path,
    )
    scaffold_pricing_model(options)
    model_toml = tmp_path / "pricing_models" / "my_model" / "model.toml"
    model_toml.write_text("# stale\n", encoding="utf-8")

    result = scaffold_pricing_model(
        ScaffoldOptions(
            model_name="MY_MODEL",
            model_label="My model",
            target_name="target",
            root=tmp_path,
            force=True,
        )
    )

    assert model_toml in result.created_files
    assert "# stale" not in model_toml.read_text(encoding="utf-8")


def test_scaffold_pricing_model_accepts_explicit_names(tmp_path):
    result = scaffold_pricing_model(
        ScaffoldOptions(
            model_name="WORK_FREQ",
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
    model_toml = (tmp_path / "pricing_models" / "work_frequency" / "model.toml").read_text(
        encoding="utf-8"
    )
    assert 'model_type = "superglm_tweedie"' in model_toml
    assert 'deployment_slot = "WORK_FREQ_PROD"' in model_toml
    assert "pricing-work-frequency-prod" not in model_toml
    dag = (tmp_path / "dags" / "price_work_frequency.py").read_text(encoding="utf-8")
    assert '"""Airflow DAG for the WORK_FREQ pricing model build."""' in dag


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
    assert "--model-name" in result.stdout
    assert "--target-name" in result.stdout
    assert "--template" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_scaffold_pricing_model_script_reports_toml_discovery(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/scaffold_pricing_model.py",
            "--model-name",
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
