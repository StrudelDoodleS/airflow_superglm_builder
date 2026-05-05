from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

from pricing_pipeline.config import Settings
from scripts import no_docker_services


def test_no_docker_env_example_targets_host_processes_and_external_sql():
    env_example = Path(".env.nodocker.example")

    assert env_example.exists()
    text = env_example.read_text(encoding="utf-8")

    assert "MSSQL_SERVER=<server-name>.database.windows.net,1433" in text
    assert "PRICING_SKIP_DATABASE_CREATE=true" in text
    assert "MLFLOW_TRACKING_URI=http://127.0.0.1:5000" in text
    assert "RATING_EXPORT_ROOT=state/rating_exports" in text
    assert "mssql,1433" not in text
    assert "/opt/pricing" not in text


def test_no_docker_scripts_exist_without_compose_dependency():
    for script in [
        Path("scripts/bootstrap_no_docker.sh"),
        Path("scripts/no_docker_services.py"),
        Path("scripts/start_no_docker_stack.sh"),
        Path("scripts/start_airflow_local.py"),
        Path("scripts/start_mlflow_local.py"),
        Path("scripts/run_pipeline_no_airflow.py"),
    ]:
        assert script.exists(), f"{script} is missing"
        text = script.read_text(encoding="utf-8")
        if script.name != "start_no_docker_stack.sh":
            assert "docker compose" not in text.lower()


def test_settings_can_skip_database_creation_for_hosted_targets():
    settings = Settings.from_env({"PRICING_SKIP_DATABASE_CREATE": "true"})

    assert settings.skip_database_create is True


def test_dag_migrations_dir_can_be_overridden_for_no_docker(monkeypatch, tmp_path):
    monkeypatch.setenv("PRICING_MIGRATIONS_DIR", str(tmp_path))

    airflow_module = types.ModuleType("airflow")
    airflow_sdk_module = types.ModuleType("airflow.sdk")

    class FakeTaskOutput:
        def __rshift__(self, other):
            return other

    def dag(**dag_kwargs):
        def decorator(func):
            def factory(*args, **kwargs):
                func(*args, **kwargs)
                return types.SimpleNamespace(dag_id=dag_kwargs["dag_id"])

            return factory

        return decorator

    def task(func):
        def task_factory(*args, **kwargs):
            return FakeTaskOutput()

        return task_factory

    airflow_sdk_module.dag = dag
    airflow_sdk_module.get_current_context = lambda: {}
    airflow_sdk_module.task = task
    airflow_module.sdk = airflow_sdk_module
    monkeypatch.setitem(sys.modules, "airflow", airflow_module)
    monkeypatch.setitem(sys.modules, "airflow.sdk", airflow_sdk_module)

    dag_path = Path("dags/pricing_superglm_pipeline.py").resolve()
    spec = importlib.util.spec_from_file_location("pricing_superglm_pipeline_override", dag_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.MIGRATIONS_DIR == tmp_path


def test_no_airflow_runner_help_runs_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/run_pipeline_no_airflow.py", "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "--ensure-database" in result.stdout
    assert "--skip-raw-load" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_no_docker_service_picker_lists_available_services_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/no_docker_services.py", "list"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "mlflow" in result.stdout
    assert "airflow" in result.stdout
    assert "migrate" in result.stdout
    assert "cloudbeaver" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_no_docker_service_picker_builds_python_commands():
    commands = no_docker_services.selected_commands(
        ["migrate", "load-raw-replace", "pipeline"],
        python_executable="/python",
    )

    assert [command.name for command in commands] == [
        "migrate",
        "load-raw-replace",
        "pipeline",
    ]
    assert commands[0].argv == ["/python", "scripts/apply_sql_migrations.py"]
    assert commands[1].argv == ["/python", "scripts/load_fremtpl_raw.py", "--replace"]
    assert commands[2].argv == ["/python", "scripts/run_pipeline_no_airflow.py"]
    assert not any("docker" in part.lower() for command in commands for part in command.argv)


def test_interactive_shell_launcher_help_documents_keyboard_menu():
    result = subprocess.run(
        ["bash", "scripts/start_no_docker_stack.sh", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "keyboard menu" in result.stdout
    assert "--services airflow,mlflow" in result.stdout
    assert "cloudbeaver" in result.stdout


def test_interactive_shell_launcher_dry_run_selected_services():
    result = subprocess.run(
        [
            "bash",
            "scripts/start_no_docker_stack.sh",
            "--dry-run",
            "--services",
            "migrate,mlflow,airflow",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "uv run python scripts/apply_sql_migrations.py" in result.stdout
    assert "uv run python scripts/start_mlflow_local.py" in result.stdout
    assert "uv run python scripts/start_airflow_local.py" in result.stdout
    assert "docker compose" not in result.stdout


def test_interactive_shell_launcher_cloudbeaver_is_explicitly_docker_backed():
    result = subprocess.run(
        [
            "bash",
            "scripts/start_no_docker_stack.sh",
            "--dry-run",
            "--services",
            "cloudbeaver",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "cloudbeaver uses Docker Compose in this repo" in result.stdout
    assert "docker compose --profile sql-ui up -d cloudbeaver" in result.stdout
