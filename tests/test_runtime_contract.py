import os
from pathlib import Path

import pytest
import yaml

from pricing_pipeline.infra.config import Settings


MSSQL_PASSWORD_DEFAULT = "${MSSQL_PASSWORD:-YourStrong(!)Password123}"
STATE_PATHS = [
    "/sources/state/mssql/data",
    "/sources/state/mlflow/artifacts",
    "/sources/state/rating_exports",
    "/sources/state/cv_splits",
    "/sources/state/db_diagrams",
    "/sources/state/cloudbeaver/workspace",
]


def test_notebook_runtime_does_not_depend_on_airflow():
    production_sources = [
        *Path("pricing_pipeline").rglob("*.py"),
        *Path("dags").glob("*.py"),
    ]

    airflow_imports = []
    for path in production_sources:
        source = path.read_text(encoding="utf-8")
        if "from airflow" in source or "import airflow" in source:
            airflow_imports.append(path.as_posix())

    assert airflow_imports == []
    assert list(Path("dags").glob("*.py")) == []


def test_compose_uses_airflow_321_services():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    for name in [
        "airflow-apiserver",
        "airflow-scheduler",
        "airflow-dag-processor",
        "airflow-worker",
        "airflow-triggerer",
        "postgres",
        "redis",
        "flower",
        "state-init",
        "mssql",
        "mssql-init",
        "mlflow",
        "db-diagram-generator",
        "db-diagrams",
    ]:
        assert name in services
    assert services["flower"]["profiles"] == ["flower"]
    assert "redis://:@redis:6379/0" in str(compose["x-airflow-common"]["environment"])


def test_mssql_password_default_is_consistent_across_runtime_services():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    common_env = compose["x-airflow-common"]["environment"]
    mssql_env = services["mssql"]["environment"]
    mlflow_env = services["mlflow"]["environment"]

    assert common_env["MSSQL_PASSWORD"] == MSSQL_PASSWORD_DEFAULT
    assert mssql_env["MSSQL_SA_PASSWORD"] == MSSQL_PASSWORD_DEFAULT
    assert mlflow_env["MSSQL_PASSWORD"] == MSSQL_PASSWORD_DEFAULT
    assert "AirflowSuperGLM!2026" not in Path("docker-compose.yml").read_text(encoding="utf-8")


def test_state_init_prepares_project_state_directories():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    state_init = services["state-init"]
    command = "\n".join(str(part) for part in state_init["command"])

    assert "${AIRFLOW_PROJ_DIR:-.}:/sources" in state_init["volumes"]
    for state_path in STATE_PATHS:
        assert state_path in command
    assert services["mssql"]["depends_on"]["state-init"] == {
        "condition": "service_completed_successfully"
    }
    assert "airflow-init" not in services["mssql"].get("depends_on", {})


def test_state_mounts_follow_airflow_project_dir():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert (
        "${AIRFLOW_PROJ_DIR:-.}/state/rating_exports:/opt/pricing/state/rating_exports"
        in compose["x-airflow-common"]["volumes"]
    )
    assert (
        "${AIRFLOW_PROJ_DIR:-.}/state/cv_splits:/opt/pricing/state/cv_splits"
        in compose["x-airflow-common"]["volumes"]
    )
    assert "${AIRFLOW_PROJ_DIR:-.}/state/mssql/data:/var/opt/mssql" in services["mssql"]["volumes"]
    assert (
        "${AIRFLOW_PROJ_DIR:-.}/state/mlflow/artifacts:/mlflow/artifacts"
        in services["mlflow"]["volumes"]
    )
    assert (
        "${AIRFLOW_PROJ_DIR:-.}/state/cloudbeaver/workspace:/opt/cloudbeaver/workspace"
        in services["cloudbeaver"]["volumes"]
    )
    assert (
        "${AIRFLOW_PROJ_DIR:-.}/state/db_diagrams:/opt/pricing/state/db_diagrams"
        in services["db-diagram-generator"]["volumes"]
    )
    assert (
        "${AIRFLOW_PROJ_DIR:-.}/state/db_diagrams:/usr/share/nginx/html:ro"
        in services["db-diagrams"]["volumes"]
    )

    rendered_volumes = str(compose["x-airflow-common"]["volumes"])
    rendered_volumes += str(services["mssql"]["volumes"])
    rendered_volumes += str(services["mlflow"]["volumes"])
    rendered_volumes += str(services["cloudbeaver"]["volumes"])
    rendered_volumes += str(services["db-diagram-generator"]["volumes"])
    rendered_volumes += str(services["db-diagrams"]["volumes"])
    assert "./state/" not in rendered_volumes


def test_mssql_init_creates_pricing_and_mlflow_databases():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    mssql_init = services["mssql-init"]
    init_command = "\n".join(str(part) for part in mssql_init["command"])
    mlflow_env = services["mlflow"]["environment"]

    assert mlflow_env["MLFLOW_DATABASE"] == "${MLFLOW_DATABASE:-MLflowTracking}"
    assert mssql_init["environment"]["MSSQL_DATABASE"] == "${MSSQL_DATABASE:-PricingLab}"
    assert mssql_init["environment"]["MLFLOW_DATABASE"] == "${MLFLOW_DATABASE:-MLflowTracking}"
    assert mssql_init["depends_on"]["mssql"] == {"condition": "service_healthy"}
    assert services["mlflow"]["depends_on"]["mssql-init"] == {
        "condition": "service_completed_successfully"
    }
    assert "pricing_pipeline.infra.db" in init_command
    assert "ensure_database" in init_command
    assert "settings.pricing_database" in init_command
    assert "settings.mlflow_database" in init_command
    assert "CREATE DATABASE [" not in init_command


def test_mlflow_serves_artifacts_through_http_proxy():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    mlflow = compose["services"]["mlflow"]
    mlflow_command = "\n".join(str(part) for part in mlflow["command"])

    assert "--serve-artifacts" in mlflow_command
    assert "--artifacts-destination /mlflow/artifacts" in mlflow_command
    assert '--allowed-hosts "$${MLFLOW_ALLOWED_HOSTS}"' in mlflow_command
    assert mlflow["environment"]["MLFLOW_ALLOWED_HOSTS"] == (
        "${MLFLOW_ALLOWED_HOSTS:-localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,mlflow,mlflow:5000}"
    )
    assert "--default-artifact-root /mlflow/artifacts" not in mlflow_command


def test_sql_database_names_can_be_overridden_from_environment():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert services["mssql-init"]["environment"]["MSSQL_DATABASE"] == (
        "${MSSQL_DATABASE:-PricingLab}"
    )
    assert services["mssql-init"]["environment"]["MLFLOW_DATABASE"] == (
        "${MLFLOW_DATABASE:-MLflowTracking}"
    )
    assert services["mlflow"]["environment"]["MLFLOW_DATABASE"] == (
        "${MLFLOW_DATABASE:-MLflowTracking}"
    )


def test_mlflow_backend_uri_is_built_with_encoded_odbc_connection():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")
    mlflow = compose["services"]["mlflow"]
    mlflow_command = "\n".join(str(part) for part in mlflow["command"])
    mlflow_env_text = str(mlflow["environment"])

    assert "MLFLOW_BACKEND_STORE_URI" not in mlflow["environment"]
    assert "sa:${MSSQL_PASSWORD" not in compose_text
    assert "mssql+pyodbc://sa:${MSSQL_PASSWORD" not in mlflow_env_text
    assert "pricing_pipeline.infra.db" in mlflow_command
    assert "build_sqlalchemy_url" in mlflow_command
    assert "PWD=" not in mlflow_command
    assert "Encrypt=" not in mlflow_command


def test_mlflow_and_mssql_init_can_import_pricing_pipeline_helpers():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    for name in ["mlflow", "mssql-init"]:
        service = services[name]
        assert (
            "${AIRFLOW_PROJ_DIR:-.}/pricing_pipeline:/opt/airflow/pricing_pipeline"
            in service["volumes"]
        )
        assert service["environment"]["PYTHONPATH"] == "/opt/airflow"


def test_db_diagram_profile_generates_and_serves_static_erds():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    generator = services["db-diagram-generator"]
    server = services["db-diagrams"]
    generator_command = "\n".join(str(part) for part in generator["command"])

    assert generator["profiles"] == ["diagrams"]
    assert server["profiles"] == ["diagrams"]
    assert "generate_db_diagrams.py" in generator_command
    assert "--schemas pricing mlops" in generator_command
    assert server["ports"] == ["8088:80"]
    assert server["depends_on"]["state-init"] == {"condition": "service_completed_successfully"}
    assert generator["depends_on"]["mssql"] == {"condition": "service_healthy"}


def test_airflow_image_uses_python_314_base():
    dockerfile = Path("airflow/Dockerfile").read_text(encoding="utf-8")
    assert "apache/airflow:3.2.1-python3.14" in dockerfile
    assert "msodbcsql18" in dockerfile
    assert '"apache-airflow==${AIRFLOW_VERSION}"' in dockerfile


def test_host_python_dependencies_do_not_install_airflow():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert "apache-airflow" not in pyproject
    assert "apache-airflow" not in requirements


def test_compose_does_not_use_env_file_required_false():
    compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "required: false" not in compose_text


def test_env_example_does_not_ship_invalid_fernet_placeholder():
    env_example = Path(".env.example").read_text(encoding="utf-8")

    assert "FERNET_KEY=airflow_fernet_key_change_me" not in env_example
    assert "FERNET_KEY=" in env_example
    assert "Fernet.generate_key()" in env_example


def test_superglm_runtime_dependency_is_pinned_to_commit():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    uv_lock = Path("uv.lock").read_text(encoding="utf-8")
    git_url = "https://github.com/StrudelDoodleS/superglm.git"
    git_sha = "25c06fc84b674bb2ee777ea99567772d8d57a17c"
    expected = (
        f"superglm[editor] @ git+{git_url}@{git_sha}"
    )

    assert expected in requirements
    assert f'"{expected}"' in pyproject
    assert "git+https://github.com/StrudelDoodleS/superglm.git\n" not in requirements
    assert f'git = "{git_url}?rev={git_sha}"' in uv_lock
    assert f'source = {{ git = "{git_url}?rev={git_sha}#{git_sha}" }}' in uv_lock
    assert (
        'name = "superglm"\n'
        'version = "0.12.0"\n'
        f'source = {{ git = "{git_url}?rev={git_sha}#{git_sha}" }}'
    ) in uv_lock
    assert "1072f7792cf255899fa6ba93579efd49a25ccdb4" not in requirements
    assert "1072f7792cf255899fa6ba93579efd49a25ccdb4" not in pyproject
    assert "1072f7792cf255899fa6ba93579efd49a25ccdb4" not in uv_lock


def test_workbench_artifact_dependency_is_direct():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "joblib" in requirements.splitlines()
    assert '"joblib"' in pyproject


def test_settings_resolves_relative_artifact_roots_from_project_root(
    monkeypatch,
    tmp_path,
):
    project_root = tmp_path / "project"
    launch_root = tmp_path / "launch"
    project_root.mkdir()
    launch_root.mkdir()
    monkeypatch.chdir(launch_root)

    settings = Settings.from_env(
        {
            "PRICING_PROJECT_ROOT": str(project_root),
            "RATING_EXPORT_ROOT": "state/rating_exports",
            "VALIDATION_SPLIT_ARTIFACT_ROOT": "state/validation_splits",
            "WORKBENCH_ARTIFACT_ROOT": "state/workbench",
        }
    )

    assert settings.rating_export_root == project_root / "state/rating_exports"
    assert settings.validation_split_artifact_root == (project_root / "state/validation_splits")
    assert settings.workbench_artifact_root == project_root / "state/workbench"


def test_settings_resolves_relative_artifact_roots_from_cwd_without_project_root(
    monkeypatch,
    tmp_path,
):
    launch_root = tmp_path / "launch"
    launch_root.mkdir()
    monkeypatch.chdir(launch_root)

    settings = Settings.from_env(
        {
            "RATING_EXPORT_ROOT": "rating",
            "VALIDATION_SPLIT_ARTIFACT_ROOT": "splits",
            "WORKBENCH_ARTIFACT_ROOT": "workbench",
        }
    )

    assert settings.rating_export_root == launch_root / "rating"
    assert settings.validation_split_artifact_root == launch_root / "splits"
    assert settings.workbench_artifact_root == launch_root / "workbench"


def test_settings_canonicalizes_absolute_artifact_roots(tmp_path):
    project_root = tmp_path / "project"
    absolute_roots = {
        "RATING_EXPORT_ROOT": tmp_path / "external" / ".." / "external-rating",
        "VALIDATION_SPLIT_ARTIFACT_ROOT": tmp_path / "external" / "../external-splits",
        "WORKBENCH_ARTIFACT_ROOT": tmp_path / "external" / "../external-workbench",
    }

    settings = Settings.from_env(
        {
            "PRICING_PROJECT_ROOT": str(project_root),
            **{name: str(path) for name, path in absolute_roots.items()},
        }
    )

    assert settings.rating_export_root == absolute_roots["RATING_EXPORT_ROOT"].resolve()
    assert (
        settings.validation_split_artifact_root
        == absolute_roots["VALIDATION_SPLIT_ARTIFACT_ROOT"].resolve()
    )
    assert settings.workbench_artifact_root == absolute_roots["WORKBENCH_ARTIFACT_ROOT"].resolve()


def test_settings_expands_user_for_project_and_artifact_roots(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    settings = Settings.from_env(
        {
            "PRICING_PROJECT_ROOT": "~/pricing/../project",
            "RATING_EXPORT_ROOT": "~/artifacts/../rating",
            "VALIDATION_SPLIT_ARTIFACT_ROOT": "state/../splits",
            "WORKBENCH_ARTIFACT_ROOT": "work/../workbench",
        }
    )

    assert settings.rating_export_root == home / "rating"
    assert settings.validation_split_artifact_root == home / "project/splits"
    assert settings.workbench_artifact_root == home / "project/workbench"


@pytest.mark.skipif(os.name == "nt", reason="POSIX/WSL-specific rejection")
@pytest.mark.parametrize(
    ("env_name", "windows_path"),
    [
        ("PRICING_PROJECT_ROOT", r"C:\pricing\project"),
        ("WORKBENCH_ARTIFACT_ROOT", "D:/pricing/workbench"),
        ("PRICING_PROJECT_ROOT", r"C:pricing\project"),
        ("WORKBENCH_ARTIFACT_ROOT", "D:pricing/workbench"),
    ],
)
def test_settings_rejects_windows_drive_paths_under_posix(
    tmp_path,
    env_name,
    windows_path,
):
    env = {"PRICING_PROJECT_ROOT": str(tmp_path / "project")}
    env[env_name] = windows_path

    with pytest.raises(ValueError, match="Windows drive-qualified path.*POSIX/WSL"):
        Settings.from_env(env)


def test_settings_repr_hides_database_secret():
    secrets = {
        "MSSQL_PASSWORD": "sentinel-mssql-secret",
    }

    rendered = repr(Settings.from_env(secrets))

    assert all(secret not in rendered for secret in secrets.values())


def test_generated_runtime_files_use_portable_exception_tuple_syntax():
    for path in [
        Path("scripts/no_docker_services.py"),
    ]:
        source = path.read_text(encoding="utf-8")
        assert "except TypeError, ValueError:" not in source
        assert "except OSError, ProcessLookupError:" not in source
        assert "except AttributeError, curses.error:" not in source


def test_deploy_api_publishes_model_scoped_deployment_history():
    deployer = Path("pricing_pipeline/publishing/deployment.py").read_text(encoding="utf-8")

    assert "model_id" in deployer
    assert "PRICING_MODEL_DEPLOYMENT" in deployer
    assert "effective_to_ts = SYSUTCDATETIME()" in deployer
    assert "deployment_slot" in deployer
    assert "PRICING_PACKAGE_POINTER" in deployer
    assert "pointer_name = src.pointer_name" in deployer
    assert "model_id = src.model_id" in deployer
    assert "deployment_note" in deployer


def test_rating_package_loader_assigns_feature_level_ids_in_numeric_order():
    loader = Path("pricing_pipeline/publishing/package_writer.py").read_text(encoding="utf-8")

    assert "ORDER BY" in loader
    assert "ls.level_set_id" in loader
    assert "s.order_index" in loader
    assert "s.lower_bound" in loader
    assert "s.upper_bound" in loader
