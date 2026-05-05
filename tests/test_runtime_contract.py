import re
from pathlib import Path

import yaml


MSSQL_PASSWORD_DEFAULT = "${MSSQL_PASSWORD:-YourStrong(!)Password123}"
STATE_PATHS = [
    "/sources/state/mssql/data",
    "/sources/state/mlflow/artifacts",
    "/sources/state/rating_exports",
    "/sources/state/cv_splits",
    "/sources/state/db_diagrams",
    "/sources/state/cloudbeaver/workspace",
]


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
    assert "AirflowSuperGLM!2026" not in Path("docker-compose.yml").read_text(
        encoding="utf-8"
    )


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
    assert (
        "${AIRFLOW_PROJ_DIR:-.}/state/mssql/data:/var/opt/mssql"
        in services["mssql"]["volumes"]
    )
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
    assert "pricing_pipeline.db" in init_command
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
    assert "--allowed-hosts \"$${MLFLOW_ALLOWED_HOSTS}\"" in mlflow_command
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
    assert "pricing_pipeline.db" in mlflow_command
    assert "build_sqlalchemy_url" in mlflow_command
    assert "PWD=" not in mlflow_command
    assert "Encrypt=" not in mlflow_command


def test_mlflow_and_mssql_init_can_import_pricing_pipeline_helpers():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    for name in ["mlflow", "mssql-init"]:
        service = services[name]
        assert "${AIRFLOW_PROJ_DIR:-.}/pricing_pipeline:/opt/airflow/pricing_pipeline" in service["volumes"]
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
    assert "--schemas pricing" in generator_command
    assert server["ports"] == ["8088:80"]
    assert server["depends_on"]["state-init"] == {"condition": "service_completed_successfully"}
    assert generator["depends_on"]["mssql"] == {"condition": "service_healthy"}


def test_airflow_services_can_import_project_package_and_hide_examples():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    common_env = compose["x-airflow-common"]["environment"]
    run_script = Path("scripts/run_local_pipeline.sh").read_text(encoding="utf-8")
    cleanup_script = Path("scripts/cleanup_airflow_examples.py").read_text(
        encoding="utf-8"
    )

    assert common_env["PYTHONPATH"] == "/opt/airflow"
    assert common_env["AIRFLOW__CORE__LOAD_EXAMPLES"] == "false"
    assert "cleanup_airflow_examples.py" in run_script
    assert "airflow dags list-import-errors" in run_script
    assert "airflow dags trigger \"${DAG_ID}\"" in run_script
    assert "bundle_name" in cleanup_script
    assert "example_dags" in cleanup_script


def test_airflow_common_env_propagates_runtime_overrides():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    common_env = compose["x-airflow-common"]["environment"]

    assert common_env["MLFLOW_TRACKING_URI"] == (
        "${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
    )
    assert common_env["RATING_EXPORT_ROOT"] == (
        "${RATING_EXPORT_ROOT:-/opt/pricing/state/rating_exports}"
    )


def test_readme_documents_db_diagram_commands():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Database Diagrams" in readme
    assert "docker compose --profile diagrams run --rm db-diagram-generator" in readme
    assert "docker compose --profile diagrams up -d db-diagrams" in readme
    assert "http://localhost:8088" in readme


def test_airflow_image_uses_python_314_base():
    dockerfile = Path("airflow/Dockerfile").read_text(encoding="utf-8")
    assert "apache/airflow:3.2.1-python3.14" in dockerfile
    assert "msodbcsql18" in dockerfile
    assert '"apache-airflow==${AIRFLOW_VERSION}"' in dockerfile


def test_host_python_dependencies_pin_airflow_321():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert '"apache-airflow==3.2.1"' in pyproject
    assert "apache-airflow==3.2.1" in requirements


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
    pinned_ref = (
        r"superglm @ git\+https://github\.com/StrudelDoodleS/superglm\.git@"
        r"[0-9a-f]{40}"
    )

    assert re.search(pinned_ref, requirements)
    assert re.search(pinned_ref, pyproject)
    assert "git+https://github.com/StrudelDoodleS/superglm.git\n" not in requirements


def test_rating_package_loader_publishes_model_scoped_deployment_history():
    loader = Path("scripts/load_staging_to_rating_package.py").read_text(
        encoding="utf-8"
    )

    assert "model_id" in loader
    assert "PRICING_MODEL_DEPLOYMENT" in loader
    assert "effective_to_ts = SYSUTCDATETIME()" in loader
    assert "deployment_slot" in loader
    assert "PRICING_PACKAGE_POINTER" in loader
    assert "pointer_name = src.pointer_name" in loader
    assert "model_id = src.model_id" in loader


def test_rating_package_loader_assigns_feature_level_ids_in_numeric_order():
    loader = Path("scripts/load_staging_to_rating_package.py").read_text(
        encoding="utf-8"
    )

    assert "ORDER BY" in loader
    assert "ls.level_set_id" in loader
    assert "s.order_index" in loader
    assert "s.lower_bound" in loader
    assert "s.upper_bound" in loader
