from pathlib import Path

import yaml


MSSQL_PASSWORD_DEFAULT = "${MSSQL_PASSWORD:-YourStrong(!)Password123}"
STATE_PATHS = [
    "/sources/state/mssql/data",
    "/sources/state/mlflow/artifacts",
    "/sources/state/rating_exports",
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

    rendered_volumes = str(compose["x-airflow-common"]["volumes"])
    rendered_volumes += str(services["mssql"]["volumes"])
    rendered_volumes += str(services["mlflow"]["volumes"])
    rendered_volumes += str(services["cloudbeaver"]["volumes"])
    assert "./state/" not in rendered_volumes


def test_mssql_init_creates_pricing_and_mlflow_databases():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    mssql_init = services["mssql-init"]
    init_command = "\n".join(str(part) for part in mssql_init["command"])
    mlflow_env = services["mlflow"]["environment"]

    assert mlflow_env["MLFLOW_DATABASE"] == "${MLFLOW_DATABASE:-MLflowTracking}"
    assert "PricingLab" in init_command
    assert "MLflowTracking" in init_command
    assert mssql_init["depends_on"]["mssql"] == {"condition": "service_healthy"}
    assert services["mlflow"]["depends_on"]["mssql-init"] == {
        "condition": "service_completed_successfully"
    }


def test_mlflow_serves_artifacts_through_http_proxy():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    mlflow_command = "\n".join(
        str(part) for part in compose["services"]["mlflow"]["command"]
    )

    assert "--serve-artifacts" in mlflow_command
    assert "--artifacts-destination /mlflow/artifacts" in mlflow_command
    assert "--default-artifact-root /mlflow/artifacts" not in mlflow_command


def test_sql_database_names_can_be_overridden_from_environment():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    init_command = "\n".join(str(part) for part in services["mssql-init"]["command"])

    assert "${MSSQL_DATABASE:-PricingLab}" in init_command
    assert "${MLFLOW_DATABASE:-MLflowTracking}" in init_command
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
    assert "quote_plus" in mlflow_command or "odbc_connect" in mlflow_command
    assert "odbc_connect=" in mlflow_command


def test_airflow_common_env_propagates_runtime_overrides():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    common_env = compose["x-airflow-common"]["environment"]

    assert common_env["MLFLOW_TRACKING_URI"] == (
        "${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
    )
    assert common_env["RATING_EXPORT_ROOT"] == (
        "${RATING_EXPORT_ROOT:-/opt/pricing/state/rating_exports}"
    )


def test_airflow_image_uses_python_314_base():
    dockerfile = Path("airflow/Dockerfile").read_text(encoding="utf-8")
    assert "apache/airflow:3.2.1-python3.14" in dockerfile
    assert "msodbcsql18" in dockerfile
    assert '"apache-airflow==${AIRFLOW_VERSION}"' in dockerfile


def test_compose_does_not_use_env_file_required_false():
    compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "required: false" not in compose_text
