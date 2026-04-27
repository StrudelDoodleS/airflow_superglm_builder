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
        "mssql",
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
    assert MSSQL_PASSWORD_DEFAULT in mlflow_env["MLFLOW_BACKEND_STORE_URI"]
    assert "AirflowSuperGLM!2026" not in Path("docker-compose.yml").read_text(
        encoding="utf-8"
    )


def test_airflow_init_prepares_project_state_directories():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    airflow_init = compose["services"]["airflow-init"]
    command = "\n".join(str(part) for part in airflow_init["command"])

    assert "${AIRFLOW_PROJ_DIR:-.}:/sources" in airflow_init["volumes"]
    for state_path in STATE_PATHS:
        assert state_path in command


def test_airflow_image_uses_python_314_base():
    dockerfile = Path("airflow/Dockerfile").read_text(encoding="utf-8")
    assert "apache/airflow:3.2.1-python3.14" in dockerfile
    assert "msodbcsql18" in dockerfile
    assert '"apache-airflow==${AIRFLOW_VERSION}"' in dockerfile


def test_compose_does_not_use_env_file_required_false():
    compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "required: false" not in compose_text
