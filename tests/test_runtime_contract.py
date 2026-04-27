from pathlib import Path

import yaml


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


def test_airflow_image_uses_python_314_base():
    dockerfile = Path("airflow/Dockerfile").read_text(encoding="utf-8")
    assert "apache/airflow:3.2.1-python3.14" in dockerfile
    assert "msodbcsql18" in dockerfile
