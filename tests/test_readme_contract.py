from pathlib import Path


README_CONTRACT_STRINGS = [
    "state/",
    "docker compose down -v",
    "No-Docker Work Quickstart",
    "Airflow 3.2.1",
    "MLflow",
    "scripts/run_local_pipeline.sh",
    "scripts/no_docker_services.py",
    "scripts/start_no_docker_runtime.sh",
    "scripts/start_no_docker_stack.sh",
    "scripts/start_airflow_local.py",
    "scripts/start_mlflow_local.py",
]


def test_readme_documents_local_pipeline_contract():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in README_CONTRACT_STRINGS:
        assert expected in readme
