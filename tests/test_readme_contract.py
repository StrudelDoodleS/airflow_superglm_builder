from pathlib import Path


README_CONTRACT_STRINGS = [
    "state/",
    "docker compose down -v",
    "No-Docker Work Quickstart",
    "Airflow 3.2.1",
    "MLflow",
    "PRICING_SCHEMA_DIR=db/migrations",
    "scripts/apply_schema.py",
    "scripts/run_local_pipeline.sh",
    "scripts/no_docker_services.py",
    "scripts/start_no_docker_stack.sh",
    "scripts/start_airflow_local.py",
    "scripts/start_mlflow_local.py",
    "Adding Models",
    "DatasetSpec",
    "pricing_models/<model_name>/",
    "pricing_pipeline/data/datasets.py",
    "pricing_pipeline/infra/db.py",
    "pricing_mtpl_frequency",
    "--model-key MTPL_FREQ",
]


def test_readme_documents_local_pipeline_contract():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in README_CONTRACT_STRINGS:
        assert expected in readme
