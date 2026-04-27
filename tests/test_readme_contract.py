from pathlib import Path


README_CONTRACT_STRINGS = [
    "state/",
    "docker compose down -v",
    "Airflow 3.2.1",
    "MLflow",
    "scripts/run_local_pipeline.sh",
]


def test_readme_documents_local_pipeline_contract():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in README_CONTRACT_STRINGS:
        assert expected in readme
