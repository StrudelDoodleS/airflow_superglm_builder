from pathlib import Path


README_CONTRACT_STRINGS = [
    "state/",
    "docker compose down -v",
    "No-Docker Work Quickstart",
    "Airflow 3.2.1",
    "MLflow",
    "PRICING_ENABLE_MLFLOW=false",
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
    "SQL Prediction Validation",
    "scripts/validate_sql_prediction_against_superglm.py",
    "pricing.PREDICT_CURRENT_RATE",
    "predict(X, offset=np.log(exposure))",
]

RATE_PACKAGE_LIFECYCLE_STRINGS = [
    "model.toml",
    "ModelPublisher",
    "pricing_deploy_rate_package",
    "model_key",
    "rate_package_id",
    "package_version",
    "deployment_slot",
    "deployment_reason",
    "deployed_by",
    "parent_rate_package_id",
    "PUBLISHED",
]

RATE_PACKAGE_DEPLOY_CONTRACT_STRINGS = [
    "The deploy run requires `model_key`, exactly one",
    "`rate_package_id` or `package_version`",
    "`deployed_by`, and `deployment_reason`",
    "`deployment_slot` is optional",
    "defaults to the model config deployment slot",
]


def _readme_section(readme: str, heading: str) -> str:
    start_marker = f"## {heading}"
    start = readme.index(start_marker)
    next_heading = readme.find("\n## ", start + len(start_marker))
    if next_heading == -1:
        return readme[start:]
    return readme[start:next_heading]


def test_readme_documents_local_pipeline_contract():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in README_CONTRACT_STRINGS:
        assert expected in readme


def test_readme_documents_rate_package_lifecycle_contract():
    readme = Path("README.md").read_text(encoding="utf-8")
    lifecycle_section = _readme_section(readme, "Rate Package Lifecycle")

    for expected in RATE_PACKAGE_LIFECYCLE_STRINGS:
        assert expected in lifecycle_section
    for expected in RATE_PACKAGE_DEPLOY_CONTRACT_STRINGS:
        assert expected in lifecycle_section
    assert "manual revision" in lifecycle_section.lower()
