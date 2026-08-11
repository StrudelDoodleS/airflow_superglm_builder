from __future__ import annotations

import re
from pathlib import Path

from pricing_pipeline import notebook

ROOT_README = Path("README.md")
NOTEBOOK_GUIDE = Path("docs/notebooks/README.md")
SQL_GUIDE = Path("docs/sql/README.md")
SCRIPT_INDEX = Path("scripts/README.md")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_root_readme_is_a_concise_entry_point():
    readme = _read(ROOT_README)

    for expected in (
        "notebook-first",
        "scripts/scaffold_pricing_model.py",
        "pricing_scaffold.example.toml",
        "01_data_ingestion.ipynb",
        "02_model_training.ipynb",
        "03_model_editor.ipynb",
        "04_model_deployment.ipynb",
        "99_scratch_work.ipynb",
        "Data-as-at",
        "docs/notebooks/README.md",
        "docs/sql/README.md",
        "scripts/README.md",
        "--expected-database PricingAudit",
    ):
        assert expected in readme

    assert len(readme.splitlines()) < 125


def test_notebook_guide_documents_boundaries_and_public_functions():
    guide = _read(NOTEBOOK_GUIDE)

    for expected in (
        "Workflow boundaries",
        "PricingModelSpec",
        "connect(...)",
        "save_model_frame",
        "inspect_model_frame",
        "load_model_frame",
        "register_model",
        "build_candidate",
        "publish_candidate",
        "load_registered_model",
        "list_candidate_versions",
        "open_candidate",
        "publish_edits",
        "deploy_package",
        "data_as_of_column",
        "SELECT DB_NAME()",
        "ALLOW_REMOTE_WRITES",
    ):
        assert expected in guide


def test_documented_notebook_helpers_are_exported_by_the_public_module():
    expected = {
        "connect",
        "save_model_frame",
        "inspect_model_frame",
        "load_model_frame",
        "register_model",
        "build_candidate",
        "publish_candidate",
        "load_registered_model",
        "list_candidate_versions",
        "open_candidate",
        "export_level_groupings",
        "load_level_groupings",
        "inspect_level_groupings",
        "apply_level_groupings",
        "publish_edits",
        "deploy_package",
    }

    assert expected <= set(notebook.__all__)


def test_notebook_guide_documents_python_groupings_and_duplicate_preflight():
    guide = _read(NOTEBOOK_GUIDE)

    for expected in (
        "export_level_groupings",
        "load_level_groupings",
        "apply_level_groupings",
        "dict[str, LevelGrouping]",
        "Grouping is Python model behaviour",
        "before SQL staging",
        "model_equivalence_sha256",
        "deduplicated=True",
        "RAW",
        "ROUTINE_EDIT",
        "EDITOR_EDIT",
        "10 decimal places",
    ):
        assert expected in guide


def test_sql_guide_documents_schema_relationships_and_operational_objects():
    guide = _read(SQL_GUIDE)

    assert guide.count("```mermaid") == 3
    for expected in (
        "pricing.DATASET_MANIFEST",
        "pricing.DATASET_COLUMN",
        "pricing.CV_SPLIT_SET",
        "pricing.MODEL_RUN",
        "mlops.MODEL_RUN_DATASET",
        "mlops.MODEL_RUN_SPLIT_SET",
        "pricing.PRICING_RATE_PACKAGE",
        "pricing.PRICING_TERM",
        "pricing.PRICING_RATE_CELL",
        "pricing.PRICING_MODEL_DEPLOYMENT",
        "pricing_stg.STG_RATING_EXPORT",
        "PRICING_PACKAGE_POINTER",
        "THROW 51000",
        "THROW 51001",
    ):
        assert expected in guide


def test_sql_guide_documents_triggers_views_and_migration_decision():
    guide = _read(SQL_GUIDE)

    for expected in (
        "TR_PRICING_RATE_PACKAGE_IMMUTABLE_UPDATE_DELETE",
        "TR_PRICING_MODEL_DEPLOYMENT_PACKAGE_GUARD",
        "V_FINAL_MODEL_RELATIVITY",
        "V_MODEL_CANDIDATE_RELATIVITY",
        "V_CURRENT_DEPLOYED_RELATIVITY",
        "V_MODEL_LINEAGE_REDUNDANCY_CHECK",
        "PREDICT_RATE_PACKAGE",
        "PREDICT_CURRENT_RATE",
        "scripts/apply_schema.py",
        "scripts/reset_remote_pricing_schema.py",
        "--i-understand-this-drops-pricing-objects",
        "Do not edit an already-applied migration",
        "conceptual ERD extracts",
    ):
        assert expected in guide


def test_sql_guide_names_every_current_trigger_view_and_procedure():
    guide = _read(SQL_GUIDE)
    migration_sql = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(Path("db/migrations").glob("V*.sql"))
    )
    object_names = {
        match.group(2)
        for match in re.finditer(
            r"CREATE\s+OR\s+ALTER\s+(VIEW|TRIGGER|PROCEDURE)\s+pricing\.([A-Z0-9_]+)",
            migration_sql,
            flags=re.IGNORECASE,
        )
    }

    assert object_names
    assert all(name in guide for name in object_names)


def test_script_index_categorizes_every_top_level_command():
    index = _read(SCRIPT_INDEX)
    expected_scripts = {
        path.name
        for path in Path("scripts").iterdir()
        if path.is_file() and path.suffix in {".py", ".sh"}
    }

    assert expected_scripts
    assert all(script_name in index for script_name in expected_scripts)
    for heading in (
        "Notebook workspace",
        "SQL schema and inspection",
        "Demo data",
        "Local development services",
    ):
        assert heading in index


def test_retired_workflow_is_not_presented_as_current():
    current_docs = "\n".join(
        _read(path) for path in (ROOT_README, NOTEBOOK_GUIDE, SQL_GUIDE, SCRIPT_INDEX)
    )
    for retired in (
        "build_pricing_model_dag",
        "run_local_pipeline.sh",
        "run_mtpl_frequency_custom.py",
    ):
        assert retired not in current_docs
