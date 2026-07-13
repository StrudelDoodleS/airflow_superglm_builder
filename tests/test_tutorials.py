import json
from pathlib import Path


TUTORIAL_NOTEBOOK = Path("tutorials/basic_sql_etl_and_schema_walkthrough.ipynb")
TUTORIAL_DDL = Path("tutorials/schema/pricing_useful_tables_ddl.sql")
REFERENCE_DDL = Path("docs/pricing_useful_tables_ddl.sql")
CANDIDATE_WORKBENCH_NOTEBOOK = Path("tutorials/scaffolded_candidate_workbench.ipynb")


def _notebook_text() -> str:
    notebook = json.loads(TUTORIAL_NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") in {"markdown", "code"}
    )


def test_tutorial_ddl_is_current_erd_reference_copy():
    assert TUTORIAL_DDL.read_text(encoding="utf-8") == REFERENCE_DDL.read_text(
        encoding="utf-8"
    )


def test_basic_sql_etl_notebook_teaches_connection_transform_and_load_pattern():
    text = _notebook_text()

    for expected in [
        "Basic SQL ETL And Schema Walkthrough",
        "REPO_ROOT",
        "sys.path.insert",
        "get_engine(settings, database=source_database)",
        "pd.read_sql_query",
        "transform_source_rows",
        "to_sql",
        "target_schema",
        "target_table",
        "Entra token",
    ]:
        assert expected in text


def test_basic_sql_etl_notebook_explains_schema_and_erd_files():
    text = _notebook_text()

    for expected in [
        "tutorials/schema/pricing_useful_tables_ddl.sql",
        "docs/pricing_useful_tables_full_ddl.sql",
        "parse_ddl_schema",
        "schema_tables",
        "schema_columns",
        "schema_foreign_keys",
        "raw.FREMTPL_RAW",
        "mlops.DATASET_MANIFEST",
        "mlops.CV_SPLIT_SET",
        "mlops.MODEL_RUN",
        "pricing.MODEL",
        "pricing.RATE_PACKAGE",
        "pricing_runtime.V_COMPILED_RATE_CELL",
        "pricing.PREDICT_CURRENT_RATE",
    ]:
        assert expected in text


def test_basic_sql_etl_notebook_shows_result_set_transform_model_and_load_shape():
    text = _notebook_text()

    for expected in [
        "Notebook DAG: Offline Load, Train, Push, Deploy",
        "sqlite3",
        "attach_pricing_lab_schemas",
        "create_offline_schema",
        "seed_source_rows",
        "source_result_set",
        "transformed_result_set",
        "train_superglm_revision",
        "persist_model_revision",
        "deploy_rate_package",
        "create_manual_uplift_package",
        "current_deployment",
        "deployment_history",
        "MODEL_SCORE",
        "raw.FREMTPL_RAW",
        "mlops.MODEL_RUN",
        "pricing.RATE_PACKAGE",
        "pricing.MODEL_DEPLOYMENT",
        "SQLite is only the offline stand-in",
        "Azure SQL with Entra auth",
        "Postgres",
        "DuckDB",
        "DDL dialect",
    ]:
        assert expected in text


def test_basic_sql_etl_notebook_explains_model_revision_and_deployment_rules():
    text = _notebook_text()

    for expected in [
        "What Counts As A Model Revision",
        "Changed feature list",
        "Changed training SQL or dataset window",
        "Changed preprocessing",
        "Changed model class or hyperparameters",
        "Manual 10% Uplift Package",
        "deployed package changes",
        "historical packages remain",
    ]:
        assert expected in text


def test_candidate_workbench_notebook_uses_runtime_facade_and_no_sql_ids():
    notebook = json.loads(CANDIDATE_WORKBENCH_NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") in {"markdown", "code"}
    )

    assert "Workbench.from_runtime()" in source
    assert "workbench.models()" in source
    assert "workbench.candidates(MODEL_NAME)" in source
    assert "candidate.editor()" in source
    assert "candidate.submit_edits(" in source
    assert "submission.status()" in source
    assert "submission.request_deployment(" in source
    assert "candidate.close_editor()" in source
    for sql_identifier in [
        "rate_package_id",
        "model_run_id",
        "manifest_id",
        "split_set_id",
    ]:
        assert sql_identifier not in source
    assert "create_engine(" not in source
    assert "AIRFLOW_API_TOKEN" not in source


def test_candidate_workbench_notebook_requires_safe_explicit_analyst_inputs():
    notebook = json.loads(CANDIDATE_WORKBENCH_NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") in {"markdown", "code"}
    )

    assert 'MODEL_NAME = ""' in source
    assert "if not MODEL_NAME.strip():" in source
    assert source.index("if not MODEL_NAME.strip():") < source.index(
        "workbench.candidates(MODEL_NAME)"
    )
    assert "history.empty" in source
    assert "No candidate history" in source
    assert 'history["Editor"].eq("Ready")' in source
    assert "editor_ready.empty" in source
    assert "PACKAGE_VERSION = None" in source
    assert "history.iloc[0]" not in source
    assert source.index("if PACKAGE_VERSION is None:") < source.index("workbench.open(")
    assert 'EDIT_REASON = ""' in source
    assert "if not EDIT_REASON.strip():" in source
    assert "reason=EDIT_REASON" in source
    assert source.index("if not EDIT_REASON.strip():") < source.index(
        "candidate.submit_edits("
    )
    assert "Explain the market" not in source
    assert "close_editor" in source
