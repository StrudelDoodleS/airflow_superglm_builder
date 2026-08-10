from pathlib import Path


def _readme() -> str:
    return Path("README.md").read_text(encoding="utf-8")


def test_readme_documents_the_notebook_first_contract():
    readme = _readme()

    for expected in (
        "notebook-first workflow",
        "scripts/scaffold_pricing_model.py",
        "01_data_ingestion.ipynb",
        "02_model_training.ipynb",
        "03_model_editor.ipynb",
        "04_model_deployment.ipynb",
        "99_scratch_work.ipynb",
        "PricingModelSpec",
        "register_model",
        "build_candidate",
        "publish_candidate",
        "open_candidate",
        "publish_edits",
        "deploy_package",
        "no generated training module",
        "Analysts never type a model or",
    ):
        assert expected in readme

    for retired_workflow in (
        "build_pricing_model_dag",
        "ModelPublisher",
        "manual revision",
        "run_local_pipeline.sh",
        "run_mtpl_frequency_custom.py",
    ):
        assert retired_workflow not in readme


def test_readme_documents_local_and_guarded_remote_writes():
    readme = _readme()

    for expected in (
        'DATABASE_MODE = "local"',
        'DATABASE_MODE = "remote"',
        'RUNTIME_MODULE = "work_runtime.database"',
        "EXPECTED_REMOTE_DATABASE",
        "ALLOW_REMOTE_WRITES",
        "SELECT DB_NAME()",
        "persistent SQLite databases",
        "does not deploy a live package",
        "Do not commit\nserver names, tokens, passwords",
    ):
        assert expected in readme


def test_readme_documents_automatic_audit_evidence_and_lineage():
    readme = _readme()

    for expected in (
        "data-as-at",
        "primary keys",
        "validation configuration",
        "model source checksum",
        "candidate bundle",
        "model-run, fold, and scoped metrics",
        "editor parent lineage",
        "champion snapshot",
        "SQL database is the audit source of truth",
        "current notebook workflow does not create or log MLflow",
    ):
        assert expected in readme


def test_readme_documents_data_version_and_python_side_duplicate_preflight():
    readme = _readme()

    for expected in (
        "Data-as-at is a dataset version",
        "manifest_signature_sha256",
        "model_equivalence_sha256",
        "10 decimal places",
        "read-only lookup",
        "does not write\n`pricing_stg.STG_RATING_EXPORT`",
        "deduplicated=True",
        "RAW`, `ROUTINE_EDIT`, and `EDITOR_EDIT",
        "before SQL staging",
        "V_MODEL_CANDIDATE_RELATIVITY",
        "V_CURRENT_DEPLOYED_RELATIVITY",
    ):
        assert expected in readme


def test_readme_documents_inspection_editor_and_audit_view_flow():
    readme = _readme()

    for expected in (
        "candidate.metrics",
        "EditorSession.from_model",
        "editor_session.to_model",
        "editor_session=editor_session",
        "V_MODEL_VALIDATION_SPLIT",
        "V_MODEL_VALIDATION_SUMMARY",
        "V_FINAL_MODEL_RELATIVITY",
        "do not inherit validation metrics",
        "continuous spline SQL export",
        "export_level_groupings",
        "load_level_groupings",
        "actual\n`dict[str, LevelGrouping]` Python objects",
        "Grouping is Python model behaviour",
        "compact run keys",
        "Windows Explorer",
    ):
        assert expected in readme


def test_readme_documents_explicit_independent_offset_and_weight_inputs():
    readme = _readme()

    for expected in (
        'frame["term_offset"] = np.log(frame["term"] / 12.0)',
        'offset_column="term_offset"',
        'offset_source_column="term"',
        'offset_label="log(term / 12)"',
        'sample_weight_column="model_weight"',
        'export_weight_column="rating_table_weight"',
    ):
        assert expected in readme

    assert "exposure_column=" not in readme


def test_readme_is_concise_enough_to_be_an_entry_point():
    assert len(_readme().splitlines()) < 350
