from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

MODEL_DIR = Path("pricing_models/mtpl_frequency")
WORKFLOW_NAMES = (
    "01_data_ingestion.ipynb",
    "02_model_training.ipynb",
    "03_model_editor.ipynb",
    "04_manual_adjustment.ipynb",
    "05_model_deployment.ipynb",
    "99_scratch_work.ipynb",
)
STRICT_NOTEBOOK_NAME = re.compile(r"^\d{2}_[a-z0-9]+(?:_[a-z0-9]+)*\.ipynb$")


def _notebook(name: str) -> dict:
    return json.loads((MODEL_DIR / name).read_text(encoding="utf-8"))


def _source(name: str) -> str:
    notebook = _notebook(name)
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def _code_cells(name: str) -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in _notebook(name)["cells"]
        if cell["cell_type"] == "code"
    ]


def test_reference_model_has_exact_six_notebook_workflow():
    assert sorted(path.name for path in MODEL_DIR.glob("*.ipynb")) == sorted(WORKFLOW_NAMES)


def test_every_repository_notebook_uses_strict_name_compiles_and_has_no_output():
    notebook_paths = sorted(Path(".").glob("**/*.ipynb"))
    assert notebook_paths

    for path in notebook_paths:
        assert STRICT_NOTEBOOK_NAME.fullmatch(path.name), path
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(source, f"{path}:cell-{index}", "exec")
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_mtpl_ingestion_owns_source_transform_and_verified_handoff():
    source = _source("01_data_ingestion.ipynb")

    assert "connect(" in source
    assert "load_fremtpl_raw" in source
    assert "pd.read_sql_query(" in source
    assert 'frame["LogDensity"]' in source
    assert 'frame["LogExposure"]' in source
    assert 'DATA_AS_OF = "2026-06-30"' in source
    assert 'frame["data_as_of"] = DATA_AS_OF' in source
    assert "save_model_frame(" in source
    assert "REPLACE_MODEL_FRAME = False" in source
    assert "build_candidate(" not in source
    assert "publish_candidate(" not in source

    source_cell = next(
        cell
        for cell in _notebook("01_data_ingestion.ipynb")["cells"]
        if "SOURCE_SQL" in "".join(cell.get("source", []))
    )
    source_code = "".join(source_cell["source"])
    assert 'if pricing.mode == "local":' in source_code
    assert source_code.count("load_fremtpl_raw(") == 1
    assert "load_fremtpl_raw(pricing.engine, replace=REFRESH_LOCAL_RAW)" in source_code


def test_mtpl_training_has_untouched_raw_and_optional_routine_model():
    source = _source("02_model_training.ipynb")
    lowered = source.lower()

    assert "load_model_frame(" in source
    assert "PricingModelSpec(" in source
    assert "register_model(" in source
    assert "ValidationSplitConfig.kfold(" in source
    assert 'data_as_of_column="data_as_of"' in source
    assert "RAW_FEATURES = {" in source
    assert "raw_superglm_model = SuperGLM(" in source
    assert 'model_kind="RAW"' in source
    assert "raw_candidate.metrics" in source
    assert "publish_candidate(" in source
    assert "load_level_groupings(" in source
    assert "apply_level_groupings(" in source
    assert "inspect_level_groupings(" in source
    assert "ROUTINE_EDIT_CONFIGURED = bool(LEVEL_GROUPINGS)" in source
    assert "LevelGrouping(" not in source
    assert 'model_kind="ROUTINE_EDIT"' in source
    assert "EditorSession" not in source
    assert "open_candidate(" not in source
    assert "model.toml" not in lowered
    assert "airflow" not in lowered

    raw_cell = next(
        cell for cell in _code_cells("02_model_training.ipynb") if "RAW_FEATURES" in cell
    )
    assert "grouping=" not in raw_cell


def test_mtpl_editor_selects_label_version_or_latest_before_editing():
    cells = _code_cells("03_model_editor.ipynb")
    source = "\n".join(cells)

    assert 'MODEL_LABEL = "Motor frequency"' in source
    assert "PACKAGE_VERSION = None" in source
    assert "load_registered_model(" in source
    assert "list_candidate_versions(" in source
    assert 'versions.iloc[0]["Package"]' in source
    assert "open_candidate(" in source
    assert "EditorSession.from_model(" in source
    assert "editor_session.widget()" in source
    assert "editor_session.to_model()" in source
    assert "publish_edits(" in source
    assert "edited.model_kind" in source
    assert ".editor()" not in source

    editor_index = next(i for i, cell in enumerate(cells) if "EditorSession.from_model(" in cell)
    preview_index = next(i for i, cell in enumerate(cells) if "editor_session.to_model()" in cell)
    publish_index = next(i for i, cell in enumerate(cells) if "publish_edits(" in cell)
    assert editor_index < preview_index < publish_index


def test_mtpl_editor_reports_no_published_candidates_cleanly():
    selection_cell = next(
        cell
        for cell in _code_cells("03_model_editor.ipynb")
        if 'raise LookupError("No candidate package versions were found.")' in cell
    )

    with pytest.raises(LookupError, match="No candidate package versions"):
        exec(  # noqa: S102 - execute the checked-in notebook cell under an empty result
            compile(selection_cell, "03_model_editor.ipynb:empty-selection", "exec"),
            {
                "versions": pd.DataFrame(
                    columns=["Package"],
                ),
                "PACKAGE_VERSION": None,
            },
        )


def test_mtpl_manual_adjustment_is_replayable_and_explicitly_deployable():
    cells = _code_cells("04_manual_adjustment.ipynb")
    source = "\n".join(cells)

    assert 'SOURCE_SELECTOR = "deployed"' in source
    assert "PACKAGE_VERSION = None" in source
    assert "list_candidate_versions(" in source
    assert "open_deployed_candidate(" in source
    assert "ManualAdjustmentPolicy.from_rows(" in source
    assert "apply_manual_adjustment_policy(" in source
    assert "manual_adjustment_policy_from_candidate(" in source
    assert "publish_manual_adjustment(" in source
    assert "POLICY_SOURCE_PACKAGE_VERSION = None" in source
    assert "DEPLOY_AFTER_PUBLISH = False" in source
    assert "deploy_package(" in source

    preview_index = next(
        i for i, cell in enumerate(cells) if "apply_manual_adjustment_policy(" in cell
    )
    publish_index = next(i for i, cell in enumerate(cells) if "publish_manual_adjustment(" in cell)
    deploy_index = next(i for i, cell in enumerate(cells) if "if DEPLOY_AFTER_PUBLISH:" in cell)
    assert preview_index < publish_index < deploy_index


def test_mtpl_deployment_selects_only_published_sql_candidate():
    source = _source("05_model_deployment.ipynb")

    assert "load_registered_model(" in source
    assert "list_candidate_versions(" in source
    assert "technical=True" in source
    assert 'versions["package_status"]' in source
    assert '.eq("PUBLISHED")' in source
    assert "PACKAGE_VERSION = None" in source
    assert "open_candidate(" in source
    assert "deploy_package(" in source
    assert "DEPLOYMENT_REASON" in source


def test_mtpl_deployment_reports_no_published_candidates_cleanly():
    selection_cell = next(
        cell
        for cell in _code_cells("05_model_deployment.ipynb")
        if 'raise LookupError("No published candidate packages were found.")' in cell
    )

    with pytest.raises(LookupError, match="No published candidate packages"):
        exec(  # noqa: S102 - execute the checked-in notebook cell under an empty result
            compile(selection_cell, "05_model_deployment.ipynb:empty-selection", "exec"),
            {
                "deployable": pd.DataFrame(
                    columns=["package_version"],
                ),
                "PACKAGE_VERSION": None,
            },
        )


def test_mtpl_scratch_is_explicitly_outside_governed_handoff():
    cells = _code_cells("99_scratch_work.ipynb")
    source = _source("99_scratch_work.ipynb")

    assert "scratch" in source.lower()
    assert "load_fremtpl_raw(" in source
    assert "scratch_raw = pd.read_sql_query(" in source
    assert "scratch_frame = scratch_raw.copy()" in source
    assert "SCRATCH_FEATURES = {" in source
    assert 'SCRATCH_FAMILY = "poisson"' in source
    assert "scratch_model = SuperGLM(" in source
    assert ").fit(scratch_X, scratch_y, offset=scratch_offset)" in source
    assert "scratch_model.predict(" in source
    assert "Blank ingestion area" in source
    assert "Blank feature area" in source
    assert "Blank modelling area" in source
    assert "unconstrained_superglm_features(" in source
    assert "unconstrained_model = SuperGLM(" in source
    assert ").fit_reml(" in source
    assert "superglm_edf_table(unconstrained_model)" in source
    assert "fit_boosted_blend(" in source
    assert "reference_superglm=unconstrained_model" in source
    assert "boosted_blend.metrics" in source
    assert 'exposure=scratch_frame.loc[scratch_X.index, "Exposure"]' in source
    assert "save_model_frame(" not in source
    assert "load_model_frame(" not in source
    assert "build_candidate(" not in source
    assert "publish_candidate(" not in source
    assert "deploy_package(" not in source
    assert "EditorSession.from_model(" in source
    assert "list_candidate_versions(" in source
    assert 'versions["Kind"].eq("RAW")' in source
    assert "open_candidate(" in source
    assert "export_level_groupings(" in source
    assert 'GROUPING_ARTIFACT_PATH = MODEL_DIR / ".local"' in source

    source_index = next(i for i, cell in enumerate(cells) if "scratch_raw =" in cell)
    transform_index = next(i for i, cell in enumerate(cells) if "scratch_frame =" in cell)
    features_index = next(i for i, cell in enumerate(cells) if "SCRATCH_FEATURES =" in cell)
    fit_index = next(i for i, cell in enumerate(cells) if "scratch_model = SuperGLM(" in cell)
    inspect_index = next(i for i, cell in enumerate(cells) if "scratch_model.predict(" in cell)
    blend_index = next(i for i, cell in enumerate(cells) if "fit_boosted_blend(" in cell)
    grouping_index = next(i for i, cell in enumerate(cells) if "load_registered_model(" in cell)
    assert (
        source_index
        < transform_index
        < features_index
        < fit_index
        < inspect_index
        < blend_index
        < grouping_index
    )


def test_mtpl_notebook_import_setup_runs_from_model_directory():
    for name in WORKFLOW_NAMES:
        code_cells = _code_cells(name)
        result = subprocess.run(
            [sys.executable, "-c", "\n\n".join(code_cells[:2])],
            cwd=MODEL_DIR.resolve(),
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"
        assert "pricing_pipeline" not in result.stderr


def test_reference_notebooks_do_not_ask_analysts_for_generated_ids():
    generated_ids = {
        "model_id",
        "manifest_id",
        "split_set_id",
        "model_run_id",
        "rate_package_id",
        "package_version",
    }
    assigned_names = set()
    for name in WORKFLOW_NAMES:
        for cell in _code_cells(name):
            tree = ast.parse(cell)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    assigned_names.add(node.id)

    assert assigned_names.isdisjoint(generated_ids)
