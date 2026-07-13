from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


NOTEBOOK_PATH = Path("pricing_models/mtpl_frequency/pricing_model.ipynb")


def _source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )


def test_mtpl_pricing_model_notebook_is_direct_python_sql_workflow():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = _source(notebook)
    lowered = source.lower()

    assert "from pricing_pipeline.notebook import" in source
    assert "connect(" in source
    assert "register_model(" in source
    assert "pd.read_sql_query(" in source
    assert 'frame["LogDensity"]' in source
    assert "def make_model()" in source
    assert "ValidationSplitConfig.kfold(" in source
    assert "build_candidate(" in source
    assert "publish_candidate(" in source
    assert "open_candidate(" in source
    assert "publish_edits(" in source
    assert "deploy_package(" in source
    assert "published.model_run_id" in source
    assert "published.rate_package_id" in source
    assert "published.package_version" in source
    assert 'DATABASE_MODE = "local"' in source
    assert 'EXPECTED_REMOTE_DATABASE = ""' in source
    assert "ALLOW_REMOTE_WRITES = False" in source
    assert "mode=DATABASE_MODE" in source
    assert 'local_root=MODEL_DIR / ".local"' in source
    assert "expected_remote_database=EXPECTED_REMOTE_DATABASE" in source
    assert "allow_remote_writes=ALLOW_REMOTE_WRITES" in source
    assert "pricing.destination" in source

    assert "model.toml" not in lowered
    assert "model_config" not in lowered
    assert "airflow" not in lowered
    assert "apply_migrations" not in source
    assert "schema_migration" not in lowered
    generated_ids = {
        "model_id",
        "manifest_id",
        "split_set_id",
        "model_run_id",
        "rate_package_id",
        "package_version",
    }
    assigned_names = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned_names.add(node.id)
    assert assigned_names.isdisjoint(generated_ids)


def test_all_pricing_model_notebooks_compile_and_have_no_saved_output():
    notebook_paths = sorted(Path("pricing_models").glob("*/pricing_model.ipynb"))
    assert notebook_paths

    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(source, f"{notebook_path}:cell-{index}", "exec")
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_mtpl_notebook_import_setup_runs_from_its_model_directory():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]

    result = subprocess.run(
        [sys.executable, "-c", "\n\n".join(code_cells[:2])],
        cwd=NOTEBOOK_PATH.parent.resolve(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "pricing_pipeline" not in result.stderr


def test_mtpl_pricing_model_notebook_keeps_a_small_analyst_surface():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    source = "\n".join(code_cells)

    assert 'DATABASE_MODE = "local"' in code_cells[0]
    assert 'EXPECTED_REMOTE_DATABASE = ""' in code_cells[0]
    assert "ALLOW_REMOTE_WRITES = False" in code_cells[0]
    assert "PricingModelSpec" in source
    globals_cell = next(cell for cell in code_cells if "DATABASE_MODE" in cell)
    assert 'DATABASE_MODE = "local"' in globals_cell
    assert 'EXPECTED_REMOTE_DATABASE = ""' in globals_cell
    assert "ALLOW_REMOTE_WRITES = False" in globals_cell
    assert "DATA_AS_OF" in globals_cell
    assert "RUN_EDITOR = False" in globals_cell
    assert "DEPLOY = False" in globals_cell
    model_cell = next(cell for cell in code_cells if "MODEL = PricingModelSpec(" in cell)
    assert "FEATURE_COLUMNS" in model_cell
    connect_cell = next(cell for cell in code_cells if "pricing = connect(" in cell)
    assert "mode=DATABASE_MODE" in connect_cell
    assert 'local_root=MODEL_DIR / ".local"' in connect_cell
    assert "expected_remote_database=EXPECTED_REMOTE_DATABASE" in connect_cell
    assert "allow_remote_writes=ALLOW_REMOTE_WRITES" in connect_cell
    assert "pricing.destination" in connect_cell
    assert "EFFECTIVE_FROM" not in source
    assert ".head(" not in source
    assert "display(raw" not in source
    assert "display(frame" not in source

    build_cell = next(cell for cell in code_cells if "candidate = build_candidate(" in cell)
    assert "pricing," in build_cell
    assert "model=model" in build_cell
    assert "frame=frame" in build_cell
    assert "model_factory=make_model" in build_cell
    assert "data_as_of=DATA_AS_OF" in build_cell
    for hidden_argument in (
        "X=X",
        "y=y",
        "scoring=",
        "dataset_name=",
        "source_system=",
        "pk_columns=",
        "weight_column=",
        "offset_contract=",
        "offset_export_options=",
    ):
        assert hidden_argument not in build_cell
