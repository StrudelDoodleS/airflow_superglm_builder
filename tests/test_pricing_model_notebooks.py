from __future__ import annotations

import ast
import json
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


def test_mtpl_pricing_model_notebook_code_cells_compile_and_have_no_saved_output():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{NOTEBOOK_PATH}:cell-{index}", "exec")
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []


def test_mtpl_pricing_model_notebook_keeps_a_small_analyst_surface():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    source = "\n".join(code_cells)

    assert "PricingModelSpec" in code_cells[0]
    assert "MODEL = PricingModelSpec(" in code_cells[1]
    assert "DATA_AS_OF" in code_cells[1]
    assert "RUN_EDITOR = False" in code_cells[2]
    assert "DEPLOY = False" in code_cells[2]
    assert "EFFECTIVE_FROM" not in source

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
