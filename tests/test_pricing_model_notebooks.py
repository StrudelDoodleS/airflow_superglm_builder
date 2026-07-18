from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


NOTEBOOK_PATH = Path("pricing_models/mtpl_frequency/pricing_model.ipynb")


def _source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def test_mtpl_pricing_model_notebook_is_direct_python_sql_workflow():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = _source(notebook)
    lowered = source.lower()

    assert "from pricing_pipeline.notebook import" in source
    assert "connect(" in source
    assert "register_model(" in source
    assert "pd.read_sql_query(" in source
    assert "load_fremtpl_raw" in source
    assert 'frame["LogDensity"]' in source
    assert "superglm_model = SuperGLM(" in source
    assert "from superglm.editor import EditorSession" in source
    assert "ValidationSplitConfig.kfold(" in source
    assert "build_candidate(" in source
    assert "publish_candidate(" in source
    assert "open_candidate(" in source
    assert "publish_edits(" in source
    assert "deploy_package(" in source
    assert "published.model_run_id" not in source
    assert "published.rate_package_id" not in source
    assert "published.package_version" in source
    assert "published.model_version" in source
    assert 'DATABASE_MODE = "local"' in source
    assert 'EXPECTED_REMOTE_DATABASE = ""' in source
    assert "ALLOW_REMOTE_WRITES = False" in source
    assert "mode=DATABASE_MODE" in source
    assert 'local_root=MODEL_DIR / ".local"' in source
    assert "expected_remote_database=EXPECTED_REMOTE_DATABASE" in source
    assert "allow_remote_writes=ALLOW_REMOTE_WRITES" in source
    assert "pricing.destination" in source
    assert "pricing.settings.workbench_artifact_root" in source

    source_cell = next(
        cell for cell in notebook["cells"] if "SOURCE_SQL" in "".join(cell.get("source", []))
    )
    source_code = "".join(source_cell["source"])
    assert 'if pricing.mode == "local":' in source_code
    assert source_code.count("load_fremtpl_raw(") == 1
    assert "load_fremtpl_raw(pricing.engine, replace=REFRESH_LOCAL_RAW)" in source_code

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
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]

    result = subprocess.run(
        [sys.executable, "-c", "\n\n".join(code_cells[:3])],
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
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    source = "\n".join(code_cells)

    assert 'DATABASE_MODE = "local"' in code_cells[0]
    assert "RUNTIME_MODULE = None" in code_cells[0]
    assert 'EXPECTED_REMOTE_DATABASE = ""' in code_cells[0]
    assert "ALLOW_REMOTE_WRITES = False" in code_cells[0]
    assert "REFRESH_LOCAL_RAW = False" in code_cells[0]
    assert "PricingModelSpec" in source
    globals_cell = next(cell for cell in code_cells if "DATABASE_MODE" in cell)
    assert 'DATABASE_MODE = "local"' in globals_cell
    assert 'EXPECTED_REMOTE_DATABASE = ""' in globals_cell
    assert "ALLOW_REMOTE_WRITES = False" in globals_cell
    assert "REFRESH_LOCAL_RAW = False" in globals_cell
    assert "DATA_AS_OF" in globals_cell
    assert "RUN_EDITOR = False" in globals_cell
    assert "DEPLOY = False" in globals_cell
    assert 'SCORING = ("deviance", "nll", "gini")' in globals_cell
    model_cell = next(cell for cell in code_cells if "MODEL = PricingModelSpec(" in cell)
    assert "from superglm import Categorical, Numeric, Spline, SuperGLM" in source
    assert source.count("FEATURES = {") == 1
    assert source.count("superglm_model = SuperGLM(") == 1
    assert "features=FEATURES," in model_cell
    assert "features=tuple(FEATURES)," in model_cell
    assert "scoring=SCORING," in model_cell
    for feature_declaration in (
        '"VehAge": Spline()',
        '"DrivAge": Spline()',
        '"BonusMalus": Spline()',
        '"LogDensity": Numeric()',
        '"Area": Categorical()',
        '"VehPower": Categorical()',
        '"VehBrand": Categorical()',
        '"VehGas": Categorical()',
        '"Region": Categorical()',
    ):
        assert feature_declaration in model_cell
    connect_cell = next(cell for cell in code_cells if "pricing = connect(" in cell)
    assert "mode=DATABASE_MODE" in connect_cell
    assert "runtime_module=RUNTIME_MODULE" in connect_cell
    assert 'local_root=MODEL_DIR / ".local"' in connect_cell
    assert "expected_remote_database=EXPECTED_REMOTE_DATABASE" in connect_cell
    assert "allow_remote_writes=ALLOW_REMOTE_WRITES" in connect_cell
    assert "pricing.destination" in connect_cell
    assert "EFFECTIVE_FROM" not in source
    assert ".head(" not in source
    assert "display(raw" not in source
    assert "display(frame" not in source
    assert 'frame["LogDensity"] = np.log(' in source
    assert 'frame["LogExposure"] = np.log(frame["Exposure"].astype(float))' in source
    assert "*FEATURES]" in source
    assert 'offset_column="LogExposure"' in model_cell
    assert 'offset_source_column="Exposure"' in model_cell
    assert 'offset_label="log(Exposure)"' in model_cell
    assert "sample_weight_column=None" in model_cell
    assert 'export_weight_column="Exposure"' in model_cell
    assert "exposure_column=" not in source

    for hidden_model_surface in (
        "FEATURE_COLUMNS",
        "make_model",
        "model_factory",
        "ensure_local_fremtpl_demo",
        "row_count=120",
    ):
        assert hidden_model_surface not in source

    build_cell = next(cell for cell in code_cells if "candidate = build_candidate(" in cell)
    assert "pricing," in build_cell
    assert "model=model" in build_cell
    assert "frame=frame" in build_cell
    assert "superglm_model=superglm_model" in build_cell
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

    model_index = code_cells.index(model_cell)
    data_index = next(
        index for index, cell in enumerate(code_cells) if "raw = pd.read_sql_query" in cell
    )
    build_index = code_cells.index(build_cell)
    validation_index = next(
        index for index, cell in enumerate(code_cells) if "candidate.validation_metrics" in cell
    )
    baseline_publish_index = next(
        index for index, cell in enumerate(code_cells) if "published = publish_candidate(" in cell
    )
    editor_index = next(
        index for index, cell in enumerate(code_cells) if "EditorSession.from_model(" in cell
    )
    materialize_index = next(
        index
        for index, cell in enumerate(code_cells)
        if "edited_model = editor_session.to_model()" in cell
    )
    edit_publish_index = next(
        index for index, cell in enumerate(code_cells) if "edited = publish_edits(" in cell
    )
    deploy_index = next(
        index for index, cell in enumerate(code_cells) if "deployment = deploy_package(" in cell
    )
    assert (
        model_index
        < data_index
        < build_index
        < validation_index
        < baseline_publish_index
        < editor_index
        < materialize_index
        < edit_publish_index
        < deploy_index
    )
    assert ".editor()" not in source
    assert "from superglm.editor import EditorSession" in code_cells[editor_index]
    assert "editor_session=editor_session" in code_cells[edit_publish_index]
    for hidden_side_effect in (".save(", "publish_", "deploy_", "open_candidate("):
        assert hidden_side_effect not in code_cells[materialize_index]
