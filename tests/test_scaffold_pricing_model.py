from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.scaffold_pricing_model import ScaffoldOptions, scaffold_pricing_model


TEMPLATE_PATH = Path("scripts/templates/pricing_model.ipynb")


def _code_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        compile("".join(cell.get("source", [])), f"{path}:cell-{index}", "exec")
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
    return cells


def test_scaffold_template_is_checked_in_as_valid_notebook_json():
    assert TEMPLATE_PATH.is_file()

    notebook = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["cells"]


def test_scaffold_orders_visible_model_validation_editor_and_deployment_steps(tmp_path):
    scaffold_pricing_model(
        ScaffoldOptions(
            model_name="MY_MODEL",
            target_name="target",
            root=tmp_path,
        )
    )

    cells = _code_cells(tmp_path / "pricing_models" / "my_model" / "pricing_model.ipynb")
    model_index = next(index for index, cell in enumerate(cells) if "FEATURES = {" in cell)
    data_index = next(index for index, cell in enumerate(cells) if "frame = pd.DataFrame" in cell)
    build_index = next(
        index for index, cell in enumerate(cells) if "candidate = build_candidate(" in cell
    )
    validation_index = next(
        index for index, cell in enumerate(cells) if "candidate.validation_metrics" in cell
    )
    baseline_publish_index = next(
        index for index, cell in enumerate(cells) if "published = publish_candidate(" in cell
    )
    editor_index = next(
        index for index, cell in enumerate(cells) if "EditorSession.from_model(" in cell
    )
    materialize_index = next(
        index
        for index, cell in enumerate(cells)
        if "edited_model = editor_session.to_model()" in cell
    )
    edit_publish_index = next(
        index for index, cell in enumerate(cells) if "edited = publish_edits(" in cell
    )
    deploy_index = next(
        index for index, cell in enumerate(cells) if "deployment = deploy_package(" in cell
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
    editor_cell = cells[editor_index]
    materialize_cell = cells[materialize_index]
    publish_cell = cells[edit_publish_index]
    assert "from superglm.editor import EditorSession" in editor_cell
    assert "reviewed = open_candidate(" in editor_cell
    assert "editor_session = EditorSession.from_model(" in editor_cell
    assert "display(editor_widget)" in editor_cell
    assert "publish_edits(" not in editor_cell
    for hidden_side_effect in (".save(", "publish_", "deploy_", "open_candidate("):
        assert hidden_side_effect not in materialize_cell
    assert "candidate=reviewed" in publish_cell
    assert "editor_session=editor_session" in publish_cell


def test_scaffold_renders_user_text_without_breaking_json_or_python(tmp_path):
    root = tmp_path / 'repo "with quotes"'
    model_label = 'Quoted "model"\nwith a second line'
    target_name = 'target"]; raise RuntimeError("not data") #'
    model_type = 'custom "model"\nkind'
    deployment_slot = 'UAT"]; raise RuntimeError("not a slot") #'

    scaffold_pricing_model(
        ScaffoldOptions(
            model_name="SAFE_MODEL",
            model_label=model_label,
            target_name=target_name,
            model_type=model_type,
            deployment_slot=deployment_slot,
            root=root,
        )
    )

    notebook_path = root / "pricing_models" / "safe_model" / "pricing_model.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(_code_cells(notebook_path))
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert f"# {model_label}" in markdown
    assert f"label={json.dumps(model_label)}" in source
    assert f"target={json.dumps(target_name)}" in source
    assert f"model_type={json.dumps(model_type)}" in source
    assert f"deployment_slot={json.dumps(deployment_slot)}" in source
    assert str(root) not in source


def test_scaffold_writes_only_the_analyst_notebook_package(tmp_path):
    result = scaffold_pricing_model(
        ScaffoldOptions(
            model_name="MY_MODEL",
            model_label="My model",
            target_name="derived_target",
            root=tmp_path,
        )
    )

    package_dir = tmp_path / "pricing_models" / "my_model"
    notebook_path = package_dir / "pricing_model.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert result.created_files == (package_dir / "__init__.py", notebook_path)
    assert not (package_dir / "model.toml").exists()
    assert not (tmp_path / "dags" / "pricing_my_model.py").exists()

    cells = _code_cells(notebook_path)
    source = "\n".join(cells)
    settings_cell = next(
        cell
        for cell in notebook["cells"]
        if "DATABASE_MODE" in "".join(cell.get("source", []))
    )
    assert settings_cell["metadata"]["tags"] == [
        "pricing-pipeline-operational-settings"
    ]
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "before package publication" in markdown
    assert "before any SQL publication" not in markdown
    assert 'DATABASE_MODE = "local"' in cells[0]
    assert "RUNTIME_MODULE = None" in cells[0]
    assert 'EXPECTED_REMOTE_DATABASE = ""' in cells[0]
    assert "ALLOW_REMOTE_WRITES = False" in cells[0]
    assert "PricingModelSpec(" in source
    assert "from superglm import Numeric, SuperGLM" in source
    assert "from superglm.editor import EditorSession" in source
    assert 'SCORING = ("deviance", "nll", "gini")' in cells[0]
    assert source.count("FEATURES = {") == 1
    assert source.count("superglm_model = SuperGLM(") == 1
    assert '"feature_1": Numeric()' in source
    assert "features=FEATURES," in source
    assert "features=tuple(FEATURES)," in source
    assert "scoring=SCORING," in source
    assert "offset_column=None" in source
    assert "offset_source_column=None" in source
    assert "offset_label=None" in source
    assert "sample_weight_column=None" in source
    assert "export_weight_column=None" in source
    assert "exposure_column=" not in source
    assert '# frame["term_offset"] = np.log(frame["term"] / 12.0)' in source
    assert '# offset_column="term_offset"' in source
    assert '# offset_source_column="term"' in source
    assert '# offset_label="log(term / 12)"' in source
    assert "register_model(" in source
    assert "build_candidate(" in source
    assert "superglm_model=superglm_model" in source
    assert "publish_candidate(" in source
    assert "open_candidate(" in source
    assert "publish_edits(" in source
    assert "deploy_package(" in source
    assert "candidate.validation_metrics" in source
    assert '"Model version": published.model_version' in source
    assert '"Artifact root": str(pricing.settings.workbench_artifact_root)' in source
    assert "runtime_module=RUNTIME_MODULE" in source
    assert "build_pricing_model_dag" not in source
    assert "MODEL_CONFIG" not in source
    assert "model.toml" not in source
    assert "Airflow" not in source
    assert "private-server" not in source
    assert "mssql_server" not in source
    for hidden_model_surface in ("FEATURE_COLUMNS", "make_model", "model_factory"):
        assert hidden_model_surface not in source


def test_scaffold_preserves_existing_notebook_and_recreates_missing_init(tmp_path):
    options = ScaffoldOptions(model_name="MY_MODEL", target_name="target", root=tmp_path)
    scaffold_pricing_model(options)
    package_dir = tmp_path / "pricing_models" / "my_model"
    notebook_path = package_dir / "pricing_model.ipynb"
    init_path = package_dir / "__init__.py"
    notebook_path.write_text(notebook_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    notebook_before = notebook_path.read_text(encoding="utf-8")
    init_path.unlink()

    result = scaffold_pricing_model(options)

    assert result.created_files == (init_path,)
    assert notebook_path.read_text(encoding="utf-8") == notebook_before
    assert scaffold_pricing_model(options).created_files == ()


def test_scaffold_force_overwrites_existing_notebook(tmp_path):
    options = ScaffoldOptions(model_name="MY_MODEL", target_name="target", root=tmp_path)
    scaffold_pricing_model(options)
    notebook_path = tmp_path / "pricing_models" / "my_model" / "pricing_model.ipynb"
    notebook_path.write_text("stale", encoding="utf-8")

    result = scaffold_pricing_model(
        ScaffoldOptions(
            model_name="MY_MODEL",
            target_name="target",
            root=tmp_path,
            force=True,
        )
    )

    assert notebook_path in result.created_files
    assert notebook_path.read_text(encoding="utf-8") != "stale"


def test_scaffold_accepts_explicit_model_names(tmp_path):
    result = scaffold_pricing_model(
        ScaffoldOptions(
            model_name="WORK_FREQ",
            model_label="Work frequency",
            target_name="claim_count",
            model_type="superglm_tweedie",
            deployment_slot="WORK_FREQ_PROD",
            package_name="work_frequency",
            root=tmp_path,
        )
    )

    assert result.package_name == "work_frequency"
    source = "\n".join(
        _code_cells(tmp_path / "pricing_models" / "work_frequency" / "pricing_model.ipynb")
    )
    assert 'name="WORK_FREQ"' in source
    assert 'label="Work frequency"' in source
    assert 'target="claim_count"' in source
    assert 'model_type="superglm_tweedie"' in source
    assert 'deployment_slot="WORK_FREQ_PROD"' in source


def test_scaffold_script_help_has_no_legacy_factory_options():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/scaffold_pricing_model.py", "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "--model-name" in result.stdout
    assert "--target-name" in result.stdout
    assert "--template" not in result.stdout
    assert "--dag-id" not in result.stdout
    assert "--experiment-name" not in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_scaffold_script_reports_notebook_path(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/scaffold_pricing_model.py",
            "--model-name",
            "SCRIPT_MODEL",
            "--target-name",
            "target",
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "pricing_models/script_model/pricing_model.ipynb" in result.stdout
    assert "model.toml" not in result.stdout
    assert "DAG" not in result.stdout
