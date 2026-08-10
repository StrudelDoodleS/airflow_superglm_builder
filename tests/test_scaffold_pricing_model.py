from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from scripts.scaffold_pricing_model import (
    _NOTEBOOK_NAMES,
    ScaffoldOptions,
    scaffold_pricing_model,
)


NOTEBOOK_NAME = re.compile(r"^\d{2}_[a-z0-9]+(?:_[a-z0-9]+)*\.ipynb$")
EXPECTED_NOTEBOOKS = (
    "01_data_ingestion.ipynb",
    "02_model_training.ipynb",
    "03_model_editor.ipynb",
    "04_model_deployment.ipynb",
    "99_scratch_work.ipynb",
)


def _notebook(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        compile("".join(cell.get("source", [])), f"{path}:cell-{index}", "exec")
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
    return notebook


def _code(path: Path) -> str:
    notebook = _notebook(path)
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def _scaffold(tmp_path: Path, **overrides) -> Path:
    options = {
        "model_name": "MY_MODEL",
        "model_label": "My model",
        "target_name": "target",
        "root": tmp_path,
        **overrides,
    }
    result = scaffold_pricing_model(ScaffoldOptions(**options))
    return tmp_path / "pricing_models" / result.package_name


def test_scaffold_has_one_strict_ordered_notebook_contract():
    assert _NOTEBOOK_NAMES == EXPECTED_NOTEBOOKS
    assert all(NOTEBOOK_NAME.fullmatch(name) for name in _NOTEBOOK_NAMES)


def test_scaffold_writes_five_notebook_workflow_and_no_legacy_factory(tmp_path):
    result = scaffold_pricing_model(
        ScaffoldOptions(
            model_name="MY_MODEL",
            model_label="My model",
            target_name="derived_target",
            root=tmp_path,
        )
    )

    package_dir = tmp_path / "pricing_models" / "my_model"
    expected = (
        package_dir / "__init__.py",
        *(package_dir / name for name in EXPECTED_NOTEBOOKS),
    )
    assert result.created_files == expected
    assert sorted(path.name for path in package_dir.glob("*.ipynb")) == sorted(EXPECTED_NOTEBOOKS)
    assert not (package_dir / "model.toml").exists()
    assert not (tmp_path / "dags" / "pricing_my_model.py").exists()
    for notebook_path in expected[1:]:
        _notebook(notebook_path)


def test_scaffold_separates_ingestion_training_editor_deployment_and_scratch(tmp_path):
    package_dir = _scaffold(tmp_path)
    ingestion = _code(package_dir / "01_data_ingestion.ipynb")
    training = _code(package_dir / "02_model_training.ipynb")
    editor = _code(package_dir / "03_model_editor.ipynb")
    deployment = _code(package_dir / "04_model_deployment.ipynb")
    scratch = _code(package_dir / "99_scratch_work.ipynb")

    assert "save_model_frame(" in ingestion
    assert 'DATA_AS_OF = ""' in ingestion
    assert '"data_as_of": [DATA_AS_OF]' in ingestion
    assert "if not DATA_AS_OF.strip()" in ingestion
    assert "build_candidate(" not in ingestion
    assert "load_model_frame(" in training
    assert 'data_as_of_column="data_as_of"' in training
    assert 'model_kind="RAW"' in training
    assert 'model_kind="ROUTINE_EDIT"' in training
    assert "RAW_FEATURES" in training
    assert "load_level_groupings(" in training
    assert "apply_level_groupings(" in training
    assert "ROUTINE_EDIT_CONFIGURED = bool(LEVEL_GROUPINGS)" in training
    assert "LevelGrouping(" not in training
    assert "EditorSession" not in training

    assert "load_registered_model(" in editor
    assert "list_candidate_versions(" in editor
    assert "PACKAGE_VERSION = None" in editor
    assert "EditorSession.from_model(" in editor
    assert "editor_session.to_model()" in editor
    assert "publish_edits(" in editor

    assert "list_candidate_versions(" in deployment
    assert 'eq("PUBLISHED")' in deployment
    assert "open_candidate(" in deployment
    assert "deploy_package(" in deployment

    assert "save_model_frame(" not in scratch
    assert "build_candidate(" not in scratch
    assert "publish_candidate(" not in scratch
    assert "deploy_package(" not in scratch
    assert "EditorSession.from_model(" in scratch
    assert "list_candidate_versions(" in scratch
    assert 'versions["Kind"].eq("RAW")' in scratch
    assert "open_candidate(" in scratch
    assert "export_level_groupings(" in scratch


def test_scaffold_keeps_editor_preview_and_publish_as_separate_cells(tmp_path):
    package_dir = _scaffold(tmp_path)
    notebook = _notebook(package_dir / "03_model_editor.ipynb")
    cells = [
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    editor_index = next(i for i, cell in enumerate(cells) if "EditorSession.from_model(" in cell)
    preview_index = next(i for i, cell in enumerate(cells) if "editor_session.to_model()" in cell)
    publish_index = next(i for i, cell in enumerate(cells) if "publish_edits(" in cell)

    assert editor_index < preview_index < publish_index
    assert "publish_edits(" not in cells[editor_index]
    assert "publish_edits(" not in cells[preview_index]
    assert "candidate=reviewed" in cells[publish_index]
    assert "editor_session=editor_session" in cells[publish_index]


def test_scaffold_renders_user_text_without_breaking_json_or_python(tmp_path):
    root = tmp_path / 'repo "with quotes"'
    model_label = 'Quoted "model"\nwith a second line'
    target_name = 'target"]; raise RuntimeError("not data") #'
    model_type = 'custom "model"\nkind'
    deployment_slot = 'UAT"]; raise RuntimeError("not a slot") #'

    package_dir = _scaffold(
        root,
        model_name="SAFE_MODEL",
        model_label=model_label,
        target_name=target_name,
        model_type=model_type,
        deployment_slot=deployment_slot,
    )
    all_source = "\n".join(_code(package_dir / name) for name in EXPECTED_NOTEBOOKS)
    all_markdown = "\n".join(
        "".join(cell.get("source", []))
        for name in EXPECTED_NOTEBOOKS
        for cell in _notebook(package_dir / name)["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert f"# {model_label}" in all_markdown
    assert f"label={json.dumps(model_label)}" in all_source
    assert f"target={json.dumps(target_name)}" in all_source
    assert f"model_type={json.dumps(model_type)}" in all_source
    assert f"DEPLOYMENT_SLOT = {json.dumps(deployment_slot)}" in all_source
    assert str(root) not in all_source


def test_scaffold_preserves_existing_files_and_recreates_only_missing_files(tmp_path):
    options = ScaffoldOptions(model_name="MY_MODEL", target_name="target", root=tmp_path)
    scaffold_pricing_model(options)
    package_dir = tmp_path / "pricing_models" / "my_model"
    training_path = package_dir / "02_model_training.ipynb"
    init_path = package_dir / "__init__.py"
    training_path.write_text(
        training_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    training_before = training_path.read_text(encoding="utf-8")
    init_path.unlink()

    result = scaffold_pricing_model(options)

    assert result.created_files == (init_path,)
    assert training_path.read_text(encoding="utf-8") == training_before
    assert scaffold_pricing_model(options).created_files == ()


def test_scaffold_force_overwrites_all_workflow_files(tmp_path):
    options = ScaffoldOptions(model_name="MY_MODEL", target_name="target", root=tmp_path)
    scaffold_pricing_model(options)
    package_dir = tmp_path / "pricing_models" / "my_model"
    training_path = package_dir / "02_model_training.ipynb"
    training_path.write_text("stale", encoding="utf-8")

    result = scaffold_pricing_model(
        ScaffoldOptions(
            model_name="MY_MODEL",
            target_name="target",
            root=tmp_path,
            force=True,
        )
    )

    assert result.created_files == (
        package_dir / "__init__.py",
        *(package_dir / name for name in EXPECTED_NOTEBOOKS),
    )
    assert training_path.read_text(encoding="utf-8") != "stale"


def test_scaffold_accepts_explicit_model_identity(tmp_path):
    package_dir = _scaffold(
        tmp_path,
        model_name="WORK_FREQ",
        model_label="Work frequency",
        target_name="claim_count",
        model_type="superglm_tweedie",
        deployment_slot="WORK_FREQ_PROD",
        package_name="work_frequency",
    )
    source = "\n".join(_code(package_dir / name) for name in EXPECTED_NOTEBOOKS)

    assert 'name="WORK_FREQ"' in source
    assert 'label="Work frequency"' in source
    assert 'target="claim_count"' in source
    assert 'model_type="superglm_tweedie"' in source
    assert 'DEPLOYMENT_SLOT = "WORK_FREQ_PROD"' in source


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


def test_scaffold_script_reports_all_notebook_paths(tmp_path):
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
    for name in EXPECTED_NOTEBOOKS:
        assert f"pricing_models/script_model/{name}" in result.stdout
    assert "model.toml" not in result.stdout
    assert "DAG" not in result.stdout
