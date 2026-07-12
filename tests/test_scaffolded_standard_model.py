from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import pricing_models
from pricing_pipeline.modeling.standard_superglm import ModelInputs
from scripts.scaffold_pricing_model import ScaffoldOptions, scaffold_pricing_model


def test_fresh_custom_scaffold_delegates_to_standard_runner(tmp_path, monkeypatch):
    scaffold_pricing_model(
        ScaffoldOptions(
            model_name="SCAFFOLD_FREQ",
            model_label="Scaffold frequency",
            target_name="claim_count",
            root=tmp_path,
        )
    )
    generated_models = tmp_path / "pricing_models"
    monkeypatch.setattr(
        pricing_models,
        "__path__",
        [*pricing_models.__path__, str(generated_models)],
    )
    for name in (
        "pricing_models.scaffold_freq",
        "pricing_models.scaffold_freq.data",
        "pricing_models.scaffold_freq.spec",
        "pricing_models.scaffold_freq.modeling",
    ):
        sys.modules.pop(name, None)
    modeling = importlib.import_module("pricing_models.scaffold_freq.modeling")

    frame = pd.DataFrame(
        {
            "policy_id": [2, 1, 3],
            "claim_count": [0.0, 1.0, 0.0],
            "age": [30.0, 20.0, 40.0],
        }
    )
    calls = []
    monkeypatch.setattr(modeling, "PK_COLUMNS", ("policy_id",))
    monkeypatch.setattr(modeling, "read_prepared_source", lambda prepared: frame.copy())
    monkeypatch.setattr(modeling, "build_final_model_frame", lambda raw: raw.copy())
    monkeypatch.setattr(
        modeling,
        "build_training_inputs",
        lambda final: ModelInputs(
            X=final[["age"]],
            y=final["claim_count"].to_numpy(),
        ),
    )
    monkeypatch.setattr(modeling, "build_model", lambda: object())
    monkeypatch.setattr(
        modeling,
        "validation_splitter",
        lambda final: [(np.array([0, 1]), np.array([2]))],
    )
    monkeypatch.setattr(
        modeling,
        "resolve_model_version_for_export",
        lambda engine, **kwargs: "v1",
    )

    def fake_standard_runner(engine, **kwargs):
        calls.append((engine, kwargs))
        return SimpleNamespace(completed_build={"candidate_artifact_path": "candidate.joblib"})

    monkeypatch.setattr(modeling, "run_standard_superglm_build", fake_standard_runner)
    settings = SimpleNamespace(
        workbench_artifact_root=tmp_path / "workbench",
        validation_split_artifact_root=tmp_path / "splits",
    )

    result = modeling.train_validate_export_model(
        {
            "run_key": "scheduled__20260712",
            "effective_from": "2026-07-12",
            "data_as_of_date": "2026-06-30",
        },
        engine="engine",
        settings=settings,
        created_by="pytest",
    )

    assert result == {"candidate_artifact_path": "candidate.joblib"}
    assert len(calls) == 1
    runner_kwargs = calls[0][1]
    assert runner_kwargs["frame"]["policy_id"].tolist() == [1, 2, 3]
    assert runner_kwargs["output_dir"] == (
        tmp_path / "workbench" / "SCAFFOLD_FREQ" / runner_kwargs["export_id"]
    )
    assert runner_kwargs["manifest_spec"].pk_columns == ("policy_id",)
    assert runner_kwargs["model_source_root"] == Path(modeling.__file__).parent
