from __future__ import annotations

from dataclasses import replace
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from superglm import Numeric, SuperGLM

from pricing_pipeline.build_identity import (
    BuildIdentityError,
    create_build_identity,
    stable_build_export_id,
    verify_build_identity,
)
from pricing_pipeline.data.manifest import ModelFrameManifestSpec
from pricing_pipeline.models.config import ModelBuildConfig, ValidationSplitConfig


def _source_tree(root: Path, content: str = "MODEL = 'poisson'\n") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "modeling.py").write_text(content, encoding="utf-8")
    return root


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy_id": [101, 102, 103, 104],
            "vehicle_age": [1.0, 3.0, 8.0, 12.0],
            "driver_age": [21.0, 35.0, 42.0, 68.0],
            "claim_count": [0.0, 1.0, 0.0, 2.0],
            "fit_weight": [1.0, 1.5, 0.8, 1.2],
            "log_term": [0.0, 0.0, np.log(3.0), np.log(3.0)],
            "term": [12.0, 12.0, 36.0, 36.0],
            "export_weight": [10.0, 15.0, 8.0, 12.0],
            "as_at": pd.to_datetime(["2026-06-30"] * 4),
        }
    )


def _validation() -> ValidationSplitConfig:
    return ValidationSplitConfig.kfold(
        n_splits=2,
        random_state=17,
        shuffle=True,
        materialize=True,
    )


def _model_config(validation: ValidationSplitConfig | None = None) -> ModelBuildConfig:
    return ModelBuildConfig(
        model_name="MOTOR_FREQ",
        model_label="Motor frequency",
        target_name="claim_count",
        model_type="superglm_poisson",
        deployment_slot="MOTOR_FREQ_CURRENT",
        validation_split=validation or _validation(),
    )


def _manifest_spec(**overrides) -> ModelFrameManifestSpec:
    values = {
        "dataset_name": "motor_model_frame",
        "source_system": "pricing_sql",
        "data_as_of_date": date(2026, 6, 30),
        "pk_columns": ("policy_id",),
        "target_column": "claim_count",
        "weight_column": "fit_weight",
        "feature_columns": ("vehicle_age", "driver_age"),
        "offset_column": "log_term",
        "offset_source_column": "term",
        "offset_label": "Policy term",
        "export_weight_column": "export_weight",
        "data_as_of_column": "as_at",
    }
    values.update(overrides)
    return ModelFrameManifestSpec(**values)


def _splits(dtype=np.int64):
    return (
        (np.array([0, 2], dtype=dtype), np.array([1, 3], dtype=dtype)),
        (np.array([1, 3], dtype=dtype), np.array([0, 2], dtype=dtype)),
    )


def _identity_args(tmp_path: Path) -> dict:
    return {
        "frame": _frame(),
        "model_config": _model_config(),
        "manifest_spec": _manifest_spec(),
        "superglm_model": SuperGLM(
            family="poisson",
            features={
                "vehicle_age": Numeric(),
                "driver_age": Numeric(),
            },
            spline_penalty=0.2,
        ),
        "split_indices": _splits(),
        "fit_mode": "fit_reml",
        "scoring": ("deviance", "nll", "gini"),
        "model_source_root": _source_tree(tmp_path / "model_source"),
        "builder_source_root": _source_tree(
            tmp_path / "builder_source",
            "BUILDER_CONTRACT = 1\n",
        ),
    }


def test_build_identity_is_deterministic_and_path_independent(tmp_path: Path):
    first_args = _identity_args(tmp_path / "first")
    second_args = _identity_args(tmp_path / "second")
    second_args["split_indices"] = _splits(np.int32)

    first = create_build_identity(**first_args)
    second = create_build_identity(**second_args)

    assert first == second
    assert len(first.build_fingerprint_sha256) == 64
    assert len(first.model_frame_sha256) == 64
    assert len(first.row_order_sha256) == 64
    assert len(first.model_source_sha256) == 64
    assert len(first.builder_source_sha256) == 64
    assert len(first.materialized_split_sha256) == 64
    assert len(first.runtime_sha256) == 64
    assert len(first.candidate_superglm_sha256) == 64
    assert first.candidate_superglm_version == "0.12.0"
    assert first.candidate_superglm_git_sha == "25c06fc84b674bb2ee777ea99567772d8d57a17c"
    assert stable_build_export_id(first) == f"build_{first.build_fingerprint_sha256}"


@pytest.mark.parametrize(
    "change",
    [
        "frame_value",
        "frame_row_order",
        "model_name",
        "model_type",
        "dataset_name",
        "source_system",
        "data_as_of",
        "pk_order",
        "target",
        "feature_order",
        "offset_column",
        "sample_weight_column",
        "export_weight_column",
        "data_as_of_column",
        "validation_definition",
        "materialized_splits",
        "model_configuration",
        "fit_mode",
        "scoring",
        "model_source",
        "builder_source",
    ],
)
def test_every_material_build_contract_change_changes_fingerprint(
    tmp_path: Path,
    change: str,
):
    baseline_args = _identity_args(tmp_path / "baseline")
    changed_args = _identity_args(tmp_path / "changed")

    if change == "frame_value":
        changed_args["frame"].loc[0, "vehicle_age"] = 99.0
    elif change == "frame_row_order":
        changed_args["frame"] = changed_args["frame"].iloc[::-1].reset_index(drop=True)
    elif change == "model_name":
        changed_args["model_config"] = replace(
            changed_args["model_config"],
            model_name="MOTOR_FREQ_ALT",
        )
    elif change == "model_type":
        changed_args["model_config"] = replace(
            changed_args["model_config"],
            model_type="superglm_tweedie",
        )
    elif change == "dataset_name":
        changed_args["manifest_spec"] = _manifest_spec(dataset_name="new_frame")
    elif change == "source_system":
        changed_args["manifest_spec"] = _manifest_spec(source_system="new_sql_source")
    elif change == "data_as_of":
        changed_args["manifest_spec"] = _manifest_spec(
            data_as_of_date=date(2026, 7, 1)
        )
    elif change == "pk_order":
        for args in (baseline_args, changed_args):
            args["frame"]["secondary_id"] = [1, 2, 3, 4]
        baseline_args["manifest_spec"] = _manifest_spec(
            pk_columns=("policy_id", "secondary_id")
        )
        changed_args["manifest_spec"] = _manifest_spec(
            pk_columns=("secondary_id", "policy_id")
        )
    elif change == "target":
        changed_args["model_config"] = replace(
            changed_args["model_config"],
            target_name="alternate_target",
        )
        changed_args["manifest_spec"] = _manifest_spec(target_column="alternate_target")
    elif change == "feature_order":
        changed_args["manifest_spec"] = _manifest_spec(
            feature_columns=("driver_age", "vehicle_age")
        )
    elif change == "offset_column":
        changed_args["manifest_spec"] = _manifest_spec(offset_column="other_log_term")
    elif change == "sample_weight_column":
        changed_args["manifest_spec"] = _manifest_spec(weight_column="other_weight")
    elif change == "export_weight_column":
        changed_args["manifest_spec"] = _manifest_spec(
            export_weight_column="other_export_weight"
        )
    elif change == "data_as_of_column":
        changed_args["manifest_spec"] = _manifest_spec(data_as_of_column="other_as_at")
    elif change == "validation_definition":
        changed_args["model_config"] = _model_config(
            ValidationSplitConfig.kfold(
                n_splits=2,
                random_state=18,
                shuffle=True,
                materialize=True,
            )
        )
    elif change == "materialized_splits":
        changed_args["split_indices"] = (
            (np.array([1, 2]), np.array([0, 3])),
            (np.array([0, 3]), np.array([1, 2])),
        )
    elif change == "model_configuration":
        changed_args["superglm_model"] = SuperGLM(
            family="poisson",
            features={
                "vehicle_age": Numeric(),
                "driver_age": Numeric(),
            },
            spline_penalty=0.3,
        )
    elif change == "fit_mode":
        changed_args["fit_mode"] = "fit"
    elif change == "scoring":
        changed_args["scoring"] = ("deviance", "gini")
    elif change == "model_source":
        _source_tree(changed_args["model_source_root"], "MODEL = 'changed'\n")
    elif change == "builder_source":
        _source_tree(changed_args["builder_source_root"], "BUILDER_CONTRACT = 2\n")
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(change)

    assert (
        create_build_identity(**baseline_args).build_fingerprint_sha256
        != create_build_identity(**changed_args).build_fingerprint_sha256
    )


def test_nonmaterial_incidental_values_are_not_build_identity_inputs(tmp_path: Path):
    args = _identity_args(tmp_path)

    identity = create_build_identity(**args)

    # There is deliberately nowhere to supply these values. They are attempt/publication
    # state, not inputs to create_build_identity.
    assert not {
        "output_dir",
        "created_by",
        "model_id",
        "model_version",
        "package_version",
        "model_run_id",
        "export_id",
        "timestamp",
    }.intersection(create_build_identity.__annotations__)
    assert stable_build_export_id(identity).startswith("build_")


def test_model_label_and_deployment_slot_are_not_fit_identity(tmp_path: Path):
    baseline_args = _identity_args(tmp_path / "baseline")
    changed_args = _identity_args(tmp_path / "changed")
    changed_args["model_config"] = replace(
        changed_args["model_config"],
        model_label="A clearer analyst label",
        deployment_slot="MOTOR_FREQ_UAT",
    )

    assert create_build_identity(**baseline_args) == create_build_identity(**changed_args)


def test_notebook_outputs_and_execution_metadata_do_not_change_model_source(
    tmp_path: Path,
):
    args = _identity_args(tmp_path)
    notebook_path = args["model_source_root"] / "pricing.ipynb"
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["model = make_model()\n"],
                "execution_count": 1,
                "outputs": [{"output_type": "stream", "text": ["first\n"]}],
                "metadata": {"collapsed": False},
            }
        ],
        "metadata": {"kernelspec": {"display_name": "Python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    baseline = create_build_identity(**args)

    notebook["cells"][0]["execution_count"] = 99
    notebook["cells"][0]["outputs"] = [
        {"output_type": "stream", "text": ["different output\n"]}
    ]
    notebook["metadata"] = {"widgets": {"state": {"incidental": "value"}}}
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

    assert create_build_identity(**args) == baseline


def test_exact_python_patch_version_is_material(tmp_path: Path, monkeypatch):
    args = _identity_args(tmp_path)
    baseline = create_build_identity(**args)
    monkeypatch.setattr(
        "pricing_pipeline.build_identity.platform.python_version",
        lambda: "99.88.77",
    )

    changed = create_build_identity(**args)

    assert changed.candidate_python_version == "99.88.77"
    assert changed.runtime_sha256 != baseline.runtime_sha256
    assert changed.build_fingerprint_sha256 != baseline.build_fingerprint_sha256


def test_numpy_and_native_split_definition_scalars_are_equivalent(tmp_path: Path):
    native_args = _identity_args(tmp_path / "native")
    numpy_args = _identity_args(tmp_path / "numpy")
    native_args["model_config"] = _model_config(
        ValidationSplitConfig.column_holdout(
            column="fold",
            train_values=(1, 2),
            test_values=(3,),
            materialize=True,
        )
    )
    numpy_args["model_config"] = _model_config(
        ValidationSplitConfig.column_holdout(
            column="fold",
            train_values=(np.int64(1), np.int64(2)),
            test_values=(np.int64(3),),
            materialize=True,
        )
    )
    native_args["split_indices"] = (_splits()[0],)
    numpy_args["split_indices"] = (_splits(np.int32)[0],)

    assert create_build_identity(**native_args) == create_build_identity(**numpy_args)


@pytest.mark.parametrize(
    "scoring",
    [
        lambda y, mu: 0.0,
        ("deviance", lambda y, mu: 0.0),
    ],
)
def test_publishable_build_identity_rejects_callable_scoring(tmp_path: Path, scoring):
    args = _identity_args(tmp_path)
    args["scoring"] = scoring

    with pytest.raises(BuildIdentityError, match="named string scorers"):
        create_build_identity(**args)


@pytest.mark.parametrize(
    "split_indices",
    [
        ((np.array([0, 2]), np.array([1, 1])),),
        ((np.array([0, 1]), np.array([1, 2])),),
        ((np.array([0, 1]), np.array([2, 4])),),
        ((np.array([[0, 1]]), np.array([2, 3])),),
        ((np.array([0.0, 1.0]), np.array([2, 3])),),
        (),
    ],
)
def test_build_identity_rejects_malformed_materialized_splits(
    tmp_path: Path,
    split_indices,
):
    args = _identity_args(tmp_path)
    args["split_indices"] = split_indices

    with pytest.raises(BuildIdentityError, match="validation split"):
        create_build_identity(**args)


def test_verify_build_identity_names_drifted_components(tmp_path: Path):
    args = _identity_args(tmp_path)
    expected = create_build_identity(**args)
    args["frame"].loc[0, "claim_count"] = 10.0

    with pytest.raises(
        BuildIdentityError,
        match="model_frame_sha256",
    ):
        verify_build_identity(expected, **args)


def test_verify_build_identity_accepts_an_unchanged_contract(tmp_path: Path):
    args = _identity_args(tmp_path)
    expected = create_build_identity(**args)

    assert verify_build_identity(expected, **args) is expected
