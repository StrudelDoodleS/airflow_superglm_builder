from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from superglm import Categorical, Numeric, SuperGLM

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.modeling.manual_adjustment import (
    ManualAdjustmentPolicy,
    ManualAdjustmentRule,
    apply_manual_adjustment_policy,
    manual_adjustment_policy_from_candidate,
)
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.notebook import NotebookContext, publish_manual_adjustment
from pricing_pipeline.workbench.artifacts import CandidateBundle
from pricing_pipeline.workbench.core import Candidate, Workbench


def _candidate(tmp_path: Path) -> Candidate:
    frame = pd.DataFrame(
        {
            "segment": ["A", "B", "C"] * 20,
            "x": np.tile([0.0, 1.0, 2.0], 20),
        }
    )
    target = np.tile([1.0, 2.0, 3.0], 20)
    model = SuperGLM(
        features={"segment": Categorical(base="first"), "x": Numeric()},
        selection_penalty=0.0,
    ).fit(frame, target)
    engine = object()
    workbench = Workbench(
        engine=engine,
        settings=Settings(workbench_artifact_root=tmp_path),
        model_config=ModelBuildConfig(
            model_name="TEST_FREQ",
            model_label="Test frequency",
            target_name="target",
            model_type="superglm_poisson",
            deployment_slot="TEST_FREQ_PROD",
        ),
    )
    bundle = CandidateBundle(
        fitted_model=model,
        X=frame,
        y=target,
        sample_weight=None,
        offset=None,
        export_weight=np.ones(len(frame)),
        export_weight_name="exposure",
        cv_report={},
        model_name="TEST_FREQ",
        model_version="v1",
        export_id="export-1",
        manifest_id="manifest-1",
        split_set_id="split-1",
        pk_columns=("row_id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        model_frame_sha256="c" * 64,
        offset_contract={"handling": "NONE"},
    )
    return Candidate(
        workbench=workbench,
        model_name="TEST_FREQ",
        package_version=7,
        rate_package_id=107,
        parent_rate_package_id=None,
        model_run_id=907,
        bundle=bundle,
        technical={
            "candidate_artifact_sha256": "d" * 64,
            "model_id": 17,
            "model_kind": "ROUTINE_EDIT",
            "data_as_of_date": "2026-07-31",
            "current_rate_package_id": 107,
        },
    )


def _policy() -> ManualAdjustmentPolicy:
    return ManualAdjustmentPolicy(
        name="market adjustment",
        version=1,
        reason="Approved market response",
        carry_forward=True,
        rules=(
            ManualAdjustmentRule.multiply_levels(
                "segment",
                ["B"],
                1.05,
                reason="Selected segment uplift",
            ),
        ),
    )


def test_relative_policy_changes_only_selected_level_and_reports_impact(tmp_path):
    candidate = _candidate(tmp_path)

    review = apply_manual_adjustment_policy(candidate, _policy())

    baseline = candidate.bundle.fitted_model.predict(candidate.bundle.X)
    edited = review.edited_model.predict(candidate.bundle.X)
    ratio = edited / baseline
    assert ratio[candidate.bundle.X["segment"].eq("B")].tolist() == pytest.approx([1.05] * 20)
    assert ratio[candidate.bundle.X["segment"].ne("B")].tolist() == pytest.approx([1.0] * 40)
    assert review.impact["Source package"] == 7
    assert review.impact["Data as-at"] == "2026-07-31"
    assert review.impact["Changed rows"] == 20
    assert review.rules.loc[0, "Change"] == "+5.00%"


def test_policy_payload_is_canonical_replayable_and_rejects_overlap():
    policy = _policy()

    reloaded = ManualAdjustmentPolicy.from_payload(policy.to_payload())

    assert reloaded == policy
    assert reloaded.sha256 == policy.sha256
    assert len(policy.sha256) == 64
    with pytest.raises(ValueError, match="overlap"):
        ManualAdjustmentPolicy(
            name="bad",
            version=1,
            reason="Duplicate scope",
            rules=(
                ManualAdjustmentRule.multiply_levels("segment", ["A", "B"], 1.05, reason="First"),
                ManualAdjustmentRule.multiply_levels("segment", ["B", "C"], 0.95, reason="Second"),
            ),
        )


def test_policy_refuses_missing_levels_instead_of_silently_skipping(tmp_path):
    candidate = _candidate(tmp_path)
    policy = ManualAdjustmentPolicy(
        name="invalid level",
        version=1,
        reason="Exercise validation",
        rules=(
            ManualAdjustmentRule.multiply_levels(
                "segment", ["missing"], 1.05, reason="Should fail"
            ),
        ),
    )

    with pytest.raises(KeyError, match="Unknown level"):
        apply_manual_adjustment_policy(candidate, policy)


def test_policy_round_trips_through_published_revision_metadata(tmp_path):
    candidate = _candidate(tmp_path)
    policy = _policy()
    candidate.technical.update(
        model_kind="MANUAL_EDIT",
        revision_metadata_json=json.dumps(
            {
                "edit_metadata": {
                    "manual_adjustment_policy": policy.to_payload(),
                    "manual_adjustment_policy_sha256": policy.sha256,
                }
            }
        ),
    )

    assert manual_adjustment_policy_from_candidate(candidate) == policy

    metadata = json.loads(candidate.technical["revision_metadata_json"])
    metadata["edit_metadata"]["manual_adjustment_policy_sha256"] = "0" * 64
    candidate.technical["revision_metadata_json"] = json.dumps(metadata)
    with pytest.raises(ValueError, match="SHA-256"):
        manual_adjustment_policy_from_candidate(candidate)


def test_notebook_publication_marks_manual_kind_and_embeds_policy(monkeypatch, tmp_path):
    candidate = _candidate(tmp_path)
    review = apply_manual_adjustment_policy(candidate, _policy())
    context = NotebookContext(
        engine=candidate.workbench.engine,
        settings=candidate.workbench.settings,
        mode="remote",
        write_allowed=True,
        destination="remote:test",
    )
    captured = {}

    def fake_save(candidate_arg, **kwargs):
        captured["candidate"] = candidate_arg
        captured["save"] = kwargs
        return SimpleNamespace(
            path="submission.json",
            sha256="e" * 64,
            submission_id="submission-1",
        )

    def fake_publish(engine, **kwargs):
        captured["engine"] = engine
        captured["publish"] = kwargs
        return SimpleNamespace(model_kind="MANUAL_EDIT")

    monkeypatch.setattr("pricing_pipeline.notebook.save_editor_submission", fake_save)
    monkeypatch.setattr("pricing_pipeline.notebook.publish_editor_submission", fake_publish)
    monkeypatch.setattr("pricing_pipeline.notebook._created_by", lambda value: "analyst")

    result = publish_manual_adjustment(context, review=review)

    assert result.model_kind == "MANUAL_EDIT"
    assert captured["candidate"] is candidate
    assert captured["save"]["model_kind"] == "MANUAL_EDIT"
    metadata = captured["save"]["edit_metadata"]
    assert metadata["manual_adjustment_policy"] == review.policy.to_payload()
    assert metadata["manual_adjustment_policy_sha256"] == review.policy.sha256
    assert captured["publish"]["dag_id"] == "notebook_publish_manual_adjustment"
