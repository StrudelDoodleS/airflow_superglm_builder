from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.workbench.artifacts import CandidateBundle
from pricing_pipeline.workbench.core import Candidate, Workbench
from pricing_pipeline.workbench.submission import (
    EditorSubmissionError,
    load_verified_submission,
    save_editor_submission,
)


class FakeEditorSession:
    def __init__(self, reference_model, *, fail_save: bool = False) -> None:
        self.reference_model = reference_model
        self.fail_save = fail_save
        self.saved_paths: list[Path] = []

    def save(self, path) -> None:
        target = Path(path)
        self.saved_paths.append(target)
        if self.fail_save:
            raise RuntimeError("injected save failure")
        target.write_text('{"edits":["age"]}\n', encoding="utf-8")


def _bundle() -> CandidateBundle:
    return CandidateBundle(
        fitted_model={"coef": [0.1]},
        X=pd.DataFrame({"age": [20.0, 30.0]}),
        y=np.array([0.0, 1.0]),
        sample_weight=np.array([2.0, 3.0]),
        offset=np.array([0.1, 0.2]),
        export_weight=np.array([10.0, 20.0]),
        cv_report={"pooled_scores": {"deviance": 0.482}},
        model_name="HOME_FREQ",
        model_version="v1",
        export_id="export-1",
        manifest_id="manifest-1",
        split_set_id="split-1",
        pk_columns=("policy_id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        build_fingerprint_sha256="c" * 64,
        builder_source_sha256="d" * 64,
        materialized_split_sha256="e" * 64,
        runtime_sha256="f" * 64,
        candidate_superglm_sha256="0" * 64,
        offset_contract={
            "handling": "ALREADY_APPLIED_SQL_EXPOSURE",
            "source_name": "Exposure",
            "label": "log(Exposure)",
        },
    )


def _candidate(tmp_path: Path) -> Candidate:
    workbench = Workbench(
        engine=object(),
        settings=Settings(workbench_artifact_root=tmp_path),
        model_config=ModelBuildConfig(
            model_name="HOME_FREQ",
            model_label="Home frequency",
            target_name="claim_count",
            model_type="superglm_poisson",
            deployment_slot="HOME_FREQ_UAT",
        ),
    )
    return Candidate(
        workbench=workbench,
        model_name="HOME_FREQ",
        package_version=7,
        rate_package_id=107,
        parent_rate_package_id=None,
        model_run_id=907,
        bundle=_bundle(),
        technical={"candidate_artifact_sha256": "c" * 64},
    )


def _save(candidate: Candidate, session: FakeEditorSession, reason: str = "Market review"):
    return save_editor_submission(
        candidate,
        editor_session=session,
        reason=reason,
        claimed_identity="analyst@example",
    )


def test_editor_session_is_explicit_and_saved_without_a_second_model_artifact(tmp_path):
    candidate = _candidate(tmp_path)
    session = FakeEditorSession(candidate.bundle.fitted_model)

    assert not hasattr(Candidate, "editor")
    assert not hasattr(Candidate, "close_editor")
    assert not hasattr(candidate, "editor_session")
    assert not hasattr(candidate, "editor_widget")
    assert not hasattr(Workbench, "create_editor_session")
    submission = _save(candidate, session)

    payload = json.loads(Path(submission.path).read_text(encoding="utf-8"))
    assert len(session.saved_paths) == 1
    assert Path(submission.editor_session_path).is_file()
    assert not (Path(submission.path).parent / "edited_model.joblib").exists()
    assert payload["parent_rate_package_id"] == 107
    assert payload["manifest_id"] == "manifest-1"
    assert payload["split_set_id"] == "split-1"
    assert payload["baseline_candidate_sha256"] == "c" * 64
    assert payload["claimed_identity"] == "analyst@example"


def test_submission_loader_verifies_manifest_and_editor_session_hashes(tmp_path):
    candidate = _candidate(tmp_path)
    session = FakeEditorSession(candidate.bundle.fitted_model)
    submission = _save(candidate, session)

    loaded = load_verified_submission(
        submission.path,
        submission.sha256,
        allowed_root=tmp_path,
    )
    assert loaded.submission_id == submission.submission_id

    Path(submission.editor_session_path).write_text("tampered", encoding="utf-8")
    with pytest.raises(EditorSubmissionError, match="SHA-256"):
        load_verified_submission(
            submission.path,
            submission.sha256,
            allowed_root=tmp_path,
        )


def test_submission_rejects_blank_reason_and_identity(tmp_path):
    candidate = _candidate(tmp_path)
    session = FakeEditorSession(candidate.bundle.fitted_model)

    with pytest.raises(ValueError, match="reason"):
        save_editor_submission(
            candidate,
            editor_session=session,
            reason=" ",
            claimed_identity="analyst@example",
        )
    with pytest.raises(ValueError, match="claimed_identity"):
        save_editor_submission(
            candidate,
            editor_session=session,
            reason="Market review",
            claimed_identity=" ",
        )


def test_identical_save_reuses_immutable_submission_and_rejects_new_reason(tmp_path):
    candidate = _candidate(tmp_path)
    session = FakeEditorSession(candidate.bundle.fitted_model)

    first = _save(candidate, session)
    retried = _save(candidate, session)

    assert retried.path == first.path
    assert retried.sha256 == first.sha256
    with pytest.raises(EditorSubmissionError, match="incompatible metadata: reason"):
        _save(candidate, session, reason="Different rationale")


def test_failed_session_save_leaves_no_submission(tmp_path):
    candidate = _candidate(tmp_path)
    session = FakeEditorSession(candidate.bundle.fitted_model, fail_save=True)

    with pytest.raises(RuntimeError, match="injected save failure"):
        _save(candidate, session)

    submission_root = tmp_path / "HOME_FREQ" / "editor_submissions"
    assert not list(submission_root.glob("submission-*"))
    assert not list(submission_root.glob(".submission-*"))


def test_submission_rejects_session_from_another_model_before_writing(tmp_path):
    candidate = _candidate(tmp_path)
    session = FakeEditorSession(object())

    with pytest.raises(EditorSubmissionError, match="reference_model"):
        _save(candidate, session)

    assert session.saved_paths == []
    assert not (tmp_path / "HOME_FREQ" / "editor_submissions").exists()
