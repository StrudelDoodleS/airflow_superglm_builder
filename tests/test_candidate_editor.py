from __future__ import annotations

import errno
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.workbench.artifacts import CandidateBundle
from pricing_pipeline.workbench import load_verified_submission
from pricing_pipeline.workbench.submission import EditorSubmissionError


class FakeWidget:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeEditorSession:
    def __init__(self, *, fail_save: bool = False) -> None:
        self.widget_value = FakeWidget()
        self.saved_json_path: Path | None = None
        self.saved_json_paths: list[Path] = []
        self.fail_save = fail_save
        self.to_model_kwargs = None

    def widget(self):
        return self.widget_value

    def save(self, path) -> None:
        self.saved_json_path = Path(path)
        self.saved_json_paths.append(self.saved_json_path)
        if self.fail_save:
            raise RuntimeError("injected editor session save failure")
        self.saved_json_path.write_text('{"edits":["age"]}\n', encoding="utf-8")

    def to_model(self, **kwargs):
        self.to_model_kwargs = kwargs
        return {"edited": True, "coef": [0.25]}


class FakeAirflowClient:
    def __init__(self) -> None:
        self.triggered = []
        self.trigger_error: Exception | None = None
        self.run_state = "queued"
        self.triggering_user_name = None

    def trigger_dag(self, dag_id, *, run_id, conf):
        self.triggered.append(SimpleNamespace(dag_id=dag_id, run_id=run_id, conf=conf))
        if self.trigger_error is not None:
            raise self.trigger_error
        return SimpleNamespace(dag_id=dag_id, dag_run_id=run_id, state="queued", payload={})

    def get_dag_run(self, dag_id, run_id):
        return SimpleNamespace(
            dag_id=dag_id,
            dag_run_id=run_id,
            state=self.run_state,
            payload={"triggering_user_name": self.triggering_user_name},
        )

    def dag_run_ui_url(self, dag_id, run_id):
        return f"http://airflow.test/dags/{dag_id}/runs/{run_id}"


def _bundle() -> CandidateBundle:
    return CandidateBundle(
        fitted_model={"coef": [0.1]},
        X=pd.DataFrame({"age": [20.0, 30.0]}),
        y=np.array([0.0, 1.0]),
        sample_weight=np.array([2.0, 3.0]),
        offset=np.array([0.1, 0.2]),
        export_weight=np.array([10.0, 20.0]),
        cv_report={"pooled_scores": {"deviance": 0.482}},
        manifest_id="manifest-1",
        split_set_id="split-1",
        pk_columns=("policy_id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "OFFSET_SEPARATE"},
    )


def _candidate(tmp_path, *, session, airflow_client):
    from pricing_pipeline.workbench.core import Candidate, Workbench

    calls = []

    def session_factory(model, *, train_data, cv_report):
        calls.append((model, train_data, cv_report))
        return session

    settings = Settings(
        workbench_artifact_root=tmp_path,
        airflow_api_url="http://127.0.0.1:8080/api/v2",
    )
    workbench = Workbench(
        engine=object(),
        settings=settings,
        config_loader=lambda name: SimpleNamespace(deployment_slot="HOME_FREQ_UAT"),
        editor_session_factory=session_factory,
        airflow_client=airflow_client,
    )
    candidate = Candidate(
        workbench=workbench,
        model_name="HOME_FREQ",
        package_version=7,
        rate_package_id=107,
        parent_rate_package_id=None,
        model_run_id=907,
        bundle=_bundle(),
        technical={
            "candidate_artifact_path": str(tmp_path / "parent.joblib"),
            "candidate_artifact_sha256": "c" * 64,
            "model_source_sha256": "b" * 64,
        },
    )
    return candidate, calls


def test_candidate_retains_live_editor_session_until_submission(tmp_path):
    session = FakeEditorSession()
    airflow_client = FakeAirflowClient()
    candidate, factory_calls = _candidate(
        tmp_path,
        session=session,
        airflow_client=airflow_client,
    )

    widget = candidate.editor()
    same_widget = candidate.editor()
    submission = candidate.submit_edits(reason="Sparse-age market calibration")

    assert widget is session.widget_value
    assert same_widget is widget
    assert candidate.editor_session is session
    assert len(factory_calls) == 1
    assert factory_calls[0][0] is candidate.bundle.fitted_model
    assert factory_calls[0][1] == (
        candidate.bundle.X,
        candidate.bundle.y,
        candidate.bundle.sample_weight,
        candidate.bundle.offset,
    )
    assert factory_calls[0][2] == candidate.bundle.cv_report
    assert session.saved_json_path is not None
    assert len(session.saved_json_paths) == 1
    assert session.to_model_kwargs == {
        "X": candidate.bundle.X,
        "y": candidate.bundle.y,
        "sample_weight": candidate.bundle.sample_weight,
        "offset": candidate.bundle.offset,
    }
    assert submission.parent_rate_package_id == candidate.rate_package_id
    assert Path(submission.edited_model_path).exists()
    assert Path(submission.editor_session_path).exists()
    assert Path(submission.path).exists()
    assert submission.state == "queued"
    assert len(airflow_client.triggered) == 1
    trigger = airflow_client.triggered[0]
    assert trigger.dag_id == "pricing_publish_editor_candidate"
    assert trigger.conf == {
        "submission_path": submission.path,
        "submission_sha256": submission.sha256,
    }

    payload = json.loads(Path(submission.path).read_text(encoding="utf-8"))
    assert payload["reason"] == "Sparse-age market calibration"
    assert payload["parent_rate_package_id"] == 107
    assert payload["manifest_id"] == "manifest-1"
    assert payload["split_set_id"] == "split-1"
    assert payload["baseline_candidate_sha256"] == "c" * 64
    assert payload["claimed_identity"] == "prototype-local-not-authenticated"


def test_submission_promotes_one_complete_staging_directory_atomically(
    tmp_path,
    monkeypatch,
):
    from pricing_pipeline.workbench import submission as submission_module

    session = FakeEditorSession()
    candidate, _ = _candidate(
        tmp_path,
        session=session,
        airflow_client=FakeAirflowClient(),
    )
    candidate.editor()
    observed = {}
    real_rename = os.rename

    def inspect_then_promote(source, target):
        staged_directory = Path(source)
        final_directory = Path(target)
        assert not final_directory.exists()
        assert {path.name for path in staged_directory.iterdir()} == {
            "editor_session.json",
            "edited_model.joblib",
            "submission.json",
        }
        payload = json.loads(
            (staged_directory / "submission.json").read_text(encoding="utf-8")
        )
        assert Path(payload["editor_session_path"]) == (
            final_directory / "editor_session.json"
        )
        assert Path(payload["edited_model_path"]) == final_directory / "edited_model.joblib"
        observed["staging"] = staged_directory
        observed["final"] = final_directory
        return real_rename(source, target)

    monkeypatch.setattr(submission_module.os, "rename", inspect_then_promote)

    result = candidate.submit_edits(reason="Atomic market calibration")

    assert observed["final"] == Path(result.path).parent
    assert not observed["staging"].exists()
    assert len(session.saved_json_paths) == 1
    assert set(observed["final"].iterdir()) == {
        Path(result.editor_session_path),
        Path(result.edited_model_path),
        Path(result.path),
    }


@pytest.mark.parametrize("failure_point", ["session", "model", "json", "promotion"])
def test_failure_before_promotion_cleans_staging_without_final_state(
    tmp_path,
    monkeypatch,
    failure_point,
):
    from pricing_pipeline.workbench import submission as submission_module

    session = FakeEditorSession(fail_save=failure_point == "session")
    candidate, _ = _candidate(
        tmp_path,
        session=session,
        airflow_client=FakeAirflowClient(),
    )
    candidate.editor()

    def injected_failure(*args, **kwargs):
        raise RuntimeError(f"injected {failure_point} failure")

    if failure_point == "model":
        monkeypatch.setattr(submission_module, "_save_edited_model", injected_failure)
    elif failure_point == "json":
        monkeypatch.setattr(submission_module, "_write_json_atomic", injected_failure)
    elif failure_point == "promotion":
        monkeypatch.setattr(submission_module.os, "rename", injected_failure)

    with pytest.raises(RuntimeError, match=f"injected .*{failure_point}.*failure"):
        candidate.submit_edits(reason="Failure boundary")

    submissions_root = tmp_path / "HOME_FREQ" / "editor_submissions"
    assert list(submissions_root.iterdir()) == []
    assert len(session.saved_json_paths) == 1


def test_concurrent_identical_submission_loser_reuses_atomic_winner(
    tmp_path,
    monkeypatch,
):
    from pricing_pipeline.workbench import submission as submission_module

    session = FakeEditorSession()
    airflow_client = FakeAirflowClient()
    candidate, _ = _candidate(
        tmp_path,
        session=session,
        airflow_client=airflow_client,
    )
    candidate.editor()
    real_rename = os.rename
    winner_paths = []

    def simulate_concurrent_winner(source, target):
        real_rename(source, target)
        winner_paths.append(Path(target))
        raise FileExistsError(errno.EEXIST, "simulated concurrent winner", target)

    monkeypatch.setattr(submission_module.os, "rename", simulate_concurrent_winner)

    submission = candidate.submit_edits(reason="Concurrent calibration")

    assert winner_paths == [Path(submission.path).parent]
    assert len(session.saved_json_paths) == 1
    assert len(airflow_client.triggered) == 1
    assert Path(submission.path).is_file()
    assert Path(submission.editor_session_path).is_file()
    assert Path(submission.edited_model_path).is_file()


def test_incomplete_deterministic_directory_is_recovered_without_touching_siblings(
    tmp_path,
):
    session = FakeEditorSession()
    candidate, _ = _candidate(
        tmp_path,
        session=session,
        airflow_client=FakeAirflowClient(),
    )
    candidate.editor()
    first = candidate.submit_edits(reason="Recover interrupted submission")
    final_directory = Path(first.path).parent
    submissions_root = final_directory.parent
    shutil.rmtree(final_directory)
    final_directory.mkdir()
    (final_directory / "legacy.partial").write_text("interrupted", encoding="utf-8")
    sibling = submissions_root / "do-not-touch"
    sibling.mkdir()
    (sibling / "sentinel").write_text("preserve", encoding="utf-8")
    save_attempts_before_retry = len(session.saved_json_paths)

    recovered = candidate.submit_edits(reason="Recover interrupted submission")

    assert recovered.submission_id == first.submission_id
    assert {path.name for path in final_directory.iterdir()} == {
        "editor_session.json",
        "edited_model.joblib",
        "submission.json",
    }
    assert (sibling / "sentinel").read_text(encoding="utf-8") == "preserve"
    assert len(session.saved_json_paths) == save_attempts_before_retry + 1


def test_trigger_failure_leaves_complete_submission_for_deterministic_retry(tmp_path):
    session = FakeEditorSession()
    airflow_client = FakeAirflowClient()
    airflow_client.trigger_error = RuntimeError("injected trigger failure")
    candidate, _ = _candidate(
        tmp_path,
        session=session,
        airflow_client=airflow_client,
    )
    candidate.editor()

    with pytest.raises(RuntimeError, match="injected trigger failure"):
        candidate.submit_edits(reason="Retry trigger safely")

    submissions_root = tmp_path / "HOME_FREQ" / "editor_submissions"
    final_directories = list(submissions_root.glob("submission-*"))
    assert len(final_directories) == 1
    assert {path.name for path in final_directories[0].iterdir()} == {
        "editor_session.json",
        "edited_model.joblib",
        "submission.json",
    }
    airflow_client.trigger_error = None

    retried = candidate.submit_edits(reason="Retry trigger safely")

    assert Path(retried.path).parent == final_directories[0]
    assert len(session.saved_json_paths) == 2
    assert len(airflow_client.triggered) == 2


def test_submission_root_cannot_escape_configured_artifact_root(tmp_path):
    configured_root = tmp_path / "configured"
    candidate, _ = _candidate(
        configured_root,
        session=FakeEditorSession(),
        airflow_client=FakeAirflowClient(),
    )
    candidate.model_name = "../outside"
    candidate.editor()

    with pytest.raises(EditorSubmissionError, match="outside configured artifact root"):
        candidate.submit_edits(reason="Contain submission artifacts")

    assert not (tmp_path / "outside").exists()


def test_public_submission_loader_rejects_omitted_allowed_root(tmp_path):
    candidate, _ = _candidate(
        tmp_path,
        session=FakeEditorSession(),
        airflow_client=FakeAirflowClient(),
    )
    candidate.editor()
    submission = candidate.submit_edits(reason="Boundary contract")

    with pytest.raises(TypeError, match="allowed_root"):
        load_verified_submission(
            Path(submission.path).resolve(),
            submission.sha256,
        )


def test_public_submission_loader_rejects_absolute_path_outside_allowed_root(tmp_path):
    candidate, _ = _candidate(
        tmp_path,
        session=FakeEditorSession(),
        airflow_client=FakeAirflowClient(),
    )
    candidate.editor()
    submission = candidate.submit_edits(reason="Boundary contract")
    configured_root = tmp_path / "different-configured-root"

    with pytest.raises(EditorSubmissionError, match="outside configured artifact root"):
        load_verified_submission(
            Path(submission.path).resolve(),
            submission.sha256,
            allowed_root=configured_root,
        )


def test_submit_requires_reason_and_an_open_live_session(tmp_path):
    candidate, _ = _candidate(
        tmp_path,
        session=FakeEditorSession(),
        airflow_client=FakeAirflowClient(),
    )

    with pytest.raises(ValueError, match="reason"):
        candidate.submit_edits(reason="   ")
    with pytest.raises(RuntimeError, match="editor"):
        candidate.submit_edits(reason="Market calibration")


def test_close_editor_stops_local_widget_server_and_discards_session(tmp_path):
    session = FakeEditorSession()
    candidate, _ = _candidate(
        tmp_path,
        session=session,
        airflow_client=FakeAirflowClient(),
    )
    candidate.editor()

    candidate.close_editor()

    assert session.widget_value.closed is True
    assert candidate.editor_session is None
    assert candidate.editor_widget is None


def test_submission_status_resolves_published_child_and_requests_deployment(tmp_path):
    session = FakeEditorSession()
    airflow_client = FakeAirflowClient()
    candidate, _ = _candidate(
        tmp_path,
        session=session,
        airflow_client=airflow_client,
    )
    candidate.editor()
    submission = candidate.submit_edits(reason="Sparse-age market calibration")
    candidate.workbench.resolve_editor_publication = lambda loaded_submission: {
        "model_name": "HOME_FREQ",
        "rate_package_id": 108,
        "package_version": 8,
        "model_run_id": 908,
        "package_status": "PUBLISHED",
    }
    airflow_client.run_state = "success"
    airflow_client.triggering_user_name = "analyst@example.test"

    status = submission.status()
    deployment_run = submission.request_deployment(
        reason="Approved market calibration",
    )

    assert status.state == "published"
    assert status.model_name == "HOME_FREQ"
    assert status.package_version == 8
    assert status.rate_package_id == 108
    assert status.airflow_url.endswith(
        "/dags/pricing_publish_editor_candidate/runs/"
        f"manual__{submission.submission_id}"
    )
    assert deployment_run.state == "queued"
    trigger = airflow_client.triggered[-1]
    assert trigger.dag_id == "pricing_deploy_rate_package"
    assert trigger.conf == {
        "model_name": "HOME_FREQ",
        "package_version": 8,
        "deployment_slot": "HOME_FREQ_UAT",
        "deployment_reason": "Approved market calibration",
    }


def test_deployment_request_requires_published_submission_and_reason(tmp_path):
    candidate, _ = _candidate(
        tmp_path,
        session=FakeEditorSession(),
        airflow_client=FakeAirflowClient(),
    )
    candidate.editor()
    submission = candidate.submit_edits(reason="Market calibration")

    with pytest.raises(RuntimeError, match="published"):
        submission.request_deployment(reason="Approved")
    with pytest.raises(ValueError, match="reason"):
        submission.request_deployment(reason=" ")


def test_identical_submission_retry_reuses_immutable_artifacts(tmp_path):
    airflow_client = FakeAirflowClient()
    candidate, _ = _candidate(
        tmp_path,
        session=FakeEditorSession(),
        airflow_client=airflow_client,
    )
    candidate.editor()

    first = candidate.submit_edits(reason="Market calibration")
    retried = candidate.submit_edits(reason="Market calibration")

    assert retried.submission_id == first.submission_id
    assert retried.path == first.path
    assert retried.sha256 == first.sha256
    assert Path(first.path).is_file()
    assert Path(first.editor_session_path).is_file()
    assert Path(first.edited_model_path).is_file()
    assert len(airflow_client.triggered) == 2


def test_incompatible_submission_retry_preserves_existing_artifacts(tmp_path):
    candidate, _ = _candidate(
        tmp_path,
        session=FakeEditorSession(),
        airflow_client=FakeAirflowClient(),
    )
    candidate.editor()
    first = candidate.submit_edits(reason="First rationale")

    with pytest.raises(EditorSubmissionError, match="already exists"):
        candidate.submit_edits(reason="Different rationale")

    assert Path(first.path).is_file()
    assert Path(first.editor_session_path).is_file()
    assert Path(first.edited_model_path).is_file()
