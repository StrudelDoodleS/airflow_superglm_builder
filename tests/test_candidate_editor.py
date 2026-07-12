from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.workbench.artifacts import CandidateBundle


class FakeWidget:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeEditorSession:
    def __init__(self) -> None:
        self.widget_value = FakeWidget()
        self.saved_json_path: Path | None = None
        self.to_model_kwargs = None

    def widget(self):
        return self.widget_value

    def save(self, path) -> None:
        self.saved_json_path = Path(path)
        self.saved_json_path.write_text('{"edits":["age"]}\n', encoding="utf-8")

    def to_model(self, **kwargs):
        self.to_model_kwargs = kwargs
        return {"edited": True, "coef": [0.25]}


class FakeAirflowClient:
    def __init__(self) -> None:
        self.triggered = []

    def trigger_dag(self, dag_id, *, run_id, conf):
        self.triggered.append(SimpleNamespace(dag_id=dag_id, run_id=run_id, conf=conf))
        return SimpleNamespace(dag_id=dag_id, dag_run_id=run_id, state="queued", payload={})


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
    assert session.saved_json_path.exists()
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
