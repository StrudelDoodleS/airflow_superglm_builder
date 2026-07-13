from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.workbench.artifacts import CandidateBundle, save_candidate_bundle


def _api():
    try:
        return importlib.import_module("pricing_pipeline.workbench.core")
    except ModuleNotFoundError as exc:
        pytest.fail(f"candidate workbench API is not implemented: {exc}")


def _settings(tmp_path=None):
    root = "state/workbench_artifacts" if tmp_path is None else tmp_path
    return Settings(
        workbench_artifact_root=root,
        airflow_api_url="http://127.0.0.1:8080/api/v2",
    )


def _bundle():
    return CandidateBundle(
        fitted_model={"coef": [0.1]},
        X=pd.DataFrame({"age": [20.0, 30.0]}),
        y=np.array([0.0, 1.0]),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={"scope": "cv"},
        manifest_id="manifest-1",
        split_set_id="split-1",
        pk_columns=("policy_id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "NONE"},
    )


def _history_rows():
    return [
        {
            "model_name": "HOME_FREQ",
            "package_version": 13,
            "rate_package_id": 113,
            "parent_rate_package_id": None,
            "parent_package_version": None,
            "completed_ts": "2026-07-10T09:00:00",
            "data_as_of_date": "2026-06-30",
            "current_rate_package_id": 113,
            "baseline_cv_deviance": 0.482,
            "baseline_is_parent": False,
            "editor_training_delta": None,
            "model_run_id": 913,
            "run_status": "SUCCESS",
            "candidate_artifact_path": "/state/home/13/candidate.joblib",
            "candidate_artifact_sha256": "c" * 64,
            "candidate_artifact_format": "superglm-candidate-joblib-v1",
            "candidate_artifact_size_bytes": 100,
            "candidate_python_version": "3.14.4",
            "candidate_superglm_version": "0.11.0",
            "model_source_sha256": "d" * 64,
            "manifest_id": "manifest-13",
            "split_set_id": "split-13",
        },
        {
            "model_name": "HOME_FREQ",
            "package_version": 12,
            "rate_package_id": 112,
            "parent_rate_package_id": 111,
            "parent_package_version": 11,
            "completed_ts": "2026-07-03T12:00:00",
            "data_as_of_date": "2026-06-23",
            "current_rate_package_id": 113,
            "baseline_cv_deviance": 0.491,
            "baseline_is_parent": True,
            "baseline_metric_scope": "inherited_cv",
            "editor_training_delta": 0.009,
            "model_run_id": 912,
            "run_status": "SUCCESS",
            "candidate_artifact_path": "/state/home/12/candidate.joblib",
            "candidate_artifact_sha256": "e" * 64,
            "candidate_artifact_format": "superglm-candidate-joblib-v1",
            "candidate_artifact_size_bytes": 101,
            "candidate_python_version": "3.14.4",
            "candidate_superglm_version": "0.11.0",
            "model_source_sha256": "f" * 64,
            "manifest_id": "manifest-11",
            "split_set_id": "split-11",
        },
    ]


def test_candidates_returns_friendly_columns_and_hides_lineage_ids(monkeypatch):
    api = _api()
    workbench = api.Workbench(
        engine=object(),
        settings=_settings(),
        config_loader=lambda name: SimpleNamespace(deployment_slot="HOME_FREQ_UAT"),
    )
    monkeypatch.setattr(workbench, "_candidate_rows", lambda model_name, slot: _history_rows())

    history = workbench.candidates("HOME_FREQ")

    assert list(history.columns) == [
        "Package",
        "Fitted",
        "Data through",
        "Parent",
        "State",
        "Baseline pooled CV deviance",
        "Editor train delta",
        "Editor",
    ]
    assert "model_run_id" not in history.columns
    assert history.iloc[0]["State"] == "Champion in HOME_FREQ_UAT"
    assert history.iloc[0]["Baseline pooled CV deviance"] == pytest.approx(0.482)
    assert history.iloc[1]["State"] == "Edited candidate"
    assert history.iloc[1]["Baseline pooled CV deviance"] == "parent: 0.491"
    assert history.iloc[1]["Editor train delta"] == pytest.approx(0.009)


def test_candidates_can_return_explicit_technical_view(monkeypatch):
    api = _api()
    workbench = api.Workbench(
        engine=object(),
        settings=_settings(),
        config_loader=lambda name: SimpleNamespace(deployment_slot="HOME_FREQ_UAT"),
    )
    monkeypatch.setattr(workbench, "_candidate_rows", lambda model_name, slot: _history_rows())

    history = workbench.candidates("HOME_FREQ", technical=True)

    assert history.iloc[0]["model_run_id"] == 913
    assert history.iloc[0]["candidate_artifact_sha256"] == "c" * 64


def test_candidate_history_binds_validation_split_to_current_manifest():
    statements = []

    class Rows:
        def mappings(self):
            return self

        def all(self):
            return []

    class Connection:
        def execute(self, statement, params):
            statements.append((str(statement), params))
            return Rows()

    class Begin:
        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class Engine:
        def begin(self):
            return Begin()

    workbench = _api().Workbench(
        engine=Engine(),
        settings=_settings(),
        config_loader=lambda name: SimpleNamespace(deployment_slot="HOME_FREQ_UAT"),
    )

    assert workbench._candidate_rows("HOME_FREQ", "HOME_FREQ_UAT") == []
    assert "split_link.manifest_id = mr.manifest_id" in statements[0][0]
    assert "split_link.dataset_role = 'training'" in statements[0][0]


@pytest.mark.parametrize("current_rate_package_id", [113, None])
def test_workbench_reads_current_champion_at_decision_time(current_rate_package_id):
    statements = []

    class Result:
        def scalar_one_or_none(self):
            return current_rate_package_id

    class Connection:
        def execute(self, statement, params):
            statements.append((str(statement), params))
            return Result()

    class Begin:
        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class Engine:
        def begin(self):
            return Begin()

    workbench = _api().Workbench(
        engine=Engine(),
        settings=_settings(),
        config_loader=lambda name: SimpleNamespace(deployment_slot="HOME_FREQ_UAT"),
    )

    result = workbench.current_champion_rate_package_id(
        "HOME_FREQ",
        deployment_slot="home_freq_prod",
    )

    assert result == current_rate_package_id
    assert "effective_to_ts IS NULL" in statements[0][0]
    assert statements[0][1] == {
        "model_name": "HOME_FREQ",
        "deployment_slot": "HOME_FREQ_PROD",
    }


def test_workbench_resolves_reviewed_champion_from_child_revision_metadata():
    row = {
        "model_name": "HOME_FREQ",
        "rate_package_id": 108,
        "package_version": 8,
        "package_status": "PUBLISHED",
        "parent_rate_package_id": 107,
        "model_run_id": 908,
        "run_status": "SUCCESS",
        "revision_metadata_json": json.dumps(
            {
                "champion_comparison": {
                    "available": True,
                    "status": "COMPARED",
                    "deployment_slot": "home_freq_uat",
                    "rate_package_id": 107,
                    "reason": None,
                }
            }
        ),
    }

    class Rows:
        def mappings(self):
            return self

        def all(self):
            return [row]

    class Connection:
        def execute(self, statement, params):
            assert "revision_metadata_json" in str(statement)
            return Rows()

    class Begin:
        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class Engine:
        def begin(self):
            return Begin()

    workbench = _api().Workbench(
        engine=Engine(),
        settings=_settings(),
        config_loader=lambda name: SimpleNamespace(deployment_slot="HOME_FREQ_UAT"),
    )
    submission = SimpleNamespace(
        submission_id="submission-1",
        model_name="HOME_FREQ",
        parent_rate_package_id=107,
    )

    resolved = workbench.resolve_editor_publication(submission)

    assert resolved["reviewed_champion_status"] == "COMPARED"
    assert resolved["reviewed_champion_rate_package_id"] == 107
    assert resolved["reviewed_deployment_slot"] == "HOME_FREQ_UAT"


def test_open_resolves_one_successful_run_and_verifies_bundle(tmp_path, monkeypatch):
    api = _api()
    metadata = save_candidate_bundle(_bundle(), tmp_path / "candidate.joblib")
    row = {
        "model_name": "HOME_FREQ",
        "package_version": 7,
        "rate_package_id": 107,
        "parent_rate_package_id": None,
        "model_run_id": 907,
        "run_status": "SUCCESS",
        "candidate_artifact_path": metadata.path,
        "candidate_artifact_sha256": metadata.sha256,
        "candidate_artifact_format": metadata.format,
        "candidate_artifact_size_bytes": metadata.size_bytes,
        "candidate_python_version": metadata.python_version,
        "candidate_superglm_version": metadata.superglm_version,
        "model_source_sha256": "b" * 64,
        "manifest_id": "manifest-1",
        "split_set_id": "split-1",
    }
    workbench = api.Workbench(
        engine=object(),
        settings=_settings(tmp_path),
        config_loader=lambda name: SimpleNamespace(deployment_slot="HOME_FREQ_UAT"),
    )
    monkeypatch.setattr(
        workbench,
        "_resolve_candidate_rows",
        lambda model_name, package_version: [row],
    )

    candidate = workbench.open("HOME_FREQ", package_version=7)

    assert candidate.package_version == 7
    assert candidate.rate_package_id == 107
    assert candidate.model_run_id == 907
    assert candidate.bundle.manifest_id == "manifest-1"


def test_open_rejects_ambiguous_run_lineage(monkeypatch):
    api = _api()
    workbench = api.Workbench(
        engine=object(),
        settings=_settings(),
        config_loader=lambda name: SimpleNamespace(deployment_slot="HOME_FREQ_UAT"),
    )
    monkeypatch.setattr(
        workbench,
        "_resolve_candidate_rows",
        lambda model_name, package_version: [{}, {}],
    )

    with pytest.raises(api.CandidateLineageError, match="exactly one successful MODEL_RUN"):
        workbench.open("HOME_FREQ", package_version=7)


def test_from_runtime_hides_engine_construction(monkeypatch):
    api = _api()
    runtime = SimpleNamespace(
        settings=_settings(),
        get_engine=lambda: "configured-engine",
    )
    monkeypatch.setattr(
        api,
        "runtime_from_env_or_module",
        lambda runtime_module=None: runtime,
    )

    workbench = api.Workbench.from_runtime("work_runtime.database")

    assert workbench.engine == "configured-engine"
    assert workbench.settings is runtime.settings


def test_models_returns_injected_discovered_logical_names_without_sql_identifiers():
    loader_calls = []

    def load_model_names():
        loader_calls.append(True)
        return ("HOME_FREQ", "MOTOR_SEVERITY")

    workbench = _api().Workbench(
        engine=object(),
        settings=_settings(),
        config_loader=lambda name: SimpleNamespace(deployment_slot="unused"),
        model_names_loader=load_model_names,
    )

    assert workbench.models() == ["HOME_FREQ", "MOTOR_SEVERITY"]
    assert loader_calls == [True]
