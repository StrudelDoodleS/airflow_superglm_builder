from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.workbench.artifacts import CandidateBundle, save_candidate_bundle

MODEL_CONFIG = ModelBuildConfig(
    model_name="HOME_FREQ",
    model_label="Home frequency",
    target_name="claim_count",
    model_type="superglm_poisson",
    deployment_slot="HOME_FREQ_UAT",
)


def _api():
    try:
        return importlib.import_module("pricing_pipeline.workbench.core")
    except ModuleNotFoundError as exc:
        pytest.fail(f"candidate workbench API is not implemented: {exc}")


def _settings(tmp_path=None):
    root = "state/workbench_artifacts" if tmp_path is None else tmp_path
    return Settings(workbench_artifact_root=root)


def _bundle():
    return CandidateBundle(
        fitted_model={"coef": [0.1]},
        X=pd.DataFrame({"age": [20.0, 30.0]}),
        y=np.array([0.0, 1.0]),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={"scope": "cv"},
        model_name="HOME_FREQ",
        model_version="20260603",
        export_id="export-1",
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
            "package_status": "DRAFT",
            "package_version": 14,
        },
        {
            "model_name": "HOME_FREQ",
            "model_version": "20260603",
            "model_kind": "RAW",
            "export_id": "export-13",
            "package_version": 13,
            "package_status": "PUBLISHED",
            "rate_package_id": 113,
            "parent_rate_package_id": None,
            "parent_package_version": None,
            "completed_ts": "2026-07-10T09:00:00",
            "data_as_of_date": "2026-06-30",
            "current_deployment_id": 713,
            "current_rate_package_id": 113,
            "baseline_cv_deviance": 0.482,
            "baseline_is_parent": False,
            "editor_training_delta": None,
            "model_run_id": 913,
            "run_status": "SUCCESS",
            "candidate_artifact_path": "/state/home/13/candidate.joblib",
            "candidate_artifact_sha256": "c" * 64,
            "candidate_artifact_format": "superglm-candidate-joblib-v2",
            "candidate_artifact_size_bytes": 100,
            "candidate_python_version": "3.14.4",
            "candidate_superglm_version": "0.11.0",
            "model_source_sha256": "d" * 64,
            "manifest_id": "manifest-13",
            "split_set_id": "split-13",
        },
        {
            "model_name": "HOME_FREQ",
            "model_version": "20260603",
            "model_kind": "EDITOR_EDIT",
            "export_id": "editor-12",
            "package_version": 12,
            "package_status": "PUBLISHED",
            "rate_package_id": 112,
            "parent_rate_package_id": 111,
            "parent_package_version": 11,
            "completed_ts": "2026-07-03T12:00:00",
            "data_as_of_date": "2026-06-23",
            "current_deployment_id": 713,
            "current_rate_package_id": 113,
            "baseline_cv_deviance": 0.491,
            "baseline_is_parent": True,
            "baseline_metric_scope": "inherited_cv",
            "editor_training_delta": 0.009,
            "model_run_id": 912,
            "run_status": "SUCCESS",
            "candidate_artifact_path": "/state/home/12/candidate.joblib",
            "candidate_artifact_sha256": "e" * 64,
            "candidate_artifact_format": "superglm-candidate-joblib-v2",
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
        model_config=MODEL_CONFIG,
    )
    monkeypatch.setattr(workbench, "_candidate_rows", lambda model_name, slot: _history_rows())

    history = workbench.candidates("HOME_FREQ")

    assert list(history.columns) == [
        "Package",
        "Kind",
        "Fitted",
        "Data through",
        "Manifest",
        "Parent",
        "State",
        "Baseline pooled CV deviance",
        "Editor train delta",
        "Editor",
    ]
    assert history["Package"].tolist() == [13, 12]
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
        model_config=MODEL_CONFIG,
    )
    monkeypatch.setattr(workbench, "_candidate_rows", lambda model_name, slot: _history_rows())

    history = workbench.candidates("HOME_FREQ", technical=True)

    assert history["package_version"].tolist() == [13, 12]
    assert set(history["package_status"]) == {"PUBLISHED"}
    assert history.iloc[0]["model_run_id"] == 913
    assert history.iloc[0]["candidate_artifact_sha256"] == "c" * 64


def test_candidates_return_stable_empty_editor_and_deployment_views(monkeypatch):
    api = _api()
    workbench = api.Workbench(
        engine=object(),
        settings=_settings(),
        model_config=MODEL_CONFIG,
    )
    monkeypatch.setattr(workbench, "_candidate_rows", lambda model_name, slot: [])

    editor_history = workbench.candidates("HOME_FREQ")
    deployment_history = workbench.candidates("HOME_FREQ", technical=True)

    assert editor_history.empty
    assert list(editor_history.columns) == api._FRIENDLY_COLUMNS
    assert deployment_history.empty
    assert list(deployment_history.columns) == api._TECHNICAL_COLUMNS
    assert {
        "package_version",
        "package_status",
        "model_kind",
        "model_equivalence_sha256",
        "data_as_of_date",
        "manifest_id",
        "parent_rate_package_id",
        "current_deployment_id",
        "current_rate_package_id",
    } <= set(deployment_history.columns)


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
        model_config=MODEL_CONFIG,
    )

    assert (
        workbench._candidate_rows(
            "HOME_FREQ",
            "HOME_FREQ_UAT",
            package_version=7,
        )
        == []
    )
    assert "split_link.manifest_id = mr.manifest_id" in statements[0][0]
    assert "split_link.dataset_role = 'training'" in statements[0][0]
    assert "mr.model_version" in statements[0][0]
    assert "mr.export_id" in statements[0][0]
    assert "mr.publication_receipt_sha256" in statements[0][0]
    assert "rp.publication_receipt_sha256 AS package_publication_receipt_sha256" in statements[0][0]
    assert "manifest.model_frame_sha256" in statements[0][0]
    assert "deployment.deployment_id AS current_deployment_id" in statements[0][0]
    assert "rp.package_status = 'PUBLISHED'" in statements[0][0]
    assert "rp.package_version = :package_version" in statements[0][0]
    assert statements[0][1]["package_version"] == 7


@pytest.mark.parametrize("frame_digest", [None, "c" * 64])
def test_open_resolves_one_successful_run_and_verifies_bundle(
    tmp_path,
    monkeypatch,
    frame_digest,
):
    api = _api()
    bundle = replace(_bundle(), model_frame_sha256=frame_digest)
    metadata = save_candidate_bundle(bundle, tmp_path / "candidate.joblib")
    row = {
        "model_name": "HOME_FREQ",
        "model_version": "20260603",
        "export_id": "export-1",
        "package_version": 7,
        "package_status": "PUBLISHED",
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
        "model_frame_sha256": frame_digest,
    }
    workbench = api.Workbench(
        engine=object(),
        settings=_settings(tmp_path),
        model_config=MODEL_CONFIG,
    )
    monkeypatch.setattr(
        workbench,
        "_candidate_rows",
        lambda model_name, deployment_slot, *, package_version=None: [row],
    )

    candidate = workbench.open("HOME_FREQ", package_version=7)

    assert candidate.package_version == 7
    assert candidate.rate_package_id == 107
    assert candidate.model_run_id == 907
    assert candidate.bundle.manifest_id == "manifest-1"


@pytest.mark.parametrize(
    ("bundle_digest", "sql_digest"),
    [
        ("c" * 64, "d" * 64),
        (None, "d" * 64),
        ("c" * 64, None),
    ],
)
def test_open_rejects_candidate_bundle_model_frame_digest_mismatch(
    tmp_path,
    monkeypatch,
    bundle_digest,
    sql_digest,
):
    api = _api()
    bundle = replace(_bundle(), model_frame_sha256=bundle_digest)
    metadata = save_candidate_bundle(bundle, tmp_path / "candidate.joblib")
    row = {
        "model_name": "HOME_FREQ",
        "model_version": "20260603",
        "export_id": "export-1",
        "package_version": 7,
        "package_status": "PUBLISHED",
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
        "model_frame_sha256": sql_digest,
    }
    workbench = api.Workbench(
        engine=object(),
        settings=_settings(tmp_path),
        model_config=MODEL_CONFIG,
    )
    monkeypatch.setattr(
        workbench,
        "_candidate_rows",
        lambda model_name, deployment_slot, *, package_version=None: [row],
    )

    with pytest.raises(api.CandidateLineageError, match="model_frame_sha256"):
        workbench.open("HOME_FREQ", package_version=7)


@pytest.mark.parametrize("field_name", ["model_name", "model_version", "export_id"])
def test_open_rejects_candidate_bundle_model_identity_mismatch(
    tmp_path,
    monkeypatch,
    field_name,
):
    api = _api()
    bundle = replace(_bundle(), **{field_name: f"wrong-{field_name}"})
    metadata = save_candidate_bundle(bundle, tmp_path / "candidate.joblib")
    row = {
        "model_name": "HOME_FREQ",
        "model_version": "20260603",
        "export_id": "export-1",
        "package_version": 7,
        "package_status": "PUBLISHED",
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
        model_config=MODEL_CONFIG,
    )
    monkeypatch.setattr(
        workbench,
        "_candidate_rows",
        lambda model_name, deployment_slot, *, package_version=None: [row],
    )

    with pytest.raises(api.CandidateLineageError, match=field_name):
        workbench.open("HOME_FREQ", package_version=7)


def test_open_rejects_ambiguous_run_lineage(monkeypatch):
    api = _api()
    workbench = api.Workbench(
        engine=object(),
        settings=_settings(),
        model_config=MODEL_CONFIG,
    )
    monkeypatch.setattr(
        workbench,
        "_candidate_rows",
        lambda model_name, deployment_slot, *, package_version=None: [
            {"package_status": "PUBLISHED"},
            {"package_status": "PUBLISHED"},
        ],
    )

    with pytest.raises(api.CandidateLineageError, match="one successful MODEL_RUN"):
        workbench.open("HOME_FREQ", package_version=7)


def test_open_rejects_a_non_published_package(monkeypatch):
    api = _api()
    workbench = api.Workbench(
        engine=object(),
        settings=_settings(),
        model_config=MODEL_CONFIG,
    )
    monkeypatch.setattr(
        workbench,
        "_candidate_rows",
        lambda model_name, deployment_slot, *, package_version=None: [{"package_status": "DRAFT"}],
    )

    with pytest.raises(api.CandidateLineageError, match="published package"):
        workbench.open("HOME_FREQ", package_version=7)
