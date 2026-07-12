from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pricing_pipeline.data.manifest import ModelFrameManifestSpec
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.modeling.standard_superglm import ModelInputs
from pricing_pipeline.models.config import ValidationSplitConfig
from scripts.scaffold_pricing_model import ScaffoldOptions, scaffold_pricing_model


class SmokeModel:
    def fit_reml(self, X, y, sample_weight=None, offset=None):
        self.columns = tuple(X.columns)
        self.mean = float(np.mean(y))
        return self

    def training_telemetry(self):
        return {"converged": True, "n_iter": 1}

    def predict(self, X, offset=None):
        prediction = np.full(len(X), max(self.mean, 0.01), dtype=float)
        if offset is not None:
            prediction *= np.exp(np.asarray(offset))
        return prediction


class SmokeEditorSession:
    def __init__(self, model):
        self.model = model
        self.widget_value = SimpleNamespace(close=lambda: None)

    def widget(self):
        return self.widget_value

    def save(self, path):
        Path(path).write_text('{"format":"smoke-editor","edits":[]}\n', encoding="utf-8")

    def to_model(self, **kwargs):
        return self.model


class SmokeAirflowClient:
    def __init__(self):
        self.triggered = []

    def trigger_dag(self, dag_id, *, run_id, conf):
        run = SimpleNamespace(
            dag_id=dag_id,
            dag_run_id=run_id,
            run_id=run_id,
            conf=conf,
            state="queued",
            payload={},
        )
        self.triggered.append(run)
        return run


def _cv_result(folds):
    return SimpleNamespace(
        fold_scores=pd.DataFrame(
            {
                "fold": [0, 1],
                "converged": [True, True],
                "deviance": [0.4, 0.5],
            }
        ),
        mean_scores={"deviance": 0.45},
        pooled_scores={"deviance": 0.44},
        std_scores={"deviance": 0.05},
        fold_indices=folds,
        oof_predictions=np.array([0.25, 0.25, 0.25, 0.25]),
    )


def run_scaffolded_workflow_smoke(tmp_path, monkeypatch):
    from pricing_pipeline.modeling import standard_superglm
    from pricing_pipeline.workbench.core import Workbench

    scaffold_root = tmp_path / "scaffold"
    scaffold_pricing_model(
        ScaffoldOptions(
            model_name="SMOKE_FREQ",
            model_label="Smoke frequency",
            target_name="claim_count",
            root=scaffold_root,
        )
    )
    source_root = scaffold_root / "pricing_models" / "smoke_freq"
    assert source_root.is_dir()

    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3, 4],
            "age": [20.0, 30.0, 40.0, 50.0],
            "claim_count": [0.0, 1.0, 0.0, 1.0],
        }
    )
    inputs = ModelInputs(
        X=frame[["age"]],
        y=frame["claim_count"].to_numpy(),
    )
    folds = [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
    ]

    def fake_export(model, X, y, exposure, output_path, **kwargs):
        del model, X, y, exposure, kwargs
        Path(output_path).write_bytes(b"smoke canonical workbook")
        return Path(output_path)

    monkeypatch.setattr(standard_superglm, "export_rating_tables", fake_export)
    monkeypatch.setattr(
        standard_superglm,
        "build_superglm_publication_receipt",
        lambda *args, **kwargs: object(),
    )

    def write_receipt(receipt, path):
        del receipt
        Path(path).write_bytes(b"smoke receipt")
        return "c" * 64

    monkeypatch.setattr(
        standard_superglm,
        "write_publication_receipt",
        write_receipt,
    )
    monkeypatch.setattr(
        standard_superglm,
        "create_model_frame_manifest_with_split",
        lambda engine, **kwargs: SimpleNamespace(
            manifest_id="manifest-smoke",
            split_set_id="split-smoke",
        ),
    )
    settings = Settings(
        workbench_artifact_root=tmp_path / "workbench",
        validation_split_artifact_root=tmp_path / "splits",
        airflow_api_url="http://127.0.0.1:8080/api/v2",
    )
    build = standard_superglm.run_standard_superglm_build(
        object(),
        frame=frame,
        inputs=inputs,
        model_factory=SmokeModel,
        split_indices=folds,
        fit_mode="fit_reml",
        scoring=("deviance",),
        output_dir=settings.workbench_artifact_root / "SMOKE_FREQ" / "scheduled_1",
        model_name="SMOKE_FREQ",
        model_version="v1",
        export_id="scheduled_1",
        effective_from="2026-07-12",
        manifest_spec=ModelFrameManifestSpec(
            dataset_name="smoke_frame",
            source_system="pytest",
            data_as_of_date="2026-06-30",
            pk_columns=("policy_id",),
            target_column="claim_count",
        ),
        validation_split=ValidationSplitConfig.custom(materialize=True),
        split_artifact_root=settings.validation_split_artifact_root,
        model_source_root=source_root,
        created_by="pytest",
        cross_validate_fn=lambda *args, **kwargs: _cv_result(folds),
    )
    completed = build.completed_build
    airflow_client = SmokeAirflowClient()
    workbench = Workbench(
        engine=object(),
        settings=settings,
        config_loader=lambda name: SimpleNamespace(deployment_slot="SMOKE_FREQ_UAT"),
        editor_session_factory=lambda model, **kwargs: SmokeEditorSession(model),
        airflow_client=airflow_client,
    )
    candidate_row = {
        "model_name": "SMOKE_FREQ",
        "package_version": 1,
        "rate_package_id": 101,
        "parent_rate_package_id": None,
        "model_run_id": 901,
        "run_status": "SUCCESS",
        "candidate_artifact_path": completed["candidate_artifact_path"],
        "candidate_artifact_sha256": completed["candidate_artifact_sha256"],
        "candidate_artifact_format": completed["candidate_artifact_format"],
        "candidate_artifact_size_bytes": completed["candidate_artifact_size_bytes"],
        "candidate_python_version": completed["candidate_python_version"],
        "candidate_superglm_version": completed["candidate_superglm_version"],
        "model_source_sha256": completed["model_source_sha256"],
        "manifest_id": "manifest-smoke",
        "split_set_id": "split-smoke",
    }
    workbench._resolve_candidate_rows = lambda model_name, package_version: [candidate_row]
    candidate = workbench.open("SMOKE_FREQ", package_version=1)
    widget = candidate.editor()
    submission = candidate.submit_edits(reason="Smoke market calibration")
    return SimpleNamespace(
        scheduled_candidate=SimpleNamespace(
            bundle_verified=candidate.bundle.manifest_id == "manifest-smoke"
        ),
        editor_session_opened=widget is not None and candidate.editor_session is not None,
        submission=submission,
        airflow_client=airflow_client,
    )


def test_scaffolded_workflow_builds_opens_edits_and_submits(tmp_path, monkeypatch):
    result = run_scaffolded_workflow_smoke(tmp_path, monkeypatch)

    assert result.scheduled_candidate.bundle_verified is True
    assert result.editor_session_opened is True
    assert result.submission.dag_id == "pricing_publish_editor_candidate"
    assert result.submission.parent_package_version == 1
    assert result.airflow_client.triggered[-1].conf == {
        "submission_path": result.submission.path,
        "submission_sha256": result.submission.sha256,
    }
