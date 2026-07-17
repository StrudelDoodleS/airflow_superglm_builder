from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.models.spec import CompletedModelBuild, CompletedModelBuildError
from pricing_pipeline.orchestration.publish_completed_build import (
    CandidateSQLLineage,
    _verify_candidate_artifact,
)
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import ModelExportResult
from pricing_pipeline.orchestration import pipeline
from pricing_pipeline.publishing.sqlite_notebook import _publish_sqlite_candidate_locked
from pricing_pipeline.workbench.artifacts import CandidateBundle, save_candidate_bundle


def _candidate(tmp_path) -> tuple[CandidateBundle, CompletedModelBuild]:
    bundle = CandidateBundle(
        fitted_model={"coef": [0.1]},
        X=pd.DataFrame({"age": [20.0]}),
        y=np.array([0.0]),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={},
        model_name="CLAIM_FREQ",
        model_version="20260603",
        export_id="export-1",
        manifest_id="manifest-1",
        split_set_id="split-1",
        pk_columns=("policy_id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        model_frame_sha256="c" * 64,
        build_fingerprint_sha256="d" * 64,
        builder_source_sha256="e" * 64,
        materialized_split_sha256="f" * 64,
        runtime_sha256="0" * 64,
        candidate_superglm_sha256="1" * 64,
        offset_contract={"handling": "NONE"},
    )
    metadata = save_candidate_bundle(bundle, tmp_path / "candidate.joblib")
    receipt = tmp_path / "publication_receipt.json"
    receipt.write_bytes(b"receipt")
    build = CompletedModelBuild(
        model_id=17,
        model_name="CLAIM_FREQ",
        rating_workbook_path="rating.xlsx",
        rating_workbook_sha256="a" * 64,
        model_version="20260603",
        model_type="superglm_poisson",
        target_name="claim_count",
        deployment_slot="CLAIM_FREQ_UAT",
        export_id="export-1",
        manifest_id="manifest-1",
        split_set_id="split-1",
        created_by="pytest",
        publication_receipt_path=str(receipt),
        publication_receipt_sha256=pipeline.sha256_file(receipt),
        candidate_artifact_path=metadata.path,
        candidate_artifact_sha256=metadata.sha256,
        candidate_artifact_format=metadata.format,
        candidate_artifact_size_bytes=metadata.size_bytes,
        candidate_python_version=metadata.python_version,
        candidate_superglm_version=metadata.superglm_version,
        candidate_superglm_git_sha=metadata.superglm_git_sha,
        build_fingerprint_sha256=bundle.build_fingerprint_sha256,
        builder_source_sha256=bundle.builder_source_sha256,
        materialized_split_sha256=bundle.materialized_split_sha256,
        runtime_sha256=bundle.runtime_sha256,
        candidate_superglm_sha256=bundle.candidate_superglm_sha256,
        row_order_sha256=bundle.row_order_sha256,
        model_source_sha256="b" * 64,
        model_frame_sha256="c" * 64,
    )
    return bundle, build


@pytest.mark.parametrize(
    ("identity_field", "published_value"),
    [
        ("model_name", "OTHER_MODEL"),
        ("model_version", "other-version"),
        ("export_id", "other-export"),
    ],
)
def test_completed_build_rejects_candidate_model_identity_mismatch(
    tmp_path,
    identity_field,
    published_value,
):
    _, build = _candidate(tmp_path)
    build = build.model_copy(update={identity_field: published_value})

    with pytest.raises(CompletedModelBuildError, match=identity_field):
        _verify_candidate_artifact(
            build,
            sql_lineage=CandidateSQLLineage(
                manifest_id="manifest-1",
                row_count=1,
                pk_columns=("policy_id",),
                split_set_id="split-1",
                split_row_order_sha256="a" * 64,
                model_frame_sha256="c" * 64,
            ),
            allowed_root=tmp_path,
        )


def test_completed_build_accepts_matching_candidate_model_identity(tmp_path):
    bundle, build = _candidate(tmp_path)

    _verify_candidate_artifact(
        build,
        sql_lineage=CandidateSQLLineage(
            manifest_id=bundle.manifest_id,
            row_count=1,
            pk_columns=bundle.pk_columns,
            split_set_id=bundle.split_set_id,
            split_row_order_sha256=bundle.row_order_sha256,
            model_frame_sha256=bundle.model_frame_sha256,
        ),
        allowed_root=tmp_path,
    )


@pytest.mark.parametrize("mismatch_source", ["artifact", "build", "sql"])
def test_completed_build_rejects_model_frame_digest_mismatch(
    tmp_path,
    mismatch_source,
):
    bundle, build = _candidate(tmp_path)
    sql_digest = bundle.model_frame_sha256
    if mismatch_source == "artifact":
        metadata = save_candidate_bundle(
            replace(bundle, model_frame_sha256="d" * 64),
            tmp_path / "mismatched-candidate.joblib",
        )
        build = build.model_copy(
            update={
                "candidate_artifact_path": metadata.path,
                "candidate_artifact_sha256": metadata.sha256,
                "candidate_artifact_format": metadata.format,
                "candidate_artifact_size_bytes": metadata.size_bytes,
                "candidate_python_version": metadata.python_version,
                "candidate_superglm_version": metadata.superglm_version,
                "candidate_superglm_git_sha": metadata.superglm_git_sha,
            }
        )
    elif mismatch_source == "build":
        build = build.model_copy(update={"model_frame_sha256": "d" * 64})
    else:
        sql_digest = "d" * 64

    with pytest.raises(CompletedModelBuildError, match="model_frame_sha256"):
        _verify_candidate_artifact(
            build,
            sql_lineage=CandidateSQLLineage(
                manifest_id=bundle.manifest_id,
                row_count=1,
                pk_columns=bundle.pk_columns,
                split_set_id=bundle.split_set_id,
                split_row_order_sha256=bundle.row_order_sha256,
                model_frame_sha256=sql_digest,
            ),
            allowed_root=tmp_path,
        )


def test_local_publication_rejects_model_frame_digest_mismatch_before_staging(
    tmp_path,
    monkeypatch,
):
    bundle, build = _candidate(tmp_path)
    workbook = tmp_path / "rating.xlsx"
    workbook.write_bytes(b"rating workbook")
    build = build.model_copy(
        update={
            "rating_workbook_path": str(workbook),
            "rating_workbook_sha256": pipeline.sha256_file(workbook),
        }
    )
    monkeypatch.setattr(
        "pricing_pipeline.publishing.sqlite_notebook.load_candidate_sql_lineage",
        lambda *args, **kwargs: CandidateSQLLineage(
            manifest_id=bundle.manifest_id,
            row_count=1,
            pk_columns=bundle.pk_columns,
            split_set_id=bundle.split_set_id,
            split_row_order_sha256=bundle.row_order_sha256,
            model_frame_sha256="d" * 64,
        ),
    )
    monkeypatch.setattr(
        "pricing_pipeline.publishing.sqlite_notebook.stage_rating_export",
        lambda *args, **kwargs: pytest.fail("staging ran before digest verification"),
    )

    with pytest.raises(CompletedModelBuildError, match="model_frame_sha256"):
        _publish_sqlite_candidate_locked(
            object(),
            model_id=17,
            model_config=ModelBuildConfig(
                model_name="CLAIM_FREQ",
                model_label="Claim frequency",
                target_name="claim_count",
                model_type="superglm_poisson",
                deployment_slot="CLAIM_FREQ_UAT",
            ),
            completed_build=build,
            created_by="pytest",
            artifact_root=tmp_path,
        )


@pytest.mark.parametrize("field_name", ["model_name", "model_version", "export_id"])
def test_existing_sql_run_rejects_candidate_model_identity_mismatch(
    tmp_path,
    monkeypatch,
    field_name,
):
    bundle, build = _candidate(tmp_path)
    artifact = save_candidate_bundle(
        replace(bundle, **{field_name: f"wrong-{field_name}"}),
        tmp_path / "existing-candidate.joblib",
    )
    workbook = tmp_path / "rating.xlsx"
    workbook.write_bytes(b"rating workbook")
    workbook_sha256 = pipeline.sha256_file(workbook)
    row = {
        "model_id": 17,
        "model_name": "CLAIM_FREQ",
        "model_version": "20260603",
        "source_export_id": "export-1",
        "rate_package_id": 42,
        "package_version": 7,
        "package_status": "PUBLISHED",
        "model_run_id": 901,
        "run_status": "SUCCESS",
        "run_model_name": "CLAIM_FREQ",
        "run_model_version": "20260603",
        "run_export_id": "export-1",
        "manifest_id": "manifest-1",
        "rating_workbook_path": str(workbook),
        "rating_workbook_sha256": workbook_sha256,
        "mlflow_run_id": "",
        "candidate_artifact_path": artifact.path,
        "candidate_artifact_sha256": artifact.sha256,
        "candidate_artifact_format": artifact.format,
        "candidate_artifact_size_bytes": artifact.size_bytes,
        "candidate_python_version": artifact.python_version,
        "candidate_superglm_version": artifact.superglm_version,
        "candidate_superglm_git_sha": artifact.superglm_git_sha,
        "model_source_sha256": "b" * 64,
    }

    class Rows:
        def __init__(self, values):
            self.values = values

        def mappings(self):
            return self

        def all(self):
            return self.values

    class Engine:
        def begin(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, statement, params):
            del params
            if "PRICING_RATE_PACKAGE AS rp" in str(statement):
                return Rows([row])
            return Rows([])

    monkeypatch.setattr(
        pipeline,
        "schema_names_from_connectable",
        lambda engine: SimpleNamespace(pricing="pricing", mlops="mlops"),
    )
    monkeypatch.setattr(pipeline, "_retry_evidence_conflicts", lambda **kwargs: [])
    export = ModelExportResult(
        model_id=17,
        model_name="CLAIM_FREQ",
        model_version="20260603",
        model_type="superglm_poisson",
        target_name="claim_count",
        deployment_slot="CLAIM_FREQ_UAT",
        manifest_id="manifest-1",
        mlflow_run_id=None,
        split_set_id="split-1",
        export_id="export-1",
        rating_workbook_path=str(workbook),
        rating_workbook_sha256=workbook_sha256,
        effective_from=None,
        created_by="pytest",
        publication_receipt_path=build.publication_receipt_path,
        publication_receipt_sha256=build.publication_receipt_sha256,
        candidate_artifact_path=build.candidate_artifact_path,
        candidate_artifact_sha256=build.candidate_artifact_sha256,
        candidate_artifact_format=build.candidate_artifact_format,
        candidate_artifact_size_bytes=build.candidate_artifact_size_bytes,
        candidate_python_version=build.candidate_python_version,
        candidate_superglm_version=build.candidate_superglm_version,
        candidate_superglm_git_sha=build.candidate_superglm_git_sha,
        build_fingerprint_sha256=build.build_fingerprint_sha256,
        builder_source_sha256=build.builder_source_sha256,
        materialized_split_sha256=build.materialized_split_sha256,
        runtime_sha256=build.runtime_sha256,
        candidate_superglm_sha256=build.candidate_superglm_sha256,
        row_order_sha256=build.row_order_sha256,
        model_source_sha256=build.model_source_sha256,
        model_frame_sha256=build.model_frame_sha256,
    )

    with pytest.raises(pipeline.PublishedRunIntegrityError, match=field_name):
        pipeline._resolve_existing_published_run(
            Engine(),
            export,
            allowed_artifact_root=tmp_path,
        )
