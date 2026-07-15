from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import ModelExportResult
from pricing_pipeline.orchestration.publish_completed_build import (
    CompletedModelBuild,
    CompletedModelBuildError,
    publish_completed_model_build,
)
from pricing_pipeline.publishing.lifecycle import CompletedModelPublishResult
from pricing_pipeline.publishing.sqlite_notebook import _publish_sqlite_candidate_locked


class _Engine:
    class _Transaction:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, traceback):
            return False

    def begin(self):
        return self._Transaction()


def _config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_name="CLAIM_FREQ",
        model_label="Claim frequency",
        target_name="claim_count",
        model_type="superglm_poisson",
        deployment_slot="CLAIM_FREQ_CURRENT",
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        pricing_database="PricingLab",
        mlflow_tracking_uri="",
        mlflow_enabled=False,
        rating_export_root=tmp_path / "rating_exports",
        validation_split_artifact_root=tmp_path / "validation_splits",
        workbench_artifact_root=tmp_path / "workbench",
    )


def _approved_build(tmp_path: Path) -> CompletedModelBuild:
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_bytes(b"rating workbook")
    candidate = tmp_path / "candidate.joblib"
    candidate.write_bytes(b"candidate")
    receipt = tmp_path / "publication_receipt.json"
    receipt.write_bytes(b"receipt")
    return CompletedModelBuild(
        model_id=17,
        model_name="CLAIM_FREQ",
        model_version="v1",
        model_type="superglm_poisson",
        target_name="claim_count",
        deployment_slot="CLAIM_FREQ_CURRENT",
        manifest_id="manifest-1",
        split_set_id=None,
        export_id="export-1",
        rating_workbook_path=str(workbook),
        rating_workbook_sha256=hashlib.sha256(workbook.read_bytes()).hexdigest(),
        effective_from=None,
        created_by="analyst",
        mlflow_run_id=None,
        publication_receipt_path=str(receipt),
        publication_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        candidate_artifact_path=str(candidate),
        candidate_artifact_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        candidate_artifact_format="superglm-candidate-joblib-v1",
        candidate_artifact_size_bytes=candidate.stat().st_size,
        candidate_python_version="3.14.4",
        candidate_superglm_version="0.11.0",
        model_source_sha256="a" * 64,
    )


def test_completed_build_and_export_are_one_record_type():
    assert CompletedModelBuild is ModelExportResult


@pytest.mark.parametrize(
    "field_name",
    [
        "publication_receipt_path",
        "publication_receipt_sha256",
        "candidate_artifact_path",
        "candidate_artifact_sha256",
        "candidate_artifact_format",
        "candidate_artifact_size_bytes",
        "candidate_python_version",
        "candidate_superglm_version",
        "model_source_sha256",
    ],
)
def test_approved_build_requires_all_audit_artifacts(tmp_path: Path, field_name: str):
    payload = _approved_build(tmp_path).model_dump()
    payload.pop(field_name)

    with pytest.raises(CompletedModelBuildError, match=field_name):
        CompletedModelBuild(**payload)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (date(2026, 6, 3), "2026-06-03"),
        (datetime(2026, 6, 3, 14, 30), "2026-06-03"),
        ("2026-06-03T14:30:00", "2026-06-03"),
    ],
)
def test_approved_build_normalises_effective_date(
    tmp_path: Path,
    raw_value,
    expected: str,
):
    payload = _approved_build(tmp_path).model_dump()
    payload["effective_from"] = raw_value

    assert CompletedModelBuild(**payload).effective_from == expected


def test_approved_build_rejects_non_finite_metrics(tmp_path: Path):
    payload = _approved_build(tmp_path).model_dump()
    payload["metrics"] = {"deviance": float("nan")}

    with pytest.raises(CompletedModelBuildError, match="finite"):
        CompletedModelBuild(**payload)


def test_approved_build_rejects_unknown_fields(tmp_path: Path):
    payload = _approved_build(tmp_path).model_dump()
    payload["unexpected"] = True

    with pytest.raises(CompletedModelBuildError, match="unexpected"):
        CompletedModelBuild(**payload)


def test_remote_publication_passes_the_approved_record_without_repacking(
    tmp_path: Path,
    monkeypatch,
):
    build = _approved_build(tmp_path)
    expected = CompletedModelPublishResult(
        model_id=17,
        model_name="CLAIM_FREQ",
        model_version="v1",
        manifest_id="manifest-1",
        split_set_id=None,
        export_id="export-1",
        rate_package_id=42,
        package_version=1,
        package_status="PUBLISHED",
        rating_workbook_path=build.rating_workbook_path,
        model_run_id=91,
    )
    captured = []
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_registered_model",
        lambda connection, config: SimpleNamespace(model_id=17),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.load_candidate_sql_lineage",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build._verify_candidate_artifact",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        lambda engine, export, **kwargs: captured.append(export) or expected,
    )

    result = publish_completed_model_build(
        _Engine(),
        settings=_settings(tmp_path),
        model_config=_config(),
        completed_build=build,
    )

    assert result is expected
    assert captured == [build]


def test_local_publication_rejects_record_from_another_registered_model(tmp_path: Path):
    build = _approved_build(tmp_path)

    with pytest.raises(CompletedModelBuildError, match="model_id"):
        _publish_sqlite_candidate_locked(
            object(),
            model_id=18,
            model_config=_config(),
            completed_build=build,
            created_by="analyst",
            artifact_root=tmp_path,
        )
