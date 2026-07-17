from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from pricing_pipeline.models import spec as model_spec
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import (
    ApprovedModelBuild,
    ApprovedModelBuildError,
    CompletedModelBuild,
    CompletedModelBuildError,
    ModelExportResult,
)
from pricing_pipeline.orchestration.publish_completed_build import (
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
        candidate_artifact_format="superglm-candidate-joblib-v3",
        candidate_artifact_size_bytes=candidate.stat().st_size,
        candidate_python_version="3.14.4",
        candidate_superglm_version="0.12.0",
        candidate_superglm_git_sha="a" * 40,
        model_source_sha256="a" * 64,
        model_frame_sha256="b" * 64,
    )


def test_completed_build_and_export_are_one_record_type():
    assert ApprovedModelBuild is CompletedModelBuild is ModelExportResult
    assert ApprovedModelBuildError is CompletedModelBuildError


def _numeric_curve_point(**overrides):
    values = {
        "validation_split_no": 1,
        "term_name": "vehicle_age",
        "point_no": 1,
        "point_kind": "NUMERIC",
        "x_numeric": 0.0,
        "level_text": None,
        "eta_contribution": 0.0,
        "relativity": 1.0,
        "support_value": 4.0,
        "reference_value": 0.0,
        "reference_level": None,
    }
    values.update(overrides)
    return values


def test_approved_build_curve_capture_is_legacy_safe_and_round_trips(tmp_path: Path):
    legacy = _approved_build(tmp_path)

    assert legacy.validation_curve_status is None
    assert legacy.validation_curve_reason is None
    assert legacy.validation_curve_points == ()

    payload = legacy.model_dump()
    payload.update(
        validation_curve_status="COMPLETE",
        validation_curve_points=(_numeric_curve_point(),),
    )
    build = ApprovedModelBuild(**payload)

    assert isinstance(build.validation_curve_points[0], model_spec.ValidationCurvePoint)
    assert build.validation_curve_points[0].eta_contribution == 0.0
    assert ApprovedModelBuild(**build.model_dump()).model_dump() == build.model_dump()


@pytest.mark.parametrize(
    ("status", "reason", "points", "match"),
    [
        (None, "capture failed", (), "legacy validation curve capture"),
        (None, None, (_numeric_curve_point(),), "legacy validation curve capture"),
        ("COMPLETE", "capture failed", (_numeric_curve_point(),), "COMPLETE.*reason"),
        ("COMPLETE", None, (), "COMPLETE.*at least one point"),
        ("UNAVAILABLE", None, (), "UNAVAILABLE.*reason"),
        ("UNAVAILABLE", "capture failed", (_numeric_curve_point(),), "UNAVAILABLE.*zero points"),
        ("UNAVAILABLE", "x" * 501, (), "at most 500 characters"),
        ("PARTIAL", None, (), "validation_curve_status"),
    ],
)
def test_approved_build_rejects_inconsistent_curve_capture(
    tmp_path: Path,
    status,
    reason,
    points,
    match,
):
    payload = _approved_build(tmp_path).model_dump()
    payload.update(
        validation_curve_status=status,
        validation_curve_reason=reason,
        validation_curve_points=points,
    )

    with pytest.raises(ApprovedModelBuildError, match=match):
        ApprovedModelBuild(**payload)


def test_approved_build_normalises_bounded_curve_capture_reason(tmp_path: Path):
    payload = _approved_build(tmp_path).model_dump()
    payload.update(
        validation_curve_status="UNAVAILABLE",
        validation_curve_reason="curve capture failed:\n  RuntimeError: broken   payload",
    )

    build = ApprovedModelBuild(**payload)

    assert build.validation_curve_reason == (
        "curve capture failed: RuntimeError: broken payload"
    )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"validation_split_no": 0}, "must be a positive integer"),
        ({"term_name": "  "}, "is required"),
        ({"point_no": True}, "must be a positive integer"),
        ({"x_numeric": [1.0]}, "must be a finite scalar"),
        ({"eta_contribution": float("nan")}, "must be a finite scalar"),
        ({"relativity": float("inf")}, "must be a finite scalar"),
        ({"support_value": -1.0}, "must be nonnegative"),
        ({"reference_value": None}, "NUMERIC.*reference_value"),
        ({"level_text": "unexpected"}, "NUMERIC.*level_text"),
        ({"reference_level": "A"}, "NUMERIC.*reference_level"),
    ],
)
def test_validation_curve_point_rejects_invalid_or_mismatched_shapes(
    override,
    match,
):
    payload = _numeric_curve_point(**override)

    with pytest.raises(ValueError, match=match):
        model_spec.ValidationCurvePoint(**payload)


def test_level_validation_curve_point_has_exact_level_shape():
    point = model_spec.ValidationCurvePoint(
        validation_split_no=2,
        term_name="region",
        point_no=3,
        point_kind="LEVEL",
        x_numeric=None,
        level_text="North",
        eta_contribution=-0.2,
        relativity=None,
        support_value=9.0,
        reference_value=None,
        reference_level="Central",
    )

    assert point.level_text == "North"
    assert point.reference_level == "Central"
    with pytest.raises(ValidationError):
        point.eta_contribution = 1.0


def test_approved_build_holds_ordered_validation_split_results(tmp_path: Path):
    payload = _approved_build(tmp_path).model_dump()
    payload["validation_splits"] = (
        {
            "validation_split_no": 1,
            "n_train": 80,
            "n_validation": 20,
            "metrics": {"deviance": 0.4, "nll": 0.2, "gini": 0.7},
        },
    )
    payload["fold_metrics"] = (
        {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},
        {"fold_no": 1, "metric_name": "nll", "metric_value": 0.2},
        {"fold_no": 1, "metric_name": "gini", "metric_value": 0.7},
    )

    build = ApprovedModelBuild(**payload)

    split = build.validation_splits[0]
    assert split.validation_split_no == 1
    assert split.n_train == 80
    assert split.n_validation == 20
    assert list(split.metrics) == ["deviance", "nll", "gini"]


def test_validation_evidence_is_deeply_immutable_and_round_trips(tmp_path: Path):
    payload = _approved_build(tmp_path).model_dump()
    payload["validation_splits"] = (
        {
            "validation_split_no": 1,
            "n_train": 80,
            "n_validation": 20,
            "metrics": {"deviance": 0.4},
        },
    )
    payload["fold_metrics"] = (
        {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},
    )
    build = ApprovedModelBuild(**payload)

    with pytest.raises(TypeError):
        build.validation_splits[0].metrics["deviance"] = 9.9
    with pytest.raises(TypeError):
        build.fold_metrics[0]["metric_value"] = 9.9

    dumped = build.model_dump()
    assert type(dumped["validation_splits"][0]["metrics"]) is dict
    assert type(dumped["fold_metrics"][0]) is dict
    assert ApprovedModelBuild(**dumped).model_dump() == dumped


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"validation_split_no": 0}, "validation_split_no.*positive integer"),
        ({"n_train": 0}, "n_train.*positive integer"),
        ({"n_validation": 0}, "n_validation.*positive integer"),
        ({"metrics": {}}, "metrics.*at least one metric"),
        ({"metrics": {"deviance": float("nan")}}, "deviance.*finite"),
    ],
)
def test_approved_build_rejects_invalid_validation_split_values(
    tmp_path: Path,
    override,
    match,
):
    payload = _approved_build(tmp_path).model_dump()
    split = {
        "validation_split_no": 1,
        "n_train": 80,
        "n_validation": 20,
        "metrics": {"deviance": 0.4},
    }
    split.update(override)
    payload["validation_splits"] = (split,)

    with pytest.raises(ApprovedModelBuildError, match=match):
        ApprovedModelBuild(**payload)


def test_approved_build_rejects_misnumbered_validation_splits(tmp_path: Path):
    payload = _approved_build(tmp_path).model_dump()
    payload["validation_splits"] = (
        {
            "validation_split_no": 1,
            "n_train": 80,
            "n_validation": 20,
            "metrics": {"deviance": 0.4},
        },
        {
            "validation_split_no": 3,
            "n_train": 80,
            "n_validation": 20,
            "metrics": {"deviance": 0.5},
        },
    )

    with pytest.raises(
        ApprovedModelBuildError,
        match="validation_splits must be numbered consecutively from 1",
    ):
        ApprovedModelBuild(**payload)


def test_approved_build_rejects_incomplete_validation_split_metrics(tmp_path: Path):
    payload = _approved_build(tmp_path).model_dump()
    payload["validation_splits"] = (
        {
            "validation_split_no": 1,
            "n_train": 80,
            "n_validation": 20,
            "metrics": {"deviance": 0.4, "nll": 0.2},
        },
        {
            "validation_split_no": 2,
            "n_train": 80,
            "n_validation": 20,
            "metrics": {"deviance": 0.5},
        },
    )

    with pytest.raises(
        ApprovedModelBuildError,
        match="validation_splits must contain the same metrics in requested order",
    ):
        ApprovedModelBuild(**payload)


@pytest.mark.parametrize(
    "fold_metrics",
    [
        (
            {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},
            {"fold_no": 1, "metric_name": "nll", "metric_value": 0.2},
            {"fold_no": 2, "metric_name": "deviance", "metric_value": 0.5},
        ),
        (
            {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},
            {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},
            {"fold_no": 1, "metric_name": "nll", "metric_value": 0.2},
            {"fold_no": 2, "metric_name": "deviance", "metric_value": 0.5},
            {"fold_no": 2, "metric_name": "nll", "metric_value": 0.3},
        ),
        (
            {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},
            {"fold_no": 1, "metric_name": "nll", "metric_value": 0.2},
            {"fold_no": 2, "metric_name": "deviance", "metric_value": 0.5},
            {"fold_no": 2, "metric_name": "nll", "metric_value": 9.9},
        ),
        (
            {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},
            {"fold_no": 1, "metric_name": "nll", "metric_value": 0.2},
            {"fold_no": 2, "metric_name": "deviance", "metric_value": 0.5},
            {"fold_no": 2, "metric_name": "nll", "metric_value": 0.3},
            {"fold_no": 2, "metric_name": "gini", "metric_value": 0.8},
        ),
    ],
)
def test_approved_build_rejects_fold_metrics_that_disagree_with_validation_splits(
    tmp_path: Path,
    fold_metrics,
):
    payload = _approved_build(tmp_path).model_dump()
    payload["validation_splits"] = (
        {
            "validation_split_no": 1,
            "n_train": 80,
            "n_validation": 20,
            "metrics": {"deviance": 0.4, "nll": 0.2},
        },
        {
            "validation_split_no": 2,
            "n_train": 80,
            "n_validation": 20,
            "metrics": {"deviance": 0.5, "nll": 0.3},
        },
    )
    payload["fold_metrics"] = fold_metrics

    with pytest.raises(
        ApprovedModelBuildError,
        match="fold_metrics must exactly match validation_splits",
    ):
        ApprovedModelBuild(**payload)


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
        "candidate_superglm_git_sha",
        "model_source_sha256",
        "model_frame_sha256",
    ],
)
def test_approved_build_requires_all_audit_artifacts(tmp_path: Path, field_name: str):
    payload = _approved_build(tmp_path).model_dump()
    payload.pop(field_name)

    with pytest.raises(CompletedModelBuildError, match=field_name):
        CompletedModelBuild(**payload)


@pytest.mark.parametrize("git_sha", ["", "A" * 40, "a" * 39, "g" * 40])
def test_approved_build_rejects_invalid_superglm_git_sha(tmp_path: Path, git_sha: str):
    payload = _approved_build(tmp_path).model_dump()
    payload["candidate_superglm_git_sha"] = git_sha

    with pytest.raises(
        CompletedModelBuildError,
        match="candidate_superglm_git_sha.*40-character lowercase hex git SHA",
    ):
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
