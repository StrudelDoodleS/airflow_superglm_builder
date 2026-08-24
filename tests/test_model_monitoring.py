from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import IntegrityError
from superglm import (
    Categorical,
    NaturalSpline,
    Numeric,
    OrderedCategorical,
    Spline,
    SuperGLM,
    collapse_levels,
)
from superglm.editor import EditorSession
from superglm.features import Constraint
from superglm.types import LambdaPolicy

from pricing_pipeline.data.manifest import model_frame_evidence
from pricing_pipeline.infra.offline_sqlite import (
    apply_offline_ddl,
    sqlite_engine_with_offline_schemas,
)
from pricing_pipeline.modeling.monitoring import (
    MonitoringError,
    MonitoringVariant,
    build_model_fit_contract,
    materialize_monitoring_model,
    persist_monitoring_fit,
    run_monitoring_fit,
)
from pricing_pipeline.publishing.superglm_metadata import build_superglm_publication_receipt
from pricing_pipeline.publishing.superglm_publication_receipt import (
    OffsetExportContract,
    canonical_receipt_bytes,
)
from pricing_pipeline.workbench.artifacts import CandidateBundle, save_candidate_bundle
from pricing_pipeline.workbench.core import Candidate


@pytest.fixture(scope="module")
def monitoring_case():
    rng = np.random.default_rng(1729)
    row_count = 360
    category = np.array(["A", "B", "C", "D"])[np.arange(row_count) % 4]
    ordered = np.array(["low", "mid", "high", "MISSING"])[np.arange(row_count) % 4]
    x = np.linspace(0.0, 1.0, row_count)
    X = pd.DataFrame({"category": category, "ordered": ordered, "x": x})
    y = rng.poisson(np.exp(-1.0 + 0.6 * x + 0.2 * (category == "D")))
    grouping = collapse_levels(
        X["category"],
        groups={"AB": ["A", "B"]},
    )
    model = SuperGLM(
        family="poisson",
        features={
            "category": Categorical(grouping=grouping, base="AB"),
            "ordered": OrderedCategorical(
                order=["low", "mid", "high", "MISSING"],
                specials=["MISSING"],
                basis=Spline(kind="ps", k=5),
                base="low",
            ),
            "x": Spline(
                kind="ps",
                k=7,
                constraint=Constraint.postfit.increasing,
            ),
        },
        selection_penalty=0.0,
    )
    model.fit_reml(X, y, max_reml_iter=5, runtime_validation="skip")
    return model, X, y


def test_fit_contract_and_variants_freeze_domain_structure(monitoring_case):
    model, _, _ = monitoring_case
    contract = build_model_fit_contract(model, continuous_points=11)
    payload = contract.payload()

    assert payload["schema_name"] == "superglm_monitoring_fit_contract"
    assert len(contract.contract_sha256) == 64
    assert payload["structure_sha256"] == contract.structure_sha256
    terms = payload["structure"]["term_metadata"]
    assert terms["category"]["declared"]["grouping"]["group_to_originals"]["AB"] == [
        "A",
        "B",
    ]
    assert terms["ordered"]["declared"]["specials"] == ["MISSING"]
    assert terms["ordered"]["fitted"]["special_levels"] == ["MISSING"]
    assert terms["x"]["declared"]["constraint_kind"] == "increasing"
    assert terms["x"]["declared"]["constraint_mode"] == "postfit"

    frozen = materialize_monitoring_model(model, MonitoringVariant.FROZEN_REFIT)
    lambda_refit = materialize_monitoring_model(
        model,
        MonitoringVariant.REESTIMATE_LAMBDA,
    )
    adaptive = materialize_monitoring_model(model, MonitoringVariant.FULL_ADAPTIVE)

    frozen_category = frozen._specs["category"]
    assert frozen_category.base == "AB"
    assert frozen_category._declared_levels == ["A", "B", "C", "D"]
    assert frozen_category._grouping == model._specs["category"]._grouping

    for refit in (frozen, lambda_refit, adaptive):
        ordered = refit._specs["ordered"]
        assert ordered._specials == ["MISSING"]
        assert ordered._grouping == model._specs["ordered"]._grouping
        assert refit._specs["x"].constraint_kind == "increasing"
        assert refit._specs["x"].constraint_mode == "postfit"

    np.testing.assert_allclose(
        frozen._specs["x"]._explicit_knots,
        model._specs["x"].fitted_knots,
    )
    np.testing.assert_allclose(
        lambda_refit._specs["x"]._explicit_knots,
        model._specs["x"].fitted_knots,
    )
    assert adaptive._specs["x"]._explicit_knots is None
    assert isinstance(frozen._specs["x"]._lambda_policy, LambdaPolicy)
    assert frozen._specs["x"]._lambda_policy.mode == "fixed"
    assert lambda_refit._specs["x"]._lambda_policy is None
    assert adaptive._specs["x"]._lambda_policy is None


def test_editor_created_grouping_is_the_monitoring_genesis():
    frame = pd.DataFrame(
        {
            "region": np.repeat(["A", "B", "C", "D"], 30),
            "x": np.tile(np.linspace(-1.0, 1.0, 30), 4),
        }
    )
    y = np.random.default_rng(1729).poisson(
        frame["region"].map({"A": 1.0, "B": 2.0, "C": 2.0, "D": 4.0}) * np.exp(0.2 * frame["x"])
    )
    raw = SuperGLM(
        features={"region": Categorical(base="first"), "x": Numeric()},
        selection_penalty=0.0,
    ).fit(frame, y)
    editor = EditorSession.from_model(raw, train_data=(frame, y))
    editor.select_levels("region", ["B", "C"])
    editor.replace_with_collapsed_levels("region", method="fit")
    deployed = editor.to_model()

    contract = build_model_fit_contract(deployed)
    grouping = contract.payload()["structure"]["term_metadata"]["region"]["declared"]["grouping"]
    assert grouping["group_to_originals"]["B+C"] == ["B", "C"]

    frozen = materialize_monitoring_model(deployed, MonitoringVariant.FROZEN_REFIT)
    assert frozen._specs["region"]._grouping == deployed._specs["region"]._grouping


def test_all_monitoring_presets_share_points_and_frozen_refit_reuses_lambdas(
    monitoring_case,
):
    model, X, y = monitoring_case
    results = {
        variant: run_monitoring_fit(
            model,
            X,
            y,
            variant=variant,
            continuous_points=11,
            max_reml_iter=5,
            runtime_validation="skip",
        )
        for variant in MonitoringVariant
    }

    expected_points = {
        (row.term_name, row.point_key)
        for row in results[MonitoringVariant.STATIC_SCORE].relativities
    }
    for result in results.values():
        assert {(row.term_name, row.point_key) for row in result.relativities} == expected_points
        assert result.metrics["row_count"] == len(X)

    baseline_lambdas = model.reml_diagnostics()["lambdas"]
    frozen = results[MonitoringVariant.FROZEN_REFIT]
    assert frozen.fitted_model.reml_diagnostics()["termination_reason"] == "fixed_lambdas"
    assert {row.component_name: row.lambda_value for row in frozen.lambdas} == baseline_lambdas
    assert {row.lambda_mode for row in frozen.lambdas} == {"FIXED"}
    assert {row.lambda_mode for row in results[MonitoringVariant.REESTIMATE_LAMBDA].lambdas} == {
        "ESTIMATED"
    }
    invariant = frozen.invariant_evidence.payload()
    assert frozen.invariant_evidence.status == "VERIFIED"
    assert len(frozen.invariant_evidence.evidence_sha256) == 64
    assert invariant["structure"]["exact_match"] is True
    assert invariant["geometry"]["protected_exact_match"] is True
    assert invariant["lambdas"]["baseline"] == invariant["lambdas"]["fitted"]
    assert invariant["lambdas"]["history_exact_for_protected_components"] is True
    assert invariant["lambdas"]["termination_reason"] == "fixed_lambdas"


def test_full_adaptive_evaluates_frozen_natural_spline_grid_with_real_extrapolation():
    baseline_x = np.linspace(0.0, 1.0, 240)
    baseline_X = pd.DataFrame({"x": baseline_x})
    mean = np.exp(-0.7 + 0.5 * baseline_x + 1.8 * baseline_x**2)
    baseline_y = np.random.default_rng(824).poisson(mean)
    baseline = SuperGLM(
        family="poisson",
        features={"x": NaturalSpline(n_knots=6, extrapolation="extend")},
        selection_penalty=0.0,
    ).fit_reml(baseline_X, baseline_y, max_reml_iter=5, runtime_validation="skip")
    adaptive_rows = baseline_X["x"].between(0.25, 0.75).to_numpy()
    adaptive_X = baseline_X.loc[adaptive_rows].reset_index(drop=True)
    adaptive_y = baseline_y[adaptive_rows]

    result = run_monitoring_fit(
        baseline,
        adaptive_X,
        adaptive_y,
        variant=MonitoringVariant.FULL_ADAPTIVE,
        continuous_points=5,
        max_reml_iter=5,
        runtime_validation="skip",
    )

    stored = {
        float(row.point_numeric): row.log_relativity
        for row in result.relativities
        if row.term_name == "x"
    }
    frozen_points = np.asarray(sorted(stored), dtype=float)
    fitted_prediction = np.asarray(
        result.fitted_model.predict(pd.DataFrame({"x": frozen_points})),
        dtype=float,
    )
    anchor = 2
    actual_delta = (
        np.asarray([stored[point] for point in frozen_points]) - stored[frozen_points[anchor]]
    )
    expected_delta = np.log(fitted_prediction) - np.log(fitted_prediction[anchor])

    assert result.fitted_model._specs["x"].fitted_boundary == pytest.approx(
        (float(adaptive_X["x"].min()), float(adaptive_X["x"].max()))
    )
    assert actual_delta == pytest.approx(expected_delta)


def test_monitoring_metrics_use_declared_sample_weights(monitoring_case):
    model, X, y = monitoring_case
    evaluation_X = X.iloc[:8].reset_index(drop=True)
    evaluation_y = np.asarray(y[:8], dtype=float)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 5.0, 1.0])

    result = run_monitoring_fit(
        model,
        evaluation_X,
        evaluation_y,
        variant=MonitoringVariant.STATIC_SCORE,
        sample_weight=weights,
        fit_sample_weight_name="exposure",
        continuous_points=11,
    )

    predictions = np.asarray(model.predict(evaluation_X), dtype=float)
    expected_prediction_sum = float(np.dot(weights, predictions))
    assert result.metrics["sample_weight_sum"] == pytest.approx(21.0)
    assert result.metrics["sample_weighted_sum_observed"] == pytest.approx(25.0)
    assert result.metrics["sample_weighted_mean_observed"] == pytest.approx(25.0 / 21.0)
    assert result.metrics["sample_weighted_sum_prediction"] == pytest.approx(
        expected_prediction_sum
    )
    assert result.metrics["sample_weighted_mean_prediction"] == pytest.approx(
        expected_prediction_sum / 21.0
    )
    assert "mean_observed" not in result.metrics
    assert "mean_prediction" not in result.metrics
    assert "sum_prediction" not in result.metrics


def _monitoring_candidate(
    tmp_path,
    model,
    X,
    y,
    *,
    offset_contract: OffsetExportContract | None = None,
    sample_weight=None,
    fit_sample_weight_name: str | None = None,
    offset=None,
    offset_source=None,
    offset_source_name: str | None = None,
    export_weight=None,
    export_weight_name: str | None = None,
) -> Candidate:
    model_frame_sha256 = model_frame_evidence(X.assign(target=y))[0]
    resolved_offset_contract = offset_contract or OffsetExportContract(handling="NONE")
    bundle = CandidateBundle(
        fitted_model=copy.deepcopy(model),
        X=X,
        y=np.asarray(y),
        sample_weight=sample_weight,
        offset=offset,
        export_weight=export_weight,
        cv_report={},
        model_name="SYNTHETIC_TARGET",
        model_version="v1",
        export_id="baseline-export-1",
        manifest_id="baseline-manifest-1",
        split_set_id=None,
        pk_columns=("PolicyID",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract=resolved_offset_contract,
        offset_source=offset_source,
        fit_sample_weight_name=fit_sample_weight_name,
        offset_source_name=offset_source_name,
        export_weight_name=export_weight_name,
        model_frame_sha256=model_frame_sha256,
    )
    artifact = save_candidate_bundle(bundle, tmp_path / "candidate.joblib")
    publication_receipt = build_superglm_publication_receipt(
        model,
        offset_contract=resolved_offset_contract,
        fit_sample_weight_name=fit_sample_weight_name,
        export_weight_name=export_weight_name,
    )
    publication_receipt_sha256 = hashlib.sha256(
        canonical_receipt_bytes(publication_receipt)
    ).hexdigest()
    workbench = SimpleNamespace(
        settings=SimpleNamespace(workbench_artifact_root=tmp_path),
        model_config=SimpleNamespace(deployment_slot="SYNTHETIC_PROD"),
    )
    technical = {
        "model_id": 91,
        "model_name": "SYNTHETIC_TARGET",
        "model_version": "v1",
        "model_equivalence_sha256": "c" * 64,
        "export_id": "baseline-export-1",
        "package_version": 1,
        "package_status": "PUBLISHED",
        "rate_package_id": 92,
        "model_run_id": "baseline-run-1",
        "run_status": "SUCCESS",
        "manifest_id": "baseline-manifest-1",
        "split_set_id": None,
        "candidate_artifact_path": artifact.path,
        "candidate_artifact_sha256": artifact.sha256,
        "candidate_artifact_format": artifact.format,
        "candidate_artifact_size_bytes": artifact.size_bytes,
        "candidate_python_version": artifact.python_version,
        "candidate_superglm_version": artifact.superglm_version,
        "model_source_sha256": "b" * 64,
        "model_frame_sha256": model_frame_sha256,
        "publication_receipt_sha256": publication_receipt_sha256,
        "package_publication_receipt_sha256": publication_receipt_sha256,
        "data_as_of_date": "2026-04-22",
        "current_deployment_id": 93,
        "current_rate_package_id": 92,
    }
    candidate = Candidate(
        workbench=workbench,
        model_name="SYNTHETIC_TARGET",
        package_version=1,
        rate_package_id=92,
        parent_rate_package_id=None,
        model_run_id="baseline-run-1",
        bundle=bundle,
        technical=dict(technical),
    )
    refreshed = replace(candidate, technical=dict(technical))
    workbench.open = lambda model_name, package_version: refreshed
    return candidate


def _monitoring_candidate_with_bundle(
    tmp_path,
    candidate: Candidate,
    bundle: CandidateBundle,
    *,
    artifact_name: str,
    preserve_publication_receipt: bool = False,
) -> Candidate:
    artifact = save_candidate_bundle(bundle, tmp_path / artifact_name)
    publication_receipt_sha256 = candidate.technical["publication_receipt_sha256"]
    if not preserve_publication_receipt:
        receipt = build_superglm_publication_receipt(
            bundle.fitted_model,
            offset_contract=bundle.offset_contract,
            fit_sample_weight_name=bundle.fit_sample_weight_name,
            export_weight_name=bundle.export_weight_name,
        )
        publication_receipt_sha256 = hashlib.sha256(canonical_receipt_bytes(receipt)).hexdigest()
    technical = {
        **candidate.technical,
        "candidate_artifact_path": artifact.path,
        "candidate_artifact_sha256": artifact.sha256,
        "candidate_artifact_format": artifact.format,
        "candidate_artifact_size_bytes": artifact.size_bytes,
        "candidate_python_version": artifact.python_version,
        "candidate_superglm_version": artifact.superglm_version,
        "publication_receipt_sha256": publication_receipt_sha256,
        "package_publication_receipt_sha256": publication_receipt_sha256,
    }
    updated = replace(candidate, bundle=bundle, technical=technical)
    updated.workbench.open = lambda model_name, package_version: replace(
        updated,
        technical=dict(technical),
    )
    return updated


def test_monitoring_reloads_and_binds_the_deployed_candidate_artifact(
    tmp_path,
    monitoring_case,
):
    model, X, y = monitoring_case
    candidate = _monitoring_candidate(tmp_path, model, X, y)
    candidate.bundle.fitted_model._specs["x"].constraint_kind = "decreasing"
    trusted_artifact_sha256 = candidate.technical["candidate_artifact_sha256"]
    candidate.technical["candidate_artifact_sha256"] = "0" * 64

    result = run_monitoring_fit(
        candidate,
        X,
        y,
        variant=MonitoringVariant.STATIC_SCORE,
        continuous_points=11,
    )

    assert result.fitted_model is not candidate.bundle.fitted_model
    assert (
        result.contract.payload()["structure"]["term_metadata"]["x"]["declared"]["constraint_kind"]
        == "increasing"
    )
    baseline = json.loads(result.fit_configuration_json)["baseline"]
    assert baseline == {
        "candidate_artifact_format": candidate.technical["candidate_artifact_format"],
        "candidate_artifact_sha256": trusted_artifact_sha256,
        "candidate_artifact_size_bytes": candidate.technical["candidate_artifact_size_bytes"],
        "candidate_python_version": candidate.technical["candidate_python_version"],
        "candidate_superglm_version": candidate.technical["candidate_superglm_version"],
        "data_as_of_date": "2026-04-22",
        "deployment_id": 93,
        "deployment_slot": "SYNTHETIC_PROD",
        "export_id": "baseline-export-1",
        "manifest_id": "baseline-manifest-1",
        "model_equivalence_sha256": "c" * 64,
        "model_frame_sha256": candidate.bundle.model_frame_sha256,
        "model_id": 91,
        "model_run_id": "baseline-run-1",
        "model_source_sha256": "b" * 64,
        "model_version": "v1",
        "package_version": 1,
        "package_publication_receipt_sha256": candidate.technical[
            "package_publication_receipt_sha256"
        ],
        "publication_receipt_sha256": candidate.technical["publication_receipt_sha256"],
        "rate_package_id": 92,
        "row_order_sha256": "a" * 64,
        "split_set_id": None,
    }

    with pytest.raises(MonitoringError, match="fit_sample_weight_name"):
        run_monitoring_fit(
            candidate,
            X,
            y,
            variant=MonitoringVariant.STATIC_SCORE,
            fit_sample_weight_name="tampered_weight",
            continuous_points=11,
        )


def test_monitoring_requires_the_deployed_fit_weight_contract(
    tmp_path,
    monitoring_case,
):
    model, X, y = monitoring_case
    candidate = _monitoring_candidate(tmp_path, model, X, y)
    weighted_bundle = replace(
        candidate.bundle,
        sample_weight=pd.Series(np.ones(len(X)), name="exposure"),
        fit_sample_weight_name="exposure",
    )
    weighted_candidate = _monitoring_candidate_with_bundle(
        tmp_path,
        candidate,
        weighted_bundle,
        artifact_name="weighted-candidate.joblib",
    )

    with pytest.raises(MonitoringError, match="sample_weight is required"):
        run_monitoring_fit(
            weighted_candidate,
            X,
            y,
            variant=MonitoringVariant.STATIC_SCORE,
            continuous_points=11,
        )


def test_monitoring_requires_the_deployed_offset_contract(
    tmp_path,
):
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 80)})
    offset = np.log(np.linspace(1.0, 2.0, len(X)))
    y = np.random.default_rng(417).poisson(np.exp(-0.4 + 0.3 * X["x"] + offset))
    model = SuperGLM(
        features={"x": Numeric()},
        selection_penalty=0.0,
    ).fit(X, y, offset=offset)
    offset_contract = OffsetExportContract(
        handling="EXPORTED_FACTOR",
        source_factor_name="exposure",
        published_factor_name="exposure",
        source_name="exposure",
        label="log exposure",
    )
    candidate = _monitoring_candidate(
        tmp_path,
        model,
        X,
        y,
        offset_contract=offset_contract,
        offset=offset,
        offset_source=np.exp(offset),
        offset_source_name="exposure",
    )
    offset_bundle = replace(
        candidate.bundle,
        offset=offset,
        offset_source=np.exp(offset),
        offset_source_name="exposure",
        offset_contract=offset_contract,
    )
    offset_candidate = _monitoring_candidate_with_bundle(
        tmp_path,
        candidate,
        offset_bundle,
        artifact_name="offset-candidate.joblib",
    )

    with pytest.raises(MonitoringError, match="offset is required"):
        run_monitoring_fit(
            offset_candidate,
            X,
            y,
            variant=MonitoringVariant.STATIC_SCORE,
            continuous_points=11,
        )


def test_monitoring_rejects_an_undeclared_candidate_offset(
    tmp_path,
    monitoring_case,
):
    model, X, y = monitoring_case
    candidate = _monitoring_candidate(tmp_path, model, X, y)

    with pytest.raises(MonitoringError, match="offset was not used"):
        run_monitoring_fit(
            candidate,
            X,
            y,
            variant=MonitoringVariant.STATIC_SCORE,
            offset=np.zeros(len(X)),
            continuous_points=11,
        )


def test_monitoring_fit_binds_the_exact_ordered_model_frame(monitoring_case):
    model, X, y = monitoring_case
    model_frame = X.assign(target=y)

    result = run_monitoring_fit(
        model,
        X,
        y,
        variant=MonitoringVariant.STATIC_SCORE,
        continuous_points=11,
        model_frame=model_frame,
        target_column="target",
    )

    assert result.model_frame_sha256 == model_frame_evidence(model_frame)[0]
    assert json.loads(result.fit_configuration_json)["target_column"] == "target"
    assert len(result.result_evidence_sha256) == 64

    changed_X = X.copy()
    changed_X.loc[0, "x"] += 0.01
    with pytest.raises(MonitoringError, match="X does not match the ordered model frame"):
        run_monitoring_fit(
            model,
            changed_X,
            y,
            variant=MonitoringVariant.STATIC_SCORE,
            model_frame=model_frame,
            target_column="target",
        )


def test_postfit_guard_rejects_a_silent_fixed_lambda_change(
    monitoring_case,
    monkeypatch,
):
    model, X, y = monitoring_case
    original_fit_reml = SuperGLM.fit_reml

    def sabotaged_fit_reml(refit, *args, **kwargs):
        fitted = original_fit_reml(refit, *args, **kwargs)
        component = next(iter(fitted._reml_result.lambdas))
        fitted._reml_result.lambdas[component] *= 1.000001
        return fitted

    monkeypatch.setattr(SuperGLM, "fit_reml", sabotaged_fit_reml)
    with pytest.raises(MonitoringError, match="fixed lambda"):
        run_monitoring_fit(
            model,
            X,
            y,
            variant=MonitoringVariant.FROZEN_REFIT,
            continuous_points=11,
            max_reml_iter=5,
            runtime_validation="skip",
        )


def test_postfit_guard_rejects_a_silent_protected_knot_change(
    monitoring_case,
    monkeypatch,
):
    model, X, y = monitoring_case
    original_fit_reml = SuperGLM.fit_reml

    def sabotaged_fit_reml(refit, *args, **kwargs):
        fitted = original_fit_reml(refit, *args, **kwargs)
        spline = fitted._specs["x"]
        spline._knots[spline.degree + 1] += 1e-9
        return fitted

    monkeypatch.setattr(SuperGLM, "fit_reml", sabotaged_fit_reml)
    with pytest.raises(MonitoringError, match="knot/boundary geometry"):
        run_monitoring_fit(
            model,
            X,
            y,
            variant=MonitoringVariant.FROZEN_REFIT,
            continuous_points=11,
            max_reml_iter=5,
            runtime_validation="skip",
        )


def test_postfit_guard_rejects_a_silent_constraint_change(
    monitoring_case,
    monkeypatch,
):
    model, X, y = monitoring_case
    original_fit_reml = SuperGLM.fit_reml

    def sabotaged_fit_reml(refit, *args, **kwargs):
        fitted = original_fit_reml(refit, *args, **kwargs)
        fitted._specs["x"].constraint_kind = "decreasing"
        return fitted

    monkeypatch.setattr(SuperGLM, "fit_reml", sabotaged_fit_reml)
    with pytest.raises(MonitoringError, match="structural change"):
        run_monitoring_fit(
            model,
            X,
            y,
            variant=MonitoringVariant.FROZEN_REFIT,
            continuous_points=11,
            max_reml_iter=5,
            runtime_validation="skip",
        )


def test_frozen_refit_retains_a_baseline_level_missing_from_the_new_snapshot(
    monitoring_case,
):
    model, X, y = monitoring_case
    keep = X["category"].ne("D").to_numpy()

    with pytest.warns(UserWarning, match="remain"):
        result = run_monitoring_fit(
            model,
            X.loc[keep].reset_index(drop=True),
            y[keep],
            variant=MonitoringVariant.FROZEN_REFIT,
            continuous_points=11,
            max_reml_iter=5,
            runtime_validation="skip",
        )

    category_levels = {
        row.point_label for row in result.relativities if row.term_name == "category"
    }
    assert "D" in category_levels


def _seed_monitoring_lineage(
    engine,
    *,
    model_frame_sha256: str,
    candidate: Candidate | None = None,
    monitor_row_count: int = 360,
    weight_column: str | None = None,
    offset_column: str | None = None,
    offset_source_column: str | None = None,
    offset_label: str | None = None,
    export_weight_column: str | None = None,
) -> None:
    technical = {} if candidate is None else candidate.technical
    baseline_frame_sha256 = str(technical.get("model_frame_sha256") or model_frame_sha256)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_MODEL (
                    model_id, model_name, model_label, target_name,
                    model_type, model_status, created_by
                ) VALUES (
                    91, 'SYNTHETIC_TARGET', 'Synthetic target', 'target_value',
                    'superglm_poisson', 'ACTIVE', 'pytest'
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.DATASET_MANIFEST (
                    manifest_id, manifest_signature_sha256, dataset_name,
                    source_system, data_as_of_date, data_as_of_column,
                    row_count, pk_columns_json, target_column,
                    model_frame_sha256, frame_hash_metadata_json, created_by
                ) VALUES (
                    'baseline-manifest-1', :manifest_sha, 'synthetic_baseline_frame',
                    'pricing_sql', '2026-04-22', 'AsAt',
                    360, '["PolicyID"]', 'target',
                    :frame_sha, '{}', 'pytest'
                )
                """
            ),
            {"manifest_sha": "a" * 64, "frame_sha": baseline_frame_sha256},
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.DATASET_MANIFEST (
                    manifest_id, manifest_signature_sha256, dataset_name,
                    source_system, data_as_of_date, data_as_of_column,
                    row_count, pk_columns_json, target_column, weight_column,
                    offset_column, offset_source_column, offset_label,
                    export_weight_column,
                    model_frame_sha256, frame_hash_metadata_json, created_by
                ) VALUES (
                    'manifest-monitor-1', :manifest_sha, 'synthetic_monitor_frame',
                    'pricing_sql', '2026-04-29', 'AsAt',
                    :row_count, '["PolicyID"]', 'target', :weight_column,
                    :offset_column, :offset_source_column, :offset_label,
                    :export_weight_column,
                    :frame_sha, '{}', 'pytest'
                )
                """
            ),
            {
                "manifest_sha": "e" * 64,
                "frame_sha": model_frame_sha256,
                "row_count": monitor_row_count,
                "weight_column": weight_column,
                "offset_column": offset_column,
                "offset_source_column": offset_source_column,
                "offset_label": offset_label,
                "export_weight_column": export_weight_column,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_RATE_PACKAGE (
                    rate_package_id, model_id, model_name, model_version,
                    package_version, base_rate, package_status,
                    publication_receipt_sha256, created_by
                ) VALUES (
                    92, 91, 'SYNTHETIC_TARGET', 'v1', 1, 1.0, 'PUBLISHED',
                    :receipt_sha, 'pytest'
                )
                """
            ),
            {"receipt_sha": technical.get("publication_receipt_sha256") or "d" * 64},
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_RUN (
                    model_run_id, model_id, model_version, export_id,
                    model_kind, model_equivalence_sha256, manifest_id,
                    rate_package_id, model_name,
                    rating_workbook_path, rating_workbook_sha256,
                    publication_receipt_path, publication_receipt_sha256,
                    candidate_artifact_path, candidate_artifact_sha256,
                    candidate_artifact_format, candidate_artifact_size_bytes,
                    candidate_python_version, candidate_superglm_version,
                    model_source_sha256,
                    run_status, created_by
                ) VALUES (
                    'baseline-run-1', 91, 'v1', 'baseline-export-1',
                    'ROUTINE_EDIT', :equivalence_sha, 'baseline-manifest-1',
                    92, 'SYNTHETIC_TARGET',
                    '/tmp/rating.xlsx', :workbook_sha,
                    '/tmp/publication_receipt.json', :receipt_sha,
                    :artifact_path, :artifact_sha, :artifact_format,
                    :artifact_size, :python_version, :superglm_version,
                    :model_source_sha,
                    'SUCCESS', 'pytest'
                )
                """
            ),
            {
                "workbook_sha": "f" * 64,
                "equivalence_sha": technical.get("model_equivalence_sha256") or "c" * 64,
                "receipt_sha": technical.get("publication_receipt_sha256") or "d" * 64,
                "artifact_path": technical.get("candidate_artifact_path")
                or "/tmp/candidate.joblib",
                "artifact_sha": technical.get("candidate_artifact_sha256") or "1" * 64,
                "artifact_format": technical.get("candidate_artifact_format")
                or "superglm-candidate-joblib-v2",
                "artifact_size": technical.get("candidate_artifact_size_bytes") or 1,
                "python_version": technical.get("candidate_python_version") or "3.14.0",
                "superglm_version": technical.get("candidate_superglm_version") or "0.26.0",
                "model_source_sha": technical.get("model_source_sha256") or "b" * 64,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT (
                    deployment_id, model_id, rate_package_id, deployment_slot,
                    effective_from_ts, deployed_by, deployment_note
                ) VALUES (
                    93, 91, 92, 'SYNTHETIC_PROD',
                    '2026-04-23 00:00:00', 'pytest', 'monitoring baseline'
                )
                """
            )
        )


@pytest.mark.parametrize(
    ("offset_contract", "manifest_offset_source"),
    [
        pytest.param(
            OffsetExportContract(
                handling="EXPORTED_FACTOR",
                source_factor_name="exposure",
                published_factor_name="exposure",
                source_name="exposure",
                label="log exposure",
            ),
            "exposure",
            id="exported-factor",
        ),
        pytest.param(
            OffsetExportContract(
                handling="ALREADY_APPLIED_SQL_EXPOSURE",
                source_name="exposure",
                label="log exposure",
            ),
            None,
            id="already-applied-sql-exposure",
        ),
    ],
)
def test_candidate_monitoring_persistence_binds_manifest_fit_and_export_roles(
    tmp_path,
    offset_contract,
    manifest_offset_source,
):
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 80)})
    exposure = np.linspace(1.0, 2.0, len(X))
    offset = np.log(exposure)
    fit_weight = np.linspace(0.5, 1.5, len(X))
    export_weight = np.linspace(10.0, 20.0, len(X))
    y = np.random.default_rng(824).poisson(np.exp(-0.4 + 0.3 * X["x"] + offset))
    model = SuperGLM(
        features={"x": Numeric()},
        selection_penalty=0.0,
    ).fit(X, y, sample_weight=fit_weight, offset=offset)
    candidate = _monitoring_candidate(
        tmp_path,
        model,
        X,
        y,
        offset_contract=offset_contract,
        sample_weight=fit_weight,
        fit_sample_weight_name="fit_weight",
        offset=offset,
        offset_source=exposure if offset_contract.handling == "EXPORTED_FACTOR" else None,
        offset_source_name=("exposure" if offset_contract.handling == "EXPORTED_FACTOR" else None),
        export_weight=export_weight,
        export_weight_name="rating_weight",
    )
    model_frame = X.assign(
        target=y,
        fit_weight=fit_weight,
        log_exposure=offset,
        exposure=exposure,
        rating_weight=export_weight,
    )
    result = run_monitoring_fit(
        candidate,
        X,
        y,
        variant=MonitoringVariant.STATIC_SCORE,
        sample_weight=fit_weight,
        offset=offset,
        continuous_points=11,
        model_frame=model_frame,
        target_column="target",
        offset_column="log_exposure",
    )
    manifest_contract = {
        "weight_column": "fit_weight",
        "offset_column": "log_exposure",
        "offset_source_column": manifest_offset_source,
        "offset_label": "log exposure",
        "export_weight_column": "rating_weight",
    }

    def seeded_engine(case_name, manifest_values):
        case_dir = tmp_path / offset_contract.handling.lower() / case_name
        engine = sqlite_engine_with_offline_schemas(
            {
                "pricing": case_dir / "pricing.sqlite",
                "pricing_stg": case_dir / "pricing_stg.sqlite",
                "mlops": case_dir / "mlops.sqlite",
            }
        )
        apply_offline_ddl(engine)
        _seed_monitoring_lineage(
            engine,
            model_frame_sha256=model_frame_evidence(model_frame)[0],
            candidate=candidate,
            monitor_row_count=len(model_frame),
            **manifest_values,
        )
        return engine

    engine = seeded_engine("matching", manifest_contract)
    persisted = persist_monitoring_fit(
        engine,
        result,
        baseline_model_run_id="baseline-run-1",
        baseline_deployment_id=93,
        manifest_id="manifest-monitor-1",
        created_by="pytest",
    )
    assert persisted.deduplicated is False

    for field_name, expected in manifest_contract.items():
        mismatch = "unexpected_role" if expected is None else None
        mismatched_contract = {**manifest_contract, field_name: mismatch}
        mismatch_engine = seeded_engine(field_name, mismatched_contract)
        with pytest.raises(MonitoringError, match=field_name):
            persist_monitoring_fit(
                mismatch_engine,
                result,
                baseline_model_run_id="baseline-run-1",
                baseline_deployment_id=93,
                manifest_id="manifest-monitor-1",
                created_by="pytest",
            )


def test_monitoring_rejects_a_sha_valid_candidate_with_receipt_metadata_drift(
    tmp_path,
    monitoring_case,
):
    model, X, y = monitoring_case
    candidate = _monitoring_candidate(tmp_path, model, X, y)
    changed_model = copy.deepcopy(candidate.bundle.fitted_model)
    changed_model._specs["x"].constraint_kind = "decreasing"
    changed_candidate = _monitoring_candidate_with_bundle(
        tmp_path,
        candidate,
        replace(candidate.bundle, fitted_model=changed_model),
        artifact_name="receipt-drift-candidate.joblib",
        preserve_publication_receipt=True,
    )

    with pytest.raises(MonitoringError, match="publication receipt"):
        run_monitoring_fit(
            changed_candidate,
            X,
            y,
            variant=MonitoringVariant.STATIC_SCORE,
            continuous_points=11,
        )


def test_monitoring_result_persists_and_is_queryable_in_standalone_sqlite(
    tmp_path,
    monitoring_case,
):
    model, X, y = monitoring_case
    candidate = _monitoring_candidate(tmp_path, model, X, y)
    model_frame = X.assign(target=y)
    result = run_monitoring_fit(
        candidate,
        X,
        y,
        variant=MonitoringVariant.STATIC_SCORE,
        continuous_points=11,
        model_frame=model_frame,
        target_column="target",
    )
    paths = {
        "pricing": tmp_path / "pricing.sqlite",
        "pricing_stg": tmp_path / "pricing_stg.sqlite",
        "mlops": tmp_path / "mlops.sqlite",
    }
    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    _seed_monitoring_lineage(
        engine,
        model_frame_sha256=model_frame_evidence(model_frame)[0],
        candidate=candidate,
    )

    persisted = persist_monitoring_fit(
        engine,
        result,
        baseline_model_run_id="baseline-run-1",
        baseline_deployment_id=93,
        manifest_id="manifest-monitor-1",
        created_by="pytest",
        component_role="SEVERITY",
    )
    retry = persist_monitoring_fit(
        engine,
        result,
        baseline_model_run_id="baseline-run-1",
        baseline_deployment_id=93,
        manifest_id="manifest-monitor-1",
        created_by="pytest",
        component_role="SEVERITY",
    )
    assert retry.monitor_run_id == persisted.monitor_run_id
    assert retry.deduplicated is True

    frequency = persist_monitoring_fit(
        engine,
        result,
        baseline_model_run_id="baseline-run-1",
        baseline_deployment_id=93,
        manifest_id="manifest-monitor-1",
        created_by="pytest",
        component_role="FREQUENCY",
    )
    assert frequency.monitor_run_id != persisted.monitor_run_id
    assert frequency.deduplicated is False

    with engine.connect() as connection:
        run = (
            connection.execute(
                text(
                    "SELECT * FROM pricing.V_MODEL_MONITORING_RUN "
                    "WHERE monitor_run_id = :monitor_run_id"
                ),
                {"monitor_run_id": persisted.monitor_run_id},
            )
            .mappings()
            .one()
        )
        lambda_rows = (
            connection.execute(
                text(
                    """
                    SELECT component_name, lambda_mode, data_as_of_date
                    FROM pricing.V_MODEL_MONITORING_LAMBDA
                    WHERE monitor_run_id = :monitor_run_id
                    ORDER BY component_name
                    """
                ),
                {"monitor_run_id": persisted.monitor_run_id},
            )
            .mappings()
            .all()
        )
        ordered_metadata = connection.execute(
            text(
                """
                    SELECT term_metadata_json
                    FROM pricing.MODEL_MONITOR_TERM
                    WHERE monitor_run_id = :monitor_run_id
                      AND term_name = 'ordered'
                    """
            ),
            {"monitor_run_id": persisted.monitor_run_id},
        ).scalar_one()

    assert run["variant_code"] == "STATIC_SCORE"
    assert run["component_role"] == "SEVERITY"
    assert run["invariant_status"] == "VERIFIED"
    assert run["invariant_evidence_sha256"] == result.invariant_evidence.evidence_sha256
    assert json.loads(run["invariant_evidence_json"])["status"] == "VERIFIED"
    assert run["data_as_of_date"] == "2026-04-29"
    assert run["data_as_of_column"] == "AsAt"
    assert run["baseline_model_run_id"] == "baseline-run-1"
    assert {row["lambda_mode"] for row in lambda_rows} == {"BASELINE"}
    assert {row["data_as_of_date"] for row in lambda_rows} == {"2026-04-29"}
    assert json.loads(ordered_metadata)["declared"]["specials"] == ["MISSING"]

    immutable_writes = (
        "UPDATE pricing.MODEL_MONITOR_RUN SET created_by = 'tampered'",
        "UPDATE pricing.MODEL_MONITOR_TERM SET term_kind = 'tampered'",
        "UPDATE pricing.MODEL_MONITOR_LAMBDA SET lambda_value = lambda_value + 1",
        "UPDATE pricing.MODEL_MONITOR_RELATIVITY SET relativity = relativity * 1.01",
        "UPDATE pricing.MODEL_MONITOR_METRIC SET metric_value = metric_value + 1",
        "DELETE FROM pricing.MODEL_MONITOR_RUN",
        "DELETE FROM pricing.MODEL_MONITOR_TERM",
        "DELETE FROM pricing.MODEL_MONITOR_LAMBDA",
        "DELETE FROM pricing.MODEL_MONITOR_RELATIVITY",
        "DELETE FROM pricing.MODEL_MONITOR_METRIC",
    )
    for statement in immutable_writes:
        with (
            pytest.raises(IntegrityError, match="monitoring evidence is immutable"),
            engine.begin() as connection,
        ):
            connection.execute(text(statement))

    with sqlite3.connect(paths["pricing"]) as standalone:
        standalone_run = standalone.execute(
            "SELECT data_as_of_date, baseline_deployment_slot FROM V_MODEL_MONITORING_RUN"
        ).fetchone()
    assert standalone_run == ("2026-04-29", "SYNTHETIC_PROD")

    with (
        pytest.raises(IntegrityError, match="model fit contracts are immutable"),
        engine.begin() as connection,
    ):
        connection.execute(
            text(
                """
                UPDATE pricing.MODEL_FIT_CONTRACT
                SET created_by = 'tampered'
                WHERE fit_contract_id = :fit_contract_id
                """
            ),
            {"fit_contract_id": persisted.fit_contract_id},
        )

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_RATE_PACKAGE (
                    rate_package_id, model_id, model_name, model_version,
                    package_version, base_rate, package_status, created_by
                ) VALUES (
                    94, 91, 'SYNTHETIC_TARGET', 'v2', 2, 1.0, 'DRAFT', 'pytest'
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_RUN (
                    model_run_id, model_id, model_version, export_id,
                    model_kind, manifest_id, rate_package_id, model_name,
                    rating_workbook_path, rating_workbook_sha256,
                    run_status, created_by
                ) VALUES (
                    'failed-run-2', 91, 'v2', 'failed-export-2',
                    'RAW', 'manifest-monitor-1', 94, 'SYNTHETIC_TARGET',
                    '/tmp/failed.xlsx', :workbook_sha, 'FAILED', 'pytest'
                )
                """
            ),
            {"workbook_sha": "d" * 64},
        )
    with (
        pytest.raises(IntegrityError, match="successful published baseline"),
        engine.begin() as connection,
    ):
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_FIT_CONTRACT (
                    fit_contract_id, baseline_model_run_id, model_id,
                    rate_package_id, contract_schema_version,
                    contract_sha256, structure_sha256, contract_json,
                    superglm_version, created_by
                ) VALUES (
                    'bad-contract', 'failed-run-2', 91,
                    94, 1, :contract_sha, :structure_sha, '{}',
                    '0.26.0', 'pytest'
                )
                """
            ),
            {"contract_sha": "e" * 64, "structure_sha": "f" * 64},
        )


def test_monitoring_persistence_rejects_a_simulation_only_baseline(
    tmp_path,
    monitoring_case,
):
    model, X, y = monitoring_case
    model_frame = X.assign(target=y)
    result = run_monitoring_fit(
        model,
        X,
        y,
        variant=MonitoringVariant.STATIC_SCORE,
        continuous_points=11,
        model_frame=model_frame,
        target_column="target",
    )
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    apply_offline_ddl(engine)
    _seed_monitoring_lineage(
        engine,
        model_frame_sha256=model_frame_evidence(model_frame)[0],
    )

    with pytest.raises(MonitoringError, match="verified deployed candidate artifact"):
        persist_monitoring_fit(
            engine,
            result,
            baseline_model_run_id="baseline-run-1",
            baseline_deployment_id=93,
            manifest_id="manifest-monitor-1",
            created_by="pytest",
        )


def test_concurrent_exact_monitoring_retries_resolve_one_observation(
    tmp_path,
    monitoring_case,
):
    model, X, y = monitoring_case
    candidate = _monitoring_candidate(tmp_path, model, X, y)
    model_frame = X.assign(target=y)
    result = run_monitoring_fit(
        candidate,
        X,
        y,
        variant=MonitoringVariant.STATIC_SCORE,
        continuous_points=11,
        model_frame=model_frame,
        target_column="target",
    )
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    apply_offline_ddl(engine)
    _seed_monitoring_lineage(
        engine,
        model_frame_sha256=model_frame_evidence(model_frame)[0],
        candidate=candidate,
    )
    persist_monitoring_fit(
        engine,
        result,
        baseline_model_run_id="baseline-run-1",
        baseline_deployment_id=93,
        manifest_id="manifest-monitor-1",
        created_by="pytest",
        component_role="FREQUENCY",
    )

    select_gate = threading.Barrier(2)
    gate_lock = threading.Lock()
    gated_selects = 0

    def synchronize_missing_observation_select(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ):
        nonlocal gated_selects
        if "FROM pricing.MODEL_MONITOR_RUN" not in statement:
            return
        with gate_lock:
            if gated_selects >= 2:
                return
            gated_selects += 1
        select_gate.wait(timeout=5)

    event.listen(engine, "before_cursor_execute", synchronize_missing_observation_select)
    try:

        def persist_exact_retry():
            return persist_monitoring_fit(
                engine,
                result,
                baseline_model_run_id="baseline-run-1",
                baseline_deployment_id=93,
                manifest_id="manifest-monitor-1",
                created_by="pytest",
                component_role="SEVERITY",
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(lambda _index: persist_exact_retry(), range(2)))
    finally:
        event.remove(engine, "before_cursor_execute", synchronize_missing_observation_select)

    assert receipts[0].monitor_run_id == receipts[1].monitor_run_id
    assert sorted(receipt.deduplicated for receipt in receipts) == [False, True]


def test_monitoring_persistence_rejects_frame_and_result_evidence_mismatch(
    tmp_path,
    monitoring_case,
):
    model, X, y = monitoring_case
    candidate = _monitoring_candidate(tmp_path, model, X, y)
    model_frame = X.assign(target=y)
    result = run_monitoring_fit(
        candidate,
        X,
        y,
        variant=MonitoringVariant.STATIC_SCORE,
        continuous_points=11,
        model_frame=model_frame,
        target_column="target",
    )
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    apply_offline_ddl(engine)
    _seed_monitoring_lineage(
        engine,
        model_frame_sha256="f" * 64,
        candidate=candidate,
    )

    with pytest.raises(MonitoringError, match="model frame does not match manifest"):
        persist_monitoring_fit(
            engine,
            result,
            baseline_model_run_id="baseline-run-1",
            baseline_deployment_id=93,
            manifest_id="manifest-monitor-1",
            created_by="pytest",
        )

    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE pricing.DATASET_MANIFEST SET model_frame_sha256 = :digest "
                "WHERE manifest_id = 'manifest-monitor-1'"
            ),
            {"digest": result.model_frame_sha256},
        )
        connection.execute(
            text(
                "UPDATE pricing.MODEL_RUN SET candidate_artifact_sha256 = :digest "
                "WHERE model_run_id = 'baseline-run-1'"
            ),
            {"digest": "9" * 64},
        )
    with pytest.raises(MonitoringError, match="candidate_artifact_sha256"):
        persist_monitoring_fit(
            engine,
            result,
            baseline_model_run_id="baseline-run-1",
            baseline_deployment_id=93,
            manifest_id="manifest-monitor-1",
            created_by="pytest",
        )

    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE pricing.MODEL_RUN SET candidate_artifact_sha256 = :digest "
                "WHERE model_run_id = 'baseline-run-1'"
            ),
            {"digest": candidate.technical["candidate_artifact_sha256"]},
        )
        connection.execute(
            text(
                "UPDATE pricing.PRICING_RATE_PACKAGE "
                "SET publication_receipt_sha256 = :digest "
                "WHERE rate_package_id = 92"
            ),
            {"digest": "8" * 64},
        )
    with pytest.raises(MonitoringError, match="package_publication_receipt_sha256"):
        persist_monitoring_fit(
            engine,
            result,
            baseline_model_run_id="baseline-run-1",
            baseline_deployment_id=93,
            manifest_id="manifest-monitor-1",
            created_by="pytest",
        )

    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE pricing.PRICING_RATE_PACKAGE "
                "SET publication_receipt_sha256 = :digest "
                "WHERE rate_package_id = 92"
            ),
            {"digest": candidate.technical["publication_receipt_sha256"]},
        )
        connection.execute(
            text(
                "UPDATE pricing.MODEL_RUN SET model_source_sha256 = :digest "
                "WHERE model_run_id = 'baseline-run-1'"
            ),
            {"digest": "7" * 64},
        )
    with pytest.raises(MonitoringError, match="model_source_sha256"):
        persist_monitoring_fit(
            engine,
            result,
            baseline_model_run_id="baseline-run-1",
            baseline_deployment_id=93,
            manifest_id="manifest-monitor-1",
            created_by="pytest",
        )

    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE pricing.MODEL_RUN SET model_source_sha256 = :digest "
                "WHERE model_run_id = 'baseline-run-1'"
            ),
            {"digest": candidate.technical["model_source_sha256"]},
        )
    tampered = replace(result, metrics={**result.metrics, "mean_prediction": 999.0})
    with pytest.raises(MonitoringError, match="result evidence digest"):
        persist_monitoring_fit(
            engine,
            tampered,
            baseline_model_run_id="baseline-run-1",
            baseline_deployment_id=93,
            manifest_id="manifest-monitor-1",
            created_by="pytest",
        )


def test_static_variant_has_no_materialized_refit_model(monitoring_case):
    model, _, _ = monitoring_case
    with pytest.raises(MonitoringError, match="STATIC_SCORE"):
        materialize_monitoring_model(model, MonitoringVariant.STATIC_SCORE)
