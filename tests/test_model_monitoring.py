from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from superglm import Categorical, Numeric, OrderedCategorical, Spline, SuperGLM, collapse_levels
from superglm.editor import EditorSession
from superglm.features import Constraint
from superglm.types import LambdaPolicy

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


def _seed_monitoring_lineage(engine) -> None:
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
                    'manifest-monitor-1', :manifest_sha, 'synthetic_frame',
                    'pricing_sql', '2026-04-22', 'AsAt',
                    360, '["PolicyID"]', 'target_value',
                    :frame_sha, '{}', 'pytest'
                )
                """
            ),
            {"manifest_sha": "a" * 64, "frame_sha": "b" * 64},
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_RATE_PACKAGE (
                    rate_package_id, model_id, model_name, model_version,
                    package_version, base_rate, package_status, created_by
                ) VALUES (
                    92, 91, 'SYNTHETIC_TARGET', 'v1', 1, 1.0, 'PUBLISHED', 'pytest'
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
                    'baseline-run-1', 91, 'v1', 'baseline-export-1',
                    'ROUTINE_EDIT', 'manifest-monitor-1', 92, 'SYNTHETIC_TARGET',
                    '/tmp/rating.xlsx', :workbook_sha, 'SUCCESS', 'pytest'
                )
                """
            ),
            {"workbook_sha": "c" * 64},
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


def test_monitoring_result_persists_and_is_queryable_in_standalone_sqlite(
    tmp_path,
    monitoring_case,
):
    model, X, y = monitoring_case
    result = run_monitoring_fit(
        model,
        X,
        y,
        variant=MonitoringVariant.STATIC_SCORE,
        continuous_points=11,
    )
    paths = {
        "pricing": tmp_path / "pricing.sqlite",
        "pricing_stg": tmp_path / "pricing_stg.sqlite",
        "mlops": tmp_path / "mlops.sqlite",
    }
    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    _seed_monitoring_lineage(engine)

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

    with engine.connect() as connection:
        run = (
            connection.execute(text("SELECT * FROM pricing.V_MODEL_MONITORING_RUN"))
            .mappings()
            .one()
        )
        lambda_rows = (
            connection.execute(
                text(
                    """
                    SELECT component_name, lambda_mode, data_as_of_date
                    FROM pricing.V_MODEL_MONITORING_LAMBDA
                    ORDER BY component_name
                    """
                )
            )
            .mappings()
            .all()
        )
        ordered_metadata = connection.execute(
            text(
                """
                SELECT term_metadata_json
                FROM pricing.MODEL_MONITOR_TERM
                WHERE term_name = 'ordered'
                """
            )
        ).scalar_one()

    assert run["variant_code"] == "STATIC_SCORE"
    assert run["component_role"] == "SEVERITY"
    assert run["invariant_status"] == "VERIFIED"
    assert run["invariant_evidence_sha256"] == result.invariant_evidence.evidence_sha256
    assert json.loads(run["invariant_evidence_json"])["status"] == "VERIFIED"
    assert run["data_as_of_date"] == "2026-04-22"
    assert run["data_as_of_column"] == "AsAt"
    assert run["baseline_model_run_id"] == "baseline-run-1"
    assert {row["lambda_mode"] for row in lambda_rows} == {"BASELINE"}
    assert {row["data_as_of_date"] for row in lambda_rows} == {"2026-04-22"}
    assert json.loads(ordered_metadata)["declared"]["specials"] == ["MISSING"]

    with sqlite3.connect(paths["pricing"]) as standalone:
        standalone_run = standalone.execute(
            "SELECT data_as_of_date, baseline_deployment_slot FROM V_MODEL_MONITORING_RUN"
        ).fetchone()
    assert standalone_run == ("2026-04-22", "SYNTHETIC_PROD")

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


def test_static_variant_has_no_materialized_refit_model(monitoring_case):
    model, _, _ = monitoring_case
    with pytest.raises(MonitoringError, match="STATIC_SCORE"):
        materialize_monitoring_model(model, MonitoringVariant.STATIC_SCORE)
