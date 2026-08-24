from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from superglm import (
    Categorical,
    CategoricalInteraction,
    FactorSmooth,
    Gaussian,
    LambdaPolicy,
    Numeric,
    NumericCategorical,
    NumericInteraction,
    Spline,
    SplineCategorical,
    SuperGLM,
    TensorInteraction,
)

import pricing_pipeline.reporting.adapters.superglm as superglm_adapter
from pricing_pipeline.reporting import UnderwriterReportOptions, build_scored_model_report
from pricing_pipeline.reporting._core import UnderwriterReportError
from pricing_pipeline.reporting.adapters.superglm import SuperGLMReportAdapter
from pricing_pipeline.reporting.evidence import (
    EvidenceFact,
    EvidenceRequest,
    ReportContext,
    normalize_model_evidence,
)


def _named(spec: object) -> object:
    """Supply the public explicit-interaction name required by SuperGLM 0.26."""
    spec.name = ":".join(spec.parent_names)
    return spec


def _context(
    frame: pd.DataFrame,
    *,
    model_name: str,
    prediction: np.ndarray,
    features: tuple[str, ...],
    comparison_unit_codes: np.ndarray | None = None,
    minimum_cell_size: int = 2,
) -> ReportContext:
    codes = (
        np.arange(len(frame), dtype=np.intp)
        if comparison_unit_codes is None
        else np.asarray(comparison_unit_codes, dtype=np.intp)
    )
    return ReportContext(
        frame=frame.loc[:, list(features)].reset_index(drop=True),
        actual=frame["actual"].to_numpy(dtype=float),
        predictions={model_name: np.asarray(prediction, dtype=float)},
        weight=frame["weight"].to_numpy(dtype=float),
        features=features,
        comparison_unit_codes=codes,
        comparison_units=len(np.unique(codes)),
        minimum_cell_size=minimum_cell_size,
        problem_type="frequency",
        deviance_power=1.0,
    )


def _interaction_frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic frame with a level repeated in rows but unsafe in comparison units."""
    unit_codes = np.repeat(np.arange(30, dtype=np.intp), 4)
    frame = pd.DataFrame(
        {
            "x": np.tile(np.linspace(0.0, 1.0, 4), 30),
            "z": np.tile(np.linspace(-1.0, 1.0, 4), 30),
            "a": np.where(unit_codes == 0, "C", np.where(unit_codes % 2, "A", "B")),
            "b": np.where(unit_codes % 3, "Q", "R"),
            "weight": np.ones(len(unit_codes)),
        }
    )
    eta = -1.1 + 0.25 * frame["x"] - 0.15 * frame["z"]
    eta += 0.15 * (frame["a"] == "B") + 0.1 * (frame["b"] == "R")
    frame["actual"] = np.exp(eta)
    return frame, unit_codes


def _fit_interaction(
    features: dict[str, object],
    interaction: object,
) -> tuple[SuperGLM, pd.DataFrame, np.ndarray]:
    frame, unit_codes = _interaction_frame()
    model = SuperGLM(
        features=features,
        interactions=[interaction],
        selection_penalty=0.0,
    ).fit(frame.loc[:, list(features)], frame["actual"], sample_weight=frame["weight"])
    return model, frame, unit_codes


def _embedded_payload(path: Path) -> dict[str, object]:
    match = re.search(
        r'<script type="application/json" id="report-data">(.*?)</script>',
        path.read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_adapter_collects_public_tensor_surface_with_normalized_density_columns():
    model, frame, unit_codes = _fit_interaction(
        {"x": Spline(k=5), "z": Spline(k=5)},
        _named(TensorInteraction("x", "z", n_knots=(4, 4))),
    )
    context = _context(
        frame,
        model_name="Tensor",
        prediction=model.predict(frame[["x", "z"]]),
        features=("x", "z"),
        comparison_unit_codes=unit_codes,
    )

    evidence = SuperGLMReportAdapter(interaction_points=32).collect(
        model_name="Tensor", source=model, context=context
    )

    term = evidence.interactions["x:z"]
    assert term.source == "SuperGLM object"
    assert term.semantic == "native_component"
    assert term.plot_kind == "surface"
    assert term.parents == ("x", "z")
    assert list(term.effect.columns) == ["x", "y", "value"]
    assert len(term.effect) == 32 * 32
    assert term.density is not None
    assert list(term.density.columns) == ["x", "y", "density", "hdr_mass"]


def test_adapter_skips_unselected_interaction_parents_before_plotting():
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "actual": [0.3, 0.5],
            "weight": [1.0, 1.0],
        }
    )
    context = _context(
        frame,
        model_name="Subset",
        prediction=np.array([0.4, 0.6]),
        features=("x",),
    )

    interactions, unavailable = superglm_adapter._model_interactions(
        _telemetry_model(
            _numeric_interaction_telemetry(),
            plot_data=lambda *args, **kwargs: pytest.fail(
                "out-of-scope interaction plotting must not run"
            ),
        ),
        context,
        n_points=8,
    )

    assert interactions == {}
    assert unavailable == []


def test_subset_report_omits_interactions_with_unselected_parent(tmp_path: Path):
    model, frame, _unit_codes = _fit_interaction(
        {"x": Numeric(), "z": Numeric()},
        _named(NumericInteraction("x", "z")),
    )
    frame["prediction"] = model.predict(frame[["x", "z"]])

    result = build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Subset": "prediction"},
        sample_weight="weight",
        features=["x"],
        evidence_requests=(EvidenceRequest("Subset", SuperGLMReportAdapter(), model),),
        output_path=tmp_path / "subset.html",
        options=UnderwriterReportOptions(
            problem_type="frequency",
            comparison_bootstrap_replicates=0,
            minimum_cell_size=2,
        ),
    )

    assert _embedded_payload(result.output_path)["interactions"] == {
        "models": {},
        "unavailable": [],
    }


@pytest.mark.parametrize(
    ("features", "interaction", "expected_kind", "columns"),
    [
        (
            {"a": Categorical(base="first"), "b": Categorical(base="first")},
            _named(CategoricalInteraction("a", "b")),
            "categorical_heatmap",
            ["left", "right", "value"],
        ),
        (
            {"x": Spline(k=5), "a": Categorical(base="first")},
            _named(SplineCategorical("x", "a")),
            "varying_coefficient",
            ["x", "level", "value"],
        ),
        (
            {"x": Numeric(), "a": Categorical(base="first")},
            _named(NumericCategorical("x", "a")),
            "numeric_categorical",
            ["level", "value"],
        ),
        (
            {"x": Numeric(), "z": Numeric()},
            _named(NumericInteraction("x", "z")),
            "numeric_numeric",
            ["value"],
        ),
    ],
    ids=["categorical", "spline-categorical", "numeric-categorical", "numeric-numeric"],
)
def test_adapter_normalizes_discrete_interaction_kinds_and_core_suppresses_unsafe_support(
    features: dict[str, object],
    interaction: object,
    expected_kind: str,
    columns: list[str],
):
    model, frame, unit_codes = _fit_interaction(features, interaction)
    model_name = "Interaction"
    context = _context(
        frame,
        model_name=model_name,
        prediction=model.predict(frame.loc[:, list(features)]),
        features=tuple(features),
        comparison_unit_codes=unit_codes,
    )

    evidence = SuperGLMReportAdapter(interaction_points=12).collect(
        model_name=model_name, source=model, context=context
    )
    term = next(iter(evidence.interactions.values()))
    assert term.plot_kind == expected_kind
    assert list(term.effect.columns) == columns

    normalized = normalize_model_evidence(model_name, evidence, context)
    normalized_term = normalized.interactions[term.name]
    if expected_kind == "categorical_heatmap":
        assert "C" not in set(normalized_term.effect["left"])
    elif expected_kind in {"varying_coefficient", "numeric_categorical"}:
        assert "C" not in set(normalized_term.effect["level"])


def _factor_smooth_model(basis: str) -> tuple[SuperGLM, pd.DataFrame]:
    rng = np.random.default_rng(314)
    frame = pd.DataFrame(
        {
            "x": np.tile(np.linspace(0.0, 1.0, 40), 3),
            "level": np.repeat(["A", "B", "C"], 40),
            "weight": np.ones(120),
        }
    )
    level_effect = np.select(
        [frame["level"] == "A", frame["level"] == "B"],
        [0.12 * np.sin(5 * frame["x"]), -0.12 * np.sin(5 * frame["x"])],
        default=0.06 * np.cos(4 * frame["x"]),
    )
    frame["actual"] = 1.2 + 0.25 * frame["x"] + level_effect + rng.normal(0.0, 0.03, len(frame))
    policy = (
        {
            "wiggle": LambdaPolicy.fixed(1.2),
            "null_0": LambdaPolicy.fixed(0.8),
            "null_1": LambdaPolicy.fixed(0.9),
        }
        if basis == "fs"
        else {"wiggle": LambdaPolicy.fixed(1.2)}
    )
    model = SuperGLM(
        family=Gaussian(),
        features={"x": Spline(k=5)},
        interactions=[
            FactorSmooth(
                "x",
                group="level",
                basis=basis,
                k=5,
                lambda_policy=policy,
                name=f"curve_{basis}",
            )
        ],
        selection_penalty=0.0,
    ).fit_reml(
        frame[["x", "level"]], frame["actual"], sample_weight=frame["weight"], max_reml_iter=2
    )
    return model, frame


@pytest.mark.parametrize(
    ("basis", "interpretation"),
    [
        ("fs", "Level-specific fitted effect"),
        ("sz", "Sum-to-zero deviation from the common smooth"),
    ],
)
def test_adapter_collects_factor_smooths_via_dedicated_public_api(
    monkeypatch: pytest.MonkeyPatch,
    basis: str,
    interpretation: str,
):
    model, frame = _factor_smooth_model(basis)
    name = f"curve_{basis}"
    original_plot_data: Callable[..., object] = model.plot_data

    def fail_factor_smooth_plot_data(term: object, **kwargs: object) -> object:
        if term == name:
            raise AssertionError("factor smooth must not use plot_data")
        return original_plot_data(term, **kwargs)

    monkeypatch.setattr(model, "plot_data", fail_factor_smooth_plot_data)
    context = _context(
        frame,
        model_name="Factor",
        prediction=np.maximum(model.predict(frame[["x", "level"]]), 1.0e-6),
        features=("x", "level"),
        comparison_unit_codes=np.where(frame["level"].eq("C"), 0, np.arange(len(frame))),
        minimum_cell_size=2,
    )

    evidence = SuperGLMReportAdapter(interaction_points=32).collect(
        model_name="Factor", source=model, context=context
    )
    term = evidence.interactions[name]

    assert term.semantic == "native_component"
    assert term.plot_kind == "factor_smooth"
    assert EvidenceFact("Basis", basis) in term.facts
    assert EvidenceFact("Interpretation", interpretation) in term.facts
    assert {fact.label for fact in term.facts} >= {
        "Lambda (wiggle)",
        "Effective DF",
    }
    assert all(np.isscalar(fact.value) or fact.value is None for fact in term.facts)
    assert term.level_diagnostics is not None
    assert set(term.level_diagnostics.columns) >= {
        "level",
        "effective_df",
        "has_information",
        "sufficient_support",
        "collapsed",
    }
    if basis == "fs":
        assert "credibility" in term.level_diagnostics
    else:
        assert "Not a standalone rating relativity" in term.warnings
    assert "C" not in set(term.effect["level"])
    assert "C" not in set(term.level_diagnostics["level"])


def test_adapter_reports_factor_smooth_lambda_boundary_warnings(
    monkeypatch: pytest.MonkeyPatch,
):
    model, frame = _factor_smooth_model("fs")
    original_factor_smooth: Callable[..., object] = model.factor_smooth

    def lower_boundary_result(*args: object, **kwargs: object) -> object:
        result = original_factor_smooth(*args, **kwargs)
        return replace(result, at_lower_boundary={"wiggle": True}, at_upper_boundary={})

    monkeypatch.setattr(model, "factor_smooth", lower_boundary_result)
    context = _context(
        frame,
        model_name="Factor",
        prediction=np.maximum(model.predict(frame[["x", "level"]]), 1.0e-6),
        features=("x", "level"),
    )

    evidence = SuperGLMReportAdapter(interaction_points=32).collect(
        model_name="Factor", source=model, context=context
    )

    assert (
        "curve_fs: lambda wiggle is at the lower boundary"
        in evidence.interactions["curve_fs"].warnings
    )


def test_adapter_keeps_successful_interactions_when_one_public_plot_capability_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    class MixedAvailabilityModel:
        def training_telemetry(self) -> dict[str, object]:
            return {
                "features": {
                    "interaction_order": ["ok", "unavailable"],
                    "interactions": {
                        "ok": {
                            "class": "NumericInteraction",
                            "parents": ["x", "z"],
                        },
                        "unavailable": {
                            "class": "NumericInteraction",
                            "parents": ["x", "z"],
                        },
                    },
                }
            }

        def plot_data(self, name: str, **kwargs: object) -> dict[str, object]:
            del kwargs
            if name == "unavailable":
                raise NotImplementedError("native interaction reporting is not supported")
            return {
                "kind": "interaction",
                "name": name,
                "parents": ["x", "z"],
                "plot_kind": "numeric_numeric",
                "effect": pd.DataFrame({"relativity_per_unit_unit": [1.1]}),
            }

        def term_importance(self, frame: object, weight: object) -> object:
            del frame, weight
            raise NotImplementedError("upstream implementation detail")

    def unsupported_editor(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise NotImplementedError("upstream implementation detail")

    monkeypatch.setattr(
        "pricing_pipeline.reporting.adapters.superglm.EditorSession.from_model",
        unsupported_editor,
    )
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "z": [1.0, 2.0, 3.0, 4.0],
            "actual": [0.3, 0.4, 0.5, 0.6],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    context = _context(
        frame,
        model_name="Mixed",
        prediction=np.array([0.4, 0.5, 0.6, 0.7]),
        features=("x", "z"),
    )

    evidence = normalize_model_evidence(
        "Mixed",
        SuperGLMReportAdapter().collect(
            model_name="Mixed", source=MixedAvailabilityModel(), context=context
        ),
        context,
    )

    assert tuple(evidence.interactions) == ("ok",)
    assert evidence.warnings == ("unavailable: native interaction reporting is not supported",)
    assert "interactions" not in {item.capability for item in evidence.unavailable}


def _telemetry_model(payload: object, *, plot_data: Callable[..., object] | None = None) -> object:
    class TelemetryModel:
        def training_telemetry(self) -> object:
            return payload

        def plot_data(self, *args: object, **kwargs: object) -> object:
            if plot_data is None:
                raise AssertionError("plot_data was not expected")
            return plot_data(*args, **kwargs)

    return TelemetryModel()


def _numeric_interaction_telemetry(order: list[str] | None = None) -> dict[str, object]:
    names = ["term"] if order is None else order
    interactions = (
        {} if not names else {"term": {"class": "NumericInteraction", "parents": ["x", "z"]}}
    )
    return {
        "features": {
            "interaction_order": names,
            "interactions": interactions,
        }
    }


@pytest.mark.parametrize(
    "payload",
    [
        {
            "features": {
                "interaction_order": ["term", "term"],
                "interactions": {"term": {"class": "NumericInteraction", "parents": ["x", "z"]}},
            }
        },
        {
            "features": {
                "interaction_order": ["term"],
                "interactions": {"term": {"class": "UnknownInteraction", "parents": ["x", "z"]}},
            }
        },
        {
            "features": {
                "interaction_order": ["term"],
                "interactions": {"term": {"class": "NumericInteraction", "parents": ["x"]}},
            }
        },
        {
            "features": {
                "interaction_order": [],
                "interactions": {"orphan": {"class": "NumericInteraction", "parents": ["x", "z"]}},
            }
        },
    ],
)
def test_adapter_rejects_malformed_or_unknown_public_interaction_telemetry(payload: object):
    with pytest.raises(UnderwriterReportError, match="telemetry"):
        superglm_adapter._interaction_telemetry(_telemetry_model(payload))


def test_adapter_rejects_unhashable_interaction_order_names_with_a_stable_error():
    payload = _numeric_interaction_telemetry()
    payload["features"]["interaction_order"] = [["term"]]

    with pytest.raises(UnderwriterReportError, match="invalid name"):
        superglm_adapter._interaction_telemetry(_telemetry_model(payload))


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kind": "main_effects",
            "name": "term",
            "parents": ["x", "z"],
            "plot_kind": "numeric_numeric",
        },
        {
            "kind": "interaction",
            "name": "other",
            "parents": ["x", "z"],
            "plot_kind": "numeric_numeric",
        },
        {
            "kind": "interaction",
            "name": "term",
            "parents": ["z", "x"],
            "plot_kind": "numeric_numeric",
        },
        {"kind": "interaction", "name": "term", "parents": ["x", "z"], "plot_kind": "surface"},
    ],
)
def test_adapter_rejects_public_plot_payload_identity_or_class_mapping_mismatches(
    payload: dict[str, object],
):
    payload["effect"] = pd.DataFrame({"relativity_per_unit_unit": [1.1]})
    with pytest.raises(UnderwriterReportError, match="interaction"):
        superglm_adapter._interaction_from_plot_data(
            "term", ("x", "z"), payload, expected_plot_kind="numeric_numeric"
        )


def test_adapter_distinguishes_empty_public_interaction_order_from_unavailable_telemetry():
    empty, empty_unavailable = superglm_adapter._model_interactions(
        _telemetry_model(_numeric_interaction_telemetry([])),
        _small_context(),
        n_points=8,
    )

    class MissingTelemetry:
        pass

    missing, missing_unavailable = superglm_adapter._model_interactions(
        MissingTelemetry(), _small_context(), n_points=8
    )

    assert empty == {}
    assert empty_unavailable == []
    assert missing == {}
    assert missing_unavailable == [
        superglm_adapter.CapabilityUnavailable(
            "interactions", "native interaction reporting is not supported"
        )
    ]


def test_adapter_sanitizes_unexpected_public_plot_errors_without_suppressing_them():
    def sensitive_plot(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise NotImplementedError("sensitive upstream detail")

    with pytest.raises(UnderwriterReportError, match="could not extract") as exc_info:
        superglm_adapter._model_interactions(
            _telemetry_model(_numeric_interaction_telemetry(), plot_data=sensitive_plot),
            _small_context(),
            n_points=8,
        )
    assert "sensitive" not in str(exc_info.value)


def test_adapter_returns_interactions_capability_unavailable_only_when_every_term_is_unavailable():
    def unavailable_plot(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise NotImplementedError("native interaction reporting is not supported")

    interactions, unavailable = superglm_adapter._model_interactions(
        _telemetry_model(_numeric_interaction_telemetry(), plot_data=unavailable_plot),
        _small_context(),
        n_points=8,
    )

    assert interactions == {}
    assert unavailable == [
        superglm_adapter.CapabilityUnavailable(
            "interactions", "term: native interaction reporting is not supported"
        )
    ]


@pytest.mark.parametrize("successful", [True, False], ids=["mixed", "all-unavailable"])
def test_adapter_recognizes_only_the_exact_factor_smooth_support_unavailable_message(
    successful: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    support_message = (
        "FactorSmooth support for 'curve' is unavailable; refit with "
        "retain_fit_state=True or direct_solve='structured'."
    )

    class FactorSmoothAvailabilityModel:
        def training_telemetry(self) -> dict[str, object]:
            order = ["curve"] if not successful else ["ok", "curve"]
            interactions: dict[str, object] = {
                "curve": {"class": "FactorSmooth", "parents": ["x", "z"]}
            }
            if successful:
                interactions["ok"] = {
                    "class": "NumericInteraction",
                    "parents": ["x", "z"],
                }
            return {"features": {"interaction_order": order, "interactions": interactions}}

        def plot_data(self, name: str, **kwargs: object) -> dict[str, object]:
            del kwargs
            assert name == "ok"
            return {
                "kind": "interaction",
                "name": "ok",
                "parents": ["x", "z"],
                "plot_kind": "numeric_numeric",
                "effect": pd.DataFrame({"relativity_per_unit_unit": [1.1]}),
            }

        def factor_smooth(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError(support_message)

    monkeypatch.setattr(superglm_adapter, "_model_main_effects", lambda *args, **kwargs: {})
    monkeypatch.setattr(superglm_adapter, "_model_importance", lambda *args, **kwargs: None)
    evidence = SuperGLMReportAdapter(interaction_points=8).collect(
        model_name="Small", source=FactorSmoothAvailabilityModel(), context=_small_context()
    )

    if successful:
        assert tuple(evidence.interactions) == ("ok",)
        assert evidence.warnings == ("curve: native interaction reporting is not supported",)
        assert "interactions" not in {item.capability for item in evidence.unavailable}
    else:
        assert evidence.interactions == {}
        assert evidence.warnings == ()
        assert (
            superglm_adapter.CapabilityUnavailable(
                "interactions", "curve: native interaction reporting is not supported"
            )
            in evidence.unavailable
        )


def test_adapter_detaches_ordinary_public_plot_tables():
    raw_effect = pd.DataFrame({"relativity_per_unit_unit": [1.1]})
    payload = {
        "kind": "interaction",
        "name": "term",
        "parents": ["x", "z"],
        "plot_kind": "numeric_numeric",
        "effect": raw_effect,
    }

    term = superglm_adapter._interaction_from_plot_data(
        "term", ("x", "z"), payload, expected_plot_kind="numeric_numeric"
    )
    raw_effect.loc[0, "relativity_per_unit_unit"] = 9.9

    assert term.effect.loc[0, "value"] == pytest.approx(1.1)


def test_adapter_detaches_surface_effect_density_and_grid_axes():
    effect = pd.DataFrame(
        {"x": [0.0, 1.0, 0.0, 1.0], "z": [2.0, 2.0, 3.0, 3.0], "relativity": [1.0, 1.1, 1.2, 1.3]}
    )
    density = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0],
            "z": [2.0, 2.0, 3.0, 3.0],
            "density": [0.1, 0.2, 0.3, 0.4],
            "hdr_mass": [0.1, 0.2, 0.3, 0.4],
        }
    )
    x_axis = np.array([0.0, 1.0])
    z_axis = np.array([2.0, 3.0])
    payload = {
        "kind": "interaction",
        "name": "surface",
        "parents": ["x", "z"],
        "plot_kind": "surface",
        "effect": effect,
        "density": density,
        "grid_axes": {"x": x_axis, "z": z_axis},
    }

    term = superglm_adapter._interaction_from_plot_data(
        "surface", ("x", "z"), payload, expected_plot_kind="surface"
    )
    effect.loc[:, "relativity"] = 9.0
    density.loc[:, "density"] = 9.0
    x_axis[:] = 9.0
    z_axis[:] = 9.0

    assert term.effect["value"].tolist() == [1.0, 1.1, 1.2, 1.3]
    assert term.density is not None
    assert term.density["density"].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert term.grid_axes["x"].tolist() == [0.0, 1.0]
    assert term.grid_axes["y"].tolist() == [2.0, 3.0]


def test_adapter_detaches_factor_smooth_results_and_matches_response_scale_exponentiation(
    monkeypatch: pytest.MonkeyPatch,
):
    model, frame = _factor_smooth_model("fs")
    result = model.factor_smooth("curve_fs", grid=32, levels=["A", "B", "C"])
    monkeypatch.setattr(model, "factor_smooth", lambda *args, **kwargs: result)
    context = _context(
        frame,
        model_name="Factor",
        prediction=np.maximum(model.predict(frame[["x", "level"]]), 1.0e-6),
        features=("x", "level"),
    )

    evidence = SuperGLMReportAdapter(interaction_points=32).collect(
        model_name="Factor", source=model, context=context
    )
    term = evidence.interactions["curve_fs"]
    assert term.effect["value"].to_numpy() == pytest.approx(np.exp(result.curves["effect"]))
    assert term.effect["lower"].to_numpy() == pytest.approx(np.exp(result.curves["lower"]))
    result.curves.loc[:, "effect"] = 8.0
    result.table.loc[:, "effective_df"] = 0.0
    assert term.effect["value"].max() < 2.0
    assert term.level_diagnostics is not None
    assert term.level_diagnostics["effective_df"].sum() > 0.0


def test_adapter_rejects_factor_smooth_response_scale_overflow(monkeypatch: pytest.MonkeyPatch):
    model, frame = _factor_smooth_model("fs")
    result = model.factor_smooth("curve_fs", grid=8, levels=["A", "B", "C"])
    overflow = replace(
        result, curves=result.curves.assign(effect=1000.0, lower=1000.0, upper=1000.0)
    )
    monkeypatch.setattr(model, "factor_smooth", lambda *args, **kwargs: overflow)
    context = _context(
        frame,
        model_name="Factor",
        prediction=np.maximum(model.predict(frame[["x", "level"]]), 1.0e-6),
        features=("x", "level"),
    )

    with pytest.raises(UnderwriterReportError, match="non-finite on the response scale"):
        SuperGLMReportAdapter(interaction_points=8).collect(
            model_name="Factor", source=model, context=context
        )


def _small_context() -> ReportContext:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "z": [1.0, 2.0],
            "actual": [0.3, 0.5],
            "weight": [1.0, 1.0],
        }
    )
    return _context(
        frame,
        model_name="Small",
        prediction=np.array([0.4, 0.6]),
        features=("x", "z"),
    )
