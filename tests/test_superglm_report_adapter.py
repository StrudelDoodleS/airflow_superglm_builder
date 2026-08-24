from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from superglm import Categorical, Numeric, Spline, SuperGLM, Tweedie
from superglm.distributions import Gamma
from superglm.editor import EditorSession
from superglm.editor.payloads import session_payload
from superglm.profiling.tweedie import tweedie_logpdf

import pricing_pipeline.reporting.adapters.superglm as superglm_adapter
from pricing_pipeline.reporting import UnderwriterReportOptions, build_scored_model_report
from pricing_pipeline.reporting._core import UnderwriterReportError
from pricing_pipeline.reporting.adapters.rating_workbook import RatingWorkbookAdapter
from pricing_pipeline.reporting.adapters.superglm import (
    SuperGLMReportAdapter,
    SuppliedTweedieLikelihoodAdapter,
)
from pricing_pipeline.reporting.evidence import (
    EvidenceRequest,
    ReportContext,
    normalize_model_evidence,
)


def _context(
    frame: pd.DataFrame,
    *,
    model_name: str,
    prediction: np.ndarray,
    features: tuple[str, ...],
    problem_type: str,
    deviance_power: float,
) -> ReportContext:
    return ReportContext(
        frame=frame.loc[:, list(features)].reset_index(drop=True),
        actual=frame["actual"].to_numpy(dtype=float),
        predictions={model_name: np.asarray(prediction, dtype=float)},
        weight=frame["weight"].to_numpy(dtype=float),
        features=features,
        comparison_unit_codes=np.arange(len(frame), dtype=np.intp),
        comparison_units=len(frame),
        minimum_cell_size=2,
        problem_type=problem_type,
        deviance_power=deviance_power,
    )


def test_adapter_package_lazily_exports_public_adapters():
    from pricing_pipeline.reporting import adapters

    assert adapters.RatingWorkbookAdapter is RatingWorkbookAdapter
    assert adapters.SuperGLMReportAdapter is SuperGLMReportAdapter
    assert adapters.SuppliedTweedieLikelihoodAdapter is SuppliedTweedieLikelihoodAdapter


def _poisson_model_and_context() -> tuple[SuperGLM, ReportContext]:
    rng = np.random.default_rng(842)
    row_count = 180
    frame = pd.DataFrame(
        {
            "segment": np.resize(["A", "B", "C"], row_count),
            "age": rng.uniform(18, 80, row_count),
            "weight": rng.uniform(0.5, 2.0, row_count),
        }
    )
    eta = (
        -1.2
        + 0.3 * (frame["segment"] == "B")
        + 0.55 * (frame["segment"] == "C")
        + 0.012 * (frame["age"] - 45)
    )
    frame["actual"] = rng.poisson(np.exp(eta))
    model = SuperGLM(
        features={"segment": Categorical(base="first"), "age": Spline(k=6)},
        selection_penalty=0.0,
    ).fit(
        frame[["segment", "age"]],
        frame["actual"],
        sample_weight=frame["weight"],
    )
    prediction = model.predict(frame[["segment", "age"]])
    return model, _context(
        frame,
        model_name="Fitted GAM",
        prediction=prediction,
        features=("segment", "age"),
        problem_type="frequency",
        deviance_power=1.0,
    )


def test_superglm_adapter_preserves_native_editor_evidence():
    model, context = _poisson_model_and_context()
    evidence = normalize_model_evidence(
        "Fitted GAM",
        SuperGLMReportAdapter().collect(
            model_name="Fitted GAM",
            source=model,
            context=context,
        ),
        context,
    )

    editor = EditorSession.from_model(
        model,
        n_points=200,
        centering="native",
        with_se=True,
        train_data=(context.frame, context.actual, context.weight),
    )
    payload = session_payload(editor)

    assert evidence.source == "SuperGLM object"
    assert evidence.importance is not None
    assert evidence.interactions == {}
    assert evidence.importance.method == "native_link_variance"
    assert set(evidence.importance.table["feature"]) == {"segment", "age"}
    for feature, effect in evidence.main_effects.items():
        term = editor.terms[feature]
        assert effect.semantic == "native_component"
        assert effect.source == "SuperGLM object"
        assert effect.effect["value"].to_numpy() == pytest.approx(np.exp(term.original_log_effect))
        assert effect.effect["lower"].to_numpy() == pytest.approx(np.exp(term.ci_lower_log_effect))
        assert effect.effect["upper"].to_numpy() == pytest.approx(np.exp(term.ci_upper_log_effect))
        assert effect.effective_df == pytest.approx(term.metadata["edf"])
    assert evidence.main_effects["segment"].effect["label"].tolist() == ["A", "B", "C"]
    age_density = evidence.main_effects["age"].density
    assert age_density is not None
    assert age_density["x"].to_numpy() == pytest.approx(payload["age"]["exposure"]["x"])
    assert age_density["density"].to_numpy() == pytest.approx(payload["age"]["exposure"]["y"])

    exact = evidence.exact_loss
    assert exact is not None
    assert exact.score_label == "Exact NLL"
    assert exact.comparison_group == "poisson"
    assert exact.size_basis == "weight_sum"
    assert exact.family == "Poisson"
    assert exact.tweedie_power == 1.0
    assert exact.dispersion == 1.0
    assert len(exact.contributions) == len(context.actual)
    assert np.isfinite(exact.contributions).all()


@pytest.mark.parametrize(
    ("family", "problem_type", "power", "target"),
    [
        ("poisson", "frequency", 1.0, np.resize([0.0, 1.0, 0.0, 2.0], 80)),
        (Gamma(), "severity", 2.0, np.linspace(0.2, 3.0, 80)),
        (Tweedie(p=1.5), "burn_cost", 1.5, np.resize([0.0, 0.4, 1.2, 2.8], 80)),
    ],
    ids=["poisson", "gamma", "compound-tweedie"],
)
def test_superglm_adapter_derives_exact_fitted_likelihood(
    family,
    problem_type: str,
    power: float,
    target: np.ndarray,
):
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, len(target)),
            "weight": np.linspace(0.5, 1.5, len(target)),
            "actual": target,
        }
    )
    model = SuperGLM(
        family=family,
        features={"x": Numeric()},
        selection_penalty=0.0,
    ).fit(frame[["x"]], target, sample_weight=frame["weight"])
    context = _context(
        frame,
        model_name="Fitted",
        prediction=model.predict(frame[["x"]]),
        features=("x",),
        problem_type=problem_type,
        deviance_power=power,
    )

    evidence = normalize_model_evidence(
        "Fitted",
        SuperGLMReportAdapter().collect(
            model_name="Fitted",
            source=model,
            context=context,
        ),
        context,
    )

    exact = evidence.exact_loss
    assert exact is not None
    assert exact.tweedie_power == power
    assert exact.dispersion == pytest.approx(1.0 if power == 1.0 else model.result.phi)
    assert exact.size_basis == ("row_count" if 1.0 < power < 2.0 else "weight_sum")
    assert exact.contributions.shape == context.actual.shape
    assert np.isfinite(exact.contributions).all()


@pytest.mark.parametrize(
    ("power", "dispersion", "actual", "prediction", "weight", "expected"),
    [
        (
            1.0,
            1.0,
            [0.0, 1.0, 3.0],
            [0.25, 1.3, 2.1],
            [0.5, 1.25, 2.75],
            [0.125, 1.2970446694156363, 4.58135544635979],
        ),
        (
            2.0,
            0.4,
            [0.5, 1.75, 3.25],
            [0.8, 1.4, 2.6],
            [0.4, 1.3, 2.2],
            [0.01532717333677023, 1.4569268355686593, 3.8274547494560385],
        ),
    ],
    ids=["poisson", "gamma"],
)
def test_supplied_exact_loss_matches_closed_form_full_vectors(
    power: float,
    dispersion: float,
    actual: list[float],
    prediction: list[float],
    weight: list[float],
    expected: list[float],
):
    frame = pd.DataFrame(
        {
            "x": np.arange(len(actual), dtype=float),
            "actual": actual,
            "weight": weight,
        }
    )
    problem_type = "frequency" if power == 1.0 else "severity"
    context = _context(
        frame,
        model_name="Exact",
        prediction=np.asarray(prediction),
        features=("x",),
        problem_type=problem_type,
        deviance_power=power,
    )

    evidence = SuppliedTweedieLikelihoodAdapter(
        tweedie_power=power,
        dispersion=dispersion,
    ).collect(model_name="Exact", source=None, context=context)

    assert evidence.exact_loss is not None
    assert evidence.exact_loss.contributions == pytest.approx(expected)


def test_fitted_compound_tweedie_matches_public_density_full_vector():
    target = np.resize([0.0, 0.3, 1.1, 2.6], 80)
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, len(target)),
            "weight": np.linspace(0.35, 1.85, len(target)),
            "actual": target,
        }
    )
    model = SuperGLM(
        family=Tweedie(p=1.5),
        features={"x": Numeric()},
        selection_penalty=0.0,
    ).fit(frame[["x"]], target, sample_weight=frame["weight"])
    prediction = model.predict(frame[["x"]])
    context = _context(
        frame,
        model_name="Compound",
        prediction=prediction,
        features=("x",),
        problem_type="burn_cost",
        deviance_power=1.5,
    )

    evidence = SuperGLMReportAdapter().collect(
        model_name="Compound",
        source=model,
        context=context,
    )

    assert evidence.exact_loss is not None
    expected = -tweedie_logpdf(
        target,
        prediction,
        model.result.phi,
        1.5,
        weights=frame["weight"].to_numpy(),
    )
    assert evidence.exact_loss.contributions == pytest.approx(expected)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("variance_eta", np.nan),
        ("variance_eta", np.inf),
        ("edf", np.nan),
        ("edf", np.inf),
    ],
)
def test_native_importance_rejects_non_finite_columns(column: str, bad_value: float):
    class InvalidImportanceModel:
        def term_importance(self, frame, weight):
            del frame, weight
            table = pd.DataFrame(
                {
                    "term": ["x"],
                    "feature": ["x"],
                    "variance_eta": [0.25],
                    "sd_eta": [0.5],
                    "edf": [1.0],
                }
            )
            table.loc[0, column] = bad_value
            return table

    frame = pd.DataFrame({"x": [0.0, 1.0], "actual": [0.2, 0.4], "weight": [1.0, 2.0]})
    context = _context(
        frame,
        model_name="Invalid",
        prediction=np.array([0.3, 0.5]),
        features=("x",),
        problem_type="burn_cost",
        deviance_power=1.5,
    )

    with pytest.raises(UnderwriterReportError, match=column):
        superglm_adapter._model_importance(InvalidImportanceModel(), context)


def test_native_importance_groups_literal_term_rows_independently():
    class MultiTermModel:
        def term_importance(self, frame, weight):
            del frame, weight
            return pd.DataFrame(
                {
                    "term": ["age-linear", "segment", "age-spline"],
                    "feature": ["age", "segment", "age"],
                    "variance_eta": [0.4, 0.2, 0.7],
                    "sd_eta": [0.632455532, 0.447213595, 0.836660027],
                    "edf": [1.2, 2.0, 2.3],
                }
            )

    frame = pd.DataFrame(
        {
            "age": [20.0, 40.0],
            "segment": ["A", "B"],
            "actual": [0.2, 0.4],
            "weight": [1.0, 2.0],
        }
    )
    context = _context(
        frame,
        model_name="Grouped",
        prediction=np.array([0.3, 0.5]),
        features=("age", "segment"),
        problem_type="burn_cost",
        deviance_power=1.5,
    )

    result = superglm_adapter._model_importance(MultiTermModel(), context).table.set_index(
        "feature"
    )

    assert result.loc["age", "magnitude"] == pytest.approx(1.1)
    assert result.loc["age", "effective_df"] == pytest.approx(3.5)
    assert result.loc["segment", "magnitude"] == pytest.approx(0.2)
    assert result.loc["segment", "effective_df"] == pytest.approx(2.0)


def test_superglm_adapter_rejects_supplied_metadata_conflicting_with_fitted_object():
    model, context = _poisson_model_and_context()

    with pytest.raises(ValueError, match="does not match the fitted SuperGLM object"):
        SuperGLMReportAdapter(tweedie_power=1.5, dispersion=0.72).collect(
            model_name="Fitted GAM",
            source=model,
            context=context,
        )


def test_unsupported_superglm_capabilities_have_stable_plain_text_reasons(monkeypatch):
    class UnsupportedModel:
        def term_importance(self, frame, weight):
            del frame, weight
            raise NotImplementedError("<upstream importance implementation detail>")

    def unsupported_editor(*args, **kwargs):
        del args, kwargs
        raise NotImplementedError("<upstream editor implementation detail>")

    monkeypatch.setattr(superglm_adapter.EditorSession, "from_model", unsupported_editor)
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, 4),
            "actual": np.linspace(0.2, 0.8, 4),
            "weight": np.ones(4),
        }
    )
    context = _context(
        frame,
        model_name="Unsupported",
        prediction=np.linspace(0.3, 0.9, 4),
        features=("x",),
        problem_type="burn_cost",
        deviance_power=1.5,
    )

    evidence = normalize_model_evidence(
        "Unsupported",
        SuperGLMReportAdapter().collect(
            model_name="Unsupported",
            source=UnsupportedModel(),
            context=context,
        ),
        context,
    )

    assert {item.capability: item.reason for item in evidence.unavailable} == {
        "main_effects": "native main effects are not supported",
        "importance": "native term importance is not supported",
        "interactions": "native interaction reporting is not supported",
        "exact_loss": "fitted distribution does not expose a supported exact likelihood",
    }
    assert all("upstream" not in item.reason for item in evidence.unavailable)


def _editor_context() -> ReportContext:
    frame = pd.DataFrame({"x": [0.0, 1.0], "actual": [0.2, 0.4], "weight": [1.0, 2.0]})
    return _context(
        frame,
        model_name="Editor",
        prediction=np.array([0.3, 0.5]),
        features=("x",),
        problem_type="burn_cost",
        deviance_power=1.5,
    )


def _valid_editor_term() -> SimpleNamespace:
    return SimpleNamespace(
        original_log_effect=np.array([0.0, 0.1]),
        levels=None,
        x=np.array([0.0, 1.0]),
        ci_lower_log_effect=np.array([-0.1, 0.0]),
        ci_upper_log_effect=np.array([0.1, 0.2]),
        metadata={"edf": 1.5},
    )


def _install_editor_result(
    monkeypatch,
    *,
    term: SimpleNamespace,
    payload: object,
) -> None:
    session = SimpleNamespace(terms={"x": term})
    monkeypatch.setattr(
        superglm_adapter.EditorSession,
        "from_model",
        lambda *args, **kwargs: session,
    )
    monkeypatch.setattr(superglm_adapter, "session_payload", lambda session: payload)


@pytest.mark.parametrize("edf", [np.nan, np.inf])
def test_native_main_effect_rejects_non_finite_edf(monkeypatch, edf: float):
    term = _valid_editor_term()
    term.metadata["edf"] = edf
    _install_editor_result(
        monkeypatch,
        term=term,
        payload={
            "x": {
                "exposure": {
                    "kind": "density",
                    "x": [0.0, 1.0],
                    "y": [1.0, 2.0],
                }
            }
        },
    )

    with pytest.raises(UnderwriterReportError, match="effective_df"):
        superglm_adapter._model_main_effects(object(), _editor_context(), n_points=2)


@pytest.mark.parametrize(
    "malformation",
    ["missing-term-attribute", "missing-payload-term", "invalid-metadata", "overflow"],
)
def test_malformed_editor_evidence_raises_report_error(monkeypatch, malformation: str):
    term = _valid_editor_term()
    payload = {
        "x": {
            "exposure": {
                "kind": "density",
                "x": [0.0, 1.0],
                "y": [1.0, 2.0],
            }
        }
    }
    if malformation == "missing-term-attribute":
        del term.original_log_effect
    elif malformation == "missing-payload-term":
        payload = {}
    elif malformation == "invalid-metadata":
        term.metadata = None
    else:
        term.original_log_effect = np.array([0.0, 1000.0])
    _install_editor_result(monkeypatch, term=term, payload=payload)

    with pytest.raises(UnderwriterReportError, match="main effect"):
        superglm_adapter._model_main_effects(object(), _editor_context(), n_points=2)


def test_malformed_term_not_implemented_error_is_not_capability_unavailable(monkeypatch):
    class MalformedTerm:
        @property
        def original_log_effect(self):
            raise NotImplementedError("malformed term attribute")

    class OtherwiseSupportedModel:
        def term_importance(self, frame, weight):
            del frame, weight
            return pd.DataFrame(
                {
                    "term": ["x"],
                    "feature": ["x"],
                    "variance_eta": [0.25],
                    "sd_eta": [0.5],
                    "edf": [1.0],
                }
            )

    _install_editor_result(
        monkeypatch,
        term=MalformedTerm(),
        payload={"x": {"exposure": None}},
    )

    with pytest.raises(UnderwriterReportError, match="main effect"):
        SuperGLMReportAdapter().collect(
            model_name="Editor",
            source=OtherwiseSupportedModel(),
            context=_editor_context(),
        )


def test_editor_receives_only_positive_weight_aligned_rows(monkeypatch, tmp_path: Path):
    frame = pd.DataFrame(
        {
            "x": [-100.0, 10.0, 20.0, 100.0, 30.0, 40.0],
            "actual": [9.0, 0.0, 1.0, 11.0, 0.0, 2.0],
            "weight": [0.0, 0.5, 1.25, 0.0, 2.0, 0.75],
        }
    )
    model = SuperGLM(
        features={"x": Spline(k=5)},
        selection_penalty=0.0,
    ).fit(frame[["x"]], frame["actual"], sample_weight=frame["weight"])
    frame["prediction"] = model.predict(frame[["x"]])
    captured: list[tuple[pd.DataFrame, np.ndarray, np.ndarray]] = []
    original = superglm_adapter.EditorSession.from_model

    def capture_train_data(*args, **kwargs):
        train_frame, actual, weight = kwargs["train_data"]
        captured.append(
            (
                train_frame.copy(deep=True),
                np.asarray(actual).copy(),
                np.asarray(weight).copy(),
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(superglm_adapter.EditorSession, "from_model", capture_train_data)

    build_scored_model_report(
        frame,
        actual="actual",
        predictions={"Filtered": "prediction"},
        sample_weight="weight",
        features=["x"],
        evidence_requests=(EvidenceRequest("Filtered", SuperGLMReportAdapter(), model),),
        output_path=tmp_path / "filtered.html",
        options=UnderwriterReportOptions(problem_type="frequency", minimum_cell_size=2),
    )

    assert len(captured) == 1
    train_frame, actual, weight = captured[0]
    assert train_frame["x"].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert actual.tolist() == [0.0, 1.0, 0.0, 2.0]
    assert weight.tolist() == [0.5, 1.25, 2.0, 0.75]


def test_supplied_tweedie_likelihood_uses_exact_superglm_series():
    actual = np.resize([0.0, 0.4, 1.2, 2.8], 40)
    prediction = np.linspace(0.25, 1.75, len(actual))
    weight = np.linspace(0.5, 1.5, len(actual))
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, len(actual)),
            "actual": actual,
            "weight": weight,
        }
    )
    context = _context(
        frame,
        model_name="Challenger",
        prediction=prediction,
        features=("x",),
        problem_type="burn_cost",
        deviance_power=1.5,
    )

    evidence = normalize_model_evidence(
        "Challenger",
        SuppliedTweedieLikelihoodAdapter(tweedie_power=1.5, dispersion=0.72).collect(
            model_name="Challenger",
            source=None,
            context=context,
        ),
        context,
    )

    exact = evidence.exact_loss
    assert exact is not None
    expected = -tweedie_logpdf(actual, prediction, 0.72, 1.5, weights=weight)
    assert exact.source == "supplied training metadata"
    assert exact.family == "Tweedie"
    assert exact.comparison_group == "tweedie:1.5"
    assert exact.size_basis == "row_count"
    assert exact.contributions == pytest.approx(expected)
    assert math.isfinite(float(exact.contributions.sum()))


def test_rating_workbook_adapter_emits_native_evidence_only(tmp_path: Path):
    raw = pd.DataFrame([[None] * 3 for _ in range(11)])
    raw.iat[4, 0] = "segment"
    raw.iloc[6, 0:3] = ["segment", "Relativity", "Weight"]
    raw.iloc[7, 0:3] = ["A", 0.8, 10.0]
    raw.iloc[8, 0:3] = ["B", 1.0, 20.0]
    raw.iloc[9, 0:3] = ["C", 1.25, 30.0]
    workbook = tmp_path / "rating_tables.xlsx"
    raw.to_excel(workbook, sheet_name="Rating Tables", header=False, index=False)
    frame = pd.DataFrame(
        {
            "segment": np.resize(["A", "B", "C"], 12),
            "actual": np.linspace(0.1, 1.2, 12),
            "weight": np.ones(12),
        }
    )
    context = _context(
        frame,
        model_name="Published GAM",
        prediction=np.linspace(0.2, 1.3, 12),
        features=("segment",),
        problem_type="burn_cost",
        deviance_power=1.5,
    )

    evidence = normalize_model_evidence(
        "Published GAM",
        RatingWorkbookAdapter().collect(
            model_name="Published GAM",
            source=workbook,
            context=context,
        ),
        context,
    )

    assert evidence.source == "rating workbook"
    assert evidence.exact_loss is None
    assert evidence.interactions == {}
    assert evidence.importance is not None
    assert evidence.importance.method == "export_log_relativity_variance"
    assert evidence.importance.table.loc[0, "magnitude"] == pytest.approx(0.02766280249617631)
    effect = evidence.main_effects["segment"]
    assert effect.semantic == "native_component"
    assert effect.effect["label"].tolist() == ["A", "B", "C"]
    assert effect.effect["value"].to_numpy() == pytest.approx([0.8, 1.0, 1.25])


def test_rating_workbook_rejects_mismatched_level_header(tmp_path: Path):
    raw = pd.DataFrame([[None] * 3 for _ in range(10)])
    raw.iat[4, 0] = "segment"
    raw.iloc[6, 0:3] = ["age", "Relativity", "Weight"]
    raw.iloc[7, 0:3] = ["A", 0.8, 10.0]
    workbook = tmp_path / "mismatched_header.xlsx"
    raw.to_excel(workbook, sheet_name="Rating Tables", header=False, index=False)

    with pytest.raises(UnderwriterReportError, match="level header"):
        RatingWorkbookAdapter().collect(
            model_name="Published",
            source=workbook,
            context=_editor_context(),
        )


def test_rating_workbook_rejects_duplicate_term_titles(tmp_path: Path):
    raw = pd.DataFrame([[None] * 6 for _ in range(10)])
    for column in (0, 3):
        raw.iat[4, column] = "x"
        raw.iloc[6, column : column + 3] = ["x", "Relativity", "Weight"]
        raw.iloc[7, column : column + 3] = ["A", 0.8, 10.0]
    workbook = tmp_path / "duplicate_titles.xlsx"
    raw.to_excel(workbook, sheet_name="Rating Tables", header=False, index=False)

    with pytest.raises(UnderwriterReportError, match="duplicate term"):
        RatingWorkbookAdapter().collect(
            model_name="Published",
            source=workbook,
            context=_editor_context(),
        )
