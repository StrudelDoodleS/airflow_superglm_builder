from __future__ import annotations

import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from superglm.links import (
    IdentityLink,
    LogLink,
    NegativeBinomialLink,
    PowerLink,
)


def _api():
    try:
        return importlib.import_module("pricing_pipeline.modeling.validation_curves")
    except ModuleNotFoundError as exc:
        pytest.fail(f"validation curve normalizer is not implemented: {exc}")


def _continuous_term():
    x = np.array([10.0, 20.0, 30.0])
    return {
        "family": "continuous",
        "domain": {"x": x.copy()},
        "support": {"x": x.copy(), "density": np.array([2.0, 8.0, 8.0])},
        "curves": {
            "response": {
                "fold_0": np.array([-999.0, -999.0, -999.0]),
                "fold_1": np.array([-999.0, -999.0, -999.0]),
            },
            "link": {
                "fold_0": np.array([1.0, 3.0, 7.0]),
                "fold_1": np.array([-2.0, 1.0, 0.0]),
            },
        },
    }


def _level_term():
    levels = ["low", "mid", "high"]
    return {
        "family": "level",
        "domain": {"levels": list(levels)},
        "support": {"levels": list(levels), "density": np.array([3.0, 9.0, 9.0])},
        "curves": {
            "response": {
                "fold_0": np.array([-999.0, -999.0, -999.0]),
                "fold_1": np.array([-999.0, -999.0, -999.0]),
            },
            "link": {
                "fold_0": np.array([0.2, 0.5, -0.1]),
                "fold_1": np.array([1.0, 0.25, 0.75]),
            },
        },
    }


def _estimators(link=None):
    resolved_link = LogLink() if link is None else link
    return [SimpleNamespace(_link=resolved_link), SimpleNamespace(_link=type(resolved_link)())]


def test_normalizes_all_supported_main_effect_families_atomically_in_stable_order():
    api = _api()
    payload = {
        "spline_term": _continuous_term(),
        "ordered_term": _level_term(),
        "numeric_term": _continuous_term(),
        "categorical_term": _level_term(),
        "polynomial_term": _continuous_term(),
    }

    capture = api.normalize_validation_curves(
        payload,
        estimators=_estimators(),
        fold_count=2,
    )

    assert capture.status == "COMPLETE"
    assert capture.reason is None
    assert len(capture.points) == 30
    assert [
        (point.validation_split_no, point.term_name, point.point_no)
        for point in capture.points
    ] == [
        (split_no, term_name, point_no)
        for split_no in (1, 2)
        for term_name in (
            "categorical_term",
            "numeric_term",
            "ordered_term",
            "polynomial_term",
            "spline_term",
        )
        for point_no in (1, 2, 3)
    ]

    numeric = [
        point
        for point in capture.points
        if point.validation_split_no == 1 and point.term_name == "numeric_term"
    ]
    assert [point.point_kind for point in numeric] == ["NUMERIC"] * 3
    assert [point.x_numeric for point in numeric] == [10.0, 20.0, 30.0]
    assert [point.level_text for point in numeric] == [None, None, None]
    assert [point.reference_value for point in numeric] == [20.0, 20.0, 20.0]
    assert [point.reference_level for point in numeric] == [None, None, None]
    assert [point.support_value for point in numeric] == [2.0, 8.0, 8.0]
    assert [point.eta_contribution for point in numeric] == [-2.0, 0.0, 4.0]
    assert [point.relativity for point in numeric] == pytest.approx(np.exp([-2.0, 0.0, 4.0]))

    ordered = [
        point
        for point in capture.points
        if point.validation_split_no == 2 and point.term_name == "ordered_term"
    ]
    assert [point.point_kind for point in ordered] == ["LEVEL"] * 3
    assert [point.x_numeric for point in ordered] == [None, None, None]
    assert [point.level_text for point in ordered] == ["low", "mid", "high"]
    assert [point.reference_value for point in ordered] == [None, None, None]
    assert [point.reference_level for point in ordered] == ["mid", "mid", "mid"]
    assert [point.eta_contribution for point in ordered] == pytest.approx([0.75, 0.0, 0.5])
    assert [point.relativity for point in ordered] == pytest.approx(np.exp([0.75, 0.0, 0.5]))


def test_non_log_link_leaves_relativity_null_and_never_trusts_response_curves():
    api = _api()
    term = _continuous_term()
    term["curves"]["response"] = object()

    capture = api.normalize_validation_curves(
        {"numeric_term": term},
        estimators=_estimators(IdentityLink()),
        fold_count=2,
    )

    assert capture.status == "COMPLETE"
    assert all(point.relativity is None for point in capture.points)


def test_capture_record_enforces_atomic_status_reason_and_points():
    api = _api()
    point = api.normalize_validation_curves(
        {"numeric_term": _continuous_term()},
        estimators=_estimators(),
        fold_count=2,
    ).points[0]

    invalid = (
        ("PARTIAL", None, ()),
        ("COMPLETE", None, ()),
        ("COMPLETE", "unexpected", (point,)),
        ("UNAVAILABLE", None, ()),
        ("UNAVAILABLE", "failed", (point,)),
        ("UNAVAILABLE", "x" * 501, ()),
    )

    for status, reason, points in invalid:
        with pytest.raises(ValueError):
            api.ValidationCurveCapture(
                status=status,
                reason=reason,
                points=points,
            )


@pytest.mark.parametrize("payload", [None, {}, [], {"unknown": {"family": "surface"}}])
def test_missing_empty_or_all_unsupported_payload_is_unavailable(payload):
    api = _api()

    capture = api.normalize_validation_curves(
        payload,
        estimators=_estimators(),
        fold_count=2,
    )

    assert capture.status == "UNAVAILABLE"
    assert capture.reason
    assert len(capture.reason) <= 500
    assert capture.points == ()


def test_unsupported_entries_are_skipped_without_weakening_supported_capture():
    api = _api()
    payload = {
        "interaction_surface": {
            "family": "surface",
            "domain": None,
            "support": None,
            "curves": None,
        },
        "interaction": {"family": "interaction"},
        "numeric_term": _continuous_term(),
        "offset": {"family": "offset"},
    }

    capture = api.normalize_validation_curves(
        payload,
        estimators=_estimators(),
        fold_count=2,
    )

    assert capture.status == "COMPLETE"
    assert {point.term_name for point in capture.points} == {"numeric_term"}


@pytest.mark.parametrize(
    "ambiguous_entry",
    [
        None,
        {},
        {"domain": {"x": [1.0]}},
        {"family": "mystery"},
    ],
    ids=("null", "empty", "missing-family", "unknown-family"),
)
def test_ambiguous_top_level_entry_invalidates_otherwise_valid_capture(
    ambiguous_entry,
):
    api = _api()

    capture = api.normalize_validation_curves(
        {
            "numeric_term": _continuous_term(),
            "ambiguous_term": ambiguous_entry,
        },
        estimators=_estimators(),
        fold_count=2,
    )

    assert capture.status == "UNAVAILABLE"
    assert capture.reason
    assert capture.points == ()


def test_normalized_top_level_term_name_collisions_are_unavailable():
    api = _api()

    capture = api.normalize_validation_curves(
        {
            "age": _continuous_term(),
            " age ": _continuous_term(),
        },
        estimators=_estimators(),
        fold_count=2,
    )

    assert capture.status == "UNAVAILABLE"
    assert "collision" in capture.reason
    assert capture.points == ()


def _malformed_continuous(case: str):
    term = copy.deepcopy(_continuous_term())
    match case:
        case "domain_not_mapping":
            term["domain"] = None
        case "domain_missing_x":
            term["domain"] = {}
        case "domain_empty":
            term["domain"]["x"] = []
        case "domain_nested":
            term["domain"]["x"] = [[10.0, 20.0, 30.0]]
        case "domain_nonfinite":
            term["domain"]["x"] = [10.0, float("nan"), 30.0]
        case "support_not_mapping":
            term["support"] = None
        case "support_missing_domain":
            term["support"].pop("x")
        case "support_domain_mismatch":
            term["support"]["x"] = [10.0, 21.0, 30.0]
        case "support_density_missing":
            term["support"].pop("density")
        case "support_density_empty":
            term["support"]["density"] = []
        case "support_density_nested":
            term["support"]["density"] = [[2.0, 8.0, 8.0]]
        case "support_density_wrong_length":
            term["support"]["density"] = [2.0, 8.0]
        case "support_density_nonfinite":
            term["support"]["density"] = [2.0, float("inf"), 8.0]
        case "support_density_negative":
            term["support"]["density"] = [2.0, -1.0, 8.0]
        case "curves_not_mapping":
            term["curves"] = None
        case "link_curves_missing":
            term["curves"].pop("link")
        case "link_curves_not_mapping":
            term["curves"]["link"] = []
        case "fold_label_missing":
            term["curves"]["link"].pop("fold_1")
        case "fold_label_extra":
            term["curves"]["link"]["fold_2"] = np.array([1.0, 2.0, 3.0])
        case "curve_nested":
            term["curves"]["link"]["fold_0"] = [[1.0, 3.0, 7.0]]
        case "curve_wrong_length":
            term["curves"]["link"]["fold_0"] = [1.0, 3.0]
        case "curve_nonfinite":
            term["curves"]["link"]["fold_0"] = [1.0, float("nan"), 7.0]
        case _:
            raise AssertionError(f"unknown malformed case {case}")
    return term


@pytest.mark.parametrize(
    "case",
    [
        "domain_not_mapping",
        "domain_missing_x",
        "domain_empty",
        "domain_nested",
        "domain_nonfinite",
        "support_not_mapping",
        "support_missing_domain",
        "support_domain_mismatch",
        "support_density_missing",
        "support_density_empty",
        "support_density_nested",
        "support_density_wrong_length",
        "support_density_nonfinite",
        "support_density_negative",
        "curves_not_mapping",
        "link_curves_missing",
        "link_curves_not_mapping",
        "fold_label_missing",
        "fold_label_extra",
        "curve_nested",
        "curve_wrong_length",
        "curve_nonfinite",
    ],
)
def test_any_malformed_supported_continuous_term_is_atomically_unavailable(case):
    api = _api()
    payload = {
        "valid_level": _level_term(),
        "broken_continuous": _malformed_continuous(case),
    }

    capture = api.normalize_validation_curves(
        payload,
        estimators=_estimators(),
        fold_count=2,
    )

    assert capture.status == "UNAVAILABLE"
    assert capture.reason
    assert len(capture.reason) <= 500
    assert capture.points == ()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda term: term["domain"].__setitem__("levels", ["low", "low", "high"]),
        lambda term: term["domain"].__setitem__("levels", ["low", "", "high"]),
        lambda term: term["support"].__setitem__("levels", ["low", "high", "mid"]),
    ],
    ids=("duplicate-level", "blank-level", "support-level-mismatch"),
)
def test_malformed_level_domains_are_atomically_unavailable(mutation):
    api = _api()
    term = _level_term()
    mutation(term)

    capture = api.normalize_validation_curves(
        {"level_term": term, "valid_numeric": _continuous_term()},
        estimators=_estimators(),
        fold_count=2,
    )

    assert capture.status == "UNAVAILABLE"
    assert capture.reason
    assert capture.points == ()


def test_scalar_levels_are_normalized_to_text_without_losing_reference_identity():
    api = _api()
    term = _level_term()
    term["domain"]["levels"] = ["low", 2, 3.5]
    term["support"]["levels"] = ["low", 2, 3.5]

    capture = api.normalize_validation_curves(
        {"level_term": term},
        estimators=_estimators(),
        fold_count=2,
    )

    assert capture.status == "COMPLETE"
    first_fold = [
        point for point in capture.points if point.validation_split_no == 1
    ]
    assert [point.level_text for point in first_fold] == ["low", "2", "3.5"]
    assert [point.reference_level for point in first_fold] == ["2", "2", "2"]


def test_all_zero_support_uses_the_first_point_as_the_shared_reference():
    api = _api()
    term = _continuous_term()
    term["support"]["density"] = [0.0, 0.0, 0.0]

    capture = api.normalize_validation_curves(
        {"numeric_term": term},
        estimators=_estimators(),
        fold_count=2,
    )

    assert capture.status == "COMPLETE"
    first_fold = [
        point for point in capture.points if point.validation_split_no == 1
    ]
    assert [point.reference_value for point in first_fold] == [10.0, 10.0, 10.0]
    assert [point.eta_contribution for point in first_fold] == [0.0, 2.0, 6.0]


def test_large_finite_support_values_do_not_require_a_finite_total():
    api = _api()
    term = _continuous_term()
    term["support"]["density"] = [1e308, 1e308, 0.0]

    capture = api.normalize_validation_curves(
        {"numeric_term": term},
        estimators=_estimators(),
        fold_count=2,
    )

    assert capture.status == "COMPLETE"
    first_fold = [
        point for point in capture.points if point.validation_split_no == 1
    ]
    assert [point.reference_value for point in first_fold] == [10.0, 10.0, 10.0]
    assert [point.support_value for point in first_fold] == [1e308, 1e308, 0.0]


@pytest.mark.parametrize(
    "estimators",
    [
        [],
        [SimpleNamespace(_link=LogLink())],
        [SimpleNamespace(_link=LogLink()), SimpleNamespace(_link=IdentityLink())],
        [SimpleNamespace(), SimpleNamespace()],
        [SimpleNamespace(_link=None), SimpleNamespace(_link=None)],
    ],
    ids=("none", "wrong-count", "mismatch", "missing", "null"),
)
def test_missing_or_mismatched_fold_estimator_links_are_unavailable(estimators):
    api = _api()

    capture = api.normalize_validation_curves(
        {"numeric_term": _continuous_term()},
        estimators=estimators,
        fold_count=2,
    )

    assert capture.status == "UNAVAILABLE"
    assert capture.reason
    assert capture.points == ()


@pytest.mark.parametrize(
    "links",
    [
        (PowerLink(power=0.5), PowerLink(power=0.5)),
        (NegativeBinomialLink(theta=2.0), NegativeBinomialLink(theta=2.0)),
    ],
    ids=("power", "negative-binomial"),
)
def test_equal_parameterized_fold_links_are_semantically_compatible(links):
    api = _api()

    capture = api.normalize_validation_curves(
        {"numeric_term": _continuous_term()},
        estimators=[SimpleNamespace(_link=link) for link in links],
        fold_count=2,
    )

    assert capture.status == "COMPLETE"
    assert all(point.relativity is None for point in capture.points)


@pytest.mark.parametrize(
    "links",
    [
        (PowerLink(power=0.5), PowerLink(power=2.0)),
        (NegativeBinomialLink(theta=1.0), NegativeBinomialLink(theta=2.0)),
    ],
    ids=("power", "negative-binomial"),
)
def test_different_parameterized_fold_links_are_semantically_incompatible(links):
    api = _api()

    capture = api.normalize_validation_curves(
        {"numeric_term": _continuous_term()},
        estimators=[SimpleNamespace(_link=link) for link in links],
        fold_count=2,
    )

    assert capture.status == "UNAVAILABLE"
    assert "link" in capture.reason
    assert capture.points == ()


def test_nonfinite_log_relativity_makes_the_whole_capture_unavailable():
    api = _api()
    term = _continuous_term()
    term["support"]["density"] = [9.0, 2.0, 1.0]
    term["curves"]["link"]["fold_0"] = [0.0, 1000.0, 1.0]

    capture = api.normalize_validation_curves(
        {"numeric_term": term},
        estimators=_estimators(),
        fold_count=2,
    )

    assert capture.status == "UNAVAILABLE"
    assert capture.reason
    assert capture.points == ()


def test_unavailable_reason_is_whitespace_collapsed_deterministic_and_bounded():
    api = _api()
    huge_name = "broken_" + "x" * 900
    term = _continuous_term()
    term["domain"] = None

    first = api.normalize_validation_curves(
        {huge_name: term},
        estimators=_estimators(),
        fold_count=2,
    )
    second = api.normalize_validation_curves(
        {huge_name: term},
        estimators=_estimators(),
        fold_count=2,
    )

    assert first.reason == second.reason
    assert first.reason.startswith(
        "validation curve capture unavailable: ValidationCurvePayloadError:"
    )
    assert "\n" not in first.reason
    assert len(first.reason) == 500
