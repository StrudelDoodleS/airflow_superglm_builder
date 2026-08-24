from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.reporting import evidence as evidence_module
from pricing_pipeline.reporting.evidence import InteractionEvidence, ReportContext


@pytest.fixture
def report_context() -> ReportContext:
    return ReportContext(
        frame=pd.DataFrame(
            {
                "age": [20.0, 40.0, 20.0, 40.0, 20.0, 40.0, 20.0, 40.0],
                "density": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0],
                "segment": ["A", "A", "B", "B", "C", "C", "D", "D"],
                "region": ["X", "X", "Y", "Y", "X", "X", "Y", "Y"],
            }
        ),
        actual=np.linspace(0.1, 0.8, 8),
        predictions={"Model A": np.linspace(0.2, 0.9, 8)},
        weight=np.ones(8),
        features=("age", "density", "segment", "region"),
        comparison_unit_codes=np.arange(8),
        comparison_units=8,
        minimum_cell_size=2,
        problem_type="burn_cost",
        deviance_power=1.5,
    )


def _surface() -> InteractionEvidence:
    return InteractionEvidence(
        name="age:density",
        parents=("age", "density"),
        semantic="partial_dependence",
        plot_kind="surface",
        effect=pd.DataFrame(
            {
                "x": [20.0, 40.0, 20.0, 40.0],
                "y": [1.0, 1.0, 2.0, 2.0],
                "value": [0.8, 1.0, 1.1, 1.3],
            }
        ),
        source="portable PDP",
        grid_axes={"x": np.array([20.0, 40.0]), "y": np.array([1.0, 2.0])},
    )


def _evidence_for(plot_kind: str) -> InteractionEvidence:
    if plot_kind == "surface":
        return _surface()
    specifications = {
        "categorical_heatmap": (
            ("segment", "region"),
            pd.DataFrame({"left": ["A"], "right": ["X"], "value": [1.0]}),
        ),
        "varying_coefficient": (
            ("age", "segment"),
            pd.DataFrame({"x": [20.0], "level": ["A"], "value": [1.0]}),
        ),
        "numeric_categorical": (
            ("age", "segment"),
            pd.DataFrame({"level": ["A"], "value": [1.0]}),
        ),
        "numeric_numeric": (
            ("age", "density"),
            pd.DataFrame({"value": [1.0]}),
        ),
        "factor_smooth": (
            ("age", "segment"),
            pd.DataFrame({"x": [20.0], "level": ["A"], "value": [1.0]}),
        ),
    }
    parents, effect = specifications[plot_kind]
    return InteractionEvidence(
        name=f"{parents[0]}:{parents[1]}",
        parents=parents,
        semantic="native_component" if plot_kind == "factor_smooth" else "partial_dependence",
        plot_kind=plot_kind,
        effect=effect,
        source="direct",
    )


def _normalize(evidence: InteractionEvidence, context: ReportContext) -> InteractionEvidence:
    return evidence_module.normalize_interaction_evidence("Model A", evidence, context)


@pytest.mark.parametrize(
    ("plot_kind", "required_column"),
    [
        ("surface", "x"),
        ("surface", "y"),
        ("surface", "value"),
        ("categorical_heatmap", "left"),
        ("categorical_heatmap", "right"),
        ("categorical_heatmap", "value"),
        ("varying_coefficient", "x"),
        ("varying_coefficient", "level"),
        ("varying_coefficient", "value"),
        ("numeric_categorical", "level"),
        ("numeric_categorical", "value"),
        ("numeric_numeric", "value"),
        ("factor_smooth", "x"),
        ("factor_smooth", "level"),
        ("factor_smooth", "value"),
    ],
)
def test_each_plot_kind_requires_its_declared_schema(
    report_context: ReportContext,
    plot_kind: str,
    required_column: str,
):
    evidence = _evidence_for(plot_kind)
    evidence = replace(evidence, effect=evidence.effect.drop(columns=required_column))

    with pytest.raises(
        ValueError,
        match=rf"interaction.effect is missing columns:.*{required_column}",
    ):
        _normalize(evidence, report_context)


@pytest.mark.parametrize(
    "plot_kind",
    [
        "surface",
        "categorical_heatmap",
        "varying_coefficient",
        "numeric_categorical",
        "numeric_numeric",
        "factor_smooth",
    ],
)
def test_each_plot_kind_rejects_unknown_and_non_finite_effect_columns(
    report_context: ReportContext,
    plot_kind: str,
):
    evidence = _evidence_for(plot_kind)
    unknown = replace(evidence, effect=evidence.effect.assign(row_secret="private"))
    non_finite = replace(evidence, effect=evidence.effect.assign(value=np.nan))

    with pytest.raises(ValueError, match="interaction.effect has unknown columns: row_secret"):
        _normalize(unknown, report_context)
    with pytest.raises(ValueError, match="interaction.effect.value.*finite"):
        _normalize(non_finite, report_context)


@pytest.mark.parametrize(
    "plot_kind",
    ["varying_coefficient", "numeric_categorical", "numeric_numeric", "factor_smooth"],
)
def test_curve_bounds_must_appear_together_and_bracket_values(
    report_context: ReportContext,
    plot_kind: str,
):
    evidence = _evidence_for(plot_kind)
    one_bound = replace(evidence, effect=evidence.effect.assign(lower=0.5))
    invalid_bounds = replace(
        evidence,
        effect=evidence.effect.assign(lower=1.1, upper=1.2),
    )

    with pytest.raises(ValueError, match="lower and upper must appear together"):
        _normalize(one_bound, report_context)
    with pytest.raises(ValueError, match="lower and upper must bracket value"):
        _normalize(invalid_bounds, report_context)


def test_interaction_parents_must_be_allowed_features(report_context: ReportContext):
    evidence = replace(_surface(), parents=("age", "policy_secret"))

    with pytest.raises(ValueError, match="interaction parent.*policy_secret.*not allowed"):
        _normalize(evidence, report_context)


def test_surface_density_has_an_exact_bounded_schema(report_context: ReportContext):
    density = pd.DataFrame(
        {
            "x": [20.0, 40.0, 20.0, 40.0],
            "y": [1.0, 1.0, 2.0, 2.0],
            "density": [0.2, 0.3, 0.1, 0.4],
            "hdr_mass": [0.5, 0.5, 0.9, 0.9],
        }
    )

    normalized = _normalize(replace(_surface(), density=density), report_context)

    assert normalized.density is not None
    assert normalized.density.columns.tolist() == ["x", "y", "density", "hdr_mass"]

    for bad_density, match in [
        (density.drop(columns="hdr_mass"), "exactly x, y, density, and hdr_mass"),
        (density.assign(row_secret="private"), "exactly x, y, density, and hdr_mass"),
        (density.assign(density=-0.1), "density must be non-negative"),
        (density.assign(hdr_mass=1.1), "hdr_mass must be between 0 and 1"),
    ]:
        with pytest.raises(ValueError, match=match):
            _normalize(replace(_surface(), density=bad_density), report_context)


def test_surface_hdr_mass_clips_only_machine_precision_boundary_noise(
    report_context: ReportContext,
):
    epsilon = np.finfo(float).eps
    density = pd.DataFrame(
        {
            "x": [20.0, 40.0, 20.0, 40.0],
            "y": [1.0, 1.0, 2.0, 2.0],
            "density": [0.2, 0.3, 0.1, 0.4],
            "hdr_mass": [-4 * epsilon, 0.5, 0.9, 1.0 + 4 * epsilon],
        }
    )

    normalized = _normalize(replace(_surface(), density=density), report_context)

    assert normalized.density is not None
    assert normalized.density["hdr_mass"].tolist() == [0.0, 0.5, 0.9, 1.0]


def test_surface_requires_a_bounded_rectangular_grid(report_context: ReportContext):
    normalized = _normalize(_surface(), report_context)

    assert len(normalized.effect) == 4


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda evidence: replace(
                evidence,
                effect=pd.concat([evidence.effect, evidence.effect.iloc[[0]]], ignore_index=True),
            ),
            "duplicate grid coordinates",
        ),
        (
            lambda evidence: replace(
                evidence,
                grid_axes={"x": np.array([20.0, 50.0]), "y": np.array([1.0, 2.0])},
            ),
            "grid_axes disagree with interaction.effect",
        ),
        (
            lambda evidence: replace(
                evidence,
                grid_axes={"x": np.arange(161, dtype=float), "y": np.array([1.0])},
                effect=pd.DataFrame({"x": np.arange(161, dtype=float), "y": 1.0, "value": 1.0}),
            ),
            "at most 160 points per axis",
        ),
        (
            lambda evidence: replace(
                evidence,
                grid_axes={"x": np.arange(160, dtype=float), "y": np.arange(161, dtype=float)},
                effect=pd.DataFrame(
                    {
                        "x": np.repeat(np.arange(160, dtype=float), 161),
                        "y": np.tile(np.arange(161, dtype=float), 160),
                        "value": 1.0,
                    }
                ),
            ),
            "at most 160 points per axis|at most 25,600 (?:cells|rows)",
        ),
    ],
)
def test_surface_rejects_unsafe_grids(
    report_context: ReportContext,
    mutation,
    match: str,
):
    with pytest.raises(ValueError, match=match):
        _normalize(mutation(_surface()), report_context)


def test_surface_density_must_match_the_effect_grid(report_context: ReportContext):
    density = pd.DataFrame(
        {
            "x": [20.0, 40.0, 20.0, 50.0],
            "y": [1.0, 1.0, 2.0, 2.0],
            "density": [0.2, 0.3, 0.1, 0.4],
            "hdr_mass": [0.5, 0.5, 0.9, 0.9],
        }
    )

    with pytest.raises(ValueError, match="density grid must match interaction.effect grid"):
        _normalize(replace(_surface(), density=density), report_context)


def _privacy_context() -> ReportContext:
    frame = pd.DataFrame(
        {
            "left_feature": ["A", "A", "A", "rare"],
            "right_feature": ["X", "X", "X", "Y"],
            "comparison_unit_label": ["secret-u1", "secret-u1", "secret-u2", "secret-u3"],
            "row_secret": ["r1", "r2", "r3", "r4"],
        }
    )
    return ReportContext(
        frame=frame,
        actual=np.ones(4),
        predictions={"Model A": np.ones(4)},
        weight=np.array([2.0, 3.0, 5.0, 7.0]),
        features=("left_feature", "right_feature"),
        comparison_unit_codes=np.array([11, 11, 12, 13]),
        comparison_units=3,
        minimum_cell_size=2,
        problem_type="frequency",
        deviance_power=1.0,
    )


def test_categorical_pairs_use_distinct_units_and_serialize_only_safe_aggregates():
    context = _privacy_context()
    evidence = InteractionEvidence(
        name="left:right",
        parents=("left_feature", "right_feature"),
        semantic="portfolio_aggregate",
        plot_kind="categorical_heatmap",
        effect=pd.DataFrame(
            {
                "left": ["A", "rare"],
                "right": ["X", "Y"],
                "value": [1.2, 9.9],
            }
        ),
        source="direct",
    )

    normalized = _normalize(evidence, context)

    assert normalized.effect.to_dict("records") == [{"left": "A", "right": "X", "value": 1.2}]
    assert normalized.support is not None
    assert normalized.support.to_dict("records") == [
        {
            "left": "A",
            "right": "X",
            "rows": 3,
            "comparison_units": 2,
            "weight": 10.0,
            "weight_share": pytest.approx(10.0 / 17.0),
        }
    ]
    serialized = json.dumps(
        {
            "effect": normalized.effect.to_dict("records"),
            "support": normalized.support.to_dict("records"),
        }
    )
    assert "rare" not in serialized
    assert "secret-u" not in serialized
    assert "row_secret" not in serialized


def _level_privacy_context() -> ReportContext:
    levels = [f"L{index}" for index in range(8)] + ["rare"]
    repeated = np.repeat(levels, 2).tolist()
    repeated[-1] = "rare"
    repeated[-2] = "rare"
    codes = np.arange(len(repeated))
    codes[-1] = codes[-2]
    frame = pd.DataFrame(
        {
            "age": np.tile([20.0, 40.0], len(levels)),
            "segment": repeated,
            "row_secret": [f"secret-{index}" for index in range(len(repeated))],
        }
    )
    weights_by_level = {f"L{index}": float(index + 1) for index in range(8)}
    weights_by_level["rare"] = 100.0
    weights = np.array([weights_by_level[level] for level in repeated])
    return ReportContext(
        frame=frame,
        actual=np.ones(len(frame)),
        predictions={"Model A": np.ones(len(frame))},
        weight=weights,
        features=("age", "segment"),
        comparison_unit_codes=codes,
        comparison_units=len(np.unique(codes)),
        minimum_cell_size=2,
        problem_type="frequency",
        deviance_power=1.0,
    )


@pytest.mark.parametrize(
    "plot_kind",
    ["varying_coefficient", "numeric_categorical", "factor_smooth"],
)
def test_categorical_levels_use_distinct_units_and_default_to_top_six_weights(
    plot_kind: str,
):
    context = _level_privacy_context()
    levels = [f"L{index}" for index in range(8)] + ["rare"]
    effect = pd.DataFrame({"level": levels, "value": np.ones(len(levels))})
    if plot_kind != "numeric_categorical":
        effect.insert(0, "x", np.arange(len(levels), dtype=float))
    evidence = InteractionEvidence(
        name="age:segment",
        parents=("age", "segment"),
        semantic="native_component" if plot_kind == "factor_smooth" else "partial_dependence",
        plot_kind=plot_kind,
        effect=effect,
        source="direct",
    )

    normalized = _normalize(evidence, context)

    assert normalized.support is not None
    assert normalized.support.columns.tolist() == [
        "level",
        "rows",
        "comparison_units",
        "weight",
        "weight_share",
    ]
    assert set(normalized.effect["level"]) == {f"L{index}" for index in range(8)}
    assert set(normalized.support["level"]) == {f"L{index}" for index in range(8)}
    assert normalized.default_levels == ("L7", "L6", "L5", "L4", "L3", "L2")
    assert "rare" not in json.dumps(normalized.support.to_dict("records"))


def test_caller_support_and_defaults_cannot_self_certify_privacy(
    report_context: ReportContext,
):
    evidence = _evidence_for("numeric_categorical")
    support = pd.DataFrame(
        {
            "level": ["A"],
            "rows": [1_000],
            "comparison_units": [1_000],
            "weight": [1_000.0],
            "weight_share": [1.0],
        }
    )

    with pytest.raises(ValueError, match="interaction.support must be None before normalization"):
        _normalize(replace(evidence, support=support), report_context)
    with pytest.raises(
        ValueError,
        match="interaction.default_levels must be empty before normalization",
    ):
        _normalize(replace(evidence, default_levels=("A",)), report_context)


def test_factor_smooth_diagnostics_are_validated_and_filtered_to_safe_levels():
    context = _level_privacy_context()
    evidence = InteractionEvidence(
        name="age:segment",
        parents=("age", "segment"),
        semantic="native_component",
        plot_kind="factor_smooth",
        effect=pd.DataFrame(
            {
                "x": [20.0, 20.0],
                "level": ["L0", "rare"],
                "value": [1.0, 8.0],
            }
        ),
        source="direct",
        level_diagnostics=pd.DataFrame(
            {
                "level": ["L0", "rare"],
                "effective_df": [1.5, 9.5],
                "credibility": [0.8, 0.1],
                "has_information": [True, True],
                "sufficient_support": [True, False],
                "collapsed": [False, False],
            }
        ),
    )

    normalized = _normalize(evidence, context)

    assert normalized.level_diagnostics is not None
    assert normalized.level_diagnostics.to_dict("records") == [
        {
            "level": "L0",
            "effective_df": 1.5,
            "credibility": 0.8,
            "has_information": True,
            "sufficient_support": True,
            "collapsed": False,
        }
    ]


@pytest.mark.parametrize(
    ("plot_kind", "diagnostics", "match"),
    [
        (
            "numeric_categorical",
            pd.DataFrame({"level": ["A"]}),
            "only valid for factor_smooth",
        ),
        (
            "factor_smooth",
            pd.DataFrame({"level": ["A"], "row_secret": ["private"]}),
            "has unknown columns: row_secret",
        ),
        (
            "factor_smooth",
            pd.DataFrame({"level": ["A", "A"]}),
            "must contain unique levels",
        ),
    ],
)
def test_level_diagnostics_reject_wrong_kind_unknown_columns_and_duplicate_levels(
    report_context: ReportContext,
    plot_kind: str,
    diagnostics: pd.DataFrame,
    match: str,
):
    evidence = replace(_evidence_for(plot_kind), level_diagnostics=diagnostics)

    with pytest.raises(ValueError, match=match):
        _normalize(evidence, report_context)


@pytest.mark.parametrize(
    ("semantic", "values", "match"),
    [
        ("native_component", [0.0, 1.0, 2.0], "native_component.*positive"),
        ("partial_dependence", [-0.1, 0.0, 0.1], "partial_dependence.*non-negative"),
        ("portfolio_aggregate", [-0.1, 0.0, 0.1], "portfolio_aggregate.*non-negative"),
    ],
)
def test_interaction_semantics_restrict_response_values(
    report_context: ReportContext,
    semantic: str,
    values: list[float],
    match: str,
):
    evidence = replace(
        _evidence_for("numeric_numeric"),
        semantic=semantic,
        effect=pd.DataFrame({"value": [values[1]], "lower": [values[0]], "upper": [values[2]]}),
    )

    with pytest.raises(ValueError, match=match):
        _normalize(evidence, report_context)


@pytest.mark.parametrize("semantic", ["shap_interaction", "accumulated_local_effect"])
def test_signed_interaction_semantics_accept_finite_values(
    report_context: ReportContext,
    semantic: str,
):
    evidence = replace(
        _evidence_for("numeric_numeric"),
        semantic=semantic,
        effect=pd.DataFrame({"value": [-1.0], "lower": [-2.0], "upper": [0.0]}),
    )

    normalized = _normalize(evidence, report_context)

    assert normalized.effect["value"].tolist() == [-1.0]


def _context_for_frame(
    frame: pd.DataFrame,
    *,
    codes: np.ndarray | None = None,
    minimum_cell_size: int = 2,
) -> ReportContext:
    rows = len(frame)
    unit_codes = np.arange(rows) if codes is None else codes
    return ReportContext(
        frame=frame,
        actual=np.ones(rows),
        predictions={"Model A": np.ones(rows)},
        weight=np.ones(rows),
        features=tuple(frame.columns),
        comparison_unit_codes=unit_codes,
        comparison_units=len(np.unique(unit_codes)),
        minimum_cell_size=minimum_cell_size,
        problem_type="frequency",
        deviance_power=1.0,
    )


@pytest.mark.parametrize(
    "plot_kind",
    ["varying_coefficient", "numeric_categorical", "factor_smooth"],
)
def test_level_support_always_uses_the_second_parent_role(plot_kind: str):
    context = _context_for_frame(
        pd.DataFrame(
            {
                "numeric_parent": [1, 1, 2],
                "categorical_parent": ["other", "other", "different"],
            }
        )
    )
    effect = pd.DataFrame({"level": ["1"], "value": [1.0]})
    if plot_kind != "numeric_categorical":
        effect.insert(0, "x", [1.0])
    evidence = InteractionEvidence(
        name="numeric:categorical",
        parents=("numeric_parent", "categorical_parent"),
        semantic="native_component" if plot_kind == "factor_smooth" else "partial_dependence",
        plot_kind=plot_kind,
        effect=effect,
        source="direct",
    )

    normalized = _normalize(evidence, context)

    assert normalized.effect.empty
    assert normalized.support is not None
    assert normalized.support.empty


def test_unambiguous_numeric_categorical_levels_remain_supported():
    context = _context_for_frame(
        pd.DataFrame(
            {
                "numeric_parent": [10.0, 20.0, 30.0],
                "category_code": [1, 1, 2],
            }
        )
    )
    evidence = InteractionEvidence(
        name="numeric:code",
        parents=("numeric_parent", "category_code"),
        semantic="partial_dependence",
        plot_kind="numeric_categorical",
        effect=pd.DataFrame({"level": ["1"], "value": [1.0]}),
        source="direct",
    )

    normalized = _normalize(evidence, context)

    assert normalized.effect["level"].tolist() == ["1"]
    assert normalized.support is not None
    assert normalized.support["comparison_units"].tolist() == [2]


@pytest.mark.parametrize("plot_kind", ["categorical_heatmap", "numeric_categorical"])
def test_raw_categories_that_collide_as_text_are_rejected(plot_kind: str):
    secret_collision_label = "8675309"
    if plot_kind == "categorical_heatmap":
        context = _context_for_frame(
            pd.DataFrame(
                {
                    "left_parent": [int(secret_collision_label), secret_collision_label],
                    "right_parent": ["X", "X"],
                }
            )
        )
        evidence = InteractionEvidence(
            name="left:right",
            parents=("left_parent", "right_parent"),
            semantic="portfolio_aggregate",
            plot_kind=plot_kind,
            effect=pd.DataFrame({"left": [secret_collision_label], "right": ["X"], "value": [1.0]}),
            source="direct",
        )
        collision_feature = "left_parent"
    else:
        context = _context_for_frame(
            pd.DataFrame(
                {
                    "numeric_parent": [10.0, 20.0],
                    "category_code": [int(secret_collision_label), secret_collision_label],
                }
            )
        )
        evidence = InteractionEvidence(
            name="numeric:category",
            parents=("numeric_parent", "category_code"),
            semantic="partial_dependence",
            plot_kind=plot_kind,
            effect=pd.DataFrame({"level": [secret_collision_label], "value": [1.0]}),
            source="direct",
        )
        collision_feature = "category_code"

    with pytest.raises(ValueError) as error:
        _normalize(evidence, context)

    message = str(error.value)
    assert collision_feature in message
    assert "distinct categories have an ambiguous text representation" in message
    assert secret_collision_label not in message


@pytest.mark.parametrize(
    ("plot_kind", "parents", "semantic", "match"),
    [
        ("surface", ("segment", "density"), "partial_dependence", "surface.*parents.*numeric"),
        (
            "numeric_numeric",
            ("age", "segment"),
            "partial_dependence",
            "numeric_numeric.*parents.*numeric",
        ),
        (
            "numeric_categorical",
            ("segment", "region"),
            "partial_dependence",
            "numeric_categorical.*first parent.*numeric",
        ),
        (
            "varying_coefficient",
            ("segment", "region"),
            "partial_dependence",
            "varying_coefficient.*first parent.*numeric",
        ),
        (
            "factor_smooth",
            ("age", "segment"),
            "partial_dependence",
            "factor_smooth.*native_component",
        ),
    ],
)
def test_plot_kind_requires_compatible_parent_roles_and_semantic(
    report_context: ReportContext,
    plot_kind: str,
    parents: tuple[str, str],
    semantic: str,
    match: str,
):
    evidence = replace(_evidence_for(plot_kind), parents=parents, semantic=semantic)

    with pytest.raises(ValueError, match=match):
        _normalize(evidence, report_context)


@pytest.mark.parametrize(
    ("plot_kind", "effect"),
    [
        (
            "varying_coefficient",
            pd.DataFrame({"x": [20.0, 20.0], "level": ["A", "A"], "value": [1.0, 1.1]}),
        ),
        (
            "factor_smooth",
            pd.DataFrame({"x": [20.0, 20.0], "level": ["A", "A"], "value": [1.0, 1.1]}),
        ),
        (
            "numeric_categorical",
            pd.DataFrame({"level": ["A", "A"], "value": [1.0, 1.1]}),
        ),
        ("numeric_numeric", pd.DataFrame({"value": [1.0, 1.1]})),
    ],
)
def test_curve_coordinates_must_be_unique(
    report_context: ReportContext,
    plot_kind: str,
    effect: pd.DataFrame,
):
    evidence = replace(_evidence_for(plot_kind), effect=effect)

    with pytest.raises(ValueError, match=rf"{plot_kind}.*unique coordinate|exactly one row"):
        _normalize(evidence, report_context)


class _CopyForbiddenFrame(pd.DataFrame):
    def copy(self, *args, **kwargs):
        raise AssertionError("oversized table was copied before its size was rejected")


@pytest.mark.parametrize("oversized_part", ["effect", "density"])
def test_oversized_interaction_tables_are_rejected_before_copying(
    report_context: ReportContext,
    oversized_part: str,
):
    rows = 25_601
    if oversized_part == "effect":
        evidence = replace(
            _evidence_for("numeric_numeric"),
            effect=_CopyForbiddenFrame({"value": np.ones(rows)}),
        )
    else:
        evidence = replace(
            _surface(),
            density=_CopyForbiddenFrame(
                {
                    "x": np.zeros(rows),
                    "y": np.zeros(rows),
                    "density": np.ones(rows),
                    "hdr_mass": np.ones(rows),
                }
            ),
        )

    with pytest.raises(ValueError, match="at most 25,600 rows"):
        _normalize(evidence, report_context)


def test_normalization_detaches_all_mutable_interaction_inputs(
    report_context: ReportContext,
):
    effect = _surface().effect.copy(deep=True)
    x_axis = np.array([20.0, 40.0])
    density = pd.DataFrame(
        {
            "x": effect["x"],
            "y": effect["y"],
            "density": [0.2, 0.3, 0.1, 0.4],
            "hdr_mass": [0.5, 0.5, 0.9, 0.9],
        }
    )
    evidence = replace(
        _surface(),
        effect=effect,
        grid_axes={"x": x_axis, "y": np.array([1.0, 2.0])},
        density=density,
    )

    normalized = _normalize(evidence, report_context)
    effect.loc[0, "value"] = 99.0
    x_axis[0] = 99.0
    density.loc[0, "density"] = 99.0

    assert normalized.effect.loc[0, "value"] == 0.8
    assert normalized.grid_axes["x"].tolist() == [20.0, 40.0]
    assert normalized.density is not None
    assert normalized.density.loc[0, "density"] == 0.2

    diagnostics = pd.DataFrame({"level": ["A"], "effective_df": [1.5]})
    factor = replace(_evidence_for("factor_smooth"), level_diagnostics=diagnostics)
    normalized_factor = _normalize(factor, report_context)
    diagnostics.loc[0, "effective_df"] = 99.0

    assert normalized_factor.level_diagnostics is not None
    assert normalized_factor.level_diagnostics.loc[0, "effective_df"] == 1.5
