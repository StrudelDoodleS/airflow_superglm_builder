from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.reporting.evidence import (
    CapabilityUnavailable,
    EvidenceFact,
    EvidenceRequest,
    ExactLossEvidence,
    FeatureImportanceEvidence,
    InteractionEvidence,
    MainEffectEvidence,
    ModelEvidence,
    ReportContext,
    collect_model_evidence,
    normalize_model_evidence,
)


@pytest.fixture
def report_context() -> ReportContext:
    return ReportContext(
        frame=pd.DataFrame(
            {
                "segment": ["A", "A", "B", "B", "C", "C", "D", "D"],
                "age": np.arange(8, dtype=float),
            }
        ),
        actual=np.linspace(0.1, 0.8, 8),
        predictions={"Model A": np.linspace(0.2, 0.9, 8)},
        weight=np.ones(8),
        features=("segment", "age"),
        comparison_unit_codes=np.arange(8),
        comparison_units=8,
        minimum_cell_size=2,
        problem_type="burn_cost",
        deviance_power=1.5,
    )


def test_model_evidence_is_library_neutral(report_context: ReportContext):
    evidence = ModelEvidence(
        source="precomputed",
        facts=(EvidenceFact("Version", "2026-W33"),),
    )

    assert normalize_model_evidence("Model A", evidence, report_context).source == "precomputed"


@pytest.mark.parametrize(
    ("model_name", "evidence", "match"),
    [
        ("Unknown", ModelEvidence(source="direct"), "unknown model name"),
        (
            "Model A",
            ModelEvidence(
                source="direct",
                main_effects={
                    "unknown": MainEffectEvidence(
                        feature="unknown",
                        semantic="partial_dependence",
                        effect=pd.DataFrame({"x": [1.0], "value": [1.0]}),
                        source="direct",
                    )
                },
            ),
            "not an allowed feature",
        ),
        (
            "Model A",
            ModelEvidence(
                source="direct",
                importance=FeatureImportanceEvidence(
                    pd.DataFrame({"feature": ["age"], "magnitude": [np.inf]}),
                    method="permutation",
                    source="direct",
                ),
            ),
            "magnitude",
        ),
        (
            "Model A",
            ModelEvidence(
                source="direct",
                exact_loss=ExactLossEvidence(
                    contributions=np.ones(7),
                    size_basis="row_count",
                    comparison_group="holdout",
                    score_label="deviance",
                    source="direct",
                    family="tweedie",
                    tweedie_power=1.5,
                    dispersion=1.0,
                ),
            ),
            "contributions",
        ),
        (
            "Model A",
            ModelEvidence(
                source="direct",
                exact_loss=ExactLossEvidence(
                    contributions=np.ones(8),
                    size_basis="rows",
                    comparison_group="holdout",
                    score_label="deviance",
                    source="direct",
                    family="tweedie",
                    tweedie_power=1.5,
                    dispersion=1.0,
                ),
            ),
            "size_basis",
        ),
        (
            "Model A",
            ModelEvidence(
                source="direct",
                exact_loss=ExactLossEvidence(
                    contributions=np.ones(8),
                    size_basis="row_count",
                    comparison_group="",
                    score_label="deviance",
                    source="direct",
                    family="tweedie",
                    tweedie_power=1.5,
                    dispersion=1.0,
                ),
            ),
            "comparison_group",
        ),
    ],
)
def test_normalization_rejects_invalid_evidence(
    report_context: ReportContext,
    model_name: str,
    evidence: ModelEvidence,
    match: str,
):
    with pytest.raises((TypeError, ValueError, KeyError), match=match):
        normalize_model_evidence(model_name, evidence, report_context)


def test_normalization_rejects_html_in_adapter_warning(report_context: ReportContext):
    class HtmlWarningAdapter:
        def collect(self, *, model_name, source, context):
            return ModelEvidence(source="adapter", warnings=("<b>unsafe</b>",))

    request = EvidenceRequest("Model A", HtmlWarningAdapter(), "artifact")

    with pytest.raises(ValueError, match="warnings"):
        collect_model_evidence(report_context, {}, (request,))


def test_collection_rejects_unknown_direct_and_requested_models(report_context: ReportContext):
    with pytest.raises(KeyError, match="unknown model name"):
        collect_model_evidence(report_context, {"Unknown": ModelEvidence(source="direct")}, ())

    request = EvidenceRequest("Unknown", object(), "artifact")
    with pytest.raises(KeyError, match="unknown model name"):
        collect_model_evidence(report_context, {}, (request,))


def test_model_names_must_match_context_keys_exactly(report_context: ReportContext):
    with pytest.raises(KeyError, match="unknown model name"):
        normalize_model_evidence(" Model A ", ModelEvidence(source="direct"), report_context)
    with pytest.raises(KeyError, match="unknown model name"):
        collect_model_evidence(report_context, {" Model A ": ModelEvidence(source="direct")}, ())

    request = EvidenceRequest(" Model A ", object(), "artifact")
    with pytest.raises(KeyError, match="unknown model name"):
        collect_model_evidence(report_context, {}, (request,))


def test_normalization_rejects_a_scalar_warning_string(report_context: ReportContext):
    evidence = ModelEvidence(source="direct", warnings="not a warning tuple")

    with pytest.raises(TypeError, match="warnings"):
        normalize_model_evidence("Model A", evidence, report_context)


def test_normalization_copies_data_and_serializes_importance_columns(report_context: ReportContext):
    table = pd.DataFrame(
        {
            "feature": ["age", "segment"],
            "magnitude": [2.0, 1.0],
            "ignored": ["discard", "discard"],
        }
    )
    evidence = ModelEvidence(
        source="direct",
        importance=FeatureImportanceEvidence(table, method="permutation", source="artifact"),
    )

    result = normalize_model_evidence("Model A", evidence, report_context)
    table.loc[0, "magnitude"] = 99.0

    assert result.importance is not None
    assert result.importance.table.to_dict("list") == {
        "feature": ["age", "segment"],
        "magnitude": [2.0, 1.0],
        "share": [pytest.approx(2 / 3), pytest.approx(1 / 3)],
        "method": ["permutation", "permutation"],
        "source": ["artifact", "artifact"],
    }


def test_main_effect_requires_a_single_coordinate_and_valid_bounds(report_context: ReportContext):
    invalid_coordinate = ModelEvidence(
        source="direct",
        main_effects={
            "age": MainEffectEvidence(
                feature="age",
                semantic="partial_dependence",
                effect=pd.DataFrame({"x": [1.0], "label": ["one"], "value": [1.0]}),
                source="direct",
            )
        },
    )
    invalid_bounds = ModelEvidence(
        source="direct",
        main_effects={
            "age": MainEffectEvidence(
                feature="age",
                semantic="partial_dependence",
                effect=pd.DataFrame({"x": [1.0], "value": [1.0], "lower": [1.1], "upper": [1.2]}),
                source="direct",
            )
        },
    )

    with pytest.raises(ValueError, match="exactly one"):
        normalize_model_evidence("Model A", invalid_coordinate, report_context)
    with pytest.raises(ValueError, match="bracket"):
        normalize_model_evidence("Model A", invalid_bounds, report_context)


def test_main_effect_normalization_retains_only_declared_columns(
    report_context: ReportContext,
):
    evidence = ModelEvidence(
        source="direct",
        main_effects={
            "age": MainEffectEvidence(
                feature="age",
                semantic="partial_dependence",
                effect=pd.DataFrame(
                    {
                        "x": [20.0, 40.0],
                        "value": [0.2, 0.4],
                        "lower": [0.1, 0.3],
                        "upper": [0.3, 0.5],
                        "row_marker": ["row-1", "row-2"],
                    }
                ),
                source="direct",
            )
        },
    )

    normalized = normalize_model_evidence("Model A", evidence, report_context)

    assert normalized.main_effects["age"].effect.columns.tolist() == [
        "x",
        "value",
        "lower",
        "upper",
    ]


@pytest.mark.parametrize(
    ("categories", "label"),
    [
        ([1, "1"], "1"),
        ([np.nan, "nan"], "nan"),
    ],
)
def test_categorical_main_effect_rejects_ambiguous_display_categories(
    categories: list[object],
    label: str,
):
    context = ReportContext(
        frame=pd.DataFrame({"segment": categories, "age": [0.0, 1.0]}),
        actual=np.array([0.1, 0.2]),
        predictions={"Model A": np.array([0.2, 0.3])},
        weight=np.ones(2),
        features=("segment", "age"),
        comparison_unit_codes=np.array([0, 1]),
        comparison_units=2,
        minimum_cell_size=2,
        problem_type="burn_cost",
        deviance_power=1.5,
    )
    evidence = ModelEvidence(
        source="direct",
        main_effects={
            "segment": MainEffectEvidence(
                feature="segment",
                semantic="partial_dependence",
                effect=pd.DataFrame({"label": [label], "value": [1.0]}),
                source="direct",
            )
        },
    )

    with pytest.raises(ValueError, match="segment.*ambiguous text representation"):
        normalize_model_evidence("Model A", evidence, context)


@pytest.mark.parametrize("oversized_part", ["effect", "density"])
def test_numeric_main_effect_grids_are_bounded(
    report_context: ReportContext,
    oversized_part: str,
):
    oversized_points = 513
    effect_points = oversized_points if oversized_part == "effect" else 2
    density_points = oversized_points if oversized_part == "density" else 2
    evidence = ModelEvidence(
        source="direct",
        main_effects={
            "age": MainEffectEvidence(
                feature="age",
                semantic="partial_dependence",
                effect=pd.DataFrame(
                    {
                        "x": np.linspace(0.0, 1.0, effect_points),
                        "value": np.linspace(0.2, 0.8, effect_points),
                    }
                ),
                density=pd.DataFrame(
                    {
                        "x": np.linspace(0.0, 1.0, density_points),
                        "density": np.ones(density_points),
                    }
                ),
                source="direct",
            )
        },
    )

    with pytest.raises(ValueError, match=r"at most 512 points"):
        normalize_model_evidence("Model A", evidence, report_context)


class StaticAdapter:
    def __init__(self, evidence: ModelEvidence):
        self.evidence = evidence

    def collect(self, *, model_name, source, context):
        assert source == "artifact"
        assert model_name in context.predictions
        return self.evidence


@pytest.mark.parametrize(
    ("direct_evidence", "match"),
    [
        (ModelEvidence(source="direct", warnings="unsafe"), "warnings"),
        (ModelEvidence(source="direct", main_effects=[]), "main_effects"),
    ],
)
def test_collection_validates_each_source_before_multisource_merge(
    report_context: ReportContext,
    direct_evidence: ModelEvidence,
    match: str,
):
    request = EvidenceRequest(
        "Model A",
        StaticAdapter(ModelEvidence(source="adapter", facts=(EvidenceFact("Family", "Poisson"),))),
        "artifact",
    )

    with pytest.raises(TypeError, match=match):
        collect_model_evidence(report_context, {"Model A": direct_evidence}, (request,))


def test_collection_composes_non_conflicting_capabilities(report_context: ReportContext):
    direct = {
        "Model A": ModelEvidence(
            source="direct",
            importance=FeatureImportanceEvidence(
                pd.DataFrame({"feature": ["age"], "magnitude": [2.0]}),
                method="permutation",
                source="direct",
            ),
        )
    }
    request = EvidenceRequest(
        "Model A",
        StaticAdapter(ModelEvidence(source="adapter", facts=(EvidenceFact("Family", "Poisson"),))),
        "artifact",
    )

    result = collect_model_evidence(report_context, direct, (request,))

    assert result["Model A"].importance is not None
    assert result["Model A"].facts == (EvidenceFact("Family", "Poisson"),)


def test_collection_rejects_conflicting_capabilities(report_context: ReportContext):
    direct = {
        "Model A": ModelEvidence(
            source="direct",
            importance=FeatureImportanceEvidence(
                pd.DataFrame({"feature": ["age"], "magnitude": [2.0]}),
                method="permutation",
                source="direct",
            ),
        )
    }
    request = EvidenceRequest(
        "Model A",
        StaticAdapter(
            ModelEvidence(
                source="adapter",
                importance=FeatureImportanceEvidence(
                    pd.DataFrame({"feature": ["age"], "magnitude": [1.0]}),
                    method="native",
                    source="adapter",
                ),
            )
        ),
        "artifact",
    )

    with pytest.raises(ValueError, match="Model A.*importance"):
        collect_model_evidence(report_context, direct, (request,))


def test_normalization_validates_interaction_parents_and_unavailable_capability(
    report_context: ReportContext,
):
    invalid = ModelEvidence(
        source="direct",
        interactions={
            "bad": InteractionEvidence(
                name="bad",
                parents=("age", "unknown"),
                semantic="partial_dependence",
                plot_kind="surface",
                effect=pd.DataFrame(),
                source="direct",
            )
        },
        unavailable=(CapabilityUnavailable("importance", "not supported"),),
    )

    with pytest.raises(ValueError, match="not allowed"):
        normalize_model_evidence("Model A", invalid, report_context)


def _evidence_with_capability(capability: str) -> ModelEvidence:
    if capability == "importance":
        return ModelEvidence(
            source="direct",
            importance=FeatureImportanceEvidence(
                pd.DataFrame({"feature": ["age"], "magnitude": [1.0]}),
                method="permutation",
                source="direct",
            ),
        )
    if capability == "main_effects":
        return ModelEvidence(
            source="direct",
            main_effects={
                "age": MainEffectEvidence(
                    feature="age",
                    semantic="partial_dependence",
                    effect=pd.DataFrame({"x": [20.0, 40.0], "value": [0.2, 0.4]}),
                    source="direct",
                )
            },
        )
    if capability == "interactions":
        return ModelEvidence(
            source="direct",
            interactions={
                "age by segment": InteractionEvidence(
                    name="age by segment",
                    parents=("age", "segment"),
                    semantic="partial_dependence",
                    plot_kind="numeric_categorical",
                    effect=pd.DataFrame({"level": ["A"], "value": [0.2]}),
                    source="direct",
                )
            },
        )
    if capability == "exact_loss":
        return ModelEvidence(
            source="direct",
            exact_loss=ExactLossEvidence(
                contributions=np.ones(8),
                size_basis="row_count",
                comparison_group="tweedie:1.5",
                score_label="Exact NLL",
                source="direct",
                family="Tweedie",
                tweedie_power=1.5,
                dispersion=0.8,
            ),
        )
    raise AssertionError(f"unsupported test capability: {capability}")


@pytest.mark.parametrize(
    "capability",
    ["importance", "main_effects", "interactions", "exact_loss"],
)
def test_normalization_rejects_populated_capability_declared_unavailable(
    report_context: ReportContext,
    capability: str,
):
    evidence = replace(
        _evidence_with_capability(capability),
        unavailable=(CapabilityUnavailable(capability, "not supported"),),
    )

    with pytest.raises(ValueError, match=rf"{capability}.*unavailable"):
        normalize_model_evidence("Model A", evidence, report_context)


def test_collection_reconciles_interaction_availability_across_sources(
    report_context: ReportContext,
):
    direct = {
        "Model A": ModelEvidence(
            source="direct",
            unavailable=(
                CapabilityUnavailable("importance", "importance unavailable"),
                CapabilityUnavailable("interactions", "interactions unavailable"),
            ),
        )
    }
    request = EvidenceRequest(
        "Model A",
        StaticAdapter(_evidence_with_capability("interactions")),
        "artifact",
    )

    result = collect_model_evidence(report_context, direct, (request,))["Model A"]

    assert result.interactions
    assert result.unavailable == (CapabilityUnavailable("importance", "importance unavailable"),)


def test_empty_private_interaction_is_removed_and_declared_unavailable(
    report_context: ReportContext,
):
    evidence = ModelEvidence(
        source="direct",
        interactions={
            "age by segment": InteractionEvidence(
                name="age by segment",
                parents=("age", "segment"),
                semantic="partial_dependence",
                plot_kind="numeric_categorical",
                effect=pd.DataFrame({"level": ["missing"], "value": [0.2]}),
                source="direct",
            )
        },
    )

    normalized = normalize_model_evidence("Model A", evidence, report_context)

    assert not normalized.interactions
    assert normalized.unavailable == (
        CapabilityUnavailable(
            "interactions",
            "age by segment: no cells meet minimum support",
        ),
    )


def test_mixed_safe_and_private_interactions_keep_capability_available(
    report_context: ReportContext,
):
    evidence = ModelEvidence(
        source="direct",
        interactions={
            "safe": InteractionEvidence(
                name="safe",
                parents=("age", "segment"),
                semantic="partial_dependence",
                plot_kind="numeric_categorical",
                effect=pd.DataFrame({"level": ["A"], "value": [0.2]}),
                source="direct",
            ),
            "private": InteractionEvidence(
                name="private",
                parents=("age", "segment"),
                semantic="partial_dependence",
                plot_kind="numeric_categorical",
                effect=pd.DataFrame({"level": ["missing"], "value": [0.2]}),
                source="direct",
            ),
        },
    )

    direct = normalize_model_evidence("Model A", evidence, report_context)
    collected = collect_model_evidence(report_context, {"Model A": evidence}, ())["Model A"]

    assert tuple(direct.interactions) == ("safe",)
    assert tuple(collected.interactions) == ("safe",)
    assert direct.unavailable == ()
    assert collected.unavailable == ()
