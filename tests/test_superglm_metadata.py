from __future__ import annotations

import math
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest
from superglm import (
    Categorical,
    LevelGrouping,
    NegativeBinomial,
    Numeric,
    OrderedCategorical,
    Polynomial,
    Spline,
    SuperGLM,
    Tweedie,
)
from superglm.features.spline import PSpline
from superglm.types import LambdaPolicy

from pricing_pipeline.publishing.superglm_metadata import (
    _grouping_metadata,
    _json_value,
    build_superglm_publication_receipt,
)
from pricing_pipeline.publishing.superglm_publication_receipt import OffsetExportContract


def _fit_model(features, *, family="poisson", offset=None, interactions=None):
    n = 90
    rng = np.random.default_rng(20260619)
    x = pd.DataFrame(
        {
            "cat": np.array(["A", "B", "C"])[np.arange(n) % 3],
            "ord": np.array(["low", "medium", "high"])[np.arange(n) % 3],
            "age": np.linspace(18.0, 90.0, n),
            "poly": np.linspace(0.0, 10.0, n),
            "num": rng.normal(size=n),
            "a/b": np.linspace(1.0, 3.0, n),
            "a b": rng.normal(loc=1.0, scale=0.25, size=n),
        }
    )
    y = rng.poisson(np.exp(-2.0 + 0.01 * x["age"].to_numpy()))
    model = SuperGLM(
        family=family,
        features=features,
        interactions=interactions,
        selection_penalty=0.0,
        discrete=True,
        n_bins=32,
        retain_fit_state=False,
    )
    return model.fit(x, y, sample_weight=np.ones(n), offset=offset)


def test_extracts_categorical_ordered_spline_polynomial_and_numeric_metadata():
    with pytest.warns(UserWarning, match="n_knots=5 clamped"):
        model = _fit_model(
            {
                "cat": Categorical(base="most_exposed"),
                "ord": OrderedCategorical(
                    order=["low", "medium", "high"],
                    basis="spline",
                    n_knots=5,
                ),
                "age": Spline(
                    kind="ps",
                    n_knots=4,
                    knot_strategy="quantile",
                    discrete=True,
                    n_bins=16,
                ),
                "poly": Polynomial(degree=2),
                "num": Numeric(),
            }
        )

    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    assert receipt.schema_name == "superglm_publication_receipt"
    assert receipt.schema_version == 1
    assert receipt.metadata_origin == "SUPERGLM_FITTED_MODEL"
    assert receipt.package_metadata["model"]["family"] == "poisson"
    assert receipt.package_metadata["model"]["family_params"] == {}
    assert receipt.package_metadata["model"]["fit_used_offset"] is False

    cat = receipt.term_metadata["cat"]
    assert cat["feature_kind"] == "categorical"
    assert cat["declared"]["base"] == "most_exposed"
    assert sorted(cat["fitted"]["levels"]) == ["A", "B", "C"]
    assert cat["fitted"]["base_level"] in {"A", "B", "C"}

    ordered = receipt.term_metadata["ord"]
    assert ordered["feature_kind"] == "ordered_categorical"
    assert list(ordered["declared"]["ordered_levels"]) == ["low", "medium", "high"]
    assert ordered["declared"]["n_knots_requested"] == 5
    assert ordered["effective"]["n_knots_effective"] == 2
    assert ordered["spline"]["fitted"]["class_name"] == "PSpline"

    spline = receipt.term_metadata["age"]
    assert spline["feature_kind"] == "spline"
    assert spline["declared"]["kind"] == "ps"
    assert spline["declared"]["spline_degree"] == 3
    assert "degree" not in spline["declared"]
    assert spline["declared"]["knot_strategy"] == "quantile"
    assert "knot_alpha" not in spline["declared"]
    assert "discrete" not in spline["declared"]
    assert "n_bins" not in spline["declared"]
    assert spline["declared"]["constraint_kind"] is None
    assert spline["declared"]["constraint_mode"] is None
    assert "degree" not in spline["effective"]
    assert "discrete" not in spline["effective"]
    assert "n_bins" not in spline["effective"]
    assert list(spline["fitted"]["boundary"]) == [18.0, 90.0]
    assert spline["fitted"]["raw_basis_count"] > 0

    poly = receipt.term_metadata["poly"]
    assert poly["feature_kind"] == "polynomial"
    assert poly["declared"]["degree"] == 2
    assert poly["fitted"]["lower_bound"] == 0.0
    assert poly["fitted"]["upper_bound"] == 10.0

    numeric = receipt.term_metadata["num"]
    assert numeric["feature_kind"] == "numeric"
    assert numeric["declared"] == {}
    assert numeric["effective"]["encoding"] == "identity"


def test_extracts_ordered_categorical_interaction_metadata_from_fitted_model():
    model = _fit_model(
        {
            "cat": Categorical(base="most_exposed"),
            "ord": Categorical(base="most_exposed"),
        },
        interactions=[("cat", "ord")],
    )

    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    interaction = receipt.term_metadata["cat_ord"]
    assert interaction["feature_kind"] == "categorical_interaction"
    assert interaction["source_term_name"] == "cat:ord"
    assert interaction["published_term_name"] == "cat_ord"
    assert list(interaction["parent_names"]) == ["cat", "ord"]
    assert list(interaction["input_column_names"]) == ["cat", "ord"]
    assert interaction["interaction_order"] == 2
    assert interaction["effective"]["encoding"] == "categorical_cross_product"


def test_rejects_unsupported_interaction_metadata_before_publication():
    model = _fit_model(
        {"age": Numeric(), "num": Numeric()},
        interactions=[("age", "num")],
    )

    with pytest.raises(ValueError, match="age:num.*categorical"):
        build_superglm_publication_receipt(
            model,
            offset_contract=OffsetExportContract(handling="NONE"),
        )


def test_receipt_records_actual_fit_and_export_weight_usage():
    model = _fit_model({"age": Numeric()})

    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
        fit_sample_weight_name="Exposure",
        export_weight_name="PortfolioExposure",
    )

    model_metadata = receipt.package_metadata["model"]
    assert model_metadata["fit_sample_weight_used"] is True
    assert model_metadata["fit_sample_weight_name"] == "Exposure"
    assert model_metadata["export_weight_used"] is True
    assert model_metadata["export_weight_name"] == "PortfolioExposure"


def test_extracts_tweedie_family_metadata():
    model = _fit_model({"age": Numeric()}, family=Tweedie(p=1.5))

    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    assert receipt.package_metadata["model"]["family"] == "tweedie"
    assert receipt.package_metadata["model"]["family_params"] == {"p": 1.5}


def test_extracts_distribution_family_params_generically():
    model = _fit_model({"age": Numeric()}, family=NegativeBinomial(theta=2.5))

    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    assert receipt.package_metadata["model"]["family"] == "negative_binomial"
    assert receipt.package_metadata["model"]["family_params"] == {"theta": 2.5}


def test_spline_factory_and_direct_pspline_normalize_to_same_kind():
    model = _fit_model(
        {
            "age": Spline(kind="ps", n_knots=4),
            "poly": PSpline(n_knots=4),
        }
    )

    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    assert receipt.term_metadata["age"]["effective"]["kind"] == "ps"
    assert receipt.term_metadata["poly"]["effective"]["kind"] == "ps"
    assert receipt.term_metadata["age"]["fitted"]["class_name"] == "PSpline"
    assert receipt.term_metadata["poly"]["fitted"]["class_name"] == "PSpline"


@pytest.mark.parametrize("kind", ["ps", "bs", "ns", "cr", "cr_cardinal"])
def test_extracts_every_spline_kind_supported_by_the_pinned_superglm(kind):
    model = _fit_model({"age": Spline(kind=kind, n_knots=4)})

    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    spline = receipt.term_metadata["age"]
    assert spline["declared"]["kind"] == kind
    assert spline["effective"]["kind"] == kind
    assert spline["fitted"]["boundary"] == (18.0, 90.0)
    assert spline["fitted"]["coefficient_width"] > 0


def test_spline_metadata_includes_knot_alpha_only_when_strategy_uses_it():
    model = _fit_model(
        {
            "age": Spline(
                kind="ps",
                n_knots=4,
                knot_strategy="quantile_tempered",
                knot_alpha=0.7,
            )
        }
    )

    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    spline = receipt.term_metadata["age"]
    assert spline["declared"]["knot_strategy"] == "quantile_tempered"
    assert spline["declared"]["knot_alpha"] == 0.7


def test_ordered_categorical_step_has_no_nested_spline():
    model = _fit_model(
        {
            "ord": OrderedCategorical(
                order=["low", "medium", "high"],
                basis="step",
                base="first",
            )
        }
    )

    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    metadata = receipt.term_metadata["ord"]
    assert metadata["feature_kind"] == "ordered_categorical"
    assert metadata["declared"]["basis"] == "step"
    assert "spline" not in metadata


def test_ordered_categorical_records_an_explicit_spline_basis_configuration():
    model = _fit_model(
        {
            "ord": OrderedCategorical(
                order=["low", "medium", "high"],
                basis=Spline(kind="bs", n_knots=2, degree=2, penalty="none"),
            )
        }
    )

    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    declared = receipt.term_metadata["ord"]["declared"]
    assert declared["kind"] == "bs"
    assert declared["n_knots_requested"] == 2
    assert declared["degree"] == 2
    assert declared["penalty"] == "none"


def test_name_collision_detection_rejects_ambiguous_published_names():
    model = _fit_model({"a/b": Numeric(), "a b": Numeric()})

    with pytest.raises(ValueError, match="canonical term name collision"):
        build_superglm_publication_receipt(
            model,
            offset_contract=OffsetExportContract(handling="NONE"),
        )


def test_receipt_builder_does_not_expose_unstageable_term_renaming():
    model = _fit_model({"age": Numeric()})

    with pytest.raises(TypeError, match="source_to_published_names"):
        build_superglm_publication_receipt(
            model,
            offset_contract=OffsetExportContract(handling="NONE"),
            source_to_published_names={"age": "renamed_age"},
        )


def test_offset_contract_is_preserved_when_fit_used_offset():
    n = 90
    offset = np.log(np.full(n, 0.75))
    model = _fit_model({"num": Numeric()}, offset=offset)
    contract = OffsetExportContract(
        handling="EXPORTED_FACTOR",
        source_factor_name="Term Months",
        published_factor_name="Term_Months",
        source_name="TermMonths",
        label="log(TermMonths / 12)",
    )

    receipt = build_superglm_publication_receipt(model, offset_contract=contract)

    assert receipt.package_metadata["model"]["fit_used_offset"] is True
    assert receipt.offset_contract == contract
    assert receipt.model_dump(mode="json")["offset_contract"] == contract.model_dump(mode="json")
    offset_metadata = receipt.term_metadata["Term_Months"]
    assert offset_metadata["feature_kind"] == "offset"
    assert offset_metadata["offset_handling"] == "EXPORTED_FACTOR"
    assert offset_metadata["fixed_log_coefficient"] == 1.0
    assert offset_metadata["coefficient_source"] == "offset"
    assert offset_metadata["offset_factor_name"] == "Term_Months"
    assert offset_metadata["offset_source_name"] == "TermMonths"
    assert offset_metadata["offset_label"] == "log(TermMonths / 12)"


def test_pre_applied_sql_offset_is_recorded_without_a_rating_factor_term():
    offset = np.log(np.full(90, 0.75))
    model = _fit_model({"num": Numeric()}, offset=offset)
    contract = OffsetExportContract(
        handling="ALREADY_APPLIED_SQL_EXPOSURE",
        source_name="Exposure",
        label="log(Exposure)",
    )

    receipt = build_superglm_publication_receipt(model, offset_contract=contract)

    assert receipt.offset_contract == contract
    assert all(metadata["feature_kind"] != "offset" for metadata in receipt.term_metadata.values())


def test_receipt_uses_installed_superglm_package_version():
    model = _fit_model({"num": Numeric()})

    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    assert receipt.superglm_version == version("superglm")


def test_grouping_metadata_preserves_level_grouping_shape():
    grouping = LevelGrouping(
        original_to_group={"A": "small", "B": "small", "C": "large"},
        group_to_originals={"small": ["A", "B"], "large": ["C"]},
        all_original_levels=["A", "B", "C"],
        grouped_levels=["small", "large"],
    )

    metadata = _grouping_metadata(grouping)

    assert metadata == {
        "class_name": "LevelGrouping",
        "original_to_group": {"A": "small", "B": "small", "C": "large"},
        "group_to_originals": {"small": ["A", "B"], "large": ["C"]},
        "all_original_levels": ["A", "B", "C"],
        "grouped_levels": ["small", "large"],
    }


def test_offset_fitted_model_rejects_none_offset_contract():
    n = 90
    offset = np.log(np.full(n, 0.75))
    model = _fit_model({"num": Numeric()}, offset=offset)

    with pytest.raises(ValueError, match="offset contract"):
        build_superglm_publication_receipt(
            model,
            offset_contract=OffsetExportContract(handling="NONE"),
        )


def test_non_offset_model_rejects_exported_offset_contract():
    model = _fit_model({"num": Numeric()})
    contract = OffsetExportContract(
        handling="EXPORTED_FACTOR",
        source_factor_name="Term Months",
        published_factor_name="Term_Months",
        source_name="TermMonths",
        label="log(TermMonths / 12)",
    )

    with pytest.raises(ValueError, match="offset contract"):
        build_superglm_publication_receipt(model, offset_contract=contract)


def test_json_value_rejects_non_finite_floats():
    with pytest.raises(ValueError, match="non-finite"):
        _json_value({"value": np.float64(math.nan)})


def test_json_value_rejects_unknown_objects():
    with pytest.raises(ValueError, match="unsupported"):
        _json_value(object())


def test_json_value_rejects_non_string_mapping_keys():
    with pytest.raises(ValueError, match="keys must be strings"):
        _json_value({object(): "x"})


def test_model_without_feature_specs_is_rejected():
    model = _fit_model({"age": Numeric()})
    model._specs = {}
    model._feature_order = []

    with pytest.raises(ValueError, match="no feature specs"):
        build_superglm_publication_receipt(
            model,
            offset_contract=OffsetExportContract(handling="NONE"),
        )


def test_unfitted_superglm_is_rejected_before_metadata_is_published():
    model = SuperGLM(features={"age": Numeric()})

    with pytest.raises(ValueError, match="must be fitted"):
        build_superglm_publication_receipt(
            model,
            offset_contract=OffsetExportContract(handling="NONE"),
        )


def test_tampered_feature_order_is_rejected_instead_of_repaired():
    model = _fit_model({"age": Numeric(), "num": Numeric()})
    model._feature_order = ["age"]

    with pytest.raises(ValueError, match="feature order"):
        build_superglm_publication_receipt(
            model,
            offset_contract=OffsetExportContract(handling="NONE"),
        )


def test_malformed_grouping_is_rejected_instead_of_published_as_a_class_name():
    model = _fit_model({"cat": Categorical()})
    model._specs["cat"]._grouping = object()

    with pytest.raises(ValueError, match="LevelGrouping"):
        build_superglm_publication_receipt(
            model,
            offset_contract=OffsetExportContract(handling="NONE"),
        )


def test_spline_lambda_policy_is_serialized_from_the_pinned_superglm_type():
    model = _fit_model({"age": Spline(kind="ps", n_knots=4)})
    model._specs["age"]._lambda_policy = LambdaPolicy.fixed(2.5)

    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    assert receipt.term_metadata["age"]["declared"]["lambda_policy"] == {
        "mode": "fixed",
        "value": 2.5,
    }


def test_receipt_records_the_resolved_fitted_link_name():
    model = _fit_model({"age": Numeric()})

    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    assert receipt.package_metadata["model"]["link"] == "log"


def test_unsupported_feature_metadata_is_rejected_clearly():
    model = _fit_model({"age": Numeric()})
    model._specs["age"] = object()

    with pytest.raises(ValueError, match="unsupported feature.*object"):
        build_superglm_publication_receipt(
            model,
            offset_contract=OffsetExportContract(handling="NONE"),
        )


def test_json_value_rejects_item_method_compatibility_objects():
    class ItemLike:
        def item(self):
            return 1

    with pytest.raises(ValueError, match="unsupported"):
        _json_value(ItemLike())
