from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from superglm import (
    Adaptive,
    BSplineSmooth,
    Binomial,
    Categorical,
    CauchitLink,
    CloglogLink,
    CubicRegressionSpline,
    Gamma,
    Gaussian,
    GroupElasticNet,
    GroupLasso,
    IdentityLink,
    InverseLink,
    InverseSquaredLink,
    LevelGrouping,
    LogitLink,
    LogLink,
    NaturalSpline,
    NegativeBinomial,
    NegativeBinomialLink,
    Numeric,
    OrderedCategorical,
    Polynomial,
    Poisson,
    PowerLink,
    ProbitLink,
    PSpline,
    Ridge,
    SparseGroupLasso,
    SqrtLink,
    SuperGLM,
    Tweedie,
)
from superglm.features.constraint import Constraint
from superglm.features.spline import CardinalCRSpline
from superglm.types import LambdaPolicy

from pricing_pipeline.modeling import superglm_identity as identity


def test_runtime_identity_resolves_the_locked_superglm_pin():
    runtime = identity.resolve_superglm_runtime_identity()

    assert runtime == identity.SuperGLMRuntimeIdentity(
        version="0.12.0",
        git_sha="25c06fc84b674bb2ee777ea99567772d8d57a17c",
    )
    with pytest.raises(FrozenInstanceError):
        runtime.version = "different"


def test_runtime_identity_fails_closed_on_a_version_mismatch(monkeypatch):
    monkeypatch.setattr(identity.metadata, "version", lambda package: "0.12.1")

    with pytest.raises(identity.SuperGLMIdentityError, match="requires SuperGLM version"):
        identity.resolve_superglm_runtime_identity()


@pytest.mark.parametrize(
    "direct_url_text",
    [
        None,
        "not-json",
        "{}",
        json.dumps({"vcs_info": {"vcs": "git", "commit_id": "A" * 40}}),
        json.dumps({"vcs_info": {"vcs": "hg", "commit_id": "a" * 40}}),
    ],
)
def test_runtime_identity_rejects_unauditable_direct_url(monkeypatch, direct_url_text):
    installed = type(
        "InstalledDistribution",
        (),
        {"read_text": lambda self, filename: direct_url_text},
    )()
    monkeypatch.setattr(identity.metadata, "version", lambda package: identity.SUPERGLM_VERSION)
    monkeypatch.setattr(identity.metadata, "distribution", lambda package: installed)

    with pytest.raises(identity.SuperGLMIdentityError, match="direct_url.json"):
        identity.resolve_superglm_runtime_identity()


def test_runtime_identity_wraps_direct_url_read_failures(monkeypatch):
    class UnreadableDistribution:
        def read_text(self, filename):
            raise OSError(f"cannot read {filename}")

    monkeypatch.setattr(identity.metadata, "version", lambda package: identity.SUPERGLM_VERSION)
    monkeypatch.setattr(identity.metadata, "distribution", lambda package: UnreadableDistribution())

    with pytest.raises(identity.SuperGLMIdentityError, match="direct_url.json"):
        identity.resolve_superglm_runtime_identity()


def test_canonical_payload_rejects_model_changes_during_capture(monkeypatch):
    model = SuperGLM()
    runtime = identity.SuperGLMRuntimeIdentity(
        version=identity.SUPERGLM_VERSION,
        git_sha=identity.SUPERGLM_GIT_SHA,
    )

    def mutate_during_runtime_resolution():
        model._tol = 0.25
        return runtime

    monkeypatch.setattr(
        identity,
        "resolve_superglm_runtime_identity",
        mutate_during_runtime_resolution,
    )

    with pytest.raises(identity.SuperGLMIdentityError, match="changed during identity capture"):
        identity.canonical_superglm_payload(model)


def test_canonical_superglm_bytes_and_sha256_are_deterministic():
    model = SuperGLM(features={"age": Numeric()})

    identity.validate_pristine_superglm(model)
    first = identity.canonical_superglm_bytes(model)
    second = identity.canonical_superglm_bytes(model)

    assert first == second
    assert identity.superglm_semantic_sha256(model) == hashlib.sha256(first).hexdigest()
    assert json.loads(first) == identity.canonical_superglm_payload(model)
    assert b"0x" not in first


def test_mapping_insertion_is_stable_but_feature_plan_order_is_semantic():
    left = SuperGLM(
        splines=["age"],
        n_bins={"age": 128, "vehicle_age": 64},
    )
    right = SuperGLM(
        splines=["age"],
        n_bins={"vehicle_age": 64, "age": 128},
    )
    assert identity.canonical_superglm_bytes(left) == identity.canonical_superglm_bytes(right)

    age_first = SuperGLM(features={"age": Numeric(), "vehicle_age": Numeric()})
    vehicle_first = SuperGLM(features={"vehicle_age": Numeric(), "age": Numeric()})
    assert identity.canonical_superglm_bytes(age_first) != identity.canonical_superglm_bytes(
        vehicle_first
    )


@pytest.mark.parametrize(
    ("shortcut", "distribution", "expected_kind"),
    [
        ("poisson", Poisson(), "poisson"),
        ("gaussian", Gaussian(), "gaussian"),
        ("gamma", Gamma(), "gamma"),
        ("binomial", Binomial(), "binomial"),
    ],
)
def test_parameter_free_family_shortcuts_equal_exact_builtin_instances(
    shortcut,
    distribution,
    expected_kind,
):
    shortcut_payload = identity.canonical_superglm_payload(SuperGLM(family=shortcut))
    instance_payload = identity.canonical_superglm_payload(SuperGLM(family=distribution))

    assert shortcut_payload == instance_payload
    assert shortcut_payload["family"] == {"kind": expected_kind}


@pytest.mark.parametrize(
    ("family", "default_link", "link_instance", "expected_kind"),
    [
        ("poisson", "log", LogLink(), "log"),
        ("gaussian", "identity", IdentityLink(), "identity"),
        ("gamma", "log", LogLink(), "log"),
        ("binomial", "logit", LogitLink(), "logit"),
    ],
)
def test_default_link_equals_explicit_builtin_link(
    family,
    default_link,
    link_instance,
    expected_kind,
):
    default_payload = identity.canonical_superglm_payload(SuperGLM(family=family))
    string_payload = identity.canonical_superglm_payload(SuperGLM(family=family, link=default_link))
    instance_payload = identity.canonical_superglm_payload(
        SuperGLM(family=family, link=link_instance)
    )

    assert default_payload == string_payload == instance_payload
    assert default_payload["link"] == {"kind": expected_kind}


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        (NegativeBinomial(theta="auto"), {"kind": "negative_binomial", "theta": "auto"}),
        (NegativeBinomial(theta=2.5), {"kind": "negative_binomial", "theta": 2.5}),
        (Tweedie(p=1.5), {"kind": "tweedie", "p": 1.5}),
    ],
)
def test_parameterized_family_payload_records_fit_driving_values(family, expected):
    payload = identity.canonical_superglm_payload(SuperGLM(family=family))

    assert payload["family"] == expected


def test_parameterized_family_values_change_identity():
    assert identity.canonical_superglm_bytes(
        SuperGLM(family=NegativeBinomial(theta=1.5))
    ) != identity.canonical_superglm_bytes(SuperGLM(family=NegativeBinomial(theta=2.5)))
    assert identity.canonical_superglm_bytes(
        SuperGLM(family=Tweedie(p=1.3))
    ) != identity.canonical_superglm_bytes(SuperGLM(family=Tweedie(p=1.7)))


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        (LogLink(), {"kind": "log"}),
        (IdentityLink(), {"kind": "identity"}),
        (LogitLink(), {"kind": "logit"}),
        (ProbitLink(), {"kind": "probit"}),
        (CloglogLink(), {"kind": "cloglog"}),
        (CauchitLink(), {"kind": "cauchit"}),
        (InverseLink(), {"kind": "inverse"}),
        (InverseSquaredLink(), {"kind": "inverse_squared"}),
        (SqrtLink(), {"kind": "sqrt"}),
        (PowerLink(power=0.4), {"kind": "power", "power": 0.4}),
        (
            NegativeBinomialLink(theta=2.5),
            {"kind": "negative_binomial", "theta": 2.5},
        ),
    ],
)
def test_all_exact_builtin_links_are_recorded(link, expected):
    payload = identity.canonical_superglm_payload(SuperGLM(link=link))

    assert payload["link"] == expected


def test_parameterized_link_values_change_identity():
    assert identity.canonical_superglm_bytes(
        SuperGLM(link=PowerLink(power=0.4))
    ) != identity.canonical_superglm_bytes(SuperGLM(link=PowerLink(power=0.6)))
    assert identity.canonical_superglm_bytes(
        SuperGLM(link=NegativeBinomialLink(theta=1.0))
    ) != identity.canonical_superglm_bytes(SuperGLM(link=NegativeBinomialLink(theta=2.0)))


@pytest.mark.parametrize(
    ("penalty", "expected_kind"),
    [
        (GroupLasso(lambda1=0.2), "group_lasso"),
        (SparseGroupLasso(lambda1=0.2, alpha=0.3), "sparse_group_lasso"),
        (GroupElasticNet(lambda1=0.2, alpha=0.3), "group_elastic_net"),
        (Ridge(lambda1=0.2), "ridge"),
    ],
)
def test_all_exact_builtin_penalties_are_recorded(penalty, expected_kind):
    payload = identity.canonical_superglm_payload(SuperGLM(penalty=penalty))

    assert payload["penalty"]["kind"] == expected_kind
    assert payload["penalty"]["lambda1"] == 0.2


def test_penalty_shortcut_and_resolved_builtin_are_equivalent():
    default = identity.canonical_superglm_bytes(SuperGLM())

    assert default == identity.canonical_superglm_bytes(SuperGLM(penalty="group_lasso"))
    assert default == identity.canonical_superglm_bytes(SuperGLM(penalty=GroupLasso()))


def test_penalty_flavor_targets_and_alpha_are_semantic_but_target_order_is_not():
    adaptive = SuperGLM(
        penalty=GroupElasticNet(
            lambda1=0.2,
            alpha=0.3,
            flavor=Adaptive(expon=2.0, eps=0.01),
            features=["vehicle", "age"],
        )
    )
    reordered_targets = SuperGLM(
        penalty=GroupElasticNet(
            lambda1=0.2,
            alpha=0.3,
            flavor=Adaptive(expon=2.0, eps=0.01),
            features=["age", "vehicle"],
        )
    )
    changed_alpha = SuperGLM(
        penalty=GroupElasticNet(
            lambda1=0.2,
            alpha=0.7,
            flavor=Adaptive(expon=2.0, eps=0.01),
            features=["age", "vehicle"],
        )
    )
    changed_flavor = SuperGLM(
        penalty=GroupElasticNet(
            lambda1=0.2,
            alpha=0.3,
            flavor=Adaptive(expon=1.0, eps=0.01),
            features=["age", "vehicle"],
        )
    )

    assert identity.canonical_superglm_bytes(adaptive) == identity.canonical_superglm_bytes(
        reordered_targets
    )
    assert identity.canonical_superglm_bytes(adaptive) != identity.canonical_superglm_bytes(
        changed_alpha
    )
    assert identity.canonical_superglm_bytes(adaptive) != identity.canonical_superglm_bytes(
        changed_flavor
    )


def test_numpy_and_native_constructor_scalars_are_equivalent():
    native = SuperGLM(
        family=NegativeBinomial(theta=2.5),
        link=PowerLink(power=0.5),
        penalty=SparseGroupLasso(lambda1=0.2, alpha=0.3),
        spline_penalty=0.7,
        splines=["age"],
        n_knots=8,
        degree=3,
        active_set=True,
        discrete=True,
        n_bins=128,
        tol=1e-7,
        max_iter=80,
    )
    numpy_values = SuperGLM(
        family=NegativeBinomial(theta=np.float64(2.5)),
        link=PowerLink(power=np.float64(0.5)),
        penalty=SparseGroupLasso(
            lambda1=np.float64(0.2),
            alpha=np.float64(0.3),
        ),
        spline_penalty=np.float64(0.7),
        splines=["age"],
        n_knots=np.int64(8),
        degree=np.int64(3),
        active_set=np.bool_(True),
        discrete=np.bool_(True),
        n_bins=np.int64(128),
        tol=np.float64(1e-7),
        max_iter=np.int64(80),
    )

    assert identity.canonical_superglm_bytes(native) == identity.canonical_superglm_bytes(
        numpy_values
    )


def test_negative_and_positive_zero_are_canonical_scalar_equivalents():
    assert identity.canonical_superglm_bytes(
        SuperGLM(spline_penalty=-0.0)
    ) == identity.canonical_superglm_bytes(SuperGLM(spline_penalty=0.0))


def test_auto_detect_and_intercept_only_are_distinct_public_modes():
    intercept_only = identity.canonical_superglm_payload(SuperGLM())
    auto_detect = identity.canonical_superglm_payload(SuperGLM(splines=[]))

    assert intercept_only["features"]["mode"] == "intercept_only"
    assert auto_detect["features"]["mode"] == "auto_detect"
    assert intercept_only != auto_detect


@pytest.mark.parametrize(
    "constructor_options",
    [
        {"spline_penalty": 0.7},
        {"active_set": True},
        {"direct_solve": "qr"},
        {"discrete": True},
        {"n_bins": 64},
        {"tol": 1e-7},
        {"max_iter": 80},
        {"convergence": "coefficients"},
        {"retain_fit_state": False},
    ],
)
def test_solver_and_lambda2_changes_change_identity(constructor_options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = SuperGLM(**constructor_options)
    assert identity.canonical_superglm_bytes(model) != identity.canonical_superglm_bytes(SuperGLM())


def test_pending_constructor_interactions_are_ordered_and_recorded():
    first = SuperGLM(
        features={"a": Numeric(), "b": Numeric(), "c": Numeric()},
        interactions=[("a", "b"), ("b", "c")],
    )
    second = SuperGLM(
        features={"a": Numeric(), "b": Numeric(), "c": Numeric()},
        interactions=[("b", "c"), ("a", "b")],
    )

    assert identity.canonical_superglm_payload(first)["interactions"] == [
        ["a", "b"],
        ["b", "c"],
    ]
    assert identity.canonical_superglm_bytes(first) != identity.canonical_superglm_bytes(second)


def _grouping(*, reverse_mapping: bool = False) -> LevelGrouping:
    mapping_items = [("a", "low"), ("b", "low"), ("c", "high")]
    inverse_items = [("low", ["a", "b"]), ("high", ["c"])]
    if reverse_mapping:
        mapping_items.reverse()
        inverse_items.reverse()
    return LevelGrouping(
        original_to_group=dict(mapping_items),
        group_to_originals=dict(inverse_items),
        all_original_levels=["a", "b", "c"],
        grouped_levels=["low", "high"],
    )


@pytest.mark.parametrize(
    ("feature", "expected_kind"),
    [
        (Numeric(), "numeric"),
        (Polynomial(degree=3), "polynomial"),
        (Categorical(base="first"), "categorical"),
        (
            OrderedCategorical(order=["a", "b", "c"], basis=PSpline(n_knots=1)),
            "ordered_categorical",
        ),
        (PSpline(), "p_spline"),
        (BSplineSmooth(), "b_spline_smooth"),
        (NaturalSpline(), "natural_spline"),
        (CubicRegressionSpline(), "cubic_regression_spline"),
        (CardinalCRSpline(), "cardinal_cr_spline"),
    ],
)
def test_all_supported_main_effect_feature_families_are_recorded(feature, expected_kind):
    payload = identity.canonical_superglm_payload(SuperGLM(features={"x": feature}))

    assert payload["features"]["plan"][0]["spec"]["kind"] == expected_kind


def test_polynomial_categorical_and_full_level_grouping_configuration_is_semantic():
    baseline = SuperGLM(
        features={
            "poly": Polynomial(degree=3),
            "cat": Categorical(base="first", grouping=_grouping()),
        }
    )
    reordered_mapping = SuperGLM(
        features={
            "poly": Polynomial(degree=3),
            "cat": Categorical(base="first", grouping=_grouping(reverse_mapping=True)),
        }
    )
    changed_degree = SuperGLM(
        features={
            "poly": Polynomial(degree=2),
            "cat": Categorical(base="first", grouping=_grouping()),
        }
    )
    changed_base = SuperGLM(
        features={
            "poly": Polynomial(degree=3),
            "cat": Categorical(base="most_exposed", grouping=_grouping()),
        }
    )

    assert identity.canonical_superglm_bytes(baseline) == identity.canonical_superglm_bytes(
        reordered_mapping
    )
    assert identity.canonical_superglm_bytes(baseline) != identity.canonical_superglm_bytes(
        changed_degree
    )
    assert identity.canonical_superglm_bytes(baseline) != identity.canonical_superglm_bytes(
        changed_base
    )


def test_ordered_categorical_step_and_spline_semantics_are_recorded():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        step = SuperGLM(
            features={
                "band": OrderedCategorical(
                    order=["a", "b", "c"],
                    basis="step",
                    base="first",
                    grouping=_grouping(),
                )
            }
        )
    spline = SuperGLM(
        features={
            "band": OrderedCategorical(
                values={"a": 1.0, "b": 2.0, "c": 4.0},
                basis=PSpline(n_knots=1, penalty="none", m=1),
                base="first",
            )
        }
    )

    step_payload = identity.canonical_superglm_payload(step)["features"]["plan"][0]["spec"]
    spline_payload = identity.canonical_superglm_payload(spline)["features"]["plan"][0]["spec"]
    assert step_payload["basis"] == "step"
    assert step_payload["ordered_levels"] == ["low", "high"]
    assert spline_payload["basis"] == "spline"
    assert spline_payload["level_values"] == {"a": 1.0, "b": 2.0, "c": 4.0}
    assert step_payload != spline_payload


def test_ordered_categorical_interactions_are_rejected():
    model = SuperGLM(
        features={
            "band": OrderedCategorical(order=["a", "b", "c"], basis=PSpline(n_knots=1)),
            "age": Numeric(),
        },
        interactions=[("band", "age")],
    )

    with pytest.raises(identity.SuperGLMIdentityError, match="ordered-categorical interactions"):
        identity.canonical_superglm_payload(model)


@pytest.mark.parametrize(
    "changed_spline",
    [
        PSpline(n_knots=6),
        PSpline(degree=2),
        PSpline(knot_strategy="quantile"),
        PSpline(knot_strategy="quantile_tempered", knot_alpha=0.6),
        PSpline(penalty="none"),
        PSpline(select=True),
        PSpline(knots=[1.0, 2.0, 3.0], boundary=(0.0, 4.0)),
        PSpline(discrete=True),
        PSpline(n_bins=64),
        PSpline(extrapolation="error"),
        PSpline(constraint=Constraint.fit.increasing),
        PSpline(m=1),
        PSpline(lambda_policy=LambdaPolicy.fixed(2.0)),
        PSpline(
            m=(1, 2),
            lambda_policy={
                "m1": LambdaPolicy.fixed(1.0),
                "m2": LambdaPolicy.estimate(),
            },
        ),
    ],
)
def test_each_fit_driving_spline_configuration_changes_identity(changed_spline):
    baseline = SuperGLM(features={"x": PSpline()})
    changed = SuperGLM(features={"x": changed_spline})

    assert identity.canonical_superglm_bytes(baseline) != identity.canonical_superglm_bytes(changed)


def test_lambda_policy_mapping_insertion_order_is_not_semantic():
    first = PSpline(
        m=(1, 2),
        lambda_policy={
            "m1": LambdaPolicy.fixed(1.0),
            "m2": LambdaPolicy.estimate(),
        },
    )
    second = PSpline(
        m=(1, 2),
        lambda_policy={
            "m2": LambdaPolicy.estimate(),
            "m1": LambdaPolicy.fixed(1.0),
        },
    )

    assert identity.canonical_superglm_bytes(
        SuperGLM(features={"x": first})
    ) == identity.canonical_superglm_bytes(SuperGLM(features={"x": second}))


class _SuperGLMSubclass(SuperGLM):
    pass


class _NumericSubclass(Numeric):
    pass


class _PoissonSubclass(Poisson):
    pass


@pytest.mark.parametrize(
    "model",
    [
        _SuperGLMSubclass(),
        SuperGLM(features={"x": _NumericSubclass()}),
        SuperGLM(family=_PoissonSubclass()),
    ],
)
def test_custom_and_subclass_types_are_rejected(model):
    with pytest.raises(identity.SuperGLMIdentityError):
        identity.canonical_superglm_payload(model)


def test_fitted_and_failed_fit_models_are_rejected():
    fitted = SuperGLM(family="gaussian", features={"x": Numeric()}, selection_penalty=0.0)
    fitted.fit(pd.DataFrame({"x": [0.0, 1.0, 2.0]}), np.array([0.0, 1.0, 2.0]))

    failed = SuperGLM(features={"cat": Categorical()})
    with pytest.raises(ValueError):
        failed.fit(pd.DataFrame({"cat": ["one", "one"]}), np.array([0.0, 1.0]))

    for model in (fitted, failed):
        with pytest.raises(identity.SuperGLMIdentityError, match="not pristine"):
            identity.canonical_superglm_payload(model)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("_result", object()),
        ("_dm", object()),
        ("_distribution", Poisson()),
        ("_last_fit_meta", {"attempted": True}),
        ("_fit_X_ref", object()),
    ],
)
def test_partial_fit_and_design_matrix_caches_are_rejected(field, value):
    model = SuperGLM()
    setattr(model, field, value)

    with pytest.raises(identity.SuperGLMIdentityError, match="not pristine"):
        identity.canonical_superglm_payload(model)


def test_mutated_nested_feature_state_is_rejected():
    polynomial = Polynomial(degree=3)
    polynomial._lo = 2.0
    categorical = Categorical()
    categorical._levels = ["a", "b"]
    spline = PSpline()
    spline._knots = np.array([1.0, 2.0])

    for feature in (polynomial, categorical, spline):
        with pytest.raises(identity.SuperGLMIdentityError, match="not pristine"):
            identity.validate_pristine_superglm(SuperGLM(features={"x": feature}))


def test_shared_feature_identity_and_malformed_feature_maps_are_rejected():
    shared = Numeric()
    shared_model = SuperGLM(features={"a": shared, "b": shared})
    malformed_order = SuperGLM(features={"a": Numeric(), "b": Numeric()})
    malformed_order._feature_order.reverse()
    malformed_map = SuperGLM(features={"a": Numeric()})
    malformed_map._specs["b"] = Numeric()

    for model in (shared_model, malformed_order, malformed_map):
        with pytest.raises(identity.SuperGLMIdentityError):
            identity.canonical_superglm_payload(model)


def test_private_resolved_interactions_and_unexpected_state_are_rejected():
    resolved = SuperGLM(features={"a": Numeric(), "b": Numeric()})
    resolved._interaction_specs["a:b"] = object()
    resolved._interaction_order.append("a:b")
    custom_model_state = SuperGLM()
    custom_model_state.extra = "unexpected"
    custom_feature_state = SuperGLM(features={"a": Numeric()})
    custom_feature_state._specs["a"].extra = "unexpected"

    for model in (resolved, custom_model_state, custom_feature_state):
        with pytest.raises(identity.SuperGLMIdentityError):
            identity.canonical_superglm_payload(model)
