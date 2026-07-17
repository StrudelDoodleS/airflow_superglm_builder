from __future__ import annotations

import copy
import inspect
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from superglm import (
    Adaptive,
    Categorical,
    Gaussian,
    GroupElasticNet,
    IdentityLink,
    Numeric,
    Polynomial,
    SuperGLM,
    cross_validate,
)

from pricing_pipeline.modeling import superglm_identity as identity
from pricing_pipeline.modeling.standard_superglm import run_cross_validation
from pricing_pipeline.modeling.superglm_identity import exact_superglm_cross_validate


class _TwoFoldSplitter:
    def split(self, X, y=None, groups=None):
        del X, y, groups
        yield np.arange(0, 8), np.arange(8, 12)
        yield np.arange(4, 12), np.arange(0, 4)


def _frame():
    return pd.DataFrame(
        {
            "age": np.linspace(18.0, 70.0, 12),
            "region": ["a", "b", "c"] * 4,
        }
    )


def _response():
    return np.linspace(0.5, 3.0, 12)


def _configured_model(*, lambda2=0.37, tol=2e-5, retain_fit_state=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return SuperGLM(
            family=Gaussian(),
            link=IdentityLink(),
            penalty=GroupElasticNet(
                lambda1=0.001,
                alpha=0.7,
                flavor=Adaptive(expon=1.5, eps=1e-5),
                features=["age", "region"],
            ),
            spline_penalty=lambda2,
            features={
                "age": Polynomial(degree=2),
                "region": Categorical(base="first"),
            },
            categorical_base="first",
            interactions=[("age", "region")],
            active_set=True,
            direct_solve="qr",
            discrete=True,
            n_bins={"age": 32, "region": 16},
            tol=tol,
            max_iter=321,
            convergence="coefficients",
            retain_fit_state=retain_fit_state,
        )


def _real_cv(model, *, return_estimators=True):
    return exact_superglm_cross_validate(
        model,
        _frame(),
        _response(),
        cv=_TwoFoldSplitter(),
        scoring="deviance",
        return_estimators=return_estimators,
        error_score="raise",
    )


def test_exact_wrapper_retains_the_full_constructor_configuration_on_every_real_fold():
    model = _configured_model()
    original_clone = cross_validate.__globals__["_clone_model"]

    result = _real_cv(model)

    assert len(result.estimators) == 2
    for estimator in result.estimators:
        assert type(estimator.family) is Gaussian
        assert type(estimator.link) is IdentityLink
        assert type(estimator.penalty) is GroupElasticNet
        assert estimator.penalty.lambda1 == 0.001
        assert estimator.penalty.alpha == 0.7
        assert type(estimator.penalty.flavor) is Adaptive
        assert estimator.penalty.features == frozenset({"age", "region"})
        assert estimator.lambda2 == 0.37
        assert estimator._feature_order == ["age", "region"]
        assert [type(estimator._specs[name]) for name in estimator._feature_order] == [
            Polynomial,
            Categorical,
        ]
        assert estimator._splines is None
        assert estimator._categorical_base == "first"
        assert estimator._interaction_order == ["age:region"]
        assert estimator._interaction_specs["age:region"].parent_names == ("age", "region")
        assert estimator._active_set is True
        assert estimator._direct_solve == "qr"
        assert estimator._discrete is True
        assert estimator._n_bins == {"age": 32, "region": 16}
        assert estimator._tol == 2e-5
        assert estimator._max_iter == 321
        assert estimator._convergence == "coefficients"
        assert estimator._retain_fit_state is False
    assert cross_validate.__globals__["_clone_model"] is original_clone
    identity.canonical_superglm_payload(model)
    pristine_copy, _ = identity._pristine_superglm_copy(model)
    assert pristine_copy._pending_interactions == [("age", "region")]


def test_scoped_function_copy_preserves_function_metadata_and_only_replaces_clone_global():
    upstream = identity._UPSTREAM_CROSS_VALIDATE
    original_clone = upstream.__globals__["_clone_model"]

    scoped = identity._scoped_exact_cross_validate()

    assert scoped is not upstream
    assert scoped.__code__ is upstream.__code__
    assert scoped.__defaults__ == upstream.__defaults__
    assert scoped.__kwdefaults__ == upstream.__kwdefaults__
    assert scoped.__closure__ is upstream.__closure__
    assert scoped.__name__ == upstream.__name__
    assert scoped.__qualname__ == upstream.__qualname__
    assert scoped.__module__ == upstream.__module__
    assert scoped.__doc__ == upstream.__doc__
    assert scoped.__annotations__ == upstream.__annotations__
    assert scoped.__dict__ == upstream.__dict__
    assert scoped.__globals__ is not upstream.__globals__
    assert scoped.__globals__["_clone_model"] is copy.deepcopy
    assert upstream.__globals__["_clone_model"] is original_clone


def test_exact_wrapper_preserves_auto_detect_spline_configuration_on_real_folds():
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.001,
        splines=["age"],
        n_knots=3,
        degree=2,
        categorical_base="first",
        tol=3e-5,
        max_iter=123,
    )

    result = _real_cv(model)

    for estimator in result.estimators:
        assert estimator._splines == ["age"]
        assert estimator._n_knots == 3
        assert estimator._degree == 2
        assert estimator._categorical_base == "first"
        assert estimator._feature_order == ["age", "region"]
        assert estimator._tol == 3e-5
        assert estimator._max_iter == 123


def test_parallel_exact_cv_calls_do_not_cross_talk_or_mutate_upstream_globals():
    original_clone = cross_validate.__globals__["_clone_model"]
    models = (
        _configured_model(lambda2=0.21, tol=1e-5, retain_fit_state=False),
        _configured_model(lambda2=0.83, tol=7e-5, retain_fit_state=True),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(_real_cv, models))

    observed = [
        {
            (estimator.lambda2, estimator._tol, estimator._retain_fit_state)
            for estimator in result.estimators
        }
        for result in results
    ]
    assert observed == [{(0.21, 1e-5, False)}, {(0.83, 7e-5, True)}]
    assert cross_validate.__globals__["_clone_model"] is original_clone


def test_exception_exit_detects_caller_model_mutation_and_chains_scorer_error():
    model = _configured_model()
    scorer_error = RuntimeError("scorer failed")

    def mutating_scorer(estimator, X, y, *, sample_weight, offset):
        del estimator, X, y, sample_weight, offset
        model._tol = 0.125
        raise scorer_error

    with pytest.raises(
        identity.SuperGLMIdentityError,
        match="SuperGLM changed during cross-validation",
    ) as raised:
        exact_superglm_cross_validate(
            model,
            _frame(),
            _response(),
            cv=_TwoFoldSplitter(),
            scoring=mutating_scorer,
            error_score="raise",
        )

    assert raised.value.__cause__ is scorer_error


def test_exception_exit_detects_snapshot_mutation_and_chains_upstream_error(monkeypatch):
    model = _configured_model()
    upstream_error = RuntimeError("upstream failed")

    def mutating_cross_validate(snapshot, X, y, **kwargs):
        del X, y, kwargs
        snapshot._tol = 0.125
        raise upstream_error

    monkeypatch.setattr(identity, "_scoped_exact_cross_validate", lambda: mutating_cross_validate)

    with pytest.raises(
        identity.SuperGLMIdentityError,
        match="SuperGLM snapshot changed during cross-validation",
    ) as raised:
        exact_superglm_cross_validate(model, _frame(), _response(), cv=_TwoFoldSplitter())

    assert raised.value.__cause__ is upstream_error


def test_exception_exit_preserves_original_error_when_models_are_unchanged(monkeypatch):
    model = _configured_model()
    upstream_error = RuntimeError("upstream failed")

    def failing_cross_validate(snapshot, X, y, **kwargs):
        del snapshot, X, y, kwargs
        raise upstream_error

    monkeypatch.setattr(identity, "_scoped_exact_cross_validate", lambda: failing_cross_validate)

    with pytest.raises(RuntimeError) as raised:
        exact_superglm_cross_validate(model, _frame(), _response(), cv=_TwoFoldSplitter())

    assert raised.value is upstream_error


def test_exact_wrapper_rejects_pin_and_function_structure_mismatches(monkeypatch):
    model = _configured_model()
    monkeypatch.setattr(identity.metadata, "version", lambda package: "0.12.1")
    with pytest.raises(identity.SuperGLMIdentityError, match="requires SuperGLM version"):
        _real_cv(model, return_estimators=False)

    monkeypatch.undo()
    monkeypatch.setattr(identity, "_UPSTREAM_CROSS_VALIDATE", lambda *args, **kwargs: None)
    with pytest.raises(identity.SuperGLMIdentityError, match="function structure"):
        _real_cv(model, return_estimators=False)


def test_scoped_function_rejects_positional_default_metadata_drift(monkeypatch):
    upstream = identity._UPSTREAM_CROSS_VALIDATE
    monkeypatch.setattr(upstream, "__defaults__", ("drift",))

    with pytest.raises(identity.SuperGLMIdentityError, match="function structure"):
        identity._scoped_exact_cross_validate()


def test_scoped_function_rejects_non_function_structure_without_attribute_leaks(monkeypatch):
    monkeypatch.setattr(identity, "_UPSTREAM_CROSS_VALIDATE", object())

    with pytest.raises(identity.SuperGLMIdentityError, match="function structure"):
        identity._scoped_exact_cross_validate()


def test_scoped_function_rejects_in_place_keyword_default_metadata_drift(monkeypatch):
    upstream = identity._UPSTREAM_CROSS_VALIDATE
    monkeypatch.setitem(upstream.__kwdefaults__, "return_oof", True)

    with pytest.raises(identity.SuperGLMIdentityError, match="function structure"):
        identity._scoped_exact_cross_validate()


def test_scoped_function_rejects_closure_metadata_drift(monkeypatch):
    monkeypatch.setattr(
        identity,
        "_PINNED_CROSS_VALIDATE_CLOSURE_METADATA",
        ("drift",),
        raising=False,
    )

    with pytest.raises(identity.SuperGLMIdentityError, match="function structure"):
        identity._scoped_exact_cross_validate()


def test_exact_wrapper_requires_a_pristine_copyable_exact_superglm(monkeypatch):
    fitted = SuperGLM(family="gaussian", features={"age": Numeric()}, selection_penalty=0.0)
    fitted.fit(_frame()[["age"]], _response())
    with pytest.raises(identity.SuperGLMIdentityError, match="not pristine"):
        _real_cv(fitted, return_estimators=False)

    def fail_copy(value):
        raise TypeError(f"cannot copy {type(value).__name__}")

    monkeypatch.setattr(identity, "_EXACT_DEEPCOPY", fail_copy)
    with pytest.raises(identity.SuperGLMIdentityError, match="copyable"):
        _real_cv(_configured_model(), return_estimators=False)


def test_standard_cross_validation_default_is_the_exact_wrapper():
    default = inspect.signature(run_cross_validation).parameters["cross_validate_fn"].default
    assert default is exact_superglm_cross_validate
