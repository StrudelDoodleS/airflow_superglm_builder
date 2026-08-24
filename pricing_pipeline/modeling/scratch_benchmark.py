"""Disposable unconstrained and boosted-tree benchmarks for scratch notebooks.

Nothing in this module registers, publishes, deploys, or writes to SQL. The
helpers intentionally return ordinary in-memory Python objects so a technical
benchmark cannot accidentally become a governed pricing model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import KFold
from superglm import Categorical, Numeric, OrderedCategorical, Poisson, Spline, SuperGLM, Tweedie
from superglm.features.spline import _SplineBase

_MISSING_CATEGORY = "__SCRATCH_MISSING__"
_MODEL_NAMES = ("catboost", "lightgbm", "xgboost")
_EXPLORATORY_BLEND_EVIDENCE = (
    "exploratory_cross_fitted_meta_over_global_base_oof_"
    "non_nested_not_unbiased_generalization_estimate"
)
EstimatorFactory = Callable[[int], Any]


class ScratchBenchmarkError(RuntimeError):
    """Raised when a disposable benchmark cannot produce trustworthy diagnostics."""


def _reference_distribution(model: SuperGLM) -> tuple[float | None, str]:
    if not isinstance(model, SuperGLM):
        raise TypeError("reference_superglm must be a fitted SuperGLM")
    try:
        model.training_telemetry()
    except RuntimeError as exc:
        raise ScratchBenchmarkError("reference_superglm must be fitted") from exc
    if model.family == "poisson" or isinstance(model.family, Poisson):
        return None, "reference_superglm"
    if isinstance(model.family, Tweedie):
        power = float(model.family.p)
        if not np.isfinite(power) or not 1.0 < power < 2.0:
            raise ValueError("reference_superglm Tweedie power must be strictly between 1 and 2")
        return power, "reference_superglm"
    raise ValueError("reference_superglm must use the Poisson or Tweedie family")


def unconstrained_superglm_features(
    frame: pd.DataFrame,
    *,
    categorical_columns: Sequence[str] = (),
    ordered_columns: Mapping[str, Sequence[Any]] | None = None,
    linear_columns: Sequence[str] = (),
    spline_kind: str = "ps",
    k: int = 10,
    knot_strategy: str = "quantile_tempered",
    knot_alpha: float = 0.2,
) -> dict[str, Any]:
    """Build a decision-light SuperGLM feature map with no grouping or constraints.

    Columns are kept in frame order. Ordinary categoricals retain every raw
    observed level; ordered columns retain only the caller-supplied ordering;
    linear columns remain linear; every other column receives an unconstrained
    spline whose lambdas and geometry are data driven.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a non-empty pandas DataFrame")
    if not isinstance(k, int) or isinstance(k, bool) or k < 3:
        raise ValueError("k must be an integer of at least 3")
    categorical = tuple(str(column) for column in categorical_columns)
    linear = tuple(str(column) for column in linear_columns)
    ordered = {str(column): tuple(values) for column, values in (ordered_columns or {}).items()}
    configured = set(categorical) | set(linear) | set(ordered)
    if len(configured) != len(categorical) + len(linear) + len(ordered):
        raise ValueError("categorical, ordered, and linear column declarations must not overlap")
    missing = configured - set(frame.columns)
    if missing:
        raise ValueError("declared benchmark columns are missing: " + ", ".join(sorted(missing)))
    if any(not values for values in ordered.values()):
        raise ValueError("every ordered benchmark column requires at least one ordered level")

    def spline() -> Any:
        return Spline(
            kind=spline_kind,
            k=k,
            knot_strategy=knot_strategy,
            knot_alpha=knot_alpha,
        )

    features: dict[str, Any] = {}
    for column in frame.columns:
        if column in categorical:
            features[column] = Categorical()
        elif column in ordered:
            features[column] = OrderedCategorical(order=list(ordered[column]), basis=spline())
        elif column in linear:
            features[column] = Numeric()
        else:
            features[column] = spline()
    return features


def superglm_edf_table(model: SuperGLM) -> pd.DataFrame:
    """Return solver-group EDF so ordered specials stay separate from their curve."""
    if not isinstance(model, SuperGLM):
        raise TypeError("model must be a fitted SuperGLM")
    try:
        telemetry = model.training_telemetry()
    except RuntimeError as exc:
        raise ScratchBenchmarkError("model must be fitted before EDF inspection") from exc

    by_group = telemetry["edf"]["by_group"]
    rows: list[dict[str, Any]] = []
    for group in model._groups:
        group_name = str(group.name)
        feature_name = str(group.feature_name)
        spec = model._specs.get(feature_name)
        subgroup = getattr(group, "subgroup_type", None)
        if subgroup == "special":
            component_kind = "special"
        elif isinstance(spec, OrderedCategorical):
            component_kind = "ordered_smooth"
        elif isinstance(spec, _SplineBase):
            component_kind = "smooth"
        elif isinstance(spec, Categorical):
            component_kind = "categorical"
        else:
            component_kind = "linear"
        available_dimension = int(group.size)
        effective_df = float(by_group.get(group_name, 0.0))
        rows.append(
            {
                "feature_name": feature_name,
                "component_name": group_name,
                "component_kind": component_kind,
                "available_dimension": available_dimension,
                "effective_df": effective_df,
                "edf_utilisation": (
                    effective_df / available_dimension if available_dimension else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["feature_name", "component_name"],
        kind="stable",
        ignore_index=True,
    )


def _required_scratch_estimators(
    *,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    thread_count: int,
    tweedie_power: float | None,
    model_parameters: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, EstimatorFactory]:
    try:
        from catboost import CatBoostRegressor
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise ScratchBenchmarkError(
            "Boosted scratch benchmarks require the optional dependencies. Run "
            "`uv sync --extra scratch`, restart the notebook kernel, and rerun this cell."
        ) from exc

    def merged(name: str, defaults: Mapping[str, Any]) -> dict[str, Any]:
        return {**defaults, **dict(model_parameters.get(name, {}))}

    catboost_loss = (
        "Poisson" if tweedie_power is None else f"Tweedie:variance_power={tweedie_power!r}"
    )
    lightgbm_objective = "poisson" if tweedie_power is None else "tweedie"
    xgboost_objective = "count:poisson" if tweedie_power is None else "reg:tweedie"
    power_parameter = {} if tweedie_power is None else {"tweedie_variance_power": tweedie_power}
    return MappingProxyType(
        {
            "catboost": lambda seed: CatBoostRegressor(
                **merged(
                    "catboost",
                    {
                        "loss_function": catboost_loss,
                        "iterations": n_estimators,
                        "depth": max_depth,
                        "learning_rate": learning_rate,
                        "random_seed": seed,
                        "thread_count": thread_count,
                        "verbose": False,
                        "allow_writing_files": False,
                    },
                )
            ),
            "lightgbm": lambda seed: LGBMRegressor(
                **merged(
                    "lightgbm",
                    {
                        "objective": lightgbm_objective,
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "random_state": seed,
                        "n_jobs": thread_count,
                        "verbosity": -1,
                        **power_parameter,
                    },
                )
            ),
            "xgboost": lambda seed: XGBRegressor(
                **merged(
                    "xgboost",
                    {
                        "objective": xgboost_objective,
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "random_state": seed,
                        "n_jobs": thread_count,
                        "tree_method": "hist",
                        "enable_categorical": True,
                        "verbosity": 0,
                        **power_parameter,
                    },
                )
            ),
        }
    )


def _category_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(_MISSING_CATEGORY)


def _category_levels(
    frame: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    return {
        column: tuple(sorted(_category_text(frame[column]).unique().tolist()))
        for column in categorical_columns
    }


def _tree_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    categorical_levels: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError("prediction frame is missing columns: " + ", ".join(sorted(missing)))
    prepared = frame.loc[:, list(feature_columns)].copy()
    categorical = set(categorical_levels)
    for column in prepared.columns:
        if column in categorical:
            prepared[column] = pd.Categorical(
                _category_text(prepared[column]),
                categories=list(categorical_levels[column]),
            )
        elif not pd.api.types.is_numeric_dtype(prepared[column]):
            raise TypeError(
                f"non-categorical benchmark column {column!r} must have a numeric dtype"
            )
    return prepared


def _catboost_frame(
    frame: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    prepared = frame.copy()
    for column in categorical_columns:
        prepared[column] = _category_text(prepared[column]).astype(str)
    return prepared


def _fit_estimator(
    name: str,
    estimator: Any,
    tree_frame: pd.DataFrame,
    catboost_frame: pd.DataFrame,
    target: np.ndarray,
    weight: np.ndarray,
    positions: np.ndarray,
    categorical_columns: Sequence[str],
) -> Any:
    if name == "catboost":
        return estimator.fit(
            catboost_frame.iloc[positions],
            target[positions],
            sample_weight=weight[positions],
            cat_features=list(categorical_columns),
        )
    return estimator.fit(
        tree_frame.iloc[positions],
        target[positions],
        sample_weight=weight[positions],
    )


def _predict_estimator(
    name: str,
    estimator: Any,
    tree_frame: pd.DataFrame,
    catboost_frame: pd.DataFrame,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    source = catboost_frame if name == "catboost" else tree_frame
    selected = source if positions is None else source.iloc[positions]
    prediction = np.asarray(estimator.predict(selected), dtype=float)
    return np.clip(prediction, np.finfo(float).tiny, None)


def _convex_tweedie_weights(
    target: np.ndarray,
    predictions: np.ndarray,
    sample_weight: np.ndarray,
    *,
    power: float,
) -> np.ndarray:
    model_count = predictions.shape[1]
    initial = np.full(model_count, 1.0 / model_count)

    def objective(weights: np.ndarray) -> float:
        blended = np.clip(predictions @ weights, np.finfo(float).tiny, None)
        return float(
            mean_tweedie_deviance(
                target,
                blended,
                sample_weight=sample_weight,
                power=power,
            )
        )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * model_count,
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.fun):
        raise ScratchBenchmarkError(f"convex blend optimisation failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    return weights / weights.sum()


def _cross_fitted_blend_predictions(
    target: np.ndarray,
    predictions: np.ndarray,
    sample_weight: np.ndarray,
    assessment_fold: np.ndarray,
    *,
    power: float,
) -> np.ndarray:
    folds = np.unique(assessment_fold)
    if len(folds) < 2:
        raise ScratchBenchmarkError(
            "at least two complete OOF assessment folds are required to evaluate blend weights"
        )
    blended = np.full(len(target), np.nan, dtype=float)
    for fold in folds:
        assessment = assessment_fold == fold
        meta_train = ~assessment
        fold_weights = _convex_tweedie_weights(
            target[meta_train],
            predictions[meta_train],
            sample_weight[meta_train],
            power=power,
        )
        blended[assessment] = np.clip(
            predictions[assessment] @ fold_weights,
            np.finfo(float).tiny,
            None,
        )
    if not np.isfinite(blended).all():
        raise ScratchBenchmarkError("cross-fitted blend evaluation is incomplete")
    return blended


@dataclass(frozen=True)
class ScratchBoostedBlend:
    """Fitted scratch-only tree ensemble and its exploratory diagnostics.

    ``weights`` are the final deployable weights fitted on all complete OOF
    rows. ``oof_predictions['blend_rate']`` instead uses assessment-fold
    cross-fitted meta-weights over one global base-OOF matrix. This assessment
    is non-nested because its meta-training features can come from base fits
    influenced by the assessed outcomes. It is exploratory and not an unbiased
    generalization estimate.
    """

    models: Mapping[str, Any]
    weights: Mapping[str, float]
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    categorical_levels: Mapping[str, tuple[str, ...]]
    uses_exposure: bool
    distribution: str
    tweedie_power: float | None
    power_source: str
    oof_predictions: pd.DataFrame
    metrics: pd.DataFrame

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict positive rates from each fitted base learner."""
        tree = _tree_frame(
            frame,
            feature_columns=self.feature_columns,
            categorical_levels=self.categorical_levels,
        )
        catboost = _catboost_frame(tree, self.categorical_columns)
        return pd.DataFrame(
            {
                name: _predict_estimator(name, self.models[name], tree, catboost)
                for name in _MODEL_NAMES
            },
            index=frame.index,
        )

    def predict_rate(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict the convexly blended response rate."""
        components = self.predict_components(frame)
        weights = np.array([self.weights[name] for name in _MODEL_NAMES])
        return np.asarray(components.loc[:, list(_MODEL_NAMES)] @ weights, dtype=float)

    def predict_expected(
        self,
        frame: pd.DataFrame,
        *,
        exposure: Any = None,
    ) -> np.ndarray:
        """Predict the aggregate response, requiring exposure for an offset-rate fit."""
        rate = self.predict_rate(frame)
        if not self.uses_exposure:
            if exposure is not None:
                raise ValueError("exposure was supplied to a blend trained without exposure")
            return rate
        resolved = np.asarray(exposure, dtype=float)
        if resolved.ndim != 1 or len(resolved) != len(frame):
            raise ValueError("exposure must be one-dimensional and match the prediction frame")
        if not np.isfinite(resolved).all() or np.any(resolved <= 0):
            raise ValueError("exposure must contain finite positive values")
        return rate * resolved


def fit_boosted_blend(
    frame: pd.DataFrame,
    target: Any,
    *,
    categorical_columns: Sequence[str] = (),
    sample_weight: Any = None,
    exposure: Any = None,
    cv: Any = None,
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    thread_count: int = -1,
    tweedie_power: float | None = None,
    reference_superglm: SuperGLM | None = None,
    model_parameters: Mapping[str, Mapping[str, Any]] | None = None,
    estimator_factories: Mapping[str, EstimatorFactory] | None = None,
) -> ScratchBoostedBlend:
    """Fit CatBoost/LightGBM/XGBoost and learn a convex blend from OOF rates.

    When exposure is supplied, each learner fits ``target / exposure`` with
    ``sample_weight * exposure ** (2 - power)``. By Tweedie deviance
    homogeneity, this is equivalent to fitting the aggregate response with a
    log-exposure offset while keeping credibility weight distinct. It reduces
    to exposure weighting for Poisson, but not for compound Tweedie.
    ``predict_expected`` converts the fitted rate back to the response scale. Leave
    ``tweedie_power`` as ``None`` for Poisson. For a compound Tweedie benchmark,
    pass one GAM-estimated power in ``(1, 2)``; that exact value is fixed across
    all three learners and the OOF blend loss rather than tuned per learner.
    Prefer ``reference_superglm=fitted_gam``: Poisson/Tweedie and the fitted
    GAM's power are then resolved automatically, and a conflicting explicit
    power is rejected.

    The reported blend deviance cross-fits only the meta weights over one
    global base-OOF matrix. That evidence is exploratory, non-nested, and not
    an unbiased generalization estimate. Final deployable weights are fitted
    separately on all complete OOF rows.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a non-empty pandas DataFrame")
    y = np.asarray(target, dtype=float)
    if y.ndim != 1 or len(y) != len(frame):
        raise ValueError("target must be one-dimensional and match frame")
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("benchmark target must contain finite non-negative values")
    explicit_power = tweedie_power
    if explicit_power is not None:
        if isinstance(explicit_power, bool):
            raise TypeError("tweedie_power must be a float strictly between 1 and 2")
        tweedie_power = float(explicit_power)
        if not np.isfinite(tweedie_power) or not 1.0 < tweedie_power < 2.0:
            raise ValueError("tweedie_power must be strictly between 1 and 2")
    power_source = "explicit" if tweedie_power is not None else "poisson_default"
    if reference_superglm is not None:
        reference_power, power_source = _reference_distribution(reference_superglm)
        if tweedie_power is not None and tweedie_power != reference_power:
            raise ValueError(
                "tweedie_power conflicts with the fitted reference_superglm distribution"
            )
        tweedie_power = reference_power
    distribution = "poisson" if tweedie_power is None else "tweedie"
    deviance_power = 1.0 if tweedie_power is None else tweedie_power
    feature_columns = tuple(str(column) for column in frame.columns)
    categorical = tuple(str(column) for column in categorical_columns)
    if len(set(categorical)) != len(categorical):
        raise ValueError("categorical_columns must not contain duplicates")
    missing_categorical = set(categorical) - set(feature_columns)
    if missing_categorical:
        raise ValueError(
            "categorical benchmark columns are missing: " + ", ".join(sorted(missing_categorical))
        )

    base_weight = (
        np.ones(len(frame), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if base_weight.ndim != 1 or len(base_weight) != len(frame):
        raise ValueError("sample_weight must be one-dimensional and match frame")
    if not np.isfinite(base_weight).all() or np.any(base_weight <= 0):
        raise ValueError("sample_weight must contain finite positive values")
    if exposure is None:
        model_target = y
        model_weight = base_weight
        resolved_exposure = np.ones(len(frame), dtype=float)
        uses_exposure = False
    else:
        resolved_exposure = np.asarray(exposure, dtype=float)
        if resolved_exposure.ndim != 1 or len(resolved_exposure) != len(frame):
            raise ValueError("exposure must be one-dimensional and match frame")
        if not np.isfinite(resolved_exposure).all() or np.any(resolved_exposure <= 0):
            raise ValueError("exposure must contain finite positive values")
        model_target = y / resolved_exposure
        model_weight = base_weight * np.power(resolved_exposure, 2.0 - deviance_power)
        uses_exposure = True

    invalid_parameter_models = set(model_parameters or {}) - set(_MODEL_NAMES)
    if invalid_parameter_models:
        raise ValueError(
            "unknown model_parameters entries: " + ", ".join(sorted(invalid_parameter_models))
        )
    protected_parameters = {
        "catboost": {"loss_function"},
        "lightgbm": {"objective", "tweedie_variance_power"},
        "xgboost": {"objective", "tweedie_variance_power"},
    }
    protected_overrides = {
        name: sorted(set(parameters) & protected_parameters[name])
        for name, parameters in (model_parameters or {}).items()
        if set(parameters) & protected_parameters[name]
    }
    if protected_overrides:
        detail = "; ".join(
            f"{name}: {', '.join(parameters)}"
            for name, parameters in sorted(protected_overrides.items())
        )
        raise ValueError(
            "model_parameters cannot override the shared distribution contract (" + detail + ")"
        )
    factories = estimator_factories or _required_scratch_estimators(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        thread_count=thread_count,
        tweedie_power=tweedie_power,
        model_parameters=model_parameters or {},
    )
    if set(factories) != set(_MODEL_NAMES):
        raise ValueError("estimator_factories must contain catboost, lightgbm, and xgboost")

    levels = _category_levels(frame, categorical)
    tree = _tree_frame(
        frame,
        feature_columns=feature_columns,
        categorical_levels=levels,
    )
    catboost = _catboost_frame(tree, categorical)
    splitter: Iterable[tuple[np.ndarray, np.ndarray]]
    if cv is None:
        if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
            raise ValueError("n_splits must be an integer of at least 2")
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        ).split(tree, model_target)
    elif hasattr(cv, "split"):
        splitter = cv.split(tree, model_target)
    else:
        splitter = iter(cv)

    oof = np.full((len(frame), len(_MODEL_NAMES)), np.nan, dtype=float)
    validation_count = np.zeros(len(frame), dtype=int)
    assessment_fold = np.full(len(frame), -1, dtype=int)
    for fold_no, (raw_train, raw_validation) in enumerate(splitter):
        train = np.asarray(raw_train, dtype=int)
        validation = np.asarray(raw_validation, dtype=int)
        if train.ndim != 1 or validation.ndim != 1 or not len(train) or not len(validation):
            raise ValueError("each CV split requires non-empty one-dimensional positions")
        if np.intersect1d(train, validation).size:
            raise ValueError("CV train and validation positions must not overlap")
        validation_count[validation] += 1
        assessment_fold[validation] = fold_no
        for model_no, name in enumerate(_MODEL_NAMES):
            estimator = factories[name](random_state + fold_no)
            fitted = _fit_estimator(
                name,
                estimator,
                tree,
                catboost,
                model_target,
                model_weight,
                train,
                categorical,
            )
            oof[validation, model_no] = _predict_estimator(
                name,
                fitted,
                tree,
                catboost,
                validation,
            )
    if np.any(validation_count > 1):
        raise ValueError("CV validation positions must not be repeated")
    eligible = (validation_count == 1) & np.isfinite(oof).all(axis=1)
    if int(eligible.sum()) < max(20, 3 * len(_MODEL_NAMES)):
        raise ScratchBenchmarkError("too few complete OOF rows to estimate blend weights")

    eligible_folds = assessment_fold[eligible]
    blended = _cross_fitted_blend_predictions(
        model_target[eligible],
        oof[eligible],
        model_weight[eligible],
        eligible_folds,
        power=deviance_power,
    )
    weights = _convex_tweedie_weights(
        model_target[eligible],
        oof[eligible],
        model_weight[eligible],
        power=deviance_power,
    )
    fitted_models: dict[str, Any] = {}
    all_positions = np.arange(len(frame), dtype=int)
    for name in _MODEL_NAMES:
        fitted_models[name] = _fit_estimator(
            name,
            factories[name](random_state + 10_000),
            tree,
            catboost,
            model_target,
            model_weight,
            all_positions,
            categorical,
        )

    metrics = []
    for model_no, name in enumerate(_MODEL_NAMES):
        metrics.append(
            {
                "model": name,
                "mean_unit_deviance": float(
                    mean_tweedie_deviance(
                        model_target[eligible],
                        oof[eligible, model_no],
                        sample_weight=model_weight[eligible],
                        power=deviance_power,
                    )
                ),
                "distribution": distribution,
                "tweedie_power": tweedie_power,
                "power_source": power_source,
                "blend_weight": float(weights[model_no]),
                "blend_weight_scope": "final_all_complete_oof",
                "oof_rows": int(eligible.sum()),
                "evaluation": "base_oof",
            }
        )
    metrics.append(
        {
            "model": "blend",
            "mean_unit_deviance": float(
                mean_tweedie_deviance(
                    model_target[eligible],
                    blended,
                    sample_weight=model_weight[eligible],
                    power=deviance_power,
                )
            ),
            "distribution": distribution,
            "tweedie_power": tweedie_power,
            "power_source": power_source,
            "blend_weight": 1.0,
            "blend_weight_scope": "final_all_complete_oof",
            "oof_rows": int(eligible.sum()),
            "evaluation": _EXPLORATORY_BLEND_EVIDENCE,
        }
    )

    positions = np.flatnonzero(eligible)
    oof_frame = pd.DataFrame(
        {
            "row_position": positions,
            "assessment_fold": eligible_folds,
            "actual": y[eligible],
            "exposure": resolved_exposure[eligible],
            "actual_rate": model_target[eligible],
            "fit_weight": model_weight[eligible],
            **{
                f"{name}_rate": oof[eligible, model_no]
                for model_no, name in enumerate(_MODEL_NAMES)
            },
            "blend_rate": blended,
            "blend_expected": blended * resolved_exposure[eligible],
            "blend_evidence": _EXPLORATORY_BLEND_EVIDENCE,
        }
    )
    return ScratchBoostedBlend(
        models=MappingProxyType(fitted_models),
        weights=MappingProxyType(
            {name: float(weights[index]) for index, name in enumerate(_MODEL_NAMES)}
        ),
        feature_columns=feature_columns,
        categorical_columns=categorical,
        categorical_levels=MappingProxyType(levels),
        uses_exposure=uses_exposure,
        distribution=distribution,
        tweedie_power=tweedie_power,
        power_source=power_source,
        oof_predictions=oof_frame,
        metrics=pd.DataFrame(metrics),
    )


__all__ = [
    "ScratchBenchmarkError",
    "ScratchBoostedBlend",
    "fit_boosted_blend",
    "superglm_edf_table",
    "unconstrained_superglm_features",
]
