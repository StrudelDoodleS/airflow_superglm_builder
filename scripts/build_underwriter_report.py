"""Build a local aggregate model-review HTML report from strict TOML."""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.reporting import (
    ModelLikelihoodSpec,
    UnderwriterReportOptions,
    build_scored_model_report,
)

_ALLOWED_SECTIONS = {
    "run",
    "data",
    "columns",
    "predictions",
    "superglm_objects",
    "rating_workbooks",
    "model_likelihoods",
}
_ALLOWED_KEYS = {
    "run": {
        "output_path",
        "title",
        "problem_type",
        "tweedie_power",
        "top_k",
        "double_lift_bins",
        "curve_bins",
        "distribution_bins",
        "movement_bins",
        "relativity_points",
        "interaction_points",
        "comparison_bootstrap_replicates",
        "comparison_bootstrap_seed",
        "minimum_cell_size",
    },
    "data": {"path"},
    "columns": {"actual", "sample_weight", "features", "comparison_unit"},
}


@dataclass(frozen=True)
class ReportRunConfig:
    """Resolved local inputs for the report CLI."""

    data_path: Path
    output_path: Path
    actual: str
    sample_weight: str
    comparison_unit: str | None
    features: tuple[str, ...]
    predictions: dict[str, str]
    superglm_objects: dict[str, Path]
    rating_workbooks: dict[str, Path]
    model_likelihoods: dict[str, ModelLikelihoodSpec]
    options: UnderwriterReportOptions


def _fixed_table(payload: Any, section: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"TOML section [{section}] must be a table")
    unknown = set(payload) - _ALLOWED_KEYS[section]
    if unknown:
        raise ValueError(f"unknown [{section}] keys: {', '.join(sorted(unknown))}")
    return payload


def _string_table(payload: Any, section: str, *, required: bool = False) -> dict[str, str]:
    if payload is None:
        if required:
            raise ValueError(f"TOML section [{section}] is required")
        return {}
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"TOML section [{section}] must be a non-empty table")
    result: dict[str, str] = {}
    for raw_name, raw_value in payload.items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"[{section}] must map non-empty names to non-empty strings")
        if name in result:
            raise ValueError(f"[{section}] contains duplicate normalized model name: {name!r}")
        result[name] = raw_value.strip()
    return result


def _path(raw_value: Any, name: str, *, relative_to: Path) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _model_likelihood_table(
    payload: Any,
    *,
    prediction_names: set[str],
) -> dict[str, ModelLikelihoodSpec]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError("TOML section [model_likelihoods] must be a table")
    normalized: dict[str, dict[str, Any]] = {}
    raw_names: dict[str, str] = {}
    for raw_name, raw_spec in payload.items():
        raw_text = str(raw_name)
        name = raw_text.strip()
        if not name or not isinstance(raw_spec, dict):
            raise TypeError("each [model_likelihoods] entry must be a named table")
        if name in normalized:
            raise ValueError(
                f"[model_likelihoods] contains duplicate normalized model name: {name!r}"
            )
        normalized[name] = raw_spec
        raw_names[name] = raw_text

    for name, raw_name in raw_names.items():
        if raw_name != name:
            raise ValueError(
                f"[model_likelihoods] model name must be canonical: {raw_name!r}; use {name!r}"
            )

    unknown_models = set(normalized) - prediction_names
    if unknown_models:
        raise ValueError(
            "[model_likelihoods] contains models without predictions: "
            + ", ".join(sorted(unknown_models))
        )
    result: dict[str, ModelLikelihoodSpec] = {}
    for name, raw_spec in normalized.items():
        unknown_fields = set(raw_spec) - {"tweedie_power", "dispersion"}
        missing_fields = {"tweedie_power", "dispersion"} - set(raw_spec)
        if unknown_fields:
            raise ValueError(
                f"[model_likelihoods.{name!r}] has unknown keys: "
                + ", ".join(sorted(unknown_fields))
            )
        if missing_fields:
            raise ValueError(
                f"[model_likelihoods.{name!r}] is missing keys: "
                + ", ".join(sorted(missing_fields))
            )
        prefix = f'[model_likelihoods."{name}"]'
        result[name] = ModelLikelihoodSpec(
            tweedie_power=_toml_number(
                raw_spec["tweedie_power"],
                f"{prefix}.tweedie_power",
            ),
            dispersion=_toml_number(
                raw_spec["dispersion"],
                f"{prefix}.dispersion",
            ),
        )
    return result


def _toml_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not boolean")
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _require_prediction_names(
    section: str,
    model_names: set[str],
    prediction_names: set[str],
) -> None:
    unknown = model_names - prediction_names
    if unknown:
        raise ValueError(
            f"[{section}] contains models without predictions: " + ", ".join(sorted(unknown))
        )


def _positive_int(value: Any, name: str, default: int) -> int:
    resolved = default if value is None else value
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < 1:
        raise ValueError(f"{name} must be a positive integer")
    return resolved


def load_report_config(path: Path) -> ReportRunConfig:
    """Load and validate a report configuration without opening its data."""
    config_path = path.expanduser().resolve()
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    unknown_sections = set(payload) - _ALLOWED_SECTIONS
    if unknown_sections:
        raise ValueError("unknown TOML sections: " + ", ".join(sorted(unknown_sections)))

    run = _fixed_table(payload.get("run"), "run")
    data = _fixed_table(payload.get("data"), "data")
    columns = _fixed_table(payload.get("columns"), "columns")
    predictions = _string_table(payload.get("predictions"), "predictions", required=True)
    object_paths = _string_table(payload.get("superglm_objects"), "superglm_objects")
    workbook_paths = _string_table(payload.get("rating_workbooks"), "rating_workbooks")
    prediction_names = set(predictions)
    _require_prediction_names("superglm_objects", set(object_paths), prediction_names)
    _require_prediction_names("rating_workbooks", set(workbook_paths), prediction_names)

    for name in ("actual", "sample_weight"):
        if not isinstance(columns.get(name), str) or not columns[name].strip():
            raise ValueError(f"[columns].{name} must be a non-empty column name")
    raw_features = columns.get("features")
    if (
        not isinstance(raw_features, list)
        or not raw_features
        or not all(isinstance(value, str) and value.strip() for value in raw_features)
    ):
        raise ValueError("[columns].features must be a non-empty string array")
    features = tuple(value.strip() for value in raw_features)
    if len(set(features)) != len(features):
        raise ValueError("[columns].features must not contain duplicates")

    data_path = _path(data.get("path"), "[data].path", relative_to=config_path.parent)
    if data_path.suffix.lower() not in {".parquet", ".csv", ".feather"}:
        raise ValueError("[data].path must end in .parquet, .csv, or .feather")
    output_path = _path(
        run.get("output_path", "state/underwriter_report/model_review.html"),
        "[run].output_path",
        relative_to=ROOT,
    )

    problem_type = str(run.get("problem_type", "burn_cost"))
    raw_title = run.get("title", "Pricing model review")
    if not isinstance(raw_title, str):
        raise TypeError("[run].title must be a string")
    raw_power = run.get("tweedie_power")
    resolved_power = None if raw_power is None else _toml_number(raw_power, "[run].tweedie_power")
    options = UnderwriterReportOptions(
        title=raw_title,
        problem_type=problem_type,  # type: ignore[arg-type]
        tweedie_power=resolved_power,
        top_k=_positive_int(run.get("top_k"), "[run].top_k", 12),
        double_lift_bins=_positive_int(
            run.get("double_lift_bins"),
            "[run].double_lift_bins",
            10,
        ),
        curve_bins=_positive_int(run.get("curve_bins"), "[run].curve_bins", 100),
        distribution_bins=_positive_int(
            run.get("distribution_bins"),
            "[run].distribution_bins",
            200,
        ),
        movement_bins=_positive_int(
            run.get("movement_bins"),
            "[run].movement_bins",
            10,
        ),
        relativity_points=_positive_int(
            run.get("relativity_points"),
            "[run].relativity_points",
            200,
        ),
        interaction_points=_positive_int(
            run.get("interaction_points"),
            "[run].interaction_points",
            80,
        ),
        comparison_bootstrap_replicates=(
            200
            if run.get("comparison_bootstrap_replicates") is None
            else run["comparison_bootstrap_replicates"]
        ),
        comparison_bootstrap_seed=(
            1729
            if run.get("comparison_bootstrap_seed") is None
            else run["comparison_bootstrap_seed"]
        ),
        minimum_cell_size=_positive_int(
            run.get("minimum_cell_size"),
            "[run].minimum_cell_size",
            20,
        ),
    )
    raw_comparison_unit = columns.get("comparison_unit")
    if raw_comparison_unit is not None and (
        not isinstance(raw_comparison_unit, str) or not raw_comparison_unit.strip()
    ):
        raise ValueError("[columns].comparison_unit must be a non-empty column name")
    model_likelihoods = _model_likelihood_table(
        payload.get("model_likelihoods"),
        prediction_names=prediction_names,
    )
    return ReportRunConfig(
        data_path=data_path,
        output_path=output_path,
        actual=str(columns["actual"]),
        sample_weight=str(columns["sample_weight"]),
        comparison_unit=(None if raw_comparison_unit is None else raw_comparison_unit.strip()),
        features=features,
        predictions=predictions,
        superglm_objects={
            name: _path(value, f"[superglm_objects].{name}", relative_to=config_path.parent)
            for name, value in object_paths.items()
        },
        rating_workbooks={
            name: _path(value, f"[rating_workbooks].{name}", relative_to=config_path.parent)
            for name, value in workbook_paths.items()
        },
        model_likelihoods=model_likelihoods,
        options=options,
    )


def _read_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"configured input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix == ".feather":
        return pd.read_feather(path, columns=columns)
    frame = pd.read_csv(path, usecols=columns)
    return frame.loc[:, columns]


def build_report_from_config(
    config: ReportRunConfig,
    *,
    allow_trusted_model_load: bool,
):
    """Load configured local artifacts and build the aggregate report."""
    if config.superglm_objects and not allow_trusted_model_load:
        raise RuntimeError(
            "loading model objects executes pickle/joblib deserialization; rerun with "
            "--allow-trusted-model-load only for artifacts you trust"
        )
    required_columns = list(
        dict.fromkeys(
            [
                config.actual,
                config.sample_weight,
                *([config.comparison_unit] if config.comparison_unit is not None else []),
                *config.features,
                *config.predictions.values(),
            ]
        )
    )
    frame = _read_frame(config.data_path, required_columns)
    if config.superglm_objects:
        import joblib

        models = {name: joblib.load(path) for name, path in config.superglm_objects.items()}
    else:
        models = {}

    if models or config.rating_workbooks or config.model_likelihoods:
        from pricing_pipeline.reporting import build_underwriter_report

        return build_underwriter_report(
            frame,
            actual=config.actual,
            predictions=config.predictions,
            sample_weight=config.sample_weight,
            features=config.features,
            output_path=config.output_path,
            superglm_models=models,
            rating_workbooks=config.rating_workbooks,
            model_likelihoods=config.model_likelihoods,
            comparison_unit=config.comparison_unit,
            options=config.options,
        )

    return build_scored_model_report(
        frame,
        actual=config.actual,
        predictions=config.predictions,
        sample_weight=config.sample_weight,
        features=config.features,
        output_path=config.output_path,
        comparison_unit=config.comparison_unit,
        options=config.options,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained aggregate underwriting model report."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the strict TOML file")
    parser.add_argument(
        "--allow-local-input",
        action="store_true",
        help="Confirm that the configured local scoring data may be read",
    )
    parser.add_argument(
        "--allow-trusted-model-load",
        action="store_true",
        help="Allow joblib loading of configured, trusted SuperGLM objects",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_local_input:
        raise SystemExit("Refusing to read local input without --allow-local-input")
    config = load_report_config(args.config)
    result = build_report_from_config(
        config,
        allow_trusted_model_load=args.allow_trusted_model_load,
    )
    print(f"Report: {result.output_path}")
    print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
