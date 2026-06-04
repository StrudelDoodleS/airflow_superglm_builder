from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_models.registry import get_model_spec  # noqa: E402
from pricing_pipeline.infra.config import Settings  # noqa: E402
from pricing_pipeline.infra.db import get_engine  # noqa: E402
from pricing_pipeline.models.spec import coerce_training_frame  # noqa: E402


PREDICT_SQL = text(
    """
    EXEC pricing.PREDICT_CURRENT_RATE
        @model_key = :model_key,
        @deployment_slot = :deployment_slot,
        @features_json = :features_json,
        @exposure = :exposure
    """
)


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_feature_payload(row: Mapping[str, Any], feature_columns: Sequence[str]) -> str:
    payload = {
        column: _json_value(row[column])
        for column in feature_columns
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def predict_with_superglm(fitted_model, training_frame) -> np.ndarray:
    return np.asarray(
        fitted_model.predict(training_frame.X, offset=training_frame.offset),
        dtype=float,
    )


def fetch_sql_predictions(
    engine,
    *,
    model_key: str,
    deployment_slot: str,
    X: pd.DataFrame,
    exposure: np.ndarray,
    feature_columns: Sequence[str],
) -> np.ndarray:
    predictions: list[float] = []
    with engine.begin() as con:
        for (_, row), exposure_value in zip(X.iterrows(), exposure, strict=True):
            result = con.execute(
                PREDICT_SQL,
                {
                    "model_key": model_key,
                    "deployment_slot": deployment_slot,
                    "features_json": build_feature_payload(row, feature_columns),
                    "exposure": float(exposure_value),
                },
            ).mappings().first()
            if result is None:
                raise RuntimeError("pricing.PREDICT_CURRENT_RATE returned no rows")
            predictions.append(float(result["prediction"]))
    return np.asarray(predictions, dtype=float)


def prediction_error_summary(expected: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    abs_error = np.abs(actual - expected)
    rel_error = np.zeros_like(abs_error)
    nonzero_expected = np.abs(expected) > 0
    rel_error[nonzero_expected] = (
        abs_error[nonzero_expected] / np.abs(expected[nonzero_expected])
    )
    return {
        "rows_checked": int(expected.size),
        "max_abs_error": float(abs_error.max(initial=0.0)),
        "mean_abs_error": float(abs_error.mean() if expected.size else 0.0),
        "max_rel_error": float(rel_error.max(initial=0.0)),
        "mean_rel_error": float(rel_error.mean() if expected.size else 0.0),
    }


def assert_predictions_close(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> None:
    close = np.isclose(actual, expected, rtol=rtol, atol=atol)
    if bool(np.all(close)):
        return
    summary = prediction_error_summary(expected, actual)
    raise AssertionError(
        "SQL prediction does not match SuperGLM prediction: "
        f"rows_outside_tolerance={int((~close).sum())} "
        f"max_abs_error={summary['max_abs_error']} "
        f"max_rel_error={summary['max_rel_error']} "
        f"rtol={rtol} atol={atol}"
    )


def find_latest_model_artifact(rating_export_root: Path, model_key: str) -> Path:
    candidates = list((rating_export_root / model_key).glob("**/superglm_model.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No superglm_model.pkl found under {rating_export_root / model_key}"
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def load_training_frame(engine, spec, *, limit: int | None):
    raw = pd.read_sql_query(text(spec.training_sql), engine)
    if limit is not None:
        raw = raw.head(limit)
    return coerce_training_frame(spec.build_training_frame(raw))


def load_fitted_model(model_path: Path):
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def validate_sql_prediction(
    engine,
    *,
    model_key: str,
    deployment_slot: str,
    model_path: Path,
    limit: int | None,
    rtol: float,
    atol: float,
) -> dict[str, float | int]:
    spec = get_model_spec(model_key)
    training_frame = load_training_frame(engine, spec, limit=limit)
    fitted_model = load_fitted_model(model_path)
    expected = predict_with_superglm(fitted_model, training_frame)
    actual = fetch_sql_predictions(
        engine,
        model_key=model_key,
        deployment_slot=deployment_slot,
        X=training_frame.X,
        exposure=training_frame.exposure,
        feature_columns=spec.feature_columns,
    )
    assert_predictions_close(expected, actual, rtol=rtol, atol=atol)
    return prediction_error_summary(expected, actual)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate SQL Server rating-table prediction against SuperGLM.predict "
            "for the same rows and offset."
        )
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--deployment-slot", required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-8)
    return parser


def main() -> int:
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    args = build_parser().parse_args()
    settings = Settings.from_env(os.environ)
    model_path = args.model_path or find_latest_model_artifact(
        settings.rating_export_root,
        args.model_key,
    )
    summary = validate_sql_prediction(
        get_engine(settings),
        model_key=args.model_key,
        deployment_slot=args.deployment_slot,
        model_path=model_path,
        limit=args.limit,
        rtol=args.rtol,
        atol=args.atol,
    )
    print("validation=ok")
    print(f"model_path={model_path}")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
