from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import validate_sql_prediction_against_superglm as validator


def test_build_feature_payload_uses_model_feature_columns_only():
    row = pd.Series(
        {
            "VehAge": 4,
            "Area": "C",
            "ClaimNb": 1,
            "Exposure": 0.75,
        }
    )

    payload = validator.build_feature_payload(row, ("VehAge", "Area"))

    assert payload == '{"Area":"C","VehAge":4}'


def test_prediction_error_summary_reports_absolute_and_relative_errors():
    expected = np.array([1.0, 2.0, 0.0])
    actual = np.array([1.1, 1.8, 0.05])

    summary = validator.prediction_error_summary(expected, actual)

    assert summary == {
        "rows_checked": 3,
        "max_abs_error": pytest.approx(0.2),
        "mean_abs_error": pytest.approx((0.1 + 0.2 + 0.05) / 3),
        "max_rel_error": pytest.approx(0.1),
        "mean_rel_error": pytest.approx((0.1 + 0.1 + 0.0) / 3),
    }


def test_assert_predictions_close_reports_rows_outside_tolerance():
    expected = np.array([1.0, 2.0])
    actual = np.array([1.0, 2.5])

    with pytest.raises(AssertionError) as exc:
        validator.assert_predictions_close(expected, actual, rtol=1e-6, atol=1e-8)

    assert "rows_outside_tolerance=1" in str(exc.value)
    assert "max_abs_error=0.5" in str(exc.value)


class FakeResult:
    def __init__(self, prediction: float):
        self.prediction = prediction

    def mappings(self):
        return self

    def first(self):
        return {"prediction": self.prediction}


class FakeConnection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params):
        self.calls.append((str(statement), params))
        return FakeResult(3.25 + len(self.calls))


class FakeBegin:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeEngine:
    def __init__(self):
        self.connection = FakeConnection()

    def begin(self):
        return FakeBegin(self.connection)


def test_fetch_sql_predictions_calls_prediction_proc_for_each_row():
    engine = FakeEngine()
    X = pd.DataFrame(
        {
            "VehAge": [4, 7],
            "Area": ["C", "A"],
            "ClaimNb": [99, 99],
        }
    )
    exposure = np.array([0.75, 1.25])

    predictions = validator.fetch_sql_predictions(
        engine,
        model_name="MTPL_FREQ",
        deployment_slot="MTPL_FREQ_UAT",
        X=X,
        exposure=exposure,
        feature_columns=("VehAge", "Area"),
    )

    np.testing.assert_allclose(predictions, np.array([4.25, 5.25]))
    assert len(engine.connection.calls) == 2
    first_sql, first_params = engine.connection.calls[0]
    assert "EXEC pricing.PREDICT_CURRENT_RATE" in first_sql
    assert first_params == {
        "model_name": "MTPL_FREQ",
        "deployment_slot": "MTPL_FREQ_UAT",
        "features_json": '{"Area":"C","VehAge":4}',
        "exposure": 0.75,
    }


def test_find_latest_model_artifact_uses_newest_superglm_model(tmp_path: Path):
    older = tmp_path / "MTPL_FREQ" / "2026-01-01" / "run1" / "superglm_model.pkl"
    newer = tmp_path / "MTPL_FREQ" / "2026-01-02" / "run2" / "superglm_model.pkl"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_bytes(b"old")
    newer.write_bytes(b"new")

    assert validator.find_latest_model_artifact(tmp_path, "MTPL_FREQ") == newer


def test_superglm_predictions_use_offset_not_sample_weight():
    calls = []

    class FakeModel:
        def predict(self, X, offset=None):
            calls.append((X.copy(), offset.copy()))
            return np.asarray([10.0, 20.0])

    training_frame = SimpleNamespace(
        X=pd.DataFrame({"x": [1, 2]}),
        offset=np.asarray([0.1, 0.2]),
    )

    predictions = validator.predict_with_superglm(FakeModel(), training_frame)

    np.testing.assert_allclose(predictions, np.asarray([10.0, 20.0]))
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][1], training_frame.offset)
