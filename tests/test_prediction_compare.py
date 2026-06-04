import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from pricing_pipeline.publishing.prediction_compare import compare_prediction_vectors


def test_compare_prediction_vectors_reports_summary_and_top_changes():
    before = pd.Series([100.0, 200.0, 300.0], name="before")
    after = pd.Series([110.0, 190.0, 330.0], name="after")

    comparison = compare_prediction_vectors(before, after, top_n=2)

    assert comparison.summary == {
        "row_count": 3.0,
        "mean_absolute_change": 16.666666666666668,
        "max_absolute_change": 30.0,
        "mean_relative_change": 0.08333333333333333,
        "max_relative_change": 0.1,
    }
    expected_changed_rows = pd.DataFrame(
        {
            "row_index": [2, 0],
            "before": [300.0, 100.0],
            "after": [330.0, 110.0],
            "absolute_change": [30.0, 10.0],
            "relative_change": [0.1, 0.1],
        },
    )
    pdt.assert_frame_equal(comparison.changed_rows, expected_changed_rows)


def test_compare_prediction_vectors_breaks_ties_by_row_index():
    before = pd.Series([100.0, 100.0, 100.0])
    after = pd.Series([90.0, 110.0, 130.0])

    comparison = compare_prediction_vectors(before, after)

    assert comparison.changed_rows["row_index"].tolist() == [2, 0, 1]


def test_compare_prediction_vectors_rejects_length_mismatch():
    before = pd.Series([1.0, 2.0])
    after = pd.Series([1.0])

    with pytest.raises(ValueError, match="same length"):
        compare_prediction_vectors(before, after)


def test_compare_prediction_vectors_rejects_empty_vectors():
    with pytest.raises(ValueError, match="must not be empty"):
        compare_prediction_vectors(pd.Series([], dtype="float64"), pd.Series([], dtype="float64"))


def test_compare_prediction_vectors_treats_zero_baseline_relative_change_as_zero():
    before = pd.Series([0.0, -4.0, 2.0])
    after = pd.Series([5.0, -2.0, 1.0])

    comparison = compare_prediction_vectors(before, after, top_n=3)

    assert comparison.summary["mean_relative_change"] == pytest.approx(1.0 / 3.0)
    assert comparison.summary["max_relative_change"] == 0.5
    assert comparison.changed_rows["relative_change"].tolist() == [0.0, 0.5, 0.5]


def test_compare_prediction_vectors_rejects_negative_top_n():
    with pytest.raises(ValueError, match="top_n"):
        compare_prediction_vectors(pd.Series([1.0]), pd.Series([2.0]), top_n=-1)


def test_compare_prediction_vectors_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        compare_prediction_vectors(pd.Series([1.0, np.inf]), pd.Series([2.0, 3.0]))

    with pytest.raises(ValueError, match="finite"):
        compare_prediction_vectors(pd.Series([1.0, 2.0]), pd.Series([2.0, np.nan]))


def test_compare_prediction_vectors_rejects_overflowed_absolute_change():
    with pytest.raises(ValueError, match="prediction changes must be finite"):
        compare_prediction_vectors(pd.Series([1.79e308]), pd.Series([-1.79e308]))


def test_compare_prediction_vectors_rejects_overflowed_relative_change():
    with pytest.raises(ValueError, match="prediction changes must be finite"):
        compare_prediction_vectors(pd.Series([5e-324]), pd.Series([1.0]))


def test_compare_prediction_vectors_rejects_overflowed_summary_values():
    max_float = np.finfo("float64").max

    with pytest.raises(ValueError, match="prediction summary values must be finite"):
        compare_prediction_vectors(pd.Series([0.0, 0.0]), pd.Series([max_float, max_float]))
