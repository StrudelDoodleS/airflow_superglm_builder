from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pricing_pipeline.data.frame_artifact import (
    ModelFrameArtifactError,
    inspect_model_frame,
    load_model_frame,
    save_model_frame,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy_id": pd.Series([1, 2, 3], dtype="int64"),
            "claim_count": pd.Series([0, 1, 0], dtype="int64"),
            "area": pd.Series(["A", "B", "A"], dtype="string"),
            "data_as_of": pd.Series(["2026-06-30"] * 3, dtype="string"),
        }
    )


def test_model_frame_handoff_round_trips_and_identical_save_is_idempotent(
    tmp_path: Path,
):
    path = tmp_path / "model_frame.joblib"
    frame = _frame()

    first = save_model_frame(frame, path)
    retry = save_model_frame(frame.copy(), path)
    loaded = load_model_frame(path)

    assert retry == first
    assert inspect_model_frame(path) == first
    assert Path(first.path) == path.resolve()
    assert first.row_count == 3
    assert first.columns == tuple(frame.columns)
    pd.testing.assert_frame_equal(loaded, frame)


def test_model_frame_handoff_requires_explicit_replace_for_changed_frame(
    tmp_path: Path,
):
    path = tmp_path / "model_frame.joblib"
    original = _frame()
    save_model_frame(original, path)
    changed = original.copy()
    changed.loc[1, "area"] = "C"

    with pytest.raises(FileExistsError, match="replace=True"):
        save_model_frame(changed, path)

    replaced = save_model_frame(changed, path, replace=True)
    assert (
        replaced.model_frame_sha256
        != save_model_frame(
            original,
            tmp_path / "original.joblib",
        ).model_frame_sha256
    )
    pd.testing.assert_frame_equal(load_model_frame(path), changed)


def test_model_frame_handoff_rejects_tampered_artifact_before_deserialization(
    tmp_path: Path,
):
    path = tmp_path / "model_frame.joblib"
    save_model_frame(_frame(), path)
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ModelFrameArtifactError, match="size does not match"):
        load_model_frame(path)
