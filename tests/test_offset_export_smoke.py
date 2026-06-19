from __future__ import annotations

import pytest

from scripts.run_offset_export_smoke import run_offset_export_smoke


def test_offset_export_smoke_publishes_transformed_offset_factor(tmp_path):
    result = run_offset_export_smoke(
        db_root=tmp_path / "offset_export_smoke",
        effective_from="2026-06-19",
        reset=True,
    )

    offset_rows = result["final_offset_rows"]
    assert [row["level_code"] for row in offset_rows] == ["12", "36"]
    assert [row["cell_key_text"] for row in offset_rows] == [
        "TermMonths=12",
        "TermMonths=36",
    ]
    assert [row["term_name"] for row in offset_rows] == ["TermMonths", "TermMonths"]
    assert offset_rows[0]["multiplier"] == pytest.approx(1.0)
    assert offset_rows[1]["multiplier"] == pytest.approx(3.0)

    staging_rows = result["staging_offset_rows"]
    assert [row["level_code"] for row in staging_rows] == ["12", "36"]
    assert [row["cell_key_text"] for row in staging_rows] == [
        "TermMonths=12",
        "TermMonths=36",
    ]
    assert staging_rows[0]["multiplier"] == pytest.approx(1.0)
    assert staging_rows[1]["multiplier"] == pytest.approx(3.0)

    tables = result["tables"]
    assert tables["pricing_stg"]["STG_RATING_EXPORT"] == 1
    assert tables["pricing"]["PRICING_RATE_PACKAGE"] == 1
    assert tables["pricing"]["PRICING_TERM"] >= 2
