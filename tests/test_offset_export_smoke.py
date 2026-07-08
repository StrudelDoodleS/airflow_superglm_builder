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


def test_offset_export_smoke_scopes_result_rows_to_current_package(tmp_path):
    db_root = tmp_path / "offset_export_smoke"
    first = run_offset_export_smoke(
        db_root=db_root,
        effective_from="2026-06-19",
        reset=True,
    )
    second = run_offset_export_smoke(
        db_root=db_root,
        effective_from="2026-06-20",
        reset=False,
    )

    assert first["rate_package_id"] != second["rate_package_id"]
    assert [row["level_code"] for row in second["final_offset_rows"]] == ["12", "36"]
    assert [row["cell_key_text"] for row in second["compiled_offset_rows"]] == [
        "TermMonths=12",
        "TermMonths=36",
    ]
    assert second["tables"]["pricing"]["PRICING_RATE_PACKAGE"] == 2
