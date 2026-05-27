from decimal import Decimal

import pandas as pd
import pytest

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.lifecycle import RatePackageSelector, RatePackageSnapshot
from pricing_pipeline.publishing.manual_revision import (
    ManualRevisionError,
    diff_rate_cell_edits,
    load_rate_package_snapshot,
    validate_rate_cell_edits,
)


def config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_key="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot="MTPL_FREQ_UAT",
        default_package_status="PUBLISHED",
    )


def metadata_row(**overrides):
    row = {
        "rate_package_id": 202,
        "parent_rate_package_id": None,
        "model_id": 17,
        "model_key": "MTPL_FREQ",
        "model_version": "2026.05",
        "package_version": 4,
        "base_rate": 1.25,
        "effective_from_date": "2026-01-01",
        "effective_to_date": None,
        "package_status": "PUBLISHED",
        "created_by": "unit-test",
    }
    row.update(overrides)
    return row


def rate_cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": [31, 29, 44],
            "term_id": [11, 11, 12],
            "cell_key_text": ["A", "B", "C"],
            "cell_key_digest": ["digest-a", "digest-b", "digest-c"],
            "is_reference": [False, True, False],
            "is_default": [False, False, True],
            "multiplier": [1.10, 1.00, 0.85],
            "log_coefficient": [0.09531, 0.0, -0.16252],
        },
    )


class ReadSqlFake:
    def __init__(self, *, metadata_rows):
        self.calls = []
        self.metadata = pd.DataFrame(metadata_rows)
        self.terms = pd.DataFrame(
            {"term_id": [11], "rate_package_id": [202], "term_name": ["Area"]},
        )
        self.rate_cells = pd.DataFrame({"cell_id": [31], "term_id": [11]})
        self.cell_levels = pd.DataFrame(
            {"cell_id": [31], "position_no": [1], "feature_level_id": [41]},
        )
        self.compiled_rate_cells = pd.DataFrame(
            {"rate_package_id": [202], "term_id": [11], "cell_key_text": ["A"]},
        )
        self.compiled_1d_bands = pd.DataFrame(
            {"rate_package_id": [202], "term_id": [11], "sort_order": [1]},
        )

    def __call__(self, sql, con, params=None):
        statement = str(sql)
        self.calls.append((statement, con, params or {}))
        if "FROM pricing.PRICING_RATE_PACKAGE" in statement:
            return self.metadata.copy()
        if "FROM pricing.PRICING_COMPILED_1D_RATE_BAND" in statement:
            return self.compiled_1d_bands.copy()
        if "FROM pricing.PRICING_COMPILED_RATE_CELL" in statement:
            return self.compiled_rate_cells.copy()
        if "FROM pricing.PRICING_RATE_CELL_LEVEL" in statement:
            return self.cell_levels.copy()
        if "FROM pricing.PRICING_RATE_CELL" in statement:
            return self.rate_cells.copy()
        if "FROM pricing.PRICING_TERM" in statement:
            return self.terms.copy()
        raise AssertionError(f"unexpected SQL: {statement}")


def test_load_rate_package_snapshot_by_id_scopes_metadata_to_model_key(monkeypatch):
    engine = object()
    fake_read_sql = ReadSqlFake(metadata_rows=[metadata_row(rate_package_id=101)])
    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision.pd.read_sql_query",
        fake_read_sql,
    )

    load_rate_package_snapshot(
        engine,
        config(),
        RatePackageSelector(rate_package_id=101),
    )

    statement, con, params = fake_read_sql.calls[0]
    assert con is engine
    assert "FROM pricing.PRICING_RATE_PACKAGE" in statement
    assert "JOIN pricing.PRICING_MODEL" in statement
    assert "m.model_key = :model_key" in statement
    assert "rp.rate_package_id = :rate_package_id" in statement
    assert params == {"model_key": "MTPL_FREQ", "rate_package_id": 101}


def test_load_rate_package_snapshot_by_version_scopes_metadata_to_model_key(monkeypatch):
    engine = object()
    fake_read_sql = ReadSqlFake(metadata_rows=[metadata_row()])
    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision.pd.read_sql_query",
        fake_read_sql,
    )

    load_rate_package_snapshot(
        engine,
        config(),
        RatePackageSelector(package_version=4),
    )

    statement, _con, params = fake_read_sql.calls[0]
    assert "m.model_key = :model_key" in statement
    assert "rp.package_version = :package_version" in statement
    assert params == {"model_key": "MTPL_FREQ", "package_version": 4}


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [metadata_row(rate_package_id=202), metadata_row(rate_package_id=303)],
    ],
)
def test_load_rate_package_snapshot_requires_one_metadata_row(monkeypatch, rows):
    fake_read_sql = ReadSqlFake(metadata_rows=rows)
    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision.pd.read_sql_query",
        fake_read_sql,
    )

    with pytest.raises(ManualRevisionError, match="exactly one package"):
        load_rate_package_snapshot(
            object(),
            config(),
            RatePackageSelector(package_version=4),
        )

    assert len(fake_read_sql.calls) == 1


def test_load_rate_package_snapshot_returns_all_tables_using_resolved_package_id(monkeypatch):
    engine = object()
    fake_read_sql = ReadSqlFake(metadata_rows=[metadata_row(rate_package_id=202)])
    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision.pd.read_sql_query",
        fake_read_sql,
    )

    snapshot = load_rate_package_snapshot(
        engine,
        config(),
        RatePackageSelector(package_version=4),
    )

    assert isinstance(snapshot, RatePackageSnapshot)
    assert snapshot.metadata == metadata_row(rate_package_id=202)
    pd.testing.assert_frame_equal(snapshot.terms, fake_read_sql.terms)
    pd.testing.assert_frame_equal(snapshot.rate_cells, fake_read_sql.rate_cells)
    pd.testing.assert_frame_equal(snapshot.cell_levels, fake_read_sql.cell_levels)
    pd.testing.assert_frame_equal(
        snapshot.compiled_rate_cells,
        fake_read_sql.compiled_rate_cells,
    )
    pd.testing.assert_frame_equal(
        snapshot.compiled_1d_bands,
        fake_read_sql.compiled_1d_bands,
    )
    subsequent_calls = fake_read_sql.calls[1:]
    assert [params for _sql, _con, params in subsequent_calls] == [
        {"rate_package_id": 202},
        {"rate_package_id": 202},
        {"rate_package_id": 202},
        {"rate_package_id": 202},
        {"rate_package_id": 202},
    ]
    assert "FROM pricing.PRICING_TERM" in subsequent_calls[0][0]
    assert "FROM pricing.PRICING_RATE_CELL" in subsequent_calls[1][0]
    assert "JOIN pricing.PRICING_TERM" in subsequent_calls[1][0]
    assert "FROM pricing.PRICING_RATE_CELL_LEVEL" in subsequent_calls[2][0]
    assert "JOIN pricing.PRICING_RATE_CELL" in subsequent_calls[2][0]
    assert "JOIN pricing.PRICING_TERM" in subsequent_calls[2][0]
    assert "FROM pricing.PRICING_COMPILED_RATE_CELL" in subsequent_calls[3][0]
    assert "FROM pricing.PRICING_COMPILED_1D_RATE_BAND" in subsequent_calls[4][0]


def test_diff_rate_cell_edits_returns_changed_multipliers_in_original_order():
    original = rate_cells()
    edited = original.copy()
    edited.loc[edited["cell_id"] == 44, "multiplier"] = 0.90
    edited.loc[edited["cell_id"] == 31, "multiplier"] = 1.25
    edited = edited.sort_values("cell_id").reset_index(drop=True)

    diff = diff_rate_cell_edits(original, edited)

    expected = pd.DataFrame(
        {
            "cell_id": [31, 44],
            "old_multiplier": [1.10, 0.85],
            "new_multiplier": [1.25, 0.90],
            "old_log_coefficient": [0.09531, -0.16252],
        },
    )
    pd.testing.assert_frame_equal(diff, expected)


def test_validate_rate_cell_edits_returns_stable_diff_shape():
    original = rate_cells()
    edited = original.copy()
    edited.loc[edited["cell_id"] == 29, "multiplier"] = 1.15

    diff = validate_rate_cell_edits(original, edited)

    assert list(diff.columns) == [
        "cell_id",
        "old_multiplier",
        "new_multiplier",
        "old_log_coefficient",
    ]
    pd.testing.assert_frame_equal(
        diff,
        pd.DataFrame(
            {
                "cell_id": [29],
                "old_multiplier": [1.00],
                "new_multiplier": [1.15],
                "old_log_coefficient": [0.0],
            },
        ),
    )


def test_validate_rate_cell_edits_rejects_empty_diff():
    original = rate_cells()

    with pytest.raises(ManualRevisionError, match="no manual rate cell changes"):
        validate_rate_cell_edits(original, original.copy())


def test_validate_rate_cell_edits_treats_decimal_and_string_same_values_as_no_diff():
    original = rate_cells()
    original["multiplier"] = [Decimal("1.10"), Decimal("1.00"), Decimal("0.85")]
    edited = original.copy()
    edited["multiplier"] = ["1.10", "1.00", "0.85"]

    with pytest.raises(ManualRevisionError, match="no manual rate cell changes"):
        validate_rate_cell_edits(original, edited)


def test_diff_rate_cell_edits_returns_numeric_diff_for_decimal_and_string_change():
    original = rate_cells()
    original["multiplier"] = [Decimal("1.10"), Decimal("1.00"), Decimal("0.85")]
    edited = original.copy()
    edited["multiplier"] = ["1.10", "1.05", "0.85"]

    diff = diff_rate_cell_edits(original, edited)

    pd.testing.assert_frame_equal(
        diff,
        pd.DataFrame(
            {
                "cell_id": [29],
                "old_multiplier": [1.0],
                "new_multiplier": [1.05],
                "old_log_coefficient": [0.0],
            },
        ),
    )


def test_diff_rate_cell_edits_returns_numeric_log_coefficient_for_decimal_input():
    original = rate_cells()
    original["multiplier"] = [Decimal("1.10"), Decimal("1.00"), Decimal("0.85")]
    original["log_coefficient"] = [
        Decimal("0.09531"),
        Decimal("0.0"),
        Decimal("-0.16252"),
    ]
    edited = original.copy()
    edited["multiplier"] = ["1.25", "1.00", "0.85"]

    diff = diff_rate_cell_edits(original, edited)

    assert pd.api.types.is_float_dtype(diff["old_log_coefficient"])
    assert diff.loc[0, "old_log_coefficient"] == pytest.approx(0.09531)


def test_diff_rate_cell_edits_rejects_non_numeric_original_log_coefficient():
    original = rate_cells()
    edited = original.copy()
    original["log_coefficient"] = original["log_coefficient"].astype(object)
    original.loc[original["cell_id"] == 31, "log_coefficient"] = "not-a-number"
    edited.loc[edited["cell_id"] == 31, "multiplier"] = 1.25

    with pytest.raises(ManualRevisionError, match="original.*log_coefficient.*numeric"):
        diff_rate_cell_edits(original, edited)


def test_validate_rate_cell_edits_ignores_tiny_multiplier_roundtrip_delta():
    original = rate_cells()
    edited = original.copy()
    edited.loc[edited["cell_id"] == 31, "multiplier"] = 1.10 + 1e-13

    with pytest.raises(ManualRevisionError, match="no manual rate cell changes"):
        validate_rate_cell_edits(original, edited)


def test_diff_rate_cell_edits_rejects_non_numeric_original_multiplier():
    original = rate_cells()
    edited = original.copy()
    original["multiplier"] = original["multiplier"].astype(object)
    original.loc[original["cell_id"] == 31, "multiplier"] = "not-a-number"
    edited.loc[edited["cell_id"] == 31, "multiplier"] = 1.25

    with pytest.raises(ManualRevisionError, match="original.*multiplier.*numeric"):
        diff_rate_cell_edits(original, edited)


def test_validate_rate_cell_edits_rejects_identity_column_change():
    original = rate_cells()
    edited = original.copy()
    edited.loc[edited["cell_id"] == 31, "term_id"] = 99
    edited.loc[edited["cell_id"] == 31, "multiplier"] = 1.25

    with pytest.raises(ManualRevisionError, match="term_id"):
        validate_rate_cell_edits(original, edited)


@pytest.mark.parametrize("multiplier", [0.0, -0.5])
def test_validate_rate_cell_edits_rejects_non_positive_multiplier(multiplier):
    original = rate_cells()
    edited = original.copy()
    edited.loc[edited["cell_id"] == 31, "multiplier"] = multiplier

    with pytest.raises(ManualRevisionError, match="positive finite numbers"):
        validate_rate_cell_edits(original, edited)


@pytest.mark.parametrize("multiplier", [float("nan"), float("inf"), float("-inf")])
def test_validate_rate_cell_edits_rejects_non_finite_multiplier(multiplier):
    original = rate_cells()
    edited = original.copy()
    edited.loc[edited["cell_id"] == 31, "multiplier"] = multiplier

    with pytest.raises(ManualRevisionError, match="positive finite numbers"):
        validate_rate_cell_edits(original, edited)


@pytest.mark.parametrize(
    ("cell_ids", "message"),
    [
        ([31, 29], "same cell_id values"),
        ([31, 29, 45], "same cell_id values"),
    ],
)
def test_validate_rate_cell_edits_rejects_missing_or_extra_cell_ids(cell_ids, message):
    original = rate_cells()
    edited = original[original["cell_id"].isin(cell_ids)].copy()
    if 45 in cell_ids:
        extra = original.iloc[[0]].copy()
        extra["cell_id"] = 45
        edited = pd.concat([edited, extra], ignore_index=True)
    edited.loc[edited["cell_id"] == 31, "multiplier"] = 1.25

    with pytest.raises(ManualRevisionError, match=message):
        validate_rate_cell_edits(original, edited)


@pytest.mark.parametrize("frame_name", ["original", "edited"])
def test_validate_rate_cell_edits_rejects_duplicate_cell_ids(frame_name):
    original = rate_cells()
    edited = original.copy()
    edited.loc[edited["cell_id"] == 31, "multiplier"] = 1.25
    if frame_name == "original":
        original = pd.concat([original, original.iloc[[0]]], ignore_index=True)
    else:
        edited = pd.concat([edited, edited.iloc[[0]]], ignore_index=True)

    with pytest.raises(ManualRevisionError, match=f"duplicate cell_id values in {frame_name}"):
        validate_rate_cell_edits(original, edited)
