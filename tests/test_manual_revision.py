from decimal import Decimal
import json

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.lifecycle import RatePackageSelector, RatePackageSnapshot
from pricing_pipeline.publishing.manual_revision import (
    ManualRevisionError,
    _write_manual_revision,
    create_manual_revision,
    diff_rate_cell_edits,
    load_rate_package_snapshot,
    validate_rate_cell_edits,
)


def config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_name="MTPL_FREQ",
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
        "model_name": "MTPL_FREQ",
        "model_version": "2026.05",
        "package_version": 4,
        "base_rate": 1.25,
        "effective_from_date": "2026-01-01",
        "effective_to_date": None,
        "package_status": "PUBLISHED",
        "created_by": "unit-test",
        "publication_receipt_json": '{"schema_version":1}',
        "publication_receipt_sha256": "a" * 64,
        "package_metadata_json": '{"model":{"family":"poisson"}}',
        "revision_metadata_json": None,
        "offset_handling": "NONE",
        "offset_factor_name": None,
        "offset_source_name": None,
        "offset_label": None,
        "metadata_origin": "SUPERGLM_FITTED_MODEL",
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


def snapshot(
    terms: pd.DataFrame | None = None,
    rate_cells_frame: pd.DataFrame | None = None,
    **metadata_overrides,
) -> RatePackageSnapshot:
    return RatePackageSnapshot(
        metadata=metadata_row(**metadata_overrides),
        terms=terms if terms is not None else pd.DataFrame(),
        rate_cells=rate_cells_frame if rate_cells_frame is not None else rate_cells(),
        cell_levels=pd.DataFrame(),
        compiled_rate_cells=pd.DataFrame(),
        compiled_1d_bands=pd.DataFrame(),
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


class ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one(self):
        return self.value


class FakeConnection:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        statement = str(sql)
        self.calls.append((statement, params))
        if "WITH (UPDLOCK, HOLDLOCK)" in statement:
            return ScalarResult(5)
        if (
            "INSERT INTO pricing.PRICING_RATE_PACKAGE" in statement
            and "OUTPUT INSERTED.rate_package_id" in statement
        ):
            return ScalarResult(303)
        return ScalarResult(None)


class FakeBegin:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeEngine:
    def __init__(self):
        self.connection = FakeConnection()
        self.begin_count = 0

    def begin(self):
        self.begin_count += 1
        return FakeBegin(self.connection)


def test_load_rate_package_snapshot_by_id_scopes_metadata_to_model_name(monkeypatch):
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
    assert "m.model_name = :model_name" in statement
    assert "rp.rate_package_id = :rate_package_id" in statement
    assert params == {"model_name": "MTPL_FREQ", "rate_package_id": 101}


def test_load_rate_package_snapshot_by_version_scopes_metadata_to_model_name(monkeypatch):
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
    assert "m.model_name = :model_name" in statement
    assert "rp.package_version = :package_version" in statement
    assert params == {"model_name": "MTPL_FREQ", "package_version": 4}


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


def test_write_manual_revision_creates_child_package_and_copies_children():
    engine = FakeEngine()
    parent = snapshot(rate_package_id=202, model_id=17, package_version=4)
    diff = pd.DataFrame(
        {
            "cell_id": [31, 44],
            "old_multiplier": [1.10, 0.85],
            "new_multiplier": [1.25, 0.90],
            "old_log_coefficient": [0.09531, -0.16252],
        },
    )

    result = _write_manual_revision(
        engine,
        config(),
        parent=parent,
        edited_rate_cells=rate_cells(),
        diff=diff,
        reason="pricing correction",
        created_by="pricing-user",
    )

    assert result == (303, 5)
    assert engine.begin_count == 1
    calls = engine.connection.calls
    statements = "\n".join(statement for statement, _params in calls)

    version_sql, version_params = next(
        (statement, params)
        for statement, params in calls
        if "WITH (UPDLOCK, HOLDLOCK)" in statement
    )
    assert "MAX(package_version)" in version_sql
    assert "WHERE model_id = :model_id" in version_sql
    assert version_params == {"model_id": 17}

    insert_sql, insert_params = next(
        (statement, params)
        for statement, params in calls
        if "INSERT INTO pricing.PRICING_RATE_PACKAGE" in statement
        and "OUTPUT INSERTED.rate_package_id" in statement
    )
    assert "parent_rate_package_id" in insert_sql
    assert "package_status" in insert_sql
    expected_revision_metadata = json.dumps(
        {
            "parent_rate_package_id": 202,
            "reason": "pricing correction",
            "revision_kind": "MANUAL",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert insert_params == {
        "parent_rate_package_id": 202,
        "model_id": 17,
        "model_name": "MTPL_FREQ",
        "model_version": "2026.05",
        "package_version": 5,
        "base_rate": 1.25,
        "effective_from_date": "2026-01-01",
        "effective_to_date": None,
        "package_status": "DRAFT",
        "publication_receipt_json": '{"schema_version":1}',
        "publication_receipt_sha256": "a" * 64,
        "package_metadata_json": '{"model":{"family":"poisson"}}',
        "revision_metadata_json": expected_revision_metadata,
        "offset_handling": "NONE",
        "offset_factor_name": None,
        "offset_source_name": None,
        "offset_label": None,
        "metadata_origin": "SUPERGLM_FITTED_MODEL",
        "created_by": "pricing-user",
    }

    assert "DROP TABLE IF EXISTS #manual_rate_cell_edits" in statements
    assert "CREATE TABLE #manual_rate_cell_edits" in statements
    edit_sql, edit_params = next(
        (statement, params)
        for statement, params in calls
        if "INSERT INTO #manual_rate_cell_edits" in statement
    )
    assert "cell_id" in edit_sql
    assert "log_coefficient" in edit_sql
    assert edit_params[0]["cell_id"] == 31
    assert edit_params[0]["multiplier"] == 1.25
    assert edit_params[0]["log_coefficient"] == pytest.approx(np.log(1.25))
    assert edit_params[1]["cell_id"] == 44
    assert edit_params[1]["multiplier"] == 0.90
    assert edit_params[1]["log_coefficient"] == pytest.approx(np.log(0.90))

    assert "#term_map" in statements
    assert "#cell_map" in statements
    assert "MERGE pricing.PRICING_TERM" in statements
    assert "MERGE pricing.PRICING_RATE_CELL" in statements
    for copied_table in [
        "pricing.PRICING_TERM_FEATURE",
        "pricing.PRICING_RATE_CELL_LEVEL",
        "pricing.PRICING_COMPILED_RATE_CELL",
        "pricing.PRICING_COMPILED_1D_RATE_BAND",
    ]:
        assert copied_table in statements
    assert "COALESCE(edit.multiplier" in statements
    assert "COALESCE(edit.log_coefficient" in statements

    finalize_sql, finalize_params = next(
        (statement, params)
        for statement, params in calls
        if "UPDATE pricing.PRICING_RATE_PACKAGE" in statement and "'PUBLISHED'" in statement
    )
    assert "SET package_status = 'PUBLISHED'" in finalize_sql
    assert finalize_params == {"rate_package_id": 303}


def test_write_manual_revision_copies_receipt_and_term_metadata():
    engine = FakeEngine()
    parent = snapshot(rate_package_id=202, model_id=17, package_version=4)
    diff = pd.DataFrame(
        {
            "cell_id": [31],
            "old_multiplier": [1.10],
            "new_multiplier": [1.25],
            "old_log_coefficient": [0.09531],
        },
    )

    _write_manual_revision(
        engine,
        config(),
        parent=parent,
        edited_rate_cells=rate_cells(),
        diff=diff,
        reason="pricing correction",
        created_by="pricing-user",
    )

    calls = engine.connection.calls
    statements = "\n".join(statement for statement, _params in calls)
    insert_sql, insert_params = next(
        (statement, params)
        for statement, params in calls
        if "INSERT INTO pricing.PRICING_RATE_PACKAGE" in statement
        and "OUTPUT INSERTED.rate_package_id" in statement
    )

    assert "publication_receipt_json" in insert_sql
    assert "publication_receipt_sha256" in insert_sql
    assert "revision_metadata_json" in insert_sql
    assert insert_params["publication_receipt_sha256"] == "a" * 64
    assert insert_params["package_metadata_json"] == '{"model":{"family":"poisson"}}'
    assert insert_params["offset_handling"] == "NONE"
    revision_metadata = json.loads(insert_params["revision_metadata_json"])
    assert revision_metadata == {
        "parent_rate_package_id": 202,
        "reason": "pricing correction",
        "revision_kind": "MANUAL",
    }
    assert "term_metadata_json" in statements


def test_write_manual_revision_payload_args_are_keyword_only():
    with pytest.raises(TypeError):
        _write_manual_revision(
            object(),
            config(),
            snapshot(),
            rate_cells(),
            pd.DataFrame(),
            "pricing correction",
            "pricing-user",
        )


def test_create_manual_revision_requires_reason(monkeypatch):
    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision._write_manual_revision",
        lambda *args: pytest.fail("writer should not be called"),
    )

    with pytest.raises(ManualRevisionError, match="reason is required"):
        create_manual_revision(
            object(),
            config(),
            parent=snapshot(),
            edited_rate_cells=rate_cells(),
            reason="  ",
            created_by="pricing-user",
        )


def test_create_manual_revision_requires_created_by(monkeypatch):
    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision._write_manual_revision",
        lambda *args: pytest.fail("writer should not be called"),
    )

    with pytest.raises(ManualRevisionError, match="created_by is required"):
        create_manual_revision(
            object(),
            config(),
            parent=snapshot(),
            edited_rate_cells=rate_cells(),
            reason="pricing correction",
            created_by="  ",
        )


def test_create_manual_revision_rejects_non_published_parent(monkeypatch):
    edited = rate_cells()
    edited.loc[edited["cell_id"] == 31, "multiplier"] = 1.25
    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision._write_manual_revision",
        lambda *args: pytest.fail("writer should not be called"),
    )

    with pytest.raises(ManualRevisionError, match="PUBLISHED"):
        create_manual_revision(
            object(),
            config(),
            parent=snapshot(package_status="DRAFT"),
            edited_rate_cells=edited,
            reason="pricing correction",
            created_by="pricing-user",
        )


def test_create_manual_revision_rejects_offset_factor_edits_before_writer(monkeypatch):
    parent = snapshot(
        terms=pd.DataFrame(
            {
                "term_id": [11, 12],
                "term_name": ["Offset_Multiplier", "Area"],
                "term_type": ["OFFSET_FACTOR", "CATEGORICAL_MAIN"],
            }
        )
    )
    edited = parent.rate_cells.copy()
    edited.loc[edited["cell_id"] == 31, "multiplier"] = 1.25
    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision._write_manual_revision",
        lambda *args, **kwargs: pytest.fail("writer should not be called"),
    )

    with pytest.raises(ManualRevisionError, match="OFFSET_FACTOR"):
        create_manual_revision(
            object(),
            config(),
            parent=parent,
            edited_rate_cells=edited,
            reason="pricing correction",
            created_by="pricing-user",
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "rate_package_id",
        "model_id",
        "model_name",
        "model_version",
        "base_rate",
        "effective_from_date",
        "effective_to_date",
        "package_status",
        "package_version",
    ],
)
def test_create_manual_revision_rejects_missing_parent_metadata_before_writer(
    monkeypatch,
    missing_key,
):
    parent = snapshot()
    parent.metadata.pop(missing_key)
    edited = parent.rate_cells.copy()
    edited.loc[edited["cell_id"] == 31, "multiplier"] = 1.25
    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision._write_manual_revision",
        lambda *args, **kwargs: pytest.fail("writer should not be called"),
    )

    with pytest.raises(ManualRevisionError, match=rf"parent metadata.*{missing_key}"):
        create_manual_revision(
            object(),
            config(),
            parent=parent,
            edited_rate_cells=edited,
            reason="pricing correction",
            created_by="pricing-user",
        )


@pytest.mark.parametrize(
    "metadata_overrides",
    [
        {"model_name": "OTHER_MODEL"},
        {"registry_model_name": "OTHER_MODEL"},
        {"package_model_name": "OTHER_MODEL"},
    ],
)
def test_create_manual_revision_rejects_parent_model_mismatch_before_writer(
    monkeypatch,
    metadata_overrides,
):
    parent = snapshot(**metadata_overrides)
    edited = parent.rate_cells.copy()
    edited.loc[edited["cell_id"] == 31, "multiplier"] = 1.25
    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision._write_manual_revision",
        lambda *args, **kwargs: pytest.fail("writer should not be called"),
    )

    with pytest.raises(
        ManualRevisionError,
        match="parent model.*configured model.*mismatch",
    ):
        create_manual_revision(
            object(),
            config(),
            parent=parent,
            edited_rate_cells=edited,
            reason="pricing correction",
            created_by="pricing-user",
        )


def test_create_manual_revision_returns_result_from_writer_and_diff(monkeypatch):
    engine = object()
    parent = snapshot(rate_package_id=202, package_status="PUBLISHED")
    edited = parent.rate_cells.copy()
    edited.loc[edited["cell_id"] == 44, "multiplier"] = 0.90
    edited.loc[edited["cell_id"] == 31, "multiplier"] = 1.25
    edited = edited.sort_values("cell_id").reset_index(drop=True)
    calls = []

    def fake_write_manual_revision(
        engine_arg,
        config_arg,
        *,
        parent,
        edited_rate_cells,
        diff,
        reason,
        created_by,
    ):
        calls.append(
            (
                engine_arg,
                config_arg,
                parent,
                edited_rate_cells,
                diff.copy(),
                reason,
                created_by,
            ),
        )
        return 303, 5

    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision._write_manual_revision",
        fake_write_manual_revision,
    )

    result = create_manual_revision(
        engine,
        config(),
        parent=parent,
        edited_rate_cells=edited,
        reason="  pricing correction  ",
        created_by="  pricing-user  ",
    )

    expected_diff = pd.DataFrame(
        {
            "cell_id": [31, 44],
            "old_multiplier": [1.10, 0.85],
            "new_multiplier": [1.25, 0.90],
            "old_log_coefficient": [0.09531, -0.16252],
        },
    )
    assert result.rate_package_id == 303
    assert result.package_version == 5
    assert result.parent_rate_package_id == 202
    assert result.changed_rate_cell_count == 2
    assert result.base_rate_changed is False
    pd.testing.assert_frame_equal(result.diff_summary, expected_diff)
    assert len(calls) == 1
    (
        engine_arg,
        config_arg,
        parent_arg,
        edited_rate_cells_arg,
        diff_arg,
        reason_arg,
        created_by_arg,
    ) = calls[0]
    assert engine_arg is engine
    assert config_arg == config()
    assert parent_arg is parent
    pd.testing.assert_frame_equal(edited_rate_cells_arg, edited)
    pd.testing.assert_frame_equal(diff_arg, expected_diff)
    assert reason_arg == "pricing correction"
    assert created_by_arg == "pricing-user"
