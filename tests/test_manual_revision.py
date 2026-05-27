import pandas as pd
import pytest

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.lifecycle import RatePackageSelector, RatePackageSnapshot
from pricing_pipeline.publishing.manual_revision import (
    ManualRevisionError,
    load_rate_package_snapshot,
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
