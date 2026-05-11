from types import SimpleNamespace

import pandas as pd
import pytest

from pricing_pipeline.data import fremtpl
from pricing_pipeline.data.fremtpl import (
    FREMTPL_COLUMNS,
    FREMTPL_DATASET_NAME,
    FREMTPL_OPENML_ID,
    bulk_insert_fremtpl_raw,
    fetch_fremtpl,
    fremtpl_insert_rows,
    load_fremtpl_raw,
    prepare_fremtpl_raw_frame,
    validate_fremtpl_raw,
)


def fremtpl_frame(**overrides):
    data = {
        "IDpol": [1],
        "ClaimNb": [0],
        "Exposure": [0.5],
        "Area": ["A"],
        "VehPower": [6],
        "VehAge": [3],
        "DrivAge": [45],
        "BonusMalus": [50],
        "VehBrand": ["B1"],
        "VehGas": ["Regular"],
        "Density": [123.0],
        "Region": ["R1"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_fetch_fremtpl_uses_openml_id_and_resets_index(monkeypatch):
    calls = []
    source = fremtpl_frame()
    source.index = [99]

    def fake_fetch_openml(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(frame=source)

    monkeypatch.setattr(fremtpl, "fetch_openml", fake_fetch_openml)

    out = fetch_fremtpl()

    assert FREMTPL_OPENML_ID == 41214
    assert FREMTPL_DATASET_NAME == "freMTPL2freq"
    assert calls == [{"data_id": 41214, "as_frame": True}]
    assert out.index.tolist() == [0]


def test_prepare_fremtpl_raw_preserves_expected_columns_and_int_keys():
    frame = fremtpl_frame(extra_column=["ignored"])

    out = prepare_fremtpl_raw_frame(frame)

    assert list(out.columns) == FREMTPL_COLUMNS
    assert out.loc[0, "Exposure"] == 0.5
    assert str(out["IDpol"].dtype) == "int64"
    assert str(out["ClaimNb"].dtype) == "int64"


def test_validate_fremtpl_raw_rejects_missing_columns():
    with pytest.raises(ValueError) as exc:
        validate_fremtpl_raw(pd.DataFrame({"IDpol": [1]}))

    message = str(exc.value)
    assert "missing columns" in message
    assert "ClaimNb" in message
    assert "Region" in message


def test_fremtpl_insert_rows_preserves_order_and_converts_missing_to_none():
    frame = fremtpl_frame(
        Area=[None],
        Density=[float("nan")],
        Region=[pd.NA],
    )

    rows = fremtpl_insert_rows(prepare_fremtpl_raw_frame(frame))

    assert rows == [
        (1, 0, 0.5, None, 6, 3, 45, 50, "B1", "Regular", None, None)
    ]


class ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one(self):
        return self.value


class FakeBeginConnection:
    def __init__(self, existing_count):
        self.existing_count = existing_count
        self.statements = []

    def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append((sql, params))
        if sql.startswith("SELECT COUNT_BIG(*) FROM pricing.FREMTPL_RAW"):
            return ScalarResult(self.existing_count)
        return ScalarResult(None)


class FakeBegin:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeCursor:
    def __init__(self, *, fail=False, events=None):
        self.fast_executemany = False
        self.execute_calls = []
        self.executemany_calls = []
        self.fail = fail
        self.closed = False
        self.events = events

    def execute(self, sql):
        self.execute_calls.append(sql)
        if self.events is not None:
            self.events.append("truncate")

    def executemany(self, sql, rows):
        if self.fail:
            raise RuntimeError("executemany failed")
        self.executemany_calls.append((sql, list(rows)))
        if self.events is not None:
            self.events.append("executemany")

    def close(self):
        self.closed = True


class FakeRawConnection:
    def __init__(self, cursor):
        self.cursor_obj = cursor
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class FakeEngine:
    def __init__(self, *, existing_count=0, raw_connection=None):
        self.begin_connection = FakeBeginConnection(existing_count)
        self.raw_connection_obj = raw_connection

    def begin(self):
        return FakeBegin(self.begin_connection)

    def raw_connection(self):
        if self.raw_connection_obj is None:
            raise AssertionError("raw_connection should not be used")
        return self.raw_connection_obj


def test_bulk_insert_fremtpl_raw_uses_raw_connection_chunks_commits_and_closes():
    frame = pd.concat(
        [
            fremtpl_frame(IDpol=[1], ClaimNb=[0]),
            fremtpl_frame(IDpol=[2], ClaimNb=[1]),
            fremtpl_frame(IDpol=[3], ClaimNb=[0]),
        ],
        ignore_index=True,
    )
    cursor = FakeCursor()
    raw_connection = FakeRawConnection(cursor)
    engine = FakeEngine(raw_connection=raw_connection)

    inserted = bulk_insert_fremtpl_raw(engine, frame, chunksize=2)

    assert inserted == 3
    assert cursor.fast_executemany is True
    assert cursor.execute_calls == []
    assert len(cursor.executemany_calls) == 2
    assert [len(rows) for _, rows in cursor.executemany_calls] == [2, 1]
    sql = cursor.executemany_calls[0][0]
    assert sql == (
        "INSERT INTO pricing.FREMTPL_RAW "
        "(IDpol, ClaimNb, Exposure, Area, VehPower, VehAge, DrivAge, "
        "BonusMalus, VehBrand, VehGas, Density, Region) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    assert raw_connection.commits == 1
    assert raw_connection.rollbacks == 0
    assert cursor.closed is True
    assert raw_connection.closed is True


def test_bulk_insert_fremtpl_raw_replace_truncates_and_inserts_in_one_transaction():
    cursor = FakeCursor()
    raw_connection = FakeRawConnection(cursor)
    engine = FakeEngine(raw_connection=raw_connection)

    inserted = bulk_insert_fremtpl_raw(engine, fremtpl_frame(), replace=True)

    assert inserted == 1
    assert cursor.execute_calls == ["TRUNCATE TABLE pricing.FREMTPL_RAW"]
    assert len(cursor.executemany_calls) == 1
    assert raw_connection.commits == 1
    assert raw_connection.rollbacks == 0
    assert cursor.closed is True
    assert raw_connection.closed is True


def test_bulk_insert_fremtpl_raw_replace_rolls_back_truncate_with_insert_failure():
    cursor = FakeCursor(fail=True)
    raw_connection = FakeRawConnection(cursor)
    engine = FakeEngine(raw_connection=raw_connection)

    with pytest.raises(RuntimeError, match="executemany failed"):
        bulk_insert_fremtpl_raw(engine, fremtpl_frame(), replace=True)

    assert cursor.execute_calls == ["TRUNCATE TABLE pricing.FREMTPL_RAW"]
    assert raw_connection.commits == 0
    assert raw_connection.rollbacks == 1
    assert cursor.closed is True
    assert raw_connection.closed is True


def test_bulk_insert_fremtpl_raw_rolls_back_and_closes_on_executemany_failure():
    cursor = FakeCursor(fail=True)
    raw_connection = FakeRawConnection(cursor)
    engine = FakeEngine(raw_connection=raw_connection)

    with pytest.raises(RuntimeError, match="executemany failed"):
        bulk_insert_fremtpl_raw(engine, fremtpl_frame())

    assert raw_connection.commits == 0
    assert raw_connection.rollbacks == 1
    assert cursor.closed is True
    assert raw_connection.closed is True


def test_load_fremtpl_raw_returns_existing_count_without_fetching(monkeypatch):
    engine = FakeEngine(existing_count=7)
    monkeypatch.setattr(
        fremtpl,
        "fetch_fremtpl",
        lambda: pytest.fail("fetch_fremtpl should not run when raw rows exist"),
    )

    rows = load_fremtpl_raw(engine, replace=False)

    assert rows == 7
    assert engine.begin_connection.statements == [
        ("SELECT COUNT_BIG(*) FROM pricing.FREMTPL_RAW", None)
    ]


def test_load_fremtpl_raw_fetches_and_prepares_before_replace_truncate(monkeypatch):
    events = []
    cursor = FakeCursor(events=events)
    raw_connection = FakeRawConnection(cursor)
    engine = FakeEngine(existing_count=7, raw_connection=raw_connection)
    source = fremtpl_frame(IDpol=["1"], ClaimNb=["0"])

    def fake_fetch_fremtpl():
        events.append("fetch")
        return source

    monkeypatch.setattr(fremtpl, "fetch_fremtpl", fake_fetch_fremtpl)
    rows = load_fremtpl_raw(engine, replace=True)

    assert rows == 1
    assert engine.begin_connection.statements == [
        ("SELECT COUNT_BIG(*) FROM pricing.FREMTPL_RAW", None),
    ]
    assert events == ["fetch", "truncate", "executemany"]
    assert cursor.execute_calls == ["TRUNCATE TABLE pricing.FREMTPL_RAW"]
    assert raw_connection.commits == 1
