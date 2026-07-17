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
    assert fremtpl.FREMTPL_EXPECTED_ROW_COUNT == 678_013
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


def test_fremtpl_data_api_has_no_synthetic_demo_seed_helpers():
    assert not hasattr(fremtpl, "synthetic_fremtpl_raw_frame")
    assert not hasattr(fremtpl, "ensure_local_fremtpl_demo")


def test_load_fremtpl_raw_loads_full_fetched_frame_into_empty_sqlite(
    monkeypatch,
    tmp_path,
):
    from sqlalchemy import text

    from pricing_pipeline.infra.offline_sqlite import open_offline_sqlite

    source = pd.concat(
        [
            fremtpl_frame(IDpol=[1], ClaimNb=[0]),
            fremtpl_frame(IDpol=[2], ClaimNb=[1]),
            fremtpl_frame(IDpol=[3], ClaimNb=[0]),
        ],
        ignore_index=True,
    )
    engine, _paths = open_offline_sqlite(tmp_path)
    monkeypatch.setattr(
        fremtpl,
        "FREMTPL_EXPECTED_ROW_COUNT",
        len(source),
        raising=False,
    )
    monkeypatch.setattr(fremtpl, "fetch_fremtpl", lambda: source)

    inserted = load_fremtpl_raw(engine)

    assert inserted == len(source)
    with engine.connect() as connection:
        count = connection.execute(
            text("SELECT COUNT(*) FROM pricing.FREMTPL_RAW")
        ).scalar_one()
    assert count == len(source)


def test_load_fremtpl_raw_second_non_replace_call_reuses_sqlite_rows(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline.infra.offline_sqlite import open_offline_sqlite

    source = pd.concat(
        [fremtpl_frame(IDpol=[1]), fremtpl_frame(IDpol=[2])],
        ignore_index=True,
    )
    engine, _paths = open_offline_sqlite(tmp_path)
    monkeypatch.setattr(
        fremtpl,
        "FREMTPL_EXPECTED_ROW_COUNT",
        len(source),
        raising=False,
    )
    monkeypatch.setattr(fremtpl, "fetch_fremtpl", lambda: source)
    assert load_fremtpl_raw(engine) == len(source)
    monkeypatch.setattr(
        fremtpl,
        "fetch_fremtpl",
        lambda: pytest.fail("fetch_fremtpl should not run when raw rows exist"),
    )

    assert load_fremtpl_raw(engine, replace=False) == len(source)


def test_load_fremtpl_raw_partial_store_survives_fetch_failure(
    monkeypatch,
    tmp_path,
):
    from sqlalchemy import text

    from pricing_pipeline.infra.offline_sqlite import open_offline_sqlite

    engine, _paths = open_offline_sqlite(tmp_path)
    bulk_insert_fremtpl_raw(engine, fremtpl_frame(IDpol=[41]))

    def fail_fetch():
        raise RuntimeError("fetch failed")

    monkeypatch.setattr(fremtpl, "fetch_fremtpl", fail_fetch)

    with pytest.raises(RuntimeError, match="fetch failed"):
        load_fremtpl_raw(engine, replace=False)

    with engine.connect() as connection:
        ids = connection.execute(
            text("SELECT IDpol FROM pricing.FREMTPL_RAW")
        ).scalars().all()
    assert ids == [41]


def test_load_fremtpl_raw_wrong_fetched_count_does_not_clear_partial_store(
    monkeypatch,
    tmp_path,
):
    from sqlalchemy import text

    from pricing_pipeline.infra.offline_sqlite import open_offline_sqlite

    engine, _paths = open_offline_sqlite(tmp_path)
    bulk_insert_fremtpl_raw(engine, fremtpl_frame(IDpol=[41]))
    source = pd.concat(
        [fremtpl_frame(IDpol=[1]), fremtpl_frame(IDpol=[2])],
        ignore_index=True,
    )
    monkeypatch.setattr(
        fremtpl,
        "FREMTPL_EXPECTED_ROW_COUNT",
        3,
        raising=False,
    )
    monkeypatch.setattr(fremtpl, "fetch_fremtpl", lambda: source)

    with pytest.raises(ValueError, match="expected 3 rows, got 2"):
        load_fremtpl_raw(engine, replace=False)

    with engine.connect() as connection:
        ids = connection.execute(
            text("SELECT IDpol FROM pricing.FREMTPL_RAW")
        ).scalars().all()
    assert ids == [41]


def test_open_offline_sqlite_adds_digest_columns_to_existing_store(tmp_path):
    import sqlite3

    from pricing_pipeline.infra.offline_sqlite import open_offline_sqlite

    engine, paths = open_offline_sqlite(tmp_path)
    engine.dispose()
    with sqlite3.connect(paths["pricing"]) as connection:
        connection.execute("ALTER TABLE PRICING_RATE_PACKAGE DROP COLUMN staging_content_sha256")
    with sqlite3.connect(paths["pricing_stg"]) as connection:
        connection.execute("ALTER TABLE STG_RATING_EXPORT DROP COLUMN staging_content_sha256")

    upgraded, _paths = open_offline_sqlite(tmp_path)
    with upgraded.connect() as connection:
        package_columns = {
            row[1]
            for row in connection.exec_driver_sql(
                "PRAGMA pricing.table_info('PRICING_RATE_PACKAGE')"
            )
        }
        staging_columns = {
            row[1]
            for row in connection.exec_driver_sql(
                "PRAGMA pricing_stg.table_info('STG_RATING_EXPORT')"
            )
        }

    assert "staging_content_sha256" in package_columns
    assert "staging_content_sha256" in staging_columns


def test_open_offline_sqlite_adds_model_run_candidate_columns_to_existing_store(
    tmp_path,
):
    import sqlite3

    from pricing_pipeline.infra.offline_sqlite import open_offline_sqlite

    candidate_columns = {
        "candidate_artifact_path",
        "candidate_artifact_sha256",
        "candidate_artifact_format",
        "candidate_artifact_size_bytes",
        "candidate_python_version",
        "candidate_superglm_version",
        "candidate_superglm_git_sha",
        "model_source_sha256",
    }
    engine, paths = open_offline_sqlite(tmp_path)
    engine.dispose()
    with sqlite3.connect(paths["pricing"]) as connection:
        for column in sorted(candidate_columns):
            connection.execute(f"ALTER TABLE MODEL_RUN DROP COLUMN {column}")

    upgraded, _paths = open_offline_sqlite(tmp_path)
    with upgraded.connect() as connection:
        model_run_columns = {
            row[1] for row in connection.exec_driver_sql("PRAGMA pricing.table_info('MODEL_RUN')")
        }

    assert candidate_columns <= model_run_columns


def test_open_offline_sqlite_adds_parent_model_run_id_to_existing_store(tmp_path):
    import sqlite3

    from pricing_pipeline.infra.offline_sqlite import open_offline_sqlite

    engine, paths = open_offline_sqlite(tmp_path)
    engine.dispose()
    with sqlite3.connect(paths["pricing"]) as connection:
        existing_columns = {row[1] for row in connection.execute("PRAGMA table_info('MODEL_RUN')")}
        if "parent_model_run_id" in existing_columns:
            connection.execute("ALTER TABLE MODEL_RUN DROP COLUMN parent_model_run_id")

    upgraded, _paths = open_offline_sqlite(tmp_path)
    with upgraded.connect() as connection:
        model_run_columns = {
            row[1] for row in connection.exec_driver_sql("PRAGMA pricing.table_info('MODEL_RUN')")
        }

    assert "parent_model_run_id" in model_run_columns


def test_open_offline_sqlite_backfills_parent_model_run_id_from_package_lineage(
    tmp_path,
):
    import sqlite3

    from pricing_pipeline.infra.offline_sqlite import open_offline_sqlite

    engine, paths = open_offline_sqlite(tmp_path)
    engine.dispose()
    with sqlite3.connect(paths["pricing"]) as connection:
        connection.executemany(
            """
            INSERT INTO PRICING_RATE_PACKAGE (
                rate_package_id,
                parent_rate_package_id,
                model_id,
                model_name,
                package_version,
                base_rate,
                package_status,
                created_by
            ) VALUES (?, ?, 17, 'HOME_FREQ', ?, 1.0, 'PUBLISHED', 'test')
            """,
            [(41, None, 1), (42, 41, 2)],
        )
        connection.executemany(
            """
            INSERT INTO MODEL_RUN (
                model_run_id,
                model_id,
                model_version,
                export_id,
                manifest_id,
                rate_package_id,
                rating_workbook_path,
                rating_workbook_sha256,
                run_status,
                created_by
            ) VALUES (?, 17, 'v1', ?, 'manifest-1', ?, '/tmp/rating.xlsx',
                      ?, 'SUCCESS', 'test')
            """,
            [
                (501, "export-parent", 41, "a" * 64),
                (502, "export-child", 42, "b" * 64),
            ],
        )
        connection.execute("ALTER TABLE MODEL_RUN DROP COLUMN parent_model_run_id")

    upgraded, _paths = open_offline_sqlite(tmp_path)
    with upgraded.connect() as connection:
        parent_model_run_id = connection.exec_driver_sql(
            """
            SELECT parent_model_run_id
            FROM pricing.MODEL_RUN
            WHERE model_run_id = '502'
            """
        ).scalar_one()

    assert parent_model_run_id == "501"


def _create_legacy_effective_date_store(offline_sqlite, tmp_path):
    paths = offline_sqlite.offline_database_paths(tmp_path)
    legacy_engine = offline_sqlite.sqlite_engine_with_offline_schemas(paths)
    connection = legacy_engine.raw_connection()
    try:
        legacy_ddl = (
            (offline_sqlite.OFFLINE_DDL_DIR / "pricing.sql")
            .read_text(encoding="utf-8")
            .replace("effective_from TEXT,", "effective_from TEXT NOT NULL,")
            .replace(
                "effective_from_date TEXT,",
                "effective_from_date TEXT NOT NULL,",
            )
        )
        connection.executescript(legacy_ddl)
        connection.execute(
            """
            INSERT INTO pricing.PRICING_RATE_PACKAGE (
                rate_package_id,
                model_id,
                model_name,
                package_version,
                base_rate,
                effective_from_date,
                package_status,
                created_by
            ) VALUES (7, 1, 'LEGACY_MODEL', 1, 1.0, '2026-01-01', 'DRAFT', 'test')
            """
        )
        connection.execute(
            """
            INSERT INTO pricing.MODEL_RUN (
                model_run_id,
                model_id,
                model_version,
                export_id,
                manifest_id,
                rate_package_id,
                rating_workbook_path,
                rating_workbook_sha256,
                effective_from,
                created_by
            ) VALUES (
                'legacy-run', 1, 'v1', 'legacy-export', 'manifest-1', 7,
                'rating_tables.xlsx', :workbook_sha256, '2026-01-01', 'test'
            )
            """,
            {"workbook_sha256": "a" * 64},
        )
        connection.commit()
    finally:
        connection.close()
        legacy_engine.dispose()
    return paths


def test_open_offline_sqlite_relaxes_legacy_effective_date_constraints(tmp_path):
    from pricing_pipeline.infra import offline_sqlite

    _create_legacy_effective_date_store(offline_sqlite, tmp_path)

    upgraded, _paths = offline_sqlite.open_offline_sqlite(tmp_path)
    with upgraded.begin() as connection:
        model_run_columns = {
            row[1]: row
            for row in connection.exec_driver_sql("PRAGMA pricing.table_info('MODEL_RUN')")
        }
        package_columns = {
            row[1]: row
            for row in connection.exec_driver_sql(
                "PRAGMA pricing.table_info('PRICING_RATE_PACKAGE')"
            )
        }
        assert model_run_columns["effective_from"][3] == 0
        assert package_columns["effective_from_date"][3] == 0
        assert (
            connection.exec_driver_sql(
                "SELECT effective_from FROM pricing.MODEL_RUN WHERE model_run_id = 'legacy-run'"
            ).scalar_one()
            == "2026-01-01"
        )
        assert (
            connection.exec_driver_sql(
                """
            SELECT effective_from_date
            FROM pricing.PRICING_RATE_PACKAGE
            WHERE rate_package_id = 7
            """
            ).scalar_one()
            == "2026-01-01"
        )
        connection.exec_driver_sql(
            """
            INSERT INTO pricing.PRICING_RATE_PACKAGE (
                rate_package_id,
                model_id,
                model_name,
                package_version,
                base_rate,
                effective_from_date,
                package_status,
                created_by
            ) VALUES (8, 1, 'LEGACY_MODEL', 2, 1.0, NULL, 'DRAFT', 'test')
            """
        )
        connection.exec_driver_sql(
            """
            INSERT INTO pricing.MODEL_RUN (
                model_run_id,
                model_id,
                model_version,
                export_id,
                manifest_id,
                    rate_package_id,
                    rating_workbook_path,
                    rating_workbook_sha256,
                    effective_from,
                    created_by
                ) VALUES (
                    'nullable-run', 1, 'v2', 'nullable-export', 'manifest-2', 8,
                    'rating_tables.xlsx',
                    'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
                    NULL, 'test'
                )
            """
        )


def test_offline_nullability_rebuild_rolls_back_on_copy_failure(tmp_path):
    from pricing_pipeline.infra import offline_sqlite

    paths = _create_legacy_effective_date_store(offline_sqlite, tmp_path)
    engine = offline_sqlite.sqlite_engine_with_offline_schemas(paths)
    underlying = engine.raw_connection()

    class FailingConnection:
        def __getattr__(self, name):
            return getattr(underlying, name)

        def execute(self, statement, *args):
            if str(statement).startswith('INSERT INTO pricing."MODEL_RUN"'):
                raise RuntimeError("simulated copy interruption")
            return underlying.execute(statement, *args)

    failing_engine = SimpleNamespace(raw_connection=lambda: FailingConnection())
    try:
        with pytest.raises(RuntimeError, match="simulated copy interruption"):
            offline_sqlite.apply_offline_ddl(failing_engine)
    finally:
        engine.dispose()

    import sqlite3

    with sqlite3.connect(paths["pricing"]) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "MODEL_RUN" in tables
        assert "__offline_upgrade_model_run" not in tables
        assert connection.execute(
            "SELECT effective_from FROM MODEL_RUN WHERE model_run_id = 'legacy-run'"
        ).fetchone() == ("2026-01-01",)


def test_fremtpl_insert_rows_preserves_order_and_converts_missing_to_none():
    frame = fremtpl_frame(
        Area=[None],
        Density=[float("nan")],
        Region=[pd.NA],
    )

    rows = fremtpl_insert_rows(prepare_fremtpl_raw_frame(frame))

    assert rows == [(1, 0, 0.5, None, 6, 3, 45, 50, "B1", "Regular", None, None)]


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
        if sql.startswith("SELECT COUNT(*) FROM pricing.FREMTPL_RAW"):
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
    def __init__(
        self,
        *,
        existing_count=0,
        raw_connection=None,
        dialect_name="mssql",
        paramstyle=None,
        execution_options=None,
    ):
        self.begin_connection = FakeBeginConnection(existing_count)
        self.raw_connection_obj = raw_connection
        self._execution_options = execution_options or {}
        self.dialect = SimpleNamespace(name=dialect_name, paramstyle=paramstyle)

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


def test_bulk_insert_fremtpl_raw_uses_format_placeholders_for_pymssql():
    cursor = FakeCursor()
    raw_connection = FakeRawConnection(cursor)
    engine = FakeEngine(raw_connection=raw_connection, paramstyle="pyformat")

    bulk_insert_fremtpl_raw(engine, fremtpl_frame())

    assert (
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        in (cursor.executemany_calls[0][0])
    )


def test_bulk_insert_fremtpl_raw_uses_configured_schema_for_raw_cursor_sql():
    cursor = FakeCursor()
    raw_connection = FakeRawConnection(cursor)
    engine = FakeEngine(
        raw_connection=raw_connection,
        execution_options={
            "pricing_schema": "python_pricing",
            "pricing_staging_schema": "python_pricing_stg",
            "mlops_schema": "python_mlops",
        },
    )

    bulk_insert_fremtpl_raw(engine, fremtpl_frame(), replace=True)

    assert cursor.execute_calls == ["TRUNCATE TABLE python_pricing.FREMTPL_RAW"]
    assert cursor.executemany_calls[0][0].startswith("INSERT INTO python_pricing.FREMTPL_RAW")


def test_bulk_insert_fremtpl_raw_sqlite_replace_deletes_existing_rows(tmp_path):
    from sqlalchemy import text

    from pricing_pipeline.infra.offline_sqlite import open_offline_sqlite

    engine, _paths = open_offline_sqlite(tmp_path)
    bulk_insert_fremtpl_raw(engine, fremtpl_frame(IDpol=[1]))
    replacement = pd.concat(
        [fremtpl_frame(IDpol=[2]), fremtpl_frame(IDpol=[3])],
        ignore_index=True,
    )

    inserted = bulk_insert_fremtpl_raw(engine, replacement, replace=True)

    assert inserted == len(replacement)
    with engine.connect() as connection:
        ids = connection.execute(
            text("SELECT IDpol FROM pricing.FREMTPL_RAW ORDER BY IDpol")
        ).scalars().all()
    assert ids == [2, 3]


def test_bulk_insert_fremtpl_raw_non_sqlite_replace_uses_truncate():
    cursor = FakeCursor()
    raw_connection = FakeRawConnection(cursor)
    engine = FakeEngine(raw_connection=raw_connection, dialect_name="mssql")

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
    assert fremtpl.FREMTPL_EXPECTED_ROW_COUNT == 678_013
    engine = FakeEngine(existing_count=fremtpl.FREMTPL_EXPECTED_ROW_COUNT)
    monkeypatch.setattr(
        fremtpl,
        "fetch_fremtpl",
        lambda: pytest.fail("fetch_fremtpl should not run when raw rows exist"),
    )

    rows = load_fremtpl_raw(engine, replace=False)

    assert rows == fremtpl.FREMTPL_EXPECTED_ROW_COUNT
    assert engine.begin_connection.statements == [
        ("SELECT COUNT(*) FROM pricing.FREMTPL_RAW", None)
    ]


def test_load_fremtpl_raw_replaces_legacy_120_row_store(monkeypatch):
    events = []
    cursor = FakeCursor(events=events)
    raw_connection = FakeRawConnection(cursor)
    engine = FakeEngine(existing_count=120, raw_connection=raw_connection)
    source = fremtpl_frame()

    def fake_fetch_fremtpl():
        events.append("fetch")
        return source

    monkeypatch.setattr(
        fremtpl,
        "FREMTPL_EXPECTED_ROW_COUNT",
        len(source),
        raising=False,
    )
    monkeypatch.setattr(fremtpl, "fetch_fremtpl", fake_fetch_fremtpl)

    rows = load_fremtpl_raw(engine, replace=False)

    assert rows == len(source)
    assert events == ["fetch", "truncate", "executemany"]
    assert cursor.execute_calls == ["TRUNCATE TABLE pricing.FREMTPL_RAW"]


def test_load_fremtpl_raw_fetches_and_prepares_before_replace_truncate(monkeypatch):
    events = []
    cursor = FakeCursor(events=events)
    raw_connection = FakeRawConnection(cursor)
    engine = FakeEngine(existing_count=7, raw_connection=raw_connection)
    source = fremtpl_frame(IDpol=["1"], ClaimNb=["0"])

    def fake_fetch_fremtpl():
        events.append("fetch")
        return source

    monkeypatch.setattr(
        fremtpl,
        "FREMTPL_EXPECTED_ROW_COUNT",
        len(source),
        raising=False,
    )
    monkeypatch.setattr(fremtpl, "fetch_fremtpl", fake_fetch_fremtpl)
    rows = load_fremtpl_raw(engine, replace=True)

    assert rows == 1
    assert engine.begin_connection.statements == [
        ("SELECT COUNT(*) FROM pricing.FREMTPL_RAW", None),
    ]
    assert events == ["fetch", "truncate", "executemany"]
    assert cursor.execute_calls == ["TRUNCATE TABLE pricing.FREMTPL_RAW"]
    assert raw_connection.commits == 1
