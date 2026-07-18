import sqlite3
import re
from pathlib import Path

import pytest
from sqlalchemy import text

from pricing_pipeline.infra.offline_sqlite import (
    apply_offline_ddl,
    sqlite_engine_with_offline_schemas,
)


_CLEAN_VALIDATION_VIEWS = (
    "V_FINAL_MODEL_RELATIVITY",
    "V_MODEL_VALIDATION_SPLIT",
    "V_MODEL_VALIDATION_SUMMARY",
    "V_MODEL_VALIDATION_SPLIT_RELATIVITY",
    "V_CURRENT_DATASET_VALIDATION_SPLIT",
)


def _sql_server_view_columns(sql: str, view_name: str) -> list[str]:
    match = re.search(
        rf"CREATE OR ALTER VIEW pricing\.{re.escape(view_name)} AS\n"
        rf"SELECT\n(?P<select>.*?)\nFROM ",
        sql,
        re.S,
    )
    assert match is not None
    columns = []
    for line in match.group("select").splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped:
            continue
        alias = re.search(r"\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)$", stripped, re.I)
        columns.append(alias.group(1) if alias else stripped.split(".")[-1])
    return columns


def _foreign_key_contract(rows) -> set[tuple[str, tuple[tuple[str, str], ...]]]:
    grouped: dict[int, tuple[str, list[tuple[int, str, str]]]] = {}
    for row in rows:
        fk_id, sequence, parent_table, child_column, parent_column = row[:5]
        table, columns = grouped.setdefault(int(fk_id), (str(parent_table), []))
        assert table == parent_table
        columns.append((int(sequence), str(child_column), str(parent_column)))
    return {
        (table, tuple((child, parent) for _, child, parent in sorted(columns)))
        for table, columns in grouped.values()
    }


_MODEL_RUN_INSERT = """
    INSERT INTO MODEL_RUN (
        model_run_id,
        model_id,
        model_version,
        export_id,
        manifest_id,
        rate_package_id,
        rating_workbook_path,
        rating_workbook_sha256,
        candidate_artifact_format,
        candidate_superglm_git_sha,
        created_by
    )
    VALUES (?, 1, 'v1', ?, 'manifest-1', ?, 'rating.xlsx', ?, ?, ?, 'pytest')
"""


def _insert_model_run(
    connection: sqlite3.Connection,
    *,
    suffix: str,
    rate_package_id: int,
    artifact_format: str,
    superglm_git_sha: str | None,
) -> None:
    connection.execute(
        _MODEL_RUN_INSERT,
        (
            f"run-{suffix}",
            f"export-{suffix}",
            rate_package_id,
            "0" * 64,
            artifact_format,
            superglm_git_sha,
        ),
    )


def _assert_superglm_git_sha_constraint(pricing_path) -> None:
    invalid_v3_shas = (
        None,
        "A" * 40,
        "a" * 39,
        "g" * 40,
    )

    with sqlite3.connect(pricing_path) as connection:
        for index, git_sha in enumerate(invalid_v3_shas, start=1):
            with pytest.raises(sqlite3.IntegrityError):
                _insert_model_run(
                    connection,
                    suffix=f"invalid-{index}",
                    rate_package_id=index,
                    artifact_format="superglm-candidate-joblib-v3",
                    superglm_git_sha=git_sha,
                )

        _insert_model_run(
            connection,
            suffix="legacy",
            rate_package_id=100,
            artifact_format="superglm-candidate-joblib-v2",
            superglm_git_sha=None,
        )
        _insert_model_run(
            connection,
            suffix="valid-v3",
            rate_package_id=101,
            artifact_format="superglm-candidate-joblib-v3",
            superglm_git_sha="0123456789abcdef0123456789abcdef01234567",
        )


def test_offline_upgrade_adds_dataset_manifest_frame_evidence_columns(tmp_path):
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE pricing.DATASET_MANIFEST (
                    manifest_id TEXT NOT NULL PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    source_system TEXT,
                    data_as_of_date TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    pk_columns_json TEXT NOT NULL,
                    target_column TEXT,
                    weight_column TEXT,
                    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT NOT NULL
                )
                """
            )
        )

    apply_offline_ddl(engine)

    with engine.connect() as connection:
        columns = {
            row[1]: row[2]
            for row in connection.exec_driver_sql("PRAGMA pricing.table_info('DATASET_MANIFEST')")
        }
    assert columns["model_frame_sha256"] == "TEXT"
    assert columns["frame_hash_metadata_json"] == "TEXT"
    assert columns["exposure_column"] == "TEXT"
    assert columns["data_as_of_column"] == "TEXT"
    assert columns["offset_column"] == "TEXT"
    assert columns["offset_source_column"] == "TEXT"
    assert columns["offset_label"] == "TEXT"
    assert columns["export_weight_column"] == "TEXT"


def test_fresh_offline_dataset_manifest_requires_frame_evidence(tmp_path):
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    apply_offline_ddl(engine)

    with engine.connect() as connection:
        columns = {
            row[1]: bool(row[3])
            for row in connection.exec_driver_sql("PRAGMA pricing.table_info('DATASET_MANIFEST')")
        }
    assert columns["model_frame_sha256"] is True
    assert columns["frame_hash_metadata_json"] is True
    assert columns["offset_column"] is False
    assert columns["offset_source_column"] is False
    assert columns["offset_label"] is False
    assert columns["export_weight_column"] is False


def test_fresh_offline_model_run_enforces_superglm_git_sha_constraint(tmp_path):
    paths = {
        "pricing": tmp_path / "pricing.sqlite",
        "pricing_stg": tmp_path / "pricing_stg.sqlite",
        "mlops": tmp_path / "mlops.sqlite",
    }
    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    engine.dispose()

    _assert_superglm_git_sha_constraint(paths["pricing"])


def test_offline_upgrade_enforces_superglm_git_sha_constraint_on_added_column(tmp_path):
    paths = {
        "pricing": tmp_path / "pricing.sqlite",
        "pricing_stg": tmp_path / "pricing_stg.sqlite",
        "mlops": tmp_path / "mlops.sqlite",
    }
    engine = sqlite_engine_with_offline_schemas(paths)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE pricing.MODEL_RUN (
                    model_run_id TEXT PRIMARY KEY,
                    parent_model_run_id TEXT,
                    model_id INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    export_id TEXT NOT NULL,
                    manifest_id TEXT NOT NULL,
                    split_set_id TEXT,
                    rate_package_id INTEGER NOT NULL,
                    rating_workbook_path TEXT NOT NULL,
                    rating_workbook_sha256 TEXT NOT NULL,
                    candidate_artifact_format TEXT,
                    effective_from TEXT,
                    created_by TEXT NOT NULL
                )
                """
            )
        )

    apply_offline_ddl(engine)
    engine.dispose()

    _assert_superglm_git_sha_constraint(paths["pricing"])


def test_fresh_offline_clean_validation_schema_matches_remote_contract(tmp_path):
    paths = {
        "pricing": tmp_path / "pricing.sqlite",
        "pricing_stg": tmp_path / "pricing_stg.sqlite",
        "mlops": tmp_path / "mlops.sqlite",
    }
    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    migration = Path("db/migrations/V035__clean_validation_evidence_workflow.sql").read_text(
        encoding="utf-8"
    )

    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        package_columns = {
            row[1]: row[2]
            for row in connection.exec_driver_sql(
                "PRAGMA pricing.table_info('PRICING_RATE_PACKAGE')"
            )
        }
        model_run_columns = {
            row[1]: row[2]
            for row in connection.exec_driver_sql("PRAGMA pricing.table_info('MODEL_RUN')")
        }
        curve_columns = list(
            connection.exec_driver_sql("PRAGMA pricing.table_info('CV_SPLIT_CURVE_POINT')")
        )
        curve_foreign_keys = list(
            connection.exec_driver_sql("PRAGMA pricing.foreign_key_list('CV_SPLIT_CURVE_POINT')")
        )
        fold_metric_foreign_keys = list(
            connection.exec_driver_sql("PRAGMA pricing.foreign_key_list('CV_FOLD_METRIC')")
        )
        model_run_foreign_keys = list(
            connection.exec_driver_sql("PRAGMA pricing.foreign_key_list('MODEL_RUN')")
        )
        indexes = {
            row[0]: row[1]
            for row in connection.exec_driver_sql(
                "SELECT name, sql FROM pricing.sqlite_master WHERE type = 'index'"
            )
            if row[1] is not None
        }
        views = {
            row[0]: row[1]
            for row in connection.exec_driver_sql(
                "SELECT name, sql FROM pricing.sqlite_master WHERE type = 'view' ORDER BY name"
            )
        }
        sqlite_view_columns = {
            view_name: [
                row[1]
                for row in connection.exec_driver_sql(f"PRAGMA pricing.table_info('{view_name}')")
            ]
            for view_name in _CLEAN_VALIDATION_VIEWS
        }
        queried_view_columns = {
            view_name: list(
                connection.exec_driver_sql(f"SELECT * FROM pricing.{view_name} WHERE 0").keys()
            )
            for view_name in _CLEAN_VALIDATION_VIEWS
        }

    assert package_columns["build_fingerprint_sha256"] == "TEXT"
    for column in (
        "builder_source_sha256",
        "materialized_split_sha256",
        "runtime_sha256",
        "candidate_superglm_sha256",
        "validation_curve_status",
        "validation_curve_reason",
        "validation_source_model_run_id",
    ):
        assert model_run_columns[column] == "TEXT"
    assert [row[1] for row in curve_columns] == [
        "model_run_id",
        "split_set_id",
        "split_no",
        "term_name",
        "point_no",
        "point_kind",
        "x_numeric",
        "level_text",
        "eta_contribution",
        "relativity",
        "support_value",
        "reference_value",
        "reference_level",
    ]
    assert [row[5] for row in curve_columns[:5]] == [1, 2, 3, 4, 5]
    assert next(row for row in curve_columns if row[1] == "support_value")[3] == 0
    assert {(row[2], row[3], row[4]) for row in curve_foreign_keys} >= {
        ("MODEL_RUN", "model_run_id", "model_run_id"),
        ("CV_FOLD", "split_set_id", "split_set_id"),
        ("CV_FOLD", "split_no", "fold_no"),
    }
    assert _foreign_key_contract(fold_metric_foreign_keys) == {
        ("MODEL_RUN", (("model_run_id", "model_run_id"),)),
        (
            "CV_FOLD",
            (("split_set_id", "split_set_id"), ("fold_no", "fold_no")),
        ),
    }
    assert ("MODEL_RUN", "validation_source_model_run_id", "model_run_id") in {
        (row[2], row[3], row[4]) for row in model_run_foreign_keys
    }
    build_fingerprint_index_sql = " ".join(
        indexes["UX_PRICING_RATE_PACKAGE_MODEL_BUILD_FINGERPRINT"].split()
    )
    assert (
        "ON PRICING_RATE_PACKAGE(model_id, build_fingerprint_sha256) "
        "WHERE parent_rate_package_id IS NULL "
        "AND build_fingerprint_sha256 IS NOT NULL"
    ) in build_fingerprint_index_sql
    assert (
        "ON CV_SPLIT_CURVE_POINT(split_set_id, split_no)"
        in indexes["IX_CV_SPLIT_CURVE_POINT_SPLIT"]
    )
    assert (
        "ON MODEL_RUN(validation_source_model_run_id)" in indexes["IX_MODEL_RUN_VALIDATION_SOURCE"]
    )
    assert set(views) == set(_CLEAN_VALIDATION_VIEWS)
    for view_name, view_sql in views.items():
        assert "mlops." not in view_sql.lower()
        assert "pricing." not in view_sql.lower()
        assert sqlite_view_columns[view_name] == _sql_server_view_columns(migration, view_name)
        assert queried_view_columns[view_name] == sqlite_view_columns[view_name]


def test_fresh_offline_build_fingerprint_constraints_and_partial_uniqueness(tmp_path):
    paths = {
        "pricing": tmp_path / "pricing.sqlite",
        "pricing_stg": tmp_path / "pricing_stg.sqlite",
        "mlops": tmp_path / "mlops.sqlite",
    }
    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    engine.dispose()

    insert_sql = """
        INSERT INTO PRICING_RATE_PACKAGE (
            rate_package_id, parent_rate_package_id, model_id, model_name,
            package_version, base_rate, package_status,
            build_fingerprint_sha256, created_by
        ) VALUES (?, ?, ?, 'HOME_FREQ', ?, 1.0, 'DRAFT', ?, 'pytest')
    """
    with sqlite3.connect(paths["pricing"]) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(insert_sql, (1, None, 7, 1, None))
        connection.execute(insert_sql, (2, 1, 7, 2, None))
        connection.execute(insert_sql, (3, None, 7, 3, "a" * 64))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(insert_sql, (4, 1, 7, 4, "b" * 64))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(insert_sql, (5, None, 7, 5, "A" * 64))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(insert_sql, (6, None, 7, 6, "a" * 63))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(insert_sql, (7, None, 7, 7, "a" * 64))
        connection.execute(insert_sql, (8, None, 8, 1, "a" * 64))


def _insert_clean_validation_model_run(connection, **overrides) -> None:
    values = {
        "model_run_id": "run-valid",
        "rate_package_id": 101,
        "builder_source_sha256": None,
        "materialized_split_sha256": None,
        "runtime_sha256": None,
        "candidate_superglm_sha256": None,
        "validation_curve_status": None,
        "validation_curve_reason": None,
        "validation_source_model_run_id": None,
    }
    values.update(overrides)
    connection.execute(
        """
        INSERT INTO MODEL_RUN (
            model_run_id, model_id, model_version, export_id, manifest_id,
            rate_package_id, rating_workbook_path, rating_workbook_sha256,
            builder_source_sha256, materialized_split_sha256, runtime_sha256,
            candidate_superglm_sha256, validation_curve_status,
            validation_curve_reason, validation_source_model_run_id, created_by
        ) VALUES (
            :model_run_id, 7, 'v1', :model_run_id, 'manifest-1',
            :rate_package_id, 'rating.xlsx', :rating_workbook_sha256,
            :builder_source_sha256, :materialized_split_sha256, :runtime_sha256,
            :candidate_superglm_sha256, :validation_curve_status,
            :validation_curve_reason, :validation_source_model_run_id, 'pytest'
        )
        """,
        {**values, "rating_workbook_sha256": "0" * 64},
    )


def _replace_fold_metric_with_legacy_table(pricing_path: Path) -> None:
    with sqlite3.connect(pricing_path) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("PRAGMA legacy_alter_table=ON")
        connection.execute("ALTER TABLE CV_FOLD_METRIC RENAME TO CV_FOLD_METRIC_CURRENT")
        connection.execute("PRAGMA legacy_alter_table=OFF")
        connection.execute(
            """
            CREATE TABLE CV_FOLD_METRIC (
                model_run_id TEXT NOT NULL,
                split_set_id TEXT NOT NULL,
                fold_no INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                PRIMARY KEY (model_run_id, split_set_id, fold_no, metric_name)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO CV_FOLD_METRIC
            SELECT model_run_id, split_set_id, fold_no, metric_name, metric_value
            FROM CV_FOLD_METRIC_CURRENT
            """
        )
        connection.execute("DROP TABLE CV_FOLD_METRIC_CURRENT")


def test_fresh_offline_fold_metric_foreign_keys_reject_orphans(tmp_path):
    paths = {
        "pricing": tmp_path / "pricing.sqlite",
        "pricing_stg": tmp_path / "pricing_stg.sqlite",
        "mlops": tmp_path / "mlops.sqlite",
    }
    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    engine.dispose()

    insert_metric = """
        INSERT INTO CV_FOLD_METRIC (
            model_run_id, split_set_id, fold_no, metric_name, metric_value
        ) VALUES (?, ?, ?, ?, 1.0)
    """
    with sqlite3.connect(paths["pricing"]) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_clean_validation_model_run(
            connection,
            model_run_id="run-metric",
            rate_package_id=201,
        )
        connection.execute(
            "INSERT INTO CV_FOLD (split_set_id, fold_no, n_train, n_test) "
            "VALUES ('split-metric', 1, 80, 20)"
        )
        connection.execute(
            insert_metric,
            ("run-metric", "split-metric", 1, "deviance"),
        )
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            connection.execute(
                insert_metric,
                ("run-missing", "split-metric", 1, "deviance"),
            )
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            connection.execute(
                insert_metric,
                ("run-metric", "split-missing", 1, "deviance"),
            )


def test_offline_upgrade_adds_fold_metric_foreign_keys_and_preserves_rows(tmp_path):
    paths = {
        "pricing": tmp_path / "pricing.sqlite",
        "pricing_stg": tmp_path / "pricing_stg.sqlite",
        "mlops": tmp_path / "mlops.sqlite",
    }
    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    engine.dispose()
    with sqlite3.connect(paths["pricing"]) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_clean_validation_model_run(
            connection,
            model_run_id="run-legacy-metric",
            rate_package_id=202,
        )
        connection.execute(
            "INSERT INTO CV_FOLD (split_set_id, fold_no, n_train, n_test) "
            "VALUES ('split-legacy-metric', 1, 80, 20)"
        )
        connection.execute(
            """
            INSERT INTO CV_FOLD_METRIC (
                model_run_id, split_set_id, fold_no, metric_name, metric_value
            ) VALUES ('run-legacy-metric', 'split-legacy-metric', 1, 'deviance', 0.25)
            """
        )
    _replace_fold_metric_with_legacy_table(paths["pricing"])

    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    apply_offline_ddl(engine)
    with engine.connect() as connection:
        foreign_keys = list(
            connection.exec_driver_sql("PRAGMA pricing.foreign_key_list('CV_FOLD_METRIC')")
        )
        rows = connection.exec_driver_sql(
            """
            SELECT model_run_id, split_set_id, fold_no, metric_name, metric_value
            FROM pricing.CV_FOLD_METRIC
            """
        ).all()
        violations = connection.exec_driver_sql(
            "PRAGMA pricing.foreign_key_check('CV_FOLD_METRIC')"
        ).all()

    assert _foreign_key_contract(foreign_keys) == {
        ("MODEL_RUN", (("model_run_id", "model_run_id"),)),
        (
            "CV_FOLD",
            (("split_set_id", "split_set_id"), ("fold_no", "fold_no")),
        ),
    }
    assert rows == [("run-legacy-metric", "split-legacy-metric", 1, "deviance", 0.25)]
    assert violations == []


def test_offline_upgrade_rejects_orphan_fold_metric_evidence(tmp_path):
    paths = {
        "pricing": tmp_path / "pricing.sqlite",
        "pricing_stg": tmp_path / "pricing_stg.sqlite",
        "mlops": tmp_path / "mlops.sqlite",
    }
    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    engine.dispose()
    _replace_fold_metric_with_legacy_table(paths["pricing"])
    with sqlite3.connect(paths["pricing"]) as connection:
        connection.execute(
            """
            INSERT INTO CV_FOLD_METRIC (
                model_run_id, split_set_id, fold_no, metric_name, metric_value
            ) VALUES ('run-missing', 'split-missing', 99, 'deviance', 1.0)
            """
        )

    engine = sqlite_engine_with_offline_schemas(paths)
    with pytest.raises(
        RuntimeError,
        match=(
            "cannot add CV_FOLD_METRIC foreign keys: orphan evidence "
            ".*missing MODEL_RUN.*missing CV_FOLD"
        ),
    ):
        apply_offline_ddl(engine)
    engine.dispose()

    with sqlite3.connect(paths["pricing"]) as connection:
        assert connection.execute("SELECT COUNT(*) FROM CV_FOLD_METRIC").fetchone() == (1,)
        assert list(connection.execute("PRAGMA foreign_key_list('CV_FOLD_METRIC')")) == []


def test_fresh_offline_model_run_sha_curve_and_self_fk_constraints(tmp_path):
    paths = {
        "pricing": tmp_path / "pricing.sqlite",
        "pricing_stg": tmp_path / "pricing_stg.sqlite",
        "mlops": tmp_path / "mlops.sqlite",
    }
    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    engine.dispose()

    with sqlite3.connect(paths["pricing"]) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_clean_validation_model_run(connection, model_run_id="run-legacy")
        _insert_clean_validation_model_run(
            connection,
            model_run_id="run-complete",
            rate_package_id=102,
            builder_source_sha256="a" * 64,
            materialized_split_sha256="b" * 64,
            runtime_sha256="c" * 64,
            candidate_superglm_sha256="d" * 64,
            validation_curve_status="COMPLETE",
            validation_source_model_run_id="run-complete",
        )
        _insert_clean_validation_model_run(
            connection,
            model_run_id="run-unavailable",
            rate_package_id=103,
            validation_curve_status="UNAVAILABLE",
            validation_curve_reason="upstream diagnostic unavailable",
            validation_source_model_run_id="run-complete",
        )
        invalid_runs = (
            {
                "model_run_id": "run-bad-sha",
                "rate_package_id": 104,
                "builder_source_sha256": "A" * 64,
            },
            {
                "model_run_id": "run-complete-reason",
                "rate_package_id": 105,
                "validation_curve_status": "COMPLETE",
                "validation_curve_reason": "not allowed",
            },
            {
                "model_run_id": "run-unavailable-blank",
                "rate_package_id": 106,
                "validation_curve_status": "UNAVAILABLE",
                "validation_curve_reason": "   ",
            },
            {
                "model_run_id": "run-unknown-status",
                "rate_package_id": 107,
                "validation_curve_status": "PARTIAL",
            },
            {
                "model_run_id": "run-reason-without-status",
                "rate_package_id": 108,
                "validation_curve_reason": "orphaned reason",
            },
            {
                "model_run_id": "run-missing-source",
                "rate_package_id": 109,
                "validation_source_model_run_id": "run-does-not-exist",
            },
        )
        for values in invalid_runs:
            with pytest.raises(sqlite3.IntegrityError):
                _insert_clean_validation_model_run(connection, **values)


def _insert_curve_point(connection, **overrides) -> None:
    values = {
        "model_run_id": "run-curve",
        "split_set_id": "split-1",
        "split_no": 1,
        "term_name": "age",
        "point_no": 1,
        "point_kind": "NUMERIC",
        "x_numeric": 25.0,
        "level_text": None,
        "eta_contribution": 0.0,
        "relativity": 1.0,
        "support_value": 10.0,
        "reference_value": 25.0,
        "reference_level": None,
    }
    values.update(overrides)
    columns = ", ".join(values)
    placeholders = ", ".join(f":{column}" for column in values)
    connection.execute(
        f"INSERT INTO CV_SPLIT_CURVE_POINT ({columns}) VALUES ({placeholders})",
        values,
    )


def test_fresh_offline_curve_point_checks_and_foreign_keys(tmp_path):
    paths = {
        "pricing": tmp_path / "pricing.sqlite",
        "pricing_stg": tmp_path / "pricing_stg.sqlite",
        "mlops": tmp_path / "mlops.sqlite",
    }
    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    engine.dispose()

    with sqlite3.connect(paths["pricing"]) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_clean_validation_model_run(
            connection,
            model_run_id="run-curve",
            validation_curve_status="COMPLETE",
            validation_source_model_run_id="run-curve",
        )
        connection.execute(
            "INSERT INTO CV_FOLD (split_set_id, fold_no, n_train, n_test) "
            "VALUES ('split-1', 1, 80, 20)"
        )
        _insert_curve_point(connection)
        _insert_curve_point(
            connection,
            term_name="region",
            point_no=2,
            point_kind="LEVEL",
            x_numeric=None,
            level_text="London",
            reference_value=None,
            reference_level="London",
        )
        _insert_curve_point(connection, term_name="income", point_no=3, support_value=None)
        invalid_points = (
            {"term_name": " ", "point_no": 14},
            {"split_no": 0, "point_no": 4},
            {"point_no": 0},
            {"point_no": 5, "point_kind": "MYSTERY"},
            {"point_no": 6, "x_numeric": None},
            {"point_no": 7, "level_text": "unexpected"},
            {"point_no": 8, "reference_value": None},
            {"point_no": 9, "reference_level": "unexpected"},
            {"point_no": 10, "support_value": -1.0},
            {"point_no": 11, "relativity": -0.1},
            {"point_no": 12, "model_run_id": "missing-run"},
            {"point_no": 13, "split_set_id": "missing-split"},
        )
        for values in invalid_points:
            with pytest.raises(sqlite3.IntegrityError):
                _insert_curve_point(connection, **values)
