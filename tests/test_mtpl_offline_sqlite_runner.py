from __future__ import annotations

import sqlite3
from pathlib import Path


def test_offline_sqlite_ddl_contains_publication_receipt_metadata():
    pricing_sql = Path("db/offline_sqlite/pricing.sql").read_text(encoding="utf-8")
    staging_sql = Path("db/offline_sqlite/pricing_stg.sql").read_text(encoding="utf-8")

    assert "publication_receipt_json" in pricing_sql
    assert "publication_receipt_sha256" in pricing_sql
    assert "revision_metadata_json" in pricing_sql
    assert "term_metadata_json" in pricing_sql
    assert "publication_receipt_json" in staging_sql
    assert "STG_TERM_METADATA" in staging_sql


def test_mtpl_offline_sqlite_runner_populates_inspectable_tables(monkeypatch, tmp_path):
    from scripts import run_mtpl_frequency_offline_sqlite

    monkeypatch.setenv("PRICING_ENABLE_MLFLOW", "false")

    assert Path("db/offline_sqlite/pricing.sql").exists()
    assert Path("db/offline_sqlite/pricing_stg.sql").exists()
    assert Path("db/offline_sqlite/mlops.sql").exists()

    result = run_mtpl_frequency_offline_sqlite.run_mtpl_frequency_offline_sqlite(
        db_root=tmp_path / "offline",
        row_count=120,
        synthetic_source=True,
        effective_from="2026-06-05",
        created_by="unit-test",
        reset=True,
    )

    db_path = Path(result["db_paths"]["pricing"])
    pricing_stg_path = Path(result["db_paths"]["pricing_stg"])
    mlops_path = Path(result["db_paths"]["mlops"])
    assert db_path.exists()
    assert pricing_stg_path.exists()
    assert mlops_path.exists()
    assert {
        table_name: result["tables"]["pricing"][table_name]
        for table_name in [
            "FREMTPL_RAW",
            "DATASET_MANIFEST",
            "DATASET_COLUMN",
            "CV_SPLIT_SET",
            "CV_FOLD",
            "CV_FOLD_METRIC",
            "PRICING_MODEL",
            "MODEL_RUN",
            "PRICING_RATE_PACKAGE",
        ]
    } == {
        "FREMTPL_RAW": 120,
        "DATASET_MANIFEST": 1,
        "DATASET_COLUMN": 12,
        "CV_SPLIT_SET": 1,
        "CV_FOLD": 5,
        "CV_FOLD_METRIC": 0,
        "PRICING_MODEL": 1,
        "MODEL_RUN": 1,
        "PRICING_RATE_PACKAGE": 1,
    }
    assert result["tables"]["mlops"] == {
        "MODEL_RUN_DATASET": 0,
        "MODEL_RUN_SPLIT_SET": 0,
        "MODEL_RUN_METRIC": 7,
    }
    for table_name in [
        "PRICING_FEATURE",
        "PRICING_FEATURE_LEVEL_SET",
        "PRICING_FEATURE_LEVEL",
        "PRICING_TERM",
        "PRICING_TERM_FEATURE",
        "PRICING_RATE_CELL",
        "PRICING_RATE_CELL_LEVEL",
        "PRICING_COMPILED_RATE_CELL",
        "PRICING_COMPILED_1D_RATE_BAND",
    ]:
        assert result["tables"]["pricing"][table_name] > 0
    assert result["tables"]["pricing_stg"]["STG_RATING_EXPORT"] == 1
    assert result["tables"]["pricing_stg"]["STG_RATE_CELL"] > 0
    assert result["tables"]["pricing_stg"]["STG_CELL_LEVEL"] > 0

    with sqlite3.connect(db_path) as con:
        manifest = con.execute(
            "SELECT dataset_name, row_count, data_as_of_date FROM DATASET_MANIFEST"
        ).fetchone()
        split = con.execute(
            "SELECT split_mode, fold_count, artifact_uri FROM CV_SPLIT_SET"
        ).fetchone()
        package = con.execute(
            "SELECT model_version, package_status, source_export_id FROM PRICING_RATE_PACKAGE"
        ).fetchone()
        rate_cells = con.execute(
            """
            SELECT t.term_name, c.cell_key_text, c.multiplier
            FROM PRICING_RATE_CELL c
            JOIN PRICING_TERM t ON t.term_id = c.term_id
            ORDER BY t.sequence_no, c.cell_id
            LIMIT 5
            """
        ).fetchall()
        compiled_bands = con.execute(
            """
            SELECT term_name, feature_name, level_code, multiplier
            FROM PRICING_COMPILED_1D_RATE_BAND
            ORDER BY term_name, sort_order
            LIMIT 5
            """
        ).fetchall()
        fold_rows = con.execute(
            "SELECT fold_no, n_train, n_test FROM CV_FOLD ORDER BY fold_no"
        ).fetchall()

    assert manifest == ("freMTPL2freq_model_frame", 120, "2026-06-05")
    assert split[0] == "MATERIALIZED"
    assert split[1] == 5
    assert Path(split[2]).exists()
    assert package == ("v1", "PUBLISHED", result["export_id"])
    assert rate_cells
    assert all(multiplier > 0 for _, _, multiplier in rate_cells)
    assert compiled_bands
    assert all(multiplier > 0 for _, _, _, multiplier in compiled_bands)
    assert len(fold_rows) == 5
    assert all(n_train > 0 and n_test > 0 for _, n_train, n_test in fold_rows)

    with sqlite3.connect(mlops_path) as con:
        metrics = con.execute(
            "SELECT metric_name FROM MODEL_RUN_METRIC ORDER BY metric_name"
        ).fetchall()

    assert metrics == [
        ("claim_count_sum",),
        ("converged",),
        ("deviance",),
        ("exposure_sum",),
        ("n_iter",),
        ("row_count",),
        ("validation_fold_count",),
    ]


def test_mtpl_offline_sqlite_runner_uses_full_fremtpl_fetch_by_default(
    monkeypatch,
    tmp_path,
):
    from scripts import run_mtpl_frequency_offline_sqlite

    full_frame = run_mtpl_frequency_offline_sqlite.fre_mtpl_like_raw_frame(7)
    calls = []
    monkeypatch.setattr(
        run_mtpl_frequency_offline_sqlite,
        "fetch_fremtpl",
        lambda: calls.append("fetch") or full_frame,
    )

    engine = run_mtpl_frequency_offline_sqlite.sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    run_mtpl_frequency_offline_sqlite.apply_offline_ddl(engine)

    seeded = run_mtpl_frequency_offline_sqlite.seed_fremtpl_raw(engine)

    assert seeded == 7
    assert calls == ["fetch"]

    with sqlite3.connect(tmp_path / "pricing.sqlite") as con:
        columns = {
            row[1]: {"not_null": bool(row[3]), "primary_key_order": int(row[5])}
            for row in con.execute("PRAGMA table_info(FREMTPL_RAW)").fetchall()
        }

    assert columns["IDpol"] == {"not_null": True, "primary_key_order": 1}
    assert columns["ClaimNb"]["not_null"]
    assert columns["Exposure"]["not_null"]
