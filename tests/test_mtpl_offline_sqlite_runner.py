from __future__ import annotations

import sqlite3
from pathlib import Path


def test_mtpl_offline_sqlite_runner_populates_inspectable_tables(monkeypatch, tmp_path):
    from scripts import run_mtpl_frequency_offline_sqlite

    assert Path("db/offline_sqlite/pricing.sql").exists()
    assert Path("db/offline_sqlite/pricing_stg.sql").exists()
    assert Path("db/offline_sqlite/mlops.sql").exists()

    def fake_fit_export(
        frame,
        *,
        split_indices,
        output_dir,
        model_version,
        effective_from,
    ):
        artifact_dir = Path(output_dir) / f"{model_version}_{effective_from}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        workbook_path = artifact_dir / "rating_tables.xlsx"
        model_path = artifact_dir / "superglm_model.pkl"
        workbook_path.write_text("offline workbook placeholder", encoding="utf-8")
        model_path.write_bytes(b"offline model placeholder")
        return (
            workbook_path,
            model_path,
            {
                "row_count": float(len(frame)),
                "validation_fold_count": float(len(split_indices)),
            },
        )

    monkeypatch.setattr(
        run_mtpl_frequency_offline_sqlite,
        "fit_validate_export_rating_tables",
        fake_fit_export,
    )

    result = run_mtpl_frequency_offline_sqlite.run_mtpl_frequency_offline_sqlite(
        db_root=tmp_path / "offline",
        row_count=40,
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
    assert result["tables"] == {
        "pricing": {
            "FREMTPL_RAW": 40,
            "DATASET_MANIFEST": 1,
            "DATASET_COLUMN": 12,
            "CV_SPLIT_SET": 1,
            "CV_FOLD": 5,
            "CV_FOLD_METRIC": 0,
            "PRICING_MODEL": 1,
            "MODEL_RUN": 1,
            "PRICING_RATE_PACKAGE": 1,
        },
        "pricing_stg": {
            "STG_RATING_EXPORT": 0,
            "STG_RATE_CELL": 0,
            "STG_CELL_LEVEL": 0,
        },
        "mlops": {
            "MODEL_RUN_DATASET": 0,
            "MODEL_RUN_SPLIT_SET": 0,
            "MODEL_RUN_METRIC": 2,
        },
    }

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
        fold_rows = con.execute(
            "SELECT fold_no, n_train, n_test FROM CV_FOLD ORDER BY fold_no"
        ).fetchall()

    assert manifest == ("freMTPL2freq_model_frame", 40, "2026-06-05")
    assert split[0] == "MATERIALIZED"
    assert split[1] == 5
    assert Path(split[2]).exists()
    assert package == ("v1", "PUBLISHED", result["export_id"])
    assert len(fold_rows) == 5
    assert all(n_train > 0 and n_test > 0 for _, n_train, n_test in fold_rows)

    with sqlite3.connect(mlops_path) as con:
        metrics = con.execute(
            "SELECT metric_name FROM MODEL_RUN_METRIC ORDER BY metric_name"
        ).fetchall()

    assert metrics == [("row_count",), ("validation_fold_count",)]
