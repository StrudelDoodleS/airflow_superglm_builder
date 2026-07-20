import pytest
from sqlalchemy import text

from pricing_pipeline.infra.offline_sqlite import (
    apply_offline_ddl,
    sqlite_engine_with_offline_schemas,
)


def test_offline_views_expose_fold_metrics_and_final_relativity(tmp_path):
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    apply_offline_ddl(engine)

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_MODEL (
                    model_id, model_name, model_label, target_name,
                    model_type, model_status, created_by
                ) VALUES (
                    17, 'HOME_FREQ', 'Home frequency', 'claim_count',
                    'superglm_poisson', 'ACTIVE', 'pytest'
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.DATASET_MANIFEST (
                    manifest_id, dataset_name, source_system, data_as_of_date,
                    row_count, pk_columns_json, target_column,
                    model_frame_sha256, frame_hash_metadata_json, created_by
                ) VALUES (
                    'manifest-1', 'home_frame', 'pricing_sql', '2026-06-30',
                    20, '["policy_id"]', 'claim_count',
                    :frame_sha, '{}', 'pytest'
                )
                """
            ),
            {"frame_sha": "a" * 64},
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.CV_SPLIT_SET (
                    split_set_id, manifest_id, split_mode, splitter_class,
                    splitter_params_json, row_order_sha256, row_count,
                    fold_count, created_by
                ) VALUES (
                    'split-1', 'manifest-1', 'MATERIALIZED', 'ColumnKFold',
                    '{"column":"fold"}', :row_sha, 20, 2, 'pytest'
                )
                """
            ),
            {"row_sha": "b" * 64},
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.CV_FOLD (split_set_id, fold_no, n_train, n_test)
                VALUES ('split-1', 1, 10, 10), ('split-1', 2, 10, 10)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_RATE_PACKAGE (
                    rate_package_id, model_id, model_name, model_version,
                    package_version, base_rate, package_status, created_by
                ) VALUES (
                    71, 17, 'HOME_FREQ', 'v7', 3, 0.12, 'DRAFT', 'pytest'
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_RUN (
                    model_run_id, model_id, model_version, export_id,
                    manifest_id, split_set_id, rate_package_id, model_name,
                    rating_workbook_path, rating_workbook_sha256,
                    run_status, created_by
                ) VALUES (
                    'run-1', 17, 'v7', 'export-1',
                    'manifest-1', 'split-1', 71, 'HOME_FREQ',
                    '/tmp/rating.xlsx', :workbook_sha,
                    'SUCCESS', 'pytest'
                )
                """
            ),
            {"workbook_sha": "c" * 64},
        )
        metric_rows = [
            {"fold": fold, "name": name, "value": value}
            for fold, values in (
                (1, {"deviance": 1.0, "nll": 2.0, "gini": 0.4}),
                (2, {"deviance": 3.0, "nll": 4.0, "gini": 0.6}),
            )
            for name, value in values.items()
        ]
        connection.execute(
            text(
                """
                INSERT INTO pricing.CV_FOLD_METRIC (
                    model_run_id, split_set_id, fold_no, metric_name, metric_value
                ) VALUES (
                    'run-1', 'split-1', :fold, :name, :value
                )
                """
            ),
            metric_rows,
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_COMPILED_RATE_CELL (
                    rate_package_id, term_id, cell_key_digest, term_name,
                    term_type, sequence_no, cell_key_text, multiplier,
                    log_coefficient, exposure_weight, record_count,
                    is_default, is_reference
                ) VALUES (
                    71, 301, 'digest-1', 'Area', 'categorical', 1,
                    'Area=Urban', 1.2, 0.1823215568, 120.0, 40, 0, 0
                )
                """
            )
        )

    with engine.connect() as connection:
        splits = connection.execute(
            text(
                """
                SELECT validation_split_no, deviance, nll, gini
                FROM pricing.V_MODEL_VALIDATION_SPLIT
                ORDER BY validation_split_no
                """
            )
        ).mappings().all()
        summary = connection.execute(
            text("SELECT * FROM pricing.V_MODEL_VALIDATION_SUMMARY")
        ).mappings().one()
        relativity = connection.execute(
            text(
                """
                SELECT term_name, level_value, model_fit_scope
                FROM pricing.V_FINAL_MODEL_RELATIVITY
                """
            )
        ).mappings().one()

    assert [dict(row) for row in splits] == [
        {"validation_split_no": 1, "deviance": 1.0, "nll": 2.0, "gini": 0.4},
        {"validation_split_no": 2, "deviance": 3.0, "nll": 4.0, "gini": 0.6},
    ]
    assert summary["recorded_split_count"] == 2
    assert summary["mean_deviance"] == 2.0
    assert summary["std_deviance"] == 1.0
    assert summary["mean_nll"] == 3.0
    assert summary["std_nll"] == 1.0
    assert summary["mean_gini"] == 0.5
    assert summary["std_gini"] == pytest.approx(0.1)
    assert summary["oof_coverage"] == 1.0
    assert dict(relativity) == {
        "term_name": "Area",
        "level_value": "Urban",
        "model_fit_scope": "PACKAGE_FINAL_MODEL",
    }

    apply_offline_ddl(engine)
    with engine.connect() as connection:
        assert connection.execute(
            text("SELECT COUNT(*) FROM pricing.V_MODEL_VALIDATION_SPLIT")
        ).scalar_one() == 2


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
