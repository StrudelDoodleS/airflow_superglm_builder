from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

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
                    manifest_id, manifest_signature_sha256, dataset_name,
                    source_system, data_as_of_date, data_as_of_column,
                    row_count, pk_columns_json, target_column,
                    model_frame_sha256, frame_hash_metadata_json, created_by
                ) VALUES (
                    'manifest-1', :manifest_signature, 'home_frame',
                        'pricing_sql', '2026-06-30', 'snapshot_date',
                        20, '["policy_id"]', 'claim_count',
                        :frame_sha, :frame_metadata, 'pytest'
                )
                """
            ),
            {
                "manifest_signature": "d" * 64,
                "frame_sha": "a" * 64,
                "frame_metadata": '{"frame_hash":{"format_version":1}}',
            },
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
                    71, 17, 'HOME_FREQ', 'v7', 3, 0.12, 'PUBLISHED', 'pytest'
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_RUN (
                    model_run_id, model_id, model_version, export_id,
                    model_kind, model_equivalence_sha256,
                    manifest_id, split_set_id, rate_package_id, model_name,
                    rating_workbook_path, rating_workbook_sha256,
                    run_status, created_by
                ) VALUES (
                    'run-1', 17, 'v7', 'export-1',
                    'ROUTINE_EDIT', :equivalence_sha,
                    'manifest-1', 'split-1', 71, 'HOME_FREQ',
                    '/tmp/rating.xlsx', :workbook_sha,
                    'SUCCESS', 'pytest'
                )
                """
            ),
            {
                "workbook_sha": "c" * 64,
                "equivalence_sha": "e" * 64,
            },
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
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT (
                    deployment_id, model_id, rate_package_id, deployment_slot,
                    effective_from_ts, effective_to_ts, deployed_by,
                    deployment_note
                ) VALUES
                    (
                        700, 17, 71, 'HOME_FREQ_UAT',
                        '2026-07-01 08:00:00', '2026-07-02 08:00:00',
                        'previous-deployer', 'superseded deployment'
                    ),
                    (
                        701, 17, 71, 'HOME_FREQ_UAT',
                        '2026-07-02 08:00:00', NULL,
                        'current-deployer', 'approved for UAT'
                    )
                """
            )
        )

    with engine.connect() as connection:
        splits = (
            connection.execute(
                text(
                    """
                SELECT validation_split_no, deviance, nll, gini
                FROM pricing.V_MODEL_VALIDATION_SPLIT
                ORDER BY validation_split_no
                """
                )
            )
            .mappings()
            .all()
        )
        summary = (
            connection.execute(text("SELECT * FROM pricing.V_MODEL_VALIDATION_SUMMARY"))
            .mappings()
            .one()
        )
        candidate_relativity = (
            connection.execute(
                text(
                    """
                SELECT
                    term_name,
                    level_value,
                    model_fit_scope,
                    model_kind,
                    model_equivalence_sha256,
                    manifest_id,
                    manifest_signature_sha256,
                    dataset_name,
                    source_system,
                    data_as_of_date,
                    data_as_of_column,
                    dataset_row_count,
                    pk_columns_json,
                    dataset_target_column,
                    model_frame_sha256,
                    frame_hash_metadata_json,
                    validation_split_set_id
                FROM pricing.V_MODEL_CANDIDATE_RELATIVITY
                """
                )
            )
            .mappings()
            .one()
        )
        published_relativity = (
            connection.execute(
                text(
                    """
                    SELECT
                        term_name,
                        level_value,
                        model_fit_scope,
                        model_kind,
                        model_equivalence_sha256,
                        manifest_id,
                        manifest_signature_sha256,
                        dataset_name,
                        source_system,
                        data_as_of_date,
                        data_as_of_column,
                        dataset_row_count,
                        pk_columns_json,
                        dataset_target_column,
                        model_frame_sha256,
                        frame_hash_metadata_json,
                        validation_split_set_id
                    FROM pricing.V_PUBLISHED_MODEL_RELATIVITY
                    """
                )
            )
            .mappings()
            .one()
        )
        deployed_relativity = (
            connection.execute(
                text(
                    """
                    SELECT
                        deployment_id,
                        deployment_slot,
                        deployment_effective_from_ts,
                        deployment_effective_to_ts,
                        deployed_by,
                        deployment_note,
                        model_kind,
                        model_equivalence_sha256,
                        manifest_id,
                        manifest_signature_sha256,
                        dataset_name,
                        data_as_of_date,
                        data_as_of_column,
                        model_frame_sha256,
                        validation_split_set_id,
                        term_name,
                        level_value
                    FROM pricing.V_CURRENT_DEPLOYED_RELATIVITY
                    """
                )
            )
            .mappings()
            .one()
        )
        lineage_check = (
            connection.execute(text("SELECT * FROM pricing.V_MODEL_LINEAGE_REDUNDANCY_CHECK"))
            .mappings()
            .one()
        )

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
    expected_candidate_relativity = {
        "term_name": "Area",
        "level_value": "Urban",
        "model_fit_scope": "PACKAGE_FINAL_MODEL",
        "model_kind": "ROUTINE_EDIT",
        "model_equivalence_sha256": "e" * 64,
        "manifest_id": "manifest-1",
        "manifest_signature_sha256": "d" * 64,
        "dataset_name": "home_frame",
        "source_system": "pricing_sql",
        "data_as_of_date": "2026-06-30",
        "data_as_of_column": "snapshot_date",
        "dataset_row_count": 20,
        "pk_columns_json": '["policy_id"]',
        "dataset_target_column": "claim_count",
        "model_frame_sha256": "a" * 64,
        "frame_hash_metadata_json": '{"frame_hash":{"format_version":1}}',
        "validation_split_set_id": "split-1",
    }
    assert dict(candidate_relativity) == expected_candidate_relativity
    assert dict(published_relativity) == expected_candidate_relativity
    assert dict(deployed_relativity) == {
        "deployment_id": 701,
        "deployment_slot": "HOME_FREQ_UAT",
        "deployment_effective_from_ts": "2026-07-02 08:00:00",
        "deployment_effective_to_ts": None,
        "deployed_by": "current-deployer",
        "deployment_note": "approved for UAT",
        "model_kind": "ROUTINE_EDIT",
        "model_equivalence_sha256": "e" * 64,
        "manifest_id": "manifest-1",
        "manifest_signature_sha256": "d" * 64,
        "dataset_name": "home_frame",
        "data_as_of_date": "2026-06-30",
        "data_as_of_column": "snapshot_date",
        "model_frame_sha256": "a" * 64,
        "validation_split_set_id": "split-1",
        "term_name": "Area",
        "level_value": "Urban",
    }
    assert lineage_check["redundancy_status"] == "OK"

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT (
                    model_id, rate_package_id, deployment_slot, deployed_by
                ) VALUES (17, 71, 'HOME_FREQ_UAT', 'conflicting-deployer')
                """
            )
        )

    with (
        pytest.raises(IntegrityError, match="must exist, match model_id"),
        engine.begin() as connection,
    ):
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT (
                    model_id, rate_package_id, deployment_slot, deployed_by
                ) VALUES (18, 71, 'MISMATCHED_MODEL_UAT', 'invalid-deployer')
                """
            )
        )

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE pricing.PRICING_RATE_PACKAGE
                SET package_status = 'DRAFT'
                WHERE rate_package_id = 71
                """
            )
        )
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT COUNT(*) FROM pricing.V_CURRENT_DEPLOYED_RELATIVITY")
            ).scalar_one()
            == 0
        )
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE pricing.PRICING_RATE_PACKAGE
                SET package_status = 'PUBLISHED'
                WHERE rate_package_id = 71
                """
            )
        )

    apply_offline_ddl(engine)
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT COUNT(*) FROM pricing.V_MODEL_VALIDATION_SPLIT")
            ).scalar_one()
            == 2
        )


def test_offline_pricing_views_remain_valid_when_database_is_opened_directly(tmp_path):
    import sqlite3

    pricing_path = tmp_path / "pricing.sqlite"
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": pricing_path,
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    apply_offline_ddl(engine)
    engine.dispose()

    with sqlite3.connect(pricing_path) as connection:
        views = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'view'")
        }
        deployment_foreign_keys = {
            (row[3], row[2], row[4])
            for row in connection.execute("PRAGMA foreign_key_list('PRICING_MODEL_DEPLOYMENT')")
        }
        connection.execute("SELECT COUNT(*) FROM V_MODEL_VALIDATION_SPLIT").fetchone()
        connection.execute("SELECT COUNT(*) FROM V_MODEL_CANDIDATE_RELATIVITY").fetchone()
        connection.execute("SELECT COUNT(*) FROM V_PUBLISHED_MODEL_RELATIVITY").fetchone()
        connection.execute("SELECT COUNT(*) FROM V_CURRENT_DEPLOYED_RELATIVITY").fetchone()

    assert "V_MODEL_VALIDATION_SPLIT" in views
    assert "V_MODEL_CANDIDATE_RELATIVITY" in views
    assert "V_PUBLISHED_MODEL_RELATIVITY" in views
    assert "V_CURRENT_DEPLOYED_RELATIVITY" in views
    assert "V_DATASET_MANIFEST_REDUNDANCY_CHECK" not in views
    assert "V_MODEL_EQUIVALENCE_REDUNDANCY_CHECK" not in views
    assert deployment_foreign_keys == {
        ("model_id", "PRICING_MODEL", "model_id"),
        ("rate_package_id", "PRICING_RATE_PACKAGE", "rate_package_id"),
    }


def test_offline_monitoring_evidence_freezes_deployment_lineage(tmp_path):
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
                ) VALUES
                    (17, 'HOME_FREQ', 'Home frequency', 'claim_count',
                     'superglm_poisson', 'ACTIVE', 'pytest'),
                    (18, 'HOME_FREQ_V2', 'Home frequency v2', 'claim_count',
                     'superglm_poisson', 'ACTIVE', 'pytest')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.DATASET_MANIFEST (
                    manifest_id, manifest_signature_sha256, dataset_name,
                    source_system, data_as_of_date, row_count, pk_columns_json,
                    target_column, model_frame_sha256, frame_hash_metadata_json,
                    created_by
                ) VALUES (
                    'manifest-1', :manifest_signature, 'home_frame',
                    'pricing_sql', '2026-06-30', 20, '["policy_id"]',
                    'claim_count', :frame_sha, :frame_metadata,
                    'pytest'
                )
                """
            ),
            {
                "manifest_signature": "a" * 64,
                "frame_sha": "b" * 64,
                "frame_metadata": '{"frame_hash":{"format_version":1}}',
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.DATASET_MANIFEST (
                    manifest_id, manifest_signature_sha256, dataset_name,
                    source_system, data_as_of_date, row_count, pk_columns_json,
                    target_column, model_frame_sha256, frame_hash_metadata_json,
                    created_by
                ) VALUES (
                    'manifest-unreferenced', :manifest_signature, 'home_frame_v2',
                    'pricing_sql', '2026-07-01', 20, '["policy_id"]',
                    'claim_count', :frame_sha, :frame_metadata, 'pytest'
                )
                """
            ),
            {
                "manifest_signature": "3" * 64,
                "frame_sha": "4" * 64,
                "frame_metadata": '{"frame_hash":{"format_version":1}}',
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_RATE_PACKAGE (
                    rate_package_id, model_id, model_name, model_version,
                    package_version, base_rate, package_status, created_by
                ) VALUES
                    (71, 17, 'HOME_FREQ', 'v1', 1, 0.12, 'PUBLISHED', 'pytest'),
                    (72, 18, 'HOME_FREQ_V2', 'v2', 1, 0.12, 'PUBLISHED', 'pytest')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_RUN (
                    model_run_id, model_id, model_version, export_id, model_kind,
                    manifest_id, rate_package_id, model_name, rating_workbook_path,
                    rating_workbook_sha256, run_status, created_by
                ) VALUES (
                    'run-1', 17, 'v1', 'export-1', 'RAW', 'manifest-1', 71,
                    'HOME_FREQ', '/tmp/rating.xlsx', :workbook_sha, 'SUCCESS', 'pytest'
                )
                """
            ),
            {"workbook_sha": "c" * 64},
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT (
                    deployment_id, model_id, rate_package_id, deployment_slot,
                    effective_from_ts, deployed_by
                ) VALUES
                    (701, 17, 71, 'HOME_FREQ_PROD', '2026-07-01 08:00:00', 'pytest'),
                    (702, 17, 71, 'HOME_FREQ_UAT', '2026-07-01 08:00:00', 'pytest')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_FIT_CONTRACT (
                    fit_contract_id, baseline_model_run_id, model_id,
                    rate_package_id, contract_schema_version, contract_sha256,
                    structure_sha256, contract_json, superglm_version, created_by
                ) VALUES (
                    'contract-1', 'run-1', 17, 71, 1, :contract_sha,
                    :structure_sha, '{}', '0.26.0', 'pytest'
                )
                """
            ),
            {"contract_sha": "d" * 64, "structure_sha": "e" * 64},
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_MONITOR_RUN (
                    monitor_run_id, fit_contract_id, baseline_deployment_id,
                    model_id, rate_package_id, manifest_id, component_role,
                    variant_code, run_signature_sha256, run_status,
                    invariant_status, invariant_evidence_sha256,
                    invariant_evidence_json, model_frame_sha256,
                    fit_configuration_json, result_evidence_sha256, created_by
                ) VALUES (
                    'monitor-1', 'contract-1', 701, 17, 71, 'manifest-1',
                    'FREQUENCY', 'STATIC_SCORE', :signature_sha, 'SUCCESS',
                    'VERIFIED', :invariant_sha, :invariant_json,
                    :frame_sha, '{}', :result_sha, 'pytest'
                )
                """
            ),
            {
                "signature_sha": "f" * 64,
                "invariant_sha": "1" * 64,
                "invariant_json": '{"status":"VERIFIED"}',
                "frame_sha": "b" * 64,
                "result_sha": "2" * 64,
            },
        )

    for statement in (
        (
            "UPDATE pricing.PRICING_MODEL_DEPLOYMENT "
            "SET deployment_slot = 'RELABELLED' WHERE deployment_id = 701"
        ),
        (
            "UPDATE pricing.PRICING_MODEL_DEPLOYMENT "
            "SET model_id = 18, rate_package_id = 72 WHERE deployment_id = 701"
        ),
        (
            "UPDATE pricing.PRICING_MODEL_DEPLOYMENT "
            "SET effective_from_ts = '2026-07-02 08:00:00' WHERE deployment_id = 701"
        ),
        "DELETE FROM pricing.PRICING_MODEL_DEPLOYMENT WHERE deployment_id = 701",
        (
            "UPDATE pricing.DATASET_MANIFEST "
            "SET dataset_name = 'RELABELLED' WHERE manifest_id = 'manifest-1'"
        ),
        "DELETE FROM pricing.DATASET_MANIFEST WHERE manifest_id = 'manifest-1'",
    ):
        with (
            pytest.raises(IntegrityError, match="monitoring evidence"),
            engine.begin() as connection,
        ):
            connection.execute(text(statement))

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE pricing.PRICING_MODEL_DEPLOYMENT
                SET effective_to_ts = '2026-07-03 08:00:00'
                WHERE deployment_id = 701
                """
            )
        )
        connection.execute(
            text(
                """
                UPDATE pricing.PRICING_MODEL_DEPLOYMENT
                SET model_id = 18, rate_package_id = 72, deployment_slot = 'HOME_FREQ_V2_UAT'
                WHERE deployment_id = 702
                """
            )
        )
        connection.execute(
            text("DELETE FROM pricing.PRICING_MODEL_DEPLOYMENT WHERE deployment_id = 702")
        )
        connection.execute(
            text(
                """
                UPDATE pricing.DATASET_MANIFEST
                SET dataset_name = 'home_frame_v2_renamed'
                WHERE manifest_id = 'manifest-unreferenced'
                """
            )
        )
        connection.execute(
            text("DELETE FROM pricing.DATASET_MANIFEST WHERE manifest_id = 'manifest-unreferenced'")
        )

    with engine.connect() as connection:
        assert (
            connection.execute(
                text(
                    "SELECT effective_to_ts FROM pricing.PRICING_MODEL_DEPLOYMENT "
                    "WHERE deployment_id = 701"
                )
            ).scalar_one()
            == "2026-07-03 08:00:00"
        )
        assert (
            connection.execute(
                text(
                    "SELECT COUNT(*) FROM pricing.PRICING_MODEL_DEPLOYMENT "
                    "WHERE deployment_id = 702"
                )
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                text(
                    "SELECT COUNT(*) FROM pricing.DATASET_MANIFEST "
                    "WHERE manifest_id = 'manifest-unreferenced'"
                )
            ).scalar_one()
            == 0
        )


def test_offline_sqlite_enforces_declared_foreign_keys(tmp_path):
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    apply_offline_ddl(engine)

    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_MONITOR_TERM (
                    monitor_run_id, term_name, term_kind, sequence_no,
                    term_structure_sha256, term_metadata_json
                ) VALUES (
                    'missing-run', 'x', 'numeric', 1, :digest, '{}'
                )
                """
            ),
            {"digest": "a" * 64},
        )


def test_offline_sqlite_freezes_canonical_monitoring_variants(tmp_path):
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    apply_offline_ddl(engine)

    for statement in (
        (
            "UPDATE pricing.MODEL_MONITOR_VARIANT "
            "SET variant_label = 'Drifted label' WHERE variant_code = 'STATIC_SCORE'"
        ),
        "DELETE FROM pricing.MODEL_MONITOR_VARIANT WHERE variant_code = 'STATIC_SCORE'",
    ):
        with (
            pytest.raises(IntegrityError, match="monitoring variant policy is immutable"),
            engine.begin() as connection,
        ):
            connection.execute(text(statement))

    raw_connection = engine.raw_connection()
    try:
        raw_connection.execute(
            "DROP TRIGGER IF EXISTS pricing.TR_MODEL_MONITOR_VARIANT_IMMUTABLE_UPDATE"
        )
        raw_connection.execute(
            "UPDATE pricing.MODEL_MONITOR_VARIANT "
            "SET variant_label = 'Drifted label' WHERE variant_code = 'STATIC_SCORE'"
        )
        raw_connection.commit()
    finally:
        raw_connection.close()

    with pytest.raises(RuntimeError, match="canonical monitoring variant policy"):
        apply_offline_ddl(engine)


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
    assert columns["manifest_signature_sha256"] == "TEXT"
    assert columns["frame_hash_metadata_json"] == "TEXT"
    assert columns["exposure_column"] == "TEXT"
    assert columns["data_as_of_column"] == "TEXT"
    assert columns["offset_column"] == "TEXT"
    assert columns["offset_source_column"] == "TEXT"
    assert columns["offset_label"] == "TEXT"
    assert columns["export_weight_column"] == "TEXT"


def test_offline_upgrade_extends_existing_model_kind_check_for_manual_edits(tmp_path):
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    legacy_pricing_sql = (
        Path("db/offline_sqlite/pricing.sql")
        .read_text(encoding="utf-8")
        .replace(", 'MANUAL_EDIT'", "")
    )
    raw_connection = engine.raw_connection()
    try:
        raw_connection.executescript(legacy_pricing_sql)
        raw_connection.executescript(
            """
            INSERT INTO pricing.PRICING_MODEL (
                model_id, model_name, model_label, target_name,
                model_type, model_status, created_by
            ) VALUES (
                17, 'TEST_MODEL', 'Test model', 'claim_count',
                'superglm_poisson', 'ACTIVE', 'pytest'
            );

            INSERT INTO pricing.DATASET_MANIFEST (
                manifest_id, manifest_signature_sha256, dataset_name,
                source_system, data_as_of_date, row_count, pk_columns_json,
                target_column, model_frame_sha256, frame_hash_metadata_json,
                created_by
            ) VALUES (
                'manifest-1', replace(printf('%064x', 0), '0', 'a'), 'test_frame',
                'pytest', '2026-07-01', 1, '["policy_id"]', 'claim_count',
                replace(printf('%064x', 0), '0', 'b'), '{"frame_hash":{"format_version":1}}',
                'pytest'
            );

            INSERT INTO pricing.PRICING_RATE_PACKAGE (
                rate_package_id, model_id, model_name, model_version,
                package_version, base_rate, package_status, created_by
            ) VALUES
                (71, 17, 'TEST_MODEL', 'v1', 1, 1.0, 'PUBLISHED', 'pytest'),
                (72, 17, 'TEST_MODEL', 'v2', 2, 1.0, 'PUBLISHED', 'pytest');

            INSERT INTO pricing.MODEL_RUN (
                model_run_id, model_id, model_version, model_kind, export_id,
                manifest_id, rate_package_id, model_name, rating_workbook_path,
                rating_workbook_sha256, run_status, created_by
            ) VALUES (
                'legacy-run', 17, 'v1', 'RAW', 'legacy-export', 'manifest-1', 71,
                'TEST_MODEL', '/tmp/legacy.xlsx',
                replace(printf('%064x', 0), '0', 'c'), 'SUCCESS', 'pytest'
            );

            INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT (
                deployment_id, model_id, rate_package_id, deployment_slot,
                effective_from_ts, deployed_by
            ) VALUES (701, 17, 71, 'TEST_PROD', '2026-07-01 00:00:00', 'pytest');

            INSERT INTO pricing.MODEL_FIT_CONTRACT (
                fit_contract_id, baseline_model_run_id, model_id,
                rate_package_id, contract_schema_version, contract_sha256,
                structure_sha256, contract_json, superglm_version, created_by
            ) VALUES (
                'contract-1', 'legacy-run', 17, 71, 1,
                replace(printf('%064x', 0), '0', 'd'),
                replace(printf('%064x', 0), '0', 'e'),
                '{}', '0.26.0', 'pytest'
            );

            INSERT INTO pricing.MODEL_MONITOR_RUN (
                monitor_run_id, fit_contract_id, baseline_deployment_id,
                model_id, rate_package_id, manifest_id, component_role,
                variant_code, run_signature_sha256, run_status,
                invariant_status, invariant_evidence_sha256,
                invariant_evidence_json, model_frame_sha256,
                fit_configuration_json, result_evidence_sha256, created_by
            ) VALUES (
                'monitor-1', 'contract-1', 701, 17, 71, 'manifest-1',
                'FREQUENCY', 'STATIC_SCORE',
                replace(printf('%064x', 0), '0', 'f'), 'SUCCESS', 'VERIFIED',
                replace(printf('%064x', 0), '0', '1'), '{"status":"VERIFIED"}',
                replace(printf('%064x', 0), '0', 'b'), '{}',
                replace(printf('%064x', 0), '0', '2'), 'pytest'
            );

            INSERT INTO pricing.MODEL_MONITOR_TERM (
                monitor_run_id, term_name, term_kind, sequence_no,
                term_structure_sha256, term_metadata_json
            ) VALUES (
                'monitor-1', 'area', 'categorical', 1,
                replace(printf('%064x', 0), '0', '3'), '{}'
            );
            """
        )
        raw_connection.commit()
    finally:
        raw_connection.close()

    apply_offline_ddl(engine)

    with engine.begin() as connection:
        stored_sql = connection.exec_driver_sql(
            """
            SELECT sql
            FROM pricing.sqlite_master
            WHERE type = 'table' AND name = 'MODEL_RUN'
            """
        ).scalar_one()
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_RUN (
                    model_run_id, model_id, model_version, model_kind,
                    export_id, manifest_id, rate_package_id, model_name,
                    rating_workbook_path, rating_workbook_sha256,
                    run_status, created_by
                ) VALUES (
                    'manual-run', 17, 'v2', 'MANUAL_EDIT',
                    'manual-export', 'manifest-1', 72, 'TEST_MODEL',
                    '/tmp/manual.xlsx', :workbook_sha,
                    'SUCCESS', 'pytest'
                )
                """
            ),
            {"workbook_sha": "a" * 64},
        )
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        assert connection.exec_driver_sql("PRAGMA pricing.foreign_key_check").all() == []
        assert (
            connection.execute(
                text("SELECT model_run_id FROM pricing.MODEL_RUN WHERE model_run_id = 'legacy-run'")
            ).scalar_one()
            == "legacy-run"
        )
        assert (
            connection.execute(
                text(
                    "SELECT baseline_model_run_id FROM pricing.MODEL_FIT_CONTRACT "
                    "WHERE fit_contract_id = 'contract-1'"
                )
            ).scalar_one()
            == "legacy-run"
        )
        assert (
            connection.execute(
                text(
                    "SELECT monitor_run_id FROM pricing.MODEL_MONITOR_RUN "
                    "WHERE monitor_run_id = 'monitor-1'"
                )
            ).scalar_one()
            == "monitor-1"
        )
        assert (
            connection.execute(
                text(
                    "SELECT monitor_run_id FROM pricing.MODEL_MONITOR_TERM "
                    "WHERE monitor_run_id = 'monitor-1'"
                )
            ).scalar_one()
            == "monitor-1"
        )

    assert "MANUAL_EDIT" in stored_sql


def test_offline_upgrade_rolls_back_legacy_orphan_before_model_run_rebuild(tmp_path):
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    legacy_pricing_sql = (
        Path("db/offline_sqlite/pricing.sql")
        .read_text(encoding="utf-8")
        .replace(", 'MANUAL_EDIT'", "")
    )
    raw_connection = engine.raw_connection()
    try:
        raw_connection.executescript(legacy_pricing_sql)
        raw_connection.executescript(
            """
            INSERT INTO pricing.PRICING_MODEL (
                model_id, model_name, model_label, target_name,
                model_type, model_status, created_by
            ) VALUES (
                17, 'TEST_MODEL', 'Test model', 'claim_count',
                'superglm_poisson', 'ACTIVE', 'pytest'
            );

            INSERT INTO pricing.DATASET_MANIFEST (
                manifest_id, manifest_signature_sha256, dataset_name,
                source_system, data_as_of_date, row_count, pk_columns_json,
                target_column, model_frame_sha256, frame_hash_metadata_json,
                created_by
            ) VALUES (
                'manifest-1', replace(printf('%064x', 0), '0', 'a'), 'test_frame',
                'pytest', '2026-07-01', 1, '["policy_id"]', 'claim_count',
                replace(printf('%064x', 0), '0', 'b'), '{"frame_hash":{"format_version":1}}',
                'pytest'
            );

            INSERT INTO pricing.PRICING_RATE_PACKAGE (
                rate_package_id, model_id, model_name, model_version,
                package_version, base_rate, package_status, created_by
            ) VALUES (71, 17, 'TEST_MODEL', 'v1', 1, 1.0, 'PUBLISHED', 'pytest');

            INSERT INTO pricing.MODEL_RUN (
                model_run_id, model_id, model_version, model_kind, export_id,
                manifest_id, rate_package_id, model_name, rating_workbook_path,
                rating_workbook_sha256, run_status, created_by
            ) VALUES (
                'legacy-run', 17, 'v1', 'RAW', 'legacy-export', 'manifest-1', 71,
                'TEST_MODEL', '/tmp/legacy.xlsx',
                replace(printf('%064x', 0), '0', 'c'), 'SUCCESS', 'pytest'
            );

            INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT (
                deployment_id, model_id, rate_package_id, deployment_slot,
                effective_from_ts, deployed_by
            ) VALUES (701, 17, 71, 'TEST_PROD', '2026-07-01 00:00:00', 'pytest');

            INSERT INTO pricing.MODEL_FIT_CONTRACT (
                fit_contract_id, baseline_model_run_id, model_id,
                rate_package_id, contract_schema_version, contract_sha256,
                structure_sha256, contract_json, superglm_version, created_by
            ) VALUES (
                'contract-1', 'legacy-run', 17, 71, 1,
                replace(printf('%064x', 0), '0', 'd'),
                replace(printf('%064x', 0), '0', 'e'),
                '{}', '0.26.0', 'pytest'
            );

            INSERT INTO pricing.MODEL_MONITOR_RUN (
                monitor_run_id, fit_contract_id, baseline_deployment_id,
                model_id, rate_package_id, manifest_id, component_role,
                variant_code, run_signature_sha256, run_status,
                invariant_status, invariant_evidence_sha256,
                invariant_evidence_json, model_frame_sha256,
                fit_configuration_json, result_evidence_sha256, created_by
            ) VALUES (
                'monitor-1', 'contract-1', 701, 17, 71, 'manifest-1',
                'FREQUENCY', 'STATIC_SCORE',
                replace(printf('%064x', 0), '0', 'f'), 'SUCCESS', 'VERIFIED',
                replace(printf('%064x', 0), '0', '1'), '{"status":"VERIFIED"}',
                replace(printf('%064x', 0), '0', 'b'), '{}',
                replace(printf('%064x', 0), '0', '2'), 'pytest'
            );

            INSERT INTO pricing.MODEL_MONITOR_TERM (
                monitor_run_id, term_name, term_kind, sequence_no,
                term_structure_sha256, term_metadata_json
            ) VALUES (
                'monitor-1', 'area', 'categorical', 1,
                replace(printf('%064x', 0), '0', '3'), '{}'
            );
            """
        )
        legacy_model_run_sql = raw_connection.execute(
            """
            SELECT sql
            FROM pricing.sqlite_master
            WHERE type = 'table' AND name = 'MODEL_RUN'
            """
        ).fetchone()[0]
        raw_connection.commit()
        raw_connection.execute("PRAGMA foreign_keys=OFF")
        raw_connection.execute("DELETE FROM pricing.MODEL_RUN WHERE model_run_id = 'legacy-run'")
        raw_connection.commit()
        raw_connection.execute("PRAGMA foreign_keys=ON")
    finally:
        raw_connection.close()

    with pytest.raises(RuntimeError, match="offline foreign-key check failed"):
        apply_offline_ddl(engine)

    with engine.connect() as connection:
        stored_sql = connection.exec_driver_sql(
            """
            SELECT sql
            FROM pricing.sqlite_master
            WHERE type = 'table' AND name = 'MODEL_RUN'
            """
        ).scalar_one()
        assert stored_sql == legacy_model_run_sql
        assert "MANUAL_EDIT" not in stored_sql
        assert (
            connection.execute(
                text("SELECT COUNT(*) FROM pricing.MODEL_RUN WHERE model_run_id = 'legacy-run'")
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                text(
                    "SELECT baseline_model_run_id FROM pricing.MODEL_FIT_CONTRACT "
                    "WHERE fit_contract_id = 'contract-1'"
                )
            ).scalar_one()
            == "legacy-run"
        )
        assert (
            connection.execute(
                text(
                    "SELECT monitor_run_id FROM pricing.MODEL_MONITOR_RUN WHERE monitor_run_id = 'monitor-1'"
                )
            ).scalar_one()
            == "monitor-1"
        )
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        assert connection.exec_driver_sql("PRAGMA pricing.foreign_key_check").all()


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
