import sqlite3

import pytest
from sqlalchemy import text

from pricing_pipeline.infra.offline_sqlite import (
    apply_offline_ddl,
    sqlite_engine_with_offline_schemas,
)


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
