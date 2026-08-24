from __future__ import annotations

import shlex
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from pricing_pipeline.infra.offline_sqlite import (
    apply_offline_ddl,
    sqlite_engine_with_offline_schemas,
)
from scripts import reset_pricing_experiments, reset_remote_pricing_schema


def test_reset_sql_deletes_dependent_pricing_tables_before_parent_tables():
    required_order = [
        "DELETE FROM mlops.MODEL_RUN_SPLIT_SET",
        "DELETE FROM mlops.MODEL_RUN_DATASET",
        "DELETE FROM mlops.MODEL_RUN_METRIC",
        "DELETE FROM pricing.CV_FOLD_METRIC",
        "DELETE FROM pricing.MODEL_RUN",
        "DELETE FROM pricing.PRICING_MODEL_DEPLOYMENT",
        "DELETE FROM pricing.PRICING_PACKAGE_POINTER",
        "DELETE FROM pricing.PRICING_COMPILED_1D_RATE_BAND",
        "DELETE FROM pricing.PRICING_COMPILED_RATE_CELL",
        "DELETE FROM pricing.PRICING_RATE_CELL_LEVEL",
        "DELETE FROM pricing.PRICING_RATE_CELL",
        "DELETE FROM pricing.PRICING_TERM_FEATURE",
        "DELETE FROM pricing.PRICING_TERM",
        "DELETE FROM pricing.PRICING_RATE_PACKAGE",
        "DELETE FROM pricing.PRICING_FEATURE_LEVEL",
        "DELETE FROM pricing.PRICING_FEATURE_LEVEL_SET",
        "DELETE FROM pricing.PRICING_FEATURE",
        "DELETE FROM pricing.CV_FOLD",
        "DELETE FROM pricing.CV_SPLIT_SET",
        "DELETE FROM pricing.DATASET_COLUMN",
        "DELETE FROM pricing.DATASET_MANIFEST",
        "DELETE FROM pricing_stg.STG_CELL_LEVEL",
        "DELETE FROM pricing_stg.STG_RATE_CELL",
        "DELETE FROM pricing_stg.STG_RATING_EXPORT",
        "DELETE FROM pricing.PRICING_MODEL",
    ]

    actual_order = [
        statement.strip()
        for statement in reset_pricing_experiments.RESET_SQL.strip().split(";")
        if statement.strip()
    ]
    assert actual_order == required_order


def test_reset_requires_explicit_confirmation_flag():
    parser = reset_pricing_experiments.build_parser()

    args = parser.parse_args(["--yes"])

    assert args.yes is True


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _RecordingConnection:
    def __init__(self, *, monitoring_exists: bool):
        self.statements: list[str] = []
        self.monitoring_exists = monitoring_exists

    def execute(self, statement):
        sql = str(statement).strip()
        self.statements.append(sql)
        return _ScalarResult(1 if self.monitoring_exists else None)


class _RecordingBegin:
    def __init__(self, connection: _RecordingConnection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, *_args):
        return False


class _RecordingEngine:
    dialect = SimpleNamespace(name="mssql")

    def __init__(self, *, monitoring_exists: bool):
        self.connection = _RecordingConnection(monitoring_exists=monitoring_exists)

    def begin(self):
        return _RecordingBegin(self.connection)


def test_sql_server_reset_refuses_monitoring_history_before_any_delete(monkeypatch):
    engine = _RecordingEngine(monitoring_exists=True)
    monkeypatch.setattr(reset_pricing_experiments, "get_engine", lambda: engine)

    with pytest.raises(SystemExit, match="reset_remote_pricing_schema.py") as exc_info:
        reset_pricing_experiments.reset_pricing_experiments()

    command = str(exc_info.value).split("Run: ", maxsplit=1)[1]
    tokens = shlex.split(command)
    script_position = tokens.index("scripts/reset_remote_pricing_schema.py")
    args = reset_remote_pricing_schema.build_parser().parse_args(tokens[script_position + 1 :])
    reset_remote_pricing_schema.validate_args(args)
    assert args.expected_database == "REPLACE_WITH_DATABASE_NAME"
    assert args.execute is True
    assert args.confirmed_destructive_reset is True

    statements = engine.connection.statements
    assert len(statements) == 2
    assert all("MODEL_FIT_CONTRACT" in statement for statement in statements)
    assert not any(statement.startswith("DELETE FROM") for statement in statements)


def test_monitoring_history_probe_accepts_a_pre_monitoring_sqlite_schema(tmp_path):
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )

    with engine.connect() as connection:
        assert (
            reset_pricing_experiments._monitoring_history_exists(
                connection,
                sqlite=True,
            )
            is False
        )


def _seed_offline_monitoring_graph(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_MODEL (
                    model_id, model_name, target_name, model_type, model_status, created_by
                ) VALUES (1, 'RESET_MODEL', 'target', 'superglm_poisson', 'ACTIVE', 'pytest')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.DATASET_MANIFEST (
                    manifest_id, manifest_signature_sha256, dataset_name, source_system,
                    data_as_of_date, row_count, pk_columns_json, target_column,
                    model_frame_sha256, frame_hash_metadata_json, created_by
                ) VALUES (
                    'manifest-1', :manifest_sha, 'reset_frame', 'pytest',
                    '2026-08-24', 10, '["id"]', 'target', :frame_sha, '{}', 'pytest'
                )
                """
            ),
            {"manifest_sha": "a" * 64, "frame_sha": "b" * 64},
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_RATE_PACKAGE (
                    rate_package_id, model_id, model_name, model_version,
                    package_version, base_rate, package_status, created_by
                ) VALUES (2, 1, 'RESET_MODEL', 'v1', 1, 1.0, 'PUBLISHED', 'pytest')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_RUN (
                    model_run_id, model_id, model_version, model_kind, export_id,
                    manifest_id, rate_package_id, model_name, rating_workbook_path,
                    rating_workbook_sha256, run_status, created_by
                ) VALUES (
                    'run-1', 1, 'v1', 'RAW', 'export-1', 'manifest-1', 2,
                    'RESET_MODEL', '/tmp/reset.xlsx', :digest, 'SUCCESS', 'pytest'
                )
                """
            ),
            {"digest": "c" * 64},
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT (
                    deployment_id, model_id, rate_package_id, deployment_slot,
                    effective_from_ts, deployed_by
                ) VALUES (3, 1, 2, 'RESET_SLOT', '2026-08-24 00:00:00', 'pytest')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_FIT_CONTRACT (
                    fit_contract_id, baseline_model_run_id, model_id, rate_package_id,
                    contract_schema_version, contract_sha256, structure_sha256,
                    contract_json, superglm_version, created_by
                ) VALUES (
                    'contract-1', 'run-1', 1, 2, 1, :contract_sha,
                    :structure_sha, '{}', '0.28.0', 'pytest'
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
                    variant_code, run_signature_sha256, run_status, invariant_status,
                    invariant_evidence_sha256, invariant_evidence_json,
                    model_frame_sha256, fit_configuration_json,
                    result_evidence_sha256, started_ts, completed_ts, created_by
                ) VALUES (
                    'monitor-1', 'contract-1', 3, 1, 2, 'manifest-1', 'OTHER',
                    'STATIC_SCORE', :signature, 'SUCCESS', 'VERIFIED',
                    :invariant_sha, '{}', :frame_sha, '{}', :result_sha,
                    '2026-08-24 00:00:00', '2026-08-24 00:01:00', 'pytest'
                )
                """
            ),
            {
                "signature": "f" * 64,
                "invariant_sha": "1" * 64,
                "frame_sha": "b" * 64,
                "result_sha": "2" * 64,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_MONITOR_TERM (
                    monitor_run_id, term_name, term_kind, sequence_no,
                    term_structure_sha256, term_metadata_json
                ) VALUES ('monitor-1', 'term', 'numeric', 1, :digest, '{}')
                """
            ),
            {"digest": "3" * 64},
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_MONITOR_LAMBDA (
                    monitor_run_id, component_name, term_name, lambda_value, lambda_mode
                ) VALUES ('monitor-1', 'term', 'term', 1.0, 'BASELINE')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_MONITOR_RELATIVITY (
                    monitor_run_id, term_name, term_kind, point_key, point_label,
                    relativity, log_relativity, is_reference
                ) VALUES ('monitor-1', 'term', 'numeric', 'point', 'point', 1.0, 0.0, 1)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.MODEL_MONITOR_METRIC (
                    monitor_run_id, metric_name, metric_value
                ) VALUES ('monitor-1', 'deviance', 1.0)
                """
            )
        )


def test_confirmed_sqlite_reset_refuses_monitoring_graph_without_mutation(
    tmp_path,
    monkeypatch,
):
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    apply_offline_ddl(engine)
    _seed_offline_monitoring_graph(engine)
    with (
        pytest.raises(IntegrityError, match="monitoring evidence is immutable"),
        engine.begin() as connection,
    ):
        connection.execute(text("DELETE FROM pricing.MODEL_MONITOR_METRIC"))
    monkeypatch.setattr(reset_pricing_experiments, "get_engine", lambda: engine)

    with pytest.raises(SystemExit, match="Local SQLite") as exc_info:
        reset_pricing_experiments.reset_pricing_experiments()

    message = str(exc_info.value)
    assert "reset_remote_pricing_schema.py" not in message
    command = message.split("Run after closing all local connections: ", maxsplit=1)[1]
    tokens = shlex.split(command)
    assert tokens == [
        "rm",
        "--",
        str(tmp_path / "coordinator.sqlite"),
        str(tmp_path / "pricing.sqlite"),
        str(tmp_path / "pricing_stg.sqlite"),
        str(tmp_path / "mlops.sqlite"),
    ]

    with engine.connect() as connection:
        for table_name in (
            "MODEL_MONITOR_METRIC",
            "MODEL_MONITOR_RELATIVITY",
            "MODEL_MONITOR_LAMBDA",
            "MODEL_MONITOR_TERM",
            "MODEL_MONITOR_RUN",
            "MODEL_FIT_CONTRACT",
            "MODEL_RUN",
        ):
            assert (
                connection.execute(text(f"SELECT COUNT(*) FROM pricing.{table_name}")).scalar_one()
                == 1
            )
        delete_triggers = {
            row[0]
            for row in connection.execute(
                text(
                    """
                    SELECT name
                    FROM pricing.sqlite_master
                    WHERE type = 'trigger'
                      AND name LIKE 'TR_MODEL_%_IMMUTABLE_DELETE'
                    """
                )
            )
        }
    assert delete_triggers == {
        "TR_MODEL_FIT_CONTRACT_IMMUTABLE_DELETE",
        "TR_MODEL_MONITOR_VARIANT_IMMUTABLE_DELETE",
        "TR_MODEL_MONITOR_RUN_IMMUTABLE_DELETE",
        "TR_MODEL_MONITOR_TERM_IMMUTABLE_DELETE",
        "TR_MODEL_MONITOR_LAMBDA_IMMUTABLE_DELETE",
        "TR_MODEL_MONITOR_RELATIVITY_IMMUTABLE_DELETE",
        "TR_MODEL_MONITOR_METRIC_IMMUTABLE_DELETE",
    }
