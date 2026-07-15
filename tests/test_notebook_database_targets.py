from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.orchestration.publish_completed_build import (
    CompletedModelBuild,
    CompletedModelBuildError,
)


class _ScalarResult:
    def __init__(self, value: str):
        self._value = value

    def scalar_one(self) -> str:
        return self._value


class _RemoteEngine:
    def __init__(self, database_name: str):
        self.database_name = database_name
        self.statements: list[str] = []

    @contextmanager
    def connect(self):
        engine = self

        class Connection:
            def execute(self, statement):
                engine.statements.append(str(statement))
                return _ScalarResult(engine.database_name)

        yield Connection()


def _install_runtime(monkeypatch, api, *, database_name: str = "PricingAudit"):
    engine = _RemoteEngine(database_name)
    runtime = SimpleNamespace(
        settings=Settings(
            mssql_server="private-server-name",
            pricing_database=database_name,
        ),
        get_engine=lambda: engine,
    )
    calls: list[str | None] = []
    monkeypatch.setattr(
        api,
        "runtime_from_env_or_module",
        lambda runtime_module=None: calls.append(runtime_module) or runtime,
    )
    return engine, calls


def _model_spec(api):
    return api.PricingModelSpec(
        name="CLAIM_FREQUENCY",
        label="Claim frequency",
        target="claim_count",
        model_type="superglm_poisson",
        deployment_slot="CLAIM_FREQUENCY_UAT",
        features=("age",),
        dataset_name="claim_frequency_frame",
        source_system="pricing_sql",
        pk_columns=("policy_id",),
    )


def test_connect_local_creates_persistent_attached_schema_databases(tmp_path):
    from pricing_pipeline import notebook as api

    local_root = tmp_path / "pricing_models" / "claim_frequency" / ".local"
    first = api.connect(mode="local", local_root=local_root)

    assert first.mode == "local"
    assert first.write_allowed is True
    assert "local SQLite" in first.destination
    assert set(first.database_paths) == {"pricing", "pricing_stg", "mlops"}
    assert all(path.exists() for path in first.database_paths.values())
    assert first.settings.mlflow_enabled is False
    assert first.settings.rating_export_root == local_root.resolve() / "rating_exports"
    assert (
        first.settings.validation_split_artifact_root == local_root.resolve() / "validation_splits"
    )
    assert first.settings.workbench_artifact_root == local_root.resolve() / "workbench_artifacts"

    with first.engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_MODEL (
                    model_name, model_label, target_name, model_type,
                    model_status, created_by
                ) VALUES (
                    'PRESERVED', 'Preserved', 'target', 'poisson',
                    'ACTIVE', 'test'
                )
                """
            )
        )
    first.engine.dispose()

    second = api.connect(mode="local", local_root=local_root)
    with second.engine.connect() as connection:
        count = connection.execute(
            text("SELECT COUNT(*) FROM pricing.PRICING_MODEL WHERE model_name = 'PRESERVED'")
        ).scalar_one()

    assert count == 1


def test_local_sqlite_uses_a_file_backed_transaction_coordinator(tmp_path):
    from pricing_pipeline import notebook as api

    local_root = tmp_path / ".local"
    context = api.connect(mode="local", local_root=local_root)

    with context.engine.connect() as connection:
        databases = {
            str(row[1]): str(row[2])
            for row in connection.exec_driver_sql("PRAGMA database_list").all()
        }

    assert databases["main"] == str((local_root / "coordinator.sqlite").resolve())
    assert Path(databases["main"]).is_file()


def test_local_publication_lock_serializes_publishers(tmp_path):
    from pricing_pipeline.infra.offline_sqlite import local_publish_lock

    first_entered = Event()
    release_first = Event()
    second_entered = Event()

    def first_publisher():
        with local_publish_lock(tmp_path):
            first_entered.set()
            assert release_first.wait(timeout=2)

    def second_publisher():
        assert first_entered.wait(timeout=2)
        with local_publish_lock(tmp_path):
            second_entered.set()

    first = Thread(target=first_publisher)
    second = Thread(target=second_publisher)
    first.start()
    second.start()
    assert first_entered.wait(timeout=2)
    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()


def test_connect_remote_requires_expected_database_before_loading_runtime(monkeypatch):
    from pricing_pipeline import notebook as api

    _engine, calls = _install_runtime(monkeypatch, api)

    with pytest.raises(ValueError, match="expected_remote_database is required"):
        api.connect(mode="remote")

    assert calls == []


def test_connect_requires_an_explicit_local_or_remote_mode():
    from pricing_pipeline import notebook as api

    with pytest.raises(TypeError, match="mode"):
        api.connect()


def test_connect_remote_rejects_the_wrong_database(monkeypatch):
    from pricing_pipeline import notebook as api

    engine, _calls = _install_runtime(
        monkeypatch,
        api,
        database_name="UnexpectedDatabase",
    )

    with pytest.raises(RuntimeError, match="Remote database mismatch"):
        api.connect(
            mode="remote",
            expected_remote_database="PricingAudit",
            allow_remote_writes=True,
        )

    assert engine.statements == ["SELECT DB_NAME()"]


@pytest.mark.parametrize("allow_writes", [False, True])
def test_connect_remote_verifies_database_and_exposes_safe_destination(
    monkeypatch,
    allow_writes,
):
    from pricing_pipeline import notebook as api

    engine, calls = _install_runtime(monkeypatch, api)

    result = api.connect(
        mode="remote",
        runtime_module="work_runtime.database",
        expected_remote_database="pricingaudit",
        allow_remote_writes=allow_writes,
    )

    assert result.engine is engine
    assert result.mode == "remote"
    assert result.write_allowed is allow_writes
    assert result.destination == "remote SQL database: PricingAudit"
    assert "private-server-name" not in result.destination
    assert result.database_paths == {}
    assert engine.statements == ["SELECT DB_NAME()"]
    assert calls == ["work_runtime.database"]


def test_connect_rejects_unknown_explicit_mode():
    from pricing_pipeline import notebook as api

    with pytest.raises(ValueError, match="mode must be 'local' or 'remote'"):
        api.connect(mode="docker")


@pytest.mark.parametrize(
    ("operation", "invoke"),
    [
        (
            "register_model",
            lambda api, context, root: api.register_model(
                context,
                api.PricingModelSpec(
                    name="BLOCKED",
                    label="Blocked",
                    target="target",
                    model_type="poisson",
                    deployment_slot="UAT",
                    features=("feature",),
                    dataset_name="blocked_frame",
                    source_system="pytest",
                    pk_columns=("row_id",),
                ),
                source_root=root,
            ),
        ),
        (
            "build_candidate",
            lambda api, context, root: api.build_candidate(
                context,
                model=object(),
                frame=object(),
                model_factory=lambda: object(),
            ),
        ),
        (
            "publish_candidate",
            lambda api, context, root: api.publish_candidate(context, object()),
        ),
        (
            "publish_edits",
            lambda api, context, root: api.publish_edits(
                context,
                candidate=object(),
                reason="blocked",
            ),
        ),
        (
            "deploy_package",
            lambda api, context, root: api.deploy_package(
                context,
                package=object(),
                reason="blocked",
            ),
        ),
    ],
)
def test_remote_context_blocks_every_mutating_notebook_entry_point(
    tmp_path,
    operation,
    invoke,
):
    from pricing_pipeline import notebook as api

    context = api.NotebookContext(
        engine=object(),
        settings=Settings(),
        mode="remote",
        write_allowed=False,
        destination="remote SQL database: PricingAudit",
    )

    with pytest.raises(PermissionError, match=operation):
        invoke(api, context, tmp_path)


def test_local_register_model_is_idempotent(tmp_path):
    from pricing_pipeline import notebook as api

    model_root = tmp_path / "pricing_models" / "claim_frequency"
    model_root.mkdir(parents=True)
    context = api.connect(mode="local", local_root=model_root / ".local")
    kwargs = {
        "spec": _model_spec(api),
        "source_root": model_root,
        "created_by": "analyst@example.test",
    }

    first = api.register_model(context, **kwargs)
    second = api.register_model(context, **kwargs)

    assert first.model_id == second.model_id
    with context.engine.connect() as connection:
        count = connection.execute(
            text("SELECT COUNT(*) FROM pricing.PRICING_MODEL WHERE model_name = 'CLAIM_FREQUENCY'")
        ).scalar_one()
    assert count == 1


def test_local_model_version_reuses_export_and_advances_trained_versions(tmp_path):
    from pricing_pipeline import notebook as api
    from pricing_pipeline.publishing.sqlite_notebook import (
        resolve_sqlite_model_version,
    )

    model_root = tmp_path / "pricing_models" / "claim_frequency"
    model_root.mkdir(parents=True)
    context = api.connect(mode="local", local_root=model_root / ".local")
    model = api.register_model(
        context,
        _model_spec(api),
        source_root=model_root,
        created_by="analyst@example.test",
    )
    with context.engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO pricing.PRICING_RATE_PACKAGE (
                    model_id, model_name, model_version, package_version,
                    base_rate, package_status, source_export_id,
                    offset_handling, created_by
                ) VALUES (
                    :model_id, 'CLAIM_FREQUENCY', 'v3', 1,
                    0.1, 'PUBLISHED', 'existing-export',
                    'NONE', 'test'
                )
                """
            ),
            {"model_id": model.model_id},
        )

    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            export_id="existing-export",
        )
        == "v3"
    )
    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            export_id="new-export",
        )
        == "v4"
    )
    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            export_id="second-new-export",
        )
        == "v5"
    )
    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            export_id="new-export",
        )
        == "v4"
    )


def test_publish_candidate_records_local_package_run_and_audit_links(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline import notebook as api
    from pricing_pipeline.publishing import sqlite_notebook

    model_root = tmp_path / "pricing_models" / "claim_frequency"
    model_root.mkdir(parents=True)
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_bytes(b"local workbook")
    context = api.connect(mode="local", local_root=model_root / ".local")
    model = api.register_model(
        context,
        _model_spec(api),
        source_root=model_root,
        created_by="analyst@example.test",
    )
    with context.engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO pricing.DATASET_MANIFEST (
                    manifest_id, dataset_name, source_system, data_as_of_date,
                    row_count, pk_columns_json, target_column, weight_column,
                    model_frame_sha256, frame_hash_metadata_json, created_by
                ) VALUES (
                    'manifest-1', 'claim_frame', 'work_sql', '2026-06-30',
                    20, '["policy_id"]', 'claim_count', 'exposure',
                    :frame_sha, '{}', 'test'
                )
                """
            ),
            {"frame_sha": "f" * 64},
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.CV_SPLIT_SET (
                    split_set_id, manifest_id, split_mode, row_order_sha256,
                    row_count, fold_count, created_by
                ) VALUES (
                    'split-1', 'manifest-1', 'MATERIALIZED', :sha,
                    20, 2, 'test'
                )
                """
            ),
            {"sha": "a" * 64},
        )

    staging_digest = {"value": "d" * 64}
    staged_digests = []

    def stage(engine, **kwargs):
        assert kwargs["model_id"] == model.model_id
        assert kwargs["replace"] is True
        staged_digests.append(staging_digest["value"])
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT OR REPLACE INTO pricing_stg.STG_RATING_EXPORT (
                        export_id, model_id, model_name, model_version,
                        base_rate, effective_from_date, source_file,
                        publication_receipt_json, publication_receipt_sha256,
                        package_metadata_json, offset_handling,
                        metadata_origin, staging_content_sha256, created_by
                    ) VALUES (
                        :export_id, :model_id, :model_name, :model_version,
                        0.25, NULL, :source_file,
                        '{}', :receipt_sha, '{}', 'NONE',
                        'SUPERGLM_EXPORTER', :staging_digest, 'test'
                    )
                    """
                ),
                {
                    "export_id": kwargs["export_id"],
                    "model_id": kwargs["model_id"],
                    "model_name": (
                        "WRONG_MODEL"
                        if kwargs["export_id"].endswith("__mismatch")
                        else kwargs["model_name"]
                    ),
                    "model_version": kwargs["model_version"],
                    "source_file": str(kwargs["workbook_path"]),
                    "receipt_sha": kwargs["publication_receipt_sha256"],
                    "staging_digest": staging_digest["value"],
                },
            )

    monkeypatch.setattr(sqlite_notebook, "stage_rating_export", stage)
    monkeypatch.setattr(
        sqlite_notebook,
        "_verify_candidate_artifact",
        lambda *args, **kwargs: None,
    )
    completed_build = CompletedModelBuild(
        model_id=model.model_id,
        model_name=model.name,
        rating_workbook_path=str(workbook),
        rating_workbook_sha256=sqlite_notebook.sha256_file(workbook),
        model_version="v1",
        model_type=model.config.model_type,
        target_name=model.config.target_name,
        deployment_slot=model.config.deployment_slot,
        effective_from=None,
        export_id="claim-frequency__run-1",
        manifest_id="manifest-1",
        split_set_id="split-1",
        created_by="analyst@example.test",
        mlflow_run_id="mlflow-old",
        publication_receipt_path=str(tmp_path / "receipt.json"),
        publication_receipt_sha256="b" * 64,
        candidate_artifact_path=str(tmp_path / "candidate.joblib"),
        candidate_artifact_sha256="c" * 64,
        candidate_artifact_format="superglm-candidate-joblib-v1",
        candidate_artifact_size_bytes=123,
        candidate_python_version=python_version(),
        candidate_superglm_version=version("superglm"),
        model_source_sha256="e" * 64,
        metrics={"cv_mean_deviance": 1.25},
        metric_scopes={"cv_mean_deviance": "cv"},
        fold_metrics=(
            {
                "fold_no": 1,
                "metric_name": "deviance",
                "metric_value": 1.1,
            },
        ),
    )
    assert (
        sqlite_notebook.resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            export_id=completed_build.export_id,
        )
        == "v1"
    )
    candidate = api.BuiltCandidate(
        model=model,
        standard_build=SimpleNamespace(completed_build=completed_build, metrics={}),
    )

    first = api.publish_candidate(
        context,
        candidate,
    )
    second = api.publish_candidate(
        context,
        candidate,
    )

    assert first.model_id == model.model_id
    assert first.model_version == "v1"
    assert first.manifest_id == "manifest-1"
    assert first.split_set_id == "split-1"
    assert first.package_version == 1
    assert first.package_status == "LOCAL_AUDIT"
    assert first.model_run_id is not None
    assert first.was_existing is False
    assert second.rate_package_id == first.rate_package_id
    assert second.model_run_id == first.model_run_id
    assert second.mlflow_run_id == "mlflow-old"
    assert second.was_existing is True
    assert staged_digests == ["d" * 64, "d" * 64]

    workbook.write_bytes(b"mutated local workbook")
    with pytest.raises(CompletedModelBuildError, match="rating workbook SHA-256"):
        api.publish_candidate(
            context,
            candidate,
        )
    workbook.write_bytes(b"local workbook")

    staging_digest["value"] = "e" * 64
    with pytest.raises(ValueError, match="staging_content_sha256"):
        api.publish_candidate(
            context,
            candidate,
        )
    staging_digest["value"] = "d" * 64

    changed_run_evidence = completed_build.model_copy(
        update={
            "mlflow_run_id": "mlflow-new",
            "metrics": {"cv_mean_deviance": 99.0},
        }
    )
    changed_run_candidate = api.BuiltCandidate(
        model=model,
        standard_build=SimpleNamespace(
            completed_build=changed_run_evidence,
            metrics={},
        ),
    )
    with pytest.raises(ValueError, match="incompatible model-run evidence"):
        api.publish_candidate(
            context,
            changed_run_candidate,
        )

    conflicting_build = completed_build.model_copy(
        update={
            "publication_receipt_path": str(tmp_path / "different-receipt.json"),
            "publication_receipt_sha256": "c" * 64,
        }
    )
    conflicting_candidate = api.BuiltCandidate(
        model=model,
        standard_build=SimpleNamespace(
            completed_build=conflicting_build,
            metrics={},
        ),
    )
    with pytest.raises(ValueError, match="incompatible publication evidence"):
        api.publish_candidate(
            context,
            conflicting_candidate,
        )
    recovered = api.publish_candidate(
        context,
        candidate,
    )
    assert recovered.was_existing is True

    with context.engine.connect() as connection:
        counts = {
            table: connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            for table in (
                "pricing.PRICING_RATE_PACKAGE",
                "pricing.MODEL_RUN",
                "mlops.MODEL_RUN_DATASET",
                "mlops.MODEL_RUN_SPLIT_SET",
                "mlops.MODEL_RUN_METRIC",
                "pricing.CV_FOLD_METRIC",
            )
        }
        staged_receipt_sha256 = connection.execute(
            text(
                "SELECT publication_receipt_sha256 "
                "FROM pricing_stg.STG_RATING_EXPORT "
                "WHERE export_id = 'claim-frequency__run-1'"
            )
        ).scalar_one()
        stored_workbook_sha256 = connection.execute(
            text(
                "SELECT rating_workbook_sha256 "
                "FROM pricing.MODEL_RUN "
                "WHERE export_id = 'claim-frequency__run-1'"
            )
        ).scalar_one()
        stored_package_status = connection.execute(
            text(
                "SELECT package_status "
                "FROM pricing.PRICING_RATE_PACKAGE "
                "WHERE source_export_id = 'claim-frequency__run-1'"
            )
        ).scalar_one()
    assert counts == {
        "pricing.PRICING_RATE_PACKAGE": 1,
        "pricing.MODEL_RUN": 1,
        "mlops.MODEL_RUN_DATASET": 1,
        "mlops.MODEL_RUN_SPLIT_SET": 1,
        "mlops.MODEL_RUN_METRIC": 1,
        "pricing.CV_FOLD_METRIC": 1,
    }
    assert staged_receipt_sha256 == "b" * 64
    assert stored_workbook_sha256 == completed_build.rating_workbook_sha256
    assert stored_package_status == "LOCAL_AUDIT"

    mismatch_export_id = "claim-frequency__mismatch"
    mismatch_version = sqlite_notebook.resolve_sqlite_model_version(
        context.engine,
        model_name=model.name,
        export_id=mismatch_export_id,
    )
    mismatch_candidate = api.BuiltCandidate(
        model=model,
        standard_build=SimpleNamespace(
            completed_build=completed_build.model_copy(
                update={
                    "export_id": mismatch_export_id,
                    "model_version": mismatch_version,
                }
            ),
            metrics={},
        ),
    )
    with pytest.raises(ValueError, match="incompatible staged evidence"):
        api.publish_candidate(
            context,
            mismatch_candidate,
        )

    unreserved_candidate = api.BuiltCandidate(
        model=model,
        standard_build=SimpleNamespace(
            completed_build=completed_build.model_copy(
                update={
                    "export_id": "claim-frequency__unreserved",
                    "model_version": "v3",
                }
            ),
            metrics={},
        ),
    )
    with pytest.raises(ValueError, match="has no reserved model version"):
        api.publish_candidate(
            context,
            unreserved_candidate,
        )

    with context.engine.begin() as connection:
        connection.execute(text("DELETE FROM mlops.MODEL_RUN_DATASET"))
    with pytest.raises(RuntimeError, match="incomplete local publication lineage"):
        api.publish_candidate(
            context,
            candidate,
        )


def test_local_publication_verifies_candidate_artifact_before_staging(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline import notebook as api
    from pricing_pipeline.orchestration.publish_completed_build import (
        CompletedModelBuildError,
    )
    from pricing_pipeline.publishing import sqlite_notebook
    from pricing_pipeline.workbench.artifacts import BUNDLE_FORMAT

    model_root = tmp_path / "pricing_models" / "claim_frequency"
    model_root.mkdir(parents=True)
    context = api.connect(mode="local", local_root=model_root / ".local")
    model = api.register_model(
        context,
        _model_spec(api),
        source_root=model_root,
        created_by="analyst@example.test",
    )
    with context.engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO pricing.DATASET_MANIFEST (
                    manifest_id, dataset_name, source_system, data_as_of_date,
                    row_count, pk_columns_json, target_column,
                    model_frame_sha256, frame_hash_metadata_json, created_by
                ) VALUES (
                    'manifest-verify', 'claim_frame', 'work_sql', '2026-06-30',
                    20, '["policy_id"]', 'claim_count', :frame_sha, '{}', 'test'
                )
                """
            ),
            {"frame_sha": "f" * 64},
        )
    monkeypatch.setattr(
        sqlite_notebook,
        "stage_rating_export",
        lambda *args, **kwargs: pytest.fail("staging ran before artifact verification"),
    )
    missing_artifact = (
        context.settings.workbench_artifact_root / "CLAIM_FREQUENCY" / "missing.joblib"
    )
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_bytes(b"rating workbook")
    completed_build = CompletedModelBuild(
        model_id=model.model_id,
        model_name=model.name,
        rating_workbook_path=str(workbook),
        rating_workbook_sha256=sqlite_notebook.sha256_file(workbook),
        model_version="v1",
        model_type=model.config.model_type,
        target_name=model.config.target_name,
        deployment_slot=model.config.deployment_slot,
        export_id="claim-frequency__verify",
        manifest_id="manifest-verify",
        created_by="analyst@example.test",
        publication_receipt_path=str(tmp_path / "receipt.json"),
        publication_receipt_sha256="c" * 64,
        candidate_artifact_path=str(missing_artifact),
        candidate_artifact_sha256="a" * 64,
        candidate_artifact_format=BUNDLE_FORMAT,
        candidate_artifact_size_bytes=10,
        candidate_python_version=python_version(),
        candidate_superglm_version=version("superglm"),
        model_source_sha256="b" * 64,
    )
    candidate = api.BuiltCandidate(
        model=model,
        standard_build=SimpleNamespace(completed_build=completed_build, metrics={}),
    )

    with pytest.raises(
        CompletedModelBuildError,
        match="candidate artifact verification failed",
    ):
        api.publish_candidate(context, candidate)


def test_local_context_refuses_real_deployment(tmp_path):
    from pricing_pipeline import notebook as api

    context = api.connect(mode="local", local_root=tmp_path / ".local")

    with pytest.raises(RuntimeError, match="Remote mode is required for deployment"):
        api.deploy_package(
            context,
            package=object(),
            reason="blocked locally",
        )


def test_local_context_explains_that_editor_publication_requires_remote(tmp_path):
    from pricing_pipeline import notebook as api

    context = api.connect(mode="local", local_root=tmp_path / ".local")

    with pytest.raises(RuntimeError, match="Remote mode is required for the editor"):
        api.open_candidate(
            context,
            model=object(),
            package_version=1,
        )
    with pytest.raises(RuntimeError, match="Remote mode is required for the editor"):
        api.publish_edits(
            context,
            candidate=object(),
            reason="blocked locally",
        )
