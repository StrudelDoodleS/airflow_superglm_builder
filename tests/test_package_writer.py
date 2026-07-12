from pathlib import Path

import pytest

from pricing_pipeline.publishing.lifecycle import PublishResult
from pricing_pipeline.publishing.model_registry import ModelRegistryError
from pricing_pipeline.publishing import package_writer
from pricing_pipeline.publishing.package_writer import load_staging_to_rating_package
from pricing_pipeline.publishing.package_writer import publish_rating_package


def test_package_writer_rejects_legacy_pointer_deployment():
    args = type(
        "Args",
        (),
        {
            "export_id": "export-1",
            "created_by": "airflow",
            "package_status": "PUBLISHED",
            "set_pointer": "MTPL_FREQ_UAT",
        },
    )()

    with pytest.raises(ValueError, match="deploy"):
        load_staging_to_rating_package(object(), args)


def test_package_writer_does_not_write_deployment_tables_during_publish():
    writer = Path("pricing_pipeline/publishing/package_writer.py").read_text(encoding="utf-8")

    assert "PRICING_MODEL_DEPLOYMENT" not in writer
    assert "PRICING_PACKAGE_POINTER" not in writer


def test_publish_rating_package_builds_args_without_deployment_pointer(monkeypatch):
    captured = []

    def fake_load(engine, args):
        captured.append((engine, args))
        args.package_version = 3
        return 42

    monkeypatch.setattr(
        "pricing_pipeline.publishing.package_writer.load_staging_to_rating_package",
        fake_load,
    )
    engine = object()

    result = publish_rating_package(
        engine,
        export_id="export-1",
        created_by="airflow",
        package_status="PUBLISHED",
    )

    assert result == PublishResult(
        mlflow_run_id="",
        export_id="export-1",
        rate_package_id=42,
        package_version=3,
        rating_workbook_path="",
    )
    args = captured[0][1]
    assert args.export_id == "export-1"
    assert args.created_by == "airflow"
    assert args.package_status == "PUBLISHED"
    assert args.set_pointer is None


def test_publish_rating_package_passes_child_revision_contract(monkeypatch):
    captured = []

    def fake_load(engine, args):
        captured.append(args)
        args.package_version = 8
        return 108

    monkeypatch.setattr(package_writer, "load_staging_to_rating_package", fake_load)

    def validator(connection, rate_package_id):
        return None

    def write_lineage(connection, rate_package_id):
        return None

    publish_rating_package(
        object(),
        export_id="editor__submission_1",
        created_by="analyst@example.test",
        parent_rate_package_id=107,
        revision_metadata_json='{"kind":"SUPERGLM_EDITOR"}',
        draft_validator=validator,
        package_lineage_writer=write_lineage,
    )

    args = captured[0]
    assert args.parent_rate_package_id == 107
    assert args.revision_metadata_json == '{"kind":"SUPERGLM_EDITOR"}'
    assert args.draft_validator is validator
    assert args.package_lineage_writer is write_lineage


def test_publish_rating_package_reports_existing_source_export(monkeypatch):
    def fake_load(engine, args):
        args.package_version = 3
        args.package_status = "DRAFT"
        args.was_existing = True
        return 42

    monkeypatch.setattr(
        "pricing_pipeline.publishing.package_writer.load_staging_to_rating_package",
        fake_load,
    )

    result = publish_rating_package(
        object(),
        export_id="export-1",
        created_by="airflow",
        package_status="PUBLISHED",
    )

    assert result.was_existing is True
    assert result.package_status == "DRAFT"


class _FakeMetaResult:
    def mappings(self):
        return self

    def one(self):
        return {
            "export_id": "export-1",
            "model_id": None,
            "model_name": "MTPL_FREQ",
            "model_version": "20260529",
            "base_rate": 1.0,
            "effective_from_date": "2026-05-29",
            "effective_to_date": None,
        }


class _FakePublishConnection:
    def execute(self, statement, params=None):
        if "FROM pricing_stg.STG_RATING_EXPORT" in str(statement):
            return _FakeMetaResult()
        raise AssertionError("publish should stop before writing package rows")


class _FakePublishBegin:
    def __enter__(self):
        return _FakePublishConnection()

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakePublishEngine:
    def begin(self):
        return _FakePublishBegin()


def _staged_meta(**overrides):
    row = {
        "export_id": "export-1",
        "model_id": 17,
        "model_name": "MTPL_FREQ",
        "model_version": "20260529",
        "base_rate": 1.0,
        "effective_from_date": "2026-05-29",
        "effective_to_date": None,
        "source_file": "/tmp/export/rating_tables.xlsx",
        "publication_receipt_json": None,
        "publication_receipt_sha256": None,
        "package_metadata_json": None,
        "offset_handling": None,
        "offset_factor_name": None,
        "offset_source_name": None,
        "offset_label": None,
        "metadata_origin": None,
    }
    row.update(overrides)
    return row


def _existing_package(**overrides):
    row = {
        "rate_package_id": 42,
        "package_version": 3,
        "model_id": 17,
        "model_name": "MTPL_FREQ",
        "model_version": "20260529",
        "effective_from_date": "2026-05-29",
        "effective_to_date": None,
        "package_status": "DRAFT",
        "source_export_id": "export-1",
        "source_file": "/tmp/export/rating_tables.xlsx",
        "publication_receipt_sha256": None,
        "parent_rate_package_id": None,
        "revision_metadata_json": None,
    }
    row.update(overrides)
    return row


class _FakeExistingPackageResult:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def one_or_none(self):
        return self.row


class _FakeMetaWithModelResult:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def one(self):
        return self.row


class _FakeExistingPackageConnection:
    def __init__(self, staged_meta=None, existing_package=None):
        self.statements = []
        self.staged_meta = staged_meta or _staged_meta()
        self.existing_package = existing_package or _existing_package()

    def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append((sql, params))
        if "FROM pricing_stg.STG_RATING_EXPORT" in sql:
            return _FakeMetaWithModelResult(self.staged_meta)
        if "source_export_id = :export_id" in sql:
            return _FakeExistingPackageResult(self.existing_package)
        raise AssertionError("existing export publish should stop before package insert")


class _FakeExistingPackageBegin:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeExistingPackageEngine:
    def __init__(self, staged_meta=None, existing_package=None):
        self.connection = _FakeExistingPackageConnection(
            staged_meta=staged_meta,
            existing_package=existing_package,
        )

    def begin(self):
        return _FakeExistingPackageBegin(self.connection)


class _FakeScalarResult:
    def __init__(self, value=None):
        self.value = value

    def scalar_one(self):
        return self.value

    def scalar_one_or_none(self):
        return self.value


class _FakeNewPackageConnection:
    def __init__(self):
        self.statements = []

    def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append((sql, params))
        if "FROM pricing_stg.STG_RATING_EXPORT" in sql:
            return _FakeMetaWithModelResult(_staged_meta())
        if "source_export_id = :export_id" in sql:
            return _FakeExistingPackageResult(None)
        if "SELECT ISNULL(MAX(package_version), 0) + 1" in sql:
            return _FakeScalarResult(3)
        if "INSERT INTO pricing.PRICING_RATE_PACKAGE" in sql:
            return _FakeScalarResult(42)
        return _FakeScalarResult()


class _FakeNewPackageBegin:
    def __init__(self, connection):
        self.connection = connection
        self.exit_exception = None

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        self.exit_exception = exc
        return False


class _FakeNewPackageEngine:
    def __init__(self):
        self.connection = _FakeNewPackageConnection()
        self.transaction = _FakeNewPackageBegin(self.connection)

    def begin(self):
        return self.transaction


def _new_package_args(**overrides):
    values = {
        "export_id": "export-1",
        "created_by": "airflow",
        "package_status": "PUBLISHED",
        "set_pointer": None,
    }
    values.update(overrides)
    args = type("Args", (), {})()
    for name, value in values.items():
        setattr(args, name, value)
    return args


def test_package_lineage_writer_runs_inside_transaction_before_final_status():
    engine = _FakeNewPackageEngine()
    events = []

    def validate_draft(connection, rate_package_id):
        assert connection is engine.connection
        assert rate_package_id == 42
        events.append("validate")

    def write_lineage(connection, rate_package_id):
        assert connection is engine.connection
        assert rate_package_id == 42
        assert not any(
            "UPDATE pricing.PRICING_RATE_PACKAGE" in sql
            for sql, _params in connection.statements
        )
        events.append("lineage")

    args = _new_package_args(
        draft_validator=validate_draft,
        package_lineage_writer=write_lineage,
    )

    assert load_staging_to_rating_package(engine, args) == 42

    status_index = next(
        index
        for index, (sql, _params) in enumerate(engine.connection.statements)
        if "UPDATE pricing.PRICING_RATE_PACKAGE" in sql
    )
    assert events == ["validate", "lineage"]
    assert engine.connection.statements[status_index][1] == {
        "package_status": "PUBLISHED",
        "rate_package_id": 42,
    }


def test_package_lineage_failure_prevents_final_status_and_rolls_back_transaction():
    engine = _FakeNewPackageEngine()
    failure = RuntimeError("lineage write failed")

    def fail_lineage(connection, rate_package_id):
        assert connection is engine.connection
        assert rate_package_id == 42
        raise failure

    args = _new_package_args(package_lineage_writer=fail_lineage)

    with pytest.raises(RuntimeError, match="lineage write failed"):
        load_staging_to_rating_package(engine, args)

    assert engine.transaction.exit_exception is failure
    assert not any(
        "UPDATE pricing.PRICING_RATE_PACKAGE" in sql
        for sql, _params in engine.connection.statements
    )


def test_existing_compatible_package_invokes_lineage_writer_on_same_connection():
    engine = _FakeExistingPackageEngine()
    calls = []

    def write_lineage(connection, rate_package_id):
        calls.append((connection, rate_package_id))

    args = _new_package_args(package_lineage_writer=write_lineage)

    assert load_staging_to_rating_package(engine, args) == 42

    assert calls == [(engine.connection, 42)]
    assert args.was_existing is True


def test_package_writer_rejects_staged_export_without_registered_model_id(monkeypatch):
    args = type(
        "Args",
        (),
        {
            "export_id": "export-1",
            "created_by": "airflow",
            "package_status": "PUBLISHED",
            "set_pointer": None,
        },
    )()
    monkeypatch.setattr(
        package_writer,
        "ensure_pricing_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("publish must not create model registry rows")
        ),
        raising=False,
    )

    with pytest.raises(ModelRegistryError, match="missing model_id"):
        load_staging_to_rating_package(_FakePublishEngine(), args)


def test_package_writer_returns_existing_package_for_existing_source_export():
    args = type(
        "Args",
        (),
        {
            "export_id": "export-1",
            "created_by": "airflow",
            "package_status": "PUBLISHED",
            "set_pointer": None,
        },
    )()
    engine = _FakeExistingPackageEngine()

    rate_package_id = load_staging_to_rating_package(engine, args)

    assert rate_package_id == 42
    assert args.package_version == 3
    assert args.package_status == "DRAFT"
    assert args.was_existing is True
    assert any(
        "source_export_id = :export_id" in sql
        and params == {"model_id": 17, "export_id": "export-1"}
        for sql, params in engine.connection.statements
    )
    assert not any(
        "INSERT INTO pricing.PRICING_RATE_PACKAGE" in sql
        for sql, _params in engine.connection.statements
    )


def test_package_writer_rejects_existing_source_export_with_different_model_version():
    args = type(
        "Args",
        (),
        {
            "export_id": "export-1",
            "created_by": "airflow",
            "package_status": "PUBLISHED",
            "set_pointer": None,
        },
    )()
    engine = _FakeExistingPackageEngine(
        staged_meta=_staged_meta(model_version="20260603"),
        existing_package=_existing_package(model_version="20260529"),
    )

    with pytest.raises(ValueError, match="model_version"):
        load_staging_to_rating_package(engine, args)

    assert not any(
        "INSERT INTO pricing.PRICING_RATE_PACKAGE" in sql
        for sql, _params in engine.connection.statements
    )


def test_package_writer_rejects_existing_source_export_with_different_effective_from():
    args = type(
        "Args",
        (),
        {
            "export_id": "export-1",
            "created_by": "airflow",
            "package_status": "PUBLISHED",
            "set_pointer": None,
        },
    )()
    engine = _FakeExistingPackageEngine(
        staged_meta=_staged_meta(effective_from_date="2026-06-03"),
        existing_package=_existing_package(effective_from_date="2026-05-29"),
    )

    with pytest.raises(ValueError, match="effective_from_date"):
        load_staging_to_rating_package(engine, args)

    assert not any(
        "INSERT INTO pricing.PRICING_RATE_PACKAGE" in sql
        for sql, _params in engine.connection.statements
    )


def test_package_writer_rejects_existing_source_export_with_different_source_file():
    args = type(
        "Args",
        (),
        {
            "export_id": "export-1",
            "created_by": "airflow",
            "package_status": "PUBLISHED",
            "set_pointer": None,
        },
    )()
    engine = _FakeExistingPackageEngine(
        staged_meta=_staged_meta(source_file="/tmp/new/rating_tables.xlsx"),
        existing_package=_existing_package(source_file="/tmp/old/rating_tables.xlsx"),
    )

    with pytest.raises(ValueError, match="source_file"):
        load_staging_to_rating_package(engine, args)

    assert not any(
        "INSERT INTO pricing.PRICING_RATE_PACKAGE" in sql
        for sql, _params in engine.connection.statements
    )


def test_package_writer_rejects_existing_source_export_with_different_receipt_hash():
    args = type(
        "Args",
        (),
        {
            "export_id": "export-1",
            "created_by": "airflow",
            "package_status": "PUBLISHED",
            "set_pointer": None,
        },
    )()
    engine = _FakeExistingPackageEngine(
        staged_meta=_staged_meta(publication_receipt_sha256="a" * 64),
        existing_package=_existing_package(publication_receipt_sha256="b" * 64),
    )

    with pytest.raises(ValueError, match="publication_receipt_sha256"):
        load_staging_to_rating_package(engine, args)

    assert not any(
        "INSERT INTO pricing.PRICING_RATE_PACKAGE" in sql
        for sql, _params in engine.connection.statements
    )


def test_package_writer_allows_existing_source_export_when_old_source_file_is_unknown():
    args = type(
        "Args",
        (),
        {
            "export_id": "export-1",
            "created_by": "airflow",
            "package_status": "PUBLISHED",
            "set_pointer": None,
        },
    )()
    engine = _FakeExistingPackageEngine(
        staged_meta=_staged_meta(source_file="/tmp/new/rating_tables.xlsx"),
        existing_package=_existing_package(source_file=None),
    )

    rate_package_id = load_staging_to_rating_package(engine, args)

    assert rate_package_id == 42
    assert args.was_existing is True


def test_package_writer_publishes_receipt_and_term_metadata_columns():
    writer = Path("pricing_pipeline/publishing/package_writer.py").read_text(encoding="utf-8")

    assert "publication_receipt_json" in writer
    assert "publication_receipt_sha256" in writer
    assert "package_metadata_json" in writer
    assert "revision_metadata_json" in writer
    assert "offset_handling" in writer
    assert "STG_TERM_METADATA" in writer
    assert "term_metadata_json" in writer
