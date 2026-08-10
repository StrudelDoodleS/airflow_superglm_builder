from inspect import signature
from pathlib import Path
from types import MappingProxyType

import pytest

from pricing_pipeline.publishing.model_registry import ModelRegistryError
from pricing_pipeline.publishing.package_writer import publish_rating_package


def load_staging_to_rating_package(engine, args):
    """Adapt old test fixtures to the explicit production API."""
    result = publish_rating_package(
        engine,
        export_id=args.export_id,
        created_by=args.created_by,
        parent_rate_package_id=getattr(args, "parent_rate_package_id", None),
        revision_metadata=getattr(args, "revision_metadata", None),
        draft_validator=getattr(args, "draft_validator", None),
        package_lineage_writer=getattr(args, "package_lineage_writer", None),
        expected_staged_metadata=getattr(args, "expected_staged_metadata", None),
    )
    args.package_version = result.package_version
    args.package_status = result.package_status
    args.was_existing = result.was_existing
    args.model_run_id = result.model_run_id
    return result.rate_package_id


def test_package_writer_does_not_write_deployment_tables_during_publish():
    writer = Path("pricing_pipeline/publishing/package_writer.py").read_text(encoding="utf-8")

    assert "PRICING_MODEL_DEPLOYMENT" not in writer
    assert "PRICING_PACKAGE_POINTER" not in writer


def test_publish_rating_package_accepts_revision_mapping_without_public_status():
    parameters = signature(publish_rating_package).parameters

    assert "package_status" not in parameters
    assert "revision_metadata" in parameters
    assert "revision_metadata_json" not in parameters


def test_package_writer_canonicalises_revision_metadata_mapping_once():
    engine = _FakeNewPackageEngine()
    args = _new_package_args(
        revision_metadata={"unicode": "München", "kind": "SUPERGLM_EDITOR"}
    )

    load_staging_to_rating_package(engine, args)

    package_insert = next(
        (sql, params)
        for sql, params in engine.connection.statements
        if "INSERT INTO pricing.PRICING_RATE_PACKAGE" in sql
    )
    assert package_insert[1]["revision_metadata_json"] == (
        '{"kind":"SUPERGLM_EDITOR","unicode":"München"}'
    )


def test_package_writer_accepts_non_dict_revision_metadata_mapping():
    engine = _FakeNewPackageEngine()
    args = _new_package_args(
        revision_metadata=MappingProxyType({"kind": "SUPERGLM_EDITOR"})
    )

    load_staging_to_rating_package(engine, args)

    package_insert = next(
        (sql, params)
        for sql, params in engine.connection.statements
        if "INSERT INTO pricing.PRICING_RATE_PACKAGE" in sql
    )
    assert package_insert[1]["revision_metadata_json"] == '{"kind":"SUPERGLM_EDITOR"}'


def test_package_writer_rejects_non_mapping_revision_metadata():
    with pytest.raises(ValueError, match="revision_metadata must be a mapping"):
        publish_rating_package(
            _FakeNewPackageEngine(),
            export_id="export-1",
            revision_metadata='{"kind":"SUPERGLM_EDITOR"}',
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_package_writer_rejects_non_finite_revision_metadata(value):
    with pytest.raises(ValueError, match="finite numbers"):
        publish_rating_package(
            _FakeNewPackageEngine(),
            export_id="export-1",
            revision_metadata={"metric": value},
        )


@pytest.mark.parametrize(
    "revision_metadata",
    [
        {1: "value"},
        {"nested": {1: "value"}},
    ],
    ids=["top-level", "nested"],
)
def test_package_writer_rejects_non_string_revision_metadata_keys(revision_metadata):
    with pytest.raises(ValueError, match="keys must be strings"):
        publish_rating_package(
            _FakeNewPackageEngine(),
            export_id="export-1",
            revision_metadata=revision_metadata,
        )


def test_package_writer_rejects_non_json_serializable_revision_metadata():
    with pytest.raises(ValueError, match="JSON-serializable values"):
        publish_rating_package(
            _FakeNewPackageEngine(),
            export_id="export-1",
            revision_metadata={"unsupported": object()},
        )


@pytest.fixture
def emitted_band_compile_sql():
    engine = _FakeNewPackageEngine()

    load_staging_to_rating_package(engine, _new_package_args())

    band_sql = next(
        sql
        for sql, _params in engine.connection.statements
        if "INSERT INTO pricing.PRICING_COMPILED_1D_RATE_BAND" in sql
    )
    return " ".join(band_sql.split())


def test_package_writer_compiles_only_interval_offset_factors_as_bands(
    emitted_band_compile_sql,
):
    assert (
        "t.term_type = 'OFFSET_FACTOR' AND ls.level_set_type IN ('NUMERIC_BAND', 'SPLINE_GRID_1D')"
    ) in emitted_band_compile_sql


def test_package_writer_opens_only_the_terminal_compiled_band(emitted_band_compile_sql):
    # SuperGLM assigns x.max to its final [left, max) bin; the compiled terminal
    # must therefore be open-ended while every internal upper bound stays audited.
    assert (
        "CASE WHEN ROW_NUMBER() OVER ( PARTITION BY t.term_id ORDER BY "
        "CASE WHEN fl.lower_bound IS NULL THEN 1 ELSE 0 END, "
        "fl.lower_bound DESC, COALESCE(fl.order_index, 0) DESC, "
        "fl.feature_level_id DESC ) = 1 THEN NULL ELSE fl.upper_bound END"
    ) in emitted_band_compile_sql


def test_package_writer_rejects_replaced_staging_before_lineage_write():
    engine = _FakeExistingPackageEngine(
        staged_meta=_staged_meta(source_file="/tmp/other/rating_tables.xlsx"),
    )
    lineage_calls = []
    args = _new_package_args(
        expected_staged_metadata={
            "export_id": "export-1",
            "model_id": 17,
            "model_name": "MTPL_FREQ",
            "model_version": "20260529",
            "effective_from_date": "2026-05-29",
            "effective_to_date": None,
            "source_file": "/tmp/export/rating_tables.xlsx",
            "publication_receipt_sha256": None,
            "staging_content_sha256": "a" * 64,
        },
        package_lineage_writer=lambda *args: lineage_calls.append(args),
    )

    with pytest.raises(ValueError, match="staged export changed.*source_file"):
        load_staging_to_rating_package(engine, args)

    assert lineage_calls == []
    assert not any(
        "source_export_id = :export_id" in sql for sql, _params in engine.connection.statements
    )


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
        "staging_content_sha256": "a" * 64,
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
        "staging_content_sha256": "a" * 64,
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
        if sql.lstrip().startswith("DELETE FROM pricing_stg."):
            return _FakeScalarResult()
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
    def __init__(self, reservation=None):
        self.statements = []
        self.reservation = reservation or {
            "model_id": 17,
            "export_id": "export-1",
            "model_version": "20260529",
        }

    def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append((sql, params))
        if "FROM pricing_stg.STG_RATING_EXPORT" in sql:
            return _FakeMetaWithModelResult(_staged_meta())
        if "source_export_id = :export_id" in sql:
            return _FakeExistingPackageResult(None)
        if "FROM pricing.PRICING_MODEL_VERSION_RESERVATION" in sql:
            return _FakeExistingPackageResult(self.reservation)
        if "SELECT ISNULL(MAX(package_version), 0) + 1" in sql:
            return _FakeScalarResult(3)
        if "INSERT INTO pricing.PRICING_RATE_PACKAGE" in sql:
            return _FakeScalarResult(42)
        return _FakeScalarResult()


class _FakeNewPackageBegin:
    def __init__(self, connection):
        self.connection = connection
        self.exit_exception = None
        self.active = False

    def __enter__(self):
        self.active = True
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        self.exit_exception = exc
        self.active = False
        return False


class _FakeNewPackageEngine:
    def __init__(self, reservation=None):
        self.connection = _FakeNewPackageConnection(reservation=reservation)
        self.transaction = _FakeNewPackageBegin(self.connection)

    def begin(self):
        return self.transaction


class _FakeEquivalentPackageConnection(_FakeNewPackageConnection):
    def execute(self, statement, params=None):
        sql = str(statement)
        if "FROM pricing.MODEL_RUN AS mr" in sql:
            self.statements.append((sql, params))
            return _FakeExistingPackageResult(
                {
                    "rate_package_id": 41,
                    "package_version": 2,
                    "package_status": "PUBLISHED",
                    "source_export_id": "original-export",
                    "source_file": "/tmp/original/rating_tables.xlsx",
                    "model_run_id": 901,
                }
            )
        return super().execute(statement, params)


def _new_package_args(**overrides):
    values = {
        "export_id": "export-1",
        "created_by": "airflow",
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
            "UPDATE pricing.PRICING_RATE_PACKAGE" in sql for sql, _params in connection.statements
        )
        events.append("lineage")

    args = _new_package_args(
        draft_validator=validate_draft,
        package_lineage_writer=write_lineage,
    )

    assert load_staging_to_rating_package(engine, args) == 42

    package_insert = next(
        (sql, params)
        for sql, params in engine.connection.statements
        if "INSERT INTO pricing.PRICING_RATE_PACKAGE" in sql
    )
    assert "staging_content_sha256" in package_insert[0]
    assert package_insert[1]["staging_content_sha256"] == "a" * 64
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


def test_successful_publish_deletes_staging_payload_after_final_status_only():
    engine = _FakeNewPackageEngine()

    load_staging_to_rating_package(engine, _new_package_args())

    statements = [sql for sql, _params in engine.connection.statements]
    status_index = next(
        index
        for index, sql in enumerate(statements)
        if "UPDATE pricing.PRICING_RATE_PACKAGE" in sql
    )
    payload_deletes = [
        (index, " ".join(sql.split()))
        for index, sql in enumerate(statements)
        if sql.lstrip().startswith("DELETE FROM pricing_stg.")
    ]

    assert payload_deletes == [
        (
            status_index + 1,
            "DELETE FROM pricing_stg.STG_TERM_METADATA WHERE export_id = :export_id",
        ),
        (
            status_index + 2,
            "DELETE FROM pricing_stg.STG_CELL_LEVEL WHERE export_id = :export_id",
        ),
        (
            status_index + 3,
            "DELETE FROM pricing_stg.STG_RATE_CELL WHERE export_id = :export_id",
        ),
    ]
    assert all(
        engine.connection.statements[index][1] == {"export_id": "export-1"}
        for index, _sql in payload_deletes
    )
    assert not any("DELETE FROM pricing_stg.STG_RATING_EXPORT" in sql for sql in statements)


def test_package_lineage_writer_return_id_is_exposed_from_same_transaction():
    engine = _FakeNewPackageEngine()

    def write_lineage(connection, rate_package_id):
        assert engine.transaction.active
        assert connection is engine.connection
        assert rate_package_id == 42
        return 908

    result = publish_rating_package(
        engine,
        export_id="export-1",
        created_by="airflow",
        package_lineage_writer=write_lineage,
    )

    assert engine.transaction.active is False
    assert result.model_run_id == 908


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
    assert not any(
        "DELETE FROM pricing_stg." in sql for sql, _params in engine.connection.statements
    )


def test_existing_compatible_package_does_not_rewrite_lineage():
    engine = _FakeExistingPackageEngine()
    calls = []

    def write_lineage(connection, rate_package_id):
        calls.append((connection, rate_package_id))

    args = _new_package_args(package_lineage_writer=write_lineage)

    assert load_staging_to_rating_package(engine, args) == 42

    assert calls == []
    assert args.was_existing is True
    assert args.model_run_id is None


def test_existing_published_package_cleans_retry_payload_but_retains_header():
    engine = _FakeExistingPackageEngine(
        existing_package=_existing_package(package_status="PUBLISHED"),
    )

    result = publish_rating_package(engine, export_id="export-1")

    assert result.was_existing is True
    statements = [sql for sql, _params in engine.connection.statements]
    assert [
        " ".join(sql.split())
        for sql in statements
        if sql.lstrip().startswith("DELETE FROM pricing_stg.")
    ] == [
        "DELETE FROM pricing_stg.STG_TERM_METADATA WHERE export_id = :export_id",
        "DELETE FROM pricing_stg.STG_CELL_LEVEL WHERE export_id = :export_id",
        "DELETE FROM pricing_stg.STG_RATE_CELL WHERE export_id = :export_id",
    ]
    assert not any("DELETE FROM pricing_stg.STG_RATING_EXPORT" in sql for sql in statements)


def test_existing_draft_package_retains_staging_payload_for_recovery():
    engine = _FakeExistingPackageEngine(
        existing_package=_existing_package(package_status="DRAFT"),
    )

    result = publish_rating_package(engine, export_id="export-1")

    assert result.package_status == "DRAFT"
    assert not any(
        "DELETE FROM pricing_stg." in sql for sql, _params in engine.connection.statements
    )


def test_equivalent_published_package_cleans_rejected_attempt_payload():
    engine = _FakeNewPackageEngine()
    engine.connection = _FakeEquivalentPackageConnection()
    engine.transaction = _FakeNewPackageBegin(engine.connection)

    result = publish_rating_package(
        engine,
        export_id="export-1",
        equivalence_key={
            "manifest_id": "manifest-1",
            "model_kind": "RAW",
            "model_equivalence_sha256": "b" * 64,
        },
    )

    assert result.was_existing is True
    assert result.deduplicated is True
    assert result.export_id == "original-export"
    statements = [sql for sql, _params in engine.connection.statements]
    assert (
        len([sql for sql in statements if sql.lstrip().startswith("DELETE FROM pricing_stg.")]) == 3
    )
    assert not any("INSERT INTO pricing.PRICING_RATE_PACKAGE" in sql for sql in statements)


def test_package_writer_rejects_replaced_staging_rate_content():
    engine = _FakeExistingPackageEngine(
        staged_meta=_staged_meta(staging_content_sha256="b" * 64),
    )
    lineage_calls = []
    args = _new_package_args(
        expected_staged_metadata={"staging_content_sha256": "a" * 64},
        package_lineage_writer=lambda *args: lineage_calls.append(args),
    )

    with pytest.raises(ValueError, match="staged export changed.*staging_content_sha256"):
        load_staging_to_rating_package(engine, args)

    assert lineage_calls == []


def test_package_writer_rejects_existing_package_built_from_other_rate_content():
    engine = _FakeExistingPackageEngine(
        staged_meta=_staged_meta(staging_content_sha256="b" * 64),
        existing_package=_existing_package(staging_content_sha256="a" * 64),
    )
    args = _new_package_args(
        expected_staged_metadata={"staging_content_sha256": "b" * 64},
    )

    with pytest.raises(ValueError, match="incompatible metadata.*staging_content_sha256"):
        load_staging_to_rating_package(engine, args)


def test_package_writer_reuses_legacy_package_without_staging_digest():
    engine = _FakeExistingPackageEngine(
        existing_package=_existing_package(staging_content_sha256=None),
    )
    args = _new_package_args(
        expected_staged_metadata={"staging_content_sha256": "a" * 64},
    )

    assert load_staging_to_rating_package(engine, args) == 42
    assert args.was_existing is True


def test_package_writer_reserves_staged_version_for_direct_root_publication():
    engine = _FakeNewPackageEngine(reservation={})
    engine.connection.reservation = None

    assert load_staging_to_rating_package(engine, _new_package_args()) == 42

    reservation_insert = next(
        statement
        for statement in engine.connection.statements
        if "INSERT INTO pricing.PRICING_MODEL_VERSION_RESERVATION" in statement[0]
    )
    package_insert = next(
        statement
        for statement in engine.connection.statements
        if "INSERT INTO pricing.PRICING_RATE_PACKAGE" in statement[0]
    )
    assert reservation_insert[1] == {
        "model_id": 17,
        "export_id": "export-1",
        "model_version": "20260529",
    }
    assert engine.connection.statements.index(
        reservation_insert
    ) < engine.connection.statements.index(package_insert)


def test_package_writer_rejects_root_package_with_different_reserved_version():
    engine = _FakeNewPackageEngine(
        reservation={
            "model_id": 17,
            "export_id": "export-1",
            "model_version": "20260530",
        }
    )

    with pytest.raises(ValueError, match="reserved model_version.*20260530.*20260529"):
        load_staging_to_rating_package(engine, _new_package_args())

    assert not any(
        "INSERT INTO pricing.PRICING_RATE_PACKAGE" in sql
        for sql, _params in engine.connection.statements
    )


def test_package_writer_rejects_staged_export_without_registered_model_id():
    args = _new_package_args()
    with pytest.raises(ModelRegistryError, match="missing model_id"):
        load_staging_to_rating_package(_FakePublishEngine(), args)


def test_package_writer_returns_existing_package_for_existing_source_export():
    args = _new_package_args()
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
    args = _new_package_args()
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
    args = _new_package_args()
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
    args = _new_package_args()
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
    args = _new_package_args()
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
    args = _new_package_args()
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
