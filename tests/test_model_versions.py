from __future__ import annotations

import pytest

from pricing_pipeline.publishing.model_versions import resolve_model_version_for_export


class FakeScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value

    def scalar_one_or_none(self):
        return self.value


class FakeScalarsResult:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return iter(self.values)


class FakeConnection:
    def __init__(
        self,
        *,
        existing_version=None,
        versions=(),
        canonical_fingerprint_version=None,
        actual_database="PricingLab",
    ):
        self.existing_version = existing_version
        self.versions = versions
        self.canonical_fingerprint_version = canonical_fingerprint_version
        self.actual_database = actual_database
        self.reservations: dict[str, str] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute(self, statement, params=None):
        sql = str(statement)
        values = dict(params or {})
        self.calls.append((sql, values))
        if "DB_NAME()" in sql:
            return FakeScalarResult(self.actual_database)
        if (
            "FROM pricing.PRICING_MODEL AS pm" in sql
            or "FROM python_pricing.PRICING_MODEL AS pm" in sql
        ):
            return FakeScalarResult(17)
        if "build_fingerprint_sha256 = :build_fingerprint_sha256" in sql:
            return FakeScalarResult(self.canonical_fingerprint_version)
        if sql.lstrip().startswith("INSERT INTO"):
            self.reservations[str(values["export_id"])] = str(values["model_version"])
            return FakeScalarResult(None)
        if "PRICING_MODEL_VERSION_RESERVATION" in sql and "export_id = :export_id" in sql:
            existing = self.existing_version or self.reservations.get(str(values["export_id"]))
            return FakeScalarsResult([] if existing is None else [existing])
        if "PRICING_MODEL_VERSION_RESERVATION" in sql:
            return FakeScalarsResult([*self.versions, *self.reservations.values()])
        if "rp.source_export_id = :export_id" in sql:
            return FakeScalarResult(self.existing_version)
        return FakeScalarsResult(self.versions)


class FakeBegin:
    def __init__(self, connection: FakeConnection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeEngine:
    def __init__(
        self,
        *,
        existing_version=None,
        versions=(),
        canonical_fingerprint_version=None,
        pricing_schema="pricing",
        actual_database="PricingLab",
    ):
        self.connection = FakeConnection(
            existing_version=existing_version,
            versions=versions,
            canonical_fingerprint_version=canonical_fingerprint_version,
            actual_database=actual_database,
        )
        self._execution_options = {"pricing_schema": pricing_schema}

    def begin(self):
        return FakeBegin(self.connection)


def test_model_version_queries_respect_configured_pricing_schema():
    engine = FakeEngine(existing_version="v3", pricing_schema="python_pricing")

    resolve_model_version_for_export(
        engine,
        model_name="MY_MODEL",
        export_id="my_model__run_1",
        build_fingerprint_sha256="a" * 64,
        expected_database="PricingLab",
    )

    sql, params = engine.connection.calls[1]
    assert "FROM python_pricing.PRICING_MODEL AS pm" in sql
    assert "WITH (UPDLOCK, HOLDLOCK)" in sql
    assert params == {"model_name": "MY_MODEL"}


def test_resolve_model_version_for_export_reuses_existing_export_version():
    engine = FakeEngine(existing_version="v7", versions=["v1", "v2"])

    assert (
        resolve_model_version_for_export(
            engine,
            model_name="MY_MODEL",
            export_id="my_model__run_1",
            build_fingerprint_sha256="a" * 64,
            expected_database="PricingLab",
        )
        == "v7"
    )
    assert len(engine.connection.calls) == 4


def test_resolve_model_version_for_export_allocates_next_version_for_new_export():
    engine = FakeEngine(existing_version=None, versions=["v1", "v4"])

    assert (
        resolve_model_version_for_export(
            engine,
            model_name="MY_MODEL",
            export_id="my_model__run_2",
            build_fingerprint_sha256="b" * 64,
            expected_database="PricingLab",
        )
        == "v5"
    )
    assert engine.connection.reservations == {"my_model__run_2": "v5"}
    assert len(engine.connection.calls) == 6


def test_resolve_model_version_reserves_distinct_versions_before_publication():
    engine = FakeEngine()

    first = resolve_model_version_for_export(
        engine,
        model_name="MY_MODEL",
        export_id="my_model__run_1",
        build_fingerprint_sha256="a" * 64,
        expected_database="PricingLab",
    )
    second = resolve_model_version_for_export(
        engine,
        model_name="MY_MODEL",
        export_id="my_model__run_2",
        build_fingerprint_sha256="b" * 64,
        expected_database="PricingLab",
    )
    retry = resolve_model_version_for_export(
        engine,
        model_name="MY_MODEL",
        export_id="my_model__run_1",
        build_fingerprint_sha256="a" * 64,
        expected_database="PricingLab",
    )

    assert (first, second, retry) == ("v1", "v2", "v1")
    assert engine.connection.reservations == {
        "my_model__run_1": "v1",
        "my_model__run_2": "v2",
    }


def test_resolve_model_version_reuses_canonical_root_fingerprint_before_reservation():
    engine = FakeEngine(canonical_fingerprint_version="v7", versions=["v1", "v6"])

    resolved = resolve_model_version_for_export(
        engine,
        model_name="MY_MODEL",
        export_id="retry-attempt-export",
        build_fingerprint_sha256="c" * 64,
        expected_database="PricingLab",
    )

    assert resolved == "v7"
    assert engine.connection.reservations == {}
    fingerprint_query_index = next(
        index
        for index, (sql, _params) in enumerate(engine.connection.calls)
        if "build_fingerprint_sha256 = :build_fingerprint_sha256" in sql
    )
    assert fingerprint_query_index == 2


def test_resolve_model_version_rejects_reservation_disagreeing_with_canonical_root():
    engine = FakeEngine(
        existing_version="v8",
        canonical_fingerprint_version="v7",
    )

    with pytest.raises(RuntimeError, match="canonical root package and.*reservation disagree"):
        resolve_model_version_for_export(
            engine,
            model_name="MY_MODEL",
            export_id="retry-attempt-export",
            build_fingerprint_sha256="c" * 64,
            expected_database="PricingLab",
        )


def test_model_version_reservation_rechecks_database_before_lock_or_insert():
    engine = FakeEngine(actual_database="OtherDb")

    with pytest.raises(RuntimeError, match="expected 'PricingLab'.*connected to 'OtherDb'"):
        resolve_model_version_for_export(
            engine,
            model_name="MY_MODEL",
            export_id="my_model__run_1",
            build_fingerprint_sha256="a" * 64,
            expected_database="PricingLab",
        )

    assert len(engine.connection.calls) == 1
    assert "DB_NAME()" in engine.connection.calls[0][0]
    assert engine.connection.reservations == {}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "model_name": None,
                "export_id": "export-1",
                "build_fingerprint_sha256": "a" * 64,
                "expected_database": "PricingLab",
            },
            "model_name",
        ),
        (
            {
                "model_name": "",
                "export_id": "export-1",
                "build_fingerprint_sha256": "a" * 64,
                "expected_database": "PricingLab",
            },
            "model_name",
        ),
        (
            {
                "model_name": "MY_MODEL",
                "export_id": "   ",
                "build_fingerprint_sha256": "a" * 64,
                "expected_database": "PricingLab",
            },
            "export_id",
        ),
        (
            {
                "model_name": "MY_MODEL",
                "export_id": None,
                "build_fingerprint_sha256": "a" * 64,
                "expected_database": "PricingLab",
            },
            "export_id",
        ),
        (
            {
                "model_name": "MY_MODEL",
                "export_id": "export-1",
                "build_fingerprint_sha256": "not-a-digest",
                "expected_database": "PricingLab",
            },
            "build_fingerprint_sha256",
        ),
    ],
)
def test_resolve_model_version_for_export_rejects_blank_identity(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_model_version_for_export(FakeEngine(), **kwargs)
