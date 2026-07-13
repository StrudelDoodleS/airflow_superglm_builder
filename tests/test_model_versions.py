from __future__ import annotations

import pytest

from pricing_pipeline.publishing.model_versions import (
    existing_model_version_for_export,
    next_trained_model_version,
    resolve_model_version_for_export,
)


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
    def __init__(self, *, existing_version=None, versions=()):
        self.existing_version = existing_version
        self.versions = versions
        self.reservations: dict[str, str] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute(self, statement, params=None):
        sql = str(statement)
        values = dict(params or {})
        self.calls.append((sql, values))
        if "WITH (UPDLOCK, HOLDLOCK)" in sql:
            return FakeScalarResult(17)
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
        pricing_schema="pricing",
    ):
        self.connection = FakeConnection(
            existing_version=existing_version,
            versions=versions,
        )
        self._execution_options = {"pricing_schema": pricing_schema}

    def begin(self):
        return FakeBegin(self.connection)


def test_existing_model_version_for_export_returns_stored_version_exactly():
    engine = FakeEngine(existing_version="20260605")

    assert (
        existing_model_version_for_export(
            engine,
            model_name="MY_MODEL",
            export_id="my_model__run_1",
        )
        == "20260605"
    )


def test_model_version_queries_respect_configured_pricing_schema():
    engine = FakeEngine(existing_version="v3", pricing_schema="python_pricing")

    existing_model_version_for_export(
        engine,
        model_name="MY_MODEL",
        export_id="my_model__run_1",
    )

    sql, params = engine.connection.calls[0]
    assert "FROM python_pricing.PRICING_RATE_PACKAGE AS rp" in sql
    assert "JOIN python_pricing.PRICING_MODEL AS pm" in sql
    assert params == {"model_name": "MY_MODEL", "export_id": "my_model__run_1"}


def test_next_trained_model_version_ignores_non_vn_history_and_child_packages():
    engine = FakeEngine(versions=["v2", "20260605", "v10", "manual"])

    assert next_trained_model_version(engine, model_name="MY_MODEL") == "v11"

    sql, params = engine.connection.calls[0]
    assert "rp.parent_rate_package_id IS NULL" in sql
    assert params == {"model_name": "MY_MODEL"}


def test_next_trained_model_version_defaults_to_v1_when_no_vn_history():
    engine = FakeEngine(versions=["20260605", "manual"])

    assert next_trained_model_version(engine, model_name="MY_MODEL") == "v1"


def test_resolve_model_version_for_export_reuses_existing_export_version():
    engine = FakeEngine(existing_version="v7", versions=["v1", "v2"])

    assert (
        resolve_model_version_for_export(
            engine,
            model_name="MY_MODEL",
            export_id="my_model__run_1",
        )
        == "v7"
    )
    assert len(engine.connection.calls) == 2


def test_resolve_model_version_for_export_allocates_next_version_for_new_export():
    engine = FakeEngine(existing_version=None, versions=["v1", "v4"])

    assert (
        resolve_model_version_for_export(
            engine,
            model_name="MY_MODEL",
            export_id="my_model__run_2",
        )
        == "v5"
    )
    assert engine.connection.reservations == {"my_model__run_2": "v5"}
    assert len(engine.connection.calls) == 4


def test_resolve_model_version_reserves_distinct_versions_before_publication():
    engine = FakeEngine()

    first = resolve_model_version_for_export(
        engine,
        model_name="MY_MODEL",
        export_id="my_model__run_1",
    )
    second = resolve_model_version_for_export(
        engine,
        model_name="MY_MODEL",
        export_id="my_model__run_2",
    )
    retry = resolve_model_version_for_export(
        engine,
        model_name="MY_MODEL",
        export_id="my_model__run_1",
    )

    assert (first, second, retry) == ("v1", "v2", "v1")
    assert engine.connection.reservations == {
        "my_model__run_1": "v1",
        "my_model__run_2": "v2",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_name": None, "export_id": "export-1"}, "model_name"),
        ({"model_name": "", "export_id": "export-1"}, "model_name"),
        ({"model_name": "MY_MODEL", "export_id": "   "}, "export_id"),
        ({"model_name": "MY_MODEL", "export_id": None}, "export_id"),
    ],
)
def test_existing_model_version_for_export_rejects_blank_identity(kwargs, message):
    with pytest.raises(ValueError, match=message):
        existing_model_version_for_export(FakeEngine(), **kwargs)
