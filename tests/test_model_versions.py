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


class FakeScalarsResult:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return iter(self.values)


class FakeConnection:
    def __init__(self, *, existing_version=None, versions=()):
        self.existing_version = existing_version
        self.versions = versions
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute(self, statement, params=None):
        sql = str(statement)
        values = dict(params or {})
        self.calls.append((sql, values))
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
            model_key="MY_MODEL",
            export_id="my_model__run_1",
        )
        == "20260605"
    )


def test_model_version_queries_respect_configured_pricing_schema():
    engine = FakeEngine(existing_version="v3", pricing_schema="python_pricing")

    existing_model_version_for_export(
        engine,
        model_key="MY_MODEL",
        export_id="my_model__run_1",
    )

    sql, params = engine.connection.calls[0]
    assert "FROM python_pricing.PRICING_RATE_PACKAGE AS rp" in sql
    assert "JOIN python_pricing.PRICING_MODEL AS pm" in sql
    assert params == {"model_key": "MY_MODEL", "export_id": "my_model__run_1"}


def test_next_trained_model_version_ignores_non_vn_history_and_child_packages():
    engine = FakeEngine(versions=["v2", "20260605", "v10", "manual"])

    assert next_trained_model_version(engine, model_key="MY_MODEL") == "v11"

    sql, params = engine.connection.calls[0]
    assert "rp.parent_rate_package_id IS NULL" in sql
    assert params == {"model_key": "MY_MODEL"}


def test_next_trained_model_version_defaults_to_v1_when_no_vn_history():
    engine = FakeEngine(versions=["20260605", "manual"])

    assert next_trained_model_version(engine, model_key="MY_MODEL") == "v1"


def test_resolve_model_version_for_export_reuses_existing_export_version():
    engine = FakeEngine(existing_version="v7", versions=["v1", "v2"])

    assert (
        resolve_model_version_for_export(
            engine,
            model_key="MY_MODEL",
            export_id="my_model__run_1",
        )
        == "v7"
    )
    assert len(engine.connection.calls) == 1


def test_resolve_model_version_for_export_allocates_next_version_for_new_export():
    engine = FakeEngine(existing_version=None, versions=["v1", "v4"])

    assert (
        resolve_model_version_for_export(
            engine,
            model_key="MY_MODEL",
            export_id="my_model__run_2",
        )
        == "v5"
    )
    assert len(engine.connection.calls) == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_key": None, "export_id": "export-1"}, "model_key"),
        ({"model_key": "", "export_id": "export-1"}, "model_key"),
        ({"model_key": "MY_MODEL", "export_id": "   "}, "export_id"),
        ({"model_key": "MY_MODEL", "export_id": None}, "export_id"),
    ],
)
def test_existing_model_version_for_export_rejects_blank_identity(kwargs, message):
    with pytest.raises(ValueError, match=message):
        existing_model_version_for_export(FakeEngine(), **kwargs)
