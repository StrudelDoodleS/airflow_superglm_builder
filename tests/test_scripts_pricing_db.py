import os
from urllib.parse import unquote_plus

import scripts.pricing_db as script_db
from pricing_pipeline import db as shared_db
from pricing_pipeline.config import Settings


def test_script_get_engine_delegates_to_shared_pricing_db_helper(monkeypatch):
    sentinel_engine = object()
    captured = {}
    original_from_env = Settings.from_env

    def fake_from_env(cls, env):
        captured["env"] = env
        return original_from_env(env)

    def fake_shared_get_engine(settings, *, database=None):
        captured["settings"] = settings
        captured["database"] = database
        return sentinel_engine

    monkeypatch.setattr(
        script_db,
        "load_env",
        lambda: captured.setdefault("load_env", True),
    )
    monkeypatch.setattr(Settings, "from_env", classmethod(fake_from_env))
    monkeypatch.setattr(shared_db, "get_engine", fake_shared_get_engine)
    monkeypatch.setattr(
        script_db,
        "create_engine",
        lambda *args, **kwargs: object(),
        raising=False,
    )
    monkeypatch.setenv("MSSQL_DATABASE", "ScriptPricing")

    assert script_db.get_engine() is sentinel_engine
    assert captured["load_env"] is True
    assert captured["env"] is os.environ
    assert captured["settings"].pricing_database == "ScriptPricing"
    assert captured["database"] is None


def test_script_build_sqlalchemy_url_uses_shared_password_escaping(monkeypatch):
    monkeypatch.setattr(script_db, "load_env", lambda: None)
    monkeypatch.setenv("MSSQL_PASSWORD", "sec;Encrypt=yes}tail")

    url = script_db.build_sqlalchemy_url()
    odbc = unquote_plus(url.split("odbc_connect=", 1)[1])

    assert "PWD={sec;Encrypt=yes}}tail};" in odbc
    assert "PWD=sec;Encrypt=yes}tail;" not in odbc
    assert ";Encrypt=yes;" not in odbc
