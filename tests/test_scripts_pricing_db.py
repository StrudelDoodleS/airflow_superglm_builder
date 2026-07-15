import os
import subprocess
import sys
from types import SimpleNamespace

import scripts.load_fremtpl_raw as load_fremtpl_raw_script
import scripts.pricing_db as script_db
from pricing_pipeline.infra import db as shared_db
from pricing_pipeline.infra.config import Settings


def _run_script_help(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, script, "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_load_fremtpl_raw_script_help_runs_without_pythonpath():
    result = _run_script_help("scripts/load_fremtpl_raw.py")

    assert result.returncode == 0
    assert "--replace" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_pricing_db_direct_script_imports_resolve_package_without_pythonpath():
    result = _run_script_help("scripts/inspect_rating_package.py")

    assert result.returncode == 0
    assert "--pointer" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


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


def test_load_fremtpl_raw_script_uses_pricing_db_engine_loader(monkeypatch, capsys):
    sentinel_engine = object()
    calls = []
    monkeypatch.setattr(
        load_fremtpl_raw_script, "parse_args", lambda: SimpleNamespace(replace=True)
    )
    monkeypatch.setattr(load_fremtpl_raw_script, "get_engine", lambda: sentinel_engine)
    monkeypatch.setattr(
        load_fremtpl_raw_script,
        "load_fremtpl_raw",
        lambda engine, *, replace: calls.append((engine, replace)) or 12,
    )

    load_fremtpl_raw_script.main()

    assert calls == [(sentinel_engine, True)]
    assert "fremtpl_raw_rows=12" in capsys.readouterr().out
