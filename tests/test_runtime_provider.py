from __future__ import annotations

import sys
import types
from pathlib import Path

from sqlalchemy import create_engine

import scripts.pricing_db as script_db
from pricing_pipeline.infra.runtime import runtime_from_module
from pricing_pipeline.infra.schema import SchemaNames, schema_names_from_connectable


def test_runtime_provider_imports_connection_module_from_src(monkeypatch, tmp_path):
    project_root = tmp_path
    runtime_dir = project_root / "src" / "work_runtime"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "__init__.py").write_text("", encoding="utf-8")
    (runtime_dir / "database.py").write_text(
        """
from sqlalchemy import create_engine


last_database = None


def get_engine(database=None):
    global last_database
    last_database = database
    return create_engine("sqlite://")


def get_schema_names():
    return {
        "pricing": "python_pricing",
        "pricing_staging": "python_pricing_stg",
        "mlops": "python_mlops",
    }


def get_runtime_settings():
    return {
        "pricing_database": "PricingWork",
        "skip_database_create": True,
        "rating_export_root": "state/work/rating_exports",
        "validation_split_artifact_root": "state/work/validation_splits",
        "workbench_artifact_root": "state/work/candidates",
        "airflow_api_url": "https://airflow.work.example/api/v2",
        "airflow_api_token": "runtime-token",
        "mlflow_tracking_uri": "http://mlflow.work:5000",
        "mlflow_enabled": False,
    }
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("PRICING_PROJECT_ROOT", str(project_root))
    sys.modules.pop("work_runtime", None)
    sys.modules.pop("work_runtime.database", None)

    runtime = runtime_from_module("work_runtime.database")
    engine = runtime.get_engine(database="PricingWork")

    assert runtime.settings.pricing_database == "PricingWork"
    assert runtime.settings.skip_database_create is True
    assert runtime.settings.mlflow_tracking_uri == "http://mlflow.work:5000"
    assert runtime.settings.mlflow_enabled is False
    assert runtime.settings.workbench_artifact_root == Path("state/work/candidates")
    assert runtime.settings.airflow_api_url == "https://airflow.work.example/api/v2"
    assert runtime.settings.airflow_api_token == "runtime-token"
    assert runtime.settings.schema_names == SchemaNames(
        pricing="python_pricing",
        pricing_staging="python_pricing_stg",
        mlops="python_mlops",
    )
    assert schema_names_from_connectable(engine) == runtime.settings.schema_names

    imported = sys.modules["work_runtime.database"]
    assert imported.last_database == "PricingWork"


def test_script_get_engine_uses_runtime_module_from_env(monkeypatch):
    sentinel_engine = create_engine("sqlite://")
    runtime_module = types.ModuleType("unit_runtime_provider")
    calls = []

    def get_engine(database=None):
        calls.append(database)
        return sentinel_engine

    runtime_module.get_engine = get_engine
    runtime_module.get_schema_names = lambda: {
        "pricing": "team_pricing",
        "pricing_staging": "team_pricing_stg",
        "mlops": "team_mlops",
    }
    runtime_module.get_runtime_settings = lambda: {
        "pricing_database": "TeamPricing",
        "skip_database_create": True,
    }

    monkeypatch.setitem(sys.modules, "unit_runtime_provider", runtime_module)
    monkeypatch.setattr(script_db, "load_env", lambda: None)
    monkeypatch.setenv("PRICING_RUNTIME_MODULE", "unit_runtime_provider")

    engine = script_db.get_engine(database="TeamPricing")

    assert engine is sentinel_engine
    assert calls == ["TeamPricing"]
    assert schema_names_from_connectable(engine) == SchemaNames(
        pricing="team_pricing",
        pricing_staging="team_pricing_stg",
        mlops="team_mlops",
    )
