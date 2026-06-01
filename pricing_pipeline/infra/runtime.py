from __future__ import annotations

import importlib
import inspect
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

from sqlalchemy.engine import Engine

from pricing_pipeline.infra import db as shared_db
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra.schema import SchemaNames, validate_schema_name


@dataclass(frozen=True)
class PipelineRuntime:
    settings: Settings
    _engine_loader: Callable[[str | None], Engine]
    _database_ensurer: Callable[[str], None] | None = None

    def get_engine(self, *, database: str | None = None) -> Engine:
        return self._engine_loader(database)

    def ensure_database(self, database: str) -> None:
        if self._database_ensurer is None:
            raise RuntimeError(
                "Runtime module did not provide ensure_database(database). "
                "Set skip_database_create=true in get_runtime_settings(), or "
                "create the database outside the pipeline."
            )
        self._database_ensurer(database)


def ensure_runtime_import_paths(
    *,
    project_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    env = env or os.environ
    root = Path(project_root or env.get("PRICING_PROJECT_ROOT", Path.cwd())).resolve()
    for path in (root, root / "src"):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def runtime_from_env(env: Mapping[str, str] | None = None) -> PipelineRuntime:
    env = env or os.environ
    settings = Settings.from_env(env)
    return PipelineRuntime(
        settings=settings,
        _engine_loader=lambda database=None: shared_db.get_engine(
            settings,
            database=database,
        ),
        _database_ensurer=lambda database: shared_db.ensure_database(settings, database),
    )


def runtime_from_env_or_module(
    runtime_module: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> PipelineRuntime:
    env = env or os.environ
    module_path = runtime_module or env.get("PRICING_RUNTIME_MODULE")
    if module_path:
        return runtime_from_module(module_path, env=env)
    return runtime_from_env(env)


def runtime_from_module(
    module_path: str,
    *,
    env: Mapping[str, str] | None = None,
) -> PipelineRuntime:
    env = env or os.environ
    ensure_runtime_import_paths(env=env)
    module = importlib.import_module(module_path)
    if not hasattr(module, "get_engine"):
        raise AttributeError(f"{module_path} must define get_engine(database=None)")

    settings = _settings_from_module(module, env=env)
    engine_loader_func = getattr(module, "get_engine")

    def _engine_loader(database: str | None = None) -> Engine:
        engine = _call_engine_loader(engine_loader_func, database)
        return shared_db.configure_engine(engine, settings.schema_names)

    database_ensurer = getattr(module, "ensure_database", None)
    return PipelineRuntime(
        settings=settings,
        _engine_loader=_engine_loader,
        _database_ensurer=database_ensurer,
    )


def _settings_from_module(module: ModuleType, *, env: Mapping[str, str]) -> Settings:
    raw_settings = _call_optional(module, "get_runtime_settings")
    if raw_settings is None:
        raw_settings = _call_optional(module, "get_settings")

    schema_names = _schema_names_from_module(module)

    if isinstance(raw_settings, Settings):
        settings = raw_settings
    else:
        # A Python runtime module means DB connectivity is provided by code, so
        # database creation should be opt-in rather than inherited from Docker defaults.
        settings = replace(Settings.from_env(env), skip_database_create=True)
        if raw_settings is not None:
            settings = _settings_from_mapping(settings, _as_mapping(raw_settings))

    if schema_names is not None:
        settings = replace(
            settings,
            pricing_schema=schema_names.pricing,
            pricing_staging_schema=schema_names.pricing_staging,
            mlops_schema=schema_names.mlops,
        )
    return settings


def _settings_from_mapping(settings: Settings, values: Mapping[str, Any]) -> Settings:
    replacements: dict[str, Any] = {}
    if "pricing_database" in values:
        replacements["pricing_database"] = str(values["pricing_database"])
    if "mlflow_database" in values:
        replacements["mlflow_database"] = str(values["mlflow_database"])
    if "mlflow_tracking_uri" in values:
        replacements["mlflow_tracking_uri"] = str(values["mlflow_tracking_uri"])
    if "mlflow_enabled" in values:
        replacements["mlflow_enabled"] = _bool_value(values["mlflow_enabled"])
    if "rating_export_root" in values:
        replacements["rating_export_root"] = Path(str(values["rating_export_root"]))
    if "validation_split_artifact_root" in values:
        replacements["validation_split_artifact_root"] = Path(
            str(values["validation_split_artifact_root"])
        )
    if "skip_database_create" in values:
        replacements["skip_database_create"] = _bool_value(values["skip_database_create"])

    schema_names = _schema_names_from_mapping(values)
    if schema_names is not None:
        replacements.update(
            {
                "pricing_schema": schema_names.pricing,
                "pricing_staging_schema": schema_names.pricing_staging,
                "mlops_schema": schema_names.mlops,
            }
        )
    return replace(settings, **replacements)


def _call_optional(module: ModuleType, name: str) -> Any | None:
    func = getattr(module, name, None)
    if func is None:
        return None
    return func()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError("runtime settings must be a Settings instance or mapping")


def _schema_names_from_module(module: ModuleType) -> SchemaNames | None:
    raw = _call_optional(module, "get_schema_names")
    if raw is None:
        return None
    return _coerce_schema_names(raw)


def _schema_names_from_mapping(values: Mapping[str, Any]) -> SchemaNames | None:
    raw = values.get("schema_names")
    if raw is not None:
        return _coerce_schema_names(raw)
    if {"pricing", "pricing_staging", "mlops"} & values.keys():
        return _coerce_schema_names(values)
    if {"pricing_schema", "pricing_staging_schema", "mlops_schema"} & values.keys():
        return _coerce_schema_names(values)
    return None


def _coerce_schema_names(value: Any) -> SchemaNames:
    if isinstance(value, SchemaNames):
        return value
    mapping = _as_mapping(value)
    pricing = str(mapping.get("pricing", mapping.get("pricing_schema", SchemaNames.pricing)))
    pricing_staging = str(
        mapping.get(
            "pricing_staging",
            mapping.get("pricing_staging_schema", SchemaNames.pricing_staging),
        )
    )
    mlops = str(mapping.get("mlops", mapping.get("mlops_schema", SchemaNames.mlops)))
    return SchemaNames(
        pricing=validate_schema_name(pricing, "pricing"),
        pricing_staging=validate_schema_name(pricing_staging, "pricing_staging"),
        mlops=validate_schema_name(mlops, "mlops"),
    )


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _call_engine_loader(func: Callable[..., Engine], database: str | None) -> Engine:
    signature = inspect.signature(func)
    params = signature.parameters
    accepts_database = "database" in params or any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    if accepts_database:
        return func(database=database)
    if database is not None:
        raise TypeError("runtime get_engine must accept database=... for this operation")
    return func()
