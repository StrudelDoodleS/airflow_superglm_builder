from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Mapping

from pricing_pipeline.infra.schema import SchemaNames, validate_schema_name


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def canonicalize_path(
    value: str | Path,
    *,
    relative_to: str | Path | None = None,
) -> Path:
    raw_value = os.fspath(value)
    path = Path(raw_value).expanduser()
    if os.name != "nt" and PureWindowsPath(raw_value).drive:
        raise ValueError(
            f"Windows drive-qualified path {raw_value!r} is unsupported under POSIX/WSL; "
            "use a path in the current POSIX namespace"
        )
    if not path.is_absolute():
        base = Path.cwd() if relative_to is None else canonicalize_path(relative_to)
        path = base / path
    return path.resolve()


def resolve_project_path(value: str | Path, env: Mapping[str, str]) -> Path:
    project_root = canonicalize_path(
        env.get("PRICING_PROJECT_ROOT") or Path.cwd()
    )
    return canonicalize_path(value, relative_to=project_root)


@dataclass(frozen=True)
class Settings:
    mssql_server: str = "mssql,1433"
    pricing_database: str = "PricingLab"
    mlflow_database: str = "MLflowTracking"
    mssql_sqlalchemy_dialect: str = "mssql+pyodbc"
    mssql_auth_mode: str = "sql_password"
    mssql_token_scope: str = "https://database.windows.net/.default"
    mssql_user: str = "sa"
    mssql_password: str = field(default="YourStrong(!)Password123", repr=False)
    mssql_driver: str = "ODBC Driver 18 for SQL Server"
    mssql_encrypt: str = "no"
    mssql_trust_server_cert: str = "yes"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_enabled: bool = True
    rating_export_root: Path = Path("/opt/pricing/state/rating_exports")
    validation_split_artifact_root: Path = Path("/opt/pricing/state/validation_splits")
    workbench_artifact_root: Path = Path("state/workbench_artifacts")
    airflow_api_url: str = "http://127.0.0.1:8080/api/v2"
    airflow_api_token: str | None = field(default=None, repr=False)
    airflow_api_username: str | None = None
    airflow_api_password: str | None = field(default=None, repr=False)
    skip_database_create: bool = False
    pricing_schema: str = "pricing"
    pricing_staging_schema: str = "pricing_stg"
    mlops_schema: str = "mlops"

    @property
    def schema_names(self) -> SchemaNames:
        return SchemaNames(
            pricing=self.pricing_schema,
            pricing_staging=self.pricing_staging_schema,
            mlops=self.mlops_schema,
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Settings":
        return cls(
            mssql_server=env.get("MSSQL_SERVER", cls.mssql_server),
            pricing_database=env.get("MSSQL_DATABASE", cls.pricing_database),
            mlflow_database=env.get("MLFLOW_DATABASE", cls.mlflow_database),
            mssql_sqlalchemy_dialect=env.get(
                "MSSQL_SQLALCHEMY_DIALECT",
                cls.mssql_sqlalchemy_dialect,
            ),
            mssql_auth_mode=env.get("MSSQL_AUTH_MODE", cls.mssql_auth_mode),
            mssql_token_scope=env.get("MSSQL_TOKEN_SCOPE", cls.mssql_token_scope),
            mssql_user=env.get("MSSQL_USER", cls.mssql_user),
            mssql_password=env.get("MSSQL_PASSWORD", cls.mssql_password),
            mssql_driver=env.get("MSSQL_DRIVER", cls.mssql_driver),
            mssql_encrypt=env.get("MSSQL_ENCRYPT", cls.mssql_encrypt),
            mssql_trust_server_cert=env.get(
                "MSSQL_TRUST_SERVER_CERT", cls.mssql_trust_server_cert
            ),
            mlflow_tracking_uri=env.get("MLFLOW_TRACKING_URI", cls.mlflow_tracking_uri),
            mlflow_enabled=_env_bool(env, "PRICING_ENABLE_MLFLOW", cls.mlflow_enabled),
            rating_export_root=resolve_project_path(
                env.get("RATING_EXPORT_ROOT", str(cls.rating_export_root)),
                env,
            ),
            validation_split_artifact_root=resolve_project_path(
                env.get(
                    "VALIDATION_SPLIT_ARTIFACT_ROOT",
                    str(cls.validation_split_artifact_root),
                ),
                env,
            ),
            workbench_artifact_root=resolve_project_path(
                env.get(
                    "WORKBENCH_ARTIFACT_ROOT",
                    str(cls.workbench_artifact_root),
                ),
                env,
            ),
            airflow_api_url=env.get("AIRFLOW_API_URL", cls.airflow_api_url).rstrip("/"),
            airflow_api_token=(env.get("AIRFLOW_API_TOKEN") or None),
            airflow_api_username=(env.get("AIRFLOW_API_USERNAME") or None),
            airflow_api_password=(
                env.get("AIRFLOW_API_PASSWORD")
                or env.get("AIRFLOW_LOCAL_PASSWORD")
                or None
            ),
            skip_database_create=_env_bool(
                env,
                "PRICING_SKIP_DATABASE_CREATE",
                cls.skip_database_create,
            ),
            pricing_schema=validate_schema_name(
                env.get("PRICING_SCHEMA", cls.pricing_schema),
                "PRICING_SCHEMA",
            ),
            pricing_staging_schema=validate_schema_name(
                env.get("PRICING_STAGING_SCHEMA", cls.pricing_staging_schema),
                "PRICING_STAGING_SCHEMA",
            ),
            mlops_schema=validate_schema_name(
                env.get("MLOPS_SCHEMA", cls.mlops_schema),
                "MLOPS_SCHEMA",
            ),
        )
