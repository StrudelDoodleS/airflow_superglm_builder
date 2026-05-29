from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from pricing_pipeline.infra.schema import SchemaNames, validate_schema_name


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    mssql_server: str = "mssql,1433"
    pricing_database: str = "PricingLab"
    mlflow_database: str = "MLflowTracking"
    mssql_sqlalchemy_dialect: str = "mssql+pyodbc"
    mssql_auth_mode: str = "sql_password"
    mssql_token_scope: str = "https://database.windows.net/.default"
    mssql_user: str = "sa"
    mssql_password: str = "YourStrong(!)Password123"
    mssql_driver: str = "ODBC Driver 18 for SQL Server"
    mssql_encrypt: str = "no"
    mssql_trust_server_cert: str = "yes"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_enabled: bool = True
    rating_export_root: Path = Path("/opt/pricing/state/rating_exports")
    validation_split_artifact_root: Path = Path("/opt/pricing/state/validation_splits")
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
            rating_export_root=Path(
                env.get("RATING_EXPORT_ROOT", str(cls.rating_export_root))
            ),
            validation_split_artifact_root=Path(
                env.get(
                    "VALIDATION_SPLIT_ARTIFACT_ROOT",
                    str(cls.validation_split_artifact_root),
                )
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
