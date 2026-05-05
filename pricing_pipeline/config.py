from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


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
    mssql_user: str = "sa"
    mssql_password: str = "YourStrong(!)Password123"
    mssql_driver: str = "ODBC Driver 18 for SQL Server"
    mssql_encrypt: str = "no"
    mssql_trust_server_cert: str = "yes"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    rating_export_root: Path = Path("/opt/pricing/state/rating_exports")
    skip_database_create: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Settings":
        return cls(
            mssql_server=env.get("MSSQL_SERVER", cls.mssql_server),
            pricing_database=env.get("MSSQL_DATABASE", cls.pricing_database),
            mlflow_database=env.get("MLFLOW_DATABASE", cls.mlflow_database),
            mssql_user=env.get("MSSQL_USER", cls.mssql_user),
            mssql_password=env.get("MSSQL_PASSWORD", cls.mssql_password),
            mssql_driver=env.get("MSSQL_DRIVER", cls.mssql_driver),
            mssql_encrypt=env.get("MSSQL_ENCRYPT", cls.mssql_encrypt),
            mssql_trust_server_cert=env.get(
                "MSSQL_TRUST_SERVER_CERT", cls.mssql_trust_server_cert
            ),
            mlflow_tracking_uri=env.get("MLFLOW_TRACKING_URI", cls.mlflow_tracking_uri),
            rating_export_root=Path(
                env.get("RATING_EXPORT_ROOT", str(cls.rating_export_root))
            ),
            skip_database_create=_env_bool(
                env,
                "PRICING_SKIP_DATABASE_CREATE",
                cls.skip_database_create,
            ),
        )
