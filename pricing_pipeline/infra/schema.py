from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SCHEMA_TOKEN_PATTERN = re.compile(r"\b(pricing_stg|pricing|mlops)\b")


@dataclass(frozen=True)
class SchemaNames:
    pricing: str = "pricing"
    pricing_staging: str = "pricing_stg"
    mlops: str = "mlops"

    def as_execution_options(self) -> dict[str, str]:
        return {
            "pricing_schema": self.pricing,
            "pricing_staging_schema": self.pricing_staging,
            "mlops_schema": self.mlops,
        }

    @classmethod
    def from_execution_options(cls, options: Mapping[str, object] | None) -> "SchemaNames":
        options = options or {}
        return cls(
            pricing=str(options.get("pricing_schema", cls.pricing)),
            pricing_staging=str(
                options.get("pricing_staging_schema", cls.pricing_staging)
            ),
            mlops=str(options.get("mlops_schema", cls.mlops)),
        )


def validate_schema_name(name: str, env_name: str) -> str:
    value = name.strip()
    if not _IDENTIFIER_PATTERN.match(value):
        raise ValueError(
            f"{env_name} must be a simple SQL identifier using letters, numbers, "
            "and underscores, and it cannot start with a number"
        )
    return value


def render_sql_schemas(sql_text: str, schemas: SchemaNames) -> str:
    mapping = {
        "pricing_stg": schemas.pricing_staging,
        "pricing": schemas.pricing,
        "mlops": schemas.mlops,
    }
    return _SCHEMA_TOKEN_PATTERN.sub(lambda match: mapping[match.group(1)], sql_text)


def schema_names_from_connectable(connectable) -> SchemaNames:
    options = getattr(connectable, "_execution_options", None)
    if not options and hasattr(connectable, "get_execution_options"):
        options = connectable.get_execution_options()
    return SchemaNames.from_execution_options(options)
