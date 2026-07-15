from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SCHEMA_TOKEN_PATTERN = re.compile(r"\b(pricing_stg|pricing|mlops)\b")
_RUNTIME_SCHEMA_QUALIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(pricing_stg|pricing|mlops)(?=\s*\.)"
)


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
            pricing_staging=str(options.get("pricing_staging_schema", cls.pricing_staging)),
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


def _quoted_sql_span_end(sql_text: str, start: int, closing: str) -> int:
    index = start + 1
    while index < len(sql_text):
        if sql_text[index] != closing:
            index += 1
            continue
        if index + 1 < len(sql_text) and sql_text[index + 1] == closing:
            index += 2
            continue
        return index + 1
    return len(sql_text)


def _block_comment_end(sql_text: str, start: int) -> int:
    depth = 1
    index = start + 2
    while index < len(sql_text) and depth:
        if sql_text.startswith("/*", index):
            depth += 1
            index += 2
        elif sql_text.startswith("*/", index):
            depth -= 1
            index += 2
        else:
            index += 1
    return index


def render_runtime_sql_schemas(sql_text: str, schemas: SchemaNames) -> str:
    """Render unquoted schema qualifiers without changing SQL data or comments."""
    mapping = {
        "pricing_stg": schemas.pricing_staging,
        "pricing": schemas.pricing,
        "mlops": schemas.mlops,
    }
    rendered: list[str] = []
    index = 0
    while index < len(sql_text):
        if sql_text.startswith("--", index):
            end = sql_text.find("\n", index + 2)
            end = len(sql_text) if end == -1 else end
            rendered.append(sql_text[index:end])
            index = end
            continue
        if sql_text.startswith("/*", index):
            end = _block_comment_end(sql_text, index)
            rendered.append(sql_text[index:end])
            index = end
            continue
        if sql_text[index] in {"'", '"'}:
            end = _quoted_sql_span_end(sql_text, index, sql_text[index])
            rendered.append(sql_text[index:end])
            index = end
            continue
        if sql_text[index] == "[":
            end = _quoted_sql_span_end(sql_text, index, "]")
            rendered.append(sql_text[index:end])
            index = end
            continue

        match = _RUNTIME_SCHEMA_QUALIFIER_PATTERN.match(sql_text, index)
        if match:
            rendered.append(mapping[match.group(1)])
            index = match.end()
            continue

        rendered.append(sql_text[index])
        index += 1
    return "".join(rendered)


def schema_names_from_connectable(connectable) -> SchemaNames:
    options = getattr(connectable, "_execution_options", None)
    if not options and hasattr(connectable, "get_execution_options"):
        options = connectable.get_execution_options()
    return SchemaNames.from_execution_options(options)
