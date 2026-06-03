from __future__ import annotations

from pathlib import Path

import pytest
from sqlfluff.api.simple import parse

from scripts.render_schema_sql import render_schema_sql


def _parse_tsql(sql: str, label: str) -> None:
    try:
        parse(sql, dialect="tsql")
    except Exception as exc:
        raise AssertionError(f"{label} failed SQLFluff T-SQL parse: {exc}") from exc


@pytest.mark.parametrize(
    "path",
    sorted(Path("db/migrations").glob("V*.sql")),
    ids=lambda path: path.name,
)
def test_migration_sql_parses_as_tsql(path: Path):
    _parse_tsql(path.read_text(encoding="utf-8"), path.as_posix())


def test_rendered_custom_schema_sql_parses_as_tsql():
    sql = render_schema_sql(
        Path("db/migrations"),
        pricing_schema="python_pricing",
        pricing_staging_schema="python_pricing_stg",
        mlops_schema="python_mlops",
    )

    assert "THROW" in sql
    assert "RAISERROR" not in sql
    _parse_tsql(sql, "rendered custom schema SQL")
