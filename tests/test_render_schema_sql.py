from __future__ import annotations

from scripts.render_schema_sql import render_schema_sql


def test_render_schema_sql_outputs_custom_schema_seed_script(tmp_path):
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "V002__view.sql").write_text(
        "CREATE OR ALTER VIEW pricing.V_TEST AS SELECT 1 AS value;\nGO\n",
        encoding="utf-8",
    )
    (migrations / "V001__table.sql").write_text(
        "IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')\n"
        "    EXEC('CREATE SCHEMA pricing');\n"
        "GO\n"
        "IF OBJECT_ID('pricing.TEST_TABLE', 'U') IS NULL\n"
        "CREATE TABLE pricing.TEST_TABLE(id INT NOT NULL);\n"
        "GO\n",
        encoding="utf-8",
    )

    sql = render_schema_sql(
        migrations,
        pricing_schema="python_pricing",
        pricing_staging_schema="python_pricing_stg",
        mlops_schema="python_mlops",
    )

    assert "dbo.SCHEMA_CONFIGURATION" in sql
    assert "dbo.SCHEMA_MIGRATION" in sql
    assert "python_pricing.TEST_TABLE" in sql
    assert "python_pricing.V_TEST" in sql
    assert "V001__table.sql" in sql
    assert "V002__view.sql" in sql
    assert sql.index("V001__table.sql") < sql.index("V002__view.sql")
    assert "'pricing_schema', N'python_pricing'" in sql
    assert "OBJECT_ID('pricing.TEST_TABLE" not in sql
    assert "CREATE TABLE pricing.TEST_TABLE" not in sql


def test_render_schema_sql_can_be_used_from_plain_python_script(tmp_path):
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "V001__minimal.sql").write_text(
        "CREATE OR ALTER VIEW pricing.V_TEST AS SELECT 1 AS value;\nGO\n",
        encoding="utf-8",
    )

    sql = render_schema_sql(
        migrations,
        pricing_schema="team_pricing",
        pricing_staging_schema="team_pricing_stg",
        mlops_schema="team_mlops",
    )

    assert "team_pricing.V_TEST" in sql
