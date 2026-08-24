# Tutorials

Open `00_basic_sql_etl_schema_walkthrough.ipynb` in Jupyter for a conceptual
SQL/ETL/schema walkthrough. For an actual pricing model, use the six-notebook
workflow created by `scripts/scaffold_pricing_model.py`, starting with
`01_data_ingestion.ipynb` and `02_model_training.ipynb`.

For ERD generation, upload `schema/pricing_useful_tables_ddl.sql`. It is the strict
SQL Server DDL copy kept in sync with `docs/pricing_useful_tables_ddl.sql`.
It is a conceptual extract; use `db/migrations` to create or upgrade the
operational schema.

Open `01_portable_underwriter_report.ipynb` for a synthetic, executable example
of copying the one-file prediction report into an unrelated project. It teaches
the frequency, severity, and burn-cost input scales and has no build, SQL, or
deployment dependency.
