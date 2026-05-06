from pathlib import Path

from pricing_pipeline.migrations import migration_files, split_sql_server_batches


def test_split_sql_server_batches_handles_go_lines():
    sql = "SELECT 1;\nGO\nSELECT 2;\ngo\n"
    assert split_sql_server_batches(sql) == ["SELECT 1;", "SELECT 2;"]


def test_migration_files_are_sorted(tmp_path: Path):
    (tmp_path / "V002__b.sql").write_text("SELECT 2", encoding="utf-8")
    (tmp_path / "V001__a.sql").write_text("SELECT 1", encoding="utf-8")
    assert [p.name for p in migration_files(tmp_path)] == ["V001__a.sql", "V002__b.sql"]


def test_model_registry_migration_keeps_history_and_guards_current_deployments():
    migration = Path("db/migrations/V006__model_registry_deployments.sql").read_text(
        encoding="utf-8"
    )

    assert "CREATE TABLE pricing.PRICING_MODEL" in migration
    assert "model_id BIGINT IDENTITY(1,1) PRIMARY KEY" in migration
    assert "CREATE TABLE pricing.PRICING_MODEL_DEPLOYMENT" in migration
    assert "effective_from_ts DATETIME2(3) NOT NULL" in migration
    assert "effective_to_ts DATETIME2(3) NULL" in migration
    assert "UX_MODEL_DEPLOYMENT_CURRENT" in migration
    assert "WHERE effective_to_ts IS NULL" in migration


def test_fresh_schema_defines_model_keys_near_table_identifiers():
    pricing_core = Path("db/migrations/V002__pricing_core_minimal.sql").read_text(
        encoding="utf-8"
    )
    fremtpl_run = Path("db/migrations/V005__fremtpl_raw_model_run.sql").read_text(
        encoding="utf-8"
    )

    assert pricing_core.index("rate_package_id        BIGINT IDENTITY") < pricing_core.index(
        "model_id               BIGINT NULL"
    ) < pricing_core.index("model_name             NVARCHAR(128)")
    assert pricing_core.index("pointer_name      NVARCHAR(128)") < pricing_core.index(
        "model_id          BIGINT NULL"
    ) < pricing_core.index("rate_package_id   BIGINT NOT NULL")
    assert pricing_core.index("level_set_id        BIGINT IDENTITY") < pricing_core.index(
        "model_id            BIGINT NULL"
    ) < pricing_core.index("feature_id          BIGINT NOT NULL")
    assert fremtpl_run.index("model_run_id BIGINT IDENTITY") < fremtpl_run.index(
        "model_id BIGINT NULL"
    ) < fremtpl_run.index("dag_id NVARCHAR(250) NOT NULL")


def test_model_registry_migration_scopes_packages_pointers_and_level_sets():
    migration = Path("db/migrations/V006__model_registry_deployments.sql").read_text(
        encoding="utf-8"
    )

    assert "ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD model_id BIGINT NULL" in migration
    assert "ALTER TABLE pricing.MODEL_RUN ADD model_id BIGINT NULL" in migration
    assert "ALTER TABLE pricing_stg.STG_RATING_EXPORT ADD model_id BIGINT NULL" in migration
    assert "ALTER TABLE pricing.PRICING_PACKAGE_POINTER ADD model_id BIGINT NULL" in migration
    assert "ALTER TABLE pricing.PRICING_FEATURE_LEVEL_SET ADD model_id BIGINT NULL" in migration
    assert "UX_LEVEL_SET_MODEL_FEATURE_NAME" in migration
    assert "UX_PACKAGE_POINTER_MODEL_SLOT" in migration


def test_model_registry_migration_exposes_current_views_not_mutable_active_flags():
    migration = Path("db/migrations/V006__model_registry_deployments.sql").read_text(
        encoding="utf-8"
    )

    assert "CREATE OR ALTER VIEW pricing.V_ACTIVE_MODEL" in migration
    assert "CREATE OR ALTER VIEW pricing.V_CURRENT_RATE_PACKAGE" in migration
    assert "CREATE OR ALTER VIEW pricing.V_CURRENT_RATE_CELL" in migration
    assert "CREATE OR ALTER VIEW pricing.V_CURRENT_1D_RATE_BAND" in migration
    assert "active_flag" not in migration.lower()


def test_compiled_band_sort_order_migration_backfills_and_rekeys_table():
    migration = Path("db/migrations/V008__compiled_band_sort_order.sql").read_text(
        encoding="utf-8"
    )

    assert "ALTER TABLE pricing.PRICING_COMPILED_1D_RATE_BAND" in migration
    assert "ADD sort_order INT NOT NULL" in migration
    assert "SET sort_order = COALESCE(fl.order_index, 0)" in migration
    assert "DROP CONSTRAINT PK_COMPILED_1D_RATE_BAND" in migration
    assert "PRIMARY KEY CLUSTERED" in migration
    assert "rate_package_id, term_id, sort_order, feature_level_id" in migration


def test_current_dataset_cv_fold_view_exposes_latest_split_metadata():
    migration = Path("db/migrations/V009__current_dataset_cv_fold_view.sql").read_text(
        encoding="utf-8"
    )

    assert "CREATE OR ALTER VIEW pricing.V_CURRENT_DATASET_CV_FOLD" in migration
    assert "ROW_NUMBER() OVER" in migration
    assert "PARTITION BY dataset_name" in migration
    assert "PARTITION BY manifest_id" in migration
    assert "STRING_AGG(CONVERT(VARCHAR(12), train_fold.fold_no), ',')" in migration
    assert "train_folds_json" in migration
    assert "test_fold_no" in migration
    assert "n_train" in migration
    assert "n_test" in migration


def test_cv_split_runtime_metadata_migration_adds_dependency_audit_json():
    migration = Path("db/migrations/V010__cv_split_runtime_metadata.sql").read_text(
        encoding="utf-8"
    )
    current_view = Path("db/migrations/V011__cv_split_runtime_metadata_view.sql").read_text(
        encoding="utf-8"
    )

    assert "ALTER TABLE pricing.CV_SPLIT_SET" in migration
    assert "ADD runtime_metadata_json NVARCHAR(MAX) NULL" in migration
    assert "CREATE OR ALTER VIEW pricing.V_CURRENT_DATASET_CV_FOLD" in current_view
    assert "runtime_metadata_json" in current_view


def test_clean_pricing_schema_migration_moves_staging_and_drops_obsolete_tables():
    migration = Path("db/migrations/V012__clean_pricing_schema_tables.sql").read_text(
        encoding="utf-8"
    )

    assert "CREATE SCHEMA pricing_stg" in migration
    assert "CREATE TABLE pricing_stg.STG_RATING_EXPORT" in migration
    assert "DROP TABLE pricing.STG_RATING_EXPORT" in migration
    assert "DROP TABLE pricing.STG_RATE_CELL" in migration
    assert "DROP TABLE pricing.STG_CELL_LEVEL" in migration
    assert "DROP TABLE pricing.STG_DATASET_ROW_KEY" in migration
    assert "DROP TABLE pricing.CV_SPLIT" in migration
    assert "DROP TABLE pricing.DATASET_ROW_KEY" in migration


def test_useful_tables_reference_ddl_excludes_staging_and_row_materialization():
    ddl = Path("docs/pricing_useful_tables_ddl.sql").read_text(encoding="utf-8")

    useful_tables = [
        "FREMTPL_RAW",
        "DATASET_MANIFEST",
        "DATASET_COLUMN",
        "PRICING_MODEL",
        "MODEL_RUN",
        "CV_SPLIT_SET",
        "CV_FOLD",
        "CV_FOLD_METRIC",
        "PRICING_RATE_PACKAGE",
        "PRICING_MODEL_DEPLOYMENT",
        "PRICING_PACKAGE_POINTER",
        "PRICING_TERM",
        "PRICING_RATE_CELL",
        "PRICING_TERM_FEATURE",
        "PRICING_FEATURE",
        "PRICING_FEATURE_LEVEL_SET",
        "PRICING_FEATURE_LEVEL",
        "PRICING_RATE_CELL_LEVEL",
        "PRICING_COMPILED_RATE_CELL",
        "PRICING_COMPILED_1D_RATE_BAND",
    ]

    for table in useful_tables:
        assert f"CREATE TABLE {table}" in ddl

    assert "pricing_stg" not in ddl
    assert "pricing." not in ddl
    assert "STG_" not in ddl
    assert "DATASET_ROW_KEY" not in ddl
    assert "CREATE TABLE CV_SPLIT (" not in ddl
    assert "FOREIGN KEY (model_id) REFERENCES PRICING_MODEL(model_id)" in ddl
    assert "FOREIGN KEY (manifest_id) REFERENCES DATASET_MANIFEST(manifest_id)" in ddl
    assert "FOREIGN KEY (rate_package_id) REFERENCES PRICING_RATE_PACKAGE(rate_package_id)" in ddl


def test_useful_tables_reference_ddl_is_plain_sql_server_ddl_for_erd_import():
    ddl = Path("docs/pricing_useful_tables_ddl.sql").read_text(encoding="utf-8")

    assert "\nGO\n" not in ddl
    assert "IF NOT EXISTS" not in ddl
    assert "EXEC(" not in ddl
    assert "CREATE TABLE IF NOT EXISTS" not in ddl
    assert "CREATE SCHEMA" not in ddl
    assert "CONSTRAINT " not in ddl
    assert "CREATE UNIQUE INDEX" not in ddl
    assert "CHECK (" not in ddl
    assert "\nWHERE " not in ddl
    assert "CREATE TABLE PRICING_MODEL (" in ddl
    assert "NVARCHAR(MAX)" in ddl
    assert "DATETIME2(3)" in ddl
    assert "IDENTITY(1,1)" in ddl
