import re
from pathlib import Path

from pricing_pipeline.infra.migrations import migration_files, split_sql_server_batches


def _create_table_bodies(ddl: str) -> dict[str, str]:
    bodies = {}
    current_table = None
    current_body = []

    for line in ddl.splitlines():
        match = re.match(r"CREATE TABLE ([a-z_]+\.[A-Z0-9_]+) \(", line)
        if match:
            current_table = match.group(1)
            current_body = []
            continue

        if current_table and line == ");":
            bodies[current_table] = "\n".join(current_body)
            current_table = None
            continue

        if current_table:
            current_body.append(line)

    return bodies


def _create_table_columns(ddl: str) -> dict[str, list[str]]:
    skipped_keywords = {
        "CHECK",
        "CONSTRAINT",
        "FOREIGN",
        "PRIMARY",
        "REFERENCES",
        "UNIQUE",
    }
    columns = {}

    for table_name, body in _create_table_bodies(ddl).items():
        table_columns = []
        for line in body.splitlines():
            stripped = line.strip().rstrip(",")
            if not stripped:
                continue

            first_token = stripped.split()[0]
            if first_token.upper() in skipped_keywords:
                continue

            table_columns.append(first_token)

        columns[table_name] = table_columns

    return columns


def _create_table_foreign_keys(ddl: str) -> dict[str, list[tuple[tuple[str, ...], str, tuple[str, ...]]]]:
    foreign_key_pattern = re.compile(
        r"FOREIGN KEY \(([^)]*)\)\s+REFERENCES\s+([a-z_]+\.[A-Z0-9_]+)\(([^)]*)\)",
        re.S,
    )
    foreign_keys = {}

    for table_name, body in _create_table_bodies(ddl).items():
        table_foreign_keys = []
        for source_columns, target_table, target_columns in foreign_key_pattern.findall(body):
            table_foreign_keys.append(
                (
                    tuple(column.strip() for column in source_columns.replace("\n", " ").split(",")),
                    target_table,
                    tuple(column.strip() for column in target_columns.replace("\n", " ").split(",")),
                )
            )

        foreign_keys[table_name] = sorted(table_foreign_keys)

    return foreign_keys


def _view_columns(ddl: str, view_name: str) -> list[str]:
    view_pattern = re.compile(
        rf"CREATE OR ALTER VIEW {re.escape(view_name)} AS\nSELECT\n(?P<select>.*?)\nFROM ",
        re.S,
    )
    match = view_pattern.search(ddl)
    assert match is not None, f"Missing view {view_name}"

    columns = []
    for line in match.group("select").splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped:
            continue

        alias = re.search(r"\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)$", stripped, re.I)
        columns.append(alias.group(1) if alias else stripped.split(".")[-1])

    return columns


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


def test_rate_package_version_guard_migration_adds_unique_model_version_index():
    migration = Path("db/migrations/V016__rate_package_version_and_deploy_guards.sql").read_text(
        encoding="utf-8"
    )

    assert "UX_PRICING_RATE_PACKAGE_MODEL_VERSION" in migration
    assert "PRICING_RATE_PACKAGE(model_id, package_version)" in migration
    assert "WHERE model_id IS NOT NULL" in migration


def test_package_writer_allocates_version_under_lock():
    writer = Path("pricing_pipeline/publishing/package_writer.py").read_text(encoding="utf-8")

    assert "WITH (UPDLOCK, HOLDLOCK)" in writer
    assert "MAX(package_version)" in writer


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


def test_model_run_lineage_migration_adds_minimal_mlops_link_tables():
    migration = Path("db/migrations/V013__model_run_lineage_tables.sql").read_text(
        encoding="utf-8"
    )

    assert "CREATE SCHEMA mlops" in migration
    assert "CREATE TABLE mlops.MODEL_RUN_DATASET" in migration
    assert "CREATE TABLE mlops.MODEL_RUN_SPLIT_SET" in migration
    assert "CREATE TABLE mlops.MODEL_RUN_METRIC" in migration
    assert "CREATE TABLE mlops.CV_SPLIT_ROW" in migration
    assert "UX_CV_SPLIT_SET_MANIFEST_SPLIT" in migration
    assert "ON pricing.CV_SPLIT_SET(manifest_id, split_set_id)" in migration
    assert "REFERENCES pricing.MODEL_RUN(model_run_id)" in migration
    assert "REFERENCES pricing.DATASET_MANIFEST(manifest_id)" in migration
    assert "REFERENCES pricing.CV_SPLIT_SET(manifest_id, split_set_id)" in migration
    assert (
        "REFERENCES mlops.MODEL_RUN_DATASET(model_run_id, dataset_role, manifest_id)"
        in migration
    )
    assert "PRICING_PACKAGE_POINTER" not in migration
    assert "pricing_stg" not in migration


def test_prediction_proc_migration_scores_current_package_from_compiled_views():
    migration = Path("db/migrations/V014__current_rate_prediction_proc.sql").read_text(
        encoding="utf-8"
    )

    assert "CREATE OR ALTER PROCEDURE pricing.PREDICT_CURRENT_RATE" in migration
    assert "@features_json NVARCHAR(MAX)" in migration
    assert "@exposure FLOAT" in migration
    assert "pricing.V_CURRENT_RATE_PACKAGE" in migration
    assert "pricing.V_CURRENT_1D_RATE_BAND" in migration
    assert "pricing.V_CURRENT_RATE_CELL" in migration
    assert "JSON_VALUE(@features_json" in migration
    assert "TRY_CONVERT(FLOAT" in migration
    assert "EXP(SUM(log_coefficient))" in migration
    assert "base_rate * @exposure * EXP(SUM(log_coefficient)) AS prediction" in migration
    assert "@include_breakdown" in migration
    assert "Input features did not match every required term" in migration


def test_package_immutability_migration_blocks_direct_edits_to_frozen_packages():
    migration = Path("db/migrations/V015__rate_package_immutability.sql").read_text(
        encoding="utf-8"
    )

    assert "TR_PRICING_RATE_PACKAGE_IMMUTABLE_UPDATE_DELETE" in migration
    assert "TR_PRICING_TERM_IMMUTABLE_WRITE" in migration
    assert "TR_PRICING_RATE_CELL_IMMUTABLE_WRITE" in migration
    assert "TR_PRICING_RATE_CELL_LEVEL_IMMUTABLE_WRITE" in migration
    assert "TR_PRICING_FEATURE_LEVEL_IMMUTABLE_WRITE" in migration
    assert "TR_PRICING_COMPILED_RATE_CELL_IMMUTABLE_WRITE" in migration
    assert "TR_PRICING_COMPILED_1D_RATE_BAND_IMMUTABLE_WRITE" in migration
    assert "package_status <> 'DRAFT'" in migration
    assert "pricing.PRICING_MODEL_DEPLOYMENT" in migration
    assert "THROW 51000" in migration
    assert "Create a new package revision" in migration
    assert "AFTER INSERT, UPDATE, DELETE" in migration


def test_rating_package_loader_builds_package_as_draft_before_final_status():
    loader = Path("pricing_pipeline/publishing/package_writer.py").read_text(
        encoding="utf-8"
    )

    assert "requested_package_status = args.package_status" in loader
    assert '"package_status": "DRAFT"' in loader
    assert "UPDATE pricing.PRICING_RATE_PACKAGE" in loader
    assert "SET package_status = :package_status" in loader
    assert "requested_package_status" in loader


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
        "raw.FREMTPL_RAW",
        "mlops.DATASET_MANIFEST",
        "mlops.DATASET_COLUMN",
        "mlops.CV_SPLIT_SET",
        "mlops.CV_FOLD",
        "mlops.CV_SPLIT_ROW",
        "mlops.MODEL_RUN",
        "mlops.MODEL_RUN_DATASET",
        "mlops.MODEL_RUN_SPLIT_SET",
        "mlops.MODEL_RUN_METRIC",
        "mlops.CV_FOLD_METRIC",
        "pricing.MODEL",
        "pricing.RATE_PACKAGE",
        "pricing.FEATURE",
        "pricing.FEATURE_LEVEL_SET",
        "pricing.FEATURE_LEVEL",
        "pricing.TERM",
        "pricing.TERM_FEATURE",
        "pricing.RATE_CELL",
        "pricing.RATE_CELL_LEVEL",
        "pricing.MODEL_DEPLOYMENT",
        "pricing_runtime.V_COMPILED_RATE_CELL",
        "pricing_runtime.V_COMPILED_RATE_CELL_LEVEL",
        "pricing_runtime.V_COMPILED_1D_RATE_BAND",
    ]

    for table in useful_tables:
        assert f"CREATE TABLE {table}" in ddl

    assert "pricing_stg" not in ddl
    assert "STG_" not in ddl
    assert "DATASET_ROW_KEY" not in ddl
    assert "PRICING_PACKAGE_POINTER" not in ddl
    assert "CREATE TABLE pricing.PACKAGE_POINTER" not in ddl
    assert "CREATE TABLE mlops.CV_SPLIT (" not in ddl
    assert "FOREIGN KEY (model_id) REFERENCES pricing.MODEL(model_id)" in ddl
    assert "FOREIGN KEY (manifest_id) REFERENCES mlops.DATASET_MANIFEST(manifest_id)" in ddl
    assert "FOREIGN KEY (rate_package_id) REFERENCES pricing.RATE_PACKAGE(rate_package_id)" in ddl
    assert (
        "FOREIGN KEY (model_id, parent_rate_package_id) "
        "REFERENCES pricing.RATE_PACKAGE(model_id, rate_package_id)"
    ) in ddl
    assert (
        "FOREIGN KEY (model_id, model_run_id) "
        "REFERENCES mlops.MODEL_RUN(model_id, model_run_id)"
    ) in ddl
    assert (
        "FOREIGN KEY (model_run_id, dataset_role, manifest_id) "
        "REFERENCES mlops.MODEL_RUN_DATASET(model_run_id, dataset_role, manifest_id)"
    ) in ddl
    assert (
        "FOREIGN KEY (manifest_id, split_set_id) "
        "REFERENCES mlops.CV_SPLIT_SET(manifest_id, split_set_id)"
    ) in ddl
    assert (
        "FOREIGN KEY (feature_id, level_set_id) "
        "REFERENCES pricing.FEATURE_LEVEL_SET(feature_id, level_set_id)"
    ) in ddl
    assert (
        "FOREIGN KEY (cell_id, term_id) "
        "REFERENCES pricing.RATE_CELL(cell_id, term_id)"
    ) in ddl
    assert (
        "FOREIGN KEY (term_id, position_no, level_set_id) "
        "REFERENCES pricing.TERM_FEATURE(term_id, position_no, level_set_id)"
    ) in ddl
    assert (
        "FOREIGN KEY (level_set_id, feature_level_id) "
        "REFERENCES pricing.FEATURE_LEVEL(level_set_id, feature_level_id)"
    ) in ddl


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
    assert "CREATE TABLE pricing.MODEL (" in ddl
    assert "CREATE TABLE pricing_runtime.V_COMPILED_RATE_CELL (" in ddl
    assert "NVARCHAR(MAX)" in ddl
    assert "DATETIME2(3)" in ddl
    assert "IDENTITY(1,1)" in ddl


def test_full_useful_tables_reference_matches_strict_erd_table_contract():
    strict_ddl = Path("docs/pricing_useful_tables_ddl.sql").read_text(encoding="utf-8")
    full_ddl = Path("docs/pricing_useful_tables_full_ddl.sql").read_text(encoding="utf-8")

    strict_columns = _create_table_columns(strict_ddl)
    full_columns = _create_table_columns(full_ddl)
    strict_foreign_keys = _create_table_foreign_keys(strict_ddl)
    full_foreign_keys = _create_table_foreign_keys(full_ddl)

    persisted_tables = {
        table_name: columns
        for table_name, columns in strict_columns.items()
        if not table_name.startswith("pricing_runtime.")
    }

    assert set(full_columns) == set(persisted_tables)
    assert {
        table_name: full_columns[table_name] for table_name in sorted(full_columns)
    } == {
        table_name: persisted_tables[table_name] for table_name in sorted(persisted_tables)
    }
    assert {
        table_name: full_foreign_keys[table_name] for table_name in sorted(full_foreign_keys)
    } == {
        table_name: strict_foreign_keys[table_name]
        for table_name in sorted(persisted_tables)
    }

    runtime_views = [
        "pricing_runtime.V_COMPILED_RATE_CELL",
        "pricing_runtime.V_COMPILED_RATE_CELL_LEVEL",
        "pricing_runtime.V_COMPILED_1D_RATE_BAND",
    ]
    for view_name in runtime_views:
        assert _view_columns(full_ddl, view_name) == strict_columns[view_name]


def test_full_useful_tables_reference_ddl_keeps_sql_server_constraints_and_indexes():
    ddl = Path("docs/pricing_useful_tables_full_ddl.sql").read_text(encoding="utf-8")

    assert "CREATE SCHEMA raw;" in ddl
    assert "CREATE SCHEMA mlops;" in ddl
    assert "CREATE SCHEMA pricing;" in ddl
    assert "CREATE SCHEMA pricing_runtime;" in ddl
    assert "CREATE TABLE pricing.MODEL (" in ddl
    assert "CREATE TABLE mlops.MODEL_RUN_DATASET (" in ddl
    assert "CREATE TABLE mlops.MODEL_RUN_SPLIT_SET (" in ddl
    assert "CREATE TABLE mlops.MODEL_RUN_METRIC (" in ddl
    assert "CREATE TABLE mlops.CV_SPLIT_ROW (" in ddl
    assert "CREATE OR ALTER VIEW pricing.V_CURRENT_RATE_PACKAGE" in ddl
    assert "CREATE OR ALTER VIEW pricing_runtime.V_COMPILED_RATE_CELL" in ddl
    assert "CONSTRAINT PK_MODEL" in ddl
    assert "CONSTRAINT FK_MODEL_RUN_MODEL" in ddl
    assert "CONSTRAINT CK_MODEL_STATUS" in ddl
    assert "CREATE UNIQUE INDEX UX_MODEL_DEPLOYMENT_CURRENT" in ddl
    assert "WHERE effective_to_ts IS NULL" in ddl
    assert "CONSTRAINT FK_RATE_PACKAGE_PARENT_SAME_MODEL" in ddl
    assert "FOREIGN KEY (model_id, parent_rate_package_id)" in ddl
    assert "REFERENCES pricing.RATE_PACKAGE(model_id, rate_package_id)" in ddl
    assert "CONSTRAINT FK_RATE_PACKAGE_MODEL_RUN" in ddl
    assert "FOREIGN KEY (model_id, model_run_id)" in ddl
    assert "REFERENCES mlops.MODEL_RUN(model_id, model_run_id)" in ddl
    assert "CONSTRAINT FK_MODEL_RUN_SPLIT_SET_DATASET" in ddl
    assert "FOREIGN KEY (model_run_id, dataset_role, manifest_id)" in ddl
    assert "REFERENCES mlops.MODEL_RUN_DATASET(model_run_id, dataset_role, manifest_id)" in ddl
    assert "CONSTRAINT FK_MODEL_RUN_SPLIT_SET_SPLIT" in ddl
    assert "FOREIGN KEY (manifest_id, split_set_id)" in ddl
    assert "REFERENCES mlops.CV_SPLIT_SET(manifest_id, split_set_id)" in ddl
    assert "CONSTRAINT FK_TERM_FEATURE_LEVEL_SET_FEATURE" in ddl
    assert "FOREIGN KEY (feature_id, level_set_id)" in ddl
    assert "REFERENCES pricing.FEATURE_LEVEL_SET(feature_id, level_set_id)" in ddl
    assert "CONSTRAINT FK_RATE_CELL_LEVEL_CELL" in ddl
    assert "FOREIGN KEY (cell_id, term_id)" in ddl
    assert "CONSTRAINT FK_RATE_CELL_LEVEL_TERM_FEATURE" in ddl
    assert "FOREIGN KEY (term_id, position_no, level_set_id)" in ddl
    assert "CONSTRAINT FK_RATE_CELL_LEVEL_FEATURE_LEVEL" in ddl
    assert "FOREIGN KEY (level_set_id, feature_level_id)" in ddl
    assert "CREATE UNIQUE INDEX UX_RATE_CELL_TERM_DIGEST_ACTIVE" in ddl
    assert "WHERE is_deleted = 0" in ddl
    assert "PACKAGE_POINTER" not in ddl
    assert "PRICING_PACKAGE_POINTER" not in ddl
    assert "pricing_stg" not in ddl
    assert "STG_" not in ddl
    assert "DATASET_ROW_KEY" not in ddl


def test_full_useful_tables_reference_ddl_documents_immutability_triggers():
    ddl = Path("docs/pricing_useful_tables_full_ddl.sql").read_text(encoding="utf-8")

    assert "TR_RATE_PACKAGE_IMMUTABLE_UPDATE_DELETE" in ddl
    assert "TR_TERM_IMMUTABLE_WRITE" in ddl
    assert "TR_RATE_CELL_IMMUTABLE_WRITE" in ddl
    assert "TR_RATE_CELL_LEVEL_IMMUTABLE_WRITE" in ddl
    assert "TR_FEATURE_LEVEL_IMMUTABLE_WRITE" in ddl
    assert "TR_COMPILED_RATE_CELL_IMMUTABLE_WRITE" in ddl
    assert "TR_COMPILED_1D_RATE_BAND_IMMUTABLE_WRITE" in ddl
    assert "package_status <> 'DRAFT'" in ddl
    assert "MODEL_DEPLOYMENT" in ddl
    assert "THROW 51000" in ddl
