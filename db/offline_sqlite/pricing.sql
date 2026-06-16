CREATE TABLE IF NOT EXISTS pricing.FREMTPL_RAW (
    IDpol INTEGER NOT NULL PRIMARY KEY,
    ClaimNb INTEGER NOT NULL,
    Exposure REAL NOT NULL,
    Area TEXT,
    VehPower INTEGER,
    VehAge INTEGER,
    DrivAge INTEGER,
    BonusMalus INTEGER,
    VehBrand TEXT,
    VehGas TEXT,
    Density REAL,
    Region TEXT
);

CREATE TABLE IF NOT EXISTS pricing.DATASET_MANIFEST (
    manifest_id TEXT NOT NULL PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    source_system TEXT,
    data_as_of_date TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    pk_columns_json TEXT NOT NULL,
    target_column TEXT,
    weight_column TEXT,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pricing.DATASET_COLUMN (
    manifest_id TEXT NOT NULL,
    ordinal_no INTEGER NOT NULL,
    column_name TEXT NOT NULL,
    column_role TEXT NOT NULL,
    pandas_dtype TEXT NOT NULL,
    null_count INTEGER NOT NULL,
    distinct_count INTEGER,
    PRIMARY KEY (manifest_id, ordinal_no)
);

CREATE TABLE IF NOT EXISTS pricing.CV_SPLIT_SET (
    split_set_id TEXT NOT NULL PRIMARY KEY,
    manifest_id TEXT NOT NULL,
    split_mode TEXT NOT NULL,
    splitter_class TEXT,
    splitter_params_json TEXT,
    row_order_sha256 TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    fold_count INTEGER NOT NULL,
    groups_column TEXT,
    stratify_column TEXT,
    artifact_uri TEXT,
    artifact_sha256 TEXT,
    runtime_metadata_json TEXT,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pricing.CV_FOLD (
    split_set_id TEXT NOT NULL,
    fold_no INTEGER NOT NULL,
    n_train INTEGER NOT NULL,
    n_test INTEGER NOT NULL,
    PRIMARY KEY (split_set_id, fold_no)
);

CREATE TABLE IF NOT EXISTS pricing.CV_FOLD_METRIC (
    model_run_id TEXT NOT NULL,
    split_set_id TEXT NOT NULL,
    fold_no INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    PRIMARY KEY (model_run_id, split_set_id, fold_no, metric_name)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_MODEL (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_key TEXT NOT NULL,
    model_label TEXT,
    target_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    model_status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL,
    retired_ts TEXT,
    UNIQUE (model_key)
);

CREATE TABLE IF NOT EXISTS pricing.MODEL_RUN (
    model_run_id TEXT PRIMARY KEY,
    model_id INTEGER NOT NULL,
    dag_id TEXT,
    airflow_run_id TEXT,
    mlflow_run_id TEXT,
    model_version TEXT NOT NULL,
    export_id TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    split_set_id TEXT,
    rate_package_id INTEGER NOT NULL,
    model_name TEXT,
    rating_workbook_path TEXT NOT NULL,
    model_artifact_path TEXT,
    effective_from TEXT NOT NULL,
    run_status TEXT NOT NULL DEFAULT 'SUCCEEDED',
    started_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_ts TEXT,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_RATE_PACKAGE (
    rate_package_id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_rate_package_id INTEGER,
    model_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    package_version INTEGER NOT NULL,
    base_rate REAL NOT NULL,
    effective_from_date TEXT NOT NULL,
    effective_to_date TEXT,
    package_status TEXT NOT NULL,
    source_export_id TEXT,
    source_file TEXT,
    manifest_id TEXT,
    split_set_id TEXT,
    rating_workbook_path TEXT,
    model_artifact_path TEXT,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL,
    UNIQUE (model_id, source_export_id),
    UNIQUE (model_id, package_version)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_FEATURE (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL UNIQUE,
    feature_value_type TEXT NOT NULL,
    is_ordered INTEGER NOT NULL DEFAULT 0,
    active_flag INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_FEATURE_LEVEL_SET (
    level_set_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER,
    feature_id INTEGER NOT NULL,
    level_set_name TEXT NOT NULL,
    level_set_type TEXT NOT NULL,
    binning_strategy TEXT,
    grid_width REAL,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_id, feature_id, level_set_name)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_FEATURE_LEVEL (
    feature_level_id INTEGER PRIMARY KEY AUTOINCREMENT,
    level_set_id INTEGER NOT NULL,
    level_code TEXT NOT NULL,
    level_label TEXT,
    order_index INTEGER,
    lower_bound REAL,
    upper_bound REAL,
    representative_value REAL,
    is_missing INTEGER NOT NULL DEFAULT 0,
    is_other INTEGER NOT NULL DEFAULT 0,
    UNIQUE (level_set_id, level_code)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_TERM (
    term_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rate_package_id INTEGER NOT NULL,
    term_name TEXT NOT NULL,
    term_type TEXT NOT NULL,
    sequence_no INTEGER NOT NULL,
    default_multiplier REAL NOT NULL DEFAULT 1.0,
    default_log_coefficient REAL NOT NULL DEFAULT 0.0,
    active_flag INTEGER NOT NULL DEFAULT 1,
    UNIQUE (rate_package_id, term_name)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_TERM_FEATURE (
    term_id INTEGER NOT NULL,
    position_no INTEGER NOT NULL,
    feature_id INTEGER NOT NULL,
    level_set_id INTEGER NOT NULL,
    input_column_name TEXT,
    PRIMARY KEY (term_id, position_no)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_RATE_CELL (
    cell_id INTEGER PRIMARY KEY AUTOINCREMENT,
    term_id INTEGER NOT NULL,
    cell_key_text TEXT NOT NULL,
    cell_key_digest TEXT NOT NULL,
    multiplier REAL NOT NULL,
    log_coefficient REAL NOT NULL,
    exposure_weight REAL,
    record_count INTEGER,
    is_reference INTEGER NOT NULL DEFAULT 0,
    is_default INTEGER NOT NULL DEFAULT 0,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    UNIQUE (term_id, cell_key_digest)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_RATE_CELL_LEVEL (
    cell_id INTEGER NOT NULL,
    position_no INTEGER NOT NULL,
    feature_level_id INTEGER NOT NULL,
    PRIMARY KEY (cell_id, position_no)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_COMPILED_RATE_CELL (
    rate_package_id INTEGER NOT NULL,
    term_id INTEGER NOT NULL,
    cell_key_digest TEXT NOT NULL,
    term_name TEXT NOT NULL,
    term_type TEXT NOT NULL,
    sequence_no INTEGER NOT NULL,
    cell_key_text TEXT NOT NULL,
    multiplier REAL NOT NULL,
    log_coefficient REAL NOT NULL,
    exposure_weight REAL,
    record_count INTEGER,
    is_default INTEGER NOT NULL,
    is_reference INTEGER NOT NULL,
    PRIMARY KEY (rate_package_id, term_id, cell_key_digest)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_COMPILED_1D_RATE_BAND (
    rate_package_id INTEGER NOT NULL,
    term_id INTEGER NOT NULL,
    feature_level_id INTEGER NOT NULL,
    term_name TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    level_code TEXT NOT NULL,
    sort_order INTEGER NOT NULL,
    lower_bound REAL,
    upper_bound REAL,
    representative_value REAL,
    multiplier REAL NOT NULL,
    log_coefficient REAL NOT NULL,
    PRIMARY KEY (rate_package_id, term_id, sort_order, feature_level_id)
);
