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
    model_id TEXT PRIMARY KEY,
    model_key TEXT NOT NULL,
    model_label TEXT NOT NULL,
    target_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    model_status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL,
    retired_ts TEXT
);

CREATE TABLE IF NOT EXISTS pricing.MODEL_RUN (
    model_run_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    export_id TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    split_set_id TEXT,
    rate_package_id TEXT NOT NULL,
    rating_workbook_path TEXT NOT NULL,
    model_artifact_path TEXT,
    effective_from TEXT NOT NULL,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_RATE_PACKAGE (
    rate_package_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    package_status TEXT NOT NULL,
    source_export_id TEXT NOT NULL,
    effective_from TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    split_set_id TEXT,
    rating_workbook_path TEXT NOT NULL,
    model_artifact_path TEXT,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL
);
