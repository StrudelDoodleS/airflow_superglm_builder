CREATE TABLE IF NOT EXISTS pricing_stg.STG_RATING_EXPORT (
    export_id TEXT NOT NULL PRIMARY KEY,
    model_id TEXT,
    model_name TEXT NOT NULL,
    model_version TEXT,
    base_rate REAL NOT NULL,
    effective_from_date TEXT,
    effective_to_date TEXT,
    source_file TEXT,
    publication_receipt_json TEXT,
    publication_receipt_sha256 TEXT,
    package_metadata_json TEXT,
    offset_handling TEXT,
    offset_factor_name TEXT,
    offset_source_name TEXT,
    offset_label TEXT,
    metadata_origin TEXT,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pricing_stg.STG_RATE_CELL (
    export_id TEXT NOT NULL,
    row_id INTEGER NOT NULL,
    term_name TEXT NOT NULL,
    term_type TEXT NOT NULL,
    sequence_no INTEGER NOT NULL,
    cell_key_text TEXT NOT NULL,
    multiplier REAL NOT NULL,
    log_coefficient REAL NOT NULL,
    exposure_weight REAL,
    record_count INTEGER,
    is_reference INTEGER NOT NULL DEFAULT 0,
    is_default INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (export_id, row_id)
);

CREATE TABLE IF NOT EXISTS pricing_stg.STG_CELL_LEVEL (
    export_id TEXT NOT NULL,
    row_id INTEGER NOT NULL,
    position_no INTEGER NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value_type TEXT NOT NULL,
    level_set_name TEXT NOT NULL,
    level_set_type TEXT NOT NULL,
    level_code TEXT NOT NULL,
    level_label TEXT,
    order_index INTEGER,
    lower_bound REAL,
    upper_bound REAL,
    representative_value REAL,
    is_missing INTEGER NOT NULL DEFAULT 0,
    is_other INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (export_id, row_id, position_no)
);

CREATE TABLE IF NOT EXISTS pricing_stg.STG_TERM_METADATA (
    export_id TEXT NOT NULL,
    term_name TEXT NOT NULL,
    term_metadata_json TEXT NOT NULL,
    PRIMARY KEY (export_id, term_name),
    FOREIGN KEY (export_id) REFERENCES STG_RATING_EXPORT(export_id)
);
