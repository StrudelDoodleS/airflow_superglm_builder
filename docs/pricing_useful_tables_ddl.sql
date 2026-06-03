CREATE TABLE raw.FREMTPL_RAW (
    IDpol BIGINT PRIMARY KEY,
    ClaimNb INT NOT NULL,
    Exposure FLOAT NOT NULL,
    Area NVARCHAR(16),
    VehPower INT,
    VehAge INT,
    DrivAge INT,
    BonusMalus INT,
    VehBrand NVARCHAR(64),
    VehGas NVARCHAR(16),
    Density FLOAT,
    Region NVARCHAR(32)
);

CREATE TABLE mlops.DATASET_MANIFEST (
    manifest_id NVARCHAR(128) PRIMARY KEY,
    dataset_name NVARCHAR(128) NOT NULL,
    source_schema NVARCHAR(128),
    source_table NVARCHAR(128),
    source_system NVARCHAR(128),
    data_as_of_date DATE NOT NULL,
    row_count BIGINT NOT NULL,
    pk_columns_json NVARCHAR(MAX) NOT NULL,
    target_column NVARCHAR(128),
    weight_column NVARCHAR(128),
    created_ts DATETIME2(3) DEFAULT SYSDATETIME(),
    created_by NVARCHAR(128) NOT NULL
);

CREATE TABLE mlops.DATASET_COLUMN (
    manifest_id NVARCHAR(128) NOT NULL,
    ordinal_no INT NOT NULL,
    column_name NVARCHAR(128) NOT NULL,
    column_role NVARCHAR(32) NOT NULL,
    pandas_dtype NVARCHAR(64) NOT NULL,
    null_count BIGINT NOT NULL,
    distinct_count BIGINT,
    PRIMARY KEY (manifest_id, ordinal_no),
    FOREIGN KEY (manifest_id) REFERENCES mlops.DATASET_MANIFEST(manifest_id)
);

CREATE TABLE pricing.MODEL (
    model_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    model_key NVARCHAR(128) UNIQUE NOT NULL,
    model_label NVARCHAR(256),
    target_name NVARCHAR(128) NOT NULL,
    model_type NVARCHAR(128) NOT NULL,
    model_status NVARCHAR(32) DEFAULT 'ACTIVE',
    created_ts DATETIME2(3) DEFAULT SYSDATETIME(),
    created_by NVARCHAR(128) NOT NULL,
    retired_ts DATETIME2(3)
);

CREATE TABLE mlops.MODEL_RUN (
    model_run_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    model_id BIGINT,
    dag_id NVARCHAR(250) NOT NULL,
    airflow_run_id NVARCHAR(250) NOT NULL,
    mlflow_experiment_id NVARCHAR(128),
    mlflow_run_id NVARCHAR(128),
    model_version NVARCHAR(64),
    run_status NVARCHAR(32) NOT NULL,
    started_ts DATETIME2(3) DEFAULT SYSDATETIME(),
    completed_ts DATETIME2(3),
    created_by NVARCHAR(128) NOT NULL,
    UNIQUE (model_id, model_run_id),
    FOREIGN KEY (model_id) REFERENCES pricing.MODEL(model_id)
);

CREATE TABLE mlops.CV_SPLIT_SET (
    split_set_id NVARCHAR(128) PRIMARY KEY,
    manifest_id NVARCHAR(128) NOT NULL,
    split_mode NVARCHAR(32) NOT NULL,
    splitter_class NVARCHAR(256),
    splitter_params_json NVARCHAR(MAX),
    row_order_sha256 CHAR(64) NOT NULL,
    row_count BIGINT NOT NULL,
    fold_count INT NOT NULL,
    groups_column NVARCHAR(128),
    stratify_column NVARCHAR(128),
    artifact_uri NVARCHAR(1024),
    artifact_sha256 CHAR(64),
    runtime_metadata_json NVARCHAR(MAX),
    created_ts DATETIME2(3) DEFAULT SYSDATETIME(),
    created_by NVARCHAR(128) NOT NULL,
    UNIQUE (manifest_id, split_set_id),
    FOREIGN KEY (manifest_id) REFERENCES mlops.DATASET_MANIFEST(manifest_id)
);

CREATE TABLE mlops.CV_FOLD (
    split_set_id NVARCHAR(128) NOT NULL,
    fold_no INT NOT NULL,
    n_train BIGINT NOT NULL,
    n_test BIGINT NOT NULL,
    PRIMARY KEY (split_set_id, fold_no),
    FOREIGN KEY (split_set_id) REFERENCES mlops.CV_SPLIT_SET(split_set_id)
);

CREATE TABLE mlops.MODEL_RUN_DATASET (
    model_run_id BIGINT NOT NULL,
    manifest_id NVARCHAR(128) NOT NULL,
    dataset_role NVARCHAR(64) NOT NULL,
    PRIMARY KEY (model_run_id, dataset_role, manifest_id),
    FOREIGN KEY (model_run_id) REFERENCES mlops.MODEL_RUN(model_run_id),
    FOREIGN KEY (manifest_id) REFERENCES mlops.DATASET_MANIFEST(manifest_id)
);

CREATE TABLE mlops.MODEL_RUN_SPLIT_SET (
    model_run_id BIGINT NOT NULL,
    manifest_id NVARCHAR(128) NOT NULL,
    split_set_id NVARCHAR(128) NOT NULL,
    dataset_role NVARCHAR(64) NOT NULL,
    split_role NVARCHAR(64) NOT NULL,
    PRIMARY KEY (model_run_id, split_set_id, split_role),
    FOREIGN KEY (model_run_id) REFERENCES mlops.MODEL_RUN(model_run_id),
    FOREIGN KEY (model_run_id, dataset_role, manifest_id) REFERENCES mlops.MODEL_RUN_DATASET(model_run_id, dataset_role, manifest_id),
    FOREIGN KEY (manifest_id, split_set_id) REFERENCES mlops.CV_SPLIT_SET(manifest_id, split_set_id)
);

CREATE TABLE mlops.MODEL_RUN_METRIC (
    model_run_id BIGINT NOT NULL,
    metric_name NVARCHAR(128) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_scope NVARCHAR(64),
    PRIMARY KEY (model_run_id, metric_name),
    FOREIGN KEY (model_run_id) REFERENCES mlops.MODEL_RUN(model_run_id)
);

CREATE TABLE mlops.CV_FOLD_METRIC (
    model_run_id BIGINT NOT NULL,
    split_set_id NVARCHAR(128) NOT NULL,
    fold_no INT NOT NULL,
    metric_name NVARCHAR(128) NOT NULL,
    metric_value FLOAT NOT NULL,
    PRIMARY KEY (model_run_id, split_set_id, fold_no, metric_name),
    FOREIGN KEY (model_run_id) REFERENCES mlops.MODEL_RUN(model_run_id),
    FOREIGN KEY (split_set_id, fold_no) REFERENCES mlops.CV_FOLD(split_set_id, fold_no)
);

CREATE TABLE pricing.RATE_PACKAGE (
    rate_package_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    parent_rate_package_id BIGINT,
    model_id BIGINT NOT NULL,
    model_run_id BIGINT,
    model_version NVARCHAR(64),
    package_version INT NOT NULL,
    base_rate DECIMAL(19,6) NOT NULL,
    effective_from_date DATE NOT NULL,
    effective_to_date DATE,
    package_status NVARCHAR(32) NOT NULL,
    source_export_id NVARCHAR(128),
    source_file NVARCHAR(1024),
    created_ts DATETIME2(3) DEFAULT SYSDATETIME(),
    created_by NVARCHAR(128) NOT NULL,
    UNIQUE (model_id, rate_package_id),
    FOREIGN KEY (model_id, parent_rate_package_id) REFERENCES pricing.RATE_PACKAGE(model_id, rate_package_id),
    FOREIGN KEY (model_id) REFERENCES pricing.MODEL(model_id),
    FOREIGN KEY (model_id, model_run_id) REFERENCES mlops.MODEL_RUN(model_id, model_run_id)
);

CREATE TABLE pricing.FEATURE (
    feature_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    feature_name NVARCHAR(128) UNIQUE NOT NULL,
    feature_value_type NVARCHAR(32) NOT NULL,
    is_ordered BIT DEFAULT 0,
    active_flag BIT DEFAULT 1
);

CREATE TABLE pricing.FEATURE_LEVEL_SET (
    level_set_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    model_id BIGINT,
    feature_id BIGINT NOT NULL,
    level_set_name NVARCHAR(128) NOT NULL,
    level_set_type NVARCHAR(64) NOT NULL,
    binning_strategy NVARCHAR(64),
    grid_width FLOAT,
    created_ts DATETIME2(3) DEFAULT SYSDATETIME(),
    UNIQUE (feature_id, level_set_id),
    FOREIGN KEY (model_id) REFERENCES pricing.MODEL(model_id),
    FOREIGN KEY (feature_id) REFERENCES pricing.FEATURE(feature_id)
);

CREATE TABLE pricing.FEATURE_LEVEL (
    feature_level_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    level_set_id BIGINT NOT NULL,
    level_code NVARCHAR(128) NOT NULL,
    level_label NVARCHAR(256),
    order_index INT,
    lower_bound FLOAT,
    upper_bound FLOAT,
    representative_value FLOAT,
    is_missing BIT DEFAULT 0,
    is_other BIT DEFAULT 0,
    UNIQUE (level_set_id, feature_level_id),
    FOREIGN KEY (level_set_id) REFERENCES pricing.FEATURE_LEVEL_SET(level_set_id)
);

CREATE TABLE pricing.TERM (
    term_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    rate_package_id BIGINT NOT NULL,
    term_name NVARCHAR(128) NOT NULL,
    term_type NVARCHAR(64) NOT NULL,
    sequence_no INT NOT NULL,
    default_multiplier DECIMAL(19,10) DEFAULT 1.0,
    default_log_coefficient DECIMAL(19,12) DEFAULT 0.0,
    active_flag BIT DEFAULT 1,
    FOREIGN KEY (rate_package_id) REFERENCES pricing.RATE_PACKAGE(rate_package_id)
);

CREATE TABLE pricing.TERM_FEATURE (
    term_id BIGINT NOT NULL,
    position_no SMALLINT NOT NULL,
    feature_id BIGINT NOT NULL,
    level_set_id BIGINT NOT NULL,
    input_column_name NVARCHAR(128),
    PRIMARY KEY (term_id, position_no),
    UNIQUE (term_id, position_no, level_set_id),
    FOREIGN KEY (term_id) REFERENCES pricing.TERM(term_id),
    FOREIGN KEY (feature_id, level_set_id) REFERENCES pricing.FEATURE_LEVEL_SET(feature_id, level_set_id)
);

CREATE TABLE pricing.RATE_CELL (
    cell_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    term_id BIGINT NOT NULL,
    cell_key_text NVARCHAR(900) NOT NULL,
    cell_key_digest VARBINARY(32) NOT NULL,
    multiplier DECIMAL(19,10) NOT NULL,
    log_coefficient DECIMAL(19,12) NOT NULL,
    exposure_weight DECIMAL(19,4),
    record_count BIGINT,
    is_reference BIT DEFAULT 0,
    is_default BIT DEFAULT 0,
    is_deleted BIT DEFAULT 0,
    UNIQUE (cell_id, term_id),
    FOREIGN KEY (term_id) REFERENCES pricing.TERM(term_id)
);

CREATE TABLE pricing.RATE_CELL_LEVEL (
    cell_id BIGINT NOT NULL,
    term_id BIGINT NOT NULL,
    position_no SMALLINT NOT NULL,
    level_set_id BIGINT NOT NULL,
    feature_level_id BIGINT NOT NULL,
    PRIMARY KEY (cell_id, position_no),
    FOREIGN KEY (cell_id, term_id) REFERENCES pricing.RATE_CELL(cell_id, term_id),
    FOREIGN KEY (term_id, position_no, level_set_id) REFERENCES pricing.TERM_FEATURE(term_id, position_no, level_set_id),
    FOREIGN KEY (level_set_id, feature_level_id) REFERENCES pricing.FEATURE_LEVEL(level_set_id, feature_level_id)
);

CREATE TABLE pricing.MODEL_DEPLOYMENT (
    deployment_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    model_id BIGINT NOT NULL,
    rate_package_id BIGINT NOT NULL,
    deployment_slot NVARCHAR(64) NOT NULL,
    effective_from_ts DATETIME2(3) DEFAULT SYSDATETIME(),
    effective_to_ts DATETIME2(3),
    deployed_by NVARCHAR(128) NOT NULL,
    deployment_note NVARCHAR(512),
    created_ts DATETIME2(3) DEFAULT SYSDATETIME(),
    FOREIGN KEY (model_id) REFERENCES pricing.MODEL(model_id),
    FOREIGN KEY (model_id, rate_package_id) REFERENCES pricing.RATE_PACKAGE(model_id, rate_package_id)
);

CREATE TABLE pricing_runtime.V_COMPILED_RATE_CELL (
    rate_package_id BIGINT NOT NULL,
    term_id BIGINT NOT NULL,
    cell_id BIGINT NOT NULL,
    term_name NVARCHAR(128) NOT NULL,
    term_type NVARCHAR(64) NOT NULL,
    sequence_no INT NOT NULL,
    cell_key_text NVARCHAR(900) NOT NULL,
    multiplier DECIMAL(19,10) NOT NULL,
    log_coefficient DECIMAL(19,12) NOT NULL,
    exposure_weight DECIMAL(19,4),
    record_count BIGINT,
    is_default BIT NOT NULL,
    is_reference BIT NOT NULL,
    FOREIGN KEY (rate_package_id) REFERENCES pricing.RATE_PACKAGE(rate_package_id),
    FOREIGN KEY (term_id) REFERENCES pricing.TERM(term_id),
    FOREIGN KEY (cell_id) REFERENCES pricing.RATE_CELL(cell_id)
);

CREATE TABLE pricing_runtime.V_COMPILED_RATE_CELL_LEVEL (
    rate_package_id BIGINT NOT NULL,
    term_id BIGINT NOT NULL,
    cell_id BIGINT NOT NULL,
    position_no SMALLINT NOT NULL,
    feature_id BIGINT NOT NULL,
    level_set_id BIGINT NOT NULL,
    feature_level_id BIGINT NOT NULL,
    feature_name NVARCHAR(128) NOT NULL,
    level_code NVARCHAR(128) NOT NULL,
    level_label NVARCHAR(256),
    FOREIGN KEY (rate_package_id) REFERENCES pricing.RATE_PACKAGE(rate_package_id),
    FOREIGN KEY (term_id) REFERENCES pricing.TERM(term_id),
    FOREIGN KEY (cell_id, term_id) REFERENCES pricing.RATE_CELL(cell_id, term_id),
    FOREIGN KEY (feature_id) REFERENCES pricing.FEATURE(feature_id),
    FOREIGN KEY (level_set_id, feature_level_id) REFERENCES pricing.FEATURE_LEVEL(level_set_id, feature_level_id)
);

CREATE TABLE pricing_runtime.V_COMPILED_1D_RATE_BAND (
    rate_package_id BIGINT NOT NULL,
    term_id BIGINT NOT NULL,
    level_set_id BIGINT NOT NULL,
    feature_level_id BIGINT NOT NULL,
    term_name NVARCHAR(128) NOT NULL,
    feature_name NVARCHAR(128) NOT NULL,
    level_code NVARCHAR(128) NOT NULL,
    sort_order INT NOT NULL,
    lower_bound FLOAT,
    upper_bound FLOAT,
    representative_value FLOAT,
    multiplier DECIMAL(19,10) NOT NULL,
    log_coefficient DECIMAL(19,12) NOT NULL,
    FOREIGN KEY (rate_package_id) REFERENCES pricing.RATE_PACKAGE(rate_package_id),
    FOREIGN KEY (term_id) REFERENCES pricing.TERM(term_id),
    FOREIGN KEY (level_set_id, feature_level_id) REFERENCES pricing.FEATURE_LEVEL(level_set_id, feature_level_id)
);
