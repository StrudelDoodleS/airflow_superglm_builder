/*
Reference SQL Server DDL for the useful persisted PricingLab tables.

Use this in a scratch database or ERD tool when you want the model/data lineage
without transient staging tables or row-per-policy CV materialization tables.
The runtime migrations under db/migrations remain the source of truth for
incremental upgrades.
*/

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
    EXEC('CREATE SCHEMA pricing');
GO

-- Dataset intake: raw freMTPL data and the dataset manifest used by runs.
CREATE TABLE pricing.FREMTPL_RAW (
    IDpol       BIGINT NOT NULL,
    ClaimNb     INT NOT NULL,
    Exposure    FLOAT NOT NULL,
    Area        NVARCHAR(16) NULL,
    VehPower    INT NULL,
    VehAge      INT NULL,
    DrivAge     INT NULL,
    BonusMalus  INT NULL,
    VehBrand    NVARCHAR(64) NULL,
    VehGas      NVARCHAR(16) NULL,
    Density     FLOAT NULL,
    Region      NVARCHAR(32) NULL,

    CONSTRAINT PK_FREMTPL_RAW
        PRIMARY KEY (IDpol)
);
GO

CREATE TABLE pricing.DATASET_MANIFEST (
    manifest_id      NVARCHAR(128) NOT NULL,
    dataset_name     NVARCHAR(128) NOT NULL,
    source_system    NVARCHAR(128) NULL,
    data_as_of_date  DATE NOT NULL,
    row_count        BIGINT NOT NULL,
    pk_columns_json  NVARCHAR(MAX) NOT NULL,
    target_column    NVARCHAR(128) NULL,
    weight_column    NVARCHAR(128) NULL,
    created_ts       DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    created_by       NVARCHAR(128) NOT NULL,

    CONSTRAINT PK_DATASET_MANIFEST
        PRIMARY KEY (manifest_id)
);
GO

CREATE TABLE pricing.DATASET_COLUMN (
    manifest_id     NVARCHAR(128) NOT NULL,
    ordinal_no      INT NOT NULL,
    column_name     NVARCHAR(128) NOT NULL,
    column_role     NVARCHAR(32) NOT NULL,
    pandas_dtype    NVARCHAR(64) NOT NULL,
    null_count      BIGINT NOT NULL,
    distinct_count  BIGINT NULL,

    CONSTRAINT PK_DATASET_COLUMN
        PRIMARY KEY (manifest_id, ordinal_no),

    CONSTRAINT FK_DATASET_COLUMN_MANIFEST
        FOREIGN KEY (manifest_id)
        REFERENCES pricing.DATASET_MANIFEST(manifest_id)
);
GO

-- Model lifecycle: model families, training runs, packages, and deployments.
CREATE TABLE pricing.PRICING_MODEL (
    model_id      BIGINT IDENTITY(1,1) NOT NULL,
    model_key     NVARCHAR(128) NOT NULL,
    model_label   NVARCHAR(256) NULL,
    target_name   NVARCHAR(128) NOT NULL,
    model_type    NVARCHAR(128) NOT NULL,
    model_status  NVARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
    created_ts    DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    created_by    NVARCHAR(128) NOT NULL,
    retired_ts    DATETIME2(3) NULL,

    CONSTRAINT PK_PRICING_MODEL
        PRIMARY KEY (model_id),

    CONSTRAINT UQ_PRICING_MODEL_KEY
        UNIQUE (model_key),

    CONSTRAINT CK_PRICING_MODEL_STATUS
        CHECK (model_status IN ('ACTIVE', 'RETIRED', 'DISABLED'))
);
GO

CREATE TABLE pricing.PRICING_RATE_PACKAGE (
    rate_package_id        BIGINT IDENTITY(1,1) NOT NULL,
    parent_rate_package_id BIGINT NULL,
    model_id               BIGINT NULL,
    model_name             NVARCHAR(128) NOT NULL,
    model_version          NVARCHAR(64) NULL,
    package_version        INT NOT NULL,
    base_rate              DECIMAL(19,6) NOT NULL,
    effective_from_date    DATE NOT NULL,
    effective_to_date      DATE NULL,
    package_status         NVARCHAR(32) NOT NULL,
    created_ts             DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    created_by             NVARCHAR(128) NOT NULL,

    CONSTRAINT PK_PRICING_RATE_PACKAGE
        PRIMARY KEY (rate_package_id),

    CONSTRAINT FK_RATE_PACKAGE_PARENT
        FOREIGN KEY (parent_rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id),

    CONSTRAINT FK_RATE_PACKAGE_MODEL
        FOREIGN KEY (model_id)
        REFERENCES pricing.PRICING_MODEL(model_id)
);
GO

CREATE TABLE pricing.MODEL_RUN (
    model_run_id         BIGINT IDENTITY(1,1) NOT NULL,
    model_id             BIGINT NULL,
    dag_id               NVARCHAR(250) NOT NULL,
    airflow_run_id       NVARCHAR(250) NOT NULL,
    mlflow_experiment_id NVARCHAR(128) NULL,
    mlflow_run_id        NVARCHAR(128) NULL,
    manifest_id          NVARCHAR(128) NULL,
    export_id            NVARCHAR(128) NULL,
    model_name           NVARCHAR(128) NOT NULL,
    model_version        NVARCHAR(64) NULL,
    rate_package_id      BIGINT NULL,
    rating_workbook_path NVARCHAR(1024) NULL,
    run_status           NVARCHAR(32) NOT NULL,
    started_ts           DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    completed_ts         DATETIME2(3) NULL,
    created_by           NVARCHAR(128) NOT NULL,

    CONSTRAINT PK_MODEL_RUN
        PRIMARY KEY (model_run_id),

    CONSTRAINT FK_MODEL_RUN_MODEL
        FOREIGN KEY (model_id)
        REFERENCES pricing.PRICING_MODEL(model_id),

    CONSTRAINT FK_MODEL_RUN_MANIFEST
        FOREIGN KEY (manifest_id)
        REFERENCES pricing.DATASET_MANIFEST(manifest_id),

    CONSTRAINT FK_MODEL_RUN_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id)
);
GO

CREATE UNIQUE INDEX UX_MODEL_RUN_AIRFLOW
ON pricing.MODEL_RUN(dag_id, airflow_run_id, model_name);
GO

CREATE UNIQUE INDEX UX_MODEL_RUN_AIRFLOW_MODEL_ID
ON pricing.MODEL_RUN(dag_id, airflow_run_id, model_id)
WHERE model_id IS NOT NULL;
GO

CREATE TABLE pricing.PRICING_MODEL_DEPLOYMENT (
    deployment_id     BIGINT IDENTITY(1,1) NOT NULL,
    model_id          BIGINT NOT NULL,
    rate_package_id   BIGINT NOT NULL,
    deployment_slot   NVARCHAR(64) NOT NULL,
    effective_from_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    effective_to_ts   DATETIME2(3) NULL,
    deployed_by       NVARCHAR(128) NOT NULL,
    deployment_note   NVARCHAR(512) NULL,
    created_ts        DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),

    CONSTRAINT PK_PRICING_MODEL_DEPLOYMENT
        PRIMARY KEY (deployment_id),

    CONSTRAINT FK_MODEL_DEPLOYMENT_MODEL
        FOREIGN KEY (model_id)
        REFERENCES pricing.PRICING_MODEL(model_id),

    CONSTRAINT FK_MODEL_DEPLOYMENT_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id),

    CONSTRAINT CK_MODEL_DEPLOYMENT_EFFECTIVE_DATES
        CHECK (effective_to_ts IS NULL OR effective_to_ts > effective_from_ts)
);
GO

CREATE UNIQUE INDEX UX_MODEL_DEPLOYMENT_CURRENT
ON pricing.PRICING_MODEL_DEPLOYMENT(model_id, deployment_slot)
WHERE effective_to_ts IS NULL;
GO

CREATE TABLE pricing.PRICING_PACKAGE_POINTER (
    pointer_name     NVARCHAR(128) NOT NULL,
    model_id         BIGINT NULL,
    rate_package_id  BIGINT NOT NULL,
    updated_ts       DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    updated_by       NVARCHAR(128) NOT NULL,

    CONSTRAINT FK_PACKAGE_POINTER_MODEL
        FOREIGN KEY (model_id)
        REFERENCES pricing.PRICING_MODEL(model_id),

    CONSTRAINT FK_PACKAGE_POINTER_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id)
);
GO

CREATE UNIQUE INDEX UX_PACKAGE_POINTER_MODEL_SLOT
ON pricing.PRICING_PACKAGE_POINTER(model_id, pointer_name)
WHERE model_id IS NOT NULL;
GO

-- CV audit: replayable or materialized fold metadata, not row-per-policy keys.
CREATE TABLE pricing.CV_SPLIT_SET (
    split_set_id          NVARCHAR(128) NOT NULL,
    manifest_id           NVARCHAR(128) NOT NULL,
    split_mode            NVARCHAR(32) NOT NULL,
    splitter_class        NVARCHAR(256) NULL,
    splitter_params_json  NVARCHAR(MAX) NULL,
    row_order_sha256      CHAR(64) NOT NULL,
    row_count             BIGINT NOT NULL,
    fold_count            INT NOT NULL,
    groups_column         NVARCHAR(128) NULL,
    stratify_column       NVARCHAR(128) NULL,
    artifact_uri          NVARCHAR(1024) NULL,
    artifact_sha256       CHAR(64) NULL,
    runtime_metadata_json NVARCHAR(MAX) NULL,
    created_ts            DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    created_by            NVARCHAR(128) NOT NULL,

    CONSTRAINT PK_CV_SPLIT_SET
        PRIMARY KEY (split_set_id),

    CONSTRAINT FK_CV_SPLIT_SET_MANIFEST
        FOREIGN KEY (manifest_id)
        REFERENCES pricing.DATASET_MANIFEST(manifest_id),

    CONSTRAINT CK_CV_SPLIT_SET_MODE
        CHECK (split_mode IN ('REPLAYABLE', 'MATERIALIZED'))
);
GO

CREATE TABLE pricing.CV_FOLD (
    split_set_id  NVARCHAR(128) NOT NULL,
    fold_no       INT NOT NULL,
    n_train       BIGINT NOT NULL,
    n_test        BIGINT NOT NULL,

    CONSTRAINT PK_CV_FOLD
        PRIMARY KEY (split_set_id, fold_no),

    CONSTRAINT FK_CV_FOLD_SPLIT_SET
        FOREIGN KEY (split_set_id)
        REFERENCES pricing.CV_SPLIT_SET(split_set_id)
);
GO

CREATE TABLE pricing.CV_FOLD_METRIC (
    model_run_id  BIGINT NOT NULL,
    split_set_id  NVARCHAR(128) NOT NULL,
    fold_no       INT NOT NULL,
    metric_name   NVARCHAR(128) NOT NULL,
    metric_value  FLOAT NOT NULL,

    CONSTRAINT PK_CV_FOLD_METRIC
        PRIMARY KEY (model_run_id, split_set_id, fold_no, metric_name),

    CONSTRAINT FK_CV_FOLD_METRIC_MODEL_RUN
        FOREIGN KEY (model_run_id)
        REFERENCES pricing.MODEL_RUN(model_run_id),

    CONSTRAINT FK_CV_FOLD_METRIC_FOLD
        FOREIGN KEY (split_set_id, fold_no)
        REFERENCES pricing.CV_FOLD(split_set_id, fold_no)
);
GO

-- Rating lookup structure: package -> term -> cell -> rated feature levels.
CREATE TABLE pricing.PRICING_FEATURE (
    feature_id          BIGINT IDENTITY(1,1) NOT NULL,
    feature_name        NVARCHAR(128) NOT NULL,
    feature_value_type  NVARCHAR(32) NOT NULL,
    is_ordered          BIT NOT NULL DEFAULT 0,
    active_flag         BIT NOT NULL DEFAULT 1,

    CONSTRAINT PK_PRICING_FEATURE
        PRIMARY KEY (feature_id),

    CONSTRAINT UQ_PRICING_FEATURE_NAME
        UNIQUE (feature_name)
);
GO

CREATE TABLE pricing.PRICING_FEATURE_LEVEL_SET (
    level_set_id      BIGINT IDENTITY(1,1) NOT NULL,
    model_id          BIGINT NULL,
    feature_id        BIGINT NOT NULL,
    level_set_name    NVARCHAR(128) NOT NULL,
    level_set_type    NVARCHAR(64) NOT NULL,
    binning_strategy  NVARCHAR(64) NULL,
    grid_width        FLOAT NULL,
    created_ts        DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),

    CONSTRAINT PK_PRICING_FEATURE_LEVEL_SET
        PRIMARY KEY (level_set_id),

    CONSTRAINT FK_LEVEL_SET_MODEL
        FOREIGN KEY (model_id)
        REFERENCES pricing.PRICING_MODEL(model_id),

    CONSTRAINT FK_LEVEL_SET_FEATURE
        FOREIGN KEY (feature_id)
        REFERENCES pricing.PRICING_FEATURE(feature_id)
);
GO

CREATE UNIQUE INDEX UX_LEVEL_SET_MODEL_FEATURE_NAME
ON pricing.PRICING_FEATURE_LEVEL_SET(model_id, feature_id, level_set_name);
GO

CREATE TABLE pricing.PRICING_FEATURE_LEVEL (
    feature_level_id      BIGINT IDENTITY(1,1) NOT NULL,
    level_set_id          BIGINT NOT NULL,
    level_code            NVARCHAR(128) NOT NULL,
    level_label           NVARCHAR(256) NULL,
    order_index           INT NULL,
    lower_bound           FLOAT NULL,
    upper_bound           FLOAT NULL,
    representative_value  FLOAT NULL,
    is_missing            BIT NOT NULL DEFAULT 0,
    is_other              BIT NOT NULL DEFAULT 0,

    CONSTRAINT PK_PRICING_FEATURE_LEVEL
        PRIMARY KEY (feature_level_id),

    CONSTRAINT FK_FEATURE_LEVEL_SET
        FOREIGN KEY (level_set_id)
        REFERENCES pricing.PRICING_FEATURE_LEVEL_SET(level_set_id),

    CONSTRAINT UQ_FEATURE_LEVEL
        UNIQUE (level_set_id, level_code)
);
GO

CREATE TABLE pricing.PRICING_TERM (
    term_id                  BIGINT IDENTITY(1,1) NOT NULL,
    rate_package_id          BIGINT NOT NULL,
    term_name                NVARCHAR(128) NOT NULL,
    term_type                NVARCHAR(64) NOT NULL,
    sequence_no              INT NOT NULL,
    default_multiplier       DECIMAL(19,10) NOT NULL DEFAULT 1.0,
    default_log_coefficient  DECIMAL(19,12) NOT NULL DEFAULT 0.0,
    active_flag              BIT NOT NULL DEFAULT 1,

    CONSTRAINT PK_PRICING_TERM
        PRIMARY KEY (term_id),

    CONSTRAINT FK_TERM_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id),

    CONSTRAINT UQ_TERM_PACKAGE_NAME
        UNIQUE (rate_package_id, term_name)
);
GO

CREATE TABLE pricing.PRICING_TERM_FEATURE (
    term_id            BIGINT NOT NULL,
    position_no        SMALLINT NOT NULL,
    feature_id         BIGINT NOT NULL,
    level_set_id       BIGINT NOT NULL,
    input_column_name  NVARCHAR(128) NULL,

    CONSTRAINT PK_TERM_FEATURE
        PRIMARY KEY (term_id, position_no),

    CONSTRAINT FK_TERM_FEATURE_TERM
        FOREIGN KEY (term_id)
        REFERENCES pricing.PRICING_TERM(term_id),

    CONSTRAINT FK_TERM_FEATURE_FEATURE
        FOREIGN KEY (feature_id)
        REFERENCES pricing.PRICING_FEATURE(feature_id),

    CONSTRAINT FK_TERM_FEATURE_LEVEL_SET
        FOREIGN KEY (level_set_id)
        REFERENCES pricing.PRICING_FEATURE_LEVEL_SET(level_set_id)
);
GO

CREATE TABLE pricing.PRICING_RATE_CELL (
    cell_id          BIGINT IDENTITY(1,1) NOT NULL,
    term_id          BIGINT NOT NULL,
    cell_key_text    NVARCHAR(900) NOT NULL,
    cell_key_digest  VARBINARY(32) NOT NULL,
    multiplier       DECIMAL(19,10) NOT NULL,
    log_coefficient  DECIMAL(19,12) NOT NULL,
    exposure_weight  DECIMAL(19,4) NULL,
    record_count     BIGINT NULL,
    is_reference     BIT NOT NULL DEFAULT 0,
    is_default       BIT NOT NULL DEFAULT 0,
    is_deleted       BIT NOT NULL DEFAULT 0,

    CONSTRAINT PK_PRICING_RATE_CELL
        PRIMARY KEY (cell_id),

    CONSTRAINT FK_RATE_CELL_TERM
        FOREIGN KEY (term_id)
        REFERENCES pricing.PRICING_TERM(term_id),

    CONSTRAINT UQ_RATE_CELL
        UNIQUE (term_id, cell_key_digest)
);
GO

CREATE TABLE pricing.PRICING_RATE_CELL_LEVEL (
    cell_id           BIGINT NOT NULL,
    position_no       SMALLINT NOT NULL,
    feature_level_id  BIGINT NOT NULL,

    CONSTRAINT PK_RATE_CELL_LEVEL
        PRIMARY KEY (cell_id, position_no),

    CONSTRAINT FK_RATE_CELL_LEVEL_CELL
        FOREIGN KEY (cell_id)
        REFERENCES pricing.PRICING_RATE_CELL(cell_id),

    CONSTRAINT FK_RATE_CELL_LEVEL_LEVEL
        FOREIGN KEY (feature_level_id)
        REFERENCES pricing.PRICING_FEATURE_LEVEL(feature_level_id)
);
GO

-- Compiled read models: flattened outputs for easier downstream lookup.
CREATE TABLE pricing.PRICING_COMPILED_RATE_CELL (
    rate_package_id  BIGINT NOT NULL,
    term_id          BIGINT NOT NULL,
    cell_key_digest  VARBINARY(32) NOT NULL,
    term_name        NVARCHAR(128) NOT NULL,
    term_type        NVARCHAR(64) NOT NULL,
    sequence_no      INT NOT NULL,
    cell_key_text    NVARCHAR(900) NOT NULL,
    multiplier       DECIMAL(19,10) NOT NULL,
    log_coefficient  DECIMAL(19,12) NOT NULL,
    exposure_weight  DECIMAL(19,4) NULL,
    record_count     BIGINT NULL,
    is_default       BIT NOT NULL,
    is_reference     BIT NOT NULL,

    CONSTRAINT PK_COMPILED_RATE_CELL
        PRIMARY KEY (rate_package_id, term_id, cell_key_digest),

    CONSTRAINT FK_COMPILED_RATE_CELL_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id),

    CONSTRAINT FK_COMPILED_RATE_CELL_TERM
        FOREIGN KEY (term_id)
        REFERENCES pricing.PRICING_TERM(term_id)
);
GO

CREATE TABLE pricing.PRICING_COMPILED_1D_RATE_BAND (
    rate_package_id       BIGINT NOT NULL,
    term_id               BIGINT NOT NULL,
    feature_level_id      BIGINT NOT NULL,
    term_name             NVARCHAR(128) NOT NULL,
    feature_name          NVARCHAR(128) NOT NULL,
    level_code            NVARCHAR(128) NOT NULL,
    sort_order            INT NOT NULL,
    lower_bound           FLOAT NULL,
    upper_bound           FLOAT NULL,
    representative_value  FLOAT NULL,
    multiplier            DECIMAL(19,10) NOT NULL,
    log_coefficient       DECIMAL(19,12) NOT NULL,

    CONSTRAINT PK_COMPILED_1D_RATE_BAND
        PRIMARY KEY CLUSTERED (rate_package_id, term_id, sort_order, feature_level_id),

    CONSTRAINT FK_COMPILED_1D_RATE_BAND_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id),

    CONSTRAINT FK_COMPILED_1D_RATE_BAND_TERM
        FOREIGN KEY (term_id)
        REFERENCES pricing.PRICING_TERM(term_id),

    CONSTRAINT FK_COMPILED_1D_RATE_BAND_LEVEL
        FOREIGN KEY (feature_level_id)
        REFERENCES pricing.PRICING_FEATURE_LEVEL(feature_level_id)
);
GO
