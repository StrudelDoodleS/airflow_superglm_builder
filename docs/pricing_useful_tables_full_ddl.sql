CREATE SCHEMA raw;
CREATE SCHEMA mlops;
CREATE SCHEMA pricing;
CREATE SCHEMA pricing_runtime;

CREATE TABLE raw.FREMTPL_RAW (
    IDpol BIGINT NOT NULL,
    ClaimNb INT NOT NULL,
    Exposure FLOAT NOT NULL,
    Area NVARCHAR(16) NULL,
    VehPower INT NULL,
    VehAge INT NULL,
    DrivAge INT NULL,
    BonusMalus INT NULL,
    VehBrand NVARCHAR(64) NULL,
    VehGas NVARCHAR(16) NULL,
    Density FLOAT NULL,
    Region NVARCHAR(32) NULL,

    CONSTRAINT PK_FREMTPL_RAW
        PRIMARY KEY (IDpol)
);

CREATE TABLE mlops.DATASET_MANIFEST (
    manifest_id NVARCHAR(128) NOT NULL,
    dataset_name NVARCHAR(128) NOT NULL,
    source_schema NVARCHAR(128) NULL,
    source_table NVARCHAR(128) NULL,
    source_system NVARCHAR(128) NULL,
    data_as_of_date DATE NOT NULL,
    row_count BIGINT NOT NULL,
    pk_columns_json NVARCHAR(MAX) NOT NULL,
    target_column NVARCHAR(128) NULL,
    weight_column NVARCHAR(128) NULL,
    created_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    created_by NVARCHAR(128) NOT NULL,

    CONSTRAINT PK_DATASET_MANIFEST
        PRIMARY KEY (manifest_id)
);

CREATE TABLE mlops.DATASET_COLUMN (
    manifest_id NVARCHAR(128) NOT NULL,
    ordinal_no INT NOT NULL,
    column_name NVARCHAR(128) NOT NULL,
    column_role NVARCHAR(32) NOT NULL,
    pandas_dtype NVARCHAR(64) NOT NULL,
    null_count BIGINT NOT NULL,
    distinct_count BIGINT NULL,

    CONSTRAINT PK_DATASET_COLUMN
        PRIMARY KEY (manifest_id, ordinal_no),

    CONSTRAINT FK_DATASET_COLUMN_MANIFEST
        FOREIGN KEY (manifest_id)
        REFERENCES mlops.DATASET_MANIFEST(manifest_id)
);

CREATE TABLE pricing.MODEL (
    model_id BIGINT IDENTITY(1,1) NOT NULL,
    model_key NVARCHAR(128) NOT NULL,
    model_label NVARCHAR(256) NULL,
    target_name NVARCHAR(128) NOT NULL,
    model_type NVARCHAR(128) NOT NULL,
    model_status NVARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
    created_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    created_by NVARCHAR(128) NOT NULL,
    retired_ts DATETIME2(3) NULL,

    CONSTRAINT PK_MODEL
        PRIMARY KEY (model_id),

    CONSTRAINT UQ_MODEL_KEY
        UNIQUE (model_key),

    CONSTRAINT CK_MODEL_STATUS
        CHECK (model_status IN ('ACTIVE', 'RETIRED', 'DISABLED'))
);

CREATE TABLE mlops.MODEL_RUN (
    model_run_id BIGINT IDENTITY(1,1) NOT NULL,
    model_id BIGINT NULL,
    dag_id NVARCHAR(250) NOT NULL,
    airflow_run_id NVARCHAR(250) NOT NULL,
    mlflow_experiment_id NVARCHAR(128) NULL,
    mlflow_run_id NVARCHAR(128) NULL,
    model_version NVARCHAR(64) NULL,
    run_status NVARCHAR(32) NOT NULL,
    started_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    completed_ts DATETIME2(3) NULL,
    created_by NVARCHAR(128) NOT NULL,

    CONSTRAINT PK_MODEL_RUN
        PRIMARY KEY (model_run_id),

    CONSTRAINT UQ_MODEL_RUN_MODEL_RUN
        UNIQUE (model_id, model_run_id),

    CONSTRAINT FK_MODEL_RUN_MODEL
        FOREIGN KEY (model_id)
        REFERENCES pricing.MODEL(model_id),

    CONSTRAINT CK_MODEL_RUN_STATUS
        CHECK (run_status IN ('RUNNING', 'SUCCEEDED', 'FAILED', 'SKIPPED'))
);

CREATE UNIQUE INDEX UX_MODEL_RUN_AIRFLOW_MODEL
ON mlops.MODEL_RUN(dag_id, airflow_run_id, model_id)
WHERE model_id IS NOT NULL;

CREATE TABLE mlops.CV_SPLIT_SET (
    split_set_id NVARCHAR(128) NOT NULL,
    manifest_id NVARCHAR(128) NOT NULL,
    split_mode NVARCHAR(32) NOT NULL,
    splitter_class NVARCHAR(256) NULL,
    splitter_params_json NVARCHAR(MAX) NULL,
    row_order_sha256 CHAR(64) NOT NULL,
    row_count BIGINT NOT NULL,
    fold_count INT NOT NULL,
    groups_column NVARCHAR(128) NULL,
    stratify_column NVARCHAR(128) NULL,
    artifact_uri NVARCHAR(1024) NULL,
    artifact_sha256 CHAR(64) NULL,
    runtime_metadata_json NVARCHAR(MAX) NULL,
    created_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    created_by NVARCHAR(128) NOT NULL,

    CONSTRAINT PK_CV_SPLIT_SET
        PRIMARY KEY (split_set_id),

    CONSTRAINT UQ_CV_SPLIT_SET_MANIFEST_SPLIT
        UNIQUE (manifest_id, split_set_id),

    CONSTRAINT FK_CV_SPLIT_SET_MANIFEST
        FOREIGN KEY (manifest_id)
        REFERENCES mlops.DATASET_MANIFEST(manifest_id),

    CONSTRAINT CK_CV_SPLIT_SET_MODE
        CHECK (split_mode IN ('REPLAYABLE', 'MATERIALIZED'))
);

CREATE TABLE mlops.CV_FOLD (
    split_set_id NVARCHAR(128) NOT NULL,
    fold_no INT NOT NULL,
    n_train BIGINT NOT NULL,
    n_test BIGINT NOT NULL,

    CONSTRAINT PK_CV_FOLD
        PRIMARY KEY (split_set_id, fold_no),

    CONSTRAINT FK_CV_FOLD_SPLIT_SET
        FOREIGN KEY (split_set_id)
        REFERENCES mlops.CV_SPLIT_SET(split_set_id)
);

CREATE TABLE mlops.MODEL_RUN_DATASET (
    model_run_id BIGINT NOT NULL,
    manifest_id NVARCHAR(128) NOT NULL,
    dataset_role NVARCHAR(64) NOT NULL,

    CONSTRAINT PK_MODEL_RUN_DATASET
        PRIMARY KEY (model_run_id, dataset_role, manifest_id),

    CONSTRAINT FK_MODEL_RUN_DATASET_RUN
        FOREIGN KEY (model_run_id)
        REFERENCES mlops.MODEL_RUN(model_run_id),

    CONSTRAINT FK_MODEL_RUN_DATASET_MANIFEST
        FOREIGN KEY (manifest_id)
        REFERENCES mlops.DATASET_MANIFEST(manifest_id)
);

CREATE TABLE mlops.MODEL_RUN_SPLIT_SET (
    model_run_id BIGINT NOT NULL,
    manifest_id NVARCHAR(128) NOT NULL,
    split_set_id NVARCHAR(128) NOT NULL,
    dataset_role NVARCHAR(64) NOT NULL,
    split_role NVARCHAR(64) NOT NULL,

    CONSTRAINT PK_MODEL_RUN_SPLIT_SET
        PRIMARY KEY (model_run_id, split_set_id, split_role),

    CONSTRAINT FK_MODEL_RUN_SPLIT_SET_RUN
        FOREIGN KEY (model_run_id)
        REFERENCES mlops.MODEL_RUN(model_run_id),

    CONSTRAINT FK_MODEL_RUN_SPLIT_SET_DATASET
        FOREIGN KEY (model_run_id, dataset_role, manifest_id)
        REFERENCES mlops.MODEL_RUN_DATASET(model_run_id, dataset_role, manifest_id),

    CONSTRAINT FK_MODEL_RUN_SPLIT_SET_SPLIT
        FOREIGN KEY (manifest_id, split_set_id)
        REFERENCES mlops.CV_SPLIT_SET(manifest_id, split_set_id)
);

CREATE TABLE mlops.MODEL_RUN_METRIC (
    model_run_id BIGINT NOT NULL,
    metric_name NVARCHAR(128) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_scope NVARCHAR(64) NULL,

    CONSTRAINT PK_MODEL_RUN_METRIC
        PRIMARY KEY (model_run_id, metric_name),

    CONSTRAINT FK_MODEL_RUN_METRIC_RUN
        FOREIGN KEY (model_run_id)
        REFERENCES mlops.MODEL_RUN(model_run_id)
);

CREATE TABLE mlops.CV_FOLD_METRIC (
    model_run_id BIGINT NOT NULL,
    split_set_id NVARCHAR(128) NOT NULL,
    fold_no INT NOT NULL,
    metric_name NVARCHAR(128) NOT NULL,
    metric_value FLOAT NOT NULL,

    CONSTRAINT PK_CV_FOLD_METRIC
        PRIMARY KEY (model_run_id, split_set_id, fold_no, metric_name),

    CONSTRAINT FK_CV_FOLD_METRIC_MODEL_RUN
        FOREIGN KEY (model_run_id)
        REFERENCES mlops.MODEL_RUN(model_run_id),

    CONSTRAINT FK_CV_FOLD_METRIC_FOLD
        FOREIGN KEY (split_set_id, fold_no)
        REFERENCES mlops.CV_FOLD(split_set_id, fold_no)
);

CREATE TABLE pricing.RATE_PACKAGE (
    rate_package_id BIGINT IDENTITY(1,1) NOT NULL,
    parent_rate_package_id BIGINT NULL,
    model_id BIGINT NOT NULL,
    model_run_id BIGINT NULL,
    model_version NVARCHAR(64) NULL,
    package_version INT NOT NULL,
    base_rate DECIMAL(19,6) NOT NULL,
    effective_from_date DATE NOT NULL,
    effective_to_date DATE NULL,
    package_status NVARCHAR(32) NOT NULL,
    created_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    created_by NVARCHAR(128) NOT NULL,

    CONSTRAINT PK_RATE_PACKAGE
        PRIMARY KEY (rate_package_id),

    CONSTRAINT UQ_RATE_PACKAGE_MODEL_PACKAGE
        UNIQUE (model_id, rate_package_id),

    CONSTRAINT FK_RATE_PACKAGE_PARENT_SAME_MODEL
        FOREIGN KEY (model_id, parent_rate_package_id)
        REFERENCES pricing.RATE_PACKAGE(model_id, rate_package_id),

    CONSTRAINT FK_RATE_PACKAGE_MODEL
        FOREIGN KEY (model_id)
        REFERENCES pricing.MODEL(model_id),

    CONSTRAINT FK_RATE_PACKAGE_MODEL_RUN
        FOREIGN KEY (model_id, model_run_id)
        REFERENCES mlops.MODEL_RUN(model_id, model_run_id),

    CONSTRAINT CK_RATE_PACKAGE_STATUS
        CHECK (package_status IN ('DRAFT', 'PUBLISHED', 'RETIRED'))
);

CREATE UNIQUE INDEX UX_RATE_PACKAGE_MODEL_VERSION
ON pricing.RATE_PACKAGE(model_id, package_version);

CREATE TABLE pricing.FEATURE (
    feature_id BIGINT IDENTITY(1,1) NOT NULL,
    feature_name NVARCHAR(128) NOT NULL,
    feature_value_type NVARCHAR(32) NOT NULL,
    is_ordered BIT NOT NULL DEFAULT 0,
    active_flag BIT NOT NULL DEFAULT 1,

    CONSTRAINT PK_FEATURE
        PRIMARY KEY (feature_id),

    CONSTRAINT UQ_FEATURE_NAME
        UNIQUE (feature_name)
);

CREATE TABLE pricing.FEATURE_LEVEL_SET (
    level_set_id BIGINT IDENTITY(1,1) NOT NULL,
    model_id BIGINT NULL,
    feature_id BIGINT NOT NULL,
    level_set_name NVARCHAR(128) NOT NULL,
    level_set_type NVARCHAR(64) NOT NULL,
    binning_strategy NVARCHAR(64) NULL,
    grid_width FLOAT NULL,
    created_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),

    CONSTRAINT PK_FEATURE_LEVEL_SET
        PRIMARY KEY (level_set_id),

    CONSTRAINT UQ_FEATURE_LEVEL_SET_FEATURE_LEVEL_SET
        UNIQUE (feature_id, level_set_id),

    CONSTRAINT FK_FEATURE_LEVEL_SET_MODEL
        FOREIGN KEY (model_id)
        REFERENCES pricing.MODEL(model_id),

    CONSTRAINT FK_FEATURE_LEVEL_SET_FEATURE
        FOREIGN KEY (feature_id)
        REFERENCES pricing.FEATURE(feature_id)
);

CREATE UNIQUE INDEX UX_LEVEL_SET_MODEL_FEATURE_NAME
ON pricing.FEATURE_LEVEL_SET(model_id, feature_id, level_set_name);

CREATE TABLE pricing.FEATURE_LEVEL (
    feature_level_id BIGINT IDENTITY(1,1) NOT NULL,
    level_set_id BIGINT NOT NULL,
    level_code NVARCHAR(128) NOT NULL,
    level_label NVARCHAR(256) NULL,
    order_index INT NULL,
    lower_bound FLOAT NULL,
    upper_bound FLOAT NULL,
    representative_value FLOAT NULL,
    is_missing BIT NOT NULL DEFAULT 0,
    is_other BIT NOT NULL DEFAULT 0,

    CONSTRAINT PK_FEATURE_LEVEL
        PRIMARY KEY (feature_level_id),

    CONSTRAINT UQ_FEATURE_LEVEL_SET_LEVEL
        UNIQUE (level_set_id, feature_level_id),

    CONSTRAINT FK_FEATURE_LEVEL_SET
        FOREIGN KEY (level_set_id)
        REFERENCES pricing.FEATURE_LEVEL_SET(level_set_id),

    CONSTRAINT UQ_FEATURE_LEVEL
        UNIQUE (level_set_id, level_code)
);

CREATE TABLE pricing.TERM (
    term_id BIGINT IDENTITY(1,1) NOT NULL,
    rate_package_id BIGINT NOT NULL,
    term_name NVARCHAR(128) NOT NULL,
    term_type NVARCHAR(64) NOT NULL,
    sequence_no INT NOT NULL,
    default_multiplier DECIMAL(19,10) NOT NULL DEFAULT 1.0,
    default_log_coefficient DECIMAL(19,12) NOT NULL DEFAULT 0.0,
    active_flag BIT NOT NULL DEFAULT 1,

    CONSTRAINT PK_TERM
        PRIMARY KEY (term_id),

    CONSTRAINT FK_TERM_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.RATE_PACKAGE(rate_package_id),

    CONSTRAINT UQ_TERM_PACKAGE_NAME
        UNIQUE (rate_package_id, term_name)
);

CREATE TABLE pricing.TERM_FEATURE (
    term_id BIGINT NOT NULL,
    position_no SMALLINT NOT NULL,
    feature_id BIGINT NOT NULL,
    level_set_id BIGINT NOT NULL,
    input_column_name NVARCHAR(128) NULL,

    CONSTRAINT PK_TERM_FEATURE
        PRIMARY KEY (term_id, position_no),

    CONSTRAINT UQ_TERM_FEATURE_TERM_POSITION_LEVEL_SET
        UNIQUE (term_id, position_no, level_set_id),

    CONSTRAINT FK_TERM_FEATURE_TERM
        FOREIGN KEY (term_id)
        REFERENCES pricing.TERM(term_id),

    CONSTRAINT FK_TERM_FEATURE_LEVEL_SET_FEATURE
        FOREIGN KEY (feature_id, level_set_id)
        REFERENCES pricing.FEATURE_LEVEL_SET(feature_id, level_set_id)
);

CREATE TABLE pricing.RATE_CELL (
    cell_id BIGINT IDENTITY(1,1) NOT NULL,
    term_id BIGINT NOT NULL,
    cell_key_text NVARCHAR(900) NOT NULL,
    cell_key_digest VARBINARY(32) NOT NULL,
    multiplier DECIMAL(19,10) NOT NULL,
    log_coefficient DECIMAL(19,12) NOT NULL,
    exposure_weight DECIMAL(19,4) NULL,
    record_count BIGINT NULL,
    is_reference BIT NOT NULL DEFAULT 0,
    is_default BIT NOT NULL DEFAULT 0,
    is_deleted BIT NOT NULL DEFAULT 0,

    CONSTRAINT PK_RATE_CELL
        PRIMARY KEY (cell_id),

    CONSTRAINT UQ_RATE_CELL_CELL_TERM
        UNIQUE (cell_id, term_id),

    CONSTRAINT FK_RATE_CELL_TERM
        FOREIGN KEY (term_id)
        REFERENCES pricing.TERM(term_id)
);

CREATE UNIQUE INDEX UX_RATE_CELL_TERM_DIGEST_ACTIVE
ON pricing.RATE_CELL(term_id, cell_key_digest)
WHERE is_deleted = 0;

CREATE TABLE pricing.RATE_CELL_LEVEL (
    cell_id BIGINT NOT NULL,
    term_id BIGINT NOT NULL,
    position_no SMALLINT NOT NULL,
    level_set_id BIGINT NOT NULL,
    feature_level_id BIGINT NOT NULL,

    CONSTRAINT PK_RATE_CELL_LEVEL
        PRIMARY KEY (cell_id, position_no),

    CONSTRAINT FK_RATE_CELL_LEVEL_CELL
        FOREIGN KEY (cell_id, term_id)
        REFERENCES pricing.RATE_CELL(cell_id, term_id),

    CONSTRAINT FK_RATE_CELL_LEVEL_TERM_FEATURE
        FOREIGN KEY (term_id, position_no, level_set_id)
        REFERENCES pricing.TERM_FEATURE(term_id, position_no, level_set_id),

    CONSTRAINT FK_RATE_CELL_LEVEL_FEATURE_LEVEL
        FOREIGN KEY (level_set_id, feature_level_id)
        REFERENCES pricing.FEATURE_LEVEL(level_set_id, feature_level_id)
);

CREATE TABLE pricing.MODEL_DEPLOYMENT (
    deployment_id BIGINT IDENTITY(1,1) NOT NULL,
    model_id BIGINT NOT NULL,
    rate_package_id BIGINT NOT NULL,
    deployment_slot NVARCHAR(64) NOT NULL,
    effective_from_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    effective_to_ts DATETIME2(3) NULL,
    deployed_by NVARCHAR(128) NOT NULL,
    deployment_note NVARCHAR(512) NULL,
    created_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),

    CONSTRAINT PK_MODEL_DEPLOYMENT
        PRIMARY KEY (deployment_id),

    CONSTRAINT FK_MODEL_DEPLOYMENT_MODEL
        FOREIGN KEY (model_id)
        REFERENCES pricing.MODEL(model_id),

    CONSTRAINT FK_MODEL_DEPLOYMENT_PACKAGE
        FOREIGN KEY (model_id, rate_package_id)
        REFERENCES pricing.RATE_PACKAGE(model_id, rate_package_id),

    CONSTRAINT CK_MODEL_DEPLOYMENT_EFFECTIVE_DATES
        CHECK (effective_to_ts IS NULL OR effective_to_ts > effective_from_ts)
);

CREATE UNIQUE INDEX UX_MODEL_DEPLOYMENT_CURRENT
ON pricing.MODEL_DEPLOYMENT(model_id, deployment_slot)
WHERE effective_to_ts IS NULL;

CREATE OR ALTER TRIGGER pricing.TR_MODEL_DEPLOYMENT_PACKAGE_GUARD
ON pricing.MODEL_DEPLOYMENT
AFTER INSERT, UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM inserted d
        LEFT JOIN pricing.RATE_PACKAGE p
            ON p.rate_package_id = d.rate_package_id
           AND p.model_id = d.model_id
        WHERE p.rate_package_id IS NULL
           OR p.package_status <> 'PUBLISHED'
    )
    BEGIN;
        THROW 51001, 'rate package deployments must reference PUBLISHED packages for the same model_id.', 1;
    END;
END;

CREATE OR ALTER VIEW pricing.V_CURRENT_RATE_PACKAGE AS
SELECT
    d.deployment_id,
    d.deployment_slot,
    d.model_id,
    m.model_key,
    m.model_label,
    d.rate_package_id,
    p.package_version,
    p.model_version,
    p.base_rate,
    p.package_status,
    p.effective_from_date,
    p.effective_to_date,
    d.effective_from_ts AS deployed_from_ts,
    d.deployed_by
FROM pricing.MODEL_DEPLOYMENT AS d
JOIN pricing.RATE_PACKAGE AS p
    ON p.rate_package_id = d.rate_package_id
   AND p.model_id = d.model_id
JOIN pricing.MODEL AS m
    ON m.model_id = d.model_id
WHERE d.effective_to_ts IS NULL;

CREATE OR ALTER VIEW pricing_runtime.V_COMPILED_RATE_CELL AS
SELECT
    p.rate_package_id,
    t.term_id,
    c.cell_id,
    t.term_name,
    t.term_type,
    t.sequence_no,
    c.cell_key_text,
    c.multiplier,
    c.log_coefficient,
    c.exposure_weight,
    c.record_count,
    c.is_default,
    c.is_reference
FROM pricing.RATE_PACKAGE AS p
JOIN pricing.TERM AS t
    ON t.rate_package_id = p.rate_package_id
JOIN pricing.RATE_CELL AS c
    ON c.term_id = t.term_id
WHERE c.is_deleted = 0;

CREATE OR ALTER VIEW pricing_runtime.V_COMPILED_RATE_CELL_LEVEL AS
SELECT
    p.rate_package_id,
    t.term_id,
    c.cell_id,
    rcl.position_no,
    f.feature_id,
    rcl.level_set_id,
    fl.feature_level_id,
    f.feature_name,
    fl.level_code,
    fl.level_label
FROM pricing.RATE_PACKAGE AS p
JOIN pricing.TERM AS t
    ON t.rate_package_id = p.rate_package_id
JOIN pricing.RATE_CELL AS c
    ON c.term_id = t.term_id
JOIN pricing.RATE_CELL_LEVEL AS rcl
    ON rcl.cell_id = c.cell_id
   AND rcl.term_id = c.term_id
JOIN pricing.FEATURE_LEVEL AS fl
    ON fl.level_set_id = rcl.level_set_id
   AND fl.feature_level_id = rcl.feature_level_id
JOIN pricing.FEATURE_LEVEL_SET AS fls
    ON fls.level_set_id = rcl.level_set_id
JOIN pricing.FEATURE AS f
    ON f.feature_id = fls.feature_id
WHERE c.is_deleted = 0;

CREATE OR ALTER VIEW pricing_runtime.V_COMPILED_1D_RATE_BAND AS
SELECT
    p.rate_package_id,
    t.term_id,
    rcl.level_set_id,
    fl.feature_level_id,
    t.term_name,
    f.feature_name,
    fl.level_code,
    fl.order_index AS sort_order,
    fl.lower_bound,
    fl.upper_bound,
    fl.representative_value,
    c.multiplier,
    c.log_coefficient
FROM pricing.RATE_PACKAGE AS p
JOIN pricing.TERM AS t
    ON t.rate_package_id = p.rate_package_id
JOIN pricing.RATE_CELL AS c
    ON c.term_id = t.term_id
JOIN pricing.RATE_CELL_LEVEL AS rcl
    ON rcl.cell_id = c.cell_id
   AND rcl.term_id = c.term_id
JOIN pricing.FEATURE_LEVEL AS fl
    ON fl.level_set_id = rcl.level_set_id
   AND fl.feature_level_id = rcl.feature_level_id
JOIN pricing.FEATURE_LEVEL_SET AS fls
    ON fls.level_set_id = rcl.level_set_id
JOIN pricing.FEATURE AS f
    ON f.feature_id = fls.feature_id
WHERE c.is_deleted = 0
  AND t.term_type IN ('GLM_1D', 'SPLINE_1D', 'DISCRETIZED_SPLINE_1D');

GO

CREATE OR ALTER TRIGGER pricing.TR_RATE_PACKAGE_IMMUTABLE_UPDATE_DELETE
ON pricing.RATE_PACKAGE
AFTER UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM deleted AS d
        WHERE d.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.MODEL_DEPLOYMENT AS md
               WHERE md.rate_package_id = d.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_TERM_IMMUTABLE_WRITE
ON pricing.TERM
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM (
            SELECT rate_package_id FROM inserted
            UNION
            SELECT rate_package_id FROM deleted
        ) AS changed
        JOIN pricing.RATE_PACKAGE AS rp
          ON rp.rate_package_id = changed.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.MODEL_DEPLOYMENT AS md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_TERM_FEATURE_IMMUTABLE_WRITE
ON pricing.TERM_FEATURE
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM (
            SELECT term_id FROM inserted
            UNION
            SELECT term_id FROM deleted
        ) AS changed
        JOIN pricing.TERM AS t
          ON t.term_id = changed.term_id
        JOIN pricing.RATE_PACKAGE AS rp
          ON rp.rate_package_id = t.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.MODEL_DEPLOYMENT AS md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_RATE_CELL_IMMUTABLE_WRITE
ON pricing.RATE_CELL
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM (
            SELECT term_id FROM inserted
            UNION
            SELECT term_id FROM deleted
        ) AS changed
        JOIN pricing.TERM AS t
          ON t.term_id = changed.term_id
        JOIN pricing.RATE_PACKAGE AS rp
          ON rp.rate_package_id = t.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.MODEL_DEPLOYMENT AS md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_RATE_CELL_LEVEL_IMMUTABLE_WRITE
ON pricing.RATE_CELL_LEVEL
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM (
            SELECT cell_id FROM inserted
            UNION
            SELECT cell_id FROM deleted
        ) AS changed
        JOIN pricing.RATE_CELL AS rc
          ON rc.cell_id = changed.cell_id
        JOIN pricing.TERM AS t
          ON t.term_id = rc.term_id
        JOIN pricing.RATE_PACKAGE AS rp
          ON rp.rate_package_id = t.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.MODEL_DEPLOYMENT AS md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_FEATURE_LEVEL_IMMUTABLE_WRITE
ON pricing.FEATURE_LEVEL
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM (
            SELECT level_set_id FROM inserted
            UNION
            SELECT level_set_id FROM deleted
        ) AS changed
        JOIN pricing.TERM_FEATURE AS tf
          ON tf.level_set_id = changed.level_set_id
        JOIN pricing.TERM AS t
          ON t.term_id = tf.term_id
        JOIN pricing.RATE_PACKAGE AS rp
          ON rp.rate_package_id = t.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.MODEL_DEPLOYMENT AS md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_FEATURE_LEVEL_SET_IMMUTABLE_WRITE
ON pricing.FEATURE_LEVEL_SET
AFTER UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM deleted AS changed
        JOIN pricing.TERM_FEATURE AS tf
          ON tf.level_set_id = changed.level_set_id
        JOIN pricing.TERM AS t
          ON t.term_id = tf.term_id
        JOIN pricing.RATE_PACKAGE AS rp
          ON rp.rate_package_id = t.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.MODEL_DEPLOYMENT AS md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_FEATURE_IMMUTABLE_WRITE
ON pricing.FEATURE
AFTER UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM deleted AS changed
        JOIN pricing.FEATURE_LEVEL_SET AS fls
          ON fls.feature_id = changed.feature_id
        JOIN pricing.TERM_FEATURE AS tf
          ON tf.level_set_id = fls.level_set_id
        JOIN pricing.TERM AS t
          ON t.term_id = tf.term_id
        JOIN pricing.RATE_PACKAGE AS rp
          ON rp.rate_package_id = t.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.MODEL_DEPLOYMENT AS md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing_runtime.TR_COMPILED_RATE_CELL_IMMUTABLE_WRITE
ON pricing_runtime.V_COMPILED_RATE_CELL
INSTEAD OF INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;
    THROW 51000, 'Compiled runtime views are immutable. Create a new package revision.', 1;
END;
GO

CREATE OR ALTER TRIGGER pricing_runtime.TR_COMPILED_1D_RATE_BAND_IMMUTABLE_WRITE
ON pricing_runtime.V_COMPILED_1D_RATE_BAND
INSTEAD OF INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;
    THROW 51000, 'Compiled runtime views are immutable. Create a new package revision.', 1;
END;
GO
