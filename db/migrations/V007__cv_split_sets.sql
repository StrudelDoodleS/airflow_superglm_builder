IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
    EXEC('CREATE SCHEMA pricing');
GO

IF OBJECT_ID('pricing.CV_SPLIT_SET', 'U') IS NULL
CREATE TABLE pricing.CV_SPLIT_SET (
    split_set_id          NVARCHAR(128) NOT NULL PRIMARY KEY,
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

    CONSTRAINT FK_CV_SPLIT_SET_MANIFEST
        FOREIGN KEY (manifest_id)
        REFERENCES pricing.DATASET_MANIFEST(manifest_id),

    CONSTRAINT CK_CV_SPLIT_SET_MODE
        CHECK (split_mode IN ('REPLAYABLE', 'MATERIALIZED'))
);
GO

IF OBJECT_ID('pricing.CV_FOLD', 'U') IS NULL
CREATE TABLE pricing.CV_FOLD (
    split_set_id     NVARCHAR(128) NOT NULL,
    fold_no          INT NOT NULL,
    n_train          BIGINT NOT NULL,
    n_test           BIGINT NOT NULL,

    CONSTRAINT PK_CV_FOLD
        PRIMARY KEY (split_set_id, fold_no),

    CONSTRAINT FK_CV_FOLD_SPLIT_SET
        FOREIGN KEY (split_set_id)
        REFERENCES pricing.CV_SPLIT_SET(split_set_id)
);
GO

IF OBJECT_ID('pricing.CV_FOLD_METRIC', 'U') IS NULL
CREATE TABLE pricing.CV_FOLD_METRIC (
    model_run_id     BIGINT NOT NULL,
    split_set_id     NVARCHAR(128) NOT NULL,
    fold_no          INT NOT NULL,
    metric_name      NVARCHAR(128) NOT NULL,
    metric_value     FLOAT NOT NULL,

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
