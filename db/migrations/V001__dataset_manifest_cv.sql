IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
    EXEC('CREATE SCHEMA pricing');
GO

IF OBJECT_ID('pricing.DATASET_MANIFEST', 'U') IS NULL
CREATE TABLE pricing.DATASET_MANIFEST (
    manifest_id      NVARCHAR(128) NOT NULL PRIMARY KEY,
    dataset_name     NVARCHAR(128) NOT NULL,
    source_system    NVARCHAR(128) NULL,
    data_as_of_date  DATE NOT NULL,
    row_count        BIGINT NOT NULL,
    pk_columns_json  NVARCHAR(MAX) NOT NULL,
    target_column    NVARCHAR(128) NULL,
    weight_column    NVARCHAR(128) NULL,
    created_ts       DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    created_by       NVARCHAR(128) NOT NULL
);
GO

IF OBJECT_ID('pricing.DATASET_COLUMN', 'U') IS NULL
CREATE TABLE pricing.DATASET_COLUMN (
    manifest_id     NVARCHAR(128) NOT NULL,
    ordinal_no      INT NOT NULL,
    column_name     NVARCHAR(128) NOT NULL,
    column_role     NVARCHAR(32) NOT NULL,
    pandas_dtype    NVARCHAR(64) NOT NULL,
    null_count      BIGINT NOT NULL,
    distinct_count  BIGINT NULL,

    CONSTRAINT PK_DATASET_COLUMN PRIMARY KEY (manifest_id, ordinal_no),
    CONSTRAINT FK_DATASET_COLUMN_MANIFEST
        FOREIGN KEY (manifest_id)
        REFERENCES pricing.DATASET_MANIFEST(manifest_id)
);
GO

IF OBJECT_ID('pricing.DATASET_ROW_KEY', 'U') IS NULL
CREATE TABLE pricing.DATASET_ROW_KEY (
    manifest_id     NVARCHAR(128) NOT NULL,
    row_key_hash    VARBINARY(32) NOT NULL,
    source_pk_text  NVARCHAR(900) NOT NULL,
    row_ordinal     BIGINT NOT NULL,
    cv_fold_no      INT NOT NULL,

    CONSTRAINT PK_DATASET_ROW_KEY PRIMARY KEY (manifest_id, row_key_hash),
    CONSTRAINT FK_DATASET_ROW_KEY_MANIFEST
        FOREIGN KEY (manifest_id)
        REFERENCES pricing.DATASET_MANIFEST(manifest_id)
);
GO

IF OBJECT_ID('pricing.CV_SPLIT', 'U') IS NULL
CREATE TABLE pricing.CV_SPLIT (
    manifest_id       NVARCHAR(128) NOT NULL,
    split_no          INT NOT NULL,
    train_folds_json  NVARCHAR(MAX) NOT NULL,
    test_fold_no      INT NOT NULL,

    CONSTRAINT PK_CV_SPLIT PRIMARY KEY (manifest_id, split_no),
    CONSTRAINT FK_CV_SPLIT_MANIFEST
        FOREIGN KEY (manifest_id)
        REFERENCES pricing.DATASET_MANIFEST(manifest_id)
);
GO

IF OBJECT_ID('pricing.STG_DATASET_ROW_KEY', 'U') IS NULL
CREATE TABLE pricing.STG_DATASET_ROW_KEY (
    manifest_id     NVARCHAR(128) NOT NULL,
    source_pk_text  NVARCHAR(900) NOT NULL,
    row_ordinal     BIGINT NOT NULL,
    cv_fold_no      INT NOT NULL
);
GO
