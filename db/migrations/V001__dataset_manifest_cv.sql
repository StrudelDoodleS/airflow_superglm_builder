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
