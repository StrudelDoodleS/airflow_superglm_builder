IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'mlops')
    EXEC('CREATE SCHEMA mlops');
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_CV_SPLIT_SET_MANIFEST_SPLIT'
      AND object_id = OBJECT_ID('pricing.CV_SPLIT_SET')
)
CREATE UNIQUE INDEX UX_CV_SPLIT_SET_MANIFEST_SPLIT
ON pricing.CV_SPLIT_SET(manifest_id, split_set_id);
GO

IF OBJECT_ID('mlops.CV_SPLIT_ROW', 'U') IS NULL
CREATE TABLE mlops.CV_SPLIT_ROW (
    split_set_id    NVARCHAR(128) NOT NULL,
    row_ordinal     BIGINT NOT NULL,
    row_key_text    NVARCHAR(900) NULL,
    row_key_digest  VARBINARY(32) NOT NULL,
    test_fold_no    INT NOT NULL,

    CONSTRAINT PK_CV_SPLIT_ROW
        PRIMARY KEY (split_set_id, row_ordinal),

    CONSTRAINT FK_CV_SPLIT_ROW_SPLIT_SET
        FOREIGN KEY (split_set_id)
        REFERENCES pricing.CV_SPLIT_SET(split_set_id),

    CONSTRAINT FK_CV_SPLIT_ROW_FOLD
        FOREIGN KEY (split_set_id, test_fold_no)
        REFERENCES pricing.CV_FOLD(split_set_id, fold_no)
);
GO

IF OBJECT_ID('mlops.MODEL_RUN_DATASET', 'U') IS NULL
CREATE TABLE mlops.MODEL_RUN_DATASET (
    model_run_id  BIGINT NOT NULL,
    manifest_id   NVARCHAR(128) NOT NULL,
    dataset_role  NVARCHAR(64) NOT NULL,

    CONSTRAINT PK_MODEL_RUN_DATASET
        PRIMARY KEY (model_run_id, dataset_role, manifest_id),

    CONSTRAINT FK_MODEL_RUN_DATASET_RUN
        FOREIGN KEY (model_run_id)
        REFERENCES pricing.MODEL_RUN(model_run_id),

    CONSTRAINT FK_MODEL_RUN_DATASET_MANIFEST
        FOREIGN KEY (manifest_id)
        REFERENCES pricing.DATASET_MANIFEST(manifest_id)
);
GO

IF OBJECT_ID('mlops.MODEL_RUN_SPLIT_SET', 'U') IS NULL
CREATE TABLE mlops.MODEL_RUN_SPLIT_SET (
    model_run_id  BIGINT NOT NULL,
    manifest_id   NVARCHAR(128) NOT NULL,
    split_set_id  NVARCHAR(128) NOT NULL,
    dataset_role  NVARCHAR(64) NOT NULL,
    split_role    NVARCHAR(64) NOT NULL,

    CONSTRAINT PK_MODEL_RUN_SPLIT_SET
        PRIMARY KEY (model_run_id, split_set_id, split_role),

    CONSTRAINT FK_MODEL_RUN_SPLIT_SET_RUN
        FOREIGN KEY (model_run_id)
        REFERENCES pricing.MODEL_RUN(model_run_id),

    CONSTRAINT FK_MODEL_RUN_SPLIT_SET_DATASET
        FOREIGN KEY (model_run_id, dataset_role, manifest_id)
        REFERENCES mlops.MODEL_RUN_DATASET(model_run_id, dataset_role, manifest_id),

    CONSTRAINT FK_MODEL_RUN_SPLIT_SET_SPLIT
        FOREIGN KEY (manifest_id, split_set_id)
        REFERENCES pricing.CV_SPLIT_SET(manifest_id, split_set_id)
);
GO

IF OBJECT_ID('mlops.MODEL_RUN_METRIC', 'U') IS NULL
CREATE TABLE mlops.MODEL_RUN_METRIC (
    model_run_id  BIGINT NOT NULL,
    metric_name   NVARCHAR(128) NOT NULL,
    metric_value  FLOAT NOT NULL,
    metric_scope  NVARCHAR(64) NULL,

    CONSTRAINT PK_MODEL_RUN_METRIC
        PRIMARY KEY (model_run_id, metric_name),

    CONSTRAINT FK_MODEL_RUN_METRIC_RUN
        FOREIGN KEY (model_run_id)
        REFERENCES pricing.MODEL_RUN(model_run_id)
);
GO
