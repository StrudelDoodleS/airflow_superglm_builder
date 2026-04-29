IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
    EXEC('CREATE SCHEMA pricing');
GO

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing_stg')
    EXEC('CREATE SCHEMA pricing_stg');
GO

IF OBJECT_ID('pricing_stg.STG_RATING_EXPORT', 'U') IS NULL
CREATE TABLE pricing_stg.STG_RATING_EXPORT (
    export_id             NVARCHAR(128) NOT NULL PRIMARY KEY,
    model_id              BIGINT NULL,
    model_name            NVARCHAR(128) NOT NULL,
    model_version         NVARCHAR(64) NULL,
    base_rate             DECIMAL(19,6) NOT NULL,
    effective_from_date   DATE NOT NULL,
    effective_to_date     DATE NULL,
    source_file           NVARCHAR(1024) NULL,
    created_ts            DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    created_by            NVARCHAR(128) NOT NULL
);
GO

IF OBJECT_ID('pricing_stg.STG_RATE_CELL', 'U') IS NULL
CREATE TABLE pricing_stg.STG_RATE_CELL (
    export_id          NVARCHAR(128) NOT NULL,
    row_id             INT NOT NULL,
    term_name          NVARCHAR(128) NOT NULL,
    term_type          NVARCHAR(64) NOT NULL,
    sequence_no        INT NOT NULL,
    cell_key_text      NVARCHAR(900) NOT NULL,
    multiplier         DECIMAL(19,10) NOT NULL,
    log_coefficient    DECIMAL(19,12) NOT NULL,
    exposure_weight    DECIMAL(19,4) NULL,
    record_count       BIGINT NULL,
    is_reference       BIT NOT NULL DEFAULT 0,
    is_default         BIT NOT NULL DEFAULT 0,

    CONSTRAINT PK_STG_RATE_CELL
        PRIMARY KEY (export_id, row_id),

    CONSTRAINT FK_STG_RATE_CELL_EXPORT
        FOREIGN KEY (export_id)
        REFERENCES pricing_stg.STG_RATING_EXPORT(export_id)
);
GO

IF OBJECT_ID('pricing_stg.STG_CELL_LEVEL', 'U') IS NULL
CREATE TABLE pricing_stg.STG_CELL_LEVEL (
    export_id              NVARCHAR(128) NOT NULL,
    row_id                 INT NOT NULL,
    position_no            SMALLINT NOT NULL,
    feature_name           NVARCHAR(128) NOT NULL,
    feature_value_type     NVARCHAR(32) NOT NULL,
    level_set_name         NVARCHAR(128) NOT NULL,
    level_set_type         NVARCHAR(64) NOT NULL,
    level_code             NVARCHAR(128) NOT NULL,
    level_label            NVARCHAR(256) NULL,
    order_index            INT NULL,
    lower_bound            FLOAT NULL,
    upper_bound            FLOAT NULL,
    representative_value   FLOAT NULL,
    is_missing             BIT NOT NULL DEFAULT 0,
    is_other               BIT NOT NULL DEFAULT 0,

    CONSTRAINT PK_STG_CELL_LEVEL
        PRIMARY KEY (export_id, row_id, position_no),

    CONSTRAINT FK_STG_CELL_LEVEL_CELL
        FOREIGN KEY (export_id, row_id)
        REFERENCES pricing_stg.STG_RATE_CELL(export_id, row_id)
);
GO
