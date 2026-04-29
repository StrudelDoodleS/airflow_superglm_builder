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

IF OBJECT_ID('pricing.STG_RATING_EXPORT', 'U') IS NOT NULL
BEGIN
    INSERT INTO pricing_stg.STG_RATING_EXPORT (
        export_id,
        model_id,
        model_name,
        model_version,
        base_rate,
        effective_from_date,
        effective_to_date,
        source_file,
        created_ts,
        created_by
    )
    SELECT
        export_id,
        CASE
            WHEN COL_LENGTH('pricing.STG_RATING_EXPORT', 'model_id') IS NULL THEN NULL
            ELSE model_id
        END,
        model_name,
        model_version,
        base_rate,
        effective_from_date,
        effective_to_date,
        source_file,
        created_ts,
        created_by
    FROM pricing.STG_RATING_EXPORT src
    WHERE NOT EXISTS (
        SELECT 1
        FROM pricing_stg.STG_RATING_EXPORT tgt
        WHERE tgt.export_id = src.export_id
    );
END;
GO

IF OBJECT_ID('pricing.STG_RATE_CELL', 'U') IS NOT NULL
BEGIN
    INSERT INTO pricing_stg.STG_RATE_CELL (
        export_id,
        row_id,
        term_name,
        term_type,
        sequence_no,
        cell_key_text,
        multiplier,
        log_coefficient,
        exposure_weight,
        record_count,
        is_reference,
        is_default
    )
    SELECT
        export_id,
        row_id,
        term_name,
        term_type,
        sequence_no,
        cell_key_text,
        multiplier,
        log_coefficient,
        exposure_weight,
        record_count,
        is_reference,
        is_default
    FROM pricing.STG_RATE_CELL src
    WHERE NOT EXISTS (
        SELECT 1
        FROM pricing_stg.STG_RATE_CELL tgt
        WHERE tgt.export_id = src.export_id
          AND tgt.row_id = src.row_id
    );
END;
GO

IF OBJECT_ID('pricing.STG_CELL_LEVEL', 'U') IS NOT NULL
BEGIN
    INSERT INTO pricing_stg.STG_CELL_LEVEL (
        export_id,
        row_id,
        position_no,
        feature_name,
        feature_value_type,
        level_set_name,
        level_set_type,
        level_code,
        level_label,
        order_index,
        lower_bound,
        upper_bound,
        representative_value,
        is_missing,
        is_other
    )
    SELECT
        export_id,
        row_id,
        position_no,
        feature_name,
        feature_value_type,
        level_set_name,
        level_set_type,
        level_code,
        level_label,
        order_index,
        lower_bound,
        upper_bound,
        representative_value,
        is_missing,
        is_other
    FROM pricing.STG_CELL_LEVEL src
    WHERE NOT EXISTS (
        SELECT 1
        FROM pricing_stg.STG_CELL_LEVEL tgt
        WHERE tgt.export_id = src.export_id
          AND tgt.row_id = src.row_id
          AND tgt.position_no = src.position_no
    );
END;
GO

IF OBJECT_ID('pricing.STG_CELL_LEVEL', 'U') IS NOT NULL
    DROP TABLE pricing.STG_CELL_LEVEL;
GO

IF OBJECT_ID('pricing.STG_RATE_CELL', 'U') IS NOT NULL
    DROP TABLE pricing.STG_RATE_CELL;
GO

IF OBJECT_ID('pricing.STG_RATING_EXPORT', 'U') IS NOT NULL
    DROP TABLE pricing.STG_RATING_EXPORT;
GO

IF OBJECT_ID('pricing.STG_DATASET_ROW_KEY', 'U') IS NOT NULL
    DROP TABLE pricing.STG_DATASET_ROW_KEY;
GO

IF OBJECT_ID('pricing.CV_SPLIT', 'U') IS NOT NULL
    DROP TABLE pricing.CV_SPLIT;
GO

IF OBJECT_ID('pricing.DATASET_ROW_KEY', 'U') IS NOT NULL
    DROP TABLE pricing.DATASET_ROW_KEY;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_STG_RATING_EXPORT_MODEL'
)
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
    ADD CONSTRAINT FK_STG_RATING_EXPORT_MODEL
        FOREIGN KEY (model_id)
        REFERENCES pricing.PRICING_MODEL(model_id);
GO
