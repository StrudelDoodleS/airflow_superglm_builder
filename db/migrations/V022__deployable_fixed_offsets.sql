IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
    EXEC('CREATE SCHEMA pricing');
GO

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing_stg')
    EXEC('CREATE SCHEMA pricing_stg');
GO

IF OBJECT_ID('pricing_stg.STG_FIXED_OFFSET', 'U') IS NULL
CREATE TABLE pricing_stg.STG_FIXED_OFFSET (
    export_id              NVARCHAR(128) NOT NULL,
    term_name              NVARCHAR(128) NOT NULL,
    source_feature_name    NVARCHAR(128) NOT NULL,
    transform_type         NVARCHAR(64) NOT NULL,
    reference_value        FLOAT NOT NULL,
    coefficient            FLOAT NOT NULL,
    sequence_no            INT NOT NULL,

    CONSTRAINT PK_STG_FIXED_OFFSET
        PRIMARY KEY (export_id, term_name),

    CONSTRAINT UQ_STG_FIXED_OFFSET_SEQUENCE
        UNIQUE (export_id, sequence_no),

    CONSTRAINT FK_STG_FIXED_OFFSET_EXPORT
        FOREIGN KEY (export_id)
        REFERENCES pricing_stg.STG_RATING_EXPORT(export_id)
        ON DELETE CASCADE,

    CONSTRAINT CK_STG_FIXED_OFFSET_TRANSFORM
        CHECK (transform_type = 'LOG_RATIO'),

    CONSTRAINT CK_STG_FIXED_OFFSET_REFERENCE
        CHECK (reference_value > 0),

    CONSTRAINT CK_STG_FIXED_OFFSET_SEQUENCE
        CHECK (sequence_no > 0)
);
GO

IF OBJECT_ID('pricing.PRICING_FIXED_OFFSET', 'U') IS NULL
CREATE TABLE pricing.PRICING_FIXED_OFFSET (
    fixed_offset_id         BIGINT IDENTITY(1,1) PRIMARY KEY,
    rate_package_id         BIGINT NOT NULL,
    term_name               NVARCHAR(128) NOT NULL,
    source_feature_name     NVARCHAR(128) NOT NULL,
    transform_type          NVARCHAR(64) NOT NULL,
    reference_value         FLOAT NOT NULL,
    coefficient             FLOAT NOT NULL,
    sequence_no             INT NOT NULL,
    active_flag             BIT NOT NULL DEFAULT 1,

    CONSTRAINT FK_FIXED_OFFSET_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id)
        ON DELETE CASCADE,

    CONSTRAINT UQ_FIXED_OFFSET_PACKAGE_NAME
        UNIQUE (rate_package_id, term_name),

    CONSTRAINT UQ_FIXED_OFFSET_PACKAGE_SEQUENCE
        UNIQUE (rate_package_id, sequence_no),

    CONSTRAINT CK_FIXED_OFFSET_TRANSFORM
        CHECK (transform_type = 'LOG_RATIO'),

    CONSTRAINT CK_FIXED_OFFSET_REFERENCE
        CHECK (reference_value > 0),

    CONSTRAINT CK_FIXED_OFFSET_SEQUENCE
        CHECK (sequence_no > 0)
);
GO

CREATE OR ALTER VIEW pricing.V_CURRENT_FIXED_OFFSET AS
SELECT
    cur.model_id,
    cur.model_name,
    cur.deployment_slot,
    cur.rate_package_id,
    cur.package_version,
    fixed.fixed_offset_id,
    fixed.term_name,
    fixed.source_feature_name,
    fixed.transform_type,
    fixed.reference_value,
    fixed.coefficient,
    fixed.sequence_no
FROM pricing.V_CURRENT_RATE_PACKAGE AS cur
JOIN pricing.PRICING_FIXED_OFFSET AS fixed
  ON fixed.rate_package_id = cur.rate_package_id
WHERE fixed.active_flag = 1;
GO

CREATE OR ALTER TRIGGER pricing.TR_PRICING_RATE_PACKAGE_COPY_FIXED_OFFSETS
ON pricing.PRICING_RATE_PACKAGE
AFTER INSERT
AS
BEGIN
    SET NOCOUNT ON;

    INSERT INTO pricing.PRICING_FIXED_OFFSET (
        rate_package_id,
        term_name,
        source_feature_name,
        transform_type,
        reference_value,
        coefficient,
        sequence_no,
        active_flag
    )
    SELECT
        package.rate_package_id,
        staged.term_name,
        staged.source_feature_name,
        staged.transform_type,
        staged.reference_value,
        staged.coefficient,
        staged.sequence_no,
        1
    FROM inserted AS package
    JOIN pricing_stg.STG_FIXED_OFFSET AS staged
      ON staged.export_id = package.source_export_id
    WHERE package.parent_rate_package_id IS NULL

    UNION ALL

    SELECT
        package.rate_package_id,
        parent_offset.term_name,
        parent_offset.source_feature_name,
        parent_offset.transform_type,
        parent_offset.reference_value,
        parent_offset.coefficient,
        parent_offset.sequence_no,
        parent_offset.active_flag
    FROM inserted AS package
    JOIN pricing.PRICING_FIXED_OFFSET AS parent_offset
      ON parent_offset.rate_package_id = package.parent_rate_package_id
    WHERE package.parent_rate_package_id IS NOT NULL;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_PRICING_FIXED_OFFSET_IMMUTABLE_WRITE
ON pricing.PRICING_FIXED_OFFSET
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
        JOIN pricing.PRICING_RATE_PACKAGE AS package
          ON package.rate_package_id = changed.rate_package_id
        WHERE package.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.PRICING_MODEL_DEPLOYMENT AS deployment
               WHERE deployment.rate_package_id = package.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER PROCEDURE pricing.PREDICT_CURRENT_RATE
    @model_name NVARCHAR(128),
    @deployment_slot NVARCHAR(64),
    @features_json NVARCHAR(MAX),
    @exposure FLOAT = 1.0,
    @include_breakdown BIT = 0
AS
BEGIN
    SET NOCOUNT ON;

    IF ISJSON(@features_json) <> 1
    BEGIN;
        THROW 50000, 'features_json must be valid JSON', 1;
    END;

    IF @exposure IS NULL OR @exposure <= 0
    BEGIN;
        THROW 50001, 'exposure must be positive', 1;
    END;

    DECLARE @rate_package_id BIGINT;
    DECLARE @base_rate FLOAT;
    DECLARE @required_terms INT;
    DECLARE @matched_terms INT;

    SELECT TOP (1)
        @rate_package_id = rate_package_id,
        @base_rate = CAST(base_rate AS FLOAT)
    FROM pricing.V_CURRENT_RATE_PACKAGE
    WHERE model_name = @model_name
      AND deployment_slot = @deployment_slot;

    IF @rate_package_id IS NULL
    BEGIN;
        THROW 50002, 'No current deployed rate package found', 1;
    END;

    DECLARE @matched TABLE (
        term_id BIGINT NOT NULL PRIMARY KEY,
        sequence_no INT NOT NULL,
        term_name NVARCHAR(128) NOT NULL,
        term_type NVARCHAR(64) NOT NULL,
        match_type NVARCHAR(32) NOT NULL,
        feature_name NVARCHAR(128) NOT NULL,
        input_value NVARCHAR(4000) NULL,
        level_code NVARCHAR(4000) NULL,
        multiplier FLOAT NOT NULL,
        log_coefficient FLOAT NOT NULL
    );

    INSERT INTO @matched (
        term_id,
        sequence_no,
        term_name,
        term_type,
        match_type,
        feature_name,
        input_value,
        level_code,
        multiplier,
        log_coefficient
    )
    SELECT
        term.term_id,
        term.sequence_no,
        band.term_name,
        term.term_type,
        'BAND',
        band.feature_name,
        JSON_VALUE(@features_json, CONCAT('$.', band.feature_name)),
        band.level_code,
        CAST(band.multiplier AS FLOAT),
        CAST(band.log_coefficient AS FLOAT)
    FROM (
        SELECT DISTINCT
            term_id,
            term_name,
            term_type,
            sequence_no
        FROM pricing.V_CURRENT_RATE_CELL
        WHERE rate_package_id = @rate_package_id
    ) AS term
    CROSS APPLY (
        SELECT TOP (1) band.*
        FROM pricing.V_CURRENT_1D_RATE_BAND AS band
        WHERE band.rate_package_id = @rate_package_id
          AND band.term_id = term.term_id
          AND TRY_CONVERT(FLOAT, JSON_VALUE(@features_json, CONCAT('$.', band.feature_name))) IS NOT NULL
          AND TRY_CONVERT(FLOAT, JSON_VALUE(@features_json, CONCAT('$.', band.feature_name))) >= band.lower_bound
          AND (
              band.upper_bound IS NULL
              OR TRY_CONVERT(FLOAT, JSON_VALUE(@features_json, CONCAT('$.', band.feature_name))) < band.upper_bound
          )
        ORDER BY band.sort_order, band.feature_level_id
    ) AS band;

    INSERT INTO @matched (
        term_id,
        sequence_no,
        term_name,
        term_type,
        match_type,
        feature_name,
        input_value,
        level_code,
        multiplier,
        log_coefficient
    )
    SELECT
        cell.term_id,
        cell.sequence_no,
        cell.term_name,
        cell.term_type,
        'CELL',
        cell.term_name,
        JSON_VALUE(@features_json, CONCAT('$.', cell.term_name)),
        cell.cell_key_text,
        CAST(cell.multiplier AS FLOAT),
        CAST(cell.log_coefficient AS FLOAT)
    FROM pricing.V_CURRENT_RATE_CELL AS cell
    WHERE cell.rate_package_id = @rate_package_id
      AND NOT EXISTS (
          SELECT 1
          FROM @matched AS matched
          WHERE matched.term_id = cell.term_id
      )
      AND cell.cell_key_text = CONCAT(
          cell.term_name,
          '=',
          JSON_VALUE(@features_json, CONCAT('$.', cell.term_name))
      );

    INSERT INTO @matched (
        term_id,
        sequence_no,
        term_name,
        term_type,
        match_type,
        feature_name,
        input_value,
        level_code,
        multiplier,
        log_coefficient
    )
    SELECT
        cell.term_id,
        cell.sequence_no,
        cell.term_name,
        cell.term_type,
        'DEFAULT',
        cell.term_name,
        JSON_VALUE(@features_json, CONCAT('$.', cell.term_name)),
        cell.cell_key_text,
        CAST(cell.multiplier AS FLOAT),
        CAST(cell.log_coefficient AS FLOAT)
    FROM pricing.V_CURRENT_RATE_CELL AS cell
    WHERE cell.rate_package_id = @rate_package_id
      AND cell.is_default = 1
      AND NOT EXISTS (
          SELECT 1
          FROM @matched AS matched
          WHERE matched.term_id = cell.term_id
      );

    IF EXISTS (
        SELECT 1
        FROM pricing.V_CURRENT_FIXED_OFFSET AS fixed
        WHERE fixed.rate_package_id = @rate_package_id
          AND (
              TRY_CONVERT(
                  FLOAT,
                  JSON_VALUE(@features_json, CONCAT('$.', fixed.source_feature_name))
              ) IS NULL
              OR TRY_CONVERT(
                  FLOAT,
                  JSON_VALUE(@features_json, CONCAT('$.', fixed.source_feature_name))
              ) <= 0
          )
    )
    BEGIN;
        THROW 50004, 'Fixed offset source features must be present, numeric, and positive', 1;
    END;

    INSERT INTO @matched (
        term_id,
        sequence_no,
        term_name,
        term_type,
        match_type,
        feature_name,
        input_value,
        level_code,
        multiplier,
        log_coefficient
    )
    SELECT
        -fixed.fixed_offset_id,
        fixed.sequence_no,
        fixed.term_name,
        'FIXED_OFFSET',
        fixed.transform_type,
        fixed.source_feature_name,
        JSON_VALUE(@features_json, CONCAT('$.', fixed.source_feature_name)),
        CONCAT(
            'log(',
            fixed.source_feature_name,
            ' / ',
            CONVERT(NVARCHAR(64), fixed.reference_value),
            ')'
        ),
        POWER(
            input_value.numeric_value / fixed.reference_value,
            fixed.coefficient
        ),
        fixed.coefficient * LOG(
            input_value.numeric_value / fixed.reference_value
        )
    FROM pricing.V_CURRENT_FIXED_OFFSET AS fixed
    CROSS APPLY (
        SELECT TRY_CONVERT(
            FLOAT,
            JSON_VALUE(@features_json, CONCAT('$.', fixed.source_feature_name))
        ) AS numeric_value
    ) AS input_value
    WHERE fixed.rate_package_id = @rate_package_id;

    SELECT @required_terms =
        (
            SELECT COUNT(DISTINCT term_id)
            FROM pricing.V_CURRENT_RATE_CELL
            WHERE rate_package_id = @rate_package_id
        )
        +
        (
            SELECT COUNT(*)
            FROM pricing.V_CURRENT_FIXED_OFFSET
            WHERE rate_package_id = @rate_package_id
        );

    SELECT @matched_terms = COUNT(*)
    FROM @matched;

    IF @matched_terms <> @required_terms
    BEGIN;
        THROW 50003, 'Input features did not match every required term', 1;
    END;

    SELECT
        @model_name AS model_name,
        @deployment_slot AS deployment_slot,
        @rate_package_id AS rate_package_id,
        @base_rate AS base_rate,
        @exposure AS exposure,
        EXP(COALESCE(SUM(log_coefficient), 0.0)) AS relativity,
        @base_rate * @exposure * EXP(COALESCE(SUM(log_coefficient), 0.0)) AS prediction,
        @required_terms AS required_terms,
        @matched_terms AS matched_terms
    FROM @matched;

    IF @include_breakdown = 1
    BEGIN
        SELECT
            term_id,
            term_name,
            term_type,
            match_type,
            feature_name,
            input_value,
            level_code,
            multiplier,
            log_coefficient
        FROM @matched
        ORDER BY sequence_no, term_id;
    END;
END;
GO
