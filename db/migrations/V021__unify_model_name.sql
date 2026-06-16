IF COL_LENGTH('pricing.PRICING_MODEL', 'model_name') IS NULL
AND COL_LENGTH('pricing.PRICING_MODEL', 'model_key') IS NOT NULL
    EXEC sp_rename 'pricing.PRICING_MODEL.model_key', 'model_name', 'COLUMN';
GO

IF EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = 'UQ_PRICING_MODEL_KEY'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_MODEL')
)
ALTER TABLE pricing.PRICING_MODEL DROP CONSTRAINT UQ_PRICING_MODEL_KEY;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = 'UQ_PRICING_MODEL_NAME'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_MODEL')
)
ALTER TABLE pricing.PRICING_MODEL
ADD CONSTRAINT UQ_PRICING_MODEL_NAME UNIQUE (model_name);
GO

CREATE OR ALTER VIEW pricing.V_ACTIVE_MODEL AS
SELECT
    model_id,
    model_name,
    model_label,
    target_name,
    model_type,
    model_status,
    created_ts,
    created_by,
    retired_ts
FROM pricing.PRICING_MODEL
WHERE model_status = 'ACTIVE';
GO

CREATE OR ALTER VIEW pricing.V_CURRENT_RATE_PACKAGE AS
SELECT
    d.deployment_id,
    d.deployment_slot,
    d.effective_from_ts,
    d.model_id,
    m.model_name,
    m.target_name,
    m.model_type,
    rp.rate_package_id,
    rp.parent_rate_package_id,
    rp.model_name AS package_model_name,
    rp.model_version,
    rp.package_version,
    rp.base_rate,
    rp.effective_from_date,
    rp.effective_to_date,
    rp.package_status,
    rp.created_ts,
    rp.created_by
FROM pricing.PRICING_MODEL_DEPLOYMENT d
JOIN pricing.PRICING_MODEL m
  ON m.model_id = d.model_id
JOIN pricing.PRICING_RATE_PACKAGE rp
  ON rp.rate_package_id = d.rate_package_id
WHERE d.effective_to_ts IS NULL;
GO

CREATE OR ALTER VIEW pricing.V_CURRENT_RATE_CELL AS
SELECT
    cur.model_id,
    cur.model_name,
    cur.deployment_slot,
    cur.rate_package_id,
    cur.package_version,
    c.term_id,
    c.cell_key_digest,
    c.term_name,
    c.term_type,
    c.sequence_no,
    c.cell_key_text,
    c.multiplier,
    c.log_coefficient,
    c.exposure_weight,
    c.record_count,
    c.is_default,
    c.is_reference
FROM pricing.V_CURRENT_RATE_PACKAGE cur
JOIN pricing.PRICING_COMPILED_RATE_CELL c
  ON c.rate_package_id = cur.rate_package_id;
GO

CREATE OR ALTER VIEW pricing.V_CURRENT_1D_RATE_BAND AS
SELECT
    cur.model_id,
    cur.model_name,
    cur.deployment_slot,
    cur.rate_package_id,
    cur.package_version,
    b.term_id,
    b.feature_level_id,
    b.term_name,
    b.feature_name,
    b.level_code,
    b.sort_order,
    b.lower_bound,
    b.upper_bound,
    b.representative_value,
    b.multiplier,
    b.log_coefficient
FROM pricing.V_CURRENT_RATE_PACKAGE cur
JOIN pricing.PRICING_COMPILED_1D_RATE_BAND b
  ON b.rate_package_id = cur.rate_package_id;
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
        term_name NVARCHAR(128) NOT NULL,
        term_type NVARCHAR(64) NOT NULL,
        match_type NVARCHAR(32) NOT NULL,
        feature_name NVARCHAR(128) NOT NULL,
        input_value NVARCHAR(4000) NULL,
        level_code NVARCHAR(128) NULL,
        multiplier FLOAT NOT NULL,
        log_coefficient FLOAT NOT NULL
    );

    INSERT INTO @matched (
        term_id,
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
            term_type
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

    SELECT @required_terms = COUNT(DISTINCT term_id)
    FROM pricing.V_CURRENT_RATE_CELL
    WHERE rate_package_id = @rate_package_id;

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
        EXP(SUM(log_coefficient)) AS relativity,
        @base_rate * @exposure * EXP(SUM(log_coefficient)) AS prediction,
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
        ORDER BY term_id;
    END;
END;
GO
