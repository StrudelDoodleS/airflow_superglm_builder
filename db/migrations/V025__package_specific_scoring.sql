IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
    EXEC('CREATE SCHEMA pricing');
GO

CREATE OR ALTER PROCEDURE pricing.PREDICT_RATE_PACKAGE
    @rate_package_id BIGINT,
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

    DECLARE @model_name NVARCHAR(128);
    DECLARE @base_rate FLOAT;
    DECLARE @required_terms INT;
    DECLARE @matched_terms INT;

    SELECT
        @model_name = model_name,
        @base_rate = CAST(base_rate AS FLOAT)
    FROM pricing.PRICING_RATE_PACKAGE
    WHERE rate_package_id = @rate_package_id
      AND package_status IN ('DRAFT', 'PUBLISHED');

    IF @model_name IS NULL
    BEGIN;
        THROW 50002, 'Rate package is not available for explicit scoring', 1;
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
        FROM pricing.PRICING_COMPILED_RATE_CELL
        WHERE rate_package_id = @rate_package_id
    ) AS term
    CROSS APPLY (
        SELECT TOP (1) band.*
        FROM pricing.PRICING_COMPILED_1D_RATE_BAND AS band
        WHERE band.rate_package_id = @rate_package_id
          AND band.term_id = term.term_id
          AND TRY_CONVERT(
              FLOAT,
              JSON_VALUE(@features_json, CONCAT('$.', band.feature_name))
          ) IS NOT NULL
          AND TRY_CONVERT(
              FLOAT,
              JSON_VALUE(@features_json, CONCAT('$.', band.feature_name))
          ) >= band.lower_bound
          AND (
              band.upper_bound IS NULL
              OR TRY_CONVERT(
                  FLOAT,
                  JSON_VALUE(@features_json, CONCAT('$.', band.feature_name))
              ) < band.upper_bound
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
        'NUMERIC',
        term_feature.input_column_name,
        raw_input.input_value,
        feature_level.level_code,
        EXP(numeric_input.numeric_value * CAST(cell.log_coefficient AS FLOAT)),
        numeric_input.numeric_value * CAST(cell.log_coefficient AS FLOAT)
    FROM pricing.PRICING_COMPILED_RATE_CELL AS cell
    JOIN pricing.PRICING_RATE_CELL AS source_cell
      ON source_cell.term_id = cell.term_id
     AND source_cell.cell_key_digest = cell.cell_key_digest
    JOIN pricing.PRICING_TERM_FEATURE AS term_feature
      ON term_feature.term_id = cell.term_id
     AND term_feature.position_no = 1
    JOIN pricing.PRICING_RATE_CELL_LEVEL AS cell_level
      ON cell_level.cell_id = source_cell.cell_id
     AND cell_level.position_no = term_feature.position_no
    JOIN pricing.PRICING_FEATURE_LEVEL AS feature_level
      ON feature_level.feature_level_id = cell_level.feature_level_id
     AND feature_level.level_set_id = term_feature.level_set_id
    CROSS APPLY (
        SELECT JSON_VALUE(
            @features_json,
            CONCAT('$.', term_feature.input_column_name)
        ) AS input_value
    ) AS raw_input
    CROSS APPLY (
        SELECT TRY_CONVERT(FLOAT, raw_input.input_value) AS numeric_value
    ) AS numeric_input
    WHERE cell.rate_package_id = @rate_package_id
      AND cell.term_type = 'NUMERIC_MAIN'
      AND LOWER(feature_level.level_code) = 'per_unit'
      AND numeric_input.numeric_value IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM @matched AS matched
          WHERE matched.term_id = cell.term_id
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
        'INTERACTION',
        cell.term_name,
        cell.cell_key_text,
        cell.cell_key_text,
        CAST(cell.multiplier AS FLOAT),
        CAST(cell.log_coefficient AS FLOAT)
    FROM pricing.PRICING_COMPILED_RATE_CELL AS cell
    JOIN pricing.PRICING_RATE_CELL AS source_cell
      ON source_cell.term_id = cell.term_id
     AND source_cell.cell_key_digest = cell.cell_key_digest
    WHERE cell.rate_package_id = @rate_package_id
      AND cell.term_type = 'CATEGORICAL_INTERACTION'
      AND 2 = (
          SELECT COUNT(*)
          FROM pricing.PRICING_TERM_FEATURE AS term_feature
          WHERE term_feature.term_id = cell.term_id
      )
      AND 2 = (
          SELECT COUNT(*)
          FROM pricing.PRICING_RATE_CELL_LEVEL AS cell_level
          WHERE cell_level.cell_id = source_cell.cell_id
      )
      AND NOT EXISTS (
          SELECT 1
          FROM pricing.PRICING_TERM_FEATURE AS term_feature
          WHERE term_feature.term_id = cell.term_id
            AND NOT EXISTS (
                SELECT 1
                FROM pricing.PRICING_RATE_CELL_LEVEL AS cell_level
                JOIN pricing.PRICING_FEATURE_LEVEL AS feature_level
                  ON feature_level.feature_level_id = cell_level.feature_level_id
                 AND feature_level.level_set_id = term_feature.level_set_id
                WHERE cell_level.cell_id = source_cell.cell_id
                  AND cell_level.position_no = term_feature.position_no
                  AND feature_level.level_code = JSON_VALUE(
                      @features_json,
                      CONCAT('$.', term_feature.input_column_name)
                  )
            )
      )
      AND NOT EXISTS (
          SELECT 1
          FROM @matched AS matched
          WHERE matched.term_id = cell.term_id
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
        'CELL',
        cell.term_name,
        JSON_VALUE(@features_json, CONCAT('$.', cell.term_name)),
        cell.cell_key_text,
        CAST(cell.multiplier AS FLOAT),
        CAST(cell.log_coefficient AS FLOAT)
    FROM pricing.PRICING_COMPILED_RATE_CELL AS cell
    WHERE cell.rate_package_id = @rate_package_id
      AND cell.term_type NOT IN ('NUMERIC_MAIN', 'CATEGORICAL_INTERACTION')
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
    FROM pricing.PRICING_COMPILED_RATE_CELL AS cell
    WHERE cell.rate_package_id = @rate_package_id
      AND cell.term_type NOT IN ('NUMERIC_MAIN', 'CATEGORICAL_INTERACTION')
      AND cell.is_default = 1
      AND NOT EXISTS (
          SELECT 1
          FROM @matched AS matched
          WHERE matched.term_id = cell.term_id
      );

    SELECT @required_terms = COUNT(DISTINCT term_id)
    FROM pricing.PRICING_COMPILED_RATE_CELL
    WHERE rate_package_id = @rate_package_id;

    SELECT @matched_terms = COUNT(*)
    FROM @matched;

    IF @matched_terms <> @required_terms
    BEGIN;
        THROW 50003, 'Input features did not match every required term', 1;
    END;

    SELECT
        @model_name AS model_name,
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
