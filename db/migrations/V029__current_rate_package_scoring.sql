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

    SELECT TOP (1)
        @rate_package_id = rate_package_id
    FROM pricing.V_CURRENT_RATE_PACKAGE
    WHERE model_name = @model_name
      AND deployment_slot = @deployment_slot;

    IF @rate_package_id IS NULL
    BEGIN;
        THROW 50002, 'No current deployed rate package found', 1;
    END;

    EXEC pricing.PREDICT_RATE_PACKAGE
        @rate_package_id = @rate_package_id,
        @features_json = @features_json,
        @exposure = @exposure,
        @include_breakdown = @include_breakdown;
END;
GO
