IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'effective_from_date') IS NOT NULL
BEGIN
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
        ALTER COLUMN effective_from_date DATE NULL;
END;
GO

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'effective_from_date') IS NOT NULL
BEGIN
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
        ALTER COLUMN effective_from_date DATE NULL;
END;
GO
