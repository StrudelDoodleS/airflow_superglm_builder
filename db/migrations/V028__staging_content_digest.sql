IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'staging_content_sha256') IS NULL
BEGIN
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
        ADD staging_content_sha256 CHAR(64) NULL;
END;
GO

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'staging_content_sha256') IS NULL
BEGIN
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
        ADD staging_content_sha256 CHAR(64) NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_PRICING_RATE_PACKAGE_CONTENT_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
)
BEGIN
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
        ADD CONSTRAINT CK_PRICING_RATE_PACKAGE_CONTENT_SHA256
        CHECK (
            staging_content_sha256 IS NULL
            OR (
                LEN(staging_content_sha256) = 64
                AND staging_content_sha256 NOT LIKE '%[^0-9a-f]%'
            )
        );
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_STG_RATING_EXPORT_CONTENT_SHA256'
      AND parent_object_id = OBJECT_ID('pricing_stg.STG_RATING_EXPORT')
)
BEGIN
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
        ADD CONSTRAINT CK_STG_RATING_EXPORT_CONTENT_SHA256
        CHECK (
            staging_content_sha256 IS NULL
            OR (
                LEN(staging_content_sha256) = 64
                AND staging_content_sha256 NOT LIKE '%[^0-9a-f]%'
            )
        );
END;
GO
