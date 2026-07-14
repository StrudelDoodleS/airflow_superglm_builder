IF EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_PRICING_RATE_PACKAGE_CONTENT_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
)
BEGIN
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
        DROP CONSTRAINT CK_PRICING_RATE_PACKAGE_CONTENT_SHA256;
END;
GO

ALTER TABLE pricing.PRICING_RATE_PACKAGE WITH CHECK
    ADD CONSTRAINT CK_PRICING_RATE_PACKAGE_CONTENT_SHA256
    CHECK (
        staging_content_sha256 IS NULL
        OR (
            LEN(staging_content_sha256) = 64
            AND staging_content_sha256 COLLATE Latin1_General_BIN2
                NOT LIKE '%[^0-9a-f]%'
        )
    );
GO

IF EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_STG_RATING_EXPORT_CONTENT_SHA256'
      AND parent_object_id = OBJECT_ID('pricing_stg.STG_RATING_EXPORT')
)
BEGIN
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
        DROP CONSTRAINT CK_STG_RATING_EXPORT_CONTENT_SHA256;
END;
GO

ALTER TABLE pricing_stg.STG_RATING_EXPORT WITH CHECK
    ADD CONSTRAINT CK_STG_RATING_EXPORT_CONTENT_SHA256
    CHECK (
        staging_content_sha256 IS NULL
        OR (
            LEN(staging_content_sha256) = 64
            AND staging_content_sha256 COLLATE Latin1_General_BIN2
                NOT LIKE '%[^0-9a-f]%'
        )
    );
GO
