IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'source_file') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
    ADD source_file NVARCHAR(1024) NULL;
GO

UPDATE rp
SET source_file = src.source_file
FROM pricing.PRICING_RATE_PACKAGE AS rp
JOIN pricing_stg.STG_RATING_EXPORT AS src
  ON src.export_id = rp.source_export_id
WHERE rp.source_file IS NULL
  AND rp.source_export_id IS NOT NULL
  AND src.source_file IS NOT NULL
  AND rp.package_status = 'DRAFT';
GO
