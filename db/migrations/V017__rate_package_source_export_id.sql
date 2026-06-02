IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'source_export_id') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
    ADD source_export_id NVARCHAR(128) NULL;
GO

;WITH package_source AS (
    SELECT
        rate_package_id,
        MIN(export_id) AS source_export_id
    FROM pricing.MODEL_RUN
    WHERE export_id IS NOT NULL
      AND rate_package_id IS NOT NULL
    GROUP BY rate_package_id
)
UPDATE rp
SET source_export_id = src.source_export_id
FROM pricing.PRICING_RATE_PACKAGE AS rp
JOIN package_source AS src
  ON src.rate_package_id = rp.rate_package_id
WHERE rp.source_export_id IS NULL;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_PRICING_RATE_PACKAGE_MODEL_SOURCE_EXPORT'
      AND object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
)
CREATE UNIQUE INDEX UX_PRICING_RATE_PACKAGE_MODEL_SOURCE_EXPORT
ON pricing.PRICING_RATE_PACKAGE(model_id, source_export_id)
WHERE model_id IS NOT NULL
  AND source_export_id IS NOT NULL;
GO
