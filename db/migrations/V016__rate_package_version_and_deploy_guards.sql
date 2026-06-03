IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_PRICING_RATE_PACKAGE_MODEL_VERSION'
      AND object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
)
CREATE UNIQUE INDEX UX_PRICING_RATE_PACKAGE_MODEL_VERSION
ON pricing.PRICING_RATE_PACKAGE(model_id, package_version)
WHERE model_id IS NOT NULL;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_PRICING_RATE_PACKAGE_MODEL_PACKAGE_ID'
      AND object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
)
CREATE UNIQUE INDEX UX_PRICING_RATE_PACKAGE_MODEL_PACKAGE_ID
ON pricing.PRICING_RATE_PACKAGE(model_id, rate_package_id);
GO

IF EXISTS (
    SELECT 1
    FROM pricing.PRICING_MODEL_DEPLOYMENT d
    LEFT JOIN pricing.PRICING_RATE_PACKAGE rp
      ON rp.rate_package_id = d.rate_package_id
     AND rp.model_id = d.model_id
    WHERE rp.rate_package_id IS NULL
       OR rp.package_status <> 'PUBLISHED'
)
BEGIN
    RAISERROR('rate package deployments must reference PUBLISHED packages for the same model_id.', 16, 1);
    RETURN;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = 'FK_MODEL_DEPLOYMENT_MODEL_PACKAGE'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_MODEL_DEPLOYMENT')
)
ALTER TABLE pricing.PRICING_MODEL_DEPLOYMENT WITH CHECK
ADD CONSTRAINT FK_MODEL_DEPLOYMENT_MODEL_PACKAGE
    FOREIGN KEY (model_id, rate_package_id)
    REFERENCES pricing.PRICING_RATE_PACKAGE(model_id, rate_package_id);
GO

CREATE OR ALTER TRIGGER pricing.TR_PRICING_MODEL_DEPLOYMENT_PACKAGE_GUARD
ON pricing.PRICING_MODEL_DEPLOYMENT
AFTER INSERT, UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM inserted d
        LEFT JOIN pricing.PRICING_RATE_PACKAGE rp
          ON rp.rate_package_id = d.rate_package_id
         AND rp.model_id = d.model_id
        WHERE rp.rate_package_id IS NULL
           OR rp.package_status <> 'PUBLISHED'
    )
    BEGIN
        RAISERROR('rate package deployments must reference PUBLISHED packages for the same model_id.', 16, 1);
        ROLLBACK TRANSACTION;
        RETURN;
    END;
END;
GO
