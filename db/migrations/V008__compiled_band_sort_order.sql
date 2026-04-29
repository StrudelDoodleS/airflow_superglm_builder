IF COL_LENGTH('pricing.PRICING_COMPILED_1D_RATE_BAND', 'sort_order') IS NULL
ALTER TABLE pricing.PRICING_COMPILED_1D_RATE_BAND
ADD sort_order INT NOT NULL
    CONSTRAINT DF_COMPILED_1D_RATE_BAND_SORT_ORDER DEFAULT (0);
GO

UPDATE b
SET sort_order = COALESCE(fl.order_index, 0)
FROM pricing.PRICING_COMPILED_1D_RATE_BAND b
JOIN pricing.PRICING_FEATURE_LEVEL fl
  ON fl.feature_level_id = b.feature_level_id;
GO

IF EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = 'PK_COMPILED_1D_RATE_BAND'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_COMPILED_1D_RATE_BAND')
)
ALTER TABLE pricing.PRICING_COMPILED_1D_RATE_BAND
DROP CONSTRAINT PK_COMPILED_1D_RATE_BAND;
GO

ALTER TABLE pricing.PRICING_COMPILED_1D_RATE_BAND
ADD CONSTRAINT PK_COMPILED_1D_RATE_BAND
PRIMARY KEY CLUSTERED (rate_package_id, term_id, sort_order, feature_level_id);
GO

CREATE OR ALTER VIEW pricing.V_CURRENT_1D_RATE_BAND AS
SELECT
    cur.model_id,
    cur.model_key,
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
