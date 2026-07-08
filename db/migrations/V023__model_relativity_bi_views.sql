IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
    EXEC('CREATE SCHEMA pricing');
GO

CREATE OR ALTER VIEW pricing.V_MODEL_RELATIVITY AS
SELECT
    m.model_id,
    m.model_name,
    m.model_label,
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
    b.term_id,
    t.sequence_no AS feature_sequence_no,
    b.feature_name,
    b.term_name,
    t.term_type,
    b.level_code AS term_level,
    b.level_code,
    b.sort_order AS level_sort_order,
    b.lower_bound,
    b.upper_bound,
    b.representative_value,
    b.multiplier AS relativity,
    b.log_coefficient,
    CAST(NULL AS DECIMAL(19,4)) AS exposure_weight,
    CAST(NULL AS BIGINT) AS record_count,
    CAST(0 AS BIT) AS is_default,
    CAST(0 AS BIT) AS is_reference,
    '1D_RATE_BAND' AS relativity_source
FROM pricing.PRICING_MODEL m
JOIN pricing.PRICING_RATE_PACKAGE rp
  ON rp.model_id = m.model_id
JOIN pricing.PRICING_COMPILED_1D_RATE_BAND b
  ON b.rate_package_id = rp.rate_package_id
JOIN pricing.PRICING_TERM t
  ON t.term_id = b.term_id

UNION ALL

SELECT
    m.model_id,
    m.model_name,
    m.model_label,
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
    c.term_id,
    c.sequence_no AS feature_sequence_no,
    COALESCE(f.feature_name, c.term_name) AS feature_name,
    c.term_name,
    c.term_type,
    c.cell_key_text AS term_level,
    CAST(NULL AS NVARCHAR(128)) AS level_code,
    CAST(NULL AS INT) AS level_sort_order,
    CAST(NULL AS FLOAT) AS lower_bound,
    CAST(NULL AS FLOAT) AS upper_bound,
    CAST(NULL AS FLOAT) AS representative_value,
    c.multiplier AS relativity,
    c.log_coefficient,
    c.exposure_weight,
    c.record_count,
    c.is_default,
    c.is_reference,
    'RATE_CELL' AS relativity_source
FROM pricing.PRICING_MODEL m
JOIN pricing.PRICING_RATE_PACKAGE rp
  ON rp.model_id = m.model_id
JOIN pricing.PRICING_COMPILED_RATE_CELL c
  ON c.rate_package_id = rp.rate_package_id
LEFT JOIN pricing.PRICING_TERM_FEATURE tf
  ON tf.term_id = c.term_id
 AND tf.position_no = 1
LEFT JOIN pricing.PRICING_FEATURE f
  ON f.feature_id = tf.feature_id
WHERE NOT EXISTS (
    SELECT 1
    FROM pricing.PRICING_COMPILED_1D_RATE_BAND b
    WHERE b.rate_package_id = c.rate_package_id
      AND b.term_id = c.term_id
);
GO

CREATE OR ALTER VIEW pricing.V_PUBLISHED_MODEL_RELATIVITY AS
SELECT
    model_id,
    model_name,
    model_label,
    target_name,
    model_type,
    rate_package_id,
    parent_rate_package_id,
    package_model_name,
    model_version,
    package_version,
    base_rate,
    effective_from_date,
    effective_to_date,
    package_status,
    term_id,
    feature_sequence_no,
    feature_name,
    term_name,
    term_type,
    term_level,
    level_code,
    level_sort_order,
    lower_bound,
    upper_bound,
    representative_value,
    relativity,
    log_coefficient,
    exposure_weight,
    record_count,
    is_default,
    is_reference,
    relativity_source
FROM pricing.V_MODEL_RELATIVITY
WHERE package_status = 'PUBLISHED';
GO
