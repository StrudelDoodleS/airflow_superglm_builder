CREATE OR ALTER VIEW pricing.V_MODEL_RELATIVITY
AS
SELECT
    m.model_id,
    m.model_name,
    m.model_label,
    m.target_name,
    m.model_type,
    mr.model_run_id,
    mr.parent_model_run_id,
    mr.run_status,
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
    t.sequence_no AS term_sequence_no,
    b.term_name,
    t.term_type,
    CAST(b.level_code AS NVARCHAR(900)) AS level_value,
    b.sort_order AS level_sort_order,
    b.lower_bound,
    b.upper_bound,
    b.representative_value,
    b.multiplier AS relativity,
    b.log_coefficient,
    crc.exposure_weight,
    crc.record_count,
    crc.is_default,
    crc.is_reference,
    '1D_RATE_BAND' AS relativity_source,
    'PACKAGE_FINAL_MODEL' AS model_fit_scope
FROM pricing.PRICING_MODEL AS m
JOIN pricing.PRICING_RATE_PACKAGE AS rp
  ON rp.model_id = m.model_id
LEFT JOIN pricing.MODEL_RUN AS mr
  ON mr.rate_package_id = rp.rate_package_id
JOIN pricing.PRICING_COMPILED_1D_RATE_BAND AS b
  ON b.rate_package_id = rp.rate_package_id
JOIN pricing.PRICING_TERM AS t
  ON t.term_id = b.term_id
JOIN pricing.PRICING_RATE_CELL_LEVEL AS rcl
  ON rcl.feature_level_id = b.feature_level_id
 AND rcl.position_no = 1
JOIN pricing.PRICING_RATE_CELL AS rc
  ON rc.cell_id = rcl.cell_id
 AND rc.term_id = b.term_id
 AND rc.is_deleted = 0
JOIN pricing.PRICING_COMPILED_RATE_CELL AS crc
  ON crc.rate_package_id = b.rate_package_id
 AND crc.term_id = b.term_id
 AND crc.cell_key_digest = rc.cell_key_digest

UNION ALL

SELECT
    m.model_id,
    m.model_name,
    m.model_label,
    m.target_name,
    m.model_type,
    mr.model_run_id,
    mr.parent_model_run_id,
    mr.run_status,
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
    c.sequence_no AS term_sequence_no,
    c.term_name,
    c.term_type,
    CASE
        WHEN LEFT(c.cell_key_text, LEN(c.term_name) + 1) = CONCAT(c.term_name, '=')
        THEN SUBSTRING(c.cell_key_text, LEN(c.term_name) + 2, 900)
        ELSE c.cell_key_text
    END AS level_value,
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
    'RATE_CELL' AS relativity_source,
    'PACKAGE_FINAL_MODEL' AS model_fit_scope
FROM pricing.PRICING_MODEL AS m
JOIN pricing.PRICING_RATE_PACKAGE AS rp
  ON rp.model_id = m.model_id
LEFT JOIN pricing.MODEL_RUN AS mr
  ON mr.rate_package_id = rp.rate_package_id
JOIN pricing.PRICING_COMPILED_RATE_CELL AS c
  ON c.rate_package_id = rp.rate_package_id
WHERE NOT EXISTS (
    SELECT 1
    FROM pricing.PRICING_COMPILED_1D_RATE_BAND AS b
    WHERE b.rate_package_id = c.rate_package_id
      AND b.term_id = c.term_id
);
GO

CREATE OR ALTER VIEW pricing.V_FINAL_MODEL_RELATIVITY
AS
SELECT
    model_id,
    model_name,
    model_label,
    target_name,
    model_type,
    model_run_id,
    parent_model_run_id,
    run_status,
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
    term_sequence_no,
    term_name,
    term_type,
    level_value,
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
    relativity_source,
    model_fit_scope
FROM pricing.V_MODEL_RELATIVITY;
GO

CREATE OR ALTER VIEW pricing.V_PUBLISHED_MODEL_RELATIVITY
AS
SELECT
    model_id,
    model_name,
    model_label,
    target_name,
    model_type,
    model_run_id,
    parent_model_run_id,
    run_status,
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
    term_sequence_no,
    term_name,
    term_type,
    level_value,
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
    relativity_source,
    model_fit_scope
FROM pricing.V_FINAL_MODEL_RELATIVITY
WHERE package_status = 'PUBLISHED';
GO

CREATE OR ALTER VIEW pricing.V_MODEL_VALIDATION_SPLIT
AS
SELECT
    mr.model_run_id,
    mr.parent_model_run_id,
    m.model_id,
    m.model_name,
    m.model_label,
    m.target_name,
    m.model_type,
    mr.model_version,
    mr.export_id,
    mr.run_status,
    rp.rate_package_id,
    rp.parent_rate_package_id,
    rp.package_version,
    rp.package_status,
    dm.manifest_id,
    dm.dataset_name,
    dm.source_system,
    dm.data_as_of_date,
    dm.row_count AS dataset_row_count,
    ss.split_set_id,
    ss.split_mode,
    ss.splitter_class,
    ss.splitter_params_json,
    ss.fold_count AS configured_fold_count,
    fold.fold_no AS validation_split_no,
    fold.n_train,
    fold.n_test,
    MAX(CASE WHEN fm.metric_name = 'deviance' THEN fm.metric_value END) AS deviance,
    MAX(CASE WHEN fm.metric_name = 'nll' THEN fm.metric_value END) AS nll,
    MAX(CASE WHEN fm.metric_name = 'gini' THEN fm.metric_value END) AS gini
FROM pricing.MODEL_RUN AS mr
JOIN pricing.PRICING_MODEL AS m
  ON m.model_id = mr.model_id
JOIN pricing.PRICING_RATE_PACKAGE AS rp
  ON rp.rate_package_id = mr.rate_package_id
JOIN mlops.MODEL_RUN_SPLIT_SET AS run_split
  ON run_split.model_run_id = mr.model_run_id
 AND run_split.dataset_role = 'training'
 AND run_split.split_role = 'validation'
JOIN pricing.DATASET_MANIFEST AS dm
  ON dm.manifest_id = run_split.manifest_id
JOIN pricing.CV_SPLIT_SET AS ss
  ON ss.manifest_id = run_split.manifest_id
 AND ss.split_set_id = run_split.split_set_id
JOIN pricing.CV_FOLD AS fold
  ON fold.split_set_id = ss.split_set_id
JOIN pricing.CV_FOLD_METRIC AS fm
  ON fm.model_run_id = mr.model_run_id
 AND fm.split_set_id = fold.split_set_id
 AND fm.fold_no = fold.fold_no
WHERE mr.run_status = 'SUCCESS'
GROUP BY
    mr.model_run_id,
    mr.parent_model_run_id,
    m.model_id,
    m.model_name,
    m.model_label,
    m.target_name,
    m.model_type,
    mr.model_version,
    mr.export_id,
    mr.run_status,
    rp.rate_package_id,
    rp.parent_rate_package_id,
    rp.package_version,
    rp.package_status,
    dm.manifest_id,
    dm.dataset_name,
    dm.source_system,
    dm.data_as_of_date,
    dm.row_count,
    ss.split_set_id,
    ss.split_mode,
    ss.splitter_class,
    ss.splitter_params_json,
    ss.fold_count,
    fold.fold_no,
    fold.n_train,
    fold.n_test;
GO

CREATE OR ALTER VIEW pricing.V_MODEL_VALIDATION_SUMMARY
AS
WITH run_metrics AS (
    SELECT
        metric.model_run_id,
        MAX(
            CASE WHEN metric.metric_name = 'cv_mean_deviance'
                THEN metric.metric_value END
        ) AS mean_deviance,
        MAX(
            CASE WHEN metric.metric_name = 'cv_std_deviance'
                THEN metric.metric_value END
        ) AS std_deviance,
        MAX(
            CASE WHEN metric.metric_name = 'cv_pooled_deviance'
                THEN metric.metric_value END
        ) AS pooled_deviance,
        MAX(
            CASE WHEN metric.metric_name = 'cv_mean_nll'
                THEN metric.metric_value END
        ) AS mean_nll,
        MAX(
            CASE WHEN metric.metric_name = 'cv_std_nll'
                THEN metric.metric_value END
        ) AS std_nll,
        MAX(
            CASE WHEN metric.metric_name = 'cv_pooled_nll'
                THEN metric.metric_value END
        ) AS pooled_nll,
        MAX(
            CASE WHEN metric.metric_name = 'cv_mean_gini'
                THEN metric.metric_value END
        ) AS mean_gini,
        MAX(
            CASE WHEN metric.metric_name = 'cv_std_gini'
                THEN metric.metric_value END
        ) AS std_gini,
        MAX(
            CASE WHEN metric.metric_name = 'cv_pooled_gini'
                THEN metric.metric_value END
        ) AS pooled_gini,
        MAX(
            CASE WHEN metric.metric_name = 'cv_oof_coverage'
                THEN metric.metric_value END
        ) AS oof_coverage
    FROM mlops.MODEL_RUN_METRIC AS metric
    GROUP BY metric.model_run_id
)
SELECT
    validation.model_run_id,
    validation.parent_model_run_id,
    validation.model_id,
    validation.model_name,
    validation.model_label,
    validation.target_name,
    validation.model_type,
    validation.model_version,
    validation.export_id,
    validation.run_status,
    validation.rate_package_id,
    validation.parent_rate_package_id,
    validation.package_version,
    validation.package_status,
    validation.manifest_id,
    validation.dataset_name,
    validation.source_system,
    validation.data_as_of_date,
    validation.dataset_row_count,
    validation.split_set_id,
    validation.split_mode,
    validation.splitter_class,
    validation.splitter_params_json,
    validation.configured_fold_count,
    COUNT_BIG(*) AS recorded_split_count,
    SUM(validation.n_test) AS total_validation_rows,
    MAX(metrics.mean_deviance) AS mean_deviance,
    MAX(metrics.std_deviance) AS std_deviance,
    MAX(metrics.pooled_deviance) AS pooled_deviance,
    MAX(metrics.mean_nll) AS mean_nll,
    MAX(metrics.std_nll) AS std_nll,
    MAX(metrics.pooled_nll) AS pooled_nll,
    MAX(metrics.mean_gini) AS mean_gini,
    MAX(metrics.std_gini) AS std_gini,
    MAX(metrics.pooled_gini) AS pooled_gini,
    MAX(metrics.oof_coverage) AS oof_coverage
FROM pricing.V_MODEL_VALIDATION_SPLIT AS validation
LEFT JOIN run_metrics AS metrics
  ON metrics.model_run_id = validation.model_run_id
GROUP BY
    validation.model_run_id,
    validation.parent_model_run_id,
    validation.model_id,
    validation.model_name,
    validation.model_label,
    validation.target_name,
    validation.model_type,
    validation.model_version,
    validation.export_id,
    validation.run_status,
    validation.rate_package_id,
    validation.parent_rate_package_id,
    validation.package_version,
    validation.package_status,
    validation.manifest_id,
    validation.dataset_name,
    validation.source_system,
    validation.data_as_of_date,
    validation.dataset_row_count,
    validation.split_set_id,
    validation.split_mode,
    validation.splitter_class,
    validation.splitter_params_json,
    validation.configured_fold_count;
GO
