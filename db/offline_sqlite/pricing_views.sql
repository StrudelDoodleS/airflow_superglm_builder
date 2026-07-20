DROP VIEW IF EXISTS pricing.V_PUBLISHED_MODEL_RELATIVITY;
DROP VIEW IF EXISTS pricing.V_FINAL_MODEL_RELATIVITY;
DROP VIEW IF EXISTS pricing.V_MODEL_RELATIVITY;
DROP VIEW IF EXISTS pricing.V_MODEL_VALIDATION_SUMMARY;
DROP VIEW IF EXISTS pricing.V_MODEL_VALIDATION_SPLIT;

CREATE VIEW pricing.V_MODEL_RELATIVITY AS
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
    CAST(b.level_code AS TEXT) AS level_value,
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
FROM PRICING_MODEL AS m
JOIN PRICING_RATE_PACKAGE AS rp
  ON rp.model_id = m.model_id
LEFT JOIN MODEL_RUN AS mr
  ON mr.rate_package_id = rp.rate_package_id
JOIN PRICING_COMPILED_1D_RATE_BAND AS b
  ON b.rate_package_id = rp.rate_package_id
JOIN PRICING_TERM AS t
  ON t.term_id = b.term_id
JOIN PRICING_RATE_CELL_LEVEL AS rcl
  ON rcl.feature_level_id = b.feature_level_id
 AND rcl.position_no = 1
JOIN PRICING_RATE_CELL AS rc
  ON rc.cell_id = rcl.cell_id
 AND rc.term_id = b.term_id
 AND rc.is_deleted = 0
JOIN PRICING_COMPILED_RATE_CELL AS crc
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
        WHEN substr(c.cell_key_text, 1, length(c.term_name) + 1) = c.term_name || '='
        THEN substr(c.cell_key_text, length(c.term_name) + 2)
        ELSE c.cell_key_text
    END AS level_value,
    NULL AS level_sort_order,
    NULL AS lower_bound,
    NULL AS upper_bound,
    NULL AS representative_value,
    c.multiplier AS relativity,
    c.log_coefficient,
    c.exposure_weight,
    c.record_count,
    c.is_default,
    c.is_reference,
    'RATE_CELL' AS relativity_source,
    'PACKAGE_FINAL_MODEL' AS model_fit_scope
FROM PRICING_MODEL AS m
JOIN PRICING_RATE_PACKAGE AS rp
  ON rp.model_id = m.model_id
LEFT JOIN MODEL_RUN AS mr
  ON mr.rate_package_id = rp.rate_package_id
JOIN PRICING_COMPILED_RATE_CELL AS c
  ON c.rate_package_id = rp.rate_package_id
WHERE NOT EXISTS (
    SELECT 1
    FROM PRICING_COMPILED_1D_RATE_BAND AS b
    WHERE b.rate_package_id = c.rate_package_id
      AND b.term_id = c.term_id
);

CREATE VIEW pricing.V_FINAL_MODEL_RELATIVITY AS
SELECT *
FROM V_MODEL_RELATIVITY;

CREATE VIEW pricing.V_PUBLISHED_MODEL_RELATIVITY AS
SELECT *
FROM V_FINAL_MODEL_RELATIVITY
WHERE package_status = 'PUBLISHED';

CREATE VIEW pricing.V_MODEL_VALIDATION_SPLIT AS
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
FROM MODEL_RUN AS mr
JOIN PRICING_MODEL AS m
  ON m.model_id = mr.model_id
JOIN PRICING_RATE_PACKAGE AS rp
  ON rp.rate_package_id = mr.rate_package_id
JOIN DATASET_MANIFEST AS dm
  ON dm.manifest_id = mr.manifest_id
JOIN CV_SPLIT_SET AS ss
  ON ss.manifest_id = mr.manifest_id
 AND ss.split_set_id = mr.split_set_id
JOIN CV_FOLD AS fold
  ON fold.split_set_id = ss.split_set_id
JOIN CV_FOLD_METRIC AS fm
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

CREATE VIEW pricing.V_MODEL_VALIDATION_SUMMARY AS
SELECT
    model_run_id,
    parent_model_run_id,
    model_id,
    model_name,
    model_label,
    target_name,
    model_type,
    model_version,
    export_id,
    run_status,
    rate_package_id,
    parent_rate_package_id,
    package_version,
    package_status,
    manifest_id,
    dataset_name,
    source_system,
    data_as_of_date,
    dataset_row_count,
    split_set_id,
    split_mode,
    splitter_class,
    splitter_params_json,
    configured_fold_count,
    COUNT(*) AS recorded_split_count,
    SUM(n_test) AS total_validation_rows,
    AVG(deviance) AS mean_deviance,
    sqrt(MAX(AVG(deviance * deviance) - AVG(deviance) * AVG(deviance), 0.0))
        AS std_deviance,
    CAST(NULL AS REAL) AS pooled_deviance,
    AVG(nll) AS mean_nll,
    sqrt(MAX(AVG(nll * nll) - AVG(nll) * AVG(nll), 0.0)) AS std_nll,
    CAST(NULL AS REAL) AS pooled_nll,
    AVG(gini) AS mean_gini,
    sqrt(MAX(AVG(gini * gini) - AVG(gini) * AVG(gini), 0.0)) AS std_gini,
    CAST(NULL AS REAL) AS pooled_gini,
    CAST(SUM(n_test) AS REAL) / dataset_row_count AS oof_coverage
FROM V_MODEL_VALIDATION_SPLIT
GROUP BY
    model_run_id,
    parent_model_run_id,
    model_id,
    model_name,
    model_label,
    target_name,
    model_type,
    model_version,
    export_id,
    run_status,
    rate_package_id,
    parent_rate_package_id,
    package_version,
    package_status,
    manifest_id,
    dataset_name,
    source_system,
    data_as_of_date,
    dataset_row_count,
    split_set_id,
    split_mode,
    splitter_class,
    splitter_params_json,
    configured_fold_count;
