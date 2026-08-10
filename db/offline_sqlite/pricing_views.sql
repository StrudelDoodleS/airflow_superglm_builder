DROP VIEW IF EXISTS pricing.V_CURRENT_DEPLOYED_RELATIVITY;
DROP VIEW IF EXISTS pricing.V_PUBLISHED_MODEL_RELATIVITY;
DROP VIEW IF EXISTS pricing.V_MODEL_CANDIDATE_RELATIVITY;
DROP VIEW IF EXISTS pricing.V_FINAL_MODEL_RELATIVITY;
DROP VIEW IF EXISTS pricing.V_MODEL_RELATIVITY;
DROP VIEW IF EXISTS pricing.V_MODEL_VALIDATION_SUMMARY;
DROP VIEW IF EXISTS pricing.V_MODEL_VALIDATION_SPLIT;
DROP VIEW IF EXISTS pricing.V_MODEL_LINEAGE_REDUNDANCY_CHECK;
-- Upgrade cleanup for short-lived audit views whose duplicate states are
-- already made impossible by filtered unique indexes.
DROP VIEW IF EXISTS pricing.V_DATASET_MANIFEST_REDUNDANCY_CHECK;
DROP VIEW IF EXISTS pricing.V_MODEL_EQUIVALENCE_REDUNDANCY_CHECK;

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
SELECT
    relativity.model_id,
    relativity.model_name,
    relativity.model_label,
    relativity.target_name,
    relativity.model_type,
    model_run.model_kind,
    model_run.model_equivalence_sha256,
    relativity.model_run_id,
    relativity.parent_model_run_id,
    relativity.run_status,
    model_run.export_id,
    relativity.rate_package_id,
    relativity.parent_rate_package_id,
    relativity.package_model_name,
    relativity.model_version,
    relativity.package_version,
    relativity.base_rate,
    relativity.effective_from_date,
    relativity.effective_to_date,
    relativity.package_status,
    manifest.manifest_id,
    manifest.manifest_signature_sha256,
    manifest.dataset_name,
    manifest.source_system,
    manifest.data_as_of_date,
    manifest.data_as_of_column,
    manifest.row_count AS dataset_row_count,
    manifest.pk_columns_json,
    manifest.target_column AS dataset_target_column,
    manifest.weight_column,
    manifest.exposure_column,
    manifest.offset_column,
    manifest.offset_source_column,
    manifest.offset_label,
    manifest.export_weight_column,
    manifest.model_frame_sha256,
    manifest.frame_hash_metadata_json,
    manifest.created_ts AS manifest_created_ts,
    manifest.created_by AS manifest_created_by,
    model_run.split_set_id AS validation_split_set_id,
    relativity.term_id,
    relativity.term_sequence_no,
    relativity.term_name,
    relativity.term_type,
    relativity.level_value,
    relativity.level_sort_order,
    relativity.lower_bound,
    relativity.upper_bound,
    relativity.representative_value,
    relativity.relativity,
    relativity.log_coefficient,
    relativity.exposure_weight,
    relativity.record_count,
    relativity.is_default,
    relativity.is_reference,
    relativity.relativity_source,
    relativity.model_fit_scope
FROM V_MODEL_RELATIVITY AS relativity
LEFT JOIN MODEL_RUN AS model_run
  ON model_run.model_run_id = relativity.model_run_id
LEFT JOIN DATASET_MANIFEST AS manifest
  ON manifest.manifest_id = model_run.manifest_id;

CREATE VIEW pricing.V_MODEL_CANDIDATE_RELATIVITY AS
SELECT *
FROM V_FINAL_MODEL_RELATIVITY
WHERE package_status = 'PUBLISHED';

-- Compatibility alias retained for existing notebook and reporting queries.
CREATE VIEW pricing.V_PUBLISHED_MODEL_RELATIVITY AS
SELECT *
FROM V_MODEL_CANDIDATE_RELATIVITY;

CREATE VIEW pricing.V_CURRENT_DEPLOYED_RELATIVITY AS
SELECT
    deployment.deployment_id,
    deployment.deployment_slot,
    deployment.effective_from_ts AS deployment_effective_from_ts,
    deployment.effective_to_ts AS deployment_effective_to_ts,
    deployment.deployed_by,
    deployment.deployment_note,
    deployment.created_ts AS deployment_created_ts,
    relativity.*
FROM PRICING_MODEL_DEPLOYMENT AS deployment
JOIN V_FINAL_MODEL_RELATIVITY AS relativity
  ON relativity.model_id = deployment.model_id
 AND relativity.rate_package_id = deployment.rate_package_id
WHERE deployment.effective_to_ts IS NULL
  AND relativity.package_status = 'PUBLISHED';

CREATE VIEW pricing.V_MODEL_VALIDATION_SPLIT AS
SELECT
    mr.model_run_id,
    mr.parent_model_run_id,
    m.model_id,
    m.model_name,
    m.model_label,
    m.target_name,
    m.model_type,
    mr.model_kind,
    mr.model_equivalence_sha256,
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
-- SQLite cannot persist a pricing view that references the separately attached
-- mlops database, and this view must remain usable when pricing.sqlite is opened
-- directly. Local publication separately writes and verifies the equivalent
-- training/validation MODEL_RUN_SPLIT_SET link. SQL Server uses that normalized
-- role-filtered link in V035.
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
    mr.model_kind,
    mr.model_equivalence_sha256,
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
    model_kind,
    model_equivalence_sha256,
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
    -- Pooled scores live in the separately attached mlops database. SQLite
    -- cannot expose them from a persistent pricing.sqlite view without copying
    -- audit data, so local callers inspect candidate.metrics or mlops directly.
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
    model_kind,
    model_equivalence_sha256,
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

CREATE VIEW pricing.V_MODEL_LINEAGE_REDUNDANCY_CHECK AS
SELECT
    model_run.model_id,
    model_run.model_run_id,
    package.rate_package_id,
    package.manifest_id AS package_manifest_id,
    model_run.manifest_id AS run_manifest_id,
    CAST(NULL AS TEXT) AS linked_training_manifest_id,
    CAST(NULL AS INTEGER) AS training_manifest_link_count,
    package.split_set_id AS package_split_set_id,
    model_run.split_set_id AS run_split_set_id,
    CAST(NULL AS TEXT) AS linked_validation_split_set_id,
    CAST(NULL AS INTEGER) AS validation_split_link_count,
    CASE
        WHEN package.manifest_id IS NOT NULL
         AND package.manifest_id <> model_run.manifest_id
        THEN 'PACKAGE_RUN_MANIFEST_MISMATCH'
        WHEN package.split_set_id IS NOT NULL
         AND package.split_set_id <> model_run.split_set_id
        THEN 'PACKAGE_RUN_SPLIT_MISMATCH'
        ELSE 'OK'
    END AS redundancy_status
FROM MODEL_RUN AS model_run
JOIN PRICING_RATE_PACKAGE AS package
  ON package.rate_package_id = model_run.rate_package_id;
