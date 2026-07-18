BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS pricing.CV_SPLIT_CURVE_POINT (
    model_run_id TEXT NOT NULL,
    split_set_id TEXT NOT NULL,
    split_no INTEGER NOT NULL,
    term_name TEXT NOT NULL,
    point_no INTEGER NOT NULL,
    point_kind TEXT NOT NULL,
    x_numeric REAL,
    level_text TEXT,
    eta_contribution REAL NOT NULL,
    relativity REAL,
    support_value REAL,
    reference_value REAL,
    reference_level TEXT,
    CONSTRAINT PK_CV_SPLIT_CURVE_POINT
        PRIMARY KEY (model_run_id, split_set_id, split_no, term_name, point_no),
    CONSTRAINT FK_CV_SPLIT_CURVE_POINT_RUN
        FOREIGN KEY (model_run_id) REFERENCES MODEL_RUN(model_run_id),
    CONSTRAINT FK_CV_SPLIT_CURVE_POINT_FOLD
        FOREIGN KEY (split_set_id, split_no) REFERENCES CV_FOLD(split_set_id, fold_no),
    CONSTRAINT CK_CV_SPLIT_CURVE_POINT_POSITIVE_KEYS
        CHECK (split_no > 0 AND point_no > 0),
    CONSTRAINT CK_CV_SPLIT_CURVE_POINT_TERM_NAME
        CHECK (length(trim(term_name)) > 0),
    CONSTRAINT CK_CV_SPLIT_CURVE_POINT_KIND
        CHECK (point_kind IN ('NUMERIC', 'LEVEL')),
    CONSTRAINT CK_CV_SPLIT_CURVE_POINT_DOMAIN CHECK (
        (
            point_kind = 'NUMERIC'
            AND x_numeric IS NOT NULL
            AND level_text IS NULL
            AND reference_value IS NOT NULL
            AND reference_level IS NULL
        )
        OR (
            point_kind = 'LEVEL'
            AND x_numeric IS NULL
            AND level_text IS NOT NULL
            AND length(trim(level_text)) > 0
            AND reference_value IS NULL
            AND reference_level IS NOT NULL
            AND length(trim(reference_level)) > 0
        )
    ),
    CONSTRAINT CK_CV_SPLIT_CURVE_POINT_SUPPORT
        CHECK (support_value IS NULL OR support_value >= 0),
    CONSTRAINT CK_CV_SPLIT_CURVE_POINT_RELATIVITY
        CHECK (relativity IS NULL OR relativity >= 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS pricing.UX_PRICING_RATE_PACKAGE_MODEL_BUILD_FINGERPRINT
ON PRICING_RATE_PACKAGE(model_id, build_fingerprint_sha256)
WHERE parent_rate_package_id IS NULL
  AND build_fingerprint_sha256 IS NOT NULL;

CREATE INDEX IF NOT EXISTS pricing.IX_MODEL_RUN_VALIDATION_SOURCE
ON MODEL_RUN(validation_source_model_run_id)
WHERE validation_source_model_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS pricing.IX_CV_SPLIT_CURVE_POINT_SPLIT
ON CV_SPLIT_CURVE_POINT(split_set_id, split_no);

DROP VIEW IF EXISTS pricing.V_PUBLISHED_MODEL_RELATIVITY;
DROP VIEW IF EXISTS pricing.V_MODEL_RELATIVITY;
DROP VIEW IF EXISTS pricing.V_CURRENT_DATASET_CV_FOLD;
DROP VIEW IF EXISTS pricing.V_FINAL_MODEL_RELATIVITY;
DROP VIEW IF EXISTS pricing.V_MODEL_VALIDATION_SPLIT;
DROP VIEW IF EXISTS pricing.V_MODEL_VALIDATION_SUMMARY;
DROP VIEW IF EXISTS pricing.V_MODEL_VALIDATION_SPLIT_RELATIVITY;
DROP VIEW IF EXISTS pricing.V_CURRENT_DATASET_VALIDATION_SPLIT;

CREATE VIEW pricing.V_FINAL_MODEL_RELATIVITY AS
SELECT
    model.model_id,
    model.model_name,
    model.model_label,
    model.target_name,
    model.model_type,
    rp.rate_package_id,
    rp.parent_rate_package_id,
    rp.model_version,
    rp.package_version,
    parent_package.package_version AS parent_package_version,
    COALESCE(json_extract(rp.revision_metadata_json, '$.kind'), CASE WHEN rp.parent_rate_package_id IS NULL THEN 'ORIGINAL' ELSE 'EDITED' END) AS package_kind,
    rp.package_status,
    rp.base_rate,
    rp.effective_from_date,
    rp.effective_to_date,
    json_extract(rp.package_metadata_json, '$.model.family') AS family,
    json_extract(rp.package_metadata_json, '$.model.family_params') AS family_params_json,
    json_extract(rp.package_metadata_json, '$.model.link') AS link,
    json_extract(rp.revision_metadata_json, '$.reason') AS edit_reason,
    'PACKAGE_FINAL_MODEL' AS model_fit_scope,
    band.term_id,
    term.sequence_no AS term_sequence_no,
    band.term_name,
    term.term_type,
    band.level_code AS term_level,
    band.sort_order AS level_sort_order,
    band.lower_bound,
    band.upper_bound,
    band.representative_value,
    band.log_coefficient AS eta_contribution,
    band.multiplier AS relativity,
    compiled_cell.exposure_weight,
    compiled_cell.record_count,
    compiled_cell.is_default,
    compiled_cell.is_reference,
    '1D_RATE_BAND' AS relativity_source
FROM PRICING_MODEL AS model
JOIN PRICING_RATE_PACKAGE AS rp
  ON rp.model_id = model.model_id
LEFT JOIN PRICING_RATE_PACKAGE AS parent_package
  ON parent_package.rate_package_id = rp.parent_rate_package_id
JOIN PRICING_COMPILED_1D_RATE_BAND AS band
  ON band.rate_package_id = rp.rate_package_id
JOIN PRICING_TERM AS term
  ON term.term_id = band.term_id
JOIN PRICING_RATE_CELL_LEVEL AS cell_level
  ON cell_level.feature_level_id = band.feature_level_id
 AND cell_level.position_no = 1
JOIN PRICING_RATE_CELL AS rate_cell
  ON rate_cell.cell_id = cell_level.cell_id
 AND rate_cell.term_id = band.term_id
 AND rate_cell.is_deleted = 0
JOIN PRICING_COMPILED_RATE_CELL AS compiled_cell
  ON compiled_cell.rate_package_id = band.rate_package_id
 AND compiled_cell.term_id = band.term_id
 AND compiled_cell.cell_key_digest = rate_cell.cell_key_digest

UNION ALL

SELECT
    model.model_id,
    model.model_name,
    model.model_label,
    model.target_name,
    model.model_type,
    rp.rate_package_id,
    rp.parent_rate_package_id,
    rp.model_version,
    rp.package_version,
    parent_package.package_version AS parent_package_version,
    COALESCE(json_extract(rp.revision_metadata_json, '$.kind'), CASE WHEN rp.parent_rate_package_id IS NULL THEN 'ORIGINAL' ELSE 'EDITED' END) AS package_kind,
    rp.package_status,
    rp.base_rate,
    rp.effective_from_date,
    rp.effective_to_date,
    json_extract(rp.package_metadata_json, '$.model.family') AS family,
    json_extract(rp.package_metadata_json, '$.model.family_params') AS family_params_json,
    json_extract(rp.package_metadata_json, '$.model.link') AS link,
    json_extract(rp.revision_metadata_json, '$.reason') AS edit_reason,
    'PACKAGE_FINAL_MODEL' AS model_fit_scope,
    cell.term_id,
    cell.sequence_no AS term_sequence_no,
    cell.term_name,
    cell.term_type,
    CASE WHEN substr(cell.cell_key_text, 1, length(cell.term_name) + 1) = cell.term_name || '=' THEN substr(cell.cell_key_text, length(cell.term_name) + 2) ELSE cell.cell_key_text END AS term_level,
    CAST(NULL AS INTEGER) AS level_sort_order,
    CAST(NULL AS REAL) AS lower_bound,
    CAST(NULL AS REAL) AS upper_bound,
    CAST(NULL AS REAL) AS representative_value,
    cell.log_coefficient AS eta_contribution,
    cell.multiplier AS relativity,
    cell.exposure_weight,
    cell.record_count,
    cell.is_default,
    cell.is_reference,
    'RATE_CELL' AS relativity_source
FROM PRICING_MODEL AS model
JOIN PRICING_RATE_PACKAGE AS rp
  ON rp.model_id = model.model_id
LEFT JOIN PRICING_RATE_PACKAGE AS parent_package
  ON parent_package.rate_package_id = rp.parent_rate_package_id
JOIN PRICING_COMPILED_RATE_CELL AS cell
  ON cell.rate_package_id = rp.rate_package_id
WHERE NOT EXISTS (
    SELECT 1
    FROM PRICING_COMPILED_1D_RATE_BAND AS band
    WHERE band.rate_package_id = cell.rate_package_id
      AND band.term_id = cell.term_id
);

CREATE VIEW pricing.V_MODEL_VALIDATION_SPLIT AS
SELECT
    model.model_id,
    model.model_name,
    model.model_label,
    model.target_name,
    model.model_type,
    final_package.rate_package_id,
    final_package.parent_rate_package_id,
    final_package.model_version,
    final_package.package_version,
    parent_package.package_version AS parent_package_version,
    COALESCE(json_extract(final_package.revision_metadata_json, '$.kind'), CASE WHEN final_package.parent_rate_package_id IS NULL THEN 'ORIGINAL' ELSE 'EDITED' END) AS package_kind,
    final_package.package_status,
    json_extract(final_package.revision_metadata_json, '$.reason') AS edit_reason,
    final_run.model_run_id,
    final_run.parent_model_run_id,
    source_run.model_run_id AS validation_source_model_run_id,
    source_package.rate_package_id AS validation_source_rate_package_id,
    source_package.package_version AS validation_source_package_version,
    CASE WHEN source_run.model_run_id = final_run.model_run_id THEN 'DIRECT' ELSE 'INHERITED_FROM_PARENT' END AS validation_evidence,
    source_package.build_fingerprint_sha256,
    source_run.builder_source_sha256,
    source_run.materialized_split_sha256,
    source_run.runtime_sha256,
    source_run.candidate_superglm_sha256,
    source_run.model_source_sha256,
    source_run.candidate_python_version,
    source_run.candidate_superglm_version,
    source_run.candidate_superglm_git_sha,
    manifest.manifest_id,
    manifest.dataset_name,
    manifest.source_system,
    manifest.data_as_of_date,
    manifest.data_as_of_column,
    manifest.row_count AS dataset_row_count,
    manifest.model_frame_sha256,
    manifest.frame_hash_metadata_json,
    split_set.split_set_id,
    split_set.split_mode,
    COALESCE(CASE WHEN json_valid(split_set.splitter_params_json) THEN json_extract(split_set.splitter_params_json, '$.method') END, CASE WHEN split_set.splitter_class LIKE '%KFold' THEN 'kfold' WHEN split_set.splitter_class LIKE '%train_test_split' THEN 'train_test_split' ELSE 'custom' END) AS split_method,
    split_set.splitter_class,
    split_set.splitter_params_json,
    split_set.row_order_sha256,
    split_set.row_count AS split_row_count,
    split_set.fold_count,
    split_set.artifact_uri,
    split_set.artifact_sha256,
    split_set.runtime_metadata_json,
    fold.fold_no AS validation_split_no,
    fold.n_train,
    fold.n_test AS n_validation,
    fold_metrics.deviance,
    fold_metrics.nll,
    fold_metrics.gini,
    json_extract(final_package.package_metadata_json, '$.model.family') AS family,
    json_extract(final_package.package_metadata_json, '$.model.family_params') AS family_params_json,
    json_extract(final_package.package_metadata_json, '$.model.link') AS link,
    source_run.validation_curve_status,
    source_run.validation_curve_reason
FROM PRICING_MODEL AS model
JOIN PRICING_RATE_PACKAGE AS final_package
  ON final_package.model_id = model.model_id
JOIN MODEL_RUN AS final_run
  ON final_run.rate_package_id = final_package.rate_package_id
JOIN MODEL_RUN AS source_run
  ON source_run.model_run_id = COALESCE(final_run.validation_source_model_run_id, final_run.model_run_id)
JOIN PRICING_RATE_PACKAGE AS source_package
  ON source_package.rate_package_id = source_run.rate_package_id
LEFT JOIN PRICING_RATE_PACKAGE AS parent_package
  ON parent_package.rate_package_id = final_package.parent_rate_package_id
JOIN DATASET_MANIFEST AS manifest
  ON manifest.manifest_id = source_run.manifest_id
JOIN CV_SPLIT_SET AS split_set
  ON split_set.split_set_id = source_run.split_set_id
 AND split_set.manifest_id = source_run.manifest_id
JOIN CV_FOLD AS fold
  ON fold.split_set_id = split_set.split_set_id
LEFT JOIN (
    SELECT
        fold_metric.model_run_id,
        fold_metric.split_set_id,
        fold_metric.fold_no,
        MAX(CASE WHEN lower(fold_metric.metric_name) = 'deviance' THEN fold_metric.metric_value END) AS deviance,
        MAX(CASE WHEN lower(fold_metric.metric_name) = 'nll' THEN fold_metric.metric_value END) AS nll,
        MAX(CASE WHEN lower(fold_metric.metric_name) = 'gini' THEN fold_metric.metric_value END) AS gini
    FROM CV_FOLD_METRIC AS fold_metric
    GROUP BY
        fold_metric.model_run_id,
        fold_metric.split_set_id,
        fold_metric.fold_no
) AS fold_metrics
  ON fold_metrics.model_run_id = source_run.model_run_id
 AND fold_metrics.split_set_id = split_set.split_set_id
 AND fold_metrics.fold_no = fold.fold_no;

CREATE VIEW pricing.V_MODEL_VALIDATION_SUMMARY AS
WITH metric_anchor AS (
    SELECT
        rate_package_id,
        model_run_id,
        MIN(deviance) AS deviance,
        MIN(nll) AS nll,
        MIN(gini) AS gini
    FROM V_MODEL_VALIDATION_SPLIT
    GROUP BY
        rate_package_id,
        model_run_id
),
centered_split AS (
    SELECT
        validation_split.*,
        validation_split.deviance - metric_anchor.deviance AS deviance_delta,
        validation_split.nll - metric_anchor.nll AS nll_delta,
        validation_split.gini - metric_anchor.gini AS gini_delta
    FROM V_MODEL_VALIDATION_SPLIT AS validation_split
    JOIN metric_anchor
      ON metric_anchor.rate_package_id = validation_split.rate_package_id
     AND metric_anchor.model_run_id = validation_split.model_run_id
)
SELECT
    validation_split.model_id,
    validation_split.model_name,
    validation_split.model_label,
    validation_split.target_name,
    validation_split.model_type,
    validation_split.rate_package_id,
    validation_split.parent_rate_package_id,
    validation_split.model_version,
    validation_split.package_version,
    validation_split.parent_package_version,
    validation_split.package_kind,
    validation_split.package_status,
    validation_split.edit_reason,
    validation_split.model_run_id,
    validation_split.parent_model_run_id,
    validation_split.validation_source_model_run_id,
    validation_split.validation_source_rate_package_id,
    validation_split.validation_source_package_version,
    validation_split.validation_evidence,
    validation_split.build_fingerprint_sha256,
    validation_split.builder_source_sha256,
    validation_split.materialized_split_sha256,
    validation_split.runtime_sha256,
    validation_split.candidate_superglm_sha256,
    validation_split.model_source_sha256,
    validation_split.candidate_python_version,
    validation_split.candidate_superglm_version,
    validation_split.candidate_superglm_git_sha,
    validation_split.manifest_id,
    validation_split.dataset_name,
    validation_split.source_system,
    validation_split.data_as_of_date,
    validation_split.data_as_of_column,
    validation_split.dataset_row_count,
    validation_split.model_frame_sha256,
    validation_split.frame_hash_metadata_json,
    validation_split.split_set_id,
    validation_split.split_mode,
    validation_split.split_method,
    validation_split.splitter_class,
    validation_split.splitter_params_json,
    validation_split.row_order_sha256,
    validation_split.split_row_count,
    validation_split.fold_count,
    validation_split.artifact_uri,
    validation_split.artifact_sha256,
    validation_split.runtime_metadata_json,
    validation_split.family,
    validation_split.family_params_json,
    validation_split.link,
    COUNT(*) AS validation_split_count,
    AVG(validation_split.deviance) AS mean_deviance,
    CASE WHEN COUNT(validation_split.deviance) = 0 THEN NULL WHEN COUNT(validation_split.deviance) = 1 THEN 0.0 ELSE sqrt(max(AVG(validation_split.deviance_delta * validation_split.deviance_delta) - AVG(validation_split.deviance_delta) * AVG(validation_split.deviance_delta), 0.0)) END AS sd_deviance,
    AVG(validation_split.nll) AS mean_nll,
    CASE WHEN COUNT(validation_split.nll) = 0 THEN NULL WHEN COUNT(validation_split.nll) = 1 THEN 0.0 ELSE sqrt(max(AVG(validation_split.nll_delta * validation_split.nll_delta) - AVG(validation_split.nll_delta) * AVG(validation_split.nll_delta), 0.0)) END AS sd_nll,
    AVG(validation_split.gini) AS mean_gini,
    CASE WHEN COUNT(validation_split.gini) = 0 THEN NULL WHEN COUNT(validation_split.gini) = 1 THEN 0.0 ELSE sqrt(max(AVG(validation_split.gini_delta * validation_split.gini_delta) - AVG(validation_split.gini_delta) * AVG(validation_split.gini_delta), 0.0)) END AS sd_gini,
    SUM(CAST(validation_split.n_validation AS REAL)) / NULLIF(MAX(CAST(validation_split.dataset_row_count AS REAL)), 0.0) AS validation_prediction_coverage,
    validation_split.validation_curve_status,
    validation_split.validation_curve_reason
FROM centered_split AS validation_split
GROUP BY
    validation_split.model_id,
    validation_split.model_name,
    validation_split.model_label,
    validation_split.target_name,
    validation_split.model_type,
    validation_split.rate_package_id,
    validation_split.parent_rate_package_id,
    validation_split.model_version,
    validation_split.package_version,
    validation_split.parent_package_version,
    validation_split.package_kind,
    validation_split.package_status,
    validation_split.edit_reason,
    validation_split.model_run_id,
    validation_split.parent_model_run_id,
    validation_split.validation_source_model_run_id,
    validation_split.validation_source_rate_package_id,
    validation_split.validation_source_package_version,
    validation_split.validation_evidence,
    validation_split.build_fingerprint_sha256,
    validation_split.builder_source_sha256,
    validation_split.materialized_split_sha256,
    validation_split.runtime_sha256,
    validation_split.candidate_superglm_sha256,
    validation_split.model_source_sha256,
    validation_split.candidate_python_version,
    validation_split.candidate_superglm_version,
    validation_split.candidate_superglm_git_sha,
    validation_split.manifest_id,
    validation_split.dataset_name,
    validation_split.source_system,
    validation_split.data_as_of_date,
    validation_split.data_as_of_column,
    validation_split.dataset_row_count,
    validation_split.model_frame_sha256,
    validation_split.frame_hash_metadata_json,
    validation_split.split_set_id,
    validation_split.split_mode,
    validation_split.split_method,
    validation_split.splitter_class,
    validation_split.splitter_params_json,
    validation_split.row_order_sha256,
    validation_split.split_row_count,
    validation_split.fold_count,
    validation_split.artifact_uri,
    validation_split.artifact_sha256,
    validation_split.runtime_metadata_json,
    validation_split.family,
    validation_split.family_params_json,
    validation_split.link,
    validation_split.validation_curve_status,
    validation_split.validation_curve_reason;

CREATE VIEW pricing.V_MODEL_VALIDATION_SPLIT_RELATIVITY AS
SELECT
    model.model_id,
    model.model_name,
    model.model_label,
    model.target_name,
    model.model_type,
    final_package.rate_package_id,
    final_package.parent_rate_package_id,
    final_package.model_version,
    final_package.package_version,
    parent_package.package_version AS parent_package_version,
    COALESCE(json_extract(final_package.revision_metadata_json, '$.kind'), CASE WHEN final_package.parent_rate_package_id IS NULL THEN 'ORIGINAL' ELSE 'EDITED' END) AS package_kind,
    final_package.package_status,
    json_extract(final_package.revision_metadata_json, '$.reason') AS edit_reason,
    final_run.model_run_id,
    final_run.parent_model_run_id,
    source_run.model_run_id AS validation_source_model_run_id,
    source_package.rate_package_id AS validation_source_rate_package_id,
    source_package.package_version AS validation_source_package_version,
    CASE WHEN source_run.model_run_id = final_run.model_run_id THEN 'DIRECT' ELSE 'INHERITED_FROM_PARENT' END AS validation_evidence,
    source_package.build_fingerprint_sha256,
    source_run.builder_source_sha256,
    source_run.materialized_split_sha256,
    source_run.runtime_sha256,
    source_run.candidate_superglm_sha256,
    source_run.model_source_sha256,
    source_run.candidate_python_version,
    source_run.candidate_superglm_version,
    source_run.candidate_superglm_git_sha,
    split_set.split_set_id,
    COALESCE(CASE WHEN json_valid(split_set.splitter_params_json) THEN json_extract(split_set.splitter_params_json, '$.method') END, CASE WHEN split_set.splitter_class LIKE '%KFold' THEN 'kfold' WHEN split_set.splitter_class LIKE '%train_test_split' THEN 'train_test_split' ELSE 'custom' END) AS split_method,
    fold.fold_no AS validation_split_no,
    fold.n_train,
    fold.n_test AS n_validation,
    json_extract(final_package.package_metadata_json, '$.model.family') AS family,
    json_extract(final_package.package_metadata_json, '$.model.family_params') AS family_params_json,
    json_extract(final_package.package_metadata_json, '$.model.link') AS link,
    'VALIDATION_TRAINING_SPLIT_MODEL' AS model_fit_scope,
    point.term_name,
    point.point_no,
    point.point_kind,
    point.x_numeric,
    point.level_text,
    point.eta_contribution,
    point.relativity,
    point.support_value,
    point.reference_value,
    point.reference_level
FROM PRICING_MODEL AS model
JOIN PRICING_RATE_PACKAGE AS final_package
  ON final_package.model_id = model.model_id
JOIN MODEL_RUN AS final_run
  ON final_run.rate_package_id = final_package.rate_package_id
JOIN MODEL_RUN AS source_run
  ON source_run.model_run_id = COALESCE(final_run.validation_source_model_run_id, final_run.model_run_id)
JOIN PRICING_RATE_PACKAGE AS source_package
  ON source_package.rate_package_id = source_run.rate_package_id
LEFT JOIN PRICING_RATE_PACKAGE AS parent_package
  ON parent_package.rate_package_id = final_package.parent_rate_package_id
JOIN CV_SPLIT_SET AS split_set
  ON split_set.split_set_id = source_run.split_set_id
 AND split_set.manifest_id = source_run.manifest_id
JOIN CV_FOLD AS fold
  ON fold.split_set_id = split_set.split_set_id
JOIN CV_SPLIT_CURVE_POINT AS point
  ON point.model_run_id = source_run.model_run_id
 AND point.split_set_id = split_set.split_set_id
 AND point.split_no = fold.fold_no;

CREATE VIEW pricing.V_CURRENT_DATASET_VALIDATION_SPLIT AS
SELECT
    current_manifest.dataset_name,
    current_manifest.manifest_id,
    current_manifest.source_system,
    current_manifest.data_as_of_date,
    current_manifest.data_as_of_column,
    current_manifest.dataset_row_count,
    current_manifest.pk_columns_json,
    current_manifest.target_column,
    current_manifest.weight_column,
    current_manifest.offset_column,
    current_manifest.offset_source_column,
    current_manifest.offset_label,
    current_manifest.export_weight_column,
    current_manifest.model_frame_sha256,
    current_manifest.frame_hash_metadata_json,
    current_manifest.manifest_created_ts,
    current_manifest.manifest_created_by,
    current_split.split_set_id,
    current_split.split_mode,
    COALESCE(CASE WHEN json_valid(current_split.splitter_params_json) THEN json_extract(current_split.splitter_params_json, '$.method') END, CASE WHEN current_split.splitter_class LIKE '%KFold' THEN 'kfold' WHEN current_split.splitter_class LIKE '%train_test_split' THEN 'train_test_split' ELSE 'custom' END) AS split_method,
    current_split.splitter_class,
    current_split.splitter_params_json,
    current_split.row_order_sha256,
    current_split.split_row_count,
    current_split.fold_count,
    current_split.groups_column,
    current_split.stratify_column,
    current_split.artifact_uri,
    current_split.artifact_sha256,
    current_split.runtime_metadata_json,
    current_split.split_created_ts,
    current_split.split_created_by,
    fold.fold_no AS validation_split_no,
    fold.n_train,
    fold.n_test AS n_validation
FROM (
    SELECT
        manifest.manifest_id,
        manifest.dataset_name,
        manifest.source_system,
        manifest.data_as_of_date,
        manifest.data_as_of_column,
        manifest.row_count AS dataset_row_count,
        manifest.pk_columns_json,
        manifest.target_column,
        manifest.weight_column,
        manifest.offset_column,
        manifest.offset_source_column,
        manifest.offset_label,
        manifest.export_weight_column,
        manifest.model_frame_sha256,
        manifest.frame_hash_metadata_json,
        manifest.created_ts AS manifest_created_ts,
        manifest.created_by AS manifest_created_by,
        ROW_NUMBER() OVER (PARTITION BY manifest.dataset_name ORDER BY manifest.created_ts DESC, manifest.manifest_id DESC) AS manifest_rank
    FROM DATASET_MANIFEST AS manifest
) AS current_manifest
JOIN (
    SELECT
        split_set.split_set_id,
        split_set.manifest_id,
        split_set.split_mode,
        split_set.splitter_class,
        split_set.splitter_params_json,
        split_set.row_order_sha256,
        split_set.row_count AS split_row_count,
        split_set.fold_count,
        split_set.groups_column,
        split_set.stratify_column,
        split_set.artifact_uri,
        split_set.artifact_sha256,
        split_set.runtime_metadata_json,
        split_set.created_ts AS split_created_ts,
        split_set.created_by AS split_created_by,
        ROW_NUMBER() OVER (PARTITION BY split_set.manifest_id ORDER BY split_set.created_ts DESC, split_set.split_set_id DESC) AS split_rank
    FROM CV_SPLIT_SET AS split_set
) AS current_split
  ON current_split.manifest_id = current_manifest.manifest_id
 AND current_split.split_rank = 1
JOIN CV_FOLD AS fold
  ON fold.split_set_id = current_split.split_set_id
WHERE current_manifest.manifest_rank = 1;

COMMIT;
