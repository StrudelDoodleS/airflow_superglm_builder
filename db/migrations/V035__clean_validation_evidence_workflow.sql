IF COL_LENGTH('pricing.MODEL_RUN', 'candidate_superglm_git_sha') IS NULL
BEGIN
    ALTER TABLE pricing.MODEL_RUN
    ADD candidate_superglm_git_sha CHAR(40) NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_MODEL_RUN_CANDIDATE_SUPERGLM_GIT_SHA'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
BEGIN
    ALTER TABLE pricing.MODEL_RUN WITH CHECK
    ADD CONSTRAINT CK_MODEL_RUN_CANDIDATE_SUPERGLM_GIT_SHA CHECK (
        (
            candidate_artifact_format IS NULL
            OR candidate_artifact_format <> 'superglm-candidate-joblib-v3'
            OR candidate_superglm_git_sha IS NOT NULL
        )
        AND (
            candidate_superglm_git_sha IS NULL
            OR (
                LEN(candidate_superglm_git_sha) = 40
                AND candidate_superglm_git_sha COLLATE Latin1_General_BIN2
                    NOT LIKE '%[^0-9a-f]%'
            )
        )
    );
END;
GO

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'build_fingerprint_sha256') IS NULL
BEGIN
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
    ADD build_fingerprint_sha256 CHAR(64) NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_PRICING_RATE_PACKAGE_BUILD_FINGERPRINT_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
)
BEGIN
    ALTER TABLE pricing.PRICING_RATE_PACKAGE WITH CHECK
    ADD CONSTRAINT CK_PRICING_RATE_PACKAGE_BUILD_FINGERPRINT_SHA256 CHECK (
        build_fingerprint_sha256 IS NULL
        OR (
            parent_rate_package_id IS NULL
            AND LEN(build_fingerprint_sha256) = 64
            AND build_fingerprint_sha256 COLLATE Latin1_General_BIN2
                NOT LIKE '%[^0-9a-f]%'
        )
    );
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_PRICING_RATE_PACKAGE_MODEL_BUILD_FINGERPRINT'
      AND object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
)
BEGIN
    CREATE UNIQUE INDEX UX_PRICING_RATE_PACKAGE_MODEL_BUILD_FINGERPRINT
    ON pricing.PRICING_RATE_PACKAGE(model_id, build_fingerprint_sha256)
    WHERE parent_rate_package_id IS NULL
      AND build_fingerprint_sha256 IS NOT NULL;
END;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'builder_source_sha256') IS NULL
BEGIN
    ALTER TABLE pricing.MODEL_RUN
    ADD builder_source_sha256 CHAR(64) NULL;
END;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'materialized_split_sha256') IS NULL
BEGIN
    ALTER TABLE pricing.MODEL_RUN
    ADD materialized_split_sha256 CHAR(64) NULL;
END;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'runtime_sha256') IS NULL
BEGIN
    ALTER TABLE pricing.MODEL_RUN
    ADD runtime_sha256 CHAR(64) NULL;
END;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'candidate_superglm_sha256') IS NULL
BEGIN
    ALTER TABLE pricing.MODEL_RUN
    ADD candidate_superglm_sha256 CHAR(64) NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_MODEL_RUN_BUILDER_SOURCE_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
BEGIN
    ALTER TABLE pricing.MODEL_RUN WITH CHECK
    ADD CONSTRAINT CK_MODEL_RUN_BUILDER_SOURCE_SHA256 CHECK (
        builder_source_sha256 IS NULL
        OR (
            LEN(builder_source_sha256) = 64
            AND builder_source_sha256 COLLATE Latin1_General_BIN2
                NOT LIKE '%[^0-9a-f]%'
        )
    );
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_MODEL_RUN_MATERIALIZED_SPLIT_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
BEGIN
    ALTER TABLE pricing.MODEL_RUN WITH CHECK
    ADD CONSTRAINT CK_MODEL_RUN_MATERIALIZED_SPLIT_SHA256 CHECK (
        materialized_split_sha256 IS NULL
        OR (
            LEN(materialized_split_sha256) = 64
            AND materialized_split_sha256 COLLATE Latin1_General_BIN2
                NOT LIKE '%[^0-9a-f]%'
        )
    );
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_MODEL_RUN_RUNTIME_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
BEGIN
    ALTER TABLE pricing.MODEL_RUN WITH CHECK
    ADD CONSTRAINT CK_MODEL_RUN_RUNTIME_SHA256 CHECK (
        runtime_sha256 IS NULL
        OR (
            LEN(runtime_sha256) = 64
            AND runtime_sha256 COLLATE Latin1_General_BIN2
                NOT LIKE '%[^0-9a-f]%'
        )
    );
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_MODEL_RUN_CANDIDATE_SUPERGLM_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
BEGIN
    ALTER TABLE pricing.MODEL_RUN WITH CHECK
    ADD CONSTRAINT CK_MODEL_RUN_CANDIDATE_SUPERGLM_SHA256 CHECK (
        candidate_superglm_sha256 IS NULL
        OR (
            LEN(candidate_superglm_sha256) = 64
            AND candidate_superglm_sha256 COLLATE Latin1_General_BIN2
                NOT LIKE '%[^0-9a-f]%'
        )
    );
END;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'validation_curve_reason') IS NULL
BEGIN
    ALTER TABLE pricing.MODEL_RUN
    ADD validation_curve_reason NVARCHAR(500) NULL;
END;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'validation_curve_status') IS NULL
BEGIN
    ALTER TABLE pricing.MODEL_RUN
    ADD validation_curve_status NVARCHAR(32) NULL;
END;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'validation_source_model_run_id') IS NULL
BEGIN
    ALTER TABLE pricing.MODEL_RUN
    ADD validation_source_model_run_id BIGINT NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_MODEL_RUN_VALIDATION_CURVE_STATUS'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
BEGIN
    ALTER TABLE pricing.MODEL_RUN WITH CHECK
    ADD CONSTRAINT CK_MODEL_RUN_VALIDATION_CURVE_STATUS CHECK (
        (validation_curve_status IS NULL AND validation_curve_reason IS NULL)
        OR (
            validation_curve_status IS NOT NULL
            AND validation_curve_status = 'COMPLETE'
            AND validation_curve_reason IS NULL
        )
        OR (
            validation_curve_status IS NOT NULL
            AND validation_curve_status = 'UNAVAILABLE'
            AND validation_curve_reason IS NOT NULL
            AND LEN(LTRIM(RTRIM(validation_curve_reason))) > 0
        )
    );
END;
GO

;WITH package_lineage AS (
    SELECT
        candidate_run.model_run_id AS candidate_model_run_id,
        candidate_package.rate_package_id,
        candidate_package.parent_rate_package_id,
        0 AS lineage_depth
    FROM pricing.MODEL_RUN AS candidate_run
    JOIN pricing.PRICING_RATE_PACKAGE AS candidate_package
      ON candidate_package.rate_package_id = candidate_run.rate_package_id

    UNION ALL

    SELECT
        package_lineage.candidate_model_run_id,
        parent_package.rate_package_id,
        parent_package.parent_rate_package_id,
        package_lineage.lineage_depth + 1
    FROM package_lineage
    JOIN pricing.PRICING_RATE_PACKAGE AS parent_package
      ON parent_package.rate_package_id = package_lineage.parent_rate_package_id
    WHERE package_lineage.lineage_depth < 100
),
provable_validation_source AS (
    SELECT
        package_lineage.candidate_model_run_id,
        source_run.model_run_id
    FROM package_lineage
    JOIN pricing.MODEL_RUN AS source_run
      ON source_run.rate_package_id = package_lineage.rate_package_id
    WHERE package_lineage.parent_rate_package_id IS NULL
      AND EXISTS (
          SELECT 1
          FROM mlops.MODEL_RUN_SPLIT_SET AS source_split
          WHERE source_split.model_run_id = source_run.model_run_id
            AND source_split.split_role = 'validation'
      )
)
UPDATE candidate_run
SET validation_source_model_run_id = source_run.model_run_id
FROM pricing.MODEL_RUN AS candidate_run
JOIN provable_validation_source AS source_run
  ON source_run.candidate_model_run_id = candidate_run.model_run_id
WHERE candidate_run.validation_source_model_run_id IS NULL
OPTION (MAXRECURSION 32767);
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = 'FK_MODEL_RUN_VALIDATION_SOURCE'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
BEGIN
    ALTER TABLE pricing.MODEL_RUN WITH CHECK
    ADD CONSTRAINT FK_MODEL_RUN_VALIDATION_SOURCE
        FOREIGN KEY (validation_source_model_run_id)
        REFERENCES pricing.MODEL_RUN(model_run_id);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'IX_MODEL_RUN_VALIDATION_SOURCE'
      AND object_id = OBJECT_ID('pricing.MODEL_RUN')
)
BEGIN
    CREATE INDEX IX_MODEL_RUN_VALIDATION_SOURCE
    ON pricing.MODEL_RUN(validation_source_model_run_id)
    WHERE validation_source_model_run_id IS NOT NULL;
END;
GO

IF OBJECT_ID('pricing.CV_SPLIT_CURVE_POINT', 'U') IS NULL
BEGIN
    CREATE TABLE pricing.CV_SPLIT_CURVE_POINT (
        model_run_id BIGINT NOT NULL,
        split_set_id NVARCHAR(128) NOT NULL,
        split_no INT NOT NULL,
        term_name NVARCHAR(256) NOT NULL,
        point_no INT NOT NULL,
        point_kind NVARCHAR(16) NOT NULL,
        x_numeric FLOAT NULL,
        level_text NVARCHAR(512) NULL,
        eta_contribution FLOAT NOT NULL,
        relativity FLOAT NULL,
        support_value FLOAT NULL,
        reference_value FLOAT NULL,
        reference_level NVARCHAR(512) NULL,

        CONSTRAINT PK_CV_SPLIT_CURVE_POINT
            PRIMARY KEY (model_run_id, split_set_id, split_no, term_name, point_no),

        CONSTRAINT FK_CV_SPLIT_CURVE_POINT_RUN
            FOREIGN KEY (model_run_id)
            REFERENCES pricing.MODEL_RUN(model_run_id),

        CONSTRAINT FK_CV_SPLIT_CURVE_POINT_FOLD
            FOREIGN KEY (split_set_id, split_no)
            REFERENCES pricing.CV_FOLD(split_set_id, fold_no),

        CONSTRAINT CK_CV_SPLIT_CURVE_POINT_POSITIVE_KEYS CHECK (
            split_no > 0
            AND point_no > 0
        ),

        CONSTRAINT CK_CV_SPLIT_CURVE_POINT_TERM_NAME CHECK (
            LEN(LTRIM(RTRIM(term_name))) > 0
        ),

        CONSTRAINT CK_CV_SPLIT_CURVE_POINT_KIND CHECK (
            point_kind IN ('NUMERIC', 'LEVEL')
        ),

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
                AND LEN(LTRIM(RTRIM(level_text))) > 0
                AND reference_value IS NULL
                AND reference_level IS NOT NULL
                AND LEN(LTRIM(RTRIM(reference_level))) > 0
            )
        ),

        CONSTRAINT CK_CV_SPLIT_CURVE_POINT_SUPPORT CHECK (
            support_value IS NULL OR support_value >= 0
        ),

        CONSTRAINT CK_CV_SPLIT_CURVE_POINT_RELATIVITY CHECK (
            relativity IS NULL OR relativity >= 0
        )
    );
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'IX_CV_SPLIT_CURVE_POINT_SPLIT'
      AND object_id = OBJECT_ID('pricing.CV_SPLIT_CURVE_POINT')
)
BEGIN
    CREATE INDEX IX_CV_SPLIT_CURVE_POINT_SPLIT
    ON pricing.CV_SPLIT_CURVE_POINT(split_set_id, split_no);
END;
GO

IF OBJECT_ID('pricing.V_PUBLISHED_MODEL_RELATIVITY', 'V') IS NOT NULL
BEGIN
    DROP VIEW pricing.V_PUBLISHED_MODEL_RELATIVITY;
END;
GO

IF OBJECT_ID('pricing.V_MODEL_RELATIVITY', 'V') IS NOT NULL
BEGIN
    DROP VIEW pricing.V_MODEL_RELATIVITY;
END;
GO

IF OBJECT_ID('pricing.V_CURRENT_DATASET_CV_FOLD', 'V') IS NOT NULL
BEGIN
    DROP VIEW pricing.V_CURRENT_DATASET_CV_FOLD;
END;
GO

CREATE OR ALTER VIEW pricing.V_FINAL_MODEL_RELATIVITY AS
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
    COALESCE(JSON_VALUE(rp.revision_metadata_json, '$.kind'), CASE WHEN rp.parent_rate_package_id IS NULL THEN 'ORIGINAL' ELSE 'EDITED' END) AS package_kind,
    rp.package_status,
    rp.base_rate,
    rp.effective_from_date,
    rp.effective_to_date,
    JSON_VALUE(rp.package_metadata_json, '$.model.family') AS family,
    JSON_QUERY(rp.package_metadata_json, '$.model.family_params') AS family_params_json,
    JSON_VALUE(rp.package_metadata_json, '$.model.link') AS link,
    JSON_VALUE(rp.revision_metadata_json, '$.reason') AS edit_reason,
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
FROM pricing.PRICING_MODEL AS model
JOIN pricing.PRICING_RATE_PACKAGE AS rp
  ON rp.model_id = model.model_id
LEFT JOIN pricing.PRICING_RATE_PACKAGE AS parent_package
  ON parent_package.rate_package_id = rp.parent_rate_package_id
JOIN pricing.PRICING_COMPILED_1D_RATE_BAND AS band
  ON band.rate_package_id = rp.rate_package_id
JOIN pricing.PRICING_TERM AS term
  ON term.term_id = band.term_id
JOIN pricing.PRICING_RATE_CELL_LEVEL AS cell_level
  ON cell_level.feature_level_id = band.feature_level_id
 AND cell_level.position_no = 1
JOIN pricing.PRICING_RATE_CELL AS rate_cell
  ON rate_cell.cell_id = cell_level.cell_id
 AND rate_cell.term_id = band.term_id
 AND rate_cell.is_deleted = 0
JOIN pricing.PRICING_COMPILED_RATE_CELL AS compiled_cell
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
    COALESCE(JSON_VALUE(rp.revision_metadata_json, '$.kind'), CASE WHEN rp.parent_rate_package_id IS NULL THEN 'ORIGINAL' ELSE 'EDITED' END) AS package_kind,
    rp.package_status,
    rp.base_rate,
    rp.effective_from_date,
    rp.effective_to_date,
    JSON_VALUE(rp.package_metadata_json, '$.model.family') AS family,
    JSON_QUERY(rp.package_metadata_json, '$.model.family_params') AS family_params_json,
    JSON_VALUE(rp.package_metadata_json, '$.model.link') AS link,
    JSON_VALUE(rp.revision_metadata_json, '$.reason') AS edit_reason,
    'PACKAGE_FINAL_MODEL' AS model_fit_scope,
    cell.term_id,
    cell.sequence_no AS term_sequence_no,
    cell.term_name,
    cell.term_type,
    CASE WHEN LEFT(cell.cell_key_text, LEN(cell.term_name) + 1) = CONCAT(cell.term_name, '=') THEN SUBSTRING(cell.cell_key_text, LEN(cell.term_name) + 2, LEN(cell.cell_key_text)) ELSE cell.cell_key_text END AS term_level,
    CAST(NULL AS INT) AS level_sort_order,
    CAST(NULL AS FLOAT) AS lower_bound,
    CAST(NULL AS FLOAT) AS upper_bound,
    CAST(NULL AS FLOAT) AS representative_value,
    cell.log_coefficient AS eta_contribution,
    cell.multiplier AS relativity,
    cell.exposure_weight,
    cell.record_count,
    cell.is_default,
    cell.is_reference,
    'RATE_CELL' AS relativity_source
FROM pricing.PRICING_MODEL AS model
JOIN pricing.PRICING_RATE_PACKAGE AS rp
  ON rp.model_id = model.model_id
LEFT JOIN pricing.PRICING_RATE_PACKAGE AS parent_package
  ON parent_package.rate_package_id = rp.parent_rate_package_id
JOIN pricing.PRICING_COMPILED_RATE_CELL AS cell
  ON cell.rate_package_id = rp.rate_package_id
WHERE NOT EXISTS (
    SELECT 1
    FROM pricing.PRICING_COMPILED_1D_RATE_BAND AS band
    WHERE band.rate_package_id = cell.rate_package_id
      AND band.term_id = cell.term_id
);
GO

CREATE OR ALTER VIEW pricing.V_MODEL_VALIDATION_SPLIT AS
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
    COALESCE(JSON_VALUE(final_package.revision_metadata_json, '$.kind'), CASE WHEN final_package.parent_rate_package_id IS NULL THEN 'ORIGINAL' ELSE 'EDITED' END) AS package_kind,
    final_package.package_status,
    JSON_VALUE(final_package.revision_metadata_json, '$.reason') AS edit_reason,
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
    COALESCE(CASE WHEN ISJSON(split_set.splitter_params_json) = 1 THEN JSON_VALUE(split_set.splitter_params_json, '$.method') END, CASE WHEN split_set.splitter_class LIKE '%KFold' THEN 'kfold' WHEN split_set.splitter_class LIKE '%train_test_split' THEN 'train_test_split' ELSE 'custom' END) AS split_method,
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
    JSON_VALUE(final_package.package_metadata_json, '$.model.family') AS family,
    JSON_QUERY(final_package.package_metadata_json, '$.model.family_params') AS family_params_json,
    JSON_VALUE(final_package.package_metadata_json, '$.model.link') AS link,
    source_run.validation_curve_status,
    source_run.validation_curve_reason
FROM pricing.PRICING_MODEL AS model
JOIN pricing.PRICING_RATE_PACKAGE AS final_package
  ON final_package.model_id = model.model_id
JOIN pricing.MODEL_RUN AS final_run
  ON final_run.rate_package_id = final_package.rate_package_id
JOIN pricing.MODEL_RUN AS source_run
  ON source_run.model_run_id = COALESCE(final_run.validation_source_model_run_id, final_run.model_run_id)
JOIN pricing.PRICING_RATE_PACKAGE AS source_package
  ON source_package.rate_package_id = source_run.rate_package_id
LEFT JOIN pricing.PRICING_RATE_PACKAGE AS parent_package
  ON parent_package.rate_package_id = final_package.parent_rate_package_id
JOIN mlops.MODEL_RUN_SPLIT_SET AS run_split
  ON run_split.model_run_id = source_run.model_run_id
 AND run_split.split_role = 'validation'
JOIN pricing.DATASET_MANIFEST AS manifest
  ON manifest.manifest_id = run_split.manifest_id
JOIN pricing.CV_SPLIT_SET AS split_set
  ON split_set.split_set_id = run_split.split_set_id
 AND split_set.manifest_id = run_split.manifest_id
JOIN pricing.CV_FOLD AS fold
  ON fold.split_set_id = split_set.split_set_id
LEFT JOIN (
    SELECT
        fold_metric.model_run_id,
        fold_metric.split_set_id,
        fold_metric.fold_no,
        MAX(CASE WHEN LOWER(fold_metric.metric_name) = 'deviance' THEN fold_metric.metric_value END) AS deviance,
        MAX(CASE WHEN LOWER(fold_metric.metric_name) = 'nll' THEN fold_metric.metric_value END) AS nll,
        MAX(CASE WHEN LOWER(fold_metric.metric_name) = 'gini' THEN fold_metric.metric_value END) AS gini
    FROM pricing.CV_FOLD_METRIC AS fold_metric
    GROUP BY
        fold_metric.model_run_id,
        fold_metric.split_set_id,
        fold_metric.fold_no
) AS fold_metrics
  ON fold_metrics.model_run_id = source_run.model_run_id
 AND fold_metrics.split_set_id = split_set.split_set_id
 AND fold_metrics.fold_no = fold.fold_no;
GO

CREATE OR ALTER VIEW pricing.V_MODEL_VALIDATION_SUMMARY AS
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
    COALESCE(JSON_VALUE(final_package.revision_metadata_json, '$.kind'), CASE WHEN final_package.parent_rate_package_id IS NULL THEN 'ORIGINAL' ELSE 'EDITED' END) AS package_kind,
    final_package.package_status,
    JSON_VALUE(final_package.revision_metadata_json, '$.reason') AS edit_reason,
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
    COALESCE(CASE WHEN ISJSON(split_set.splitter_params_json) = 1 THEN JSON_VALUE(split_set.splitter_params_json, '$.method') END, CASE WHEN split_set.splitter_class LIKE '%KFold' THEN 'kfold' WHEN split_set.splitter_class LIKE '%train_test_split' THEN 'train_test_split' ELSE 'custom' END) AS split_method,
    split_set.splitter_class,
    split_set.splitter_params_json,
    split_set.row_order_sha256,
    split_set.row_count AS split_row_count,
    split_set.fold_count,
    split_set.artifact_uri,
    split_set.artifact_sha256,
    split_set.runtime_metadata_json,
    JSON_VALUE(final_package.package_metadata_json, '$.model.family') AS family,
    JSON_QUERY(final_package.package_metadata_json, '$.model.family_params') AS family_params_json,
    JSON_VALUE(final_package.package_metadata_json, '$.model.link') AS link,
    validation_summary.validation_split_count,
    validation_summary.mean_deviance,
    validation_summary.sd_deviance,
    validation_summary.mean_nll,
    validation_summary.sd_nll,
    validation_summary.mean_gini,
    validation_summary.sd_gini,
    validation_summary.validation_row_count / NULLIF(CAST(manifest.row_count AS FLOAT), 0.0) AS validation_prediction_coverage,
    source_run.validation_curve_status,
    source_run.validation_curve_reason
FROM pricing.PRICING_MODEL AS model
JOIN pricing.PRICING_RATE_PACKAGE AS final_package
  ON final_package.model_id = model.model_id
JOIN pricing.MODEL_RUN AS final_run
  ON final_run.rate_package_id = final_package.rate_package_id
JOIN pricing.MODEL_RUN AS source_run
  ON source_run.model_run_id = COALESCE(final_run.validation_source_model_run_id, final_run.model_run_id)
JOIN pricing.PRICING_RATE_PACKAGE AS source_package
  ON source_package.rate_package_id = source_run.rate_package_id
LEFT JOIN pricing.PRICING_RATE_PACKAGE AS parent_package
  ON parent_package.rate_package_id = final_package.parent_rate_package_id
JOIN mlops.MODEL_RUN_SPLIT_SET AS run_split
  ON run_split.model_run_id = source_run.model_run_id
 AND run_split.split_role = 'validation'
JOIN pricing.DATASET_MANIFEST AS manifest
  ON manifest.manifest_id = run_split.manifest_id
JOIN pricing.CV_SPLIT_SET AS split_set
  ON split_set.split_set_id = run_split.split_set_id
 AND split_set.manifest_id = run_split.manifest_id
OUTER APPLY (
    SELECT
        COUNT(*) AS validation_split_count,
        AVG(held_out.deviance) AS mean_deviance,
        CASE WHEN COUNT(held_out.deviance) = 1 THEN CAST(0.0 AS FLOAT) ELSE STDEVP(held_out.deviance) END AS sd_deviance,
        AVG(held_out.nll) AS mean_nll,
        CASE WHEN COUNT(held_out.nll) = 1 THEN CAST(0.0 AS FLOAT) ELSE STDEVP(held_out.nll) END AS sd_nll,
        AVG(held_out.gini) AS mean_gini,
        CASE WHEN COUNT(held_out.gini) = 1 THEN CAST(0.0 AS FLOAT) ELSE STDEVP(held_out.gini) END AS sd_gini,
        SUM(CAST(held_out.n_validation AS FLOAT)) AS validation_row_count
    FROM (
        SELECT
            fold.fold_no,
            fold.n_test AS n_validation,
            MAX(CASE WHEN LOWER(fold_metric.metric_name) = 'deviance' THEN fold_metric.metric_value END) AS deviance,
            MAX(CASE WHEN LOWER(fold_metric.metric_name) = 'nll' THEN fold_metric.metric_value END) AS nll,
            MAX(CASE WHEN LOWER(fold_metric.metric_name) = 'gini' THEN fold_metric.metric_value END) AS gini
        FROM pricing.CV_FOLD AS fold
        LEFT JOIN pricing.CV_FOLD_METRIC AS fold_metric
          ON fold_metric.model_run_id = source_run.model_run_id
         AND fold_metric.split_set_id = fold.split_set_id
         AND fold_metric.fold_no = fold.fold_no
        WHERE fold.split_set_id = split_set.split_set_id
        GROUP BY
            fold.fold_no,
            fold.n_test
    ) AS held_out
) AS validation_summary;
GO

CREATE OR ALTER VIEW pricing.V_MODEL_VALIDATION_SPLIT_RELATIVITY AS
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
    COALESCE(JSON_VALUE(final_package.revision_metadata_json, '$.kind'), CASE WHEN final_package.parent_rate_package_id IS NULL THEN 'ORIGINAL' ELSE 'EDITED' END) AS package_kind,
    final_package.package_status,
    JSON_VALUE(final_package.revision_metadata_json, '$.reason') AS edit_reason,
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
    COALESCE(CASE WHEN ISJSON(split_set.splitter_params_json) = 1 THEN JSON_VALUE(split_set.splitter_params_json, '$.method') END, CASE WHEN split_set.splitter_class LIKE '%KFold' THEN 'kfold' WHEN split_set.splitter_class LIKE '%train_test_split' THEN 'train_test_split' ELSE 'custom' END) AS split_method,
    fold.fold_no AS validation_split_no,
    fold.n_train,
    fold.n_test AS n_validation,
    JSON_VALUE(final_package.package_metadata_json, '$.model.family') AS family,
    JSON_QUERY(final_package.package_metadata_json, '$.model.family_params') AS family_params_json,
    JSON_VALUE(final_package.package_metadata_json, '$.model.link') AS link,
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
FROM pricing.PRICING_MODEL AS model
JOIN pricing.PRICING_RATE_PACKAGE AS final_package
  ON final_package.model_id = model.model_id
JOIN pricing.MODEL_RUN AS final_run
  ON final_run.rate_package_id = final_package.rate_package_id
JOIN pricing.MODEL_RUN AS source_run
  ON source_run.model_run_id = COALESCE(final_run.validation_source_model_run_id, final_run.model_run_id)
JOIN pricing.PRICING_RATE_PACKAGE AS source_package
  ON source_package.rate_package_id = source_run.rate_package_id
LEFT JOIN pricing.PRICING_RATE_PACKAGE AS parent_package
  ON parent_package.rate_package_id = final_package.parent_rate_package_id
JOIN mlops.MODEL_RUN_SPLIT_SET AS run_split
  ON run_split.model_run_id = source_run.model_run_id
 AND run_split.split_role = 'validation'
JOIN pricing.CV_SPLIT_SET AS split_set
  ON split_set.split_set_id = run_split.split_set_id
 AND split_set.manifest_id = run_split.manifest_id
JOIN pricing.CV_FOLD AS fold
  ON fold.split_set_id = split_set.split_set_id
JOIN pricing.CV_SPLIT_CURVE_POINT AS point
  ON point.model_run_id = source_run.model_run_id
 AND point.split_set_id = split_set.split_set_id
 AND point.split_no = fold.fold_no;
GO

CREATE OR ALTER VIEW pricing.V_CURRENT_DATASET_VALIDATION_SPLIT AS
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
    COALESCE(CASE WHEN ISJSON(current_split.splitter_params_json) = 1 THEN JSON_VALUE(current_split.splitter_params_json, '$.method') END, CASE WHEN current_split.splitter_class LIKE '%KFold' THEN 'kfold' WHEN current_split.splitter_class LIKE '%train_test_split' THEN 'train_test_split' ELSE 'custom' END) AS split_method,
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
    FROM pricing.DATASET_MANIFEST AS manifest
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
    FROM pricing.CV_SPLIT_SET AS split_set
) AS current_split
  ON current_split.manifest_id = current_manifest.manifest_id
 AND current_split.split_rank = 1
JOIN pricing.CV_FOLD AS fold
  ON fold.split_set_id = current_split.split_set_id
WHERE current_manifest.manifest_rank = 1;
GO
