IF COL_LENGTH('pricing.MODEL_RUN', 'model_kind') IS NULL
BEGIN
    ALTER TABLE pricing.MODEL_RUN
        ADD model_kind NVARCHAR(32) NULL;
END;
GO

UPDATE pricing.MODEL_RUN
SET model_kind = CASE
    WHEN parent_model_run_id IS NOT NULL THEN 'EDITOR_EDIT'
    ELSE 'RAW'
END
WHERE model_kind IS NULL;
GO

ALTER TABLE pricing.MODEL_RUN
    ALTER COLUMN model_kind NVARCHAR(32) NOT NULL;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_MODEL_RUN_MODEL_KIND'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
BEGIN
    ALTER TABLE pricing.MODEL_RUN WITH CHECK
        ADD CONSTRAINT CK_MODEL_RUN_MODEL_KIND
        CHECK (model_kind IN ('RAW', 'ROUTINE_EDIT', 'EDITOR_EDIT'));
END;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'model_equivalence_sha256') IS NULL
BEGIN
    ALTER TABLE pricing.MODEL_RUN
        ADD model_equivalence_sha256 CHAR(64) NULL;
END;
GO

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'model_equivalence_sha256') IS NULL
BEGIN
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
        ADD model_equivalence_sha256 CHAR(64) NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_MODEL_RUN_EQUIVALENCE_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
BEGIN
    ALTER TABLE pricing.MODEL_RUN WITH CHECK
        ADD CONSTRAINT CK_MODEL_RUN_EQUIVALENCE_SHA256
        CHECK (
            model_equivalence_sha256 IS NULL
            OR (
                LEN(model_equivalence_sha256) = 64
                AND model_equivalence_sha256 COLLATE Latin1_General_BIN2
                    NOT LIKE '%[^0-9a-f]%'
            )
        );
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_STG_RATING_EXPORT_EQUIVALENCE_SHA256'
      AND parent_object_id = OBJECT_ID('pricing_stg.STG_RATING_EXPORT')
)
BEGIN
    ALTER TABLE pricing_stg.STG_RATING_EXPORT WITH CHECK
        ADD CONSTRAINT CK_STG_RATING_EXPORT_EQUIVALENCE_SHA256
        CHECK (
            model_equivalence_sha256 IS NULL
            OR (
                LEN(model_equivalence_sha256) = 64
                AND model_equivalence_sha256 COLLATE Latin1_General_BIN2
                    NOT LIKE '%[^0-9a-f]%'
            )
        );
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_MODEL_RUN_EQUIVALENT_SUCCESS'
      AND object_id = OBJECT_ID('pricing.MODEL_RUN')
)
BEGIN
    CREATE UNIQUE INDEX UX_MODEL_RUN_EQUIVALENT_SUCCESS
        ON pricing.MODEL_RUN(
            model_id,
            manifest_id,
            model_kind,
            model_equivalence_sha256
        )
        WHERE model_equivalence_sha256 IS NOT NULL
          AND run_status = 'SUCCESS';
END;
GO

IF COL_LENGTH('pricing.DATASET_MANIFEST', 'manifest_signature_sha256') IS NULL
BEGIN
    ALTER TABLE pricing.DATASET_MANIFEST
        ADD manifest_signature_sha256 CHAR(64) NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_DATASET_MANIFEST_SIGNATURE_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.DATASET_MANIFEST')
)
BEGIN
    ALTER TABLE pricing.DATASET_MANIFEST WITH CHECK
        ADD CONSTRAINT CK_DATASET_MANIFEST_SIGNATURE_SHA256
        CHECK (
            manifest_signature_sha256 IS NULL
            OR (
                LEN(manifest_signature_sha256) = 64
                AND manifest_signature_sha256 COLLATE Latin1_General_BIN2
                    NOT LIKE '%[^0-9a-f]%'
            )
        );
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_DATASET_MANIFEST_SIGNATURE'
      AND object_id = OBJECT_ID('pricing.DATASET_MANIFEST')
)
BEGIN
    CREATE UNIQUE INDEX UX_DATASET_MANIFEST_SIGNATURE
        ON pricing.DATASET_MANIFEST(manifest_signature_sha256)
        WHERE manifest_signature_sha256 IS NOT NULL;
END;
GO

CREATE OR ALTER VIEW pricing.V_FINAL_MODEL_RELATIVITY
AS
WITH validation_split_links AS (
    SELECT
        model_run_id,
        COUNT_BIG(*) AS validation_split_link_count,
        CASE
            WHEN COUNT_BIG(*) = 1 THEN MAX(manifest_id)
            ELSE NULL
        END AS validation_manifest_id,
        CASE
            WHEN COUNT_BIG(*) = 1 THEN MAX(split_set_id)
            ELSE NULL
        END AS validation_split_set_id
    FROM mlops.MODEL_RUN_SPLIT_SET
    WHERE dataset_role = 'training'
      AND split_role = 'validation'
    GROUP BY model_run_id
)
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
    validation_split.validation_manifest_id
        AS validation_split_manifest_id,
    validation_split.validation_split_set_id,
    COALESCE(validation_split.validation_split_link_count, 0)
        AS validation_split_link_count,
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
FROM pricing.V_MODEL_RELATIVITY AS relativity
LEFT JOIN pricing.MODEL_RUN AS model_run
  ON model_run.model_run_id = relativity.model_run_id
LEFT JOIN mlops.MODEL_RUN_DATASET AS run_dataset
  ON run_dataset.model_run_id = model_run.model_run_id
 AND run_dataset.dataset_role = 'training'
 AND run_dataset.manifest_id = model_run.manifest_id
LEFT JOIN pricing.DATASET_MANIFEST AS manifest
  ON manifest.manifest_id = COALESCE(run_dataset.manifest_id, model_run.manifest_id)
LEFT JOIN validation_split_links AS validation_split
  ON validation_split.model_run_id = model_run.model_run_id;
GO

CREATE OR ALTER VIEW pricing.V_PUBLISHED_MODEL_RELATIVITY
AS
SELECT *
FROM pricing.V_FINAL_MODEL_RELATIVITY
WHERE package_status = 'PUBLISHED';
GO

CREATE OR ALTER VIEW pricing.V_MODEL_CANDIDATE_RELATIVITY
AS
SELECT *
FROM pricing.V_FINAL_MODEL_RELATIVITY
WHERE package_status = 'PUBLISHED';
GO

CREATE OR ALTER VIEW pricing.V_CURRENT_DEPLOYED_RELATIVITY
AS
SELECT
    deployment.deployment_id,
    deployment.deployment_slot,
    deployment.effective_from_ts AS deployment_effective_from_ts,
    deployment.effective_to_ts AS deployment_effective_to_ts,
    deployment.deployed_by,
    deployment.deployment_note,
    deployment.created_ts AS deployment_created_ts,
    relativity.*
FROM pricing.PRICING_MODEL_DEPLOYMENT AS deployment
JOIN pricing.V_FINAL_MODEL_RELATIVITY AS relativity
  ON relativity.model_id = deployment.model_id
 AND relativity.rate_package_id = deployment.rate_package_id
WHERE deployment.effective_to_ts IS NULL
  AND relativity.package_status = 'PUBLISHED';
GO

CREATE OR ALTER VIEW pricing.V_MODEL_LINEAGE_REDUNDANCY_CHECK
AS
WITH dataset_links AS (
    SELECT
        model_run_id,
        SUM(CASE WHEN dataset_role = 'training' THEN 1 ELSE 0 END)
            AS training_manifest_link_count,
        MAX(CASE WHEN dataset_role = 'training' THEN manifest_id END)
            AS linked_training_manifest_id
    FROM mlops.MODEL_RUN_DATASET
    GROUP BY model_run_id
),
split_links AS (
    SELECT
        model_run_id,
        SUM(
            CASE
                WHEN dataset_role = 'training' AND split_role = 'validation'
                THEN 1 ELSE 0
            END
        ) AS validation_split_link_count,
        MAX(
            CASE
                WHEN dataset_role = 'training' AND split_role = 'validation'
                THEN split_set_id
            END
        ) AS linked_validation_split_set_id,
        MAX(
            CASE
                WHEN dataset_role = 'training' AND split_role = 'validation'
                THEN manifest_id
            END
        ) AS linked_validation_manifest_id
    FROM mlops.MODEL_RUN_SPLIT_SET
    GROUP BY model_run_id
)
SELECT
    model_run.model_id,
    model_run.model_run_id,
    package.rate_package_id,
    model_run.manifest_id AS run_manifest_id,
    dataset_links.linked_training_manifest_id,
    dataset_links.training_manifest_link_count,
    split_links.linked_validation_manifest_id,
    split_links.linked_validation_split_set_id,
    split_links.validation_split_link_count,
    CASE
        WHEN model_run.manifest_id IS NULL
         AND COALESCE(dataset_links.training_manifest_link_count, 0) <> 0
        THEN 'UNEXPECTED_TRAINING_MANIFEST_LINK'
        WHEN model_run.manifest_id IS NOT NULL
         AND COALESCE(dataset_links.training_manifest_link_count, 0) <> 1
        THEN 'TRAINING_MANIFEST_LINK_COUNT'
        WHEN dataset_links.linked_training_manifest_id <> model_run.manifest_id
        THEN 'RUN_MANIFEST_LINK_MISMATCH'
        WHEN COALESCE(split_links.validation_split_link_count, 0) > 1
        THEN 'VALIDATION_SPLIT_LINK_COUNT'
        WHEN COALESCE(split_links.validation_split_link_count, 0) = 1
         AND (
                model_run.manifest_id IS NULL
                OR split_links.linked_validation_manifest_id
                    <> model_run.manifest_id
            )
        THEN 'VALIDATION_SPLIT_MANIFEST_MISMATCH'
        ELSE 'OK'
    END AS redundancy_status
FROM pricing.MODEL_RUN AS model_run
JOIN pricing.PRICING_RATE_PACKAGE AS package
  ON package.rate_package_id = model_run.rate_package_id
LEFT JOIN dataset_links
  ON dataset_links.model_run_id = model_run.model_run_id
LEFT JOIN split_links
  ON split_links.model_run_id = model_run.model_run_id;
GO
