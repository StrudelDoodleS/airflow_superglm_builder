IF COL_LENGTH('pricing.DATASET_MANIFEST', 'offset_column') IS NULL
BEGIN
    ALTER TABLE pricing.DATASET_MANIFEST
        ADD offset_column NVARCHAR(128) NULL;
END;
GO

IF COL_LENGTH('pricing.DATASET_MANIFEST', 'offset_source_column') IS NULL
BEGIN
    ALTER TABLE pricing.DATASET_MANIFEST
        ADD offset_source_column NVARCHAR(128) NULL;
END;
GO

IF COL_LENGTH('pricing.DATASET_MANIFEST', 'offset_label') IS NULL
BEGIN
    ALTER TABLE pricing.DATASET_MANIFEST
        ADD offset_label NVARCHAR(512) NULL;
END;
GO

IF COL_LENGTH('pricing.DATASET_MANIFEST', 'export_weight_column') IS NULL
BEGIN
    ALTER TABLE pricing.DATASET_MANIFEST
        ADD export_weight_column NVARCHAR(128) NULL;
END;
GO

IF EXISTS (
    SELECT 1
    FROM sys.columns
    WHERE object_id = OBJECT_ID('pricing.DATASET_COLUMN')
      AND name = 'column_role'
      AND max_length <> -1
      AND max_length < 256
)
BEGIN
    ALTER TABLE pricing.DATASET_COLUMN
        ALTER COLUMN column_role NVARCHAR(128) NOT NULL;
END;
GO

CREATE OR ALTER VIEW pricing.V_CURRENT_DATASET_CV_FOLD
AS
WITH current_manifest AS (
    SELECT
        dm.manifest_id,
        dm.dataset_name,
        dm.source_system,
        dm.data_as_of_date,
        dm.data_as_of_column,
        dm.row_count AS dataset_row_count,
        dm.pk_columns_json,
        dm.target_column,
        dm.weight_column,
        dm.offset_column,
        dm.offset_source_column,
        dm.offset_label,
        dm.export_weight_column,
        dm.model_frame_sha256,
        dm.frame_hash_metadata_json,
        dm.created_ts AS manifest_created_ts,
        dm.created_by AS manifest_created_by,
        ROW_NUMBER() OVER (
            PARTITION BY dataset_name
            ORDER BY dm.created_ts DESC, dm.manifest_id DESC
        ) AS manifest_rank
    FROM pricing.DATASET_MANIFEST dm
),
current_split_set AS (
    SELECT
        ss.split_set_id,
        ss.manifest_id,
        ss.split_mode,
        ss.splitter_class,
        ss.splitter_params_json,
        ss.row_order_sha256,
        ss.row_count AS split_row_count,
        ss.fold_count,
        ss.groups_column,
        ss.stratify_column,
        ss.artifact_uri,
        ss.artifact_sha256,
        ss.runtime_metadata_json,
        ss.created_ts AS split_created_ts,
        ss.created_by AS split_created_by,
        ROW_NUMBER() OVER (
            PARTITION BY manifest_id
            ORDER BY ss.created_ts DESC, ss.split_set_id DESC
        ) AS split_rank
    FROM pricing.CV_SPLIT_SET ss
)
SELECT
    cm.dataset_name,
    cm.manifest_id,
    cm.source_system,
    cm.data_as_of_date,
    cm.data_as_of_column,
    cm.dataset_row_count,
    cm.pk_columns_json,
    cm.target_column,
    cm.weight_column,
    cm.offset_column,
    cm.offset_source_column,
    cm.offset_label,
    cm.export_weight_column,
    cm.model_frame_sha256,
    cm.frame_hash_metadata_json,
    cm.manifest_created_ts,
    cm.manifest_created_by,
    css.split_set_id,
    css.split_mode,
    css.splitter_class,
    css.splitter_params_json,
    css.row_order_sha256,
    css.split_row_count,
    css.fold_count,
    css.groups_column,
    css.stratify_column,
    css.artifact_uri,
    css.artifact_sha256,
    css.runtime_metadata_json,
    css.split_created_ts,
    css.split_created_by,
    fold.fold_no AS split_no,
    CONCAT('[', COALESCE(train_folds.train_folds_csv, ''), ']') AS train_folds_json,
    fold.fold_no AS test_fold_no,
    fold.n_train,
    fold.n_test
FROM current_manifest cm
JOIN current_split_set css
  ON css.manifest_id = cm.manifest_id
 AND css.split_rank = 1
JOIN pricing.CV_FOLD fold
  ON fold.split_set_id = css.split_set_id
OUTER APPLY (
    SELECT
        STRING_AGG(CONVERT(VARCHAR(12), train_fold.fold_no), ',')
            WITHIN GROUP (ORDER BY train_fold.fold_no) AS train_folds_csv
    FROM pricing.CV_FOLD train_fold
    WHERE train_fold.split_set_id = fold.split_set_id
      AND train_fold.fold_no <> fold.fold_no
) train_folds
WHERE cm.manifest_rank = 1;
GO
