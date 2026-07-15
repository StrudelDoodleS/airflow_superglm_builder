IF COL_LENGTH('pricing.DATASET_MANIFEST', 'model_frame_sha256') IS NULL
BEGIN
    ALTER TABLE pricing.DATASET_MANIFEST
        ADD model_frame_sha256 CHAR(64) NULL;
END;
GO

IF COL_LENGTH('pricing.DATASET_MANIFEST', 'frame_hash_metadata_json') IS NULL
BEGIN
    ALTER TABLE pricing.DATASET_MANIFEST
        ADD frame_hash_metadata_json NVARCHAR(MAX) NULL;
END;
GO

IF COL_LENGTH('pricing.DATASET_MANIFEST', 'exposure_column') IS NULL
BEGIN
    ALTER TABLE pricing.DATASET_MANIFEST
        ADD exposure_column NVARCHAR(128) NULL;
END;
GO

IF COL_LENGTH('pricing.DATASET_MANIFEST', 'data_as_of_column') IS NULL
BEGIN
    ALTER TABLE pricing.DATASET_MANIFEST
        ADD data_as_of_column NVARCHAR(128) NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_DATASET_MANIFEST_MODEL_FRAME_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.DATASET_MANIFEST')
)
BEGIN
    ALTER TABLE pricing.DATASET_MANIFEST WITH CHECK
        ADD CONSTRAINT CK_DATASET_MANIFEST_MODEL_FRAME_SHA256
        CHECK (
            model_frame_sha256 IS NULL
            OR (
                LEN(model_frame_sha256) = 64
                AND model_frame_sha256 COLLATE Latin1_General_BIN2
                    NOT LIKE '%[^0-9a-f]%'
            )
        );
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_DATASET_MANIFEST_FRAME_HASH_METADATA_JSON'
      AND parent_object_id = OBJECT_ID('pricing.DATASET_MANIFEST')
)
BEGIN
    ALTER TABLE pricing.DATASET_MANIFEST WITH CHECK
        ADD CONSTRAINT CK_DATASET_MANIFEST_FRAME_HASH_METADATA_JSON
        CHECK (
            frame_hash_metadata_json IS NULL
            OR ISJSON(frame_hash_metadata_json) = 1
        );
END;
GO
