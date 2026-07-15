IF COL_LENGTH('pricing.MODEL_RUN', 'candidate_artifact_path') IS NULL
    ALTER TABLE pricing.MODEL_RUN ADD candidate_artifact_path NVARCHAR(1024) NULL;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'candidate_artifact_sha256') IS NULL
    ALTER TABLE pricing.MODEL_RUN ADD candidate_artifact_sha256 NVARCHAR(64) NULL;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'candidate_artifact_format') IS NULL
    ALTER TABLE pricing.MODEL_RUN ADD candidate_artifact_format NVARCHAR(64) NULL;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'candidate_artifact_size_bytes') IS NULL
    ALTER TABLE pricing.MODEL_RUN ADD candidate_artifact_size_bytes BIGINT NULL;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'candidate_python_version') IS NULL
    ALTER TABLE pricing.MODEL_RUN ADD candidate_python_version NVARCHAR(64) NULL;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'candidate_superglm_version') IS NULL
    ALTER TABLE pricing.MODEL_RUN ADD candidate_superglm_version NVARCHAR(64) NULL;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'model_source_sha256') IS NULL
    ALTER TABLE pricing.MODEL_RUN ADD model_source_sha256 NVARCHAR(64) NULL;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_MODEL_RUN_CANDIDATE_ARTIFACT_FIELDS'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
ALTER TABLE pricing.MODEL_RUN WITH CHECK
ADD CONSTRAINT CK_MODEL_RUN_CANDIDATE_ARTIFACT_FIELDS
CHECK (
    (
        candidate_artifact_path IS NULL
        AND candidate_artifact_sha256 IS NULL
        AND candidate_artifact_format IS NULL
        AND candidate_artifact_size_bytes IS NULL
        AND candidate_python_version IS NULL
        AND candidate_superglm_version IS NULL
        AND model_source_sha256 IS NULL
    )
    OR
    (
        candidate_artifact_path IS NOT NULL
        AND candidate_artifact_sha256 IS NOT NULL
        AND candidate_artifact_format IS NOT NULL
        AND candidate_artifact_size_bytes > 0
        AND candidate_python_version IS NOT NULL
        AND candidate_superglm_version IS NOT NULL
        AND model_source_sha256 IS NOT NULL
    )
);
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_MODEL_RUN_CANDIDATE_ARTIFACT_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
ALTER TABLE pricing.MODEL_RUN WITH CHECK
ADD CONSTRAINT CK_MODEL_RUN_CANDIDATE_ARTIFACT_SHA256
CHECK (
    candidate_artifact_sha256 IS NULL
    OR (
        LEN(candidate_artifact_sha256) = 64
        AND candidate_artifact_sha256 COLLATE Latin1_General_BIN2 NOT LIKE '%[^0-9a-f]%'
    )
);
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_MODEL_RUN_SOURCE_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
ALTER TABLE pricing.MODEL_RUN WITH CHECK
ADD CONSTRAINT CK_MODEL_RUN_SOURCE_SHA256
CHECK (
    model_source_sha256 IS NULL
    OR (
        LEN(model_source_sha256) = 64
        AND model_source_sha256 COLLATE Latin1_General_BIN2 NOT LIKE '%[^0-9a-f]%'
    )
);
GO

IF EXISTS (
    SELECT rate_package_id
    FROM pricing.MODEL_RUN
    WHERE rate_package_id IS NOT NULL
    GROUP BY rate_package_id
    HAVING COUNT_BIG(*) > 1
)
BEGIN
    THROW 51024, 'Cannot enforce one MODEL_RUN per rate package while duplicates exist', 1;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_MODEL_RUN_RATE_PACKAGE'
      AND object_id = OBJECT_ID('pricing.MODEL_RUN')
)
CREATE UNIQUE INDEX UX_MODEL_RUN_RATE_PACKAGE
ON pricing.MODEL_RUN(rate_package_id)
WHERE rate_package_id IS NOT NULL;
GO
