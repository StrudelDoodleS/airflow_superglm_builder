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
