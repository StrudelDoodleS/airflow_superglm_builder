IF COL_LENGTH('pricing.MODEL_RUN', 'rating_workbook_sha256') IS NULL
BEGIN
    ALTER TABLE pricing.MODEL_RUN
    ADD rating_workbook_sha256 CHAR(64) NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_MODEL_RUN_RATING_WORKBOOK_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
BEGIN
    ALTER TABLE pricing.MODEL_RUN WITH CHECK
    ADD CONSTRAINT CK_MODEL_RUN_RATING_WORKBOOK_SHA256 CHECK (
        rating_workbook_sha256 IS NULL
        OR (
            LEN(rating_workbook_sha256) = 64
            AND rating_workbook_sha256 COLLATE Latin1_General_BIN2
                NOT LIKE '%[^0-9a-f]%'
        )
    );
END;
GO
