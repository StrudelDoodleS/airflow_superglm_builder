IF COL_LENGTH('pricing.MODEL_RUN', 'parent_model_run_id') IS NULL
BEGIN
    ALTER TABLE pricing.MODEL_RUN
    ADD parent_model_run_id BIGINT NULL;
END;
GO

UPDATE child_run
SET parent_model_run_id = parent_run.model_run_id
FROM pricing.MODEL_RUN AS child_run
JOIN pricing.PRICING_RATE_PACKAGE AS child_package
  ON child_package.rate_package_id = child_run.rate_package_id
JOIN pricing.MODEL_RUN AS parent_run
  ON parent_run.rate_package_id = child_package.parent_rate_package_id
WHERE child_run.parent_model_run_id IS NULL
  AND child_package.parent_rate_package_id IS NOT NULL;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = 'FK_MODEL_RUN_PARENT'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
BEGIN
    ALTER TABLE pricing.MODEL_RUN WITH CHECK
    ADD CONSTRAINT FK_MODEL_RUN_PARENT
        FOREIGN KEY (parent_model_run_id)
        REFERENCES pricing.MODEL_RUN(model_run_id);
END;
GO
