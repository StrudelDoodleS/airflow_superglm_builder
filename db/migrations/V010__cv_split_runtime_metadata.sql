IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
    EXEC('CREATE SCHEMA pricing');
GO

IF COL_LENGTH('pricing.CV_SPLIT_SET', 'runtime_metadata_json') IS NULL
    ALTER TABLE pricing.CV_SPLIT_SET
    ADD runtime_metadata_json NVARCHAR(MAX) NULL;
GO
