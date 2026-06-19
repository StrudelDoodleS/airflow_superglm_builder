IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
    EXEC('CREATE SCHEMA pricing');
GO

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing_stg')
    EXEC('CREATE SCHEMA pricing_stg');
GO

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'publication_receipt_json') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
    ADD publication_receipt_json NVARCHAR(MAX) NULL;
GO

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'publication_receipt_sha256') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
    ADD publication_receipt_sha256 CHAR(64) NULL;
GO

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'package_metadata_json') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
    ADD package_metadata_json NVARCHAR(MAX) NULL;
GO

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'revision_metadata_json') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
    ADD revision_metadata_json NVARCHAR(MAX) NULL;
GO

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'offset_handling') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
    ADD offset_handling NVARCHAR(64) NOT NULL
        CONSTRAINT DF_PRICING_RATE_PACKAGE_OFFSET_HANDLING DEFAULT 'UNKNOWN';
GO

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'offset_factor_name') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
    ADD offset_factor_name NVARCHAR(256) NULL;
GO

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'offset_source_name') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
    ADD offset_source_name NVARCHAR(256) NULL;
GO

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'offset_label') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
    ADD offset_label NVARCHAR(1024) NULL;
GO

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'metadata_origin') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
    ADD metadata_origin NVARCHAR(128) NULL;
GO

UPDATE pricing.PRICING_RATE_PACKAGE
SET offset_handling = 'UNKNOWN'
WHERE offset_handling IS NULL;
GO

IF EXISTS (
    SELECT 1
    FROM sys.columns
    WHERE object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
      AND name = 'offset_handling'
      AND is_nullable = 1
)
ALTER TABLE pricing.PRICING_RATE_PACKAGE
ALTER COLUMN offset_handling NVARCHAR(64) NOT NULL;
GO

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'offset_handling') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.default_constraints dc
    JOIN sys.columns c
      ON c.object_id = dc.parent_object_id
     AND c.column_id = dc.parent_column_id
    WHERE dc.parent_object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
      AND c.name = 'offset_handling'
)
ALTER TABLE pricing.PRICING_RATE_PACKAGE
ADD CONSTRAINT DF_PRICING_RATE_PACKAGE_OFFSET_HANDLING
    DEFAULT 'UNKNOWN' FOR offset_handling;
GO

IF COL_LENGTH('pricing.PRICING_TERM', 'term_metadata_json') IS NULL
    ALTER TABLE pricing.PRICING_TERM
    ADD term_metadata_json NVARCHAR(MAX) NULL;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'publication_receipt_path') IS NULL
    ALTER TABLE pricing.MODEL_RUN
    ADD publication_receipt_path NVARCHAR(1024) NULL;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'publication_receipt_sha256') IS NULL
    ALTER TABLE pricing.MODEL_RUN
    ADD publication_receipt_sha256 CHAR(64) NULL;
GO

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'publication_receipt_json') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
    ADD publication_receipt_json NVARCHAR(MAX) NULL;
GO

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'publication_receipt_sha256') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
    ADD publication_receipt_sha256 CHAR(64) NULL;
GO

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'package_metadata_json') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
    ADD package_metadata_json NVARCHAR(MAX) NULL;
GO

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'offset_handling') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
    ADD offset_handling NVARCHAR(64) NULL;
GO

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'offset_factor_name') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
    ADD offset_factor_name NVARCHAR(256) NULL;
GO

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'offset_source_name') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
    ADD offset_source_name NVARCHAR(256) NULL;
GO

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'offset_label') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
    ADD offset_label NVARCHAR(1024) NULL;
GO

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'metadata_origin') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
    ADD metadata_origin NVARCHAR(128) NULL;
GO

IF OBJECT_ID('pricing_stg.STG_TERM_METADATA', 'U') IS NULL
CREATE TABLE pricing_stg.STG_TERM_METADATA (
    export_id NVARCHAR(128) NOT NULL,
    term_name NVARCHAR(256) NOT NULL,
    term_metadata_json NVARCHAR(MAX) NOT NULL,
    CONSTRAINT PK_STG_TERM_METADATA PRIMARY KEY (export_id, term_name),
    CONSTRAINT FK_STG_TERM_METADATA_EXPORT FOREIGN KEY (export_id)
        REFERENCES pricing_stg.STG_RATING_EXPORT(export_id),
    CONSTRAINT CK_STG_TERM_METADATA_JSON CHECK (ISJSON(term_metadata_json) = 1)
);
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_PRICING_RATE_PACKAGE_OFFSET_HANDLING'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
)
ALTER TABLE pricing.PRICING_RATE_PACKAGE WITH CHECK
ADD CONSTRAINT CK_PRICING_RATE_PACKAGE_OFFSET_HANDLING
    CHECK (
        offset_handling IN (
            'NONE',
            'EXPORTED_FACTOR',
            'ALREADY_APPLIED_SQL_EXPOSURE',
            'UNKNOWN'
        )
    );
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_PRICING_RATE_PACKAGE_RECEIPT_JSON'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
)
ALTER TABLE pricing.PRICING_RATE_PACKAGE WITH CHECK
ADD CONSTRAINT CK_PRICING_RATE_PACKAGE_RECEIPT_JSON
    CHECK (publication_receipt_json IS NULL OR ISJSON(publication_receipt_json) = 1);
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_PRICING_RATE_PACKAGE_METADATA_JSON'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
)
ALTER TABLE pricing.PRICING_RATE_PACKAGE WITH CHECK
ADD CONSTRAINT CK_PRICING_RATE_PACKAGE_METADATA_JSON
    CHECK (package_metadata_json IS NULL OR ISJSON(package_metadata_json) = 1);
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_PRICING_RATE_PACKAGE_REVISION_METADATA_JSON'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
)
ALTER TABLE pricing.PRICING_RATE_PACKAGE WITH CHECK
ADD CONSTRAINT CK_PRICING_RATE_PACKAGE_REVISION_METADATA_JSON
    CHECK (revision_metadata_json IS NULL OR ISJSON(revision_metadata_json) = 1);
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_PRICING_RATE_PACKAGE_PUBLICATION_RECEIPT_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
)
ALTER TABLE pricing.PRICING_RATE_PACKAGE WITH CHECK
ADD CONSTRAINT CK_PRICING_RATE_PACKAGE_PUBLICATION_RECEIPT_SHA256
    CHECK (
        publication_receipt_sha256 IS NULL
        OR (
            LEN(publication_receipt_sha256) = 64
            AND publication_receipt_sha256 COLLATE Latin1_General_BIN2 NOT LIKE '%[^0-9a-f]%'
        )
    );
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_MODEL_RUN_PUBLICATION_RECEIPT_SHA256'
      AND parent_object_id = OBJECT_ID('pricing.MODEL_RUN')
)
ALTER TABLE pricing.MODEL_RUN WITH CHECK
ADD CONSTRAINT CK_MODEL_RUN_PUBLICATION_RECEIPT_SHA256
    CHECK (
        publication_receipt_sha256 IS NULL
        OR (
            LEN(publication_receipt_sha256) = 64
            AND publication_receipt_sha256 COLLATE Latin1_General_BIN2 NOT LIKE '%[^0-9a-f]%'
        )
    );
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_STG_RATING_EXPORT_PUBLICATION_RECEIPT_SHA256'
      AND parent_object_id = OBJECT_ID('pricing_stg.STG_RATING_EXPORT')
)
ALTER TABLE pricing_stg.STG_RATING_EXPORT WITH CHECK
ADD CONSTRAINT CK_STG_RATING_EXPORT_PUBLICATION_RECEIPT_SHA256
    CHECK (
        publication_receipt_sha256 IS NULL
        OR (
            LEN(publication_receipt_sha256) = 64
            AND publication_receipt_sha256 COLLATE Latin1_General_BIN2 NOT LIKE '%[^0-9a-f]%'
        )
    );
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_PRICING_TERM_METADATA_JSON'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_TERM')
)
ALTER TABLE pricing.PRICING_TERM WITH CHECK
ADD CONSTRAINT CK_PRICING_TERM_METADATA_JSON
    CHECK (term_metadata_json IS NULL OR ISJSON(term_metadata_json) = 1);
GO
