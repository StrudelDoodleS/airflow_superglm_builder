DECLARE @old_error_keyword NVARCHAR(16) = N'TH' + N'ROW';
DECLARE @old_raiserror_keyword NVARCHAR(16) = N'RAIS' + N'ERROR';
DECLARE @object_name SYSNAME;
DECLARE @sql NVARCHAR(MAX);

DECLARE @guard_modules TABLE (
    object_name SYSNAME NOT NULL PRIMARY KEY,
    create_statement NVARCHAR(512) NOT NULL
);

INSERT INTO @guard_modules(object_name, create_statement)
VALUES
    (N'pricing.PREDICT_CURRENT_RATE', N'CREATE OR ALTER PROCEDURE pricing.PREDICT_CURRENT_RATE'),
    (N'pricing.TR_PRICING_RATE_PACKAGE_IMMUTABLE_UPDATE_DELETE', N'CREATE OR ALTER TRIGGER pricing.TR_PRICING_RATE_PACKAGE_IMMUTABLE_UPDATE_DELETE'),
    (N'pricing.TR_PRICING_TERM_IMMUTABLE_WRITE', N'CREATE OR ALTER TRIGGER pricing.TR_PRICING_TERM_IMMUTABLE_WRITE'),
    (N'pricing.TR_PRICING_TERM_FEATURE_IMMUTABLE_WRITE', N'CREATE OR ALTER TRIGGER pricing.TR_PRICING_TERM_FEATURE_IMMUTABLE_WRITE'),
    (N'pricing.TR_PRICING_RATE_CELL_IMMUTABLE_WRITE', N'CREATE OR ALTER TRIGGER pricing.TR_PRICING_RATE_CELL_IMMUTABLE_WRITE'),
    (N'pricing.TR_PRICING_RATE_CELL_LEVEL_IMMUTABLE_WRITE', N'CREATE OR ALTER TRIGGER pricing.TR_PRICING_RATE_CELL_LEVEL_IMMUTABLE_WRITE'),
    (N'pricing.TR_PRICING_FEATURE_LEVEL_IMMUTABLE_WRITE', N'CREATE OR ALTER TRIGGER pricing.TR_PRICING_FEATURE_LEVEL_IMMUTABLE_WRITE'),
    (N'pricing.TR_PRICING_FEATURE_LEVEL_SET_IMMUTABLE_WRITE', N'CREATE OR ALTER TRIGGER pricing.TR_PRICING_FEATURE_LEVEL_SET_IMMUTABLE_WRITE'),
    (N'pricing.TR_PRICING_FEATURE_IMMUTABLE_WRITE', N'CREATE OR ALTER TRIGGER pricing.TR_PRICING_FEATURE_IMMUTABLE_WRITE'),
    (N'pricing.TR_PRICING_COMPILED_RATE_CELL_IMMUTABLE_WRITE', N'CREATE OR ALTER TRIGGER pricing.TR_PRICING_COMPILED_RATE_CELL_IMMUTABLE_WRITE'),
    (N'pricing.TR_PRICING_COMPILED_1D_RATE_BAND_IMMUTABLE_WRITE', N'CREATE OR ALTER TRIGGER pricing.TR_PRICING_COMPILED_1D_RATE_BAND_IMMUTABLE_WRITE'),
    (N'pricing.TR_PRICING_MODEL_DEPLOYMENT_PACKAGE_GUARD', N'CREATE OR ALTER TRIGGER pricing.TR_PRICING_MODEL_DEPLOYMENT_PACKAGE_GUARD');

DECLARE guard_module_cursor CURSOR LOCAL FAST_FORWARD FOR
SELECT object_name
FROM @guard_modules;

OPEN guard_module_cursor;
FETCH NEXT FROM guard_module_cursor INTO @object_name;

WHILE @@FETCH_STATUS = 0
BEGIN
    SET @sql = OBJECT_DEFINITION(OBJECT_ID(@object_name));

    IF @sql IS NOT NULL
    BEGIN
        SET @sql = REPLACE(
            @sql,
            @old_raiserror_keyword + N'(''features_json must be valid JSON'', 16, 1);
        RETURN;',
            N';THROW 50000, ''features_json must be valid JSON'', 1;'
        );
        SET @sql = REPLACE(
            @sql,
            @old_raiserror_keyword + N'(''exposure must be positive'', 16, 1);
        RETURN;',
            N';THROW 50001, ''exposure must be positive'', 1;'
        );
        SET @sql = REPLACE(
            @sql,
            @old_raiserror_keyword + N'(''No current deployed rate package found'', 16, 1);
        RETURN;',
            N';THROW 50002, ''No current deployed rate package found'', 1;'
        );
        SET @sql = REPLACE(
            @sql,
            @old_raiserror_keyword + N'(''Input features did not match every required term'', 16, 1);
        RETURN;',
            N';THROW 50003, ''Input features did not match every required term'', 1;'
        );
        SET @sql = REPLACE(
            @sql,
            @old_raiserror_keyword + N'(''Immutable rate packages cannot be changed directly. Create a new package revision.'', 16, 1);
        ROLLBACK TRANSACTION;
        RETURN;',
            N';THROW 51000, ''Immutable rate packages cannot be changed directly. Create a new package revision.'', 1;'
        );
        SET @sql = REPLACE(
            @sql,
            @old_raiserror_keyword + N'(''rate package deployments must reference PUBLISHED packages for the same model_id.'', 16, 1);
        ROLLBACK TRANSACTION;
        RETURN;',
            N';THROW 51001, ''rate package deployments must reference PUBLISHED packages for the same model_id.'', 1;'
        );
        SET @sql = REPLACE(
            @sql,
            @old_error_keyword + N' 50000, ''features_json must be valid JSON'', 1;',
            N';THROW 50000, ''features_json must be valid JSON'', 1;'
        );
        SET @sql = REPLACE(
            @sql,
            @old_error_keyword + N' 50001, ''exposure must be positive'', 1;',
            N';THROW 50001, ''exposure must be positive'', 1;'
        );
        SET @sql = REPLACE(
            @sql,
            @old_error_keyword + N' 50002, ''No current deployed rate package found'', 1;',
            N';THROW 50002, ''No current deployed rate package found'', 1;'
        );
        SET @sql = REPLACE(
            @sql,
            @old_error_keyword + N' 50003, ''Input features did not match every required term'', 1;',
            N';THROW 50003, ''Input features did not match every required term'', 1;'
        );
        SET @sql = REPLACE(
            @sql,
            @old_error_keyword + N' 51000, ''Immutable rate packages cannot be changed directly. Create a new package revision.'', 1;',
            N';THROW 51000, ''Immutable rate packages cannot be changed directly. Create a new package revision.'', 1;'
        );
        SET @sql = REPLACE(
            @sql,
            @old_error_keyword + N' 51001, ''rate package deployments must reference PUBLISHED packages for the same model_id.'', 1;',
            N';THROW 51001, ''rate package deployments must reference PUBLISHED packages for the same model_id.'', 1;'
        );

        EXEC sys.sp_executesql @sql;
    END;

    FETCH NEXT FROM guard_module_cursor INTO @object_name;
END;

CLOSE guard_module_cursor;
DEALLOCATE guard_module_cursor;
GO
