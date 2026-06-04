CREATE OR ALTER TRIGGER pricing.TR_PRICING_RATE_PACKAGE_IMMUTABLE_UPDATE_DELETE
ON pricing.PRICING_RATE_PACKAGE
AFTER UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM deleted d
        WHERE d.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.PRICING_MODEL_DEPLOYMENT md
               WHERE md.rate_package_id = d.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_PRICING_TERM_IMMUTABLE_WRITE
ON pricing.PRICING_TERM
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM (
            SELECT rate_package_id FROM inserted
            UNION
            SELECT rate_package_id FROM deleted
        ) changed
        JOIN pricing.PRICING_RATE_PACKAGE rp
          ON rp.rate_package_id = changed.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.PRICING_MODEL_DEPLOYMENT md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_PRICING_TERM_FEATURE_IMMUTABLE_WRITE
ON pricing.PRICING_TERM_FEATURE
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM (
            SELECT term_id FROM inserted
            UNION
            SELECT term_id FROM deleted
        ) changed
        JOIN pricing.PRICING_TERM t
          ON t.term_id = changed.term_id
        JOIN pricing.PRICING_RATE_PACKAGE rp
          ON rp.rate_package_id = t.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.PRICING_MODEL_DEPLOYMENT md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_PRICING_RATE_CELL_IMMUTABLE_WRITE
ON pricing.PRICING_RATE_CELL
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM (
            SELECT term_id FROM inserted
            UNION
            SELECT term_id FROM deleted
        ) changed
        JOIN pricing.PRICING_TERM t
          ON t.term_id = changed.term_id
        JOIN pricing.PRICING_RATE_PACKAGE rp
          ON rp.rate_package_id = t.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.PRICING_MODEL_DEPLOYMENT md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_PRICING_RATE_CELL_LEVEL_IMMUTABLE_WRITE
ON pricing.PRICING_RATE_CELL_LEVEL
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM (
            SELECT cell_id FROM inserted
            UNION
            SELECT cell_id FROM deleted
        ) changed
        JOIN pricing.PRICING_RATE_CELL rc
          ON rc.cell_id = changed.cell_id
        JOIN pricing.PRICING_TERM t
          ON t.term_id = rc.term_id
        JOIN pricing.PRICING_RATE_PACKAGE rp
          ON rp.rate_package_id = t.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.PRICING_MODEL_DEPLOYMENT md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_PRICING_FEATURE_LEVEL_IMMUTABLE_WRITE
ON pricing.PRICING_FEATURE_LEVEL
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM (
            SELECT level_set_id FROM inserted
            UNION
            SELECT level_set_id FROM deleted
        ) changed
        JOIN pricing.PRICING_TERM_FEATURE tf
          ON tf.level_set_id = changed.level_set_id
        JOIN pricing.PRICING_TERM t
          ON t.term_id = tf.term_id
        JOIN pricing.PRICING_RATE_PACKAGE rp
          ON rp.rate_package_id = t.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.PRICING_MODEL_DEPLOYMENT md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_PRICING_FEATURE_LEVEL_SET_IMMUTABLE_WRITE
ON pricing.PRICING_FEATURE_LEVEL_SET
AFTER UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM deleted changed
        JOIN pricing.PRICING_FEATURE_LEVEL fl
          ON fl.level_set_id = changed.level_set_id
        JOIN pricing.PRICING_RATE_CELL_LEVEL rcl
          ON rcl.feature_level_id = fl.feature_level_id
        JOIN pricing.PRICING_RATE_CELL rc
          ON rc.cell_id = rcl.cell_id
        JOIN pricing.PRICING_TERM t
          ON t.term_id = rc.term_id
        JOIN pricing.PRICING_RATE_PACKAGE rp
          ON rp.rate_package_id = t.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.PRICING_MODEL_DEPLOYMENT md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_PRICING_FEATURE_IMMUTABLE_WRITE
ON pricing.PRICING_FEATURE
AFTER UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM deleted changed
        JOIN pricing.PRICING_FEATURE_LEVEL_SET fls
          ON fls.feature_id = changed.feature_id
        JOIN pricing.PRICING_FEATURE_LEVEL fl
          ON fl.level_set_id = fls.level_set_id
        JOIN pricing.PRICING_RATE_CELL_LEVEL rcl
          ON rcl.feature_level_id = fl.feature_level_id
        JOIN pricing.PRICING_RATE_CELL rc
          ON rc.cell_id = rcl.cell_id
        JOIN pricing.PRICING_TERM t
          ON t.term_id = rc.term_id
        JOIN pricing.PRICING_RATE_PACKAGE rp
          ON rp.rate_package_id = t.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.PRICING_MODEL_DEPLOYMENT md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_PRICING_COMPILED_RATE_CELL_IMMUTABLE_WRITE
ON pricing.PRICING_COMPILED_RATE_CELL
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM (
            SELECT rate_package_id FROM inserted
            UNION
            SELECT rate_package_id FROM deleted
        ) changed
        JOIN pricing.PRICING_RATE_PACKAGE rp
          ON rp.rate_package_id = changed.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.PRICING_MODEL_DEPLOYMENT md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_PRICING_COMPILED_1D_RATE_BAND_IMMUTABLE_WRITE
ON pricing.PRICING_COMPILED_1D_RATE_BAND
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM (
            SELECT rate_package_id FROM inserted
            UNION
            SELECT rate_package_id FROM deleted
        ) changed
        JOIN pricing.PRICING_RATE_PACKAGE rp
          ON rp.rate_package_id = changed.rate_package_id
        WHERE rp.package_status <> 'DRAFT'
           OR EXISTS (
               SELECT 1
               FROM pricing.PRICING_MODEL_DEPLOYMENT md
               WHERE md.rate_package_id = rp.rate_package_id
           )
    )
    BEGIN;
        THROW 51000, 'Immutable rate packages cannot be changed directly. Create a new package revision.', 1;
    END;
END;
GO
