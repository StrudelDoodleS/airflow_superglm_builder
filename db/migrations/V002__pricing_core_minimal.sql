IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
    EXEC('CREATE SCHEMA pricing');
GO

IF OBJECT_ID('pricing.PRICING_RATE_PACKAGE', 'U') IS NULL
CREATE TABLE pricing.PRICING_RATE_PACKAGE (
    rate_package_id        BIGINT IDENTITY(1,1) PRIMARY KEY,
    parent_rate_package_id BIGINT NULL,
    model_name             NVARCHAR(128) NOT NULL,
    model_version          NVARCHAR(64) NULL,
    package_version        INT NOT NULL,
    base_rate              DECIMAL(19,6) NOT NULL,
    effective_from_date    DATE NOT NULL,
    effective_to_date      DATE NULL,
    package_status         NVARCHAR(32) NOT NULL,
    created_ts             DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    created_by             NVARCHAR(128) NOT NULL,

    CONSTRAINT FK_RATE_PACKAGE_PARENT
        FOREIGN KEY (parent_rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id)
);
GO

IF OBJECT_ID('pricing.PRICING_PACKAGE_POINTER', 'U') IS NULL
CREATE TABLE pricing.PRICING_PACKAGE_POINTER (
    pointer_name      NVARCHAR(128) PRIMARY KEY,
    rate_package_id   BIGINT NOT NULL,
    updated_ts        DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    updated_by        NVARCHAR(128) NOT NULL,

    CONSTRAINT FK_PACKAGE_POINTER_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id)
);
GO

IF OBJECT_ID('pricing.PRICING_FEATURE', 'U') IS NULL
CREATE TABLE pricing.PRICING_FEATURE (
    feature_id          BIGINT IDENTITY(1,1) PRIMARY KEY,
    feature_name        NVARCHAR(128) NOT NULL UNIQUE,
    feature_value_type  NVARCHAR(32) NOT NULL,
    is_ordered          BIT NOT NULL DEFAULT 0,
    active_flag         BIT NOT NULL DEFAULT 1
);
GO

IF OBJECT_ID('pricing.PRICING_FEATURE_LEVEL_SET', 'U') IS NULL
CREATE TABLE pricing.PRICING_FEATURE_LEVEL_SET (
    level_set_id        BIGINT IDENTITY(1,1) PRIMARY KEY,
    feature_id          BIGINT NOT NULL,
    level_set_name      NVARCHAR(128) NOT NULL,
    level_set_type      NVARCHAR(64) NOT NULL,
    binning_strategy    NVARCHAR(64) NULL,
    grid_width          FLOAT NULL,
    created_ts          DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),

    CONSTRAINT FK_LEVEL_SET_FEATURE
        FOREIGN KEY (feature_id)
        REFERENCES pricing.PRICING_FEATURE(feature_id),

    CONSTRAINT UQ_LEVEL_SET
        UNIQUE (feature_id, level_set_name)
);
GO

IF OBJECT_ID('pricing.PRICING_FEATURE_LEVEL', 'U') IS NULL
CREATE TABLE pricing.PRICING_FEATURE_LEVEL (
    feature_level_id      BIGINT IDENTITY(1,1) PRIMARY KEY,
    level_set_id          BIGINT NOT NULL,
    level_code            NVARCHAR(128) NOT NULL,
    level_label           NVARCHAR(256) NULL,
    order_index           INT NULL,
    lower_bound           FLOAT NULL,
    upper_bound           FLOAT NULL,
    representative_value  FLOAT NULL,
    is_missing            BIT NOT NULL DEFAULT 0,
    is_other              BIT NOT NULL DEFAULT 0,

    CONSTRAINT FK_FEATURE_LEVEL_SET
        FOREIGN KEY (level_set_id)
        REFERENCES pricing.PRICING_FEATURE_LEVEL_SET(level_set_id),

    CONSTRAINT UQ_FEATURE_LEVEL
        UNIQUE (level_set_id, level_code)
);
GO

IF OBJECT_ID('pricing.PRICING_TERM', 'U') IS NULL
CREATE TABLE pricing.PRICING_TERM (
    term_id                  BIGINT IDENTITY(1,1) PRIMARY KEY,
    rate_package_id          BIGINT NOT NULL,
    term_name                NVARCHAR(128) NOT NULL,
    term_type                NVARCHAR(64) NOT NULL,
    sequence_no              INT NOT NULL,
    default_multiplier       DECIMAL(19,10) NOT NULL DEFAULT 1.0,
    default_log_coefficient  DECIMAL(19,12) NOT NULL DEFAULT 0.0,
    active_flag              BIT NOT NULL DEFAULT 1,

    CONSTRAINT FK_TERM_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id),

    CONSTRAINT UQ_TERM_PACKAGE_NAME
        UNIQUE (rate_package_id, term_name)
);
GO

IF OBJECT_ID('pricing.PRICING_TERM_FEATURE', 'U') IS NULL
CREATE TABLE pricing.PRICING_TERM_FEATURE (
    term_id           BIGINT NOT NULL,
    position_no       SMALLINT NOT NULL,
    feature_id        BIGINT NOT NULL,
    level_set_id      BIGINT NOT NULL,
    input_column_name NVARCHAR(128) NULL,

    CONSTRAINT PK_TERM_FEATURE
        PRIMARY KEY (term_id, position_no),

    CONSTRAINT FK_TERM_FEATURE_TERM
        FOREIGN KEY (term_id)
        REFERENCES pricing.PRICING_TERM(term_id),

    CONSTRAINT FK_TERM_FEATURE_FEATURE
        FOREIGN KEY (feature_id)
        REFERENCES pricing.PRICING_FEATURE(feature_id),

    CONSTRAINT FK_TERM_FEATURE_LEVEL_SET
        FOREIGN KEY (level_set_id)
        REFERENCES pricing.PRICING_FEATURE_LEVEL_SET(level_set_id)
);
GO

IF OBJECT_ID('pricing.PRICING_RATE_CELL', 'U') IS NULL
CREATE TABLE pricing.PRICING_RATE_CELL (
    cell_id              BIGINT IDENTITY(1,1) PRIMARY KEY,
    term_id              BIGINT NOT NULL,
    cell_key_text        NVARCHAR(900) NOT NULL,
    cell_key_digest      VARBINARY(32) NOT NULL,
    multiplier           DECIMAL(19,10) NOT NULL,
    log_coefficient      DECIMAL(19,12) NOT NULL,
    exposure_weight      DECIMAL(19,4) NULL,
    record_count         BIGINT NULL,
    is_reference         BIT NOT NULL DEFAULT 0,
    is_default           BIT NOT NULL DEFAULT 0,
    is_deleted           BIT NOT NULL DEFAULT 0,

    CONSTRAINT FK_RATE_CELL_TERM
        FOREIGN KEY (term_id)
        REFERENCES pricing.PRICING_TERM(term_id),

    CONSTRAINT UQ_RATE_CELL
        UNIQUE (term_id, cell_key_digest)
);
GO

IF OBJECT_ID('pricing.PRICING_RATE_CELL_LEVEL', 'U') IS NULL
CREATE TABLE pricing.PRICING_RATE_CELL_LEVEL (
    cell_id            BIGINT NOT NULL,
    position_no        SMALLINT NOT NULL,
    feature_level_id   BIGINT NOT NULL,

    CONSTRAINT PK_RATE_CELL_LEVEL
        PRIMARY KEY (cell_id, position_no),

    CONSTRAINT FK_RATE_CELL_LEVEL_CELL
        FOREIGN KEY (cell_id)
        REFERENCES pricing.PRICING_RATE_CELL(cell_id),

    CONSTRAINT FK_RATE_CELL_LEVEL_LEVEL
        FOREIGN KEY (feature_level_id)
        REFERENCES pricing.PRICING_FEATURE_LEVEL(feature_level_id)
);
GO
