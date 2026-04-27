IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
    EXEC('CREATE SCHEMA pricing');
GO

IF OBJECT_ID('pricing.PRICING_COMPILED_RATE_CELL', 'U') IS NULL
CREATE TABLE pricing.PRICING_COMPILED_RATE_CELL (
    rate_package_id    BIGINT NOT NULL,
    term_id            BIGINT NOT NULL,
    cell_key_digest    VARBINARY(32) NOT NULL,
    term_name          NVARCHAR(128) NOT NULL,
    term_type          NVARCHAR(64) NOT NULL,
    sequence_no        INT NOT NULL,
    cell_key_text      NVARCHAR(900) NOT NULL,
    multiplier         DECIMAL(19,10) NOT NULL,
    log_coefficient    DECIMAL(19,12) NOT NULL,
    exposure_weight    DECIMAL(19,4) NULL,
    record_count       BIGINT NULL,
    is_default         BIT NOT NULL,
    is_reference       BIT NOT NULL,

    CONSTRAINT PK_COMPILED_RATE_CELL
        PRIMARY KEY (rate_package_id, term_id, cell_key_digest),

    CONSTRAINT FK_COMPILED_RATE_CELL_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id),

    CONSTRAINT FK_COMPILED_RATE_CELL_TERM
        FOREIGN KEY (term_id)
        REFERENCES pricing.PRICING_TERM(term_id)
);
GO

IF OBJECT_ID('pricing.PRICING_COMPILED_1D_RATE_BAND', 'U') IS NULL
CREATE TABLE pricing.PRICING_COMPILED_1D_RATE_BAND (
    rate_package_id       BIGINT NOT NULL,
    term_id               BIGINT NOT NULL,
    feature_level_id      BIGINT NOT NULL,
    term_name             NVARCHAR(128) NOT NULL,
    feature_name          NVARCHAR(128) NOT NULL,
    level_code            NVARCHAR(128) NOT NULL,
    lower_bound           FLOAT NULL,
    upper_bound           FLOAT NULL,
    representative_value  FLOAT NULL,
    multiplier            DECIMAL(19,10) NOT NULL,
    log_coefficient       DECIMAL(19,12) NOT NULL,

    CONSTRAINT PK_COMPILED_1D_RATE_BAND
        PRIMARY KEY (rate_package_id, term_id, feature_level_id),

    CONSTRAINT FK_COMPILED_1D_RATE_BAND_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id),

    CONSTRAINT FK_COMPILED_1D_RATE_BAND_TERM
        FOREIGN KEY (term_id)
        REFERENCES pricing.PRICING_TERM(term_id),

    CONSTRAINT FK_COMPILED_1D_RATE_BAND_LEVEL
        FOREIGN KEY (feature_level_id)
        REFERENCES pricing.PRICING_FEATURE_LEVEL(feature_level_id)
);
GO
