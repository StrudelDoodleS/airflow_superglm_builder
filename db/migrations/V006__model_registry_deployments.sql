IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
    EXEC('CREATE SCHEMA pricing');
GO

IF OBJECT_ID('pricing.PRICING_MODEL', 'U') IS NULL
CREATE TABLE pricing.PRICING_MODEL (
    model_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    model_key NVARCHAR(128) NOT NULL,
    model_label NVARCHAR(256) NULL,
    target_name NVARCHAR(128) NOT NULL,
    model_type NVARCHAR(128) NOT NULL,
    model_status NVARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
    created_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    created_by NVARCHAR(128) NOT NULL,
    retired_ts DATETIME2(3) NULL,

    CONSTRAINT UQ_PRICING_MODEL_KEY UNIQUE (model_key),
    CONSTRAINT CK_PRICING_MODEL_STATUS
        CHECK (model_status IN ('ACTIVE', 'RETIRED', 'DISABLED'))
);
GO

INSERT INTO pricing.PRICING_MODEL (
    model_key,
    target_name,
    model_type,
    model_status,
    created_by
)
SELECT DISTINCT
    src.model_name,
    'ClaimNb',
    'superglm_poisson',
    'ACTIVE',
    'migration'
FROM (
    SELECT model_name FROM pricing.PRICING_RATE_PACKAGE
    UNION
    SELECT model_name FROM pricing_stg.STG_RATING_EXPORT
    UNION
    SELECT model_name FROM pricing.MODEL_RUN
) AS src
WHERE src.model_name IS NOT NULL
  AND NOT EXISTS (
      SELECT 1
      FROM pricing.PRICING_MODEL m
      WHERE m.model_key = src.model_name
  );
GO

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'model_id') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD model_id BIGINT NULL;
GO

IF COL_LENGTH('pricing.MODEL_RUN', 'model_id') IS NULL
    ALTER TABLE pricing.MODEL_RUN ADD model_id BIGINT NULL;
GO

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'model_id') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT ADD model_id BIGINT NULL;
GO

IF COL_LENGTH('pricing.PRICING_PACKAGE_POINTER', 'model_id') IS NULL
    ALTER TABLE pricing.PRICING_PACKAGE_POINTER ADD model_id BIGINT NULL;
GO

IF COL_LENGTH('pricing.PRICING_FEATURE_LEVEL_SET', 'model_id') IS NULL
    ALTER TABLE pricing.PRICING_FEATURE_LEVEL_SET ADD model_id BIGINT NULL;
GO

UPDATE rp
SET model_id = m.model_id
FROM pricing.PRICING_RATE_PACKAGE rp
JOIN pricing.PRICING_MODEL m
  ON m.model_key = rp.model_name
WHERE rp.model_id IS NULL;
GO

UPDATE mr
SET model_id = m.model_id
FROM pricing.MODEL_RUN mr
JOIN pricing.PRICING_MODEL m
  ON m.model_key = mr.model_name
WHERE mr.model_id IS NULL;
GO

UPDATE stg
SET model_id = m.model_id
FROM pricing_stg.STG_RATING_EXPORT stg
JOIN pricing.PRICING_MODEL m
  ON m.model_key = stg.model_name
WHERE stg.model_id IS NULL;
GO

UPDATE pp
SET model_id = rp.model_id
FROM pricing.PRICING_PACKAGE_POINTER pp
JOIN pricing.PRICING_RATE_PACKAGE rp
  ON rp.rate_package_id = pp.rate_package_id
WHERE pp.model_id IS NULL
  AND rp.model_id IS NOT NULL;
GO

UPDATE ls
SET model_id = rp.model_id
FROM pricing.PRICING_FEATURE_LEVEL_SET ls
JOIN pricing.PRICING_TERM_FEATURE tf
  ON tf.level_set_id = ls.level_set_id
JOIN pricing.PRICING_TERM t
  ON t.term_id = tf.term_id
JOIN pricing.PRICING_RATE_PACKAGE rp
  ON rp.rate_package_id = t.rate_package_id
WHERE ls.model_id IS NULL
  AND rp.model_id IS NOT NULL;
GO

IF (SELECT COUNT(*) FROM pricing.PRICING_MODEL) = 1
BEGIN
    UPDATE pricing.PRICING_FEATURE_LEVEL_SET
    SET model_id = (SELECT MIN(model_id) FROM pricing.PRICING_MODEL)
    WHERE model_id IS NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_RATE_PACKAGE_MODEL'
)
ALTER TABLE pricing.PRICING_RATE_PACKAGE
ADD CONSTRAINT FK_RATE_PACKAGE_MODEL
    FOREIGN KEY (model_id)
    REFERENCES pricing.PRICING_MODEL(model_id);
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_MODEL_RUN_MODEL'
)
ALTER TABLE pricing.MODEL_RUN
ADD CONSTRAINT FK_MODEL_RUN_MODEL
    FOREIGN KEY (model_id)
    REFERENCES pricing.PRICING_MODEL(model_id);
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_STG_RATING_EXPORT_MODEL'
)
ALTER TABLE pricing_stg.STG_RATING_EXPORT
ADD CONSTRAINT FK_STG_RATING_EXPORT_MODEL
    FOREIGN KEY (model_id)
    REFERENCES pricing.PRICING_MODEL(model_id);
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_PACKAGE_POINTER_MODEL'
)
ALTER TABLE pricing.PRICING_PACKAGE_POINTER
ADD CONSTRAINT FK_PACKAGE_POINTER_MODEL
    FOREIGN KEY (model_id)
    REFERENCES pricing.PRICING_MODEL(model_id);
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_LEVEL_SET_MODEL'
)
ALTER TABLE pricing.PRICING_FEATURE_LEVEL_SET
ADD CONSTRAINT FK_LEVEL_SET_MODEL
    FOREIGN KEY (model_id)
    REFERENCES pricing.PRICING_MODEL(model_id);
GO

IF EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = 'UQ_LEVEL_SET'
      AND parent_object_id = OBJECT_ID('pricing.PRICING_FEATURE_LEVEL_SET')
)
ALTER TABLE pricing.PRICING_FEATURE_LEVEL_SET DROP CONSTRAINT UQ_LEVEL_SET;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_LEVEL_SET_MODEL_FEATURE_NAME'
      AND object_id = OBJECT_ID('pricing.PRICING_FEATURE_LEVEL_SET')
)
CREATE UNIQUE INDEX UX_LEVEL_SET_MODEL_FEATURE_NAME
ON pricing.PRICING_FEATURE_LEVEL_SET(model_id, feature_id, level_set_name);
GO

DECLARE @pointer_pk_name sysname;

SELECT @pointer_pk_name = kc.name
FROM sys.key_constraints kc
WHERE kc.parent_object_id = OBJECT_ID('pricing.PRICING_PACKAGE_POINTER')
  AND kc.type = 'PK';

IF @pointer_pk_name IS NOT NULL
BEGIN
    DECLARE @drop_pointer_pk_sql NVARCHAR(MAX);
    SET @drop_pointer_pk_sql =
        N'ALTER TABLE pricing.PRICING_PACKAGE_POINTER DROP CONSTRAINT '
        + QUOTENAME(@pointer_pk_name);
    EXEC sp_executesql @drop_pointer_pk_sql;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_PACKAGE_POINTER_MODEL_SLOT'
      AND object_id = OBJECT_ID('pricing.PRICING_PACKAGE_POINTER')
)
CREATE UNIQUE INDEX UX_PACKAGE_POINTER_MODEL_SLOT
ON pricing.PRICING_PACKAGE_POINTER(model_id, pointer_name)
WHERE model_id IS NOT NULL;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_MODEL_RUN_AIRFLOW_MODEL_ID'
      AND object_id = OBJECT_ID('pricing.MODEL_RUN')
)
CREATE UNIQUE INDEX UX_MODEL_RUN_AIRFLOW_MODEL_ID
ON pricing.MODEL_RUN(dag_id, airflow_run_id, model_id)
WHERE model_id IS NOT NULL;
GO

IF OBJECT_ID('pricing.PRICING_MODEL_DEPLOYMENT', 'U') IS NULL
CREATE TABLE pricing.PRICING_MODEL_DEPLOYMENT (
    deployment_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    model_id BIGINT NOT NULL,
    rate_package_id BIGINT NOT NULL,
    deployment_slot NVARCHAR(64) NOT NULL,
    effective_from_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    effective_to_ts DATETIME2(3) NULL,
    deployed_by NVARCHAR(128) NOT NULL,
    deployment_note NVARCHAR(512) NULL,
    created_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),

    CONSTRAINT FK_MODEL_DEPLOYMENT_MODEL
        FOREIGN KEY (model_id)
        REFERENCES pricing.PRICING_MODEL(model_id),

    CONSTRAINT FK_MODEL_DEPLOYMENT_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id),

    CONSTRAINT CK_MODEL_DEPLOYMENT_EFFECTIVE_DATES
        CHECK (effective_to_ts IS NULL OR effective_to_ts > effective_from_ts)
);
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_MODEL_DEPLOYMENT_CURRENT'
      AND object_id = OBJECT_ID('pricing.PRICING_MODEL_DEPLOYMENT')
)
CREATE UNIQUE INDEX UX_MODEL_DEPLOYMENT_CURRENT
ON pricing.PRICING_MODEL_DEPLOYMENT(model_id, deployment_slot)
WHERE effective_to_ts IS NULL;
GO

INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT (
    model_id,
    rate_package_id,
    deployment_slot,
    effective_from_ts,
    deployed_by
)
SELECT
    pp.model_id,
    pp.rate_package_id,
    pp.pointer_name,
    pp.updated_ts,
    pp.updated_by
FROM pricing.PRICING_PACKAGE_POINTER pp
WHERE pp.model_id IS NOT NULL
  AND NOT EXISTS (
      SELECT 1
      FROM pricing.PRICING_MODEL_DEPLOYMENT d
      WHERE d.model_id = pp.model_id
        AND d.deployment_slot = pp.pointer_name
        AND d.effective_to_ts IS NULL
  );
GO

CREATE OR ALTER VIEW pricing.V_ACTIVE_MODEL AS
SELECT
    model_id,
    model_key,
    model_label,
    target_name,
    model_type,
    model_status,
    created_ts,
    created_by,
    retired_ts
FROM pricing.PRICING_MODEL
WHERE model_status = 'ACTIVE';
GO

CREATE OR ALTER VIEW pricing.V_CURRENT_RATE_PACKAGE AS
SELECT
    d.deployment_id,
    d.deployment_slot,
    d.effective_from_ts,
    d.model_id,
    m.model_key,
    m.target_name,
    m.model_type,
    rp.rate_package_id,
    rp.parent_rate_package_id,
    rp.model_name,
    rp.model_version,
    rp.package_version,
    rp.base_rate,
    rp.effective_from_date,
    rp.effective_to_date,
    rp.package_status,
    rp.created_ts,
    rp.created_by
FROM pricing.PRICING_MODEL_DEPLOYMENT d
JOIN pricing.PRICING_MODEL m
  ON m.model_id = d.model_id
JOIN pricing.PRICING_RATE_PACKAGE rp
  ON rp.rate_package_id = d.rate_package_id
WHERE d.effective_to_ts IS NULL;
GO

CREATE OR ALTER VIEW pricing.V_CURRENT_RATE_CELL AS
SELECT
    cur.model_id,
    cur.model_key,
    cur.deployment_slot,
    cur.rate_package_id,
    cur.package_version,
    c.term_id,
    c.cell_key_digest,
    c.term_name,
    c.term_type,
    c.sequence_no,
    c.cell_key_text,
    c.multiplier,
    c.log_coefficient,
    c.exposure_weight,
    c.record_count,
    c.is_default,
    c.is_reference
FROM pricing.V_CURRENT_RATE_PACKAGE cur
JOIN pricing.PRICING_COMPILED_RATE_CELL c
  ON c.rate_package_id = cur.rate_package_id;
GO

CREATE OR ALTER VIEW pricing.V_CURRENT_1D_RATE_BAND AS
SELECT
    cur.model_id,
    cur.model_key,
    cur.deployment_slot,
    cur.rate_package_id,
    cur.package_version,
    b.term_id,
    b.feature_level_id,
    b.term_name,
    b.feature_name,
    b.level_code,
    b.sort_order,
    b.lower_bound,
    b.upper_bound,
    b.representative_value,
    b.multiplier,
    b.log_coefficient
FROM pricing.V_CURRENT_RATE_PACKAGE cur
JOIN pricing.PRICING_COMPILED_1D_RATE_BAND b
  ON b.rate_package_id = cur.rate_package_id;
GO
