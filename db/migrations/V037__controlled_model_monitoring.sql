IF OBJECT_ID('mlops.MODEL_MONITOR_VARIANT', 'U') IS NULL
BEGIN
    CREATE TABLE mlops.MODEL_MONITOR_VARIANT (
        variant_code NVARCHAR(32) NOT NULL,
        variant_label NVARCHAR(128) NOT NULL,
        refit_coefficients BIT NOT NULL,
        reestimate_lambdas BIT NOT NULL,
        reposition_data_driven_knots BIT NOT NULL,
        structure_frozen BIT NOT NULL
            CONSTRAINT DF_MODEL_MONITOR_VARIANT_STRUCTURE_FROZEN DEFAULT (1),
        CONSTRAINT PK_MODEL_MONITOR_VARIANT PRIMARY KEY (variant_code),
        CONSTRAINT CK_MODEL_MONITOR_VARIANT_CODE CHECK (
            variant_code IN (
                'STATIC_SCORE',
                'FROZEN_REFIT',
                'REESTIMATE_LAMBDA',
                'FULL_ADAPTIVE'
            )
        )
    );
END;
GO

MERGE mlops.MODEL_MONITOR_VARIANT AS target
USING (
    VALUES
        ('STATIC_SCORE', 'Deployed model, no refit', 0, 0, 0, 1),
        ('FROZEN_REFIT', 'Refit coefficients only', 1, 0, 0, 1),
        ('REESTIMATE_LAMBDA', 'Refit coefficients and REML lambdas', 1, 1, 0, 1),
        ('FULL_ADAPTIVE', 'Refit coefficients, lambdas, and data-driven knots', 1, 1, 1, 1)
) AS source (
    variant_code,
    variant_label,
    refit_coefficients,
    reestimate_lambdas,
    reposition_data_driven_knots,
    structure_frozen
)
ON target.variant_code = source.variant_code
WHEN NOT MATCHED THEN INSERT (
    variant_code,
    variant_label,
    refit_coefficients,
    reestimate_lambdas,
    reposition_data_driven_knots,
    structure_frozen
) VALUES (
    source.variant_code,
    source.variant_label,
    source.refit_coefficients,
    source.reestimate_lambdas,
    source.reposition_data_driven_knots,
    source.structure_frozen
);
GO

IF EXISTS (
    SELECT 1
    FROM (
        VALUES
            ('STATIC_SCORE', 'Deployed model, no refit', 0, 0, 0, 1),
            ('FROZEN_REFIT', 'Refit coefficients only', 1, 0, 0, 1),
            ('REESTIMATE_LAMBDA', 'Refit coefficients and REML lambdas', 1, 1, 0, 1),
            ('FULL_ADAPTIVE', 'Refit coefficients, lambdas, and data-driven knots', 1, 1, 1, 1)
    ) AS canonical (
        variant_code,
        variant_label,
        refit_coefficients,
        reestimate_lambdas,
        reposition_data_driven_knots,
        structure_frozen
    )
    LEFT JOIN mlops.MODEL_MONITOR_VARIANT AS variant
      ON variant.variant_code = canonical.variant_code
    WHERE variant.variant_code IS NULL
       OR variant.variant_label <> canonical.variant_label
       OR variant.refit_coefficients <> canonical.refit_coefficients
       OR variant.reestimate_lambdas <> canonical.reestimate_lambdas
       OR variant.reposition_data_driven_knots <> canonical.reposition_data_driven_knots
       OR variant.structure_frozen <> canonical.structure_frozen
)
BEGIN
    THROW 51026, 'Monitoring variants differ from the canonical policy.', 1;
END;
GO

CREATE OR ALTER TRIGGER mlops.TR_MODEL_MONITOR_VARIANT_IMMUTABLE
ON mlops.MODEL_MONITOR_VARIANT
INSTEAD OF UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;
    THROW 51027, 'Monitoring variant policy is immutable.', 1;
END;
GO

IF OBJECT_ID('mlops.MODEL_FIT_CONTRACT', 'U') IS NULL
BEGIN
    CREATE TABLE mlops.MODEL_FIT_CONTRACT (
        fit_contract_id UNIQUEIDENTIFIER NOT NULL,
        baseline_model_run_id BIGINT NOT NULL,
        model_id BIGINT NOT NULL,
        rate_package_id BIGINT NOT NULL,
        contract_schema_version INT NOT NULL,
        contract_sha256 CHAR(64) COLLATE Latin1_General_BIN2 NOT NULL,
        structure_sha256 CHAR(64) COLLATE Latin1_General_BIN2 NOT NULL,
        contract_json NVARCHAR(MAX) NOT NULL,
        superglm_version NVARCHAR(64) NOT NULL,
        created_ts DATETIME2(3) NOT NULL
            CONSTRAINT DF_MODEL_FIT_CONTRACT_CREATED_TS DEFAULT SYSUTCDATETIME(),
        created_by NVARCHAR(255) NOT NULL,
        CONSTRAINT PK_MODEL_FIT_CONTRACT PRIMARY KEY (fit_contract_id),
        CONSTRAINT UQ_MODEL_FIT_CONTRACT_BASELINE_RUN UNIQUE (baseline_model_run_id),
        CONSTRAINT FK_MODEL_FIT_CONTRACT_RUN FOREIGN KEY (baseline_model_run_id)
            REFERENCES pricing.MODEL_RUN(model_run_id),
        CONSTRAINT FK_MODEL_FIT_CONTRACT_MODEL FOREIGN KEY (model_id)
            REFERENCES pricing.PRICING_MODEL(model_id),
        CONSTRAINT FK_MODEL_FIT_CONTRACT_PACKAGE FOREIGN KEY (rate_package_id)
            REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id),
        CONSTRAINT CK_MODEL_FIT_CONTRACT_SCHEMA_VERSION CHECK (contract_schema_version >= 1),
        CONSTRAINT CK_MODEL_FIT_CONTRACT_JSON CHECK (ISJSON(contract_json) = 1),
        CONSTRAINT CK_MODEL_FIT_CONTRACT_DIGESTS CHECK (
            LEN(contract_sha256) = 64
            AND contract_sha256 NOT LIKE '%[^0-9a-f]%'
            AND LEN(structure_sha256) = 64
            AND structure_sha256 NOT LIKE '%[^0-9a-f]%'
        )
    );
END;
GO

IF OBJECT_ID('mlops.MODEL_MONITOR_RUN', 'U') IS NULL
BEGIN
    CREATE TABLE mlops.MODEL_MONITOR_RUN (
        monitor_run_id UNIQUEIDENTIFIER NOT NULL,
        fit_contract_id UNIQUEIDENTIFIER NOT NULL,
        baseline_deployment_id BIGINT NOT NULL,
        model_id BIGINT NOT NULL,
        rate_package_id BIGINT NOT NULL,
        manifest_id NVARCHAR(128) NOT NULL,
        component_role NVARCHAR(32) NOT NULL,
        variant_code NVARCHAR(32) NOT NULL,
        run_signature_sha256 CHAR(64) COLLATE Latin1_General_BIN2 NOT NULL,
        run_status NVARCHAR(32) NOT NULL,
        invariant_status NVARCHAR(16) NOT NULL,
        invariant_evidence_sha256 CHAR(64) COLLATE Latin1_General_BIN2 NOT NULL,
        invariant_evidence_json NVARCHAR(MAX) NOT NULL,
        model_frame_sha256 CHAR(64) COLLATE Latin1_General_BIN2 NOT NULL,
        fit_configuration_json NVARCHAR(MAX) NOT NULL,
        result_evidence_sha256 CHAR(64) COLLATE Latin1_General_BIN2 NOT NULL,
        started_ts DATETIME2(3) NOT NULL
            CONSTRAINT DF_MODEL_MONITOR_RUN_STARTED_TS DEFAULT SYSUTCDATETIME(),
        completed_ts DATETIME2(3) NOT NULL
            CONSTRAINT DF_MODEL_MONITOR_RUN_COMPLETED_TS DEFAULT SYSUTCDATETIME(),
        created_by NVARCHAR(255) NOT NULL,
        CONSTRAINT PK_MODEL_MONITOR_RUN PRIMARY KEY (monitor_run_id),
        CONSTRAINT UQ_MODEL_MONITOR_RUN_SIGNATURE UNIQUE (run_signature_sha256),
        CONSTRAINT UQ_MODEL_MONITOR_RUN_OBSERVATION UNIQUE (
            baseline_deployment_id,
            manifest_id,
            component_role,
            variant_code
        ),
        CONSTRAINT FK_MODEL_MONITOR_RUN_CONTRACT FOREIGN KEY (fit_contract_id)
            REFERENCES mlops.MODEL_FIT_CONTRACT(fit_contract_id),
        CONSTRAINT FK_MODEL_MONITOR_RUN_DEPLOYMENT FOREIGN KEY (baseline_deployment_id)
            REFERENCES pricing.PRICING_MODEL_DEPLOYMENT(deployment_id),
        CONSTRAINT FK_MODEL_MONITOR_RUN_MODEL FOREIGN KEY (model_id)
            REFERENCES pricing.PRICING_MODEL(model_id),
        CONSTRAINT FK_MODEL_MONITOR_RUN_PACKAGE FOREIGN KEY (rate_package_id)
            REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id),
        CONSTRAINT FK_MODEL_MONITOR_RUN_MANIFEST FOREIGN KEY (manifest_id)
            REFERENCES pricing.DATASET_MANIFEST(manifest_id),
        CONSTRAINT FK_MODEL_MONITOR_RUN_VARIANT FOREIGN KEY (variant_code)
            REFERENCES mlops.MODEL_MONITOR_VARIANT(variant_code),
        CONSTRAINT CK_MODEL_MONITOR_RUN_COMPONENT CHECK (
            component_role IN ('FREQUENCY', 'SEVERITY', 'OTHER')
        ),
        CONSTRAINT CK_MODEL_MONITOR_RUN_STATUS CHECK (
            run_status IN ('SUCCESS', 'FAILED')
        ),
        CONSTRAINT CK_MODEL_MONITOR_RUN_INVARIANT_STATUS CHECK (
            invariant_status = 'VERIFIED'
        ),
        CONSTRAINT CK_MODEL_MONITOR_RUN_INVARIANT_JSON CHECK (
            ISJSON(invariant_evidence_json) = 1
        ),
        CONSTRAINT CK_MODEL_MONITOR_RUN_FIT_CONFIGURATION_JSON CHECK (
            ISJSON(fit_configuration_json) = 1
        ),
        CONSTRAINT CK_MODEL_MONITOR_RUN_INVARIANT_DIGEST CHECK (
            LEN(invariant_evidence_sha256) = 64
            AND invariant_evidence_sha256 NOT LIKE '%[^0-9a-f]%'
        ),
        CONSTRAINT CK_MODEL_MONITOR_RUN_SIGNATURE CHECK (
            LEN(run_signature_sha256) = 64
            AND run_signature_sha256 NOT LIKE '%[^0-9a-f]%'
        ),
        CONSTRAINT CK_MODEL_MONITOR_RUN_RESULT_DIGESTS CHECK (
            LEN(model_frame_sha256) = 64
            AND model_frame_sha256 NOT LIKE '%[^0-9a-f]%'
            AND LEN(result_evidence_sha256) = 64
            AND result_evidence_sha256 NOT LIKE '%[^0-9a-f]%'
        ),
        CONSTRAINT CK_MODEL_MONITOR_RUN_TIMES CHECK (completed_ts >= started_ts)
    );
END;
GO

IF OBJECT_ID('mlops.MODEL_MONITOR_TERM', 'U') IS NULL
BEGIN
    CREATE TABLE mlops.MODEL_MONITOR_TERM (
        monitor_run_id UNIQUEIDENTIFIER NOT NULL,
        term_name NVARCHAR(255) NOT NULL,
        term_kind NVARCHAR(64) NOT NULL,
        sequence_no INT NOT NULL,
        term_structure_sha256 CHAR(64) COLLATE Latin1_General_BIN2 NOT NULL,
        term_metadata_json NVARCHAR(MAX) NOT NULL,
        CONSTRAINT PK_MODEL_MONITOR_TERM PRIMARY KEY (monitor_run_id, term_name),
        CONSTRAINT UQ_MODEL_MONITOR_TERM_SEQUENCE UNIQUE (monitor_run_id, sequence_no),
        CONSTRAINT FK_MODEL_MONITOR_TERM_RUN FOREIGN KEY (monitor_run_id)
            REFERENCES mlops.MODEL_MONITOR_RUN(monitor_run_id),
        CONSTRAINT CK_MODEL_MONITOR_TERM_SEQUENCE CHECK (sequence_no >= 1),
        CONSTRAINT CK_MODEL_MONITOR_TERM_JSON CHECK (ISJSON(term_metadata_json) = 1),
        CONSTRAINT CK_MODEL_MONITOR_TERM_DIGEST CHECK (
            LEN(term_structure_sha256) = 64
            AND term_structure_sha256 NOT LIKE '%[^0-9a-f]%'
        )
    );
END;
GO

IF OBJECT_ID('mlops.MODEL_MONITOR_LAMBDA', 'U') IS NULL
BEGIN
    CREATE TABLE mlops.MODEL_MONITOR_LAMBDA (
        monitor_run_id UNIQUEIDENTIFIER NOT NULL,
        component_name NVARCHAR(255) NOT NULL,
        term_name NVARCHAR(255) NULL,
        lambda_value FLOAT NOT NULL,
        lambda_mode NVARCHAR(16) NOT NULL,
        CONSTRAINT PK_MODEL_MONITOR_LAMBDA PRIMARY KEY (monitor_run_id, component_name),
        CONSTRAINT FK_MODEL_MONITOR_LAMBDA_RUN FOREIGN KEY (monitor_run_id)
            REFERENCES mlops.MODEL_MONITOR_RUN(monitor_run_id),
        CONSTRAINT CK_MODEL_MONITOR_LAMBDA_VALUE CHECK (lambda_value >= 0),
        CONSTRAINT CK_MODEL_MONITOR_LAMBDA_MODE CHECK (
            lambda_mode IN ('BASELINE', 'FIXED', 'ESTIMATED')
        )
    );
END;
GO

IF OBJECT_ID('mlops.MODEL_MONITOR_RELATIVITY', 'U') IS NULL
BEGIN
    CREATE TABLE mlops.MODEL_MONITOR_RELATIVITY (
        monitor_run_id UNIQUEIDENTIFIER NOT NULL,
        term_name NVARCHAR(255) NOT NULL,
        term_kind NVARCHAR(64) NOT NULL,
        point_key NVARCHAR(512) NOT NULL,
        point_label NVARCHAR(512) NULL,
        point_numeric FLOAT NULL,
        relativity FLOAT NOT NULL,
        log_relativity FLOAT NOT NULL,
        is_reference BIT NOT NULL,
        CONSTRAINT PK_MODEL_MONITOR_RELATIVITY
            PRIMARY KEY (monitor_run_id, term_name, point_key),
        CONSTRAINT FK_MODEL_MONITOR_RELATIVITY_RUN FOREIGN KEY (monitor_run_id)
            REFERENCES mlops.MODEL_MONITOR_RUN(monitor_run_id),
        CONSTRAINT CK_MODEL_MONITOR_RELATIVITY_POINT CHECK (
            (point_label IS NULL AND point_numeric IS NOT NULL)
            OR (point_label IS NOT NULL AND point_numeric IS NULL)
        ),
        CONSTRAINT CK_MODEL_MONITOR_RELATIVITY_VALUE CHECK (relativity > 0)
    );
END;
GO

IF OBJECT_ID('mlops.MODEL_MONITOR_METRIC', 'U') IS NULL
BEGIN
    CREATE TABLE mlops.MODEL_MONITOR_METRIC (
        monitor_run_id UNIQUEIDENTIFIER NOT NULL,
        metric_name NVARCHAR(128) NOT NULL,
        metric_value FLOAT NOT NULL,
        CONSTRAINT PK_MODEL_MONITOR_METRIC PRIMARY KEY (monitor_run_id, metric_name),
        CONSTRAINT FK_MODEL_MONITOR_METRIC_RUN FOREIGN KEY (monitor_run_id)
            REFERENCES mlops.MODEL_MONITOR_RUN(monitor_run_id)
    );
END;
GO

CREATE OR ALTER TRIGGER mlops.TR_MODEL_FIT_CONTRACT_IMMUTABLE
ON mlops.MODEL_FIT_CONTRACT
INSTEAD OF UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;
    THROW 51020, 'Model fit contracts are immutable; deploy a new baseline.', 1;
END;
GO

CREATE OR ALTER TRIGGER mlops.TR_MODEL_FIT_CONTRACT_LINEAGE_GUARD
ON mlops.MODEL_FIT_CONTRACT
AFTER INSERT
AS
BEGIN
    SET NOCOUNT ON;
    IF EXISTS (
        SELECT 1
        FROM inserted AS contract
        LEFT JOIN pricing.MODEL_RUN AS baseline_run
          ON baseline_run.model_run_id = contract.baseline_model_run_id
        LEFT JOIN pricing.PRICING_RATE_PACKAGE AS package
          ON package.rate_package_id = contract.rate_package_id
        WHERE baseline_run.model_run_id IS NULL
           OR package.rate_package_id IS NULL
           OR baseline_run.model_id <> contract.model_id
           OR baseline_run.rate_package_id <> contract.rate_package_id
           OR baseline_run.run_status <> 'SUCCESS'
           OR package.model_id <> contract.model_id
           OR package.package_status <> 'PUBLISHED'
    )
    BEGIN
        THROW 51022, 'A fit contract must identify one successful published baseline run.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_PRICING_MODEL_DEPLOYMENT_MONITORING_LINEAGE_GUARD
ON pricing.PRICING_MODEL_DEPLOYMENT
AFTER UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM deleted AS historical_deployment
        JOIN mlops.MODEL_MONITOR_RUN AS monitor_run
          ON monitor_run.baseline_deployment_id = historical_deployment.deployment_id
        WHERE NOT EXISTS (
            SELECT 1
            FROM inserted AS current_deployment
            WHERE current_deployment.deployment_id = historical_deployment.deployment_id
              AND current_deployment.model_id = historical_deployment.model_id
              AND current_deployment.rate_package_id = historical_deployment.rate_package_id
              AND current_deployment.deployment_slot = historical_deployment.deployment_slot
              AND current_deployment.effective_from_ts = historical_deployment.effective_from_ts
        )
    )
    BEGIN
        THROW 51024, 'Deployments referenced by monitoring evidence retain their lineage identity.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER pricing.TR_DATASET_MANIFEST_MONITORING_LINEAGE_GUARD
ON pricing.DATASET_MANIFEST
AFTER UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM deleted AS historical_manifest
        JOIN mlops.MODEL_MONITOR_RUN AS monitor_run
          ON monitor_run.manifest_id = historical_manifest.manifest_id
    )
    BEGIN
        THROW 51025, 'Dataset manifests referenced by monitoring evidence are immutable.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER mlops.TR_MODEL_MONITOR_RUN_LINEAGE_GUARD
ON mlops.MODEL_MONITOR_RUN
AFTER INSERT, UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    IF EXISTS (
        SELECT 1
        FROM inserted AS monitor_run
        JOIN mlops.MODEL_FIT_CONTRACT AS contract
          ON contract.fit_contract_id = monitor_run.fit_contract_id
        JOIN pricing.MODEL_RUN AS baseline_run
          ON baseline_run.model_run_id = contract.baseline_model_run_id
        JOIN pricing.PRICING_MODEL_DEPLOYMENT AS deployment
          ON deployment.deployment_id = monitor_run.baseline_deployment_id
        WHERE contract.model_id <> monitor_run.model_id
           OR contract.rate_package_id <> monitor_run.rate_package_id
           OR baseline_run.model_id <> monitor_run.model_id
           OR baseline_run.rate_package_id <> monitor_run.rate_package_id
           OR deployment.model_id <> monitor_run.model_id
           OR deployment.rate_package_id <> monitor_run.rate_package_id
    )
    BEGIN
        THROW 51021, 'Monitoring contract, baseline run, and deployment must identify one model package.', 1;
    END;
END;
GO

CREATE OR ALTER TRIGGER mlops.TR_MODEL_MONITOR_RUN_IMMUTABLE
ON mlops.MODEL_MONITOR_RUN
INSTEAD OF UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;
    THROW 51023, 'Monitoring evidence is immutable; write a new observation.', 1;
END;
GO

CREATE OR ALTER TRIGGER mlops.TR_MODEL_MONITOR_TERM_IMMUTABLE
ON mlops.MODEL_MONITOR_TERM
INSTEAD OF UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;
    THROW 51023, 'Monitoring evidence is immutable; write a new observation.', 1;
END;
GO

CREATE OR ALTER TRIGGER mlops.TR_MODEL_MONITOR_LAMBDA_IMMUTABLE
ON mlops.MODEL_MONITOR_LAMBDA
INSTEAD OF UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;
    THROW 51023, 'Monitoring evidence is immutable; write a new observation.', 1;
END;
GO

CREATE OR ALTER TRIGGER mlops.TR_MODEL_MONITOR_RELATIVITY_IMMUTABLE
ON mlops.MODEL_MONITOR_RELATIVITY
INSTEAD OF UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;
    THROW 51023, 'Monitoring evidence is immutable; write a new observation.', 1;
END;
GO

CREATE OR ALTER TRIGGER mlops.TR_MODEL_MONITOR_METRIC_IMMUTABLE
ON mlops.MODEL_MONITOR_METRIC
INSTEAD OF UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;
    THROW 51023, 'Monitoring evidence is immutable; write a new observation.', 1;
END;
GO

CREATE OR ALTER VIEW pricing.V_MODEL_MONITORING_RUN
AS
SELECT
    monitor_run.monitor_run_id,
    monitor_run.variant_code,
    variant.variant_label,
    variant.refit_coefficients,
    variant.reestimate_lambdas,
    variant.reposition_data_driven_knots,
    monitor_run.component_role,
    monitor_run.run_status,
    monitor_run.invariant_status,
    monitor_run.invariant_evidence_sha256,
    monitor_run.invariant_evidence_json,
    monitor_run.model_frame_sha256 AS observed_model_frame_sha256,
    monitor_run.fit_configuration_json,
    monitor_run.result_evidence_sha256,
    monitor_run.started_ts,
    monitor_run.completed_ts,
    monitor_run.created_by,
    monitor_run.run_signature_sha256,
    contract.fit_contract_id,
    contract.baseline_model_run_id,
    contract.contract_sha256,
    contract.structure_sha256,
    contract.superglm_version,
    monitor_run.baseline_deployment_id,
    deployment.deployment_slot AS baseline_deployment_slot,
    monitor_run.model_id,
    model.model_name,
    model.model_label,
    model.target_name,
    model.model_type,
    monitor_run.rate_package_id,
    package.model_version AS baseline_model_version,
    package.package_version AS baseline_package_version,
    monitor_run.manifest_id,
    manifest.manifest_signature_sha256,
    manifest.dataset_name,
    manifest.source_system,
    manifest.data_as_of_date,
    manifest.data_as_of_column,
    manifest.row_count AS dataset_row_count,
    manifest.model_frame_sha256
FROM mlops.MODEL_MONITOR_RUN AS monitor_run
JOIN mlops.MODEL_MONITOR_VARIANT AS variant
  ON variant.variant_code = monitor_run.variant_code
JOIN mlops.MODEL_FIT_CONTRACT AS contract
  ON contract.fit_contract_id = monitor_run.fit_contract_id
JOIN pricing.PRICING_MODEL_DEPLOYMENT AS deployment
  ON deployment.deployment_id = monitor_run.baseline_deployment_id
JOIN pricing.PRICING_MODEL AS model
  ON model.model_id = monitor_run.model_id
JOIN pricing.PRICING_RATE_PACKAGE AS package
  ON package.rate_package_id = monitor_run.rate_package_id
JOIN pricing.DATASET_MANIFEST AS manifest
  ON manifest.manifest_id = monitor_run.manifest_id;
GO

CREATE OR ALTER VIEW pricing.V_MODEL_MONITORING_RELATIVITY
AS
SELECT
    monitoring_run.*,
    relativity.term_name,
    term.sequence_no AS term_sequence_no,
    relativity.term_kind,
    term.term_structure_sha256,
    term.term_metadata_json,
    relativity.point_key,
    relativity.point_label,
    relativity.point_numeric,
    relativity.relativity,
    relativity.log_relativity,
    relativity.is_reference
FROM pricing.V_MODEL_MONITORING_RUN AS monitoring_run
JOIN mlops.MODEL_MONITOR_RELATIVITY AS relativity
  ON relativity.monitor_run_id = monitoring_run.monitor_run_id
JOIN mlops.MODEL_MONITOR_TERM AS term
  ON term.monitor_run_id = relativity.monitor_run_id
 AND term.term_name = relativity.term_name;
GO

CREATE OR ALTER VIEW pricing.V_MODEL_MONITORING_LAMBDA
AS
SELECT
    monitoring_run.*,
    lambda.component_name,
    lambda.term_name,
    lambda.lambda_value,
    lambda.lambda_mode
FROM pricing.V_MODEL_MONITORING_RUN AS monitoring_run
JOIN mlops.MODEL_MONITOR_LAMBDA AS lambda
  ON lambda.monitor_run_id = monitoring_run.monitor_run_id;
GO
