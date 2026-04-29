IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'pricing')
    EXEC('CREATE SCHEMA pricing');
GO

IF OBJECT_ID('pricing.FREMTPL_RAW', 'U') IS NULL
CREATE TABLE pricing.FREMTPL_RAW (
    IDpol BIGINT NOT NULL PRIMARY KEY,
    ClaimNb INT NOT NULL,
    Exposure FLOAT NOT NULL,
    Area NVARCHAR(16) NULL,
    VehPower INT NULL,
    VehAge INT NULL,
    DrivAge INT NULL,
    BonusMalus INT NULL,
    VehBrand NVARCHAR(64) NULL,
    VehGas NVARCHAR(16) NULL,
    Density FLOAT NULL,
    Region NVARCHAR(32) NULL
);
GO

IF OBJECT_ID('pricing.MODEL_RUN', 'U') IS NULL
CREATE TABLE pricing.MODEL_RUN (
    model_run_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    model_id BIGINT NULL,
    dag_id NVARCHAR(250) NOT NULL,
    airflow_run_id NVARCHAR(250) NOT NULL,
    mlflow_experiment_id NVARCHAR(128) NULL,
    mlflow_run_id NVARCHAR(128) NULL,
    manifest_id NVARCHAR(128) NULL,
    export_id NVARCHAR(128) NULL,
    model_name NVARCHAR(128) NOT NULL,
    model_version NVARCHAR(64) NULL,
    rate_package_id BIGINT NULL,
    rating_workbook_path NVARCHAR(1024) NULL,
    run_status NVARCHAR(32) NOT NULL,
    started_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    completed_ts DATETIME2(3) NULL,
    created_by NVARCHAR(128) NOT NULL,

    CONSTRAINT FK_MODEL_RUN_MANIFEST
        FOREIGN KEY (manifest_id)
        REFERENCES pricing.DATASET_MANIFEST(manifest_id),

    CONSTRAINT FK_MODEL_RUN_PACKAGE
        FOREIGN KEY (rate_package_id)
        REFERENCES pricing.PRICING_RATE_PACKAGE(rate_package_id)
);
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_MODEL_RUN_AIRFLOW'
      AND object_id = OBJECT_ID('pricing.MODEL_RUN')
)
CREATE UNIQUE INDEX UX_MODEL_RUN_AIRFLOW
ON pricing.MODEL_RUN(dag_id, airflow_run_id, model_name);
GO
