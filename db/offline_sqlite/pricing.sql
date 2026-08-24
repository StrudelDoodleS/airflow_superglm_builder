CREATE TABLE IF NOT EXISTS pricing.FREMTPL_RAW (
    IDpol INTEGER NOT NULL PRIMARY KEY,
    ClaimNb INTEGER NOT NULL,
    Exposure REAL NOT NULL,
    Area TEXT,
    VehPower INTEGER,
    VehAge INTEGER,
    DrivAge INTEGER,
    BonusMalus INTEGER,
    VehBrand TEXT,
    VehGas TEXT,
    Density REAL,
    Region TEXT
);

CREATE TABLE IF NOT EXISTS pricing.DATASET_MANIFEST (
    manifest_id TEXT NOT NULL PRIMARY KEY,
    manifest_signature_sha256 TEXT,
    dataset_name TEXT NOT NULL,
    source_system TEXT,
    data_as_of_date TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    pk_columns_json TEXT NOT NULL,
    target_column TEXT,
    weight_column TEXT,
    model_frame_sha256 TEXT NOT NULL,
    frame_hash_metadata_json TEXT NOT NULL,
    exposure_column TEXT,
    data_as_of_column TEXT,
    offset_column TEXT,
    offset_source_column TEXT,
    offset_label TEXT,
    export_weight_column TEXT,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pricing.DATASET_COLUMN (
    manifest_id TEXT NOT NULL,
    ordinal_no INTEGER NOT NULL,
    column_name TEXT NOT NULL,
    column_role TEXT NOT NULL,
    pandas_dtype TEXT NOT NULL,
    null_count INTEGER NOT NULL,
    distinct_count INTEGER,
    PRIMARY KEY (manifest_id, ordinal_no)
);

CREATE TABLE IF NOT EXISTS pricing.CV_SPLIT_SET (
    split_set_id TEXT NOT NULL PRIMARY KEY,
    manifest_id TEXT NOT NULL,
    split_mode TEXT NOT NULL,
    splitter_class TEXT,
    splitter_params_json TEXT,
    row_order_sha256 TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    fold_count INTEGER NOT NULL,
    groups_column TEXT,
    stratify_column TEXT,
    artifact_uri TEXT,
    artifact_sha256 TEXT,
    runtime_metadata_json TEXT,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pricing.CV_FOLD (
    split_set_id TEXT NOT NULL,
    fold_no INTEGER NOT NULL,
    n_train INTEGER NOT NULL,
    n_test INTEGER NOT NULL,
    PRIMARY KEY (split_set_id, fold_no)
);

CREATE TABLE IF NOT EXISTS pricing.CV_FOLD_METRIC (
    model_run_id TEXT NOT NULL,
    split_set_id TEXT NOT NULL,
    fold_no INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    PRIMARY KEY (model_run_id, split_set_id, fold_no, metric_name)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_MODEL (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_label TEXT,
    target_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    model_status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL,
    retired_ts TEXT,
    UNIQUE (model_name)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_MODEL_VERSION_RESERVATION (
    model_id INTEGER NOT NULL,
    export_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    reserved_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_id, export_id),
    UNIQUE (model_id, model_version)
);

CREATE TABLE IF NOT EXISTS pricing.MODEL_RUN (
    model_run_id TEXT PRIMARY KEY,
    parent_model_run_id TEXT,
    model_id INTEGER NOT NULL,
    dag_id TEXT,
    airflow_run_id TEXT,
    mlflow_run_id TEXT,
    model_version TEXT NOT NULL,
    model_kind TEXT NOT NULL DEFAULT 'RAW'
        CHECK (model_kind IN ('RAW', 'ROUTINE_EDIT', 'EDITOR_EDIT', 'MANUAL_EDIT')),
    model_equivalence_sha256 TEXT,
    export_id TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    split_set_id TEXT,
    rate_package_id INTEGER NOT NULL,
    model_name TEXT,
    rating_workbook_path TEXT NOT NULL,
    rating_workbook_sha256 TEXT NOT NULL,
    publication_receipt_path TEXT,
    publication_receipt_sha256 TEXT,
    model_artifact_path TEXT,
    candidate_artifact_path TEXT,
    candidate_artifact_sha256 TEXT,
    candidate_artifact_format TEXT,
    candidate_artifact_size_bytes INTEGER,
    candidate_python_version TEXT,
    candidate_superglm_version TEXT,
    model_source_sha256 TEXT,
    effective_from TEXT,
    run_status TEXT NOT NULL DEFAULT 'SUCCEEDED',
    started_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_ts TEXT,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS pricing.UX_MODEL_RUN_RATE_PACKAGE
ON MODEL_RUN(rate_package_id)
WHERE rate_package_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS pricing.PRICING_RATE_PACKAGE (
    rate_package_id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_rate_package_id INTEGER,
    model_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    package_version INTEGER NOT NULL,
    base_rate REAL NOT NULL,
    effective_from_date TEXT,
    effective_to_date TEXT,
    package_status TEXT NOT NULL,
    source_export_id TEXT,
    source_file TEXT,
    publication_receipt_json TEXT,
    publication_receipt_sha256 TEXT,
    staging_content_sha256 TEXT,
    package_metadata_json TEXT,
    revision_metadata_json TEXT,
    offset_handling TEXT NOT NULL DEFAULT 'UNKNOWN',
    offset_factor_name TEXT,
    offset_source_name TEXT,
    offset_label TEXT,
    metadata_origin TEXT,
    manifest_id TEXT,
    split_set_id TEXT,
    rating_workbook_path TEXT,
    model_artifact_path TEXT,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL,
    UNIQUE (model_id, source_export_id),
    UNIQUE (model_id, package_version)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_MODEL_DEPLOYMENT (
    deployment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    rate_package_id INTEGER NOT NULL,
    deployment_slot TEXT NOT NULL,
    effective_from_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    effective_to_ts TEXT,
    deployed_by TEXT NOT NULL,
    deployment_note TEXT,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (effective_to_ts IS NULL OR effective_to_ts > effective_from_ts),
    FOREIGN KEY (model_id) REFERENCES PRICING_MODEL(model_id),
    FOREIGN KEY (rate_package_id) REFERENCES PRICING_RATE_PACKAGE(rate_package_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS pricing.UX_MODEL_DEPLOYMENT_CURRENT
ON PRICING_MODEL_DEPLOYMENT(model_id, deployment_slot)
WHERE effective_to_ts IS NULL;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_DEPLOYMENT_PACKAGE_GUARD_INSERT
BEFORE INSERT ON PRICING_MODEL_DEPLOYMENT
WHEN NOT EXISTS (
    SELECT 1
    FROM PRICING_RATE_PACKAGE AS package
    WHERE package.rate_package_id = NEW.rate_package_id
      AND package.model_id = NEW.model_id
      AND package.package_status = 'PUBLISHED'
)
BEGIN
    SELECT RAISE(
        ABORT,
        'deployment package must exist, match model_id, and be PUBLISHED'
    );
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_DEPLOYMENT_PACKAGE_GUARD_UPDATE
BEFORE UPDATE OF model_id, rate_package_id ON PRICING_MODEL_DEPLOYMENT
WHEN NOT EXISTS (
    SELECT 1
    FROM PRICING_RATE_PACKAGE AS package
    WHERE package.rate_package_id = NEW.rate_package_id
      AND package.model_id = NEW.model_id
      AND package.package_status = 'PUBLISHED'
)
BEGIN
    SELECT RAISE(
        ABORT,
        'deployment package must exist, match model_id, and be PUBLISHED'
    );
END;

-- SQL Server owns these logical tables in mlops.  The local mirror lives in
-- pricing.sqlite so its persistent monitoring views also work when that file
-- is opened without the attached-schema coordinator.
CREATE TABLE IF NOT EXISTS pricing.MODEL_MONITOR_VARIANT (
    variant_code TEXT NOT NULL PRIMARY KEY,
    variant_label TEXT NOT NULL,
    refit_coefficients INTEGER NOT NULL CHECK (refit_coefficients IN (0, 1)),
    reestimate_lambdas INTEGER NOT NULL CHECK (reestimate_lambdas IN (0, 1)),
    reposition_data_driven_knots INTEGER NOT NULL
        CHECK (reposition_data_driven_knots IN (0, 1)),
    structure_frozen INTEGER NOT NULL DEFAULT 1 CHECK (structure_frozen = 1),
    CHECK (
        variant_code IN (
            'STATIC_SCORE',
            'FROZEN_REFIT',
            'REESTIMATE_LAMBDA',
            'FULL_ADAPTIVE'
        )
    )
);

INSERT INTO pricing.MODEL_MONITOR_VARIANT (
    variant_code,
    variant_label,
    refit_coefficients,
    reestimate_lambdas,
    reposition_data_driven_knots,
    structure_frozen
) VALUES
    ('STATIC_SCORE', 'Deployed model, no refit', 0, 0, 0, 1),
    ('FROZEN_REFIT', 'Refit coefficients only', 1, 0, 0, 1),
    ('REESTIMATE_LAMBDA', 'Refit coefficients and REML lambdas', 1, 1, 0, 1),
    ('FULL_ADAPTIVE', 'Refit coefficients, lambdas, and data-driven knots', 1, 1, 1, 1)
ON CONFLICT(variant_code) DO NOTHING;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_VARIANT_IMMUTABLE_UPDATE
BEFORE UPDATE ON MODEL_MONITOR_VARIANT
BEGIN
    SELECT RAISE(ABORT, 'monitoring variant policy is immutable');
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_VARIANT_IMMUTABLE_DELETE
BEFORE DELETE ON MODEL_MONITOR_VARIANT
BEGIN
    SELECT RAISE(ABORT, 'monitoring variant policy is immutable');
END;

CREATE TABLE IF NOT EXISTS pricing.MODEL_FIT_CONTRACT (
    fit_contract_id TEXT NOT NULL PRIMARY KEY,
    baseline_model_run_id TEXT NOT NULL UNIQUE,
    model_id INTEGER NOT NULL,
    rate_package_id INTEGER NOT NULL,
    contract_schema_version INTEGER NOT NULL CHECK (contract_schema_version >= 1),
    contract_sha256 TEXT NOT NULL
        CHECK (
            length(contract_sha256) = 64
            AND contract_sha256 = lower(contract_sha256)
            AND contract_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
    structure_sha256 TEXT NOT NULL
        CHECK (
            length(structure_sha256) = 64
            AND structure_sha256 = lower(structure_sha256)
            AND structure_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
    contract_json TEXT NOT NULL CHECK (json_valid(contract_json)),
    superglm_version TEXT NOT NULL,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL,
    FOREIGN KEY (baseline_model_run_id) REFERENCES MODEL_RUN(model_run_id),
    FOREIGN KEY (model_id) REFERENCES PRICING_MODEL(model_id),
    FOREIGN KEY (rate_package_id) REFERENCES PRICING_RATE_PACKAGE(rate_package_id)
);

CREATE TABLE IF NOT EXISTS pricing.MODEL_MONITOR_RUN (
    monitor_run_id TEXT NOT NULL PRIMARY KEY,
    fit_contract_id TEXT NOT NULL,
    baseline_deployment_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    rate_package_id INTEGER NOT NULL,
    manifest_id TEXT NOT NULL,
    component_role TEXT NOT NULL
        CHECK (component_role IN ('FREQUENCY', 'SEVERITY', 'OTHER')),
    variant_code TEXT NOT NULL,
    run_signature_sha256 TEXT NOT NULL UNIQUE
        CHECK (
            length(run_signature_sha256) = 64
            AND run_signature_sha256 = lower(run_signature_sha256)
            AND run_signature_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
    run_status TEXT NOT NULL CHECK (run_status IN ('SUCCESS', 'FAILED')),
    invariant_status TEXT NOT NULL
        CHECK (invariant_status IN ('VERIFIED', 'LEGACY_UNVERIFIED')),
    invariant_evidence_sha256 TEXT,
    invariant_evidence_json TEXT,
    model_frame_sha256 TEXT NOT NULL
        CHECK (
            length(model_frame_sha256) = 64
            AND model_frame_sha256 = lower(model_frame_sha256)
            AND model_frame_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
    fit_configuration_json TEXT NOT NULL CHECK (json_valid(fit_configuration_json)),
    result_evidence_sha256 TEXT NOT NULL
        CHECK (
            length(result_evidence_sha256) = 64
            AND result_evidence_sha256 = lower(result_evidence_sha256)
            AND result_evidence_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
    started_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL,
    CHECK (completed_ts >= started_ts),
    CHECK (
        (
            invariant_status = 'VERIFIED'
            AND invariant_evidence_sha256 IS NOT NULL
            AND length(invariant_evidence_sha256) = 64
            AND invariant_evidence_sha256 = lower(invariant_evidence_sha256)
            AND invariant_evidence_sha256 NOT GLOB '*[^0-9a-f]*'
            AND invariant_evidence_json IS NOT NULL
            AND json_valid(invariant_evidence_json)
        )
        OR (
            invariant_status = 'LEGACY_UNVERIFIED'
            AND invariant_evidence_sha256 IS NULL
            AND invariant_evidence_json IS NULL
        )
    ),
    UNIQUE (baseline_deployment_id, manifest_id, component_role, variant_code),
    FOREIGN KEY (fit_contract_id) REFERENCES MODEL_FIT_CONTRACT(fit_contract_id),
    FOREIGN KEY (baseline_deployment_id)
        REFERENCES PRICING_MODEL_DEPLOYMENT(deployment_id),
    FOREIGN KEY (model_id) REFERENCES PRICING_MODEL(model_id),
    FOREIGN KEY (rate_package_id) REFERENCES PRICING_RATE_PACKAGE(rate_package_id),
    FOREIGN KEY (manifest_id) REFERENCES DATASET_MANIFEST(manifest_id),
    FOREIGN KEY (variant_code) REFERENCES MODEL_MONITOR_VARIANT(variant_code)
);

CREATE TABLE IF NOT EXISTS pricing.MODEL_MONITOR_TERM (
    monitor_run_id TEXT NOT NULL,
    term_name TEXT NOT NULL,
    term_kind TEXT NOT NULL,
    sequence_no INTEGER NOT NULL CHECK (sequence_no >= 1),
    term_structure_sha256 TEXT NOT NULL
        CHECK (
            length(term_structure_sha256) = 64
            AND term_structure_sha256 = lower(term_structure_sha256)
            AND term_structure_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
    term_metadata_json TEXT NOT NULL CHECK (json_valid(term_metadata_json)),
    PRIMARY KEY (monitor_run_id, term_name),
    UNIQUE (monitor_run_id, sequence_no),
    FOREIGN KEY (monitor_run_id) REFERENCES MODEL_MONITOR_RUN(monitor_run_id)
);

CREATE TABLE IF NOT EXISTS pricing.MODEL_MONITOR_LAMBDA (
    monitor_run_id TEXT NOT NULL,
    component_name TEXT NOT NULL,
    term_name TEXT,
    lambda_value REAL NOT NULL CHECK (lambda_value >= 0),
    lambda_mode TEXT NOT NULL CHECK (lambda_mode IN ('BASELINE', 'FIXED', 'ESTIMATED')),
    PRIMARY KEY (monitor_run_id, component_name),
    FOREIGN KEY (monitor_run_id) REFERENCES MODEL_MONITOR_RUN(monitor_run_id)
);

CREATE TABLE IF NOT EXISTS pricing.MODEL_MONITOR_RELATIVITY (
    monitor_run_id TEXT NOT NULL,
    term_name TEXT NOT NULL,
    term_kind TEXT NOT NULL,
    point_key TEXT NOT NULL,
    point_label TEXT,
    point_numeric REAL,
    relativity REAL NOT NULL CHECK (relativity > 0),
    log_relativity REAL NOT NULL,
    is_reference INTEGER NOT NULL CHECK (is_reference IN (0, 1)),
    PRIMARY KEY (monitor_run_id, term_name, point_key),
    CHECK (
        (point_label IS NULL AND point_numeric IS NOT NULL)
        OR (point_label IS NOT NULL AND point_numeric IS NULL)
    ),
    FOREIGN KEY (monitor_run_id) REFERENCES MODEL_MONITOR_RUN(monitor_run_id)
);

CREATE TABLE IF NOT EXISTS pricing.MODEL_MONITOR_METRIC (
    monitor_run_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    PRIMARY KEY (monitor_run_id, metric_name),
    FOREIGN KEY (monitor_run_id) REFERENCES MODEL_MONITOR_RUN(monitor_run_id)
);

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_FIT_CONTRACT_LINEAGE_INSERT
BEFORE INSERT ON MODEL_FIT_CONTRACT
WHEN NOT EXISTS (
    SELECT 1
    FROM MODEL_RUN AS baseline_run
    JOIN PRICING_RATE_PACKAGE AS package
      ON package.rate_package_id = NEW.rate_package_id
    WHERE baseline_run.model_run_id = NEW.baseline_model_run_id
      AND baseline_run.model_id = NEW.model_id
      AND baseline_run.rate_package_id = NEW.rate_package_id
      AND baseline_run.run_status = 'SUCCESS'
      AND package.model_id = NEW.model_id
      AND package.package_status = 'PUBLISHED'
)
BEGIN
    SELECT RAISE(
        ABORT,
        'fit contract must identify one successful published baseline run'
    );
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_FIT_CONTRACT_IMMUTABLE_UPDATE
BEFORE UPDATE ON MODEL_FIT_CONTRACT
BEGIN
    SELECT RAISE(ABORT, 'model fit contracts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_FIT_CONTRACT_IMMUTABLE_DELETE
BEFORE DELETE ON MODEL_FIT_CONTRACT
BEGIN
    SELECT RAISE(ABORT, 'model fit contracts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_RUN_LINEAGE_INSERT
BEFORE INSERT ON MODEL_MONITOR_RUN
WHEN NOT EXISTS (
    SELECT 1
    FROM MODEL_FIT_CONTRACT AS contract
    JOIN MODEL_RUN AS baseline_run
      ON baseline_run.model_run_id = contract.baseline_model_run_id
    JOIN PRICING_MODEL_DEPLOYMENT AS deployment
      ON deployment.deployment_id = NEW.baseline_deployment_id
    WHERE contract.fit_contract_id = NEW.fit_contract_id
      AND contract.model_id = NEW.model_id
      AND contract.rate_package_id = NEW.rate_package_id
      AND baseline_run.model_id = NEW.model_id
      AND baseline_run.rate_package_id = NEW.rate_package_id
      AND deployment.model_id = NEW.model_id
      AND deployment.rate_package_id = NEW.rate_package_id
)
BEGIN
    SELECT RAISE(
        ABORT,
        'monitoring contract, run, and deployment must identify one model package'
    );
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_RUN_LINEAGE_UPDATE
BEFORE UPDATE OF fit_contract_id, baseline_deployment_id, model_id, rate_package_id
ON MODEL_MONITOR_RUN
WHEN NOT EXISTS (
    SELECT 1
    FROM MODEL_FIT_CONTRACT AS contract
    JOIN MODEL_RUN AS baseline_run
      ON baseline_run.model_run_id = contract.baseline_model_run_id
    JOIN PRICING_MODEL_DEPLOYMENT AS deployment
      ON deployment.deployment_id = NEW.baseline_deployment_id
    WHERE contract.fit_contract_id = NEW.fit_contract_id
      AND contract.model_id = NEW.model_id
      AND contract.rate_package_id = NEW.rate_package_id
      AND baseline_run.model_id = NEW.model_id
      AND baseline_run.rate_package_id = NEW.rate_package_id
      AND deployment.model_id = NEW.model_id
      AND deployment.rate_package_id = NEW.rate_package_id
)
BEGIN
    SELECT RAISE(
        ABORT,
        'monitoring contract, run, and deployment must identify one model package'
    );
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_DEPLOYMENT_MONITORING_LINEAGE_GUARD_UPDATE
BEFORE UPDATE OF deployment_slot, model_id, rate_package_id, effective_from_ts
ON PRICING_MODEL_DEPLOYMENT
WHEN EXISTS (
    SELECT 1
    FROM MODEL_MONITOR_RUN AS monitor_run
    WHERE monitor_run.baseline_deployment_id = OLD.deployment_id
)
AND (
    NEW.deployment_slot IS NOT OLD.deployment_slot
    OR NEW.model_id IS NOT OLD.model_id
    OR NEW.rate_package_id IS NOT OLD.rate_package_id
    OR NEW.effective_from_ts IS NOT OLD.effective_from_ts
)
BEGIN
    SELECT RAISE(
        ABORT,
        'deployment referenced by monitoring evidence has immutable lineage'
    );
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_DEPLOYMENT_MONITORING_LINEAGE_GUARD_DELETE
BEFORE DELETE ON PRICING_MODEL_DEPLOYMENT
WHEN EXISTS (
    SELECT 1
    FROM MODEL_MONITOR_RUN AS monitor_run
    WHERE monitor_run.baseline_deployment_id = OLD.deployment_id
)
BEGIN
    SELECT RAISE(
        ABORT,
        'deployment referenced by monitoring evidence has immutable lineage'
    );
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_DATASET_MANIFEST_MONITORING_LINEAGE_GUARD_UPDATE
BEFORE UPDATE ON DATASET_MANIFEST
WHEN EXISTS (
    SELECT 1
    FROM MODEL_MONITOR_RUN AS monitor_run
    WHERE monitor_run.manifest_id = OLD.manifest_id
)
BEGIN
    SELECT RAISE(
        ABORT,
        'dataset manifest referenced by monitoring evidence is immutable'
    );
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_DATASET_MANIFEST_MONITORING_LINEAGE_GUARD_DELETE
BEFORE DELETE ON DATASET_MANIFEST
WHEN EXISTS (
    SELECT 1
    FROM MODEL_MONITOR_RUN AS monitor_run
    WHERE monitor_run.manifest_id = OLD.manifest_id
)
BEGIN
    SELECT RAISE(
        ABORT,
        'dataset manifest referenced by monitoring evidence is immutable'
    );
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_RUN_IMMUTABLE_UPDATE
BEFORE UPDATE ON MODEL_MONITOR_RUN
BEGIN
    SELECT RAISE(ABORT, 'monitoring evidence is immutable');
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_RUN_IMMUTABLE_DELETE
BEFORE DELETE ON MODEL_MONITOR_RUN
BEGIN
    SELECT RAISE(ABORT, 'monitoring evidence is immutable');
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_TERM_IMMUTABLE_UPDATE
BEFORE UPDATE ON MODEL_MONITOR_TERM
BEGIN
    SELECT RAISE(ABORT, 'monitoring evidence is immutable');
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_TERM_IMMUTABLE_DELETE
BEFORE DELETE ON MODEL_MONITOR_TERM
BEGIN
    SELECT RAISE(ABORT, 'monitoring evidence is immutable');
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_LAMBDA_IMMUTABLE_UPDATE
BEFORE UPDATE ON MODEL_MONITOR_LAMBDA
BEGIN
    SELECT RAISE(ABORT, 'monitoring evidence is immutable');
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_LAMBDA_IMMUTABLE_DELETE
BEFORE DELETE ON MODEL_MONITOR_LAMBDA
BEGIN
    SELECT RAISE(ABORT, 'monitoring evidence is immutable');
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_RELATIVITY_IMMUTABLE_UPDATE
BEFORE UPDATE ON MODEL_MONITOR_RELATIVITY
BEGIN
    SELECT RAISE(ABORT, 'monitoring evidence is immutable');
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_RELATIVITY_IMMUTABLE_DELETE
BEFORE DELETE ON MODEL_MONITOR_RELATIVITY
BEGIN
    SELECT RAISE(ABORT, 'monitoring evidence is immutable');
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_METRIC_IMMUTABLE_UPDATE
BEFORE UPDATE ON MODEL_MONITOR_METRIC
BEGIN
    SELECT RAISE(ABORT, 'monitoring evidence is immutable');
END;

CREATE TRIGGER IF NOT EXISTS pricing.TR_MODEL_MONITOR_METRIC_IMMUTABLE_DELETE
BEFORE DELETE ON MODEL_MONITOR_METRIC
BEGIN
    SELECT RAISE(ABORT, 'monitoring evidence is immutable');
END;

CREATE TABLE IF NOT EXISTS pricing.PRICING_FEATURE (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL UNIQUE,
    feature_value_type TEXT NOT NULL,
    is_ordered INTEGER NOT NULL DEFAULT 0,
    active_flag INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_FEATURE_LEVEL_SET (
    level_set_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER,
    feature_id INTEGER NOT NULL,
    level_set_name TEXT NOT NULL,
    level_set_type TEXT NOT NULL,
    binning_strategy TEXT,
    grid_width REAL,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_id, feature_id, level_set_name)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_FEATURE_LEVEL (
    feature_level_id INTEGER PRIMARY KEY AUTOINCREMENT,
    level_set_id INTEGER NOT NULL,
    level_code TEXT NOT NULL,
    level_label TEXT,
    order_index INTEGER,
    lower_bound REAL,
    upper_bound REAL,
    representative_value REAL,
    is_missing INTEGER NOT NULL DEFAULT 0,
    is_other INTEGER NOT NULL DEFAULT 0,
    UNIQUE (level_set_id, level_code)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_TERM (
    term_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rate_package_id INTEGER NOT NULL,
    term_name TEXT NOT NULL,
    term_type TEXT NOT NULL,
    sequence_no INTEGER NOT NULL,
    default_multiplier REAL NOT NULL DEFAULT 1.0,
    default_log_coefficient REAL NOT NULL DEFAULT 0.0,
    term_metadata_json TEXT,
    active_flag INTEGER NOT NULL DEFAULT 1,
    UNIQUE (rate_package_id, term_name)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_TERM_FEATURE (
    term_id INTEGER NOT NULL,
    position_no INTEGER NOT NULL,
    feature_id INTEGER NOT NULL,
    level_set_id INTEGER NOT NULL,
    input_column_name TEXT,
    PRIMARY KEY (term_id, position_no)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_RATE_CELL (
    cell_id INTEGER PRIMARY KEY AUTOINCREMENT,
    term_id INTEGER NOT NULL,
    cell_key_text TEXT NOT NULL,
    cell_key_digest TEXT NOT NULL,
    multiplier REAL NOT NULL,
    log_coefficient REAL NOT NULL,
    exposure_weight REAL,
    record_count INTEGER,
    is_reference INTEGER NOT NULL DEFAULT 0,
    is_default INTEGER NOT NULL DEFAULT 0,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    UNIQUE (term_id, cell_key_digest)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_RATE_CELL_LEVEL (
    cell_id INTEGER NOT NULL,
    position_no INTEGER NOT NULL,
    feature_level_id INTEGER NOT NULL,
    PRIMARY KEY (cell_id, position_no)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_COMPILED_RATE_CELL (
    rate_package_id INTEGER NOT NULL,
    term_id INTEGER NOT NULL,
    cell_key_digest TEXT NOT NULL,
    term_name TEXT NOT NULL,
    term_type TEXT NOT NULL,
    sequence_no INTEGER NOT NULL,
    cell_key_text TEXT NOT NULL,
    multiplier REAL NOT NULL,
    log_coefficient REAL NOT NULL,
    exposure_weight REAL,
    record_count INTEGER,
    is_default INTEGER NOT NULL,
    is_reference INTEGER NOT NULL,
    PRIMARY KEY (rate_package_id, term_id, cell_key_digest)
);

CREATE TABLE IF NOT EXISTS pricing.PRICING_COMPILED_1D_RATE_BAND (
    rate_package_id INTEGER NOT NULL,
    term_id INTEGER NOT NULL,
    feature_level_id INTEGER NOT NULL,
    term_name TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    level_code TEXT NOT NULL,
    sort_order INTEGER NOT NULL,
    lower_bound REAL,
    upper_bound REAL,
    representative_value REAL,
    multiplier REAL NOT NULL,
    log_coefficient REAL NOT NULL,
    PRIMARY KEY (rate_package_id, term_id, sort_order, feature_level_id)
);
