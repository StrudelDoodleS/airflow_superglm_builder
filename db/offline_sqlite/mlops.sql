CREATE TABLE IF NOT EXISTS mlops.MODEL_RUN_DATASET (
    model_run_id TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    dataset_role TEXT NOT NULL,
    PRIMARY KEY (model_run_id, dataset_role, manifest_id)
);

CREATE TABLE IF NOT EXISTS mlops.MODEL_RUN_SPLIT_SET (
    model_run_id TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    split_set_id TEXT NOT NULL,
    dataset_role TEXT NOT NULL,
    split_role TEXT NOT NULL,
    PRIMARY KEY (model_run_id, split_set_id, split_role)
);

CREATE TABLE IF NOT EXISTS mlops.MODEL_RUN_METRIC (
    model_run_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_scope TEXT,
    PRIMARY KEY (model_run_id, metric_name)
);
