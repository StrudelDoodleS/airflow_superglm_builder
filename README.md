# SuperGLM pricing workbench

This repository provides a notebook-first workflow for fitting a SuperGLM,
recording its audit evidence, publishing immutable SQL rating packages, making
optional market edits, and explicitly deploying a reviewed package.

The analyst owns the visible modeling decisions. The library owns generated
identifiers, model versions, dataset and validation records, artifact hashes,
lineage, package versions, parent links, and deployment concurrency checks.

Airflow is not part of the approved workflow. When durable orchestration becomes
available, it can call the same typed publication/deployment functions without
changing the analyst notebook or introducing a second model configuration.

## The analyst workflow

Create a model notebook:

```bash
uv run python scripts/scaffold_pricing_model.py \
  --model-name CLAIM_FREQUENCY \
  --target-name claim_count \
  --model-label "Claim frequency"
```

This creates only:

```text
pricing_models/claim_frequency/
├── __init__.py
└── pricing_model.ipynb
```

Open the notebook and work from top to bottom:

1. Select local or remote database mode in the first cell.
2. Load the source data and perform visible feature transforms in Python.
3. Declare `PricingModelSpec` and the validation strategy.
4. Define the feature objects and `SuperGLM` with ordinary Python.
5. Call `register_model`, `build_candidate`, and `publish_candidate`.
6. Optionally open the published package in the editor and publish retained
   edits.
7. Optionally deploy the exact package that was opened and reviewed.

There is no generated training module, DAG factory, TOML model factory, or
analyst-facing metadata form. The notebook is the model definition.

The bundled reference is
`pricing_models/mtpl_frequency/pricing_model.ipynb`.

## What remains visible

The analyst supplies the decisions that genuinely belong to the model. Offset
transforms are ordinary visible Python too:

```python
frame["term_offset"] = np.log(frame["term"] / 12.0)

MODEL = PricingModelSpec(
    name="CLAIM_FREQUENCY",
    label="Claim frequency",
    target="claim_count",
    model_type="superglm_poisson",
    deployment_slot="CLAIM_FREQUENCY_UAT",
    features=("driver_age", "vehicle_age", "region"),
    dataset_name="claim_frequency_model_frame",
    source_system="pricing_sql",
    pk_columns=("policy_id",),
    offset_column="term_offset",
    offset_source_column="term",
    offset_label="log(term / 12)",
    sample_weight_column="model_weight",
    export_weight_column="rating_table_weight",
    data_as_of_column="data_as_of",
    validation=ValidationSplitConfig.kfold(
        n_splits=5,
        random_state=42,
        shuffle=True,
    ),
)
```

`offset_column` is passed to fitting exactly as stored; the pipeline never logs
it. `offset_source_column` preserves the upstream values used as workbook levels
(for example, `term` 12 and 36 produce relativities 1 and 3 after fitting
`log(term / 12)`). `sample_weight_column` affects fitting only, while
`export_weight_column` affects rating-table aggregation only. These inputs do
not fall back to one another. Set any unused field to `None`.

Feature transforms also remain normal Python:

```python
frame["log_density"] = np.log1p(frame["density"])
frame["young_driver"] = frame["driver_age"].lt(25).astype(int)
```

The export receives the fitted model and final feature frame. There is no
separate transform language to keep in sync with the notebook. Consequently,
SQL scoring expects those final transformed columns; it does not reconstruct
`log1p(density)` from raw `density`. Any production data preparation must apply
the same visible transform before using the rating table.

Save the notebook before building. The source checksum is calculated from the
notebook file on disk; Jupyter cannot prove that unsaved in-memory cells match
that file.

`data_as_of` means the date through which the input data is complete. It can be
passed explicitly or derived from one constant-valued frame column. It is not a
deployment date. A candidate has no deployment date until it is actually
deployed.

Source-owned validation columns are supported with
`ValidationSplitConfig.column_kfold(...)` and
`ValidationSplitConfig.column_holdout(...)`. The notebook supports generated
k-fold/holdout and these two column-based strategies; it materializes the exact
fold evidence automatically.

## What is recorded automatically

Building and publishing records the following without additional notebook
arguments:

- stable SQL model ID and generated trained-model version;
- dataset name, source system, data-as-of date, exact final-frame SHA-256, row
  count, primary-key columns, and explicit
  key/feature/target/offset/offset-source/sample-weight/export-weight/data-as-of
  roles;
- ordered column names, dtypes, null counts, distinct counts, and the runtime
  metadata needed to interpret the frame checksum;
- validation method, parameters, folds, split membership evidence, and split
  artifact checksum when materialized;
- model source checksum and runtime dependency metadata;
- candidate bundle path, format, size, checksum, Python version, and SuperGLM
  version;
- rating workbook and publication-receipt checksums, rechecked across staging;
- model-run and fold metrics with their scope;
- immutable rate-package version and status;
- edited package parent, parent model run, edit reason, and claimed identity;
- deployment slot, prior champion, selected package, reason, and deployer.

The SQL database is the audit source of truth. The schema retains an optional
`mlflow_run_id`, but the current notebook workflow does not create or log MLflow
runs. MLflow is not a required registry or artifact store.

Actor text such as `created_by` defaults to the local notebook username. It is
useful attribution, not authentication. Remote database permissions and the
private runtime connection remain the actual write-control boundary.

Each call to `build_candidate` creates a new trained artifact and receives its
version from the store. Re-publishing the same export with the same evidence is
idempotent; incompatible reuse is rejected. Analysts never type a model or
package version.

## Local mode

The generated notebook starts in local mode:

```python
DATABASE_MODE = "local"
RUNTIME_MODULE = None
EXPECTED_REMOTE_DATABASE = ""
ALLOW_REMOTE_WRITES = False
```

`connect(mode="local", ...)` creates persistent SQLite databases under the
model's `.local/` directory. That directory is ignored by Git. Local mode writes
real dataset, split, model-run, metric, package, and artifact audit records; it
does not deploy a live package or run the SuperGLM editor. Local package rows use
the explicit `LOCAL_AUDIT` status so they cannot be mistaken for remotely
published rating packages.

Nothing is written to a server in local mode. Deleting `.local/` deletes that
model's local audit store, so retain it for as long as the local evidence is
needed.

## Remote SQL Server mode

Keep work-specific connectivity in a private importable module. Do not commit
server names, tokens, passwords, or copied connection code to this repository.
The module needs `get_engine()` and may provide schema/runtime settings:

```python
# work_runtime/database.py -- private work code, shown only as an interface
def get_engine(database=None):
    ...  # return the SQLAlchemy engine that already works at work

def get_schema_names():
    return {
        "pricing": "python_pricing",
        "pricing_staging": "python_pricing_stg",
        "mlops": "python_mlops",
    }

def get_runtime_settings():
    return {
        "pricing_database": "PricingAudit",
        "workbench_artifact_root": "state/workbench_artifacts",
        "validation_split_artifact_root": "state/validation_splits",
        "mlflow_enabled": False,
        "skip_database_create": True,
    }
```

Set the notebook globals only after confirming the target:

```python
DATABASE_MODE = "remote"
RUNTIME_MODULE = "work_runtime.database"
EXPECTED_REMOTE_DATABASE = "PricingAudit"
ALLOW_REMOTE_WRITES = True
```

Remote connection performs `SELECT DB_NAME()` and refuses writes when the
reported database does not exactly match `EXPECTED_REMOTE_DATABASE`.

The artifact root must be readable from every process that later opens or
publishes a candidate. On one Windows Cloud PC, keep Jupyter, Python, and any
optional Airflow process in the same WSL filesystem namespace. A transient local
artifact path is not durable storage: SQL audit rows will remain, but a lost
candidate bundle cannot later be reopened in the editor. Do not put a SQLite or
MLflow database on a OneDrive/SharePoint synced path; file syncing does not
provide safe concurrent database writes.

## Schema setup

Schema administration is separate from a pricing-model notebook. Apply the
versioned migrations with the same private runtime module:

```bash
uv run python scripts/apply_schema.py --runtime-module work_runtime.database
```

The command records each migration and checksum. Re-running it skips migrations
already applied and fails if an applied migration's contents changed. Use the
separate reset tooling only for an explicitly disposable database.

Local notebook mode applies its SQLite-compatible schema automatically.

## Editor and deployment rules

Publishing a trained candidate does not change a live deployment.

`open_candidate(...)` verifies the package, successful model run, dataset/split
lineage, candidate artifact checksum, runtime compatibility, and the bundle's
model name/version/export identity before loading it.

`publish_edits(...)` saves the authoritative editor session, replays it against
the verified parent model, publishes a child package, and records both package
and model-run parentage. The original package remains immutable.

`deploy_package(...)` accepts only a `Candidate` returned by `open_candidate`.
It carries the champion snapshot seen during review, so deployment fails rather
than silently overwriting a champion that changed in the meantime.

## Optional services

The notebook workflow needs no Docker and no local service manager. For separate
experiments, install the optional extra with `uv sync --extra mlflow` and start a
local server with `scripts/start_mlflow_local.py`; the pricing publication path
does not currently send data to it.

The SQL lineage and rate-package publication path does not depend on an MLflow
tracking database.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Never commit model-local `.local/` databases, notebook outputs, credentials, or
work connection modules.
