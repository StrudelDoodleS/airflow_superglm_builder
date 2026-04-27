# Airflow SuperGLM Pricing Pipeline Design

## Purpose

Build a production-minded local pipeline that uses Apache Airflow 3.2.1 to load
the freMTPL dataset into an external SQL Server pricing database, train a
SuperGLM Poisson frequency model, export deployment rating tables, track the run
in MLflow, and publish normalized rating tables back to SQL Server.

The first implementation is intentionally narrow: one freMTPL frequency model
using `ClaimNb` as the target and `log(Exposure)` as the model offset. The design
keeps boundaries clear so later changes to data sources, model families, rating
table shape, or production infrastructure can be made without replacing the
whole pipeline.

## Current Starter Context

The starter archive already contains useful pieces:

- SQL Server DDL migrations for dataset metadata, staging tables, normalized
  pricing package tables, and compiled rating cells.
- Python scripts for applying migrations, loading freMTPL row-key metadata,
  parsing SuperGLM-style Excel rating tables, loading staged rating tables into
  normalized pricing tables, and inspecting a package.
- A minimal `docker-compose.yml` with SQL Server and CloudBeaver.
- Example Airflow 2 style DAGs that call the scripts with Bash operators.

The main gaps are:

- freMTPL is not stored as a full source table in SQL Server.
- SuperGLM training is not yet part of Airflow.
- MLflow tracking is not present.
- The compose file is not based on the official Airflow 3.2.1 layout.
- Durable local state is not clearly separated from Compose-owned volumes.

## Sources Checked

- Apache Airflow Docker Compose guide:
  https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html
- Apache Airflow 3.2.1 compose file:
  https://airflow.apache.org/docs/apache-airflow/3.2.1/docker-compose.yaml
- Apache Airflow release notes, including Python 3.14 support:
  https://airflow.apache.org/docs/apache-airflow/stable/release_notes.html
- Apache Airflow Docker image tags:
  https://hub.docker.com/r/apache/airflow/tags
- SuperGLM repository:
  https://github.com/StrudelDoodleS/superglm
- SuperGLM rating-table PR:
  https://github.com/StrudelDoodleS/superglm/pull/109
- MLflow tracking server architecture:
  https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/
- MLflow artifact store architecture:
  https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/

## Architecture

Use the official Airflow 3.2.1 CeleryExecutor compose structure as the baseline:

- `airflow-apiserver`
- `airflow-scheduler`
- `airflow-dag-processor`
- `airflow-worker`
- `airflow-triggerer`
- Airflow metadata Postgres
- Redis broker

Extend that compose file with:

- A custom Airflow image based on `apache/airflow:3.2.1-python3.14`.
- SQL Server as the external pricing database for local development.
- Optional CloudBeaver for SQL inspection.
- MLflow tracking server.
- Durable project-local state directories mounted as bind mounts.

Airflow metadata remains in Postgres and is treated as orchestration state.
Pricing data, training lineage, rating staging tables, and deployable rating
packages live in SQL Server. MLflow tracks model experiments and points to the
model/rating-table artifacts.

Python 3.14 is the preferred runtime. If a dependency build or smoke test shows
that a required package does not work under Python 3.14, fallback to an Airflow
3.2.1 Python 3.13 image is allowed only after capturing the build failure.
Airflow itself supports Python 3.14 in the 3.2.x line.

## Custom Airflow Image

The custom image will install:

- Microsoft ODBC Driver 18 for SQL Server.
- `unixodbc-dev` and related OS packages needed by `pyodbc`.
- Python dependencies:
  - `superglm @ git+https://github.com/StrudelDoodleS/superglm.git`
  - `mlflow`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `sqlalchemy`
  - `pyodbc`
  - `python-dotenv`
  - `openpyxl`
  - `apache-airflow-providers-microsoft-mssql`

The implementation will prefer a built image over runtime
`_PIP_ADDITIONAL_REQUIREMENTS` because Airflow's own docs describe runtime pip
installation as a quick-check option only.

## Durable State

Use a project-local `state/` directory for local durable files. It is not
committed to git.

```text
state/
  mssql/
    data/
  mlflow/
    artifacts/
  rating_exports/
    MTPL_FREQ/
      2026-04-27/
        mtpl_freq__20260427T103000Z/
          rating_tables.xlsx
          model_summary.txt
          training_metrics.json
```

Compose should bind-mount these directories rather than create anonymous or
Compose-owned volumes:

```yaml
./state/mssql/data:/var/opt/mssql
./state/mlflow/artifacts:/mlflow/artifacts
./state/rating_exports:/opt/pricing/state/rating_exports
```

This does not make data impossible to delete, but it prevents accidental removal
through normal Compose volume cleanup. Documentation should avoid
`docker compose down -v` for this project.

## MLflow

MLflow will run as a separate service.

- Tracking URI for Airflow tasks: `http://mlflow:5000`.
- Backend store: SQL Server database such as `MLflowTracking`, using a
  SQLAlchemy/pyodbc connection string.
- Artifact store: `/mlflow/artifacts`, backed by
  `./state/mlflow/artifacts`.

The training task logs:

- model parameters and feature configuration
- training metrics
- validation metrics
- fitted model artifact
- rating table workbook
- compact model summary
- run tags linking Airflow run id, dataset manifest id, export id, and rate
  package id when available

The rating workbook is saved first to the project-local rating export directory,
then logged to MLflow. SQL loading uses the saved workbook path, so the workbook
used to publish a rating package is both human-accessible and linked to the
MLflow run.

If MLflow's SQL Server backend fails compatibility checks, only the MLflow
backend store should move to the Airflow Postgres service or a dedicated
Postgres service. Pricing data and rating packages remain in SQL Server.

## SQL Server Data Model

Use the existing `pricing` schema and add the missing pieces.

### Source Dataset

`pricing.FREMTPL_RAW`

Stores freMTPL as fetched from OpenML, as-is. No cleaning or derived model-ready
columns are written into this raw table. The first implementation should preserve
column names and values from the fetched DataFrame. Load metadata belongs in the
manifest and lineage tables, not in `pricing.FREMTPL_RAW`.

Expected identifying metadata outside the raw table:

- OpenML data id
- dataset name
- load timestamp
- source row ordinal if no stable key exists

### Dataset Metadata

Existing starter tables remain useful:

- `pricing.DATASET_MANIFEST`
- `pricing.DATASET_COLUMN`
- `pricing.DATASET_ROW_KEY`
- `pricing.CV_SPLIT`
- `pricing.STG_DATASET_ROW_KEY`

The current manifest loader should be changed so it records full-table metadata
for `pricing.FREMTPL_RAW`, not only row-key metadata from a transient OpenML
fetch.

### Model Run Lineage

Add `pricing.MODEL_RUN`.

Purpose: bridge operational lineage across Airflow, MLflow, dataset snapshot,
rating export, and final rate package.

Fields should include:

- model run id
- Airflow DAG id
- Airflow run id
- MLflow experiment id
- MLflow run id
- dataset manifest id
- export id
- model name
- model version
- rate package id
- rating workbook path
- run status
- timestamps
- created by

### Rating Tables

Keep and extend the starter's rating table structures:

- `pricing.STG_RATING_EXPORT`
- `pricing.STG_RATE_CELL`
- `pricing.STG_CELL_LEVEL`
- `pricing.PRICING_RATE_PACKAGE`
- `pricing.PRICING_PACKAGE_POINTER`
- `pricing.PRICING_FEATURE`
- `pricing.PRICING_FEATURE_LEVEL_SET`
- `pricing.PRICING_FEATURE_LEVEL`
- `pricing.PRICING_TERM`
- `pricing.PRICING_TERM_FEATURE`
- `pricing.PRICING_RATE_CELL`
- `pricing.PRICING_RATE_CELL_LEVEL`
- `pricing.PRICING_COMPILED_RATE_CELL`
- `pricing.PRICING_COMPILED_1D_RATE_BAND`

The SuperGLM PR adds `model.export_rating_tables(...)`. The starter's Excel
parser already expects a workbook with `Rating Tables`, base rate in `C2`, term
names on row 5, headers on row 7, and data from row 8. That aligns with the new
SuperGLM export layout and should be reused where possible.

## Airflow DAG

One first DAG should own the complete training and publishing flow:

```text
apply_migrations
  -> load_fremtpl_raw
  -> create_dataset_manifest
  -> train_superglm
  -> export_rating_tables
  -> stage_rating_export
  -> publish_rating_package
  -> record_model_run
  -> inspect_package
```

### `apply_migrations`

Applies idempotent SQL Server migrations.

### `load_fremtpl_raw`

Fetches freMTPL from OpenML and writes the dataset as-is to
`pricing.FREMTPL_RAW`. The load should be idempotent so rerunning the DAG does
not duplicate rows for the same source snapshot.

### `create_dataset_manifest`

Reads `pricing.FREMTPL_RAW`, records metadata, computes deterministic row keys,
and creates deterministic cross-validation folds.

### `train_superglm`

Reads training data from SQL Server, trains a Poisson SuperGLM frequency model,
and logs the run to MLflow.

The initial model uses:

- target: `ClaimNb`
- offset: `log(Exposure)`
- family: Poisson
- recommended fit path: `fit_reml()` with `selection_penalty=0.0`
- initial features:
  - `VehAge`: spline
  - `DrivAge`: spline
  - `BonusMalus`: spline
  - `Density`: transformed to `LogDensity` in the training frame, then used as
    a numeric feature
  - `Area`: categorical
  - `VehPower`: categorical
  - `VehBrand`: categorical
  - `VehGas`: categorical
  - `Region`: categorical

The raw SQL table remains unchanged when `LogDensity` is derived. If the OpenML
schema differs from these expected columns, the task should fail with a clear
schema validation error.

The task should persist the fitted model artifact and compact summary.

### `export_rating_tables`

Calls `model.export_rating_tables(...)`, writes the workbook under
`state/rating_exports`, and logs it to MLflow.

### `stage_rating_export`

Parses the saved workbook into staging tables.

### `publish_rating_package`

Loads staged cells into normalized pricing tables, compiles deployable rating
tables, and updates a named pointer such as `MTPL_FREQ_UAT`.

### `record_model_run`

Writes or updates `pricing.MODEL_RUN` with lineage identifiers and paths.

### `inspect_package`

Prints a compact package summary to Airflow logs.

## Error Handling And Idempotency

- Migration scripts are idempotent and tracked in `dbo.SCHEMA_MIGRATION`.
- freMTPL raw loading uses a source snapshot key to avoid duplicate loads.
- Dataset manifest creation creates a new immutable manifest id per data
  snapshot.
- Rating export ids should be unique per training run.
- Staging loaders support `--replace` for the same export id.
- Publishing a rating package creates a new package version, then atomically
  updates the package pointer.
- MLflow logging should tag failed runs and still preserve partial logs when a
  later SQL publishing step fails.

## Testing And Verification

Minimum tests and checks for the first implementation:

- Unit tests for SQL batch splitting and migration discovery.
- Unit tests for freMTPL raw table schema creation/idempotent load behavior.
- Unit tests for rating workbook parser compatibility with SuperGLM's current
  workbook layout.
- A smoke test that imports SuperGLM in the custom Airflow image and verifies
  `SuperGLM.export_rating_tables` exists.
- A Docker Compose smoke test:
  - start SQL Server, MLflow, and Airflow dependencies
  - apply migrations
  - run the DAG or task sequence on a small sample
  - verify a row exists in `pricing.MODEL_RUN`
  - verify a package pointer exists
  - verify the rating workbook exists under `state/rating_exports`
  - verify the workbook is logged to MLflow

## Initial Scope

In scope for the first build:

- Unpack/adapt the starter project into a real repo layout.
- Airflow 3.2.1 compose baseline.
- Python 3.14 custom Airflow image where dependency builds pass.
- Project-local durable state directories.
- SQL Server pricing database.
- MLflow tracking service.
- freMTPL raw load into SQL Server.
- Poisson SuperGLM training from SQL Server.
- Rating workbook export and MLflow artifact logging.
- Rating workbook staging and normalized package publishing.

Out of scope for the first build:

- Cloud object storage.
- Kubernetes or Helm deployment.
- Multiple model families.
- Champion/challenger promotion workflow.
- A UI beyond Airflow, MLflow, and SQL inspection.
- Production secrets management beyond environment-based local configuration.

## Expected Evolution

This design is a starting point. Likely future changes include:

- replacing OpenML with internal source data
- adding severity or pure premium models
- adding richer validation and model comparison
- introducing cloud object storage for MLflow artifacts
- adding promotion gates before pointer updates
- replacing local SQL Server with managed SQL Server
- separating training execution from Airflow workers if model dependencies grow
  too heavy for the Airflow image

The first implementation should keep each step as a callable Python module so
Airflow orchestration is thin and future changes stay localized.
