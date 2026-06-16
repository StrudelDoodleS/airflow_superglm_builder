# Airflow SuperGLM Builder

Production-minded Airflow 3.2.1 pipeline for building, validating, publishing,
and deploying SQL-backed SuperGLM rate packages.

The reusable pipeline stores dataset manifests and model lineage in SQL Server,
trains model-specific SuperGLM estimators, optionally logs runs to MLflow,
exports normalized rating tables, and publishes immutable rate packages back to
SQL Server. The repository includes a freMTPL motor pricing model as a runnable
demo/reference implementation.

## Contents

- [Seed Database Schema](#seed-database-schema)
- [Docker Quickstart](#docker-quickstart)
- [No-Docker Work Quickstart](#no-docker-work-quickstart)
- [Adding Models](#adding-models)
- [Optional Local Tools](#optional-local-tools)
- [Demo Data](#demo-data)
- [Work SQL Server Targeting](#work-sql-server-targeting)
- [Local Smoke Run](#local-smoke-run)
- [Local Services](#local-services)
- [Database Diagrams](#database-diagrams)
- [Durable Local State](#durable-local-state)
- [Pricing Model History](#pricing-model-history)
- [Rate Package Lifecycle](#rate-package-lifecycle)
- [CV Split Storage](#cv-split-storage)
- [SQL Prediction Validation](#sql-prediction-validation)
- [Demo Model Variants](#demo-model-variants)

## Seed Database Schema

The target SQL Server database should already exist. Pick the schema names you
want for this project, then use a small Python script to render and execute the
DDL with a normal SQLAlchemy connection. The reusable DDL renderer lives in
`scripts/render_schema_sql.py`.

```python
# seed_pricing_schema.py
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from pricing_pipeline.infra.migrations import split_sql_server_batches
from scripts.render_schema_sql import render_schema_sql


# Replace this block with the SQLAlchemy create_engine(...) code that already
# works in your environment.
odbc_connect = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=<server>.database.windows.net,1433;"
    "DATABASE=<database>;"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
)
connection_url = URL.create(
    "mssql+pyodbc",
    query={"odbc_connect": odbc_connect},
)
engine = create_engine(connection_url, future=True)

schema_sql = render_schema_sql(
    Path("db/migrations"),
    pricing_schema="python_pricing",
    pricing_staging_schema="python_pricing_stg",
    mlops_schema="python_mlops",
)

with engine.begin() as conn:
    for batch in split_sql_server_batches(schema_sql):
        conn.exec_driver_sql(batch)
```

The rendered SQL creates the configured pricing/staging/mlops schemas, core
tables, lineage tables, views, stored procedures, triggers, and the
`dbo.SCHEMA_CONFIGURATION` / `dbo.SCHEMA_MIGRATION` housekeeping tables.
`dbo.SCHEMA_CONFIGURATION` records the selected schema names on first apply and
throws if a later apply asks for a different set.

## Docker Quickstart

These steps assume Docker with the Compose v2 plugin is already installed.

1. Clone the repository and create a local environment file:

   ```bash
   git clone <repo-url>
   cd airflow_superglm_builder
   cp .env.example .env
   ```

2. Review `.env`. The defaults are for a fully local Docker setup:

   ```env
   MSSQL_SERVER=mssql,1433
   MSSQL_DATABASE=PricingLab
   MLFLOW_DATABASE=MLflowTracking
   MSSQL_USER=sa
   MSSQL_PASSWORD=YourStrong(!)Password123
   MSSQL_ENCRYPT=no
   MSSQL_TRUST_SERVER_CERT=yes
   ```

3. Build the Airflow image and start the core stack:

   ```bash
   docker compose build
   docker compose up -d
   ```

4. Confirm containers are healthy:

   ```bash
   docker compose ps
   ```

5. Apply schema DDL and load the bundled freMTPL demo source table:

   ```bash
   docker compose exec -T airflow-apiserver python /opt/pricing/scripts/apply_schema.py
   docker compose exec -T airflow-apiserver python /opt/pricing/scripts/load_fremtpl_raw.py --replace
   ```

6. Open the core services:

   - Airflow: http://localhost:8080, login `airflow` / `airflow`
   - MLflow: http://localhost:5000
   - SQL Server: `localhost,1433`, database `PricingLab`

7. Trigger the bundled demo pipeline from Airflow, or from the CLI:

   ```bash
   docker compose exec -T airflow-apiserver airflow dags trigger pricing_mtpl_frequency
   ```

The full one-command local smoke path is also available:

```bash
scripts/run_local_pipeline.sh
```

That script builds the image, starts the services, cleans stale Airflow example
DAG metadata, waits for the bundled `pricing_mtpl_frequency` demo DAG to be
visible, and triggers the DAG.

## No-Docker Work Quickstart

Use this path when Docker is blocked but local Python processes are allowed. It
runs Airflow and MLflow on the host and writes durable local artifacts under
`state/`. SQL Server connectivity and source-data access should live in Python
modules that your DAGs import.

Prerequisites:

- Python 3.14 and `uv`.
- Any Python/ODBC/Azure packages your own connection module needs.
- Network access to the hosted SQL Server.
- A target database seeded with the required schema DDL.

1. Bootstrap local folders and dependencies:

   ```bash
   scripts/bootstrap_no_docker.sh
   ```

   If `.env` does not exist, this copies `.env.nodocker.example` to `.env`.

2. Keep `.env` focused on local runtime paths:

   ```env
   AIRFLOW_HOME=state/no_docker/airflow
   PRICING_PROJECT_ROOT=.
   PRICING_RUNTIME_MODULE=work_runtime.database
   PRICING_SCHEMA_DIR=db/migrations
   MLFLOW_TRACKING_URI=http://127.0.0.1:5000
   MLFLOW_BACKEND_STORE_URI=sqlite:///state/no_docker/mlflow/mlflow.db
   MLFLOW_ARTIFACT_ROOT=state/no_docker/mlflow/artifacts
   RATING_EXPORT_ROOT=state/no_docker/rating_exports
   VALIDATION_SPLIT_ARTIFACT_ROOT=state/no_docker/validation_splits
   ```

   Do not put SQL server names, tokens, table names, or source-data rules in
   `.env`; keep those in the Python modules your DAG imports.

3. Seed the target database with the required DDL using the plain Python script
   shown in [Seed Database Schema](#seed-database-schema).

4. Put your existing connection wrapper somewhere importable, for example
   `src/work_runtime/database.py`, and point DAGs at
   `runtime_module="work_runtime.database"`. Airflow local startup adds both the
   repo root and `src/` to `PYTHONPATH`, so that module is importable inside DAG
   tasks.

5. Start MLflow in one terminal:

   ```bash
   uv run python scripts/start_mlflow_local.py
   ```

   By default, MLflow metadata is stored in `state/no_docker/mlflow/mlflow.db`
   and artifacts are stored in `state/no_docker/mlflow/artifacts`.

   MLflow is optional for model builds. If the Python package or tracking
   server is unavailable, training/export/publish continues with tracking
   calls treated as no-ops and `mlflow_run_id` recorded as blank. Set
   `PRICING_ENABLE_MLFLOW=false` to force this no-op mode.

6. Start Airflow in another terminal:

   ```bash
   uv run python scripts/start_airflow_local.py
   ```

   This runs `airflow standalone` with `AIRFLOW_HOME=state/no_docker/airflow`,
   the repo `dags/` folder, example DAGs disabled, and repo-local rating
   exports.
   The local Airflow UI login defaults to `admin` / `admin`; set
   `AIRFLOW_LOCAL_PASSWORD` before starting if you want a different local
   password.

   You can also use the interactive shell launcher. It opens a persistent TUI
   runtime manager grouped into Services, Pipeline Tasks, and Utilities:

   ```bash
   scripts/start_no_docker_stack.sh
   ```

   `Enter` or `Space` starts/stops the selected service or runs the selected
   one-shot task. The screen shows `[running]`, `[stopped]`, `[succeeded]`, or
   `[failed]` beside each item. Press `l` to show/hide the selected item's log
   tail, `r` to restart, `x` to stop all managed services, and `q` to quit.
   Logs are written under `state/runtime/logs`.

   The shell wrapper is intentionally tiny and delegates the runtime manager to
   Python, so it works from normal zsh or Bash terminals while inheriting your
   current environment.

   The TUI groups are:

   - Services: `airflow`, `mlflow`, and local-only Docker-backed `cloudbeaver`.
   - Pipeline Tasks: schema apply, bundled demo data load/reload, direct
     pipeline run, and demo model seeding.
   - Utilities: bootstrap and ERD generation.

   The same launcher can be scripted without the menu:

   ```bash
   uv run python scripts/no_docker_services.py list
   scripts/start_no_docker_stack.sh --services airflow,mlflow
   ```

   For one-shot setup tasks, select only the pieces you want:

   ```bash
   scripts/start_no_docker_stack.sh --services apply-schema,load-fremtpl
   scripts/start_no_docker_stack.sh --services apply-schema,load-fremtpl-replace,pipeline
   scripts/start_no_docker_stack.sh --services diagrams
   ```

   `cloudbeaver` is present in the menu as a local-only Docker Compose option.
   Do not select it on work machines where Docker or Docker Hub access is
   blocked.

7. Load the bundled freMTPL demo data once if you want to run the demo model:

   ```bash
   uv run python scripts/load_fremtpl_raw.py --replace
   ```

8. Trigger the bundled `pricing_mtpl_frequency` demo DAG from the Airflow UI, or
   run it directly without Airflow:

   ```bash
   uv run python scripts/run_mtpl_frequency_custom.py \
     --runtime-module work_runtime.database
   ```

   The direct runner uses the same explicit custom path as the DAG: prepare the
   freMTPL source data, build the final model frame, record the 5-fold split and
   frame manifest, export the rating workbook/model artifact, and publish to SQL.

   To inspect the same freMTPL model-frame and CV metadata flow without Airflow,
   SQL Server, or OpenML, run the offline SQLite smoke build:

   ```bash
   uv run python scripts/run_mtpl_frequency_offline_sqlite.py --reset
   ```

   This creates a local offline DDL structure with deterministic freMTPL-like
   source rows, manifest/CV metadata, model-run metrics, and package rows for
   inspection:

   ```text
   state/offline/mtpl_frequency/pricing.sqlite
   state/offline/mtpl_frequency/pricing_stg.sqlite
   state/offline/mtpl_frequency/mlops.sqlite
   ```

   It is an offline smoke check, not the production publish path.

## Adding Models

The pipeline is split into global plumbing and model-specific code. For normal
model development, most edits should be under `pricing_models/`.

```text
pricing_models/<model_name>/
  model.toml   # model key, label, target, deployment slot, validation split
  spec.py       # loads MODEL_CONFIG from model.toml
  sql/          # optional model-local SQL scripts used by data.py
  data.py       # source reads/staging and small run metadata
  modeling.py   # final frame construction, fit/validate/export, manifest payload
  airflow_tasks.py # thin TaskFlow wrappers around data/modeling code

pricing_pipeline/
  data/          # dataset specs, raw loaders, manifests, CV split metadata
  infra/         # env config, SQL connection, schema application, MLflow setup
  models/        # shared model/data contracts and SuperGLM diagnostics capture
  orchestration/ # Airflow task wrappers and direct train/export/publish helpers
  publishing/    # rating package publish, model registry, run lineage
  tools/         # optional local ERD generation
```

Create the starting files with the scaffold helper:

```bash
uv run python scripts/scaffold_pricing_model.py \
  --model-key MY_MODEL \
  --model-label "My model" \
  --target-name derived_target
```

By default, the helper writes a custom-DAG starter:

- `pricing_models/<model_name>/model.toml`: model identity and validation split
  config.
- `pricing_models/<model_name>/spec.py`: loads `MODEL_CONFIG` only.
- `pricing_models/<model_name>/sql/source_data.sql`: a comment-only placeholder
  for model-local source SQL. Use it when useful, or ignore it and call your
  team's existing data-access helper from `data.py`.
- `pricing_models/<model_name>/data.py`: where you read or stage source data
  and return small run metadata such as output paths, `effective_from`, and
  `data_as_of_date`.
- `pricing_models/<model_name>/modeling.py`: where you edit
  `read_prepared_source(...)`, `build_final_model_frame(...)`, and
  `fit_validate_export_rating_tables(...)`. If you use
  `validation_split.method = "custom"`, edit
  `validation_split_indices_for_model(...)` to return your model's
  zero-based `(train_idx, test_idx)` folds. The generated
  `train_validate_export_model(...)` recipe wires those functions into
  versioning, frame-backed manifest creation, and the completed-build payload;
  start by leaving that recipe alone unless your model needs a different flow.
- `pricing_models/<model_name>/airflow_tasks.py`: thin TaskFlow wrappers around
  your data/modeling code.
- `dags/<dag_id>.py`: register -> prepare -> train/export/create-manifest -> publish.

`prepare_source_data(...)` should return a small dictionary of paths, table
names, IDs, or metadata. The Airflow wrapper merges in `run_key`, `output_dir`,
`effective_from`, and `data_as_of_date`; `modeling.py` receives that merged
dictionary as `prepared`. Keys such as `source_data_path` are model-owned
examples, not framework-required field names.

It creates missing scaffold files and leaves existing files unchanged; pass
`--force` only when you intentionally want to overwrite existing scaffold files.
Model configs are auto-discovered from `pricing_models/<model_name>/model.toml`;
no registry import edits are needed for normal use. The older all-in-one
`ModelSpec` / `build_pricing_model_dag(...)` scaffold is still available with
`--template factory`.

- Global code in `pricing_pipeline/` owns SQL lifecycle access for schema
  application, dataset manifests, rating export publishing, and lineage writes.
  Source data access stays model/team-owned in `data.py`.
- `DatasetSpec.manifest_sql` is only a compatibility path for SQL-backed final
  model frames and the older factory/demo flow. New custom DAGs should usually
  create the manifest from the final pandas model frame instead of re-reading
  source SQL.
- `ModelSpec` is only needed for the older all-in-one factory path. Custom DAGs
  can ignore it.
- `target_name` is the final training DataFrame column after your data/modeling
  code runs; it does not need to be a physical source column. Use your model
  data prep code for derived targets, exposure/offset columns, filters, and
  feature cleanup when the source SQL view is read-only.
- Validation split behavior belongs in the model `model.toml`. The final
  published model still trains on the full dataset; the split is retained for
  review/validation lineage. Use `method = "train_test_split"` for a holdout,
  `method = "kfold"` for cross-validation, or `method = "none"` for no
  validation split. If the final frame already contains a simple fold column or
  train/holdout flag, use `method = "column_kfold"` with
  `column = "fold_number"` or `method = "column_holdout"` with `column`,
  `train_values`, and `test_values`; those methods need no split code and mark
  the split column as validation metadata. If the split comes from a SQL lookup,
  external mapping file, grouping rule, temporal rule, or any other model-owned
  logic, use `method = "custom"` with `materialize = true` and define
  `validation_split_indices_for_model(...)` in `modeling.py`; custom folds must
  be materialized because they are not replayable from TOML alone. Source split
  columns are validation metadata and should not be exported as rating features
  unless that is an intentional model decision. Set
  `materialize = true` to write compressed `.npz` fold indexes under
  `VALIDATION_SPLIT_ARTIFACT_ROOT`; SQL stores only the artifact path, artifact
  SHA256, and dataset row-order SHA256.
- `pricing_models/registry.py` scans model folders for `model.toml`. Config-only
  paths such as deployment read TOML without importing model code; full model
  builds lazy-load only the selected model's `spec.py`.
- Add one DAG per model in `dags/`. Prefer the explicit custom TaskFlow shape:
  register the model, prepare source data, train/export/create the frame
  manifest, then bolt on the completed-build publish task. The older
  `build_pricing_model_dag(...)` helper remains available only for the
  `--template factory` compatibility scaffold.

### Custom DAG Publish Task

For production-style builds, keep your model-specific Airflow tasks in your
model package and import the common SQL lifecycle tasks from
`pricing_pipeline.orchestration`:

```python
from airflow.sdk import dag

from pricing_models.claim_freq.airflow_tasks import (
    prepare_source_data_task,
    train_validate_export_task,
)
from pricing_models.claim_freq.spec import MODEL_CONFIG
from pricing_pipeline.orchestration.model_registry_tasks import register_pricing_model_task
from pricing_pipeline.orchestration.publish_completed_build import (
    publish_completed_model_build_task,
)


@dag(dag_id="claim_freq_build", schedule=None, catchup=False)
def claim_freq_build():
    registered = register_pricing_model_task(model_config=MODEL_CONFIG)()
    prepared = prepare_source_data_task()()
    build = train_validate_export_task()(prepared)

    published = publish_completed_model_build_task(model_config=MODEL_CONFIG)(build)
    registered >> prepared >> build >> published


claim_freq_build()
```

The upstream `train_validate_export_task` should return a small dictionary with
paths and metadata, not a DataFrame. At minimum it needs the rating workbook
path, model version, and effective-from date. In a real DAG those values should
come from the run context and SQL history, not hardcoded strings:

```python
from airflow.sdk import get_current_context

from pricing_models.claim_freq.data import DATASET_NAME, PK_COLUMNS, SOURCE_SYSTEM
from pricing_pipeline.data.manifest import (
    ModelFrameManifestSpec,
    create_model_frame_manifest_with_split,
)
from pricing_pipeline.orchestration.completed_build_helpers import (
    completed_model_build_payload,
    effective_from_for_run,
)
from pricing_pipeline.publishing.model_versions import resolve_model_version_for_export
from pricing_pipeline.publishing.rating_export import build_export_id


context = get_current_context()
logical_date = context["logical_date"]
run_id = context["run_id"]

# These are usually derived inside the Airflow task, not hardcoded.
# Resolve export_id first, then resolve model_version for that export:
# - same export_id reuses the already-published model_version on rerun
# - new export_id gets the next vN from SQL package history
# - effective_from from Airflow logical date, a DAG param, or business as-of date
# - data_as_of_date from the source data snapshot/as-at date
# - export_id from model_key + Airflow run_id, so reruns are idempotent
effective_from = effective_from_for_run(logical_date)
data_as_of_date = prepared["data_as_of_date"]
export_id = build_export_id(MODEL_CONFIG.model_key, run_id)
model_version = resolve_model_version_for_export(
    engine,
    model_key=MODEL_CONFIG.model_key,
    export_id=export_id,
)
split_indices = ...  # The exact folds used by your fitting/validation code.
manifest = create_model_frame_manifest_with_split(
    engine,
    frame=final_model_frame,
    spec=ModelFrameManifestSpec(
        dataset_name=DATASET_NAME,
        source_system=SOURCE_SYSTEM,
        data_as_of_date=data_as_of_date,
        pk_columns=PK_COLUMNS,
        target_column=MODEL_CONFIG.target_name,
        weight_column="exposure",
    ),
    validation_split=MODEL_CONFIG.validation_split,
    validation_split_artifact_root=settings.validation_split_artifact_root,
    split_indices=split_indices,
)

return completed_model_build_payload(
    # Required: the SuperGLM workbook exported by this task. It must be readable
    # by the worker running the downstream publish task.
    rating_workbook_path=str(workbook_path),
    # Required: the version label for this trained model candidate.
    model_version=model_version,
    # Required: normalized to the package effective-from date.
    effective_from=effective_from,
    # Optional in Airflow: the wrapper fills this if omitted.
    created_by="airflow",
    # Optional but recommended: deterministic id for rerun/idempotency checks.
    export_id=export_id,
    # Optional: set only if this task used MLflow.
    mlflow_run_id=None,
    # Optional: path to the fitted model artifact, if you save one.
    model_artifact_path=str(model_path),
    # Required in the recommended custom flow: create these from the final
    # pandas model frame after feature engineering.
    manifest_id=manifest.manifest_id,
    split_set_id=manifest.split_set_id,
    # Optional: small numeric validation/training metrics for future review helpers.
    metrics={"deviance": float(model.result.deviance)},
).to_dict()
```

The model-owned train/export task creates the frame-backed manifest from the
actual final pandas model frame and records optional validation split metadata.
The final publish task records model-run lineage and rate package rows to SQL.
It does not deploy. It creates a deployable package candidate but does not move
any deployment slot pointer. If MLflow is disabled or unavailable, leave
`mlflow_run_id` blank or `None`.

A runnable example of this shape lives in:

- `pricing_models/demo_custom_publish/data.py`: demo-only data/source helpers
  and run-scoped handoff metadata.
- `pricing_models/demo_custom_publish/modeling.py`: demo-only SuperGLM fit,
  final model-frame manifest creation, `model.summary(...)`, workbook export,
  and completed-build payload construction.
- `pricing_models/demo_custom_publish/airflow_tasks.py`: thin Airflow wrappers
  around the demo ETL/modeling functions using shared run metadata helpers.
- `dags/demo_custom_publish.py`: Airflow TaskFlow DAG using custom model tasks
  plus global SQL lifecycle tasks from `pricing_pipeline.orchestration`. The
  DAG explicitly registers the demo model first, then prepares source data,
  trains/exports and creates manifest/split metadata from the final frame, and
  publishes by using those manifest IDs.
- `scripts/run_demo_custom_publish.py`: normal Python runner for the same path,
  useful when testing outside Airflow. Set `PRICING_DEMO_CUSTOM_OUTPUT_DIR`
  when a container or work runtime needs a writable artifact directory.

For production DAGs, avoid shared mutable handoff paths like a fixed
`training_frame.csv` or a fixed work table when separate Airflow runs can
overlap. Write any temporary files/model artifacts under a run-specific
directory. The frame-backed manifest records the final pandas model frame; it
does not require writing engineered features back to SQL Server.

For a work SQL table or view that already contains the final model frame,
`DatasetSpec.manifest_sql` remains available as a compatibility path:

```python
DatasetSpec(
    dataset_name="work_motor_frequency_2026q1",
    source_system="azure_sql",
    manifest_sql="SELECT * FROM actuarial.motor_frequency ORDER BY policy_id",
    pk_columns=("policy_id",),
    target_column="claim_count",
    weight_column="exposure",
)
```

Only local/demo datasets like freMTPL need `raw_loader=...` to fetch and seed a
source table.

For example, the current MTPL frequency model is implemented in
`pricing_models/mtpl_frequency/`, registered as `MTPL_FREQ`, and exposed as the
explicit `pricing_mtpl_frequency` custom DAG. The matching no-Airflow runner is
`scripts/run_mtpl_frequency_custom.py`.

For a work deployment, CloudBeaver is not required and should normally not be
started. Work SQL connectivity should usually live in an importable Python
runtime module such as `src/work_runtime/database.py`; the DAG can pass
`runtime_module="work_runtime.database"` explicitly, or local scripts can read
`PRICING_RUNTIME_MODULE` from `.env`.

## Optional Local Tools

These are for local inspection only. They are not required for the training or
publishing pipeline.

Start CloudBeaver, a browser SQL viewer:

```bash
docker compose --profile sql-ui up -d cloudbeaver
```

Create the optional local viewer login used by the examples:

```bash
docker compose exec -T mssql /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P 'YourStrong(!)Password123' -C -d PricingLab -b -Q "IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = N'pricing_user') CREATE LOGIN pricing_user WITH PASSWORD = N'pricinguser_12345', CHECK_POLICY = OFF; IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = N'pricing_user') CREATE USER pricing_user FOR LOGIN pricing_user; GRANT SELECT ON SCHEMA::pricing TO pricing_user; DENY SELECT ON SCHEMA::pricing_stg TO pricing_user;"
```

Open http://localhost:8978. For the local Docker SQL Server, connect with:

```text
Host: mssql
Port: 1433
Database: PricingLab
User: pricing_user
Password: pricinguser_12345
Trust server certificate: enabled
```

Start Flower, the Celery worker monitor:

```bash
docker compose --profile flower up -d flower
```

Open http://localhost:5555.

Generate and serve the database ERD:

```bash
docker compose --profile diagrams run --rm db-diagram-generator
docker compose --profile diagrams up -d db-diagrams
```

Open http://localhost:8088.

## Demo Data

The repository includes a freMTPL-based demo dataset and model specs so local
smoke tests have runnable data. Work custom DAGs should use their own source
read/stage code instead of loading demo data; `DatasetSpec` remains available
only for SQL-backed final frames and the older factory/demo path.

After the schema and bundled demo data are loaded, seed simulated model builds:

```bash
docker compose exec -T airflow-apiserver python /opt/pricing/scripts/seed_demo_model_variants.py
```

This creates:

- `MTPL_FREQ_DEMO` package versions `v1_base`, `v2_more_data`, and
  `v3_manual_vehage_uplift`.
- `MTPL_SEV_DEMO` package version `v1_base`.

To reset local pricing experiment history while keeping `FREMTPL_RAW`:

```bash
docker compose exec -T airflow-apiserver python /opt/pricing/scripts/reset_pricing_experiments.py --yes
```

For a complete local `PricingLab` rebuild with identity counters reset, drop and
recreate the database, then rerun schema apply and the bundled demo data load:

```bash
docker compose exec -T mssql /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P 'YourStrong(!)Password123' -C -d master -b -Q "IF DB_ID(N'PricingLab') IS NOT NULL BEGIN ALTER DATABASE [PricingLab] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; DROP DATABASE [PricingLab]; END; CREATE DATABASE [PricingLab];"
docker compose exec -T airflow-apiserver python /opt/pricing/scripts/apply_schema.py
docker compose exec -T airflow-apiserver python /opt/pricing/scripts/load_fremtpl_raw.py --replace
```

Only do this against the local Docker SQL Server.

## Work SQL Server Targeting

CloudBeaver is not part of the production workflow. At work, Airflow can still
run locally while the pricing schema lives on a hosted SQL Server or Azure SQL
database:

```text
local Airflow / Python / MLflow
        |
        | src/work_runtime/database.py
        v
hosted SQL Server / Azure SQL database
```

The target database is controlled by your runtime module. For SQL login,
Microsoft Entra token, service principal, or any other team-approved auth path,
put the connection code in `src/work_runtime/database.py` and keep `.env` for
Airflow/MLflow/runtime paths.

The runtime module contract is intentionally small:

```python
def get_engine(database: str | None = None):
    ...


def get_schema_names():
    ...


def get_runtime_settings():
    ...
```

The schema names returned by `get_schema_names()` should match the names used
when rendering/running the database DDL.

Then reference it from a model DAG:

```python
from airflow.sdk import dag

from pricing_models.motor_frequency.airflow_tasks import (
    prepare_source_data_task,
    train_validate_export_task,
)
from pricing_models.motor_frequency.spec import MODEL_CONFIG
from pricing_pipeline.orchestration.model_registry_tasks import register_pricing_model_task
from pricing_pipeline.orchestration.publish_completed_build import (
    publish_completed_model_build_task,
)


@dag(
    dag_id="pricing.motor_frequency.build",
    schedule=None,
    catchup=False,
    tags=["pricing", "motor", "frequency", "model-build"],
)
def pricing_motor_frequency():
    registered = register_pricing_model_task(
        model_config=MODEL_CONFIG,
        runtime_module="work_runtime.database",
    )()
    prepared = prepare_source_data_task(runtime_module="work_runtime.database")()
    completed = train_validate_export_task(runtime_module="work_runtime.database")(prepared)
    published = publish_completed_model_build_task(
        model_config=MODEL_CONFIG,
        runtime_module="work_runtime.database",
    )(completed)
    registered >> prepared >> completed >> published


pricing_motor_frequency()
```

The DBA or platform owner must create the database user and grants inside the
target database; cloud login alone is not enough.

Recommended database permissions for the pipeline user:

- read access to approved source tables/views.
- write access to `pricing` model, run, package, and rating tables.
- execute/apply schema DDL only in non-production, or through a controlled DBA
  schema-change process in production.
- no `sa`, no `db_owner`, and no permission to drop the database.

The destructive local reset commands above should not be run against work
databases. Use separate environment files or Airflow connections for `local`,
`dev`, `uat`, and `prod`, and make the active target visible in Airflow logs
before publishing a model package.

## Local Smoke Run

Run the local Airflow 3.2.1 stack and trigger the bundled end-to-end smoke DAG:

```bash
scripts/run_local_pipeline.sh
```

The script creates required project directories, runs `docker compose build`,
starts PostgreSQL, Redis, SQL Server, MLflow, and the Airflow apiserver,
scheduler, dag processor, worker, and triggerer services. It then runs the
container smoke check with
`docker compose run --rm airflow-apiserver python /opt/pricing/scripts/smoke_check.py`
cleans stale example DAG metadata from existing local Airflow databases, waits
for the bundled `pricing_mtpl_frequency` demo DAG to be loaded by the DAG
processor, and triggers it through the Airflow CLI.

## Local Services

- Airflow: http://localhost:8080, running Airflow 3.2.1.
  Example/tutorial DAGs are disabled by default.
- MLflow: http://localhost:5000. MLflow records SuperGLM experiment runs and
  serves run artifacts from the local artifact store.
- SQL Server: localhost,1433, with the default local development credentials
  from `docker-compose.yml`.
- CloudBeaver: optional SQL UI at http://localhost:8978 with
  `docker compose --profile sql-ui up -d cloudbeaver`. This is local-only and
  not required at work.
- Flower: optional Celery monitor at http://localhost:5555 with
  `docker compose --profile flower up -d flower`.
- Database diagrams: generated ERD site at http://localhost:8088 with the
  `diagrams` profile.

## Database Diagrams

Generate and serve a local ERD site for the SQL Server `pricing` and `mlops`
schemas:

```bash
docker compose --profile diagrams run --rm db-diagram-generator
docker compose --profile diagrams up -d db-diagrams
```

Open http://localhost:8088 after generation. The generated files live in
`state/db_diagrams/`:

- `index.html`: searchable self-contained diagram site.
- `schema.mmd`: Mermaid ER source for copying into docs.
- `metadata.json`: table, column, and foreign-key metadata snapshot.

For strict third-party ERD importers, use `docs/pricing_useful_tables_ddl.sql`.
It uses simple SQL Server table syntax with unnamed primary and foreign keys,
split across `raw`, `mlops`, `pricing`, and `pricing_runtime`. The three
`pricing_runtime.V_*` objects are represented as table-shaped objects there so
strict ERD tools can draw them.

For the richer SQL Server reference, use
`docs/pricing_useful_tables_full_ddl.sql`; it keeps schema creation,
schema-qualified names, named constraints, checks, unique constraints, filtered
unique indexes, and actual view definitions. The current package pointer is now
`pricing.V_CURRENT_RATE_PACKAGE`, derived from `pricing.MODEL_DEPLOYMENT`, so
there is no package pointer table. Both files deliberately exclude
`pricing_stg`, old `STG_*` tables, and `DATASET_ROW_KEY`. These files are
fresh-create references for ERDs and review, not upgrade scripts for an existing
database.

The default diagram focuses on the persisted pricing model. Import staging
tables live in the separate `pricing_stg` schema so they do not clutter the
main `pricing` schema. Generate the full technical view when needed:

```bash
DIAGRAM_INCLUDE_STAGING=1 DIAGRAM_INCLUDE_ROW_KEYS=1 docker compose --profile diagrams run --rm db-diagram-generator
```

Regenerate the site whenever schema DDL or table relationships change.

## Durable Local State

Durable local project files live under `state/`:

- `state/no_docker/airflow`: host-process Airflow home for the no-Docker
  workflow.
- `state/mssql/data`: SQL Server database files.
- `state/no_docker/mlflow/mlflow.db`: host-process MLflow SQLite metadata store
  for the no-Docker workflow.
- `state/no_docker/mlflow/artifacts`: no-Docker MLflow model and run artifacts.
- `state/rating_exports`: Docker Airflow rating export workbooks and normalized
  rating packages.
- `state/no_docker/rating_exports`: no-Docker rating export workbooks and
  normalized rating packages.
- `state/cv_splits`: exported or materialized cross-validation fold index
  artifacts.
- `state/db_diagrams`: generated schema diagrams and metadata snapshots.
- `state/cloudbeaver/workspace`: CloudBeaver workspace files when the SQL UI
  profile is used.

Do not run `docker compose down -v` unless you intentionally want to delete
Docker-managed volumes, such as the PostgreSQL metadata volume. The project
`state/` directory is ordinary filesystem state; it is not removed by
`docker compose down -v` and can be deleted manually when you want a clean local
pricing workspace.

The final rating export task requires a current local SuperGLM build that
includes PR #109 rating export support, specifically
`SuperGLM.export_rating_tables`.

## Pricing Model History

The pricing tables are historical by default. `pricing.PRICING_MODEL` stores
model families such as `MTPL_FREQ`; each training run writes `pricing.MODEL_RUN`;
each published rating-table export creates a new `pricing.PRICING_RATE_PACKAGE`.
Rate cells still hang from `rate_package_id`, which is the versioned package
grain.

Current/live selection is handled by deployment history rather than mutable
flags on the rate tables. `pricing.PRICING_MODEL_DEPLOYMENT` records deployment
slots such as `production`, `staging`, or `MTPL_FREQ_UAT`, and SQL Server enforces
one current row per `(model_id, deployment_slot)` with a filtered unique index on
rows where `effective_to_ts IS NULL`.
Rows with a non-null `effective_to_ts` are closed historical deployments; use
`pricing.V_CURRENT_RATE_PACKAGE` when you only want the live deployment per slot.

Convenience views expose the current state:

- `pricing.V_ACTIVE_MODEL`
- `pricing.V_CURRENT_RATE_PACKAGE`
- `pricing.V_CURRENT_RATE_CELL`
- `pricing.V_CURRENT_1D_RATE_BAND`

## Rate Package Lifecycle

Production model builds use stable model metadata from each model's
`model.toml`. That config records housekeeping identity such as `model_key`,
`target_name`, `model_type`, and the default deployment slot; SQL Server owns
the generated `model_id`.

`ModelPublisher` is the Python API for publishing training exports, deploying
packages, loading package snapshots, creating manual revisions, and comparing
prediction vectors. Airflow build DAGs use it to create immutable `PUBLISHED`
candidate rate packages, but they do not move live deployment pointers by
default.

Live deployments happen through the generic manual DAG
`pricing_deploy_rate_package`. The deploy run requires `model_key`, exactly one
reviewed package selector (`rate_package_id` or `package_version`),
`deployed_by`, and `deployment_reason` as the audit reason.
`deployment_slot` is optional and defaults to the model config deployment slot.
Manual rate changes follow the same lifecycle: load the package snapshot, edit
the constrained rate-cell DataFrames, create a child package with
`parent_rate_package_id`, then deploy that child package through
`pricing_deploy_rate_package`.

Published package rows are never edited directly. Once a package is no longer
`DRAFT`, SQL Server immutability triggers block direct updates or deletes to
the package and its rating rows, so changes must be published as new packages
or manual revisions.

## CV Split Storage

Cross-validation split lineage is metadata-first. Dataset manifests can write a
`pricing.CV_SPLIT_SET` row with the replayable splitter class, splitter params,
row-order SHA-256 fingerprint, row count, fold count, and dependency/runtime
metadata for the split environment. Per-fold sizes are stored in
`pricing.CV_FOLD`.

Model runs are linked to their training data and split set through small
`mlops` lineage tables:

- `mlops.MODEL_RUN_DATASET`: which dataset manifest a run used.
- `mlops.MODEL_RUN_SPLIT_SET`: which CV split set a run used for that dataset.
- `mlops.MODEL_RUN_METRIC`: optional run-level metrics.

`mlops.CV_SPLIT_ROW` exists for the cases where SQL itself must answer "which
rows were in the test fold?" It is intentionally optional. For large datasets,
the default remains metadata plus compressed artifacts rather than inserting a
row for every policy on every split set.

When exact fold indices need to be locked down, materialize the split set to a
compressed NumPy artifact under `state/cv_splits`. The database row is updated
to `split_mode = 'MATERIALIZED'` and stores both `artifact_uri` and
`artifact_sha256`, so loaders verify the file before returning indices:

```bash
docker compose --profile debug run --rm airflow-cli bash -c "python /opt/pricing/scripts/export_cv_indices.py --split-set-id <split_set_id> --materialize --out /opt/pricing/state/cv_splits/<split_set_id>.npz"
```

The current split metadata, artifact path, hash, runtime metadata, and
train/test fold definitions are exposed through
`pricing.V_CURRENT_DATASET_CV_FOLD`.

## SQL Prediction Validation

`pricing.PREDICT_CURRENT_RATE` is a thin SQL Server scorer over the current
deployed rating package. It reads one JSON feature row, resolves
`pricing.V_CURRENT_RATE_PACKAGE`, matches the deployed compiled rating cells,
and returns relativity plus predicted count. It is intended for auditable SQL
serving, not for training.

Example SQL call:

```sql
EXEC pricing.PREDICT_CURRENT_RATE
    @model_key = N'MTPL_FREQ',
    @deployment_slot = N'MTPL_FREQ_UAT',
    @exposure = 0.75,
    @features_json = N'{
        "VehAge": 4,
        "DrivAge": 44,
        "BonusMalus": 82,
        "LogDensity": 6.2,
        "Area": "C",
        "VehPower": 7,
        "VehBrand": "B1",
        "VehGas": "Regular",
        "Region": "R24"
    }',
    @include_breakdown = 1;
```

The proc must be validated against the actual fitted SuperGLM artifact before
being trusted for a deployed model. The validator rebuilds the model input
frame, calls `fitted_model.predict(X, offset=offset)`, calls the SQL proc for
the same rows, and fails if the results exceed tolerance:

```bash
uv run python scripts/validate_sql_prediction_against_superglm.py \
  --model-key MTPL_FREQ \
  --deployment-slot MTPL_FREQ_UAT \
  --limit 1000 \
  --rtol 1e-4 \
  --atol 1e-8
```

`sample_weight` is used when exporting rating tables from SuperGLM, but
prediction validation uses the model's prediction API with the exposure offset:
`predict(X, offset=np.log(exposure))`.

## Demo Model Variants

To seed extra model/package history for CloudBeaver inspection:

```bash
docker compose exec -T airflow-apiserver python /opt/pricing/scripts/seed_demo_model_variants.py
```

To clear local pricing experiment history before rerunning the Airflow pipeline
and demo seeds:

```bash
docker compose exec -T airflow-apiserver python /opt/pricing/scripts/reset_pricing_experiments.py --yes
```

The reset leaves `pricing.FREMTPL_RAW` in place and clears model runs, rating
packages, dataset manifests, CV metadata, staging rows, and normalized pricing
tables.

The seeder publishes:

- `MTPL_FREQ_DEMO` with three package versions:
  `v1_base`, `v2_more_data`, and `v3_manual_vehage_uplift`.
  The demo `VehAge` term is marked as a discretized spline:
  `term_type = DISCRETIZED_SPLINE_1D` and `level_set_type = SPLINE_GRID_1D`.
- `MTPL_SEV_DEMO` with one separate severity package:
  `v1_base`.

Useful inspection queries:

```sql
SELECT model_id, model_key, target_name, model_type, model_status
FROM pricing.PRICING_MODEL
ORDER BY model_id;

SELECT model_key, deployment_slot, rate_package_id, model_version, package_version
FROM pricing.V_CURRENT_RATE_PACKAGE
ORDER BY model_key, deployment_slot;

SELECT model_name, model_version, package_version, package_status
FROM pricing.PRICING_RATE_PACKAGE
ORDER BY rate_package_id;

SELECT
    rp.model_name,
    rp.model_version,
    t.term_name,
    c.cell_key_text,
    c.multiplier
FROM pricing.PRICING_RATE_PACKAGE rp
JOIN pricing.PRICING_TERM t
  ON t.rate_package_id = rp.rate_package_id
JOIN pricing.PRICING_RATE_CELL c
  ON c.term_id = t.term_id
WHERE rp.model_name = 'MTPL_SEV_DEMO'
  AND t.term_name = 'VehBrand'
  AND c.cell_key_text = 'VehBrand=B12'
ORDER BY rp.package_version;

SELECT
    rp.model_name,
    rp.model_version,
    t.term_name,
    c.cell_key_text,
    c.multiplier
FROM pricing.PRICING_RATE_PACKAGE rp
JOIN pricing.PRICING_TERM t
  ON t.rate_package_id = rp.rate_package_id
JOIN pricing.PRICING_RATE_CELL c
  ON c.term_id = t.term_id
WHERE rp.model_name = 'MTPL_FREQ_DEMO'
  AND t.term_name = 'VehAge'
  AND c.cell_key_text = 'VehAge=[10, 20)'
ORDER BY rp.package_version;
```
