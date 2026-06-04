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
   uv run python scripts/run_pipeline_no_airflow.py \
     --runtime-module work_runtime.database \
     --model-key MTPL_FREQ
   ```

   The direct runner uses the same schema DDL, dataset manifest/CV metadata,
   MLflow logging, rating export, and SQL publish code as the DAG.

## Adding Models

The pipeline is split into global plumbing and model-specific code. For normal
model development, most edits should be under `pricing_models/`.

```text
pricing_models/<model_name>/
  model.toml   # model key, label, target, deployment slot, validation split
  spec.py       # model key, model metadata, dataset choice, feature list
  training.py   # model-specific SQL, feature prep, and SuperGLM construction

pricing_pipeline/
  data/          # dataset specs, raw loaders, manifests, CV split metadata
  infra/         # env config, SQL connection, schema application, MLflow setup
  models/        # shared model/data contracts and SuperGLM diagnostics capture
  orchestration/ # Airflow DAG factory and direct train/export/publish flow
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

The helper writes `pricing_models/<model_name>/model.toml`, `training.py`,
`spec.py`, and a thin DAG under `dags/`. It refuses to overwrite existing files
unless `--force` is passed. Model configs are auto-discovered from
`pricing_models/<model_name>/model.toml`; no registry import edits are needed for
normal use.

- Global code in `pricing_pipeline/` owns database access, schema application,
  dataset manifests, MLflow setup, rating export publishing, and lineage writes.
- Reusable datasets are described with `DatasetSpec` objects, currently in
  `pricing_pipeline/data/datasets.py`, or directly in a model `spec.py`.
- Model code lives under `pricing_models/<model_name>/`. A model package should
  provide `training.py` for feature preparation/model construction and `spec.py`
  with a `ModelSpec` that points at the dataset it trains on.
- `target_name` is the final training DataFrame column after
  `build_training_frame()` runs; it does not need to be a physical source
  column. Use that transform function for derived targets, exposure/offset
  columns, filters, and feature cleanup when the source SQL view is read-only.
- Validation split behavior belongs in the model `model.toml`. The final
  published model still trains on the full dataset; the split is retained for
  review/validation lineage. Use `method = "train_test_split"` for a holdout,
  `method = "kfold"` for cross-validation, or `method = "none"` for no
  validation split. Set `materialize = true` to write compressed `.npz` fold
  indexes under `VALIDATION_SPLIT_ARTIFACT_ROOT`; SQL stores only the artifact
  path, artifact SHA256, and dataset row-order SHA256.
- `pricing_models/registry.py` scans model folders for `model.toml`. Config-only
  paths such as deployment read TOML without importing model code; full model
  builds lazy-load only the selected model's `spec.py`.
- Add one DAG per model in `dags/`. For quick demos you can use
  `pricing_pipeline.orchestration.dag_factory.build_pricing_model_dag(...)`;
  for serious model builds, a custom DAG can own ingestion, transforms,
  training, validation, and then bolt on the completed-build publish task.

### Custom DAG Publish Task

For production-style builds, keep your model-specific Airflow tasks in your
model package and import the common SQL lifecycle tasks from
`pricing_pipeline.orchestration`:

```python
from airflow.sdk import dag

from pricing_models.claim_freq.airflow_tasks import (
    prepare_training_data,
    train_and_export_rates,
)
from pricing_models.claim_freq.data import dataset_spec_from_prepared_training
from pricing_models.claim_freq.spec import MODEL_CONFIG
from pricing_pipeline.orchestration.manifest_tasks import (
    create_prepared_dataset_manifest_task,
)
from pricing_pipeline.orchestration.model_registry_tasks import register_pricing_model_task
from pricing_pipeline.orchestration.publish_completed_build import (
    publish_completed_model_build_task,
)


@dag(dag_id="claim_freq_build", schedule=None, catchup=False)
def claim_freq_build():
    registered = register_pricing_model_task(model_config=MODEL_CONFIG)()
    prepared = prepare_training_data()
    manifested = create_prepared_dataset_manifest_task(
        model_config=MODEL_CONFIG,
        dataset_builder=dataset_spec_from_prepared_training,
    )(prepared)
    build = train_and_export_rates(manifested)

    published = publish_completed_model_build_task(model_config=MODEL_CONFIG)(build)
    registered >> prepared >> manifested >> build >> published


claim_freq_build()
```

The upstream `train_and_export_rates` task should return a small dictionary with
paths and metadata, not a DataFrame. At minimum it needs the rating workbook
path, model version, and effective-from date. In a real DAG those values should
come from the run context and SQL history, not hardcoded strings:

```python
from airflow.sdk import get_current_context

from pricing_pipeline.orchestration.publish_completed_build import CompletedModelBuild
from pricing_pipeline.publishing.rating_export import build_export_id


context = get_current_context()
logical_date = context["logical_date"]
run_id = context["run_id"]

# These are usually derived inside the Airflow task, not hardcoded:
# - model_version from SQL package history, e.g. next vN for this model_key
# - effective_from from Airflow logical date, a DAG param, or business as-of date
# - export_id from model_key + Airflow run_id, so reruns are idempotent
model_version = next_trained_model_version(engine, model_key=MODEL_CONFIG.model_key)
effective_from = effective_from_for_run(logical_date)
export_id = build_export_id(MODEL_CONFIG.model_key, run_id)

return CompletedModelBuild(
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
    # Optional: include if the prior prepare/materialize task created these.
    manifest_id=prepared.get("manifest_id"),
    split_set_id=prepared.get("split_set_id"),
    # Optional: small numeric validation/training metrics for future review helpers.
    metrics={"deviance": float(model.result.deviance)},
).to_dict()
```

The manifest task records the model-ready dataset and optional validation split
metadata. The final publish task records model-run lineage and rate package rows
to SQL. It does not deploy. It creates a deployable package candidate but does
not move any deployment slot pointer. If MLflow is disabled or unavailable,
leave `mlflow_run_id` blank or `None`.

A runnable example of this shape lives in:

- `pricing_models/demo_custom_publish/data.py`: demo-only data generation,
  run-scoped SQL staging materialization, and `DatasetSpec` construction.
- `pricing_models/demo_custom_publish/modeling.py`: demo-only SuperGLM fit,
  `model.summary(...)`, workbook export, and dynamic
  model-version/effective-date helpers.
- `pricing_models/demo_custom_publish/airflow_tasks.py`: thin Airflow wrappers
  around the demo ETL/modeling functions.
- `dags/demo_custom_publish.py`: Airflow TaskFlow DAG using custom model tasks
  plus global SQL lifecycle tasks from `pricing_pipeline.orchestration`. The
  DAG explicitly registers the demo model first, then prepares a run-scoped
  training source, creates manifest/split metadata for that source,
  trains/exports, and publishes by reusing the prepared manifest IDs.
- `scripts/run_demo_custom_publish.py`: normal Python runner for the same path,
  useful when testing outside Airflow. Set `PRICING_DEMO_CUSTOM_OUTPUT_DIR`
  when a container or work runtime needs a writable artifact directory.

For production DAGs, avoid shared mutable handoff paths like a fixed
`training_frame.csv` or a fixed work table when separate Airflow runs can
overlap. Materialize the model-ready frame to a run-specific table or filtered
view, create the manifest from that same stable source, and write workbook/model
artifacts under a run-specific directory.

For a work SQL table or view that already exists, the dataset definition can be
just metadata and SQL. It does not need a custom Python loader:

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
`pricing_mtpl_frequency` DAG. `pricing_superglm_pipeline` remains as a
compatibility alias for older local commands.

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
smoke tests have runnable data. Work deployments usually point `DatasetSpec`
objects at approved source tables or views instead of loading demo data.

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
pricing_motor_frequency = build_pricing_model_dag(
    dag_id="pricing.motor_frequency.build",
    spec=MODEL_SPEC,
    model_config=MODEL_CONFIG,
    runtime_module="work_runtime.database",
    tags=["pricing", "motor", "frequency", "model-build"],
)
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
