# Airflow SuperGLM Builder

Production-minded local Airflow 3.2.1 pipeline for freMTPL pricing experiments.

The pipeline stores raw freMTPL rows in SQL Server, trains a SuperGLM Poisson
frequency model, logs model runs to MLflow, exports rating tables, and publishes
normalized rating packages back to SQL Server.

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

5. Apply SQL migrations and load the raw freMTPL source table:

   ```bash
   docker compose exec -T airflow-apiserver python /opt/pricing/scripts/apply_sql_migrations.py
   docker compose exec -T airflow-apiserver python /opt/pricing/scripts/load_fremtpl_raw.py --replace
   ```

6. Open the core services:

   - Airflow: http://localhost:8080, login `airflow` / `airflow`
   - MLflow: http://localhost:5000
   - SQL Server: `localhost,1433`, database `PricingLab`

7. Trigger the pipeline from Airflow, or from the CLI:

   ```bash
   docker compose exec -T airflow-apiserver airflow dags trigger pricing_superglm_pipeline
   ```

The full one-command local smoke path is also available:

```bash
scripts/run_local_pipeline.sh
```

That script builds the image, starts the services, cleans stale Airflow example
DAG metadata, waits for `pricing_superglm_pipeline` to be visible, and triggers
the DAG.

## No-Docker Work Quickstart

Use this path when Docker is blocked but local Python processes are allowed. It
runs Airflow and MLflow on the host, writes durable artifacts under `state/`,
and targets an external SQL Server or Azure SQL database through ODBC.

Prerequisites:

- Python 3.14 and `uv`.
- Microsoft ODBC Driver 18 for SQL Server.
- Network access to the hosted SQL Server.
- A target database that already exists, unless your login is allowed to create
  databases.

1. Bootstrap local folders and dependencies:

   ```bash
   scripts/bootstrap_no_docker.sh
   ```

   If `.env` does not exist, this copies `.env.nodocker.example` to `.env`.

2. Edit `.env` for the work SQL Server:

   ```env
   MSSQL_SERVER=<server-name>.database.windows.net,1433
   MSSQL_DATABASE=PricingLab_UAT
   MSSQL_USER=pricing_pipeline_writer
   MSSQL_PASSWORD=<from-secret-store>
   MSSQL_ENCRYPT=yes
   MSSQL_TRUST_SERVER_CERT=no
   PRICING_SKIP_DATABASE_CREATE=true
   ```

   Keep `PRICING_SKIP_DATABASE_CREATE=true` when the DBA has already created the
   database and your pipeline login should only manage objects inside it.

3. Start MLflow in one terminal:

   ```bash
   uv run python scripts/start_mlflow_local.py
   ```

   By default, MLflow metadata is stored in `state/mlflow/mlflow.db` and
   artifacts are stored in `state/mlflow/artifacts`.

4. Start Airflow in another terminal:

   ```bash
   uv run python scripts/start_airflow_local.py
   ```

   This runs `airflow standalone` with `AIRFLOW_HOME=state/airflow`, the repo
   `dags/` folder, example DAGs disabled, and repo-local rating exports.
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
   - Pipeline Tasks: migrations, freMTPL raw load/reload, direct pipeline run,
     and demo model seeding.
   - Utilities: bootstrap and ERD generation.

   The same launcher can be scripted without the menu:

   ```bash
   uv run python scripts/no_docker_services.py list
   scripts/start_no_docker_stack.sh --services airflow,mlflow
   ```

   For one-shot setup tasks, select only the pieces you want:

   ```bash
   scripts/start_no_docker_stack.sh --services migrate,load-raw
   scripts/start_no_docker_stack.sh --services migrate,load-raw-replace,pipeline
   scripts/start_no_docker_stack.sh --services diagrams
   ```

   `cloudbeaver` is present in the menu as a local-only Docker Compose option.
   Do not select it on work machines where Docker or Docker Hub access is
   blocked.

5. Apply migrations and load raw freMTPL data once:

   ```bash
   uv run python scripts/apply_sql_migrations.py
   uv run python scripts/load_fremtpl_raw.py --replace
   ```

6. Trigger `pricing_superglm_pipeline` from the Airflow UI, or run it directly
   without Airflow:

   ```bash
   uv run python scripts/run_pipeline_no_airflow.py
   ```

   The direct runner uses the same SQL migrations, freMTPL loader, manifest/CV
   metadata, MLflow logging, rating export, and SQL publish code as the DAG.

For a work deployment, CloudBeaver is not required and should normally not be
started. Changing from local testing to a work SQL Server is just an `.env`
change as long as the authentication mode is SQL username/password. If the
work server requires Microsoft Entra token authentication, add a pyodbc token
connection path in `pricing_pipeline/db.py` and keep the rest of the pipeline
unchanged.

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

After the schema and raw data are loaded, seed simulated model builds:

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
recreate the database, then rerun migrations and raw load:

```bash
docker compose exec -T mssql /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P 'YourStrong(!)Password123' -C -d master -b -Q "IF DB_ID(N'PricingLab') IS NOT NULL BEGIN ALTER DATABASE [PricingLab] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; DROP DATABASE [PricingLab]; END; CREATE DATABASE [PricingLab];"
docker compose exec -T airflow-apiserver python /opt/pricing/scripts/apply_sql_migrations.py
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
        | ODBC Driver 18 for SQL Server
        v
hosted SQL Server / Azure SQL database
```

The target database is controlled by environment variables. For SQL
username/password targets, the code does not change between local and work; only
`.env` or Airflow secrets/connections change.

Typical hosted SQL Server settings:

```env
MSSQL_SERVER=<server-name>.database.windows.net,1433
MSSQL_DATABASE=PricingLab_UAT
MSSQL_DRIVER=ODBC Driver 18 for SQL Server
MSSQL_ENCRYPT=yes
MSSQL_TRUST_SERVER_CERT=no
```

If the work database allows SQL authentication, use a restricted pipeline login:

```env
MSSQL_USER=pricing_pipeline_writer
MSSQL_PASSWORD=<from-secret-store>
```

If the work database requires Microsoft Entra authentication, add an explicit
auth mode to the SQL connection helper before running the work deployment. The
unattended Airflow-friendly option is usually an Entra service principal or an
access token helper. The DBA or platform owner must also create the database user
and grants inside the target database; Entra login alone is not enough.

Recommended database permissions for the pipeline user:

- read access to approved source tables/views.
- write access to `pricing` model, run, package, and rating tables.
- execute/apply migrations only in non-production, or through a controlled DBA
  migration process in production.
- no `sa`, no `db_owner`, and no permission to drop the database.

The destructive local reset commands above should not be run against work
databases. Use separate environment files or Airflow connections for `local`,
`dev`, `uat`, and `prod`, and make the active target visible in Airflow logs
before publishing a model package.

## Local Smoke Run

Run the local Airflow 3.2.1 stack and trigger the end-to-end smoke DAG:

```bash
scripts/run_local_pipeline.sh
```

The script creates required project directories, runs `docker compose build`,
starts PostgreSQL, Redis, SQL Server, MLflow, and the Airflow apiserver,
scheduler, dag processor, worker, and triggerer services. It then runs the
container smoke check with
`docker compose run --rm airflow-apiserver python /opt/pricing/scripts/smoke_check.py`
cleans stale example DAG metadata from existing local Airflow databases, waits
for `pricing_superglm_pipeline` to be loaded by the DAG processor, and triggers
it through the Airflow CLI.

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

Generate and serve a local ERD site for the SQL Server `pricing` schema:

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
fresh-create references for ERDs and review, not migrations for an existing
database.

The default diagram focuses on the persisted pricing model. Import staging
tables live in the separate `pricing_stg` schema so they do not clutter the
main `pricing` schema. Generate the full technical view when needed:

```bash
DIAGRAM_INCLUDE_STAGING=1 DIAGRAM_INCLUDE_ROW_KEYS=1 docker compose --profile diagrams run --rm db-diagram-generator
```

Regenerate the site whenever migrations or table relationships change.

## Durable Local State

Durable local project files live under `state/`:

- `state/airflow`: host-process Airflow home for the no-Docker workflow.
- `state/mssql/data`: SQL Server database files.
- `state/mlflow/mlflow.db`: host-process MLflow SQLite metadata store for the
  no-Docker workflow.
- `state/mlflow/artifacts`: MLflow model and run artifacts.
- `state/rating_exports`: rating export workbooks and normalized rating
  packages produced by the pipeline.
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

## CV Split Storage

Cross-validation split lineage is metadata-first. New freMTPL manifests write a
`pricing.CV_SPLIT_SET` row with the replayable splitter class, splitter params,
row-order SHA-256 fingerprint, row count, fold count, and dependency/runtime
metadata for the split environment. Per-fold sizes are stored in
`pricing.CV_FOLD`.

The older row-key materialization tables were removed. Exact fold indices are
kept in compressed artifacts when needed rather than in one database row per
policy. This avoids inserting hundreds of thousands of rows per manifest while
still allowing exact replay from the stored artifact.

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
