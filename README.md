# Airflow SuperGLM Builder

Production-minded local Airflow 3.2.1 pipeline for freMTPL pricing experiments.

The pipeline stores raw freMTPL rows in SQL Server, trains a SuperGLM Poisson
frequency model, logs model runs to MLflow, exports rating tables, and publishes
normalized rating packages back to SQL Server.

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
and triggers `pricing_superglm_pipeline` through the Airflow CLI.

## Local Services

- Airflow: http://localhost:8080, running Airflow 3.2.1.
- MLflow: http://localhost:5000. MLflow records SuperGLM experiment runs and
  serves run artifacts from the local artifact store.
- SQL Server: localhost,1433, with the default local development credentials
  from `docker-compose.yml`.
- CloudBeaver: optional SQL UI at http://localhost:8978 with
  `docker compose --profile sql-ui up -d cloudbeaver`.
- Flower: optional Celery monitor at http://localhost:5555 with
  `docker compose --profile flower up -d flower`.

## Durable Local State

Durable local project files live under `state/`:

- `state/mssql/data`: SQL Server database files.
- `state/mlflow/artifacts`: MLflow model and run artifacts.
- `state/rating_exports`: rating export workbooks and normalized rating
  packages produced by the pipeline.
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
