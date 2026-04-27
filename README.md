# Airflow SuperGLM Builder

Production-minded local Airflow 3.2.1 pipeline for freMTPL pricing experiments.

The pipeline stores raw freMTPL rows in SQL Server, trains a SuperGLM Poisson
frequency model, logs model runs to MLflow, exports rating tables, and publishes
normalized rating packages back to SQL Server.

Durable local state lives under `state/`. Do not run `docker compose down -v`
unless you intend to remove Docker-managed service state.
