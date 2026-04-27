# Airflow SuperGLM Pricing Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Airflow 3.2.1 pipeline that stores freMTPL raw data in SQL Server, trains a SuperGLM Poisson frequency model, tracks it in MLflow, exports rating tables, and publishes normalized rating packages back to SQL Server.

**Architecture:** Use a custom Airflow 3.2.1 Python 3.14 image with SuperGLM, MLflow, and SQL Server ODBC dependencies. Keep Airflow orchestration thin by putting ETL, training, export, and SQL publishing logic in a small `pricing_pipeline` package. Persist business data in SQL Server and artifacts under project-local `state/` bind mounts.

**Tech Stack:** Apache Airflow 3.2.1, Python 3.14 preferred, SQL Server 2022, MLflow, SuperGLM from GitHub, pandas, scikit-learn, SQLAlchemy, pyodbc, openpyxl, pytest.

---

## File Structure

- Create `.gitignore`: ignore zip archives, local env files, caches, logs, and `state/`.
- Create `pyproject.toml`: package/test configuration for local development.
- Create `.env.example`: local defaults for Airflow, SQL Server, MLflow, and artifact paths.
- Create `requirements.txt`: runtime dependencies installed into the custom Airflow image.
- Create `airflow/Dockerfile`: Airflow 3.2.1 Python 3.14 image with ODBC and Python dependencies.
- Replace `docker-compose.yml`: official Airflow 3.2.1 CeleryExecutor shape plus SQL Server, MLflow, and CloudBeaver.
- Create `db/migrations/*.sql`: keep starter migrations and add raw freMTPL and model-run lineage migration.
- Create `pricing_pipeline/`: importable application package used by tests and Airflow tasks.
- Create `dags/pricing_superglm_pipeline.py`: Airflow 3 Task SDK DAG that calls package functions.
- Create `tests/`: focused unit tests for config, SQL migration helpers, freMTPL load, manifest creation, rating parsing, and pipeline smoke imports.
- Create `scripts/smoke_check.py`: local/container smoke check for imports and key entry points.
- Create `README.md`: runbook for building, starting, running, and preserving local state.

---

### Task 1: Repo Hygiene And Starter Scaffold

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `pyproject.toml`
- Modify: tracked repo contents by unpacking starter files from `pricing_python_starter.zip`

- [ ] **Step 1: Write the failing hygiene test**

Create `tests/test_repo_hygiene.py`:

```python
from pathlib import Path


def test_zip_archives_are_ignored():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "*.zip" in gitignore
    assert "state/" in gitignore


def test_project_package_can_be_imported():
    import pricing_pipeline

    assert pricing_pipeline.__version__
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
rtk pytest tests/test_repo_hygiene.py -v
```

Expected: FAIL because `pricing_pipeline` does not exist before the scaffold is added.

- [ ] **Step 3: Add repo hygiene files**

Create `.gitignore`:

```gitignore
*.zip
.env
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.mypy_cache/
logs/
state/
airflow/logs/
airflow/config/
airflow/plugins/
```

Create `pricing_pipeline/__init__.py`:

```python
"""Pricing pipeline package used by Airflow tasks and local tests."""

__version__ = "0.1.0"
```

Create `pyproject.toml`:

```toml
[project]
name = "airflow-superglm-builder"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
  "apache-airflow-providers-microsoft-mssql",
  "mlflow",
  "numpy",
  "openpyxl",
  "pandas",
  "pyodbc",
  "python-dotenv",
  "scikit-learn",
  "sqlalchemy",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-q"

[tool.ruff]
target-version = "py314"
line-length = 100
```

Create `README.md`:

```markdown
# Airflow SuperGLM Builder

Production-minded local Airflow 3.2.1 pipeline for freMTPL pricing experiments.

The pipeline stores raw freMTPL rows in SQL Server, trains a SuperGLM Poisson
frequency model, logs model runs to MLflow, exports rating tables, and publishes
normalized rating packages back to SQL Server.

Durable local state lives under `state/`. Do not run `docker compose down -v`
unless you intend to remove Docker-managed service state.
```

- [ ] **Step 4: Unpack starter files without committing the zip**

Run:

```bash
rtk unzip -o pricing_python_starter.zip -d /tmp/airflow_superglm_builder_starter
rtk cp -R /tmp/airflow_superglm_builder_starter/pricing_python_starter/db ./db
rtk cp -R /tmp/airflow_superglm_builder_starter/pricing_python_starter/scripts ./scripts
```

Expected: `db/migrations` and `scripts` exist in the repo. `pricing_python_starter.zip` remains untracked and ignored.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk pytest tests/test_repo_hygiene.py -v
rtk git status --short
```

Expected: PASS. `pricing_python_starter.zip` is not shown in `git status --short`.

Commit:

```bash
rtk git add .gitignore README.md pyproject.toml pricing_pipeline/__init__.py tests/test_repo_hygiene.py db scripts
rtk git commit -m "chore: scaffold pricing pipeline repo"
```

---

### Task 2: Configuration And SQL Server Connection Layer

**Files:**
- Create: `pricing_pipeline/config.py`
- Create: `pricing_pipeline/db.py`
- Create: `tests/test_config_db.py`
- Create: `.env.example`

- [ ] **Step 1: Write failing config tests**

Create `tests/test_config_db.py`:

```python
from pricing_pipeline.config import Settings
from pricing_pipeline.db import build_odbc_connect_string


def test_settings_defaults_are_local_dev_safe():
    settings = Settings.from_env({})
    assert settings.mssql_server == "mssql,1433"
    assert settings.pricing_database == "PricingLab"
    assert settings.mlflow_tracking_uri == "http://mlflow:5000"


def test_odbc_connection_string_targets_database():
    settings = Settings.from_env(
        {
            "MSSQL_SERVER": "localhost,1433",
            "MSSQL_DATABASE": "PricingLab",
            "MSSQL_USER": "sa",
            "MSSQL_PASSWORD": "secret",
            "MSSQL_DRIVER": "ODBC Driver 18 for SQL Server",
            "MSSQL_ENCRYPT": "no",
            "MSSQL_TRUST_SERVER_CERT": "yes",
        }
    )
    odbc = build_odbc_connect_string(settings, database=settings.pricing_database)
    assert "SERVER=localhost,1433" in odbc
    assert "DATABASE=PricingLab" in odbc
    assert "PWD=secret" in odbc
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk pytest tests/test_config_db.py -v
```

Expected: FAIL because `pricing_pipeline.config` and `pricing_pipeline.db` do not exist.

- [ ] **Step 3: Implement settings and connection helpers**

Create `pricing_pipeline/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Settings:
    mssql_server: str = "mssql,1433"
    pricing_database: str = "PricingLab"
    mlflow_database: str = "MLflowTracking"
    mssql_user: str = "sa"
    mssql_password: str = "YourStrong(!)Password123"
    mssql_driver: str = "ODBC Driver 18 for SQL Server"
    mssql_encrypt: str = "no"
    mssql_trust_server_cert: str = "yes"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    rating_export_root: Path = Path("/opt/pricing/state/rating_exports")

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Settings":
        return cls(
            mssql_server=env.get("MSSQL_SERVER", cls.mssql_server),
            pricing_database=env.get("MSSQL_DATABASE", cls.pricing_database),
            mlflow_database=env.get("MLFLOW_DATABASE", cls.mlflow_database),
            mssql_user=env.get("MSSQL_USER", cls.mssql_user),
            mssql_password=env.get("MSSQL_PASSWORD", cls.mssql_password),
            mssql_driver=env.get("MSSQL_DRIVER", cls.mssql_driver),
            mssql_encrypt=env.get("MSSQL_ENCRYPT", cls.mssql_encrypt),
            mssql_trust_server_cert=env.get(
                "MSSQL_TRUST_SERVER_CERT", cls.mssql_trust_server_cert
            ),
            mlflow_tracking_uri=env.get("MLFLOW_TRACKING_URI", cls.mlflow_tracking_uri),
            rating_export_root=Path(
                env.get("RATING_EXPORT_ROOT", str(cls.rating_export_root))
            ),
        )
```

Create `pricing_pipeline/db.py`:

```python
from __future__ import annotations

from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from pricing_pipeline.config import Settings


def build_odbc_connect_string(settings: Settings, *, database: str) -> str:
    return (
        f"DRIVER={{{settings.mssql_driver}}};"
        f"SERVER={settings.mssql_server};"
        f"DATABASE={database};"
        f"UID={settings.mssql_user};"
        f"PWD={settings.mssql_password};"
        f"Encrypt={settings.mssql_encrypt};"
        f"TrustServerCertificate={settings.mssql_trust_server_cert};"
    )


def build_sqlalchemy_url(settings: Settings, *, database: str) -> str:
    odbc = build_odbc_connect_string(settings, database=database)
    return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}"


def get_engine(settings: Settings, *, database: str | None = None) -> Engine:
    return create_engine(
        build_sqlalchemy_url(settings, database=database or settings.pricing_database),
        fast_executemany=True,
        future=True,
    )


def ensure_database(settings: Settings, database: str) -> None:
    master = get_engine(settings, database="master")
    escaped = database.replace("]", "]]")
    with master.begin() as con:
        exists = con.execute(
            text("SELECT 1 FROM sys.databases WHERE name = :database"),
            {"database": database},
        ).scalar()
        if not exists:
            con.execute(text(f"CREATE DATABASE [{escaped}]"))
```

Create `.env.example`:

```dotenv
AIRFLOW_UID=50000
AIRFLOW_IMAGE_NAME=airflow-superglm:3.2.1-python3.14
FERNET_KEY=airflow_fernet_key_change_me
AIRFLOW__API_AUTH__JWT_SECRET=airflow_jwt_secret_change_me

MSSQL_SERVER=mssql,1433
MSSQL_DATABASE=PricingLab
MLFLOW_DATABASE=MLflowTracking
MSSQL_USER=sa
MSSQL_PASSWORD=YourStrong(!)Password123
MSSQL_DRIVER=ODBC Driver 18 for SQL Server
MSSQL_ENCRYPT=no
MSSQL_TRUST_SERVER_CERT=yes

MLFLOW_TRACKING_URI=http://mlflow:5000
RATING_EXPORT_ROOT=/opt/pricing/state/rating_exports
```

- [ ] **Step 4: Run tests**

Run:

```bash
rtk pytest tests/test_config_db.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add .env.example pricing_pipeline/config.py pricing_pipeline/db.py tests/test_config_db.py
rtk git commit -m "feat: add configuration and SQL Server helpers"
```

---

### Task 3: SQL Migration Runner And New DDL

**Files:**
- Create: `pricing_pipeline/migrations.py`
- Create: `db/migrations/V005__fremtpl_raw_model_run.sql`
- Create: `tests/test_migrations.py`
- Modify: `scripts/apply_sql_migrations.py`

- [ ] **Step 1: Write failing migration tests**

Create `tests/test_migrations.py`:

```python
from pathlib import Path

from pricing_pipeline.migrations import migration_files, split_sql_server_batches


def test_split_sql_server_batches_handles_go_lines():
    sql = "SELECT 1;\nGO\nSELECT 2;\ngo\n"
    assert split_sql_server_batches(sql) == ["SELECT 1;", "SELECT 2;"]


def test_migration_files_are_sorted(tmp_path: Path):
    (tmp_path / "V002__b.sql").write_text("SELECT 2", encoding="utf-8")
    (tmp_path / "V001__a.sql").write_text("SELECT 1", encoding="utf-8")
    assert [p.name for p in migration_files(tmp_path)] == ["V001__a.sql", "V002__b.sql"]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
rtk pytest tests/test_migrations.py -v
```

Expected: FAIL because `pricing_pipeline.migrations` does not exist.

- [ ] **Step 3: Implement migration runner module**

Create `pricing_pipeline/migrations.py`:

```python
from __future__ import annotations

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine


def split_sql_server_batches(sql_text: str) -> list[str]:
    batches: list[str] = []
    current: list[str] = []
    for line in sql_text.splitlines():
        if line.strip().upper() == "GO":
            batch = "\n".join(current).strip()
            if batch:
                batches.append(batch)
            current = []
        else:
            current.append(line)
    batch = "\n".join(current).strip()
    if batch:
        batches.append(batch)
    return batches


def migration_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("V*.sql"))


def apply_migrations(engine: Engine, migrations_dir: Path) -> list[str]:
    with engine.begin() as con:
        con.execute(
            text(
                """
                IF OBJECT_ID('dbo.SCHEMA_MIGRATION', 'U') IS NULL
                CREATE TABLE dbo.SCHEMA_MIGRATION (
                    version_file NVARCHAR(256) NOT NULL PRIMARY KEY,
                    applied_ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()
                );
                """
            )
        )

    applied: list[str] = []
    for path in migration_files(migrations_dir):
        with engine.begin() as con:
            exists = con.execute(
                text("SELECT 1 FROM dbo.SCHEMA_MIGRATION WHERE version_file = :name"),
                {"name": path.name},
            ).scalar()
        if exists:
            continue
        sql_text = path.read_text(encoding="utf-8")
        with engine.begin() as con:
            for batch in split_sql_server_batches(sql_text):
                con.execute(text(batch))
            con.execute(
                text("INSERT INTO dbo.SCHEMA_MIGRATION(version_file) VALUES (:name)"),
                {"name": path.name},
            )
        applied.append(path.name)
    return applied
```

Create `db/migrations/V005__fremtpl_raw_model_run.sql`:

```sql
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

CREATE UNIQUE INDEX UX_MODEL_RUN_AIRFLOW
ON pricing.MODEL_RUN(dag_id, airflow_run_id, model_name);
GO
```

Modify `scripts/apply_sql_migrations.py` so it imports and calls `pricing_pipeline.migrations.apply_migrations`.

- [ ] **Step 4: Run tests**

Run:

```bash
rtk pytest tests/test_migrations.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/migrations.py db/migrations/V005__fremtpl_raw_model_run.sql scripts/apply_sql_migrations.py tests/test_migrations.py
rtk git commit -m "feat: add pricing database migrations"
```

---

### Task 4: Official Airflow 3.2.1 Compose And Custom Image

**Files:**
- Create: `airflow/Dockerfile`
- Replace: `docker-compose.yml`
- Modify: `requirements.txt`
- Create: `scripts/smoke_check.py`
- Create: `tests/test_runtime_contract.py`

- [ ] **Step 1: Write failing runtime contract tests**

Create `tests/test_runtime_contract.py`:

```python
from pathlib import Path

import yaml


def test_compose_uses_airflow_321_services():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    for name in [
        "airflow-apiserver",
        "airflow-scheduler",
        "airflow-dag-processor",
        "airflow-worker",
        "airflow-triggerer",
        "postgres",
        "redis",
        "flower",
        "mssql",
        "mlflow",
    ]:
        assert name in services
    assert services["flower"]["profiles"] == ["flower"]
    assert "redis://:@redis:6379/0" in str(compose["x-airflow-common"]["environment"])


def test_airflow_image_uses_python_314_base():
    dockerfile = Path("airflow/Dockerfile").read_text(encoding="utf-8")
    assert "apache/airflow:3.2.1-python3.14" in dockerfile
    assert "msodbcsql18" in dockerfile
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
rtk pytest tests/test_runtime_contract.py -v
```

Expected: FAIL because the compose and Dockerfile do not match the Airflow 3.2.1 architecture.

- [ ] **Step 3: Implement runtime files**

Create `requirements.txt`:

```text
apache-airflow-providers-microsoft-mssql
git+https://github.com/StrudelDoodleS/superglm.git
mlflow
numpy
openpyxl
pandas
pyodbc
python-dotenv
PyYAML
scikit-learn
sqlalchemy
```

Create `airflow/Dockerfile`:

```dockerfile
FROM apache/airflow:3.2.1-python3.14

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gnupg2 apt-transport-https ca-certificates unixodbc-dev git \
    && curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft-prod.gpg \
    && curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
```

Create `docker-compose.yml` by downloading the official Airflow 3.2.1 CeleryExecutor file and applying the project additions in the same edit. Keep the official Redis broker and optional Flower profile; do not replace Redis with RabbitMQ for the first build.

```bash
rtk curl -Lf https://airflow.apache.org/docs/apache-airflow/3.2.1/docker-compose.yaml -o docker-compose.yml
```

Set the Airflow image/build section in `x-airflow-common` to:

```yaml
x-airflow-common:
  &airflow-common
  image: ${AIRFLOW_IMAGE_NAME:-airflow-superglm:3.2.1-python3.14}
  build:
    context: .
    dockerfile: airflow/Dockerfile
```

Keep the official Airflow services and add these project services:

```yaml
services:
  mssql:
    image: mcr.microsoft.com/mssql/server:2022-latest
    container_name: pricing_mssql
    environment:
      ACCEPT_EULA: "Y"
      MSSQL_SA_PASSWORD: "${MSSQL_PASSWORD:-YourStrong(!)Password123}"
      MSSQL_PID: "Developer"
    ports:
      - "1433:1433"
    volumes:
      - ./state/mssql/data:/var/opt/mssql
    healthcheck:
      test: ["CMD-SHELL", "/opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P \"$${MSSQL_SA_PASSWORD}\" -C -Q \"SELECT 1\""]
      interval: 30s
      timeout: 10s
      retries: 10
      start_period: 60s

  mlflow:
    image: ${AIRFLOW_IMAGE_NAME:-airflow-superglm:3.2.1-python3.14}
    container_name: pricing_mlflow
    command: >
      bash -c "mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri mssql+pyodbc:///?odbc_connect=$${MLFLOW_BACKEND_ODBC}
      --artifacts-destination /mlflow/artifacts
      --serve-artifacts"
    environment:
      MLFLOW_BACKEND_ODBC: "DRIVER={ODBC Driver 18 for SQL Server};SERVER=mssql,1433;DATABASE=MLflowTracking;UID=sa;PWD=${MSSQL_PASSWORD:-YourStrong(!)Password123};Encrypt=no;TrustServerCertificate=yes;"
    ports:
      - "5000:5000"
    volumes:
      - ./state/mlflow/artifacts:/mlflow/artifacts
    depends_on:
      mssql:
        condition: service_healthy

  cloudbeaver:
    image: dbeaver/cloudbeaver:latest
    container_name: pricing_cloudbeaver
    ports:
      - "8978:8978"
    volumes:
      - ./state/cloudbeaver/workspace:/opt/cloudbeaver/workspace
    depends_on:
      mssql:
        condition: service_healthy
```

Add these mounts to `x-airflow-common.volumes`:

```yaml
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./config:/opt/airflow/config
      - ./plugins:/opt/airflow/plugins
      - ./pricing_pipeline:/opt/airflow/pricing_pipeline
      - ./db:/opt/pricing/db
      - ./state/rating_exports:/opt/pricing/state/rating_exports
```

Keep the official optional Flower service with:

```yaml
  flower:
    <<: *airflow-common
    command: celery flower
    profiles:
      - flower
    ports:
      - "5555:5555"
```

Create `scripts/smoke_check.py`:

```python
from __future__ import annotations

from superglm import SuperGLM


def main() -> None:
    assert hasattr(SuperGLM, "export_rating_tables")
    print("smoke_check=ok")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and image smoke check**

Run:

```bash
rtk pytest tests/test_runtime_contract.py -v
rtk docker compose build airflow-apiserver
rtk docker compose run --rm airflow-apiserver python /opt/pricing/scripts/smoke_check.py
```

Expected: tests PASS; smoke check prints `smoke_check=ok`.

- [ ] **Step 5: Commit**

```bash
rtk git add airflow/Dockerfile docker-compose.yml requirements.txt scripts/smoke_check.py tests/test_runtime_contract.py
rtk git commit -m "feat: add Airflow 3 runtime stack"
```

---

### Task 5: freMTPL Raw Loader

**Files:**
- Create: `pricing_pipeline/fremtpl.py`
- Create: `tests/test_fremtpl.py`
- Modify: `scripts/load_fremtpl_manifest.py` or create `scripts/load_fremtpl_raw.py`

- [ ] **Step 1: Write failing freMTPL tests**

Create `tests/test_fremtpl.py`:

```python
import pandas as pd
import pytest

from pricing_pipeline.fremtpl import (
    FREMTPL_COLUMNS,
    fremtpl_insert_rows,
    prepare_fremtpl_raw_frame,
    validate_fremtpl_raw,
)


def test_prepare_fremtpl_raw_preserves_expected_columns():
    frame = pd.DataFrame(
        {
            "IDpol": [1],
            "ClaimNb": [0],
            "Exposure": [0.5],
            "Area": ["A"],
            "VehPower": [6],
            "VehAge": [3],
            "DrivAge": [45],
            "BonusMalus": [50],
            "VehBrand": ["B1"],
            "VehGas": ["Regular"],
            "Density": [123.0],
            "Region": ["R1"],
        }
    )
    out = prepare_fremtpl_raw_frame(frame)
    assert list(out.columns) == FREMTPL_COLUMNS
    assert out.loc[0, "Exposure"] == 0.5


def test_validate_fremtpl_raw_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        validate_fremtpl_raw(pd.DataFrame({"IDpol": [1]}))


def test_fremtpl_insert_rows_preserves_order_and_converts_missing_to_none():
    frame = pd.DataFrame(
        {
            "IDpol": [1],
            "ClaimNb": [0],
            "Exposure": [0.5],
            "Area": [None],
            "VehPower": [6],
            "VehAge": [3],
            "DrivAge": [45],
            "BonusMalus": [50],
            "VehBrand": ["B1"],
            "VehGas": ["Regular"],
            "Density": [float("nan")],
            "Region": ["R1"],
        }
    )
    rows = fremtpl_insert_rows(prepare_fremtpl_raw_frame(frame))
    assert rows == [
        (1, 0, 0.5, None, 6, 3, 45, 50, "B1", "Regular", None, "R1")
    ]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk pytest tests/test_fremtpl.py -v
```

Expected: FAIL because `pricing_pipeline.fremtpl` does not exist.

- [ ] **Step 3: Implement raw loader module**

Create `pricing_pipeline/fremtpl.py`:

```python
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml
from sqlalchemy import text
from sqlalchemy.engine import Engine

FREMTPL_OPENML_ID = 41214
FREMTPL_DATASET_NAME = "freMTPL2freq"
FREMTPL_COLUMNS = [
    "IDpol",
    "ClaimNb",
    "Exposure",
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region",
]


def fetch_fremtpl() -> pd.DataFrame:
    return fetch_openml(data_id=FREMTPL_OPENML_ID, as_frame=True).frame.reset_index(drop=True)


def validate_fremtpl_raw(frame: pd.DataFrame) -> None:
    missing = [column for column in FREMTPL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"freMTPL raw data missing columns: {missing}")


def prepare_fremtpl_raw_frame(frame: pd.DataFrame) -> pd.DataFrame:
    validate_fremtpl_raw(frame)
    out = frame.loc[:, FREMTPL_COLUMNS].copy()
    out["IDpol"] = out["IDpol"].astype("int64")
    out["ClaimNb"] = out["ClaimNb"].astype("int64")
    return out


def _db_value(value: Any) -> Any:
    return None if pd.isna(value) else value


def fremtpl_insert_rows(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [
        tuple(_db_value(value) for value in row)
        for row in frame.loc[:, FREMTPL_COLUMNS].itertuples(index=False, name=None)
    ]


def bulk_insert_fremtpl_raw(engine: Engine, frame: pd.DataFrame) -> int:
    rows = fremtpl_insert_rows(frame)
    if not rows:
        return 0
    placeholders = ", ".join(["?"] * len(FREMTPL_COLUMNS))
    columns = ", ".join(FREMTPL_COLUMNS)
    sql = f"INSERT INTO pricing.FREMTPL_RAW ({columns}) VALUES ({placeholders})"
    connection = engine.raw_connection()
    try:
        cursor = connection.cursor()
        cursor.fast_executemany = True
        cursor.executemany(sql, rows)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return len(rows)


def load_fremtpl_raw(engine: Engine, *, replace: bool = False) -> int:
    frame = prepare_fremtpl_raw_frame(fetch_fremtpl())
    with engine.begin() as con:
        existing = con.execute(text("SELECT COUNT_BIG(*) FROM pricing.FREMTPL_RAW")).scalar_one()
        if existing and not replace:
            return int(existing)
        if replace:
            con.execute(text("TRUNCATE TABLE pricing.FREMTPL_RAW"))
    return bulk_insert_fremtpl_raw(engine, frame)
```

Create `scripts/load_fremtpl_raw.py`:

```python
from __future__ import annotations

import argparse
import os

from pricing_pipeline.config import Settings
from pricing_pipeline.db import get_engine
from pricing_pipeline.fremtpl import load_fremtpl_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env(os.environ)
    rows = load_fremtpl_raw(get_engine(settings), replace=args.replace)
    print(f"fremtpl_raw_rows={rows}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run:

```bash
rtk pytest tests/test_fremtpl.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/fremtpl.py scripts/load_fremtpl_raw.py tests/test_fremtpl.py
rtk git commit -m "feat: add freMTPL raw loader"
```

---

### Task 6: Dataset Manifest From Raw Table

This task persists row-level CV indices. `pricing.DATASET_ROW_KEY.row_ordinal`
is the deterministic dataset index, `cv_fold_no` is the assigned fold for that
row, and `pricing.CV_SPLIT` records which folds are train/test for each split.

**Files:**
- Create: `pricing_pipeline/manifest.py`
- Create: `tests/test_manifest.py`
- Modify: `scripts/load_fremtpl_manifest.py`

- [ ] **Step 1: Write failing manifest tests**

Create `tests/test_manifest.py`:

```python
import pandas as pd

from pricing_pipeline.manifest import build_column_metadata, build_cv_splits, build_row_keys


def test_build_row_keys_uses_idpol_and_deterministic_folds():
    frame = pd.DataFrame({"IDpol": [10, 20, 30, 40]})
    keys = build_row_keys(frame, manifest_id="m1", n_splits=2, random_state=42)
    assert keys["source_pk_text"].tolist() == ["IDpol=10", "IDpol=20", "IDpol=30", "IDpol=40"]
    assert sorted(keys["cv_fold_no"].unique().tolist()) == [1, 2]


def test_build_cv_splits_records_train_folds():
    splits = build_cv_splits("m1", n_splits=3)
    assert splits.loc[0, "train_folds_json"] == "[2, 3]"
    assert splits.loc[2, "test_fold_no"] == 3


def test_build_column_metadata_marks_roles():
    frame = pd.DataFrame({"IDpol": [1], "ClaimNb": [0], "Exposure": [1.0], "Area": ["A"]})
    cols = build_column_metadata(frame, manifest_id="m1")
    roles = dict(zip(cols["column_name"], cols["column_role"], strict=True))
    assert roles["IDpol"] == "KEY"
    assert roles["ClaimNb"] == "TARGET"
    assert roles["Exposure"] == "WEIGHT"
    assert roles["Area"] == "FEATURE"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk pytest tests/test_manifest.py -v
```

Expected: FAIL because `pricing_pipeline.manifest` does not exist.

- [ ] **Step 3: Implement manifest module**

Create `pricing_pipeline/manifest.py`:

```python
from __future__ import annotations

import json
from datetime import date
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sqlalchemy import text
from sqlalchemy.engine import Engine


def new_manifest_id(dataset_name: str) -> str:
    return f"{dataset_name}_{date.today().isoformat()}_{uuid4().hex[:10]}"


def build_column_metadata(frame: pd.DataFrame, *, manifest_id: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "manifest_id": manifest_id,
            "ordinal_no": np.arange(1, len(frame.columns) + 1),
            "column_name": frame.columns,
            "pandas_dtype": frame.dtypes.astype(str).values,
            "null_count": frame.isna().sum().astype(int).values,
            "distinct_count": frame.nunique(dropna=True).astype(int).values,
        }
    )
    out["column_role"] = "FEATURE"
    out.loc[out["column_name"].eq("IDpol"), "column_role"] = "KEY"
    out.loc[out["column_name"].eq("ClaimNb"), "column_role"] = "TARGET"
    out.loc[out["column_name"].eq("Exposure"), "column_role"] = "WEIGHT"
    return out[
        [
            "manifest_id",
            "ordinal_no",
            "column_name",
            "column_role",
            "pandas_dtype",
            "null_count",
            "distinct_count",
        ]
    ]


def build_row_keys(
    frame: pd.DataFrame, *, manifest_id: str, n_splits: int, random_state: int
) -> pd.DataFrame:
    fold_no = np.empty(len(frame), dtype=np.int16)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (_, test_idx) in enumerate(kf.split(frame), start=1):
        fold_no[test_idx] = fold
    return pd.DataFrame(
        {
            "manifest_id": manifest_id,
            "source_pk_text": "IDpol=" + frame["IDpol"].astype(str),
            "row_ordinal": np.arange(1, len(frame) + 1, dtype=np.int64),
            "cv_fold_no": fold_no.astype(int),
        }
    )


def build_cv_splits(manifest_id: str, *, n_splits: int) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "manifest_id": manifest_id,
            "split_no": np.arange(1, n_splits + 1),
            "test_fold_no": np.arange(1, n_splits + 1),
        }
    )
    out["train_folds_json"] = out["test_fold_no"].map(
        lambda test_fold: json.dumps([f for f in range(1, n_splits + 1) if f != test_fold])
    )
    return out[["manifest_id", "split_no", "train_folds_json", "test_fold_no"]]


def create_fremtpl_manifest(
    engine: Engine,
    *,
    manifest_id: str,
    n_splits: int = 5,
    random_state: int = 42,
    created_by: str = "airflow",
) -> str:
    frame = pd.read_sql_query("SELECT * FROM pricing.FREMTPL_RAW ORDER BY IDpol", engine)
    manifest = pd.DataFrame(
        [
            {
                "manifest_id": manifest_id,
                "dataset_name": "freMTPL2freq",
                "source_system": "openml_41214",
                "data_as_of_date": date.today(),
                "row_count": int(len(frame)),
                "pk_columns_json": json.dumps(["IDpol"]),
                "target_column": "ClaimNb",
                "weight_column": "Exposure",
                "created_by": created_by,
            }
        ]
    )
    manifest.to_sql("DATASET_MANIFEST", engine, schema="pricing", if_exists="append", index=False)
    build_column_metadata(frame, manifest_id=manifest_id).to_sql(
        "DATASET_COLUMN", engine, schema="pricing", if_exists="append", index=False
    )
    row_keys = build_row_keys(
        frame, manifest_id=manifest_id, n_splits=n_splits, random_state=random_state
    )
    with engine.begin() as con:
        con.execute(text("TRUNCATE TABLE pricing.STG_DATASET_ROW_KEY"))
    row_keys.to_sql(
        "STG_DATASET_ROW_KEY",
        engine,
        schema="pricing",
        if_exists="append",
        index=False,
        chunksize=20000,
    )
    with engine.begin() as con:
        con.execute(
            text(
                """
                INSERT INTO pricing.DATASET_ROW_KEY (
                    manifest_id, row_key_hash, source_pk_text, row_ordinal, cv_fold_no
                )
                SELECT manifest_id, HASHBYTES('SHA2_256', source_pk_text), source_pk_text,
                       row_ordinal, cv_fold_no
                FROM pricing.STG_DATASET_ROW_KEY
                WHERE manifest_id = :manifest_id
                """
            ),
            {"manifest_id": manifest_id},
        )
    build_cv_splits(manifest_id, n_splits=n_splits).to_sql(
        "CV_SPLIT", engine, schema="pricing", if_exists="append", index=False
    )
    return manifest_id
```

Modify `scripts/load_fremtpl_manifest.py` to read from `pricing.FREMTPL_RAW` through `create_fremtpl_manifest(...)`.

- [ ] **Step 4: Run tests**

Run:

```bash
rtk pytest tests/test_manifest.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/manifest.py scripts/load_fremtpl_manifest.py tests/test_manifest.py
rtk git commit -m "feat: create manifests from raw freMTPL table"
```

---

### Task 7: SuperGLM Training And MLflow Tracking

**Files:**
- Create: `pricing_pipeline/training.py`
- Create: `pricing_pipeline/mlflow_tracking.py`
- Create: `tests/test_training.py`
- Create: `scripts/train_superglm.py`

- [ ] **Step 1: Write failing training tests**

Create `tests/test_training.py`:

```python
import numpy as np
import pandas as pd

from pricing_pipeline.training import FEATURE_COLUMNS, build_training_frame


def test_build_training_frame_derives_log_density_and_offset():
    raw = pd.DataFrame(
        {
            "ClaimNb": [0, 1],
            "Exposure": [0.5, 1.0],
            "Area": ["A", "B"],
            "VehPower": [6, 7],
            "VehAge": [3, 5],
            "DrivAge": [45, 52],
            "BonusMalus": [50, 70],
            "VehBrand": ["B1", "B2"],
            "VehGas": ["Regular", "Diesel"],
            "Density": [100.0, 400.0],
            "Region": ["R1", "R2"],
        }
    )
    X, y, exposure, offset = build_training_frame(raw)
    assert "LogDensity" in X.columns
    assert list(X.columns) == FEATURE_COLUMNS
    assert y.tolist() == [0, 1]
    assert exposure.tolist() == [0.5, 1.0]
    assert np.allclose(offset, np.log(exposure))
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk pytest tests/test_training.py -v
```

Expected: FAIL because `pricing_pipeline.training` does not exist.

- [ ] **Step 3: Implement training module and MLflow helpers**

Create `pricing_pipeline/training.py`:

```python
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine
from superglm import Categorical, Numeric, Spline, SuperGLM

FEATURE_COLUMNS = [
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "LogDensity",
    "Area",
    "VehPower",
    "VehBrand",
    "VehGas",
    "Region",
]


def build_training_frame(raw: pd.DataFrame):
    frame = raw.copy()
    required = {"ClaimNb", "Exposure", "Density", *FEATURE_COLUMNS} - {"LogDensity"}
    missing = [column for column in sorted(required) if column not in frame.columns]
    if missing:
        raise ValueError(f"training data missing columns: {missing}")
    frame = frame.loc[frame["Exposure"].astype(float) > 0].copy()
    frame["LogDensity"] = np.log(frame["Density"].astype(float).clip(lower=1.0))
    X = frame.loc[:, FEATURE_COLUMNS].copy()
    y = frame["ClaimNb"].astype(float).to_numpy()
    exposure = frame["Exposure"].astype(float).to_numpy()
    offset = np.log(exposure)
    return X, y, exposure, offset


def build_model() -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "VehAge": Spline(),
            "DrivAge": Spline(),
            "BonusMalus": Spline(),
            "LogDensity": Numeric(),
            "Area": Categorical(),
            "VehPower": Categorical(),
            "VehBrand": Categorical(),
            "VehGas": Categorical(),
            "Region": Categorical(),
        },
        discrete=True,
        n_bins=256,
    )


def train_superglm(engine: Engine, *, model_dir: Path, mlflow_experiment: str) -> dict[str, str]:
    raw = pd.read_sql_query("SELECT * FROM pricing.FREMTPL_RAW ORDER BY IDpol", engine)
    X, y, exposure, offset = build_training_frame(raw)
    model = build_model()
    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run() as run:
        mlflow.log_param("family", "poisson")
        mlflow.log_param("target", "ClaimNb")
        mlflow.log_param("offset", "log(Exposure)")
        mlflow.log_param("row_count", len(X))
        model.fit_reml(X, y, sample_weight=exposure, offset=offset)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "superglm_model.pkl"
        import pickle

        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_metric("deviance", float(model.result.deviance))
        return {"mlflow_run_id": run.info.run_id, "model_path": str(model_path)}
```

Create `pricing_pipeline/mlflow_tracking.py`:

```python
from __future__ import annotations

import mlflow


def configure_mlflow(tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
```

Create `scripts/train_superglm.py`:

```python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from pricing_pipeline.config import Settings
from pricing_pipeline.db import get_engine
from pricing_pipeline.mlflow_tracking import configure_mlflow
from pricing_pipeline.training import train_superglm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/opt/pricing/state/rating_exports/manual")
    parser.add_argument("--experiment", default="pricing-mtpl-frequency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env(os.environ)
    configure_mlflow(settings.mlflow_tracking_uri)
    result = train_superglm(
        get_engine(settings),
        model_dir=Path(args.model_dir),
        mlflow_experiment=args.experiment,
    )
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run:

```bash
rtk pytest tests/test_training.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/training.py pricing_pipeline/mlflow_tracking.py scripts/train_superglm.py tests/test_training.py
rtk git commit -m "feat: train SuperGLM with MLflow tracking"
```

---

### Task 8: Rating Export, Staging, Publishing, And Lineage

**Files:**
- Create: `pricing_pipeline/rating_export.py`
- Create: `pricing_pipeline/rating_package.py`
- Create: `pricing_pipeline/lineage.py`
- Create: `pricing_pipeline/pipeline.py`
- Create: `tests/test_rating_export.py`
- Modify: `scripts/load_superglm_excel_to_staging.py`
- Modify: `scripts/load_staging_to_rating_package.py`

- [ ] **Step 1: Write failing rating export tests**

Create `tests/test_rating_export.py`:

```python
from pathlib import Path

from pricing_pipeline.rating_export import build_export_id, build_rating_export_path


def test_build_export_id_is_path_safe():
    export_id = build_export_id("MTPL_FREQ", "scheduled__2026-04-27T10:30:00+00:00")
    assert export_id == "mtpl_freq__scheduled__20260427t1030000000"


def test_build_rating_export_path_uses_model_and_date(tmp_path: Path):
    path = build_rating_export_path(
        tmp_path,
        model_name="MTPL_FREQ",
        logical_date="2026-04-27",
        export_id="mtpl_freq__run1",
    )
    assert path == tmp_path / "MTPL_FREQ" / "2026-04-27" / "mtpl_freq__run1" / "rating_tables.xlsx"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk pytest tests/test_rating_export.py -v
```

Expected: FAIL because `pricing_pipeline.rating_export` does not exist.

- [ ] **Step 3: Implement export and lineage helpers**

Create `pricing_pipeline/rating_export.py`:

```python
from __future__ import annotations

import re
from pathlib import Path

import mlflow


def build_export_id(model_name: str, run_id: str) -> str:
    raw = f"{model_name}__{run_id}".lower()
    raw = re.sub(r"[^a-z0-9_]+", "", raw.replace(":", "").replace("-", ""))
    return re.sub(r"_+", "_", raw).strip("_")


def build_rating_export_path(
    root: Path, *, model_name: str, logical_date: str, export_id: str
) -> Path:
    return root / model_name / logical_date / export_id / "rating_tables.xlsx"


def export_rating_tables(model, X, y, exposure, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.export_rating_tables(output_path, X, y, sample_weight=exposure, n_bins=150)
    mlflow.log_artifact(str(output_path), artifact_path="rating_tables")
    return output_path
```

Modify `scripts/load_superglm_excel_to_staging.py` so it exposes this callable in addition to `main()`:

```python
def stage_rating_export(
    engine,
    *,
    workbook_path: Path,
    export_id: str,
    model_name: str,
    model_version: str | None,
    effective_from: str,
    effective_to: str | None = None,
    created_by: str = "python",
    replace: bool = False,
) -> None:
    args = argparse.Namespace(
        xlsx=str(workbook_path),
        sheet="Rating Tables",
        export_id=export_id,
        model_name=model_name,
        model_version=model_version,
        effective_from=effective_from,
        effective_to=effective_to,
        base_rate=None,
        base_rate_cell="C2",
        term_row=5,
        header_row=7,
        data_start_row=8,
        term_type_map_json="{}",
        interaction_features_json="{}",
        created_by=created_by,
        replace=replace,
    )
    export_df, rate_df, level_df = build_staging_frames(args)
    if replace:
        with engine.begin() as con:
            con.execute(text("DELETE FROM pricing.STG_CELL_LEVEL WHERE export_id = :export_id"), {"export_id": export_id})
            con.execute(text("DELETE FROM pricing.STG_RATE_CELL WHERE export_id = :export_id"), {"export_id": export_id})
            con.execute(text("DELETE FROM pricing.STG_RATING_EXPORT WHERE export_id = :export_id"), {"export_id": export_id})
    export_df.to_sql("STG_RATING_EXPORT", engine, schema="pricing", if_exists="append", index=False)
    rate_df.to_sql("STG_RATE_CELL", engine, schema="pricing", if_exists="append", index=False, chunksize=5000)
    level_df.to_sql("STG_CELL_LEVEL", engine, schema="pricing", if_exists="append", index=False, chunksize=5000)
```

Modify `scripts/load_staging_to_rating_package.py` so it exposes this callable in addition to `main()`:

```python
def publish_rating_package(
    engine,
    *,
    export_id: str,
    pointer_name: str | None,
    created_by: str = "python",
    package_status: str = "DRAFT",
) -> int:
    args = argparse.Namespace(
        export_id=export_id,
        created_by=created_by,
        package_status=package_status,
        set_pointer=pointer_name,
    )
    return load_staging_to_rating_package(engine, args)
```

Create `pricing_pipeline/rating_package.py`:

```python
from __future__ import annotations

from pathlib import Path

from scripts.load_staging_to_rating_package import publish_rating_package as _publish
from scripts.load_superglm_excel_to_staging import stage_rating_export as _stage


def stage_rating_export(
    engine,
    *,
    workbook_path: Path,
    export_id: str,
    model_name: str,
    model_version: str | None,
    effective_from: str,
    created_by: str,
    replace: bool,
) -> None:
    _stage(
        engine,
        workbook_path=workbook_path,
        export_id=export_id,
        model_name=model_name,
        model_version=model_version,
        effective_from=effective_from,
        created_by=created_by,
        replace=replace,
    )


def publish_rating_package(
    engine,
    *,
    export_id: str,
    pointer_name: str,
    created_by: str,
) -> int:
    return _publish(
        engine,
        export_id=export_id,
        pointer_name=pointer_name,
        created_by=created_by,
        package_status="DRAFT",
    )
```

Create `pricing_pipeline/lineage.py`:

```python
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


def record_model_run(
    engine: Engine,
    *,
    dag_id: str,
    airflow_run_id: str,
    mlflow_run_id: str,
    manifest_id: str,
    export_id: str,
    model_name: str,
    model_version: str,
    rate_package_id: int | None,
    rating_workbook_path: str,
    run_status: str,
    created_by: str,
) -> None:
    with engine.begin() as con:
        con.execute(
            text(
                """
                INSERT INTO pricing.MODEL_RUN (
                    dag_id, airflow_run_id, mlflow_run_id, manifest_id, export_id,
                    model_name, model_version, rate_package_id, rating_workbook_path,
                    run_status, completed_ts, created_by
                )
                VALUES (
                    :dag_id, :airflow_run_id, :mlflow_run_id, :manifest_id, :export_id,
                    :model_name, :model_version, :rate_package_id, :rating_workbook_path,
                    :run_status, SYSUTCDATETIME(), :created_by
                )
                """
            ),
            {
                "dag_id": dag_id,
                "airflow_run_id": airflow_run_id,
                "mlflow_run_id": mlflow_run_id,
                "manifest_id": manifest_id,
                "export_id": export_id,
                "model_name": model_name,
                "model_version": model_version,
                "rate_package_id": rate_package_id,
                "rating_workbook_path": rating_workbook_path,
                "run_status": run_status,
                "created_by": created_by,
            },
        )
```

Create `pricing_pipeline/pipeline.py`:

```python
from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
import pandas as pd

from pricing_pipeline.config import Settings
from pricing_pipeline.lineage import record_model_run
from pricing_pipeline.mlflow_tracking import configure_mlflow
from pricing_pipeline.rating_export import build_export_id, build_rating_export_path
from pricing_pipeline.rating_package import publish_rating_package, stage_rating_export
from pricing_pipeline.training import build_model, build_training_frame


def run_training_export_publish(
    engine,
    *,
    settings: Settings,
    manifest_id: str,
    dag_id: str,
    airflow_run_id: str,
    logical_date: str,
    created_by: str = "airflow",
) -> dict[str, str]:
    configure_mlflow(settings.mlflow_tracking_uri)
    model_name = "MTPL_FREQ"
    model_version = logical_date.replace("-", "")
    export_id = build_export_id(model_name, airflow_run_id)
    workbook_path = build_rating_export_path(
        settings.rating_export_root,
        model_name=model_name,
        logical_date=logical_date,
        export_id=export_id,
    )

    raw = pd.read_sql_query("SELECT * FROM pricing.FREMTPL_RAW ORDER BY IDpol", engine)
    X, y, exposure, offset = build_training_frame(raw)
    model = build_model()

    mlflow.set_experiment("pricing-mtpl-frequency")
    with mlflow.start_run() as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("manifest_id", manifest_id)
        mlflow.log_param("target", "ClaimNb")
        mlflow.log_param("offset", "log(Exposure)")
        model.fit_reml(X, y, sample_weight=exposure, offset=offset)
        mlflow.log_metric("deviance", float(model.result.deviance))

        model_path = workbook_path.parent / "superglm_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        model.export_rating_tables(workbook_path, X, y, sample_weight=exposure, n_bins=150)
        mlflow.log_artifact(str(workbook_path), artifact_path="rating_tables")

        stage_rating_export(
            engine,
            workbook_path=workbook_path,
            export_id=export_id,
            model_name=model_name,
            model_version=model_version,
            effective_from=logical_date,
            created_by=created_by,
            replace=True,
        )
        rate_package_id = publish_rating_package(
            engine,
            export_id=export_id,
            pointer_name="MTPL_FREQ_UAT",
            created_by=created_by,
        )
        record_model_run(
            engine,
            dag_id=dag_id,
            airflow_run_id=airflow_run_id,
            mlflow_run_id=run.info.run_id,
            manifest_id=manifest_id,
            export_id=export_id,
            model_name=model_name,
            model_version=model_version,
            rate_package_id=rate_package_id,
            rating_workbook_path=str(workbook_path),
            run_status="SUCCESS",
            created_by=created_by,
        )
        return {
            "mlflow_run_id": run.info.run_id,
            "export_id": export_id,
            "rate_package_id": str(rate_package_id),
            "rating_workbook_path": str(workbook_path),
        }
```

Refactor `scripts/load_superglm_excel_to_staging.py` and `scripts/load_staging_to_rating_package.py` so their main logic is callable without shelling out from Airflow.

- [ ] **Step 4: Run tests**

Run:

```bash
rtk pytest tests/test_rating_export.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/rating_export.py pricing_pipeline/rating_package.py pricing_pipeline/lineage.py pricing_pipeline/pipeline.py scripts/load_superglm_excel_to_staging.py scripts/load_staging_to_rating_package.py tests/test_rating_export.py
rtk git commit -m "feat: publish rating exports with lineage"
```

---

### Task 9: Airflow 3.2.1 DAG

**Files:**
- Create: `dags/pricing_superglm_pipeline.py`
- Create: `tests/test_dag_import.py`

- [ ] **Step 1: Write failing DAG import test**

Create `tests/test_dag_import.py`:

```python
import importlib.util
from pathlib import Path


def test_pricing_dag_file_imports_without_airflow_execution():
    path = Path("dags/pricing_superglm_pipeline.py")
    spec = importlib.util.spec_from_file_location("pricing_superglm_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "pricing_superglm_pipeline")
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
rtk pytest tests/test_dag_import.py -v
```

Expected: FAIL because the DAG does not exist.

- [ ] **Step 3: Implement Airflow Task SDK DAG**

Create `dags/pricing_superglm_pipeline.py`:

```python
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from airflow.sdk import dag, get_current_context, task

from pricing_pipeline.config import Settings
from pricing_pipeline.db import ensure_database, get_engine
from pricing_pipeline.fremtpl import load_fremtpl_raw
from pricing_pipeline.manifest import create_fremtpl_manifest, new_manifest_id
from pricing_pipeline.migrations import apply_migrations
from pricing_pipeline.pipeline import run_training_export_publish


def _settings() -> Settings:
    return Settings.from_env(os.environ)


@dag(
    dag_id="pricing_superglm_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["pricing", "superglm", "mlflow"],
)
def pricing_superglm_pipeline():
    @task
    def apply_pricing_migrations() -> list[str]:
        settings = _settings()
        ensure_database(settings, settings.pricing_database)
        engine = get_engine(settings)
        return apply_migrations(engine, Path("/opt/pricing/db/migrations"))

    @task
    def load_raw() -> int:
        settings = _settings()
        return load_fremtpl_raw(get_engine(settings))

    @task
    def create_manifest() -> str:
        settings = _settings()
        manifest_id = new_manifest_id("freMTPL2freq")
        return create_fremtpl_manifest(get_engine(settings), manifest_id=manifest_id)

    @task
    def train_and_publish(manifest_id: str) -> dict[str, str]:
        settings = _settings()
        context = get_current_context()
        logical_date = context["logical_date"].date().isoformat()
        return run_training_export_publish(
            get_engine(settings),
            settings=settings,
            manifest_id=manifest_id,
            dag_id=context["dag"].dag_id,
            airflow_run_id=context["run_id"],
            logical_date=logical_date,
        )

    migrations = apply_pricing_migrations()
    rows = load_raw()
    manifest = create_manifest()
    publish = train_and_publish(manifest)

    migrations >> rows >> manifest >> publish


pricing_superglm_pipeline()
```

- [ ] **Step 4: Run tests**

Run:

```bash
rtk pytest tests/test_dag_import.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add dags/pricing_superglm_pipeline.py tests/test_dag_import.py
rtk git commit -m "feat: add Airflow pricing pipeline DAG"
```

---

### Task 10: End-To-End Smoke Verification And Runbook

**Files:**
- Create: `scripts/run_local_pipeline.sh`
- Modify: `README.md`
- Create: `tests/test_readme_contract.py`

- [ ] **Step 1: Write failing runbook test**

Create `tests/test_readme_contract.py`:

```python
from pathlib import Path


def test_readme_documents_state_and_no_down_v():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "state/" in text
    assert "docker compose down -v" in text
    assert "Airflow 3.2.1" in text
    assert "MLflow" in text
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
rtk pytest tests/test_readme_contract.py -v
```

Expected: FAIL until README has the required runbook details.

- [ ] **Step 3: Add local run script and README runbook**

Create `scripts/run_local_pipeline.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p state/mssql/data state/mlflow/artifacts state/rating_exports logs config plugins
docker compose build
docker compose up -d postgres redis mssql mlflow airflow-apiserver airflow-scheduler airflow-dag-processor airflow-worker airflow-triggerer
docker compose run --rm airflow-apiserver python /opt/pricing/scripts/smoke_check.py
docker compose exec airflow-apiserver airflow dags trigger pricing_superglm_pipeline
```

Append to `README.md`:

```markdown
## Local Runbook

1. Create `.env` from `.env.example`.
2. Run `./scripts/run_local_pipeline.sh`.
3. Open Airflow at <http://localhost:8080>.
4. Open MLflow at <http://localhost:5000>.
5. Optionally start Flower with `docker compose --profile flower up -d flower`
   and open it at <http://localhost:5555>.
6. Inspect SQL Server with CloudBeaver or another SQL client.

Durable local files are stored under `state/`, including MLflow artifacts and
rating table workbooks. Do not run `docker compose down -v` unless you intend to
remove Docker-managed volumes. The critical project artifacts are bind-mounted
under `state/`, but ordinary filesystem deletion can still remove them.
```

- [ ] **Step 4: Run full verification**

Run:

```bash
rtk pytest -v
rtk docker compose config
rtk docker compose build
```

Expected: tests PASS; compose config renders; Docker image builds.

Run this only after the image builds:

```bash
rtk docker compose run --rm airflow-apiserver python /opt/pricing/scripts/smoke_check.py
```

Expected: `smoke_check=ok`.

- [ ] **Step 5: Commit**

```bash
rtk chmod +x scripts/run_local_pipeline.sh
rtk git add README.md scripts/run_local_pipeline.sh tests/test_readme_contract.py
rtk git commit -m "docs: add local pipeline runbook"
```

---

## Final Verification

- [ ] Run all unit tests:

```bash
rtk pytest -v
```

- [ ] Render Compose:

```bash
rtk docker compose config
```

- [ ] Build the custom image:

```bash
rtk docker compose build
```

- [ ] Run container smoke check:

```bash
rtk docker compose run --rm airflow-apiserver python /opt/pricing/scripts/smoke_check.py
```

- [ ] Confirm git state:

```bash
rtk git status --short
```

Expected: no uncommitted tracked changes. `pricing_python_starter.zip` is ignored.

---

## Spec Coverage Review

- Airflow 3.2.1 official-style compose: Task 4.
- Python 3.14 custom image with ODBC and SuperGLM: Task 4.
- Project-local durable `state/`: Tasks 1, 4, and 10.
- SQL Server pricing database: Tasks 2, 3, 4.
- MLflow tracking: Tasks 4, 7, 8.
- freMTPL raw table in SQL Server: Tasks 3 and 5.
- Dataset manifest and CV folds: Task 6.
- Poisson SuperGLM training with `ClaimNb` and `log(Exposure)`: Task 7.
- Rating workbook export and MLflow artifact logging: Task 8.
- Rating table staging and normalized publishing: Task 8.
- Airflow DAG orchestration: Task 9.
- Runbook and verification: Task 10.
