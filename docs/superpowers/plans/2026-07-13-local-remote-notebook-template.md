# Local and Remote Notebook Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make scaffolded pricing notebooks use persistent local SQLite by default and a privately supplied, explicitly confirmed SQL Server runtime at work.

**Architecture:** `pricing_pipeline.infra.offline_sqlite` owns reusable attached-schema SQLite setup. `pricing_pipeline.notebook` owns the analyst-facing connection modes, safe destination description, write guard, and a small SQLite publication path; existing runtime-provider behavior remains compatible. The scaffold script emits a real notebook, and the MTPL notebook demonstrates the same three database controls.

**Tech Stack:** Python 3.12, SQLAlchemy, SQLite, pandas, nbformat-compatible JSON, pytest.

---

### Task 1: Reusable notebook database targets

**Files:**
- Create: `pricing_pipeline/infra/offline_sqlite.py`
- Modify: `pricing_pipeline/notebook.py`
- Modify: `scripts/run_mtpl_frequency_offline_sqlite.py`
- Test: `tests/test_notebook_database_targets.py`

- [x] **Step 1: Write failing target tests**

Add tests which call this public API:

```python
context = connect(mode="local", local_root=tmp_path / "model" / ".local")
assert context.mode == "local"
assert context.write_allowed is True
assert "local SQLite" in context.destination
assert all(path.exists() for path in context.database_paths.values())
```

Insert a sentinel row, reconnect, and assert the row remains. Add fake-runtime tests proving remote mode requires a non-empty expected database, executes `SELECT DB_NAME()`, rejects a mismatch, exposes no server name, and sets `write_allowed` only from `allow_remote_writes`. Keep the existing positional `connect("private.module")` test green.

- [x] **Step 2: Run target tests and verify RED**

Run: `pytest tests/test_notebook_database_targets.py tests/test_notebook_workflow.py::test_connect_uses_runtime_module_without_airflow -q`

Expected: failures because explicit modes and local SQLite context fields do not exist.

- [x] **Step 3: Implement target creation and guards**

Create reusable constants and functions:

```python
OFFLINE_DDL_DIR = Path(__file__).resolve().parents[2] / "db" / "offline_sqlite"
SCHEMA_DB_FILES = {
    "pricing": "pricing.sqlite",
    "pricing_stg": "pricing_stg.sqlite",
    "mlops": "mlops.sqlite",
}

def offline_database_paths(root: str | Path) -> dict[str, Path]:
    resolved = Path(root).expanduser().resolve()
    return {name: resolved / filename for name, filename in SCHEMA_DB_FILES.items()}

def sqlite_engine_with_offline_schemas(db_paths: Mapping[str, Path]):
    for path in db_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    coordinator = next(iter(db_paths.values())).parent / "coordinator.sqlite"
    engine = create_engine(
        f"sqlite:///{coordinator.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    @event.listens_for(engine, "connect")
    def attach(dbapi_connection, _connection_record):
        for schema, path in db_paths.items():
            dbapi_connection.execute(
                f"ATTACH DATABASE ? AS {schema}",
                (str(path),),
            )
    return engine

def apply_offline_ddl(engine) -> None:
    connection = engine.raw_connection()
    try:
        for schema in SCHEMA_DB_FILES:
            connection.executescript(
                (OFFLINE_DDL_DIR / f"{schema}.sql").read_text(encoding="utf-8")
            )
        connection.commit()
    finally:
        connection.close()

def open_offline_sqlite(root: str | Path):
    paths = offline_database_paths(root)
    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    return engine, paths
```

Extend the context and connection API without breaking its old positional form:

```python
@dataclass(frozen=True)
class NotebookContext:
    engine: Any
    settings: Settings
    mode: str = "runtime"
    write_allowed: bool = True
    destination: str = "configured runtime"
    database_paths: Mapping[str, Path] = field(default_factory=dict)

    def require_write(self, operation: str) -> None:
        if not self.write_allowed:
            raise PermissionError(
                f"Remote writes are disabled for {operation}. "
                "Confirm EXPECTED_REMOTE_DATABASE and set ALLOW_REMOTE_WRITES=True."
            )

def connect(
    runtime_module: str | None = None,
    *,
    mode: str | None = None,
    local_root: str | Path | None = None,
    expected_remote_database: str | None = None,
    allow_remote_writes: bool = False,
) -> NotebookContext:
    selected = None if mode is None else mode.strip().lower()
    if selected == "local":
        return _connect_local(local_root)
    if selected == "remote":
        return _connect_remote(
            runtime_module,
            expected_database=expected_remote_database,
            allow_writes=allow_remote_writes,
        )
    if selected is not None:
        raise ValueError("mode must be 'local' or 'remote'")
    runtime = runtime_from_env_or_module(runtime_module)
    return NotebookContext(engine=runtime.get_engine(), settings=runtime.settings)
```

Local settings disable MLflow and put rating, split, and workbench artifacts below the local root. Remote mode uses `runtime_from_env_or_module`, verifies `SELECT DB_NAME()` case-insensitively, never runs DDL, and returns only the confirmed database name in `destination`. Import the reusable SQLite setup from the offline runner so it retains its current behavior.

- [x] **Step 4: Run target tests and verify GREEN**

Run: `pytest tests/test_notebook_database_targets.py tests/test_notebook_workflow.py::test_connect_uses_runtime_module_without_airflow tests/test_mtpl_offline_sqlite_runner.py -q`

Expected: all selected tests pass.

### Task 2: Guard mutations and support local audit publication

**Files:**
- Create: `pricing_pipeline/publishing/sqlite_notebook.py`
- Modify: `pricing_pipeline/notebook.py`
- Test: `tests/test_notebook_database_targets.py`

- [x] **Step 1: Write failing mutation tests**

Parametrize calls to `register_model`, `build_candidate`, `publish_candidate`, `publish_edits`, and `deploy_package` with a remote context whose writes are disabled. Assert every call raises `PermissionError` naming the operation before touching its other inputs. Add an integration test using local SQLite which registers a model twice without duplication, resolves the next model version, and publishes a minimal completed candidate into `PRICING_RATE_PACKAGE`, `MODEL_RUN`, `MODEL_RUN_METRIC`, `MODEL_RUN_DATASET`, and `MODEL_RUN_SPLIT_SET`.

- [x] **Step 2: Run mutation tests and verify RED**

Run: `pytest tests/test_notebook_database_targets.py -q`

Expected: failures because mutation entry points do not call a central guard and SQL Server-only registration/version/publication statements cannot run on SQLite.

- [x] **Step 3: Implement the SQLite notebook writer**

Add dialect-local operations with bound parameters and idempotent export identity. Route only local contexts through them; SQL Server keeps the existing publisher:

```python
pricing.require_write("publish_candidate")
if pricing.mode == "local":
    return publish_sqlite_candidate(
        pricing.engine,
        settings=pricing.settings,
        model=model,
        completed_build=candidate.completed_build,
        created_by=identity,
    )
return publish_completed_model_build(
    pricing.engine,
    settings=pricing.settings,
    model_config=model.config,
    dataset=None,
    completed_build=candidate.completed_build,
    created_by=identity,
)
```

The local publisher stages the generated workbook with the existing staging parser, stores an immutable package and run, writes dataset/split/metric links, and returns generated IDs. It does not deploy; `deploy_package` gives an explicit local-mode error. Call `pricing.require_write(operation_name)` first in every mutating public helper. Remote/runtime contexts keep using the existing SQL Server implementation.

Use a file-backed coordinator in rollback-journal mode so transactions spanning
the attached schema files remain atomic. Serialize local publication with a
model-local file lock, reserve trained model versions transactionally, verify
candidate bundle bytes before staging, and compare the full stored model-run
evidence on idempotent retries.

- [x] **Step 4: Run mutation tests and verify GREEN**

Run: `pytest tests/test_notebook_database_targets.py tests/test_notebook_workflow.py -q`

Expected: all selected tests pass.

### Task 3: Generate and demonstrate the real notebook

**Files:**
- Modify: `scripts/scaffold_pricing_model.py`
- Modify: `pricing_models/mtpl_frequency/pricing_model.ipynb`
- Modify: `tests/test_scaffold_pricing_model.py`
- Modify: `tests/test_pricing_model_notebooks.py`

- [x] **Step 1: Write failing notebook/scaffold tests**

Assert a custom scaffold includes `pricing_model.ipynb`. Parse it as JSON, compile every code cell, require empty outputs, and assert the early global cell contains exactly the visible database controls:

```python
DATABASE_MODE = "local"
EXPECTED_REMOTE_DATABASE = ""
ALLOW_REMOTE_WRITES = False
```

Assert its connection cell passes `mode`, `local_root`, `expected_remote_database`, and `allow_remote_writes`, and displays `pricing.destination`. Add the same assertions for the MTPL notebook.

- [x] **Step 2: Run notebook tests and verify RED**

Run: `pytest tests/test_scaffold_pricing_model.py tests/test_pricing_model_notebooks.py -q`

Expected: failures because scaffolding does not create a notebook and MTPL uses implicit Docker runtime defaults.

- [x] **Step 3: Implement notebook JSON generation and update MTPL**

Generate notebook JSON with `json.dumps` from small markdown/code-cell helpers. The notebook has imports, a top global settings cell, visible destination, one plainly marked analyst-owned data cell, `PricingModelSpec`, ordinary Python transforms/model factory, build/publish, and disabled editor/deploy cells. It contains no executed outputs, work imports, credentials, hostnames, or database names. Update MTPL to use the explicit controls and keep its existing model workflow.

- [x] **Step 4: Run notebook tests and verify GREEN**

Run: `pytest tests/test_scaffold_pricing_model.py tests/test_pricing_model_notebooks.py -q`

Expected: all selected tests pass.

### Task 4: Documentation and complete verification

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`

- [x] **Step 1: Document the two-mode workflow**

Document local SQLite as the safe default, `PRICING_RUNTIME_MODULE` as the private remote integration point, the exact database confirmation/write switch, and that DDL administration and real deployment remain separate. Ignore model-local `.local/` directories and `*.sqlite` files.

- [x] **Step 2: Run focused static checks**

Run: `python -m compileall pricing_pipeline scripts -q`

Run: `ruff check pricing_pipeline scripts tests`

Expected: both commands exit successfully.

- [x] **Step 3: Run the complete suite**

Run: `pytest -q`

Expected: all tests pass.

- [x] **Step 4: Check repository safety**

Run: `git diff --check`

Run: `git status --short`

Expected: no SQLite files or private screenshot files are staged; `builder_screenshot/` remains untracked.
