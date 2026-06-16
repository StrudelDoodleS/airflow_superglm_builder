# Offline SQLite DDL Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the freMTPL offline smoke runner create and populate an explicit offline SQLite DDL structure with `pricing`, `pricing_stg`, and `mlops` database files.

**Architecture:** Add small SQLite-specific DDL files rather than trying to auto-convert SQL Server migrations. The runner attaches three SQLite files as `pricing`, `pricing_stg`, and `mlops`, applies the offline DDL, then writes source/manifest/package rows to `pricing` and metrics to `mlops`.

**Tech Stack:** Python, SQLAlchemy, SQLite attached databases, pandas, pytest.

---

### Task 1: Lock Down Three-Database Offline Output

**Files:**
- Modify: `tests/test_mtpl_offline_sqlite_runner.py`
- Modify: `scripts/run_mtpl_frequency_offline_sqlite.py`

- [ ] **Step 1: Write the failing test**

Update the offline runner test to expect:

```python
assert Path(result["db_paths"]["pricing"]).exists()
assert Path(result["db_paths"]["pricing_stg"]).exists()
assert Path(result["db_paths"]["mlops"]).exists()
assert result["tables"]["pricing"]["FREMTPL_RAW"] == 40
assert result["tables"]["pricing"]["DATASET_MANIFEST"] == 1
assert result["tables"]["pricing"]["CV_FOLD"] == 5
assert result["tables"]["pricing"]["PRICING_RATE_PACKAGE"] == 1
assert result["tables"]["pricing_stg"]["STG_RATING_EXPORT"] == 0
assert result["tables"]["pricing_stg"]["STG_RATE_CELL"] == 0
assert result["tables"]["pricing_stg"]["STG_CELL_LEVEL"] == 0
assert result["tables"]["mlops"]["MODEL_RUN_METRIC"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
rtk uv run pytest -q tests/test_mtpl_offline_sqlite_runner.py
```

Expected: fail because the current runner returns one `db_path`, one flat table-count mapping, and writes `MODEL_RUN_METRIC` into `pricing`.

- [ ] **Step 3: Implement minimal three-DB attachment**

Update the runner to return:

```python
"db_paths": {
    "pricing": str(pricing_db_path),
    "pricing_stg": str(pricing_stg_db_path),
    "mlops": str(mlops_db_path),
}
```

Attach each file with SQLite:

```python
ATTACH DATABASE '<pricing.sqlite>' AS pricing
ATTACH DATABASE '<pricing_stg.sqlite>' AS pricing_stg
ATTACH DATABASE '<mlops.sqlite>' AS mlops
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
rtk uv run pytest -q tests/test_mtpl_offline_sqlite_runner.py
```

Expected: pass.

### Task 2: Add Offline SQLite DDL Files

**Files:**
- Create: `db/offline_sqlite/pricing.sql`
- Create: `db/offline_sqlite/pricing_stg.sql`
- Create: `db/offline_sqlite/mlops.sql`
- Modify: `scripts/run_mtpl_frequency_offline_sqlite.py`

- [ ] **Step 1: Write failing assertions**

Assert the files exist and that the runner applies them before writing rows.

- [ ] **Step 2: Run focused test red**

Run:

```bash
rtk uv run pytest -q tests/test_mtpl_offline_sqlite_runner.py
```

Expected: fail until the DDL files exist and are applied.

- [ ] **Step 3: Add DDL files**

Create explicit SQLite DDL for the offline smoke path:

- `pricing.sql`: `FREMTPL_RAW`, manifest/CV tables, `PRICING_MODEL`, `MODEL_RUN`, `PRICING_RATE_PACKAGE`.
- `pricing_stg.sql`: `STG_RATING_EXPORT`, `STG_RATE_CELL`, `STG_CELL_LEVEL`.
- `mlops.sql`: `MODEL_RUN_METRIC`.

- [ ] **Step 4: Apply DDL in the runner**

Execute each DDL script with `sqlite3.Connection.executescript(...)` after attachment and before seeding rows.

- [ ] **Step 5: Run focused test green**

Run:

```bash
rtk uv run pytest -q tests/test_mtpl_offline_sqlite_runner.py
```

Expected: pass.

### Task 3: Documentation and Verification

**Files:**
- Modify: `README.md`
- Modify: `tests/test_no_docker_runtime.py`

- [ ] **Step 1: Update README**

Document that offline smoke creates:

```text
state/offline/mtpl_frequency/pricing.sqlite
state/offline/mtpl_frequency/pricing_stg.sqlite
state/offline/mtpl_frequency/mlops.sqlite
```

- [ ] **Step 2: Run verification**

Run:

```bash
rtk uv run pytest -q
rtk uv run ruff check scripts/run_mtpl_frequency_offline_sqlite.py tests/test_mtpl_offline_sqlite_runner.py tests/test_no_docker_runtime.py
rtk uv run ruff format --check scripts/run_mtpl_frequency_offline_sqlite.py tests/test_mtpl_offline_sqlite_runner.py tests/test_no_docker_runtime.py
rtk git diff --check
```

- [ ] **Step 3: Run actual offline build**

Run:

```bash
rtk uv run python scripts/run_mtpl_frequency_offline_sqlite.py --reset --row-count 120 --effective-from 2026-06-05
```

Expected: all three SQLite files exist and the result JSON shows populated `pricing` and `mlops` tables plus empty `pricing_stg` staging tables.
