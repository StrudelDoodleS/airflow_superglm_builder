# Shared Scaffold Helper Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move generic scaffold publish/version/run-metadata boilerplate into focused shared modules while keeping custom model code visibly model-owned.

**Architecture:** Add three small shared modules: completed-build payload/date helpers, model-version lookup helpers, and Airflow run metadata helpers. Update the generated custom scaffold and demo custom publish model to import those helpers while preserving the DAG shape and model-owned extension points.

**Tech Stack:** Python, SQLAlchemy `text`, pytest, ruff, Airflow TaskFlow wrappers.

---

### Task 1: Completed-Build Helper Module

**Files:**
- Create: `pricing_pipeline/orchestration/completed_build_helpers.py`
- Test: `tests/test_completed_build_helpers.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_completed_build_helpers.py` with tests for:
- date/datetime/ISO date normalization;
- blank, malformed, and numeric date rejection;
- required payload text;
- `mlflow_run_id` passthrough in `completed_model_build_payload(...)`.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
rtk uv run pytest -q tests/test_completed_build_helpers.py
```

Expected: import failure for `pricing_pipeline.orchestration.completed_build_helpers`.

- [ ] **Step 3: Implement minimal module**

Implement:

```python
effective_from_for_run(...)
required_payload_text(...)
completed_model_build_payload(...)
```

Use `CompletedModelBuild(...).to_dict()` as the final validation boundary.

- [ ] **Step 4: Verify tests pass**

Run:

```bash
rtk uv run pytest -q tests/test_completed_build_helpers.py
```

Expected: all tests pass.

### Task 2: Model-Version Helper Module

**Files:**
- Create: `pricing_pipeline/publishing/model_versions.py`
- Test: `tests/test_model_versions.py`

- [ ] **Step 1: Write failing tests**

Create tests proving:
- existing `model_key + source_export_id` returns stored version exactly;
- non-`vN` existing export versions are returned exactly;
- new export allocates next `vN`;
- non-`vN` historical versions are ignored for next `vN`;
- child/manual packages where `parent_rate_package_id IS NOT NULL` are ignored for next `vN`;
- configured pricing schema names are used through `schema_names_from_connectable(engine)`.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
rtk uv run pytest -q tests/test_model_versions.py
```

Expected: import failure for `pricing_pipeline.publishing.model_versions`.

- [ ] **Step 3: Implement minimal module**

Implement:

```python
existing_model_version_for_export(...)
next_trained_model_version(...)
resolve_model_version_for_export(...)
```

Use `schema_names_from_connectable(engine)` and SQLAlchemy `text(...)`.

- [ ] **Step 4: Verify tests pass**

Run:

```bash
rtk uv run pytest -q tests/test_model_versions.py
```

Expected: all tests pass.

### Task 3: Airflow Run Metadata Helper Module

**Files:**
- Create: `pricing_pipeline/orchestration/airflow_run_metadata.py`
- Test: `tests/test_airflow_run_metadata.py`

- [ ] **Step 1: Write failing tests**

Create tests proving:
- no module import of `airflow`;
- `run_id` drives `run_key`;
- logical date drives `effective_from` and `data_as_of_date`;
- retry of the same run produces stable metadata;
- missing logical date can fall back to manual/default behavior;
- `merge_prepared_payload_metadata(...)` rejects mismatched `run_key`;
- payload can override `output_dir`, `effective_from`, and `data_as_of_date`.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
rtk uv run pytest -q tests/test_airflow_run_metadata.py
```

Expected: import failure for `pricing_pipeline.orchestration.airflow_run_metadata`.

- [ ] **Step 3: Implement minimal module**

Implement:

```python
context_logical_date(...)
task_run_metadata(...)
merge_prepared_payload_metadata(...)
```

Keep Airflow imports out of the module.

- [ ] **Step 4: Verify tests pass**

Run:

```bash
rtk uv run pytest -q tests/test_airflow_run_metadata.py
```

Expected: all tests pass.

### Task 4: Update Scaffold Templates

**Files:**
- Modify: `scripts/scaffold_pricing_model.py`
- Modify tests: `tests/test_scaffold_pricing_model.py`

- [ ] **Step 1: Update scaffold tests first**

Change scaffold assertions so generated custom `modeling.py`:
- imports `completed_build_helpers`;
- imports `resolve_model_version_for_export`;
- still exposes `read_prepared_source`, `build_final_model_frame`, and `fit_validate_export_rating_tables`;
- no longer defines copied version/date/payload helpers.

Change scaffold assertions so generated `data.py` uses `output_dir`, and generated `airflow_tasks.py` imports `airflow_run_metadata`.

- [ ] **Step 2: Verify scaffold tests fail**

Run:

```bash
rtk uv run pytest -q tests/test_scaffold_pricing_model.py
```

Expected: failures showing old copied helper names and old `output_root` scaffold contract.

- [ ] **Step 3: Update scaffold templates**

Edit `_custom_data_template`, `_custom_modeling_template`, and `_custom_airflow_tasks_template` to use the shared modules and `output_dir` handoff.

- [ ] **Step 4: Verify scaffold tests pass**

Run:

```bash
rtk uv run pytest -q tests/test_scaffold_pricing_model.py
```

Expected: all scaffold tests pass.

### Task 5: Update Demo Custom Publish Code

**Files:**
- Modify: `pricing_models/demo_custom_publish/modeling.py`
- Modify: `pricing_models/demo_custom_publish/airflow_tasks.py`
- Modify: `pricing_models/demo_custom_publish/data.py`
- Modify: `scripts/run_demo_custom_publish.py`
- Modify tests: `tests/test_demo_custom_publish_example.py`, `tests/test_demo_custom_publish_runner.py`

- [ ] **Step 1: Update tests first**

Adjust demo tests so the demo imports shared helpers rather than defining copied date/version helpers. Move the effective-date behavior assertion to `tests/test_completed_build_helpers.py` and the version-resolution behavior assertions to `tests/test_model_versions.py`; keep demo tests focused on imports, orchestration order, and payload fields.

- [ ] **Step 2: Verify demo tests fail**

Run:

```bash
rtk uv run pytest -q tests/test_demo_custom_publish_example.py tests/test_demo_custom_publish_runner.py
```

Expected: failures because demo files still define/import copied helpers.

- [ ] **Step 3: Update demo code**

Make demo data prep accept `output_dir`, demo modeling import shared helpers, demo Airflow wrappers use `task_run_metadata(...)` and `merge_prepared_payload_metadata(...)`, and demo direct runner call `effective_from_for_run(...)` from `completed_build_helpers`.

- [ ] **Step 4: Verify demo tests pass**

Run:

```bash
rtk uv run pytest -q tests/test_demo_custom_publish_example.py tests/test_demo_custom_publish_runner.py
```

Expected: all demo tests pass.

### Task 6: Documentation and Verification

**Files:**
- Modify: `README.md` if scaffold wording still implies authors should edit copied helper functions.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
rtk uv run pytest -q tests/test_completed_build_helpers.py tests/test_model_versions.py tests/test_airflow_run_metadata.py tests/test_scaffold_pricing_model.py tests/test_demo_custom_publish_example.py tests/test_demo_custom_publish_runner.py
```

Expected: all targeted tests pass.

- [ ] **Step 2: Run full verification**

Run:

```bash
rtk uv run pytest -q
rtk uv run ruff check pricing_pipeline/orchestration/completed_build_helpers.py pricing_pipeline/publishing/model_versions.py pricing_pipeline/orchestration/airflow_run_metadata.py scripts/scaffold_pricing_model.py pricing_models/demo_custom_publish scripts/run_demo_custom_publish.py tests/test_completed_build_helpers.py tests/test_model_versions.py tests/test_airflow_run_metadata.py tests/test_scaffold_pricing_model.py tests/test_demo_custom_publish_example.py tests/test_demo_custom_publish_runner.py
rtk uv run ruff format --check pricing_pipeline/orchestration/completed_build_helpers.py pricing_pipeline/publishing/model_versions.py pricing_pipeline/orchestration/airflow_run_metadata.py scripts/scaffold_pricing_model.py pricing_models/demo_custom_publish scripts/run_demo_custom_publish.py tests/test_completed_build_helpers.py tests/test_model_versions.py tests/test_airflow_run_metadata.py tests/test_scaffold_pricing_model.py tests/test_demo_custom_publish_example.py tests/test_demo_custom_publish_runner.py
rtk uv run python scripts/no_docker_services.py menu --dry-run
rtk git diff --check
```

Expected: all commands exit 0. No SQL migration or re-seed is required.
