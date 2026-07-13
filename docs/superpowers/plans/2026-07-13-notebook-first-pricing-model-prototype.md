# Notebook-First Pricing Model Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an MTPL pricing-model notebook under `pricing_models/` that trains, audits, publishes, edits, and optionally deploys through direct Python calls without loading TOML or invoking Airflow.

**Architecture:** Add one thin `pricing_pipeline.notebook` facade over the existing runtime, model registry, standard SuperGLM builder, completed-build publisher, editor publisher, and deployment primitives. The notebook contains modelling decisions and SQL-reading code; the facade derives generated identifiers and audit metadata. Keep schema migration application outside the pricing notebook, while repairing the V019 upgrade defect that currently prevents an existing database from reaching the required schema.

**Tech Stack:** Python 3.14, Jupyter notebook JSON, pandas, SuperGLM, SQLAlchemy, SQL Server, pytest.

---

### Task 1: Notebook-facing registration, build, and publication API

**Files:**
- Create: `pricing_pipeline/notebook.py`
- Create: `tests/test_notebook_workflow.py`
- Modify: `pricing_pipeline/orchestration/publish_completed_build.py`

- [ ] **Step 1: Write failing tests for the wished-for API**

Test that `connect()` wraps the configured runtime, `register_model()` inserts or validates a stable SQL model and returns its generated `model_id`, `build_candidate()` derives row identity, folds, export ID, model version, artifact roots, and manifest input before delegating to `run_standard_superglm_build()`, and `publish_candidate()` returns the model-run and rate-package identifiers from SQL publication.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `rtk pytest tests/test_notebook_workflow.py -q`

Expected: collection fails because `pricing_pipeline.notebook` does not exist.

- [ ] **Step 3: Implement the minimal facade**

Expose these direct Python functions and small return records:

```python
pricing = connect(runtime_module=None)
model = register_model(
    pricing,
    name="MTPL_FREQUENCY",
    label="MTPL claim frequency",
    target="ClaimNb",
    model_type="Poisson",
    deployment_slot="PRODUCTION",
    validation_split=ValidationSplitConfig.kfold(n_splits=5),
    source_root=Path("pricing_models/mtpl_frequency"),
)
candidate = build_candidate(
    pricing,
    model=model,
    frame=frame,
    X=X,
    y=y,
    model_factory=build_model,
    scoring=("deviance",),
    dataset_name="freMTPL2freq_model_frame",
    source_system="freMTPL_raw_sql",
    data_as_of="2026-06-30",
    pk_columns=("IDpol",),
    effective_from="2026-08-01",
    sample_weight=exposure,
    weight_column="Exposure",
    offset=offset,
    offset_contract=offset_contract,
    offset_export_options=offset_export_options,
)
published = publish_candidate(pricing, candidate)
```

Generated usernames, run keys, export IDs, model versions, manifests, split sets, hashes, artifact paths, `model_run_id`, `rate_package_id`, and `package_version` remain library-owned.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `rtk pytest tests/test_notebook_workflow.py tests/test_publish_completed_build.py -q`

Expected: all focused tests pass.

### Task 2: Synchronous notebook editing and deployment

**Files:**
- Modify: `pricing_pipeline/notebook.py`
- Modify: `pricing_pipeline/publishing/editor_candidate.py`
- Modify: `tests/test_notebook_workflow.py`
- Modify: `tests/test_editor_candidate_publisher.py`

- [ ] **Step 1: Write failing direct-workflow tests**

Test that `open_candidate()` loads the just-published candidate with an in-memory Python config, `publish_edits()` saves the retained editor session and calls the existing editor publisher synchronously without an Airflow client, and `deploy_package()` reads the current champion immediately before invoking the stale-safe deployment primitive.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `rtk pytest tests/test_notebook_workflow.py tests/test_editor_candidate_publisher.py -q`

Expected: failures for the missing notebook functions and missing explicit model-config injection in editor publication.

- [ ] **Step 3: Implement direct editor publication**

Allow `publish_editor_submission()` and `load_parent_candidate()` to accept an explicit `ModelBuildConfig`; retain registry discovery as the compatibility default for existing DAG callers. The notebook facade supplies its registered Python config and uses a local submission recorder rather than an Airflow REST client.

- [ ] **Step 4: Implement direct deployment**

Resolve the current package in the configured deployment slot immediately before calling `deploy_rate_package()`. Require a non-empty business reason and use the current OS identity unless an explicit identity is supplied.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run: `rtk pytest tests/test_notebook_workflow.py tests/test_candidate_editor.py tests/test_editor_candidate_publisher.py tests/test_deployment.py -q`

Expected: all focused tests pass.

### Task 3: MTPL notebook beside the pricing model

**Files:**
- Create: `pricing_models/mtpl_frequency/pricing_model.ipynb`
- Create: `tests/test_pricing_model_notebooks.py`
- Modify: `pricing_pipeline/modeling/standard_superglm.py`
- Modify: `tests/test_standard_superglm.py`

- [ ] **Step 1: Write the notebook contract test**

Parse the notebook as JSON and assert it calls the notebook facade, reads the source frame from SQL, defines feature transforms/model/CV in Python, publishes audit lineage, and includes optional edit/deploy cells. Assert it does not mention `model.toml`, `MODEL_CONFIG`, Airflow, or analyst-entered SQL IDs.

- [ ] **Step 2: Run the contract test and verify RED**

Run: `rtk pytest tests/test_pricing_model_notebooks.py -q`

Expected: failure because the notebook does not exist.

- [ ] **Step 3: Add the executable notebook**

Create concise cells for connection/model registration, SQL reading, final-frame transforms, SuperGLM factory, validation configuration, audited build, SQL publication, optional editor child publication, and optional stale-safe deployment. Leave only actual environment/model/date/reason decisions for the analyst to edit.

- [ ] **Step 4: Include notebooks in model-source hashing**

Extend `hash_model_source()` to include `.ipynb` alongside `.py`, `.sql`, and `.toml`, with a regression test proving a notebook-only source directory is accepted and changes alter the digest.

- [ ] **Step 5: Run notebook and source-hash tests**

Run: `rtk pytest tests/test_pricing_model_notebooks.py tests/test_standard_superglm.py -q`

Expected: all focused tests pass.

### Task 4: Repair the existing-database V019 migration path

**Files:**
- Modify: `db/migrations/V019__terminate_throw_guard_errors.sql`
- Modify: `tests/test_migrations.py`

- [ ] **Step 1: Add a regression assertion for safe THROW rewrites and error visibility**

Assert V019 starts with `SET NOCOUNT ON`, does not rewrite a conditional `THROW` to the invalid `;THROW` form, and wraps replacement throws in a valid statement block.

- [ ] **Step 2: Run migration tests and verify RED**

Run: `rtk pytest tests/test_migrations.py -q`

Expected: the new V019 assertions fail.

- [ ] **Step 3: Correct V019**

Suppress intermediate row-count result sets so pyodbc surfaces dynamic-SQL errors, and rewrite old guard statements to `BEGIN; THROW ...; END;` blocks that remain valid both after a single-statement `IF` and inside an existing `BEGIN` block.

- [ ] **Step 4: Verify the real SQL Server upgrade**

Run the schema application against the existing local SQL Server database that currently has V001-V018 in the two-column migration recorder. Confirm V019-V025 apply, the recorder gains checksum/status columns, and a second invocation skips every migration cleanly.

- [ ] **Step 5: Run full verification**

Run: `rtk pytest tests/ -q`

Run: `rtk ruff check .`

Run: `rtk git diff --check`

Expected: all tests and checks pass.
