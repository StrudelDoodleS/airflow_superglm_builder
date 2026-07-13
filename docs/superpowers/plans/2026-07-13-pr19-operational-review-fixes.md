# PR 19 Operational Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six operational compatibility findings raised against PR 19 without adding analyst-facing configuration or weakening publication invariants.

**Architecture:** Extend existing compatibility seams: local additive DDL upgrades, notebook input provenance, attempt-relative artifact paths, a forward SQL migration, publisher-owned identity canonicalization, and transactional root-version reservation. Each behavior receives a focused regression test before production code changes.

**Tech Stack:** Python 3.12, pytest, SQLAlchemy, SQLite, SQL Server T-SQL migrations, Ruff.

---

### Task 1: Upgrade Existing Local Model-Run Tables

**Files:**
- Modify: `pricing_pipeline/infra/offline_sqlite.py`
- Test: `tests/test_fremtpl.py`

- [x] Drop the seven candidate-artifact columns from a temporary existing SQLite store, reopen it, and assert all columns are restored.
- [x] Run `rtk proxy .venv/bin/pytest tests/test_fremtpl.py::test_open_offline_sqlite_adds_model_run_candidate_columns_to_existing_store -q` and verify it fails before the upgrade list changes.
- [x] Add the path, digest, format, size, Python version, SuperGLM version, and model-source digest columns to `_OFFLINE_COLUMN_UPGRADES` with their SQLite types.
- [x] Rerun the focused test and verify it passes.

### Task 2: Keep Caller-Supplied Offset Semantics Explicit

**Files:**
- Modify: `pricing_pipeline/notebook.py`
- Test: `tests/test_notebook_workflow.py`

- [x] Add a spec-based notebook test that supplies a custom offset without metadata and expects a `ValueError` requiring both `offset_contract` and `offset_export_options`.
- [x] Run the new test and verify the current helper incorrectly reaches the build runner.
- [x] Record whether the helper derived the offset. Infer the exposure contract/options only in that case; reject incomplete metadata for a caller-supplied offset.
- [x] Rerun the offset tests and verify both derived and explicit paths pass.

### Task 3: Preserve Nested Editor Review Artifacts and Trigger Identity

**Files:**
- Modify: `pricing_pipeline/publishing/editor_candidate.py`
- Test: `tests/test_editor_candidate_publisher.py`

- [x] Change the editor-export test hook to return `reports/rating_tables_review.xlsx` and assert the bundle records the corresponding nested final path.
- [x] Extend the publication test to assert revision metadata contains the Airflow identity in both `claimed_identity` and `published_by`.
- [x] Run both tests and verify the path and identity assertions fail.
- [x] Rebase the validated review path from the staging directory to the final directory and canonicalize both identity fields in `_revision_with_publisher_identity()`.
- [x] Rerun the editor tests and verify they pass.

### Task 4: Upgrade Current-Package SQL Scoring

**Files:**
- Create: `db/migrations/V029__current_rate_package_scoring.sql`
- Test: `tests/test_migrations.py`

- [x] Add a migration contract test requiring a forward `CREATE OR ALTER PROCEDURE pricing.PREDICT_CURRENT_RATE` migration that resolves the current package and delegates to `PREDICT_RATE_PACKAGE` with every scoring argument.
- [x] Run the test and verify it fails because V029 is absent.
- [x] Add V029 using the current model/deployment lookup and delegate matching plus optional breakdown output to the authoritative package scorer.
- [x] Rerun migration tests and verify they pass.

### Task 5: Reserve Versions for Direct Root Publication

**Files:**
- Modify: `pricing_pipeline/publishing/package_writer.py`
- Test: `tests/test_package_writer.py`
- Test: `tests/test_demo_model_variants.py`

- [x] Replace the missing-reservation rejection test with a test asserting an exact reservation insert occurs before the root package insert.
- [x] Run it and verify the current guard raises.
- [x] Insert the staged root version into `PRICING_MODEL_VERSION_RESERVATION` when absent, inside the existing locked transaction; keep exact comparison for an existing reservation.
- [x] Rerun package-writer and demo-seed tests and verify direct paths publish while conflicts still fail.

### Task 6: Verify and Publish the Review Fixes

**Files:**
- Modify: the files listed above only.

- [x] Run `rtk proxy .venv/bin/pytest tests/ -q` and require exit 0.
- [x] Run `rtk ruff check pricing_pipeline scripts dags tests` and require no diagnostics.
- [x] Run `rtk proxy .venv/bin/python -m compileall -q pricing_pipeline scripts dags pricing_models` and require exit 0.
- [x] Run `rtk git diff --check` and require exit 0.
- [ ] Commit the exact reviewed files, push PR 19, reply in each thread, resolve all six threads, and post `@codex review` with the verification summary.
