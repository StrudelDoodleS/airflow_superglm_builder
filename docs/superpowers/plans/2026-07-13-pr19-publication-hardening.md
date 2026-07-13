# PR 19 Publication Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all four unresolved PR 19 publication-concurrency and retry-integrity findings without changing the analyst notebook interface.

**Architecture:** SQL Server owns remote model-version and staging serialization through transactional rows/locks; a focused package precondition binds staged evidence to lineage. Existing publications are reusable only after a complete SQL evidence comparison, while a small standard-library sentinel-file lock provides the equivalent trusted-host serialization on Windows and POSIX.

**Tech Stack:** Python 3.12, SQLAlchemy, SQL Server T-SQL, SQLite, pytest, Ruff

---

### Task 1: Cross-platform local publication lock

**Files:**
- Create: `pricing_pipeline/infra/file_lock.py`
- Modify: `pricing_pipeline/infra/offline_sqlite.py`
- Modify: `pricing_pipeline/publishing/editor_candidate.py`
- Modify: `pricing_pipeline/workbench/submission.py`
- Test: `tests/test_file_lock.py`

- [ ] **Step 1: Write the failing platform tests**

Create tests that patch the platform selector and a fake `msvcrt.locking`, assert one
lock and one unlock call on byte zero, and import the three consumers while blocking any
top-level `fcntl` import.

- [ ] **Step 2: Run the focused test and confirm RED**

Run `rtk proxy uv run pytest tests/test_file_lock.py -q`. Expect failure because the
shared helper does not exist and consumers import `fcntl` directly.

- [ ] **Step 3: Implement the sentinel-file helper and replace all consumers**

Implement `exclusive_file_lock(path)` with a securely opened sentinel, lazy POSIX or
Windows backend acquisition, guaranteed unlock/close, and yielding the binary handle.
Make the submission lock use `<submissions-root>/.submission.lock`.

- [ ] **Step 4: Confirm GREEN**

Run `rtk proxy uv run pytest tests/test_file_lock.py tests/test_offline_sqlite.py tests/test_editor_candidate_publisher.py tests/test_candidate_workbench.py -q` and expect all tests to pass.

### Task 2: Transactional remote model-version reservations

**Files:**
- Create: `db/migrations/V027__model_version_reservations.sql`
- Modify: `pricing_pipeline/publishing/model_versions.py`
- Modify: `tests/test_model_versions.py`
- Modify: `tests/test_migrations.py`

- [ ] **Step 1: Write failing reservation and migration tests**

Use a stateful fake SQL connection to resolve two distinct export IDs before either is
published and require `v1` then `v2`; require the same export to reuse `v1`. Assert the
migration defines model/export primary-key semantics, model/version uniqueness, model
foreign key, and a deduplicated historical-package backfill.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run `rtk proxy uv run pytest tests/test_model_versions.py tests/test_migrations.py -q`.
Expect the second export to receive the same version and the V027 assertions to fail.

- [ ] **Step 3: Add V027 and make resolve atomic**

Within one `engine.begin()`, lock the registered model row with
`WITH (UPDLOCK, HOLDLOCK)`, check package and reservation identity, union root-package
and reservation versions, choose the next numeric `vN`, and insert the reservation.
Return the stored version for deterministic retries.

- [ ] **Step 4: Confirm GREEN**

Re-run `rtk proxy uv run pytest tests/test_model_versions.py tests/test_migrations.py -q`
and expect all tests to pass.

### Task 3: Bind package lineage to locked staging evidence

**Files:**
- Create: `pricing_pipeline/publishing/staging_lock.py`
- Modify: `pricing_pipeline/publishing/staging.py`
- Modify: `pricing_pipeline/publishing/package_writer.py`
- Modify: `pricing_pipeline/publishing/publisher.py`
- Modify: `pricing_pipeline/publishing/editor_candidate.py`
- Test: `tests/test_model_publisher.py`
- Test: `tests/test_package_writer.py`
- Test: `tests/test_rating_export.py`

- [ ] **Step 1: Write failing lock and mismatch tests**

Require staging replacement and package loading to acquire the same export-scoped lock.
Change the staging header between those operations and assert publication raises before
the lineage callback. Assert both scheduled and editor callers pass their expected
staging metadata.

- [ ] **Step 2: Run focused tests and confirm RED**

Run `rtk proxy uv run pytest tests/test_model_publisher.py tests/test_package_writer.py tests/test_rating_export.py -k 'staging or lineage or expected' -q`. Expect missing lock/precondition failures.

- [ ] **Step 3: Implement the SQL Server transaction lock and precondition**

Use transaction-owned `sys.sp_getapplock` for
`pricing_staging_export:<export_id>`. Under the package lock compare expected export,
model, version, date, resolved source path, and receipt hash with the staging header;
raise a field-specific integrity error before existing/new package handling.

- [ ] **Step 4: Confirm GREEN**

Run the complete three focused files and expect all tests to pass.

### Task 4: Validate complete retry evidence

**Files:**
- Modify: `pricing_pipeline/orchestration/pipeline.py`
- Modify: `tests/test_rating_export.py`

- [ ] **Step 1: Write failing conflict tests**

Build a complete fake existing publication and parameterize conflicts in package dates
and source, run identity/paths/hashes/runtime metadata, dataset/split links, metrics, and
fold metrics. Require every conflict to raise `PublishedRunIntegrityError`; require one
exact match to return canonical persisted values.

- [ ] **Step 2: Run the resolver tests and confirm RED**

Run `rtk proxy uv run pytest tests/test_rating_export.py -k 'existing_published_run' -q`.
Expect the conflict cases to return `was_existing=True` incorrectly.

- [ ] **Step 3: Query and compare all immutable evidence**

Extend the resolver row query, read linked evidence tables on the same connection,
normalize dates/paths/scopes/floats, compare exact sets/maps, report mismatches, and only
then verify/load the committed candidate artifact.

- [ ] **Step 4: Confirm GREEN**

Run `rtk proxy uv run pytest tests/test_rating_export.py -q` and expect all tests to pass.

### Task 5: Verify and update PR 19

**Files:** all intentional source, migration, test, design, and plan files above.

- [ ] **Step 1: Run repository verification**

Run `rtk ruff check pricing_pipeline scripts tests`,
`rtk proxy python -m compileall pricing_pipeline scripts -q`,
`rtk git diff --check`, and `rtk proxy uv run pytest -q`. Expect zero errors/failures.

- [ ] **Step 2: Commit and push only intentional files**

Explicitly stage the named files, excluding the private untracked `builder_screenshot/`,
commit as `fix: harden remote notebook publication`, and push
`feature/scaffolded-candidate-workbench`.

- [ ] **Step 3: Close the review loop**

Reply to each of the four threads with its concrete fix and test evidence, resolve all
four threads, verify the unresolved-thread count is zero, and post `@codex review` on
PR 19 for a fresh review of the pushed commit.
