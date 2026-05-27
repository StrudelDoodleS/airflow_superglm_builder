# Rate Package Lifecycle API Design

## Purpose

Improve the production model-build and rate-package workflow so Airflow DAGs
have a simple, safe API for publishing trained model packages, reviewing them,
deploying them, and creating controlled manual package revisions.

The design targets serious production-like builds through Airflow. Notebooks and
standalone scripts can still demonstrate the lifecycle, but they are secondary
surfaces. The core requirements are safety, auditability, and a clean API that
does not hide the important model/package lifecycle concepts.

## Current Context

The repository already has the main pricing pipeline pieces:

- `ModelSpec` objects describe model-specific training behavior.
- Airflow DAGs call `build_pricing_model_dag(...)`.
- Training writes MLflow artifacts and a SuperGLM rating workbook.
- `stage_rating_export(...)` and `publish_rating_package(...)` expose Python
  wrappers, but they delegate to script modules.
- SQL Server stores `PRICING_MODEL`, `PRICING_RATE_PACKAGE`, package content,
  package pointers, and deployment history.
- Database triggers block direct edits to non-draft or deployed packages.

The gaps are:

- The Python publishing API is still script-shaped.
- `ensure_pricing_model(...)` silently updates existing model metadata.
- Production builds can create a new model row if the model key is missing.
- Successful packages default to `DRAFT`, which weakens immutability unless the
  package is deployed.
- Package version allocation uses plain `MAX(package_version) + 1`.
- The actual migration set does not enforce unique `(model_id, package_version)`.
- Build and deploy are currently coupled by passing the deployment slot into
  package publish.
- Manual rate changes do not have a safe first-class workflow.

## Goals

- Make Airflow DAG authors interact with one clear publishing object instead of
  many loose helper functions.
- Keep implementation internals small and testable.
- Add stable per-model config for SQL schema housekeeping metadata.
- Make model registry behavior strict during production builds.
- Publish successful production packages as immutable `PUBLISHED` packages.
- Separate candidate package creation from live deployment.
- Add a reusable deploy DAG that can deploy any configured model package.
- Add a controlled manual revision path based on loaded package frames.
- Preserve existing CLI/script entry points as thin compatibility shims.

## Non-Goals

- Build a UI for manual rate editing.
- Support fuzzy package selectors such as "latest -1" in v1.
- Support temporal `as_of` package selection in v1.
- Implement embedded Airflow human-in-the-loop approval tasks in v1.
- Put executable model construction or feature engineering into TOML.
- Store `model_id` as required config. SQL Server owns `model_id`.

## Public API Shape

Use a hybrid API:

- A small `ModelPublisher` class is the primary UX for DAG authors and
  production operators.
- Typed dataclasses define configs, selectors, snapshots, and results.
- Internal implementation remains function-based where that keeps testing and
  boundaries clear.

Example production usage:

```python
config = load_model_build_config("pricing_models/mtpl_frequency/model.toml")
publisher = ModelPublisher(engine, config)

publisher.validate_registered_model()
publish_result = publisher.publish_training_export(export_result)
```

Example manual revision usage:

```python
config = load_model_build_config("pricing_models/mtpl_frequency/model.toml")
publisher = ModelPublisher(engine, config)

snapshot = publisher.load_rate_package(
    RatePackageSelector(rate_package_id=123)
)
edited_cells = snapshot.rate_cells.copy()

revision = publisher.create_manual_revision(
    parent=snapshot,
    edited_rate_cells=edited_cells,
    reason="Temporary relativity adjustment after portfolio review",
    created_by="mhick",
)
```

The class is not a large domain manager. It owns the engine and model config,
then delegates to focused functions for validation, SQL loading, package
creation, manual revision diffing, and deployment.

## Model Config

Each production model gets a stable config file next to the Python model spec:

```text
pricing_models/mtpl_frequency/
  model.toml
  spec.py
  training.py
```

Initial TOML shape:

```toml
model_key = "MTPL_FREQ"
model_label = "Motor frequency"
target_name = "ClaimNb"
model_type = "superglm_poisson"
deployment_slot = "MTPL_FREQ_UAT"
default_package_status = "PUBLISHED"
```

Config fields are stable housekeeping metadata. They are not per-run knobs.
Training SQL, feature engineering, and model construction remain in Python.

`target_name` remains explicit in v1 because the current SQL registry requires
it. If the schema later stops needing it, config can be simplified.

`model_id` is not required in config. An optional `expected_model_id` can be
added later as an extra assertion, but the primary stable identifier is
`model_key`.

## Model Registry Policy

Production builds must validate an existing registry row. They must not silently
create or update model metadata.

New behavior:

- `register_model(config)` explicitly creates the first DB model row.
- `ModelPublisher.validate_registered_model()` checks that `model_key` exists.
- Missing model rows fail with a clear bootstrap instruction.
- Mismatched `target_name`, `model_type`, label, or status fail.
- The build/publish path never updates an existing model row as a side effect.

This prevents a typo such as `MTPL_FRQ` from creating and publishing under a new
model family.

## Package Status Semantics

Use `package_status` as a real lifecycle flag:

- `DRAFT`: package is being assembled or reviewed. Direct edits are allowed by
  database triggers.
- `PUBLISHED`: package content is frozen. It can be reviewed, deployed, or used
  for audit. Direct edits are blocked by database triggers.

Production model builds finalize packages as `PUBLISHED` by default through
`default_package_status = "PUBLISHED"`.

Manual revisions may be internally created as `DRAFT` while rows are inserted,
but the public API should return a finalized `PUBLISHED` revision unless
explicitly extended later.

## Trained Package Publish Flow

Production Airflow model build DAGs create candidate packages. They do not move
live deployment pointers by default.

Flow:

1. DAG loads `model.toml` and constructs the Python `ModelSpec`.
2. `ModelPublisher.validate_registered_model()` validates SQL registry state.
3. Training runs, logs MLflow artifacts, and writes the rating workbook.
4. Publish stages the workbook for the configured `model_key`.
5. Publisher creates a new package for the model:
   - atomically assigns the next `package_version`,
   - inserts package terms, cells, cell mappings, and compiled tables,
   - finalizes the package as `PUBLISHED`.
6. Publish returns a `PublishResult` with model, run, package, version, and
   artifact identifiers.

For v1, trained publishes do not set `parent_rate_package_id`. Retrains are
treated as independent trained candidates. Manual revisions use
`parent_rate_package_id`.

## Reusable Deploy DAG

Add one generic manually triggered DAG, for example
`pricing_deploy_rate_package`. It deploys a candidate package for any configured
model.

Expected Airflow params:

```text
model_key = "MTPL_FREQ"
rate_package_id = 123
deployment_slot = "MTPL_FREQ_UAT"
deployment_reason = "Reviewed diagnostics and SQL prediction validation"
deployed_by = "mhick"
```

Rules:

- `rate_package_id` is the preferred v1 target selector.
- `model_key + package_version` is also supported.
- `deployment_slot` defaults from `model.toml` if omitted.
- Deploy accepts only `PUBLISHED` packages.
- Deploy validates the package belongs to the configured model.
- Deploy fails if the package is already current in the target slot.
- Deploy writes `deployment_note` or equivalent reason metadata.
- Deploy closes the previous current deployment row.
- Deploy inserts a new deployment history row.
- Deploy updates the legacy/current pointer table if that table remains part of
  the serving/query compatibility layer.

This deploy DAG is also the deployment path for manual revisions.

Airflow 3.2 has human-in-the-loop operators, so an embedded approval task can be
added later. V1 keeps deploy as a separate reusable DAG because it is simpler,
auditable, and common to trained and manual package candidates.

## Manual Rate Revision Flow

Manual rate changes are controlled package revisions. They are not SQL updates
against existing published package rows.

V1 supports exact selectors only:

- `rate_package_id`
- `model_key + package_version`

Manual revision flow:

1. Load a package snapshot with metadata and typed DataFrames.
2. User edits a constrained rate-cell or band DataFrame.
3. API computes a diff from the original snapshot.
4. API validates the diff.
5. API creates a new package with the next `package_version`.
6. New package sets `parent_rate_package_id` to the source package.
7. New package finalizes as `PUBLISHED`.
8. Deployment remains separate through the reusable deploy DAG.

Allowed v1 edits:

- `multiplier` on editable rate rows or compiled band rows.
- Base rate through a dedicated metadata field or helper, not by arbitrary
  column mutation.

Validation:

- Empty diffs fail.
- Multipliers must be positive finite numbers.
- Editing forbidden identity columns fails.
- `reason` and `created_by` are required.
- Source package must belong to the publisher's configured model.
- Source package must be `PUBLISHED`.

The revision result includes:

- new `rate_package_id`
- new `package_version`
- `parent_rate_package_id`
- changed row count
- changed base rate flag
- a concise diff summary

## Prediction Comparison Helper

Add an advisory helper for manual revisions:

```python
comparison = publisher.compare_predictions(
    before=snapshot,
    edited_rate_cells=edited_cells,
    sample=validation_frame,
)
```

The helper compares old and edited package predictions over a supplied sample
frame. It reports summary statistics and top changed rows. It does not block
revision creation in v1. Threshold-based blocking can be added later.

## Database Changes

Add schema support for stronger safety:

- Unique index or constraint on `(model_id, package_version)` for
  `pricing.PRICING_RATE_PACKAGE`.
- Atomic package version allocation using transaction locking rather than plain
  `MAX(package_version) + 1`.
- Use the existing `deployment_note` column for deploy reason.
- Keep existing immutability triggers and ensure they cover any new revision
  write path.

Package content insert behavior should stay append-only for published packages.
Updates to existing published package rows remain disallowed.

## Script Compatibility

Move package lifecycle logic into package modules under
`pricing_pipeline.publishing`. Keep existing scripts as thin CLI shims:

- `scripts/load_superglm_excel_to_staging.py`
- `scripts/load_staging_to_rating_package.py`

This preserves current local/tutorial commands while making production code
import normal package modules rather than dynamically importing scripts.

## Error Handling

Errors should name the failed lifecycle invariant:

- missing registered model
- registry metadata mismatch
- unknown package selector
- package belongs to another model
- package is not `PUBLISHED`
- package already deployed to requested slot
- empty manual revision diff
- forbidden column edited
- invalid multiplier
- missing reason or actor

The deploy DAG should emit a final summary with old package, new package,
deployment slot, package version, actor, and reason.

## Testing Strategy

Unit tests:

- TOML config loading and validation.
- `ModelPublisher` construction and registry validation.
- Missing model and metadata mismatch failures.
- Explicit registration path.
- Publish finalizes packages as `PUBLISHED`.
- Deploy selector resolution and validation.
- Manual revision diff creation.
- Empty diff failure.
- Invalid multiplier failure.
- Forbidden edit failure.
- Base-rate edit diff summary.

Migration/schema tests:

- Unique `(model_id, package_version)` index exists.
- Version allocation SQL uses locking.
- Deployment reason is persisted.
- Immutability triggers still cover package content tables.

Airflow/DAG tests:

- Build DAG creates a package candidate and does not deploy by default.
- Deploy DAG validates params and calls the deploy API.
- Deploy DAG defaults slot from config when omitted.

Integration tests where SQL Server is available:

- Two sequential publishes produce package versions 1 and 2.
- Concurrent version allocation cannot create duplicate versions.
- Published package content cannot be edited directly.
- Manual revision creates a child package and leaves parent unchanged.

## Rollout Plan

1. Add model config and registry validation without changing existing DAG
   behavior.
2. Move script internals into package modules while keeping scripts working.
3. Add `ModelPublisher` and update DAG publish code to use it.
4. Change default package finalization to `PUBLISHED`.
5. Add package-version uniqueness and atomic allocation.
6. Split build and deploy by adding the reusable deploy DAG.
7. Add manual package snapshot/revision support.
8. Add prediction comparison helper.

The implementation should preserve current tutorial/script affordances while
making the Airflow production path stricter and clearer.
