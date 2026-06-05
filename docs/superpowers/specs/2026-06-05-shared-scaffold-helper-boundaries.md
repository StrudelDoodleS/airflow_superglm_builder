# Shared Scaffold Helper Boundaries

## Status

Draft design for simplifying the default custom pricing model scaffold after the
frame-backed manifest work in PR #3.

## Problem

The custom scaffold now has the right high-level shape:

```text
register model
prepare source data
train/export/create frame manifest
publish completed build
```

However, the generated model package still contains too many helper functions in
`modeling.py` and `airflow_tasks.py`. Several of those functions are not
model-specific. They exist only to support stable publish metadata, model-version
idempotency, Airflow run metadata, or `CompletedModelBuild` payload creation.

That makes a newly scaffolded model harder to understand. A model author should
be able to quickly see:

```text
what they must edit
what the library handles
what the DAG wires together
```

The cleanup should not create another all-in-one DAG factory. It should only move
boilerplate that is genuinely reusable across models.

## Reported Issue

A generated custom model currently has many `def` statements in `modeling.py`.
Most model authors only need to edit a small subset of them. The confusing part
is that model-agnostic SQL publish/version/payload helpers are copied into every
generated model file.

That raises the core question this design answers:

```text
if a function is really needed only for the shared SQL publish contract, why is
it copied into each model instead of living in shared library code?
```

## Design Principle

Move a helper into `pricing_pipeline` only when it is required for the shared
SQL publish, manifest, versioning, or Airflow runtime contract and does not need
model-specific business logic.

Keep a helper in the model package when it defines:

```text
source access
feature engineering
target construction
model fitting
validation metrics
rating workbook export
model-specific task composition
```

The goal is not zero boilerplate. The goal is that scaffolded model files show
only meaningful model-owned code plus a small readable orchestration function.

## Desired File Responsibilities

### `model.toml`

Keep as-is. It is the model identity and housekeeping config.

It should define:

```text
model_key
model_label
target_name
model_type
deployment_slot
default_package_status
validation_split
```

No helper extraction needed.

### `spec.py`

Keep as-is. It should stay tiny:

```python
MODEL_CONFIG = load_model_build_config(Path(__file__).with_name("model.toml"))
```

This file is generic, but it is intentionally local so registry/config discovery
has a consistent import surface. Do not add `MODEL_SPEC` to the default custom
scaffold.

### `data.py`

Mostly model-specific. Leave it model-owned.

Keep:

```text
DATASET_NAME
SOURCE_SYSTEM
PK_COLUMNS
WEIGHT_COLUMN
DEFAULT_OUTPUT_ROOT
prepare_source_data(...)
```

`prepare_source_data(...)` is team/source-specific. It may read a SQL file, call a
work connection helper, stage a parquet/csv file, or write a run-specific table.
The library should not try to generalize that.

### `modeling.py`

Primary cleanup target.

Keep model-owned extension points:

```python
def read_prepared_source(prepared) -> pd.DataFrame:
    ...

def build_final_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    ...

def fit_validate_export_rating_tables(...) -> tuple[path, path | None, dict[str, float]]:
    ...
```

Keep one readable model orchestration function:

```python
def train_validate_export_model(prepared, *, engine, settings, created_by="airflow"):
    ...
```

This function should remain visible because it tells the model author the build
recipe. It should call shared helpers instead of redefining version/payload
boilerplate.

Move generic helpers out of scaffolded `modeling.py`:

```text
effective_from_for_run(...)
required_payload_text(...)
existing_model_version_for_export(...)
next_trained_model_version(...)
resolve_model_version_for_export(...)
completed_model_build_payload(...)
```

### `airflow_tasks.py`

Keep as thin TaskFlow wrappers, but move generic context/runtime helpers out.

Keep local wrappers:

```python
def prepare_source_data_task(...):
    ...

def train_validate_export_task(...):
    ...
```

Those wrappers show the model's task boundary and make the DAG simple to read.
They should still call model-owned functions.

Move generic Airflow helper behavior out:

```text
derive logical date from Airflow context
derive stable run key
build default run metadata payload fields
```

Do not replace the wrappers with a large all-purpose DAG factory.
Keep runtime loading local in the generated wrappers unless it becomes genuinely
more complex; `runtime_from_env_or_module(runtime_module)` is not the confusing
part of the scaffold.

## Proposed Shared Modules

Avoid vague `helpers` dumping grounds. Shared module names should describe the
contract they serve.

### `pricing_pipeline/orchestration/completed_build_helpers.py`

Owns small helpers that are useful for any completed-build publish flow.

Public API:

```python
def effective_from_for_run(value: date | datetime | str | None = None) -> str:
    ...

def required_payload_text(payload: Mapping[str, Any], field_name: str) -> str:
    ...

def completed_model_build_payload(
    *,
    rating_workbook_path: str | Path,
    model_version: str,
    effective_from: str,
    export_id: str,
    created_by: str,
    manifest_id: str,
    split_set_id: str | None,
    mlflow_run_id: str | None = None,
    model_artifact_path: str | Path | None = None,
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    ...
```

`effective_from_for_run(...)` should be stricter than the old scaffold code:

```text
date(2026, 6, 5)             -> "2026-06-05"
datetime(2026, 6, 5, 14, 30) -> "2026-06-05"
"2026-06-05"                 -> "2026-06-05"
"2026-06-05T14:30:00"        -> "2026-06-05"
None                         -> today, for manual/direct defaults only
"" / "abc" / 20260605        -> ValueError
```

Airflow task metadata should normally pass a logical date, not `None`, so
retries and backfills stay stable.

`required_payload_text(...)` should stay boring. It only checks that a payload
field exists and is non-blank. It must not become a generic prepared-payload
schema system.

The payload helper is only a convenience wrapper around
`CompletedModelBuild(...).to_dict()`. It should not publish, deploy, inspect SQL,
or create manifests.

For the recommended custom path, `completed_model_build_payload(...)` should
require `manifest_id`. Lower-level `CompletedModelBuild` can remain more flexible
for compatibility.

The helper should keep `mlflow_run_id` optional. MLflow remains optional for the
overall workflow, but models that do use it should not need to bypass this helper
just to include the run ID.

### `pricing_pipeline/publishing/model_versions.py`

Owns idempotent model-version resolution for rate package exports.

Public API:

```python
def existing_model_version_for_export(
    engine,
    *,
    model_key: str,
    export_id: str,
) -> str | None:
    ...

def next_trained_model_version(engine, *, model_key: str) -> str:
    ...

def resolve_model_version_for_export(
    engine,
    *,
    model_key: str,
    export_id: str,
) -> str:
    ...
```

Rules:

```text
same model_key + source_export_id -> reuse existing model_version
new export_id -> allocate next vN from non-manual/non-child package history
manual child packages must not bump the trained model version sequence
non-vN package versions are ignored when allocating the next vN
existing source_export_id returns its stored model_version exactly, even if not vN
```

This is shared publish correctness, not model logic.

`model_versions.py` should use `schema_names_from_connectable(engine)` or the
same schema-rendering approach as the publishing code. It must respect configured
pricing schema names rather than hard-coding `pricing.PRICING_RATE_PACKAGE`.

`next_trained_model_version(...)` should scan only rows where:

```sql
parent_rate_package_id IS NULL
```

and only count versions matching:

```text
v<integer>
```

It should ignore non-`vN` history rather than failing. Older/manual models may
use versions like `202606`, `2026.06`, or `2026-06-05`.

This helper is not a transactional global allocator. If two brand-new exports
for the same model resolve a next version concurrently before either publishes,
they may both choose the same `vN`. Package idempotency is still enforced by
`source_export_id`. If a team requires globally unique trained model versions
under concurrent builds, that should be enforced in the publish/catalogue
transaction or by serializing builds per `model_key`.

### `pricing_pipeline/orchestration/airflow_run_metadata.py`

Owns small Airflow-only helper functions used by generated task wrappers.

Public API:

```python
def context_logical_date(context: Mapping[str, Any]) -> object | None:
    ...

def task_run_metadata(
    context: Mapping[str, Any],
    *,
    output_root: str | Path,
) -> dict[str, str]:
    ...

def merge_prepared_payload_metadata(
    metadata: Mapping[str, str],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    ...
```

`task_run_metadata(...)` should return:

```text
run_key
output_dir
effective_from
data_as_of_date
```

It may use `run_key_for_value(...)` and `effective_from_for_run(...)`.

Precedence must be explicit:

```python
logical_date = context_logical_date(context)
run_value = context.get("run_id") or logical_date or "manual"

run_key = run_key_for_value(run_value)
effective_from = effective_from_for_run(logical_date)
data_as_of_date = effective_from
output_dir = str(Path(output_root) / run_key)
```

The distinction is important:

```text
run_key:
  usually comes from Airflow run_id because it identifies the run attempt.

effective_from / data_as_of_date:
  usually come from logical_date because that is stable for retries and
  backfills and is date-like.
```

Do not derive `effective_from` from `run_id` unless there is no logical date and
the run ID is intentionally date-like.

Do not import Airflow at module import time. The generated wrappers should still
import Airflow inside task-factory functions so non-Airflow tests/direct runners
can import model code.

## Generated Scaffold After Cleanup

The default custom `modeling.py` should read roughly like:

```python
from pricing_pipeline.orchestration.completed_build_helpers import (
    completed_model_build_payload,
    required_payload_text,
)
from pricing_pipeline.publishing.model_versions import resolve_model_version_for_export


def read_prepared_source(prepared):
    raise NotImplementedError(...)


def build_final_model_frame(raw):
    return raw.copy()


def fit_validate_export_rating_tables(...):
    raise NotImplementedError(...)


def train_validate_export_model(prepared, *, engine, settings, created_by="airflow"):
    run_key = str(prepared.get("run_key") or "manual")
    export_id = build_export_id(MODEL_CONFIG.model_key, run_key)
    model_version = resolve_model_version_for_export(
        engine,
        model_key=MODEL_CONFIG.model_key,
        export_id=export_id,
    )

    raw = read_prepared_source(prepared)
    frame = build_final_model_frame(raw)
    frame = frame.sort_values(list(PK_COLUMNS)).reset_index(drop=True)

    split_indices = validation_split_indices(frame, MODEL_CONFIG.validation_split)
    workbook_path, model_path, metrics = fit_validate_export_rating_tables(...)

    manifest = create_model_frame_manifest_with_split(...)

    return completed_model_build_payload(...)
```

The default custom `airflow_tasks.py` should read roughly like:

```python
from pricing_pipeline.orchestration.airflow_run_metadata import (
    merge_prepared_payload_metadata,
    task_run_metadata,
)


def prepare_source_data_task(...):
    from airflow.sdk import get_current_context, task

    @task(task_id=task_id)
    def _prepare_source_data():
        runtime = runtime_from_env_or_module(runtime_module)
        metadata = task_run_metadata(get_current_context(), output_root=output_root)
        payload = prepare_source_data(
            runtime.get_engine(),
            run_key=metadata["run_key"],
            output_dir=Path(metadata["output_dir"]),
        )
        return merge_prepared_payload_metadata(metadata, payload)

    return _prepare_source_data
```

The generated `data.py` should therefore make `prepare_source_data(...)` accept
`output_dir`, not `output_root`. The shared metadata helper computes the
run-specific output directory once, and the model-owned data function receives
the exact location where it should write handoff artifacts.

The generated wrapper must not silently let model-owned payload data contradict
the run metadata. In particular, `run_key` should be stable for the task and must
not be overridden to a different value after `prepare_source_data(...)` has
already received it.

The merge rule should be:

```python
if "run_key" in payload and payload["run_key"] != metadata["run_key"]:
    raise ValueError("prepare_source_data returned a run_key that differs from task metadata")

return {
    **metadata,
    **payload,
    "run_key": metadata["run_key"],
    "output_dir": payload.get("output_dir", metadata["output_dir"]),
    "effective_from": payload.get("effective_from", metadata["effective_from"]),
    "data_as_of_date": payload.get("data_as_of_date", metadata["data_as_of_date"]),
}
```

This lets a model/source override `data_as_of_date` when the source has a better
as-of date, while preventing `run_key`, paths, and export IDs from drifting apart.

The generated DAG stays unchanged:

```python
registered = register_pricing_model_task(model_config=MODEL_CONFIG)()
prepared = prepare_source_data_task()()
completed = train_validate_export_task()(prepared)
published = publish_completed_model_build_task(model_config=MODEL_CONFIG)(completed)

registered >> prepared >> completed >> published
```

## Non-Goals

- Do not reintroduce `MODEL_SPEC` into the default custom scaffold.
- Do not require `DatasetSpec`, `TRAINING_SQL`, or `feature_columns` for custom
  DAGs.
- Do not create a new all-in-one DAG factory.
- Do not generate runtime scoring SQL, views, procedures, or application feature
  code.
- Do not change SQL DDL or require a database re-seed.
- Do not generalize team-specific data access.

## Compatibility

Existing generated custom models can keep working because the old local helper
functions are just normal Python. This cleanup only changes newly generated
scaffolds and introduces reusable library helpers.

Factory/spec models remain separate:

```text
Factory path:
  MODEL_SPEC
  DatasetSpec / manifest_sql
  build_pricing_model_dag(...)
  run_pipeline_no_airflow.py

Custom path:
  MODEL_CONFIG
  model-owned data/modeling/tasks
  frame-backed manifest
  CompletedModelBuild
  publish_completed_model_build_task(...)
```

## Error Handling

Shared helpers should raise clear `ValueError` messages for missing required
payload fields, blank `model_key` / `export_id` values, and malformed date
inputs. They should not treat non-`vN` historical `model_version` values as
errors when allocating the next scaffold `vN`; those values are ignored for
sequence purposes.

The version helpers should not hide SQL errors from version lookup, because those
indicate registry/package catalogue problems.

`CompletedModelBuild` validation remains the final boundary for the publish
payload.

## Testing

Add tests for shared helper behavior:

```text
effective_from_for_run normalizes date/datetime/string values
effective_from_for_run rejects blank, non-date text, and numeric values
required_payload_text rejects missing/blank fields
completed_model_build_payload returns a valid CompletedModelBuild dict
completed_model_build_payload includes optional mlflow_run_id when provided
existing_model_version_for_export reuses the version for a known export_id
resolve_model_version_for_export allocates next vN for a new export_id
manual child packages do not affect next trained model version
non-vN versions are ignored when allocating the next vN
existing source_export_id returns non-vN model_version exactly if that is what was stored
model version lookup respects configured pricing schema names
task_run_metadata uses run_id for run_key when run_id is present
task_run_metadata uses logical_date for effective_from and data_as_of_date
task_run_metadata produces the same metadata for an Airflow retry of the same run
task_run_metadata does not use wall-clock today when logical_date is present
Airflow run metadata module does not import Airflow at module import time
prepared payload merge rejects run_key mismatches
prepared payload merge allows data_as_of_date override
```

Update scaffold tests:

```text
generated modeling.py imports shared helper modules
generated modeling.py still exposes the three model-owned extension points
generated modeling.py no longer defines copied version/payload helpers
generated airflow_tasks.py imports shared Airflow task helper
generated data.py prepare_source_data receives output_dir instead of output_root
generated DAG shape stays register -> prepare -> train/export -> publish
custom scaffold still does not contain MODEL_SPEC, DatasetSpec, manifest_sql, or
create_prepared_dataset_manifest_task
```

Run verification:

```bash
rtk uv run pytest -q
rtk uv run ruff check .
rtk uv run ruff format --check <changed files>
rtk uv run python scripts/no_docker_services.py menu --dry-run
rtk git diff --check
```

No SQL migration or `sqlfluff` verification is required unless implementation
unexpectedly edits SQL files.

## Implementation Order

1. Add shared helper modules and tests.
2. Update generated custom scaffold templates to import shared helpers.
3. Update scaffold tests to assert the simplified output.
4. Regenerate or inspect a temporary scaffold and compile/lint it.
5. Update README wording if it currently implies model authors should edit copied
   lifecycle/version helpers.

## Acceptance Criteria

- A newly scaffolded custom model has fewer copied helper functions.
- The model author clearly sees only:

```text
read source
build final frame
fit/validate/export
small train/export recipe
thin Airflow wrappers
```

- Shared publish/version/metadata helpers live under `pricing_pipeline`.
- Existing factory/spec path behavior is unchanged.
- Existing custom publish/deploy SQL lifecycle behavior is unchanged.
- No DDL change and no database re-seed are required.
