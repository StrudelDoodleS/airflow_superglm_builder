# Completed Model Build Publish API Design

## Purpose

Add a clean Python and Airflow API for publishing a completed model build into
the SQL Server pricing/MLOps catalogue without forcing production DAG authors to
use the all-in-one `build_pricing_model_dag(...)` factory.

The desired serious-workflow shape is:

```text
user task: get data
  -> user task: transform/materialize data
  -> user task: train, validate, and export rating tables
  -> library task: publish manifest, lineage, and rate package to SQL
```

The final library task should be the boring lifecycle boilerplate. It should
know how to validate the model registry row, create dataset/split metadata,
stage the exported rating workbook, allocate the next package version, write the
normalized rate package tables, and record model-run lineage.

## Current Context

The repository already has most of the lower-level pieces:

- `model.toml` stores model housekeeping config and is auto-discovered by
  `pricing_models.registry`.
- `ModelBuildConfig` stores `model_key`, `target_name`, `model_type`,
  `deployment_slot`, default package status, and validation split config.
- `DatasetSpec` describes lineage metadata and a SQL query used for manifest
  creation.
- `create_dataset_manifest_with_split(...)` writes dataset manifest rows,
  column metadata, validation split set metadata, and optional `.npz` split
  artifacts.
- `ModelPublisher.publish_training_export(...)` validates the registered model,
  stages a rating workbook, and publishes a package.
- `publish_model_export(...)` records model-run lineage after publishing.
- `build_pricing_model_dag(...)` wires schema apply, manifest creation,
  training/export, and publish into one generated DAG.

The problem is not missing backend capability. The problem is UX and
composability: the current DAG factory owns too much of the user's workflow.
Production users want to define their own Airflow tasks for ingestion,
transforms, validation, and training, then bolt on one clear publish task at the
end.

## Goals

- Let DAG authors control data ingestion, transforms, training, validation, and
  intermediate disk handoffs.
- Provide one importable Python helper for publishing a completed build.
- Provide one importable Airflow TaskFlow task factory wrapping that helper.
- Keep publishing separate from deployment. The new helper creates a
  deployable candidate package, but never marks it current.
- Reuse the existing SQL lifecycle functions instead of creating a parallel
  publisher.
- Call importable package modules and lifecycle APIs, not CLI script modules.
- Keep `build_pricing_model_dag(...)` available as a demo/simple wrapper, but no
  longer position it as the serious-production API.
- Avoid large Airflow XCom payloads. Tasks pass small dictionaries containing
  paths, IDs, counts, and metadata.

## Non-Goals

- Rebuild the training pipeline.
- Replace `ModelPublisher`.
- Add a new CLI.
- Deploy packages as part of the completed-build publish step.
- Move package pointers or write deployment rows.
- Create or repair `PRICING_MODEL` rows.
- Infer `model_version` or `effective_from`.
- Accept arbitrary in-memory pandas DataFrames through Airflow XCom.
- Design a full rate-table dataframe API in v1. The v1 publish helper expects
  the existing rating workbook format that `stage_rating_export(...)` already
  understands.
- Guarantee that a mutable SQL work table still matches the training data
  unless the user materializes it in a run-stable way.
- Make `build_pricing_model_dag(...)` disappear. It remains a convenience API.

## Public API Shape

Add a new module:

```text
pricing_pipeline/orchestration/publish_completed_build.py
```

It exposes:

```python
from pricing_pipeline.orchestration.publish_completed_build import (
    CompletedModelBuild,
    CompletedModelPublishResult,
    publish_completed_model_build,
    publish_completed_model_build_task,
)
```

### Completed Build Contract

`CompletedModelBuild` is the small object that a user training task returns or
constructs from a returned dictionary. The public contract is dataclass-like:
construct it with keyword arguments, call `to_dict()`, or pass a plain mapping to
`from_mapping()`. Internally the implementation uses Pydantic v2 for boundary
validation and converts validation failures into `CompletedModelBuildError`.

```python
class CompletedModelBuild:
    rating_workbook_path: str
    model_version: str
    effective_from: str
    created_by: str | None = None
    export_id: str | None = None
    dag_id: str | None = None
    airflow_run_id: str | None = None
    mlflow_run_id: str | None = None
    manifest_id: str | None = None
    split_set_id: str | None = None
    model_artifact_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: "CompletedModelBuild | Mapping[str, Any]") -> "CompletedModelBuild":
        ...

    def to_dict(self) -> dict[str, Any]:
        ...
```

Required values:

- `rating_workbook_path`: path to the rating workbook produced by the user's
  training/export task.
- `model_version`: user-selected model version string, such as `20260603` or a
  run-specific version.
- `effective_from`: package effective date as `YYYY-MM-DD`.

Optional values:

- `created_by`: actor to record in SQL. The Airflow task wrapper can fill this
  from its `created_by` argument; the pure Python helper must receive a
  non-blank resolved value.
- `export_id`: source export identity. If omitted in an Airflow task wrapper,
  the wrapper derives it from `model_key` and Airflow `run_id` using the existing
  `build_export_id(...)` logic. The pure Python helper requires either
  `export_id` or `airflow_run_id`.
- `manifest_id` and `split_set_id`: if supplied, the helper reuses them and does
  not create new manifest/split rows.
- `model_artifact_path` and `metrics`: retained for future extension and
  optional model-run metadata. V1 does not create a model artifact store in SQL.

User Airflow tasks should return plain dictionaries rather than relying on
Airflow dataclass serialization:

```python
return CompletedModelBuild(
    rating_workbook_path=str(workbook_path),
    model_version=model_version,
    effective_from=effective_from,
    mlflow_run_id=mlflow_run_id,
    metrics=metrics,
).to_dict()
```

### Publish Result Contract

`CompletedModelPublishResult` returns the IDs the DAG needs for review or later
deployment.

```python
@dataclass(frozen=True)
class CompletedModelPublishResult:
    model_id: int
    model_key: str
    model_version: str
    manifest_id: str
    split_set_id: str | None
    export_id: str
    rate_package_id: int
    package_version: int
    package_status: str
    rating_workbook_path: str
    mlflow_run_id: str | None = None
    was_existing: bool = False
```

It has `to_dict()` for Airflow XCom compatibility.

## Pure Python Helper

Signature:

```python
def publish_completed_model_build(
    engine,
    *,
    settings: Settings,
    model_config: ModelBuildConfig,
    dataset: DatasetSpec | None,
    completed_build: CompletedModelBuild | Mapping[str, Any],
    package_status: str | None = None,
    created_by: str | None = None,
) -> CompletedModelPublishResult:
    ...
```

Responsibilities:

1. Coerce the completed-build mapping into `CompletedModelBuild`.
2. Fill helper-level runtime defaults when absent: `dag_id`,
   `airflow_run_id`, `export_id`, and `created_by`.
3. Validate required business fields: `rating_workbook_path`, `model_version`,
   `effective_from`, and resolved `created_by`.
4. Validate the registered `PRICING_MODEL` row read-only against
   `ModelBuildConfig`. The helper must not create or mutate the model registry
   row.
5. Resolve package status as `package_status or model_config.default_package_status`.
   The current production config default is `PUBLISHED`, which means immutable
   deployable candidate, not current/live.
6. Resolve manifest/split metadata:
   - when `completed_build.manifest_id` is supplied, validate/reuse it and use
     `completed_build.split_set_id` as-is;
   - when `completed_build.manifest_id` is absent, require `dataset` and create
     dataset manifest plus validation split metadata from `dataset.manifest_sql`.
7. Build `ModelExportResult` from the completed-build fields.
8. Publish the rate package through the existing lifecycle publisher with
   deploy/set-pointer disabled. The helper should call importable package
   modules and `ModelPublisher`/publisher functions, not CLI script modules.
9. Record model-run lineage.
10. Return `CompletedModelPublishResult`, including package version, package
    status, MLflow run ID, and whether the package already existed.

The helper does not call training code, does not call `apply_migrations(...)`,
does not call `DatasetSpec.raw_loader`, and does not deploy.

For pure Python calls outside Airflow:

- if `dag_id` is absent, use `"python_publish_completed_model_build"`;
- if `airflow_run_id` is absent but `export_id` is supplied, use `export_id` as
  the model-run lineage identity;
- if both `export_id` and `airflow_run_id` are absent, raise a clear error.

## Airflow Task Wrapper

Signature:

```python
def publish_completed_model_build_task(
    *,
    model_config: ModelBuildConfig,
    dataset: DatasetSpec | None = None,
    runtime_module: str | None = None,
    created_by: str = "airflow",
    task_id: str = "publish_completed_model_build",
):
    ...
```

The function returns an Airflow `@task` callable that accepts the small
completed-build dictionary from the user's training task.

Example:

```python
from airflow.sdk import dag

from pricing_models.claim_freq.spec import DATASET_SPEC, MODEL_CONFIG
from pricing_models.claim_freq.tasks import prepare_training_data, train_and_export_rates
from pricing_pipeline.orchestration.publish_completed_build import (
    publish_completed_model_build_task,
)


@dag(dag_id="claim_freq_build", schedule=None, catchup=False)
def claim_freq_build():
    data = prepare_training_data()
    build = train_and_export_rates(data)

    publish_completed_model_build_task(
        model_config=MODEL_CONFIG,
        dataset=DATASET_SPEC,
    )(build)


claim_freq_build()
```

The wrapper resolves runtime with `runtime_from_env_or_module(...)`, reads the
Airflow context, and fills missing `dag_id`, `airflow_run_id`, `export_id`, and
`created_by` only when the user did not supply them. It does not invent
`model_version` or `effective_from`; the user's training/export task must return
those values because they are model/package business decisions. `dataset` may be
omitted when the completed-build payload already includes `manifest_id` and
`split_set_id` from a separate manifest task.

## Dataset Manifest Boundary

V1 keeps the manifest source explicit and SQL-backed:

- User tasks can use disk handoffs between themselves.
- Before the publish task runs, the final model-ready dataset must be queryable
  through `dataset.manifest_sql`.
- That query can be as simple as `SELECT * FROM work_schema.claim_freq_training`.
- The manifest query must be stable for the build. It should point to an
  immutable run-specific table/view or include a run/export filter, not a
  moving work table that can change between training and publishing.
- `rating_workbook_path` and `model_artifact_path`, when supplied, must be
  readable by the worker that runs the publish task. In distributed Airflow,
  that means a shared volume, object-store mount, or another durable path
  convention.

This fits the current manifest writer and keeps the first implementation small.
It also matches production SQL Server workflows where upstream tasks refresh a
work table or view, then the lifecycle task records metadata from that final
query.

Disk-only manifest support is a later feature. It should be added as a separate
source type, not squeezed into this v1 helper.

If validation metrics are reported against a specific split, the training task
must return the `manifest_id` and `split_set_id` used for those metrics.
Auto-created split metadata is only valid when the training task used the same
`DatasetSpec` split configuration or when no split-specific metrics are being
claimed.

## Model Folder Pattern

The serious-model folder can be organized however the model owner wants:

```text
pricing_models/claim_freq/
  model.toml
  spec.py
  tasks.py
  data.py
  transforms.py
  train.py
```

Only `model.toml` and `spec.py` are conventional integration points:

- `model.toml` is discovered for model config.
- `spec.py` exposes `MODEL_CONFIG` and `DATASET_SPEC`.
- User task files can be named anything and can be imported by the DAG directly.

The scaffold can continue creating `training.py` for simple starts, but docs
should state that file names are conventions, not framework requirements.

## Error Handling

The helper fails fast with clear errors when:

- `rating_workbook_path` is missing or does not exist.
- neither `export_id` nor `airflow_run_id` is available.
- `model_version`, `effective_from`, or `created_by` is blank.
- the SQL model registry row is missing or does not match `model_config`.
- `dataset.manifest_sql` cannot be read.
- the rating workbook cannot be staged.
- the export has already been published but conflicts with the requested model.

Idempotency key:

```text
model_id + source_export_id
```

If the key already exists:

- return the existing package when model version, effective dates, and source
  workbook identity are compatible;
- raise a conflict error if the existing package belongs to another model or
  materially different export metadata.

Re-running the same successful export should return `was_existing=True` in the
publish result.

## Testing Strategy

Unit tests:

- `CompletedModelBuild` accepts mappings and validates required fields.
- `publish_completed_model_build(...)` creates a manifest when one is not
  supplied.
- `publish_completed_model_build(...)` skips manifest creation when
  `manifest_id` is supplied.
- The helper builds the expected `ModelExportResult` and delegates to
  `publish_model_export(...)`.
- The Airflow task wrapper fills `dag_id`, `airflow_run_id`, and `export_id` from
  context when absent.
- Missing `created_by` in `CompletedModelBuild` is filled by the Airflow wrapper.
- Missing `created_by` in a pure Python call raises a clear error unless a
  helper-level default is supplied.
- A supplied `manifest_id` is validated and reused without reading
  `dataset.manifest_sql`.
- A missing `manifest_id` requires `dataset.manifest_sql` and creates a
  `split_set_id` that appears in the result.
- Model registry mismatch raises and does not call staging or publish code.
- Existing `source_export_id` with compatible metadata returns
  `was_existing=True`.
- Existing `source_export_id` with conflicting model ID, model version, or
  effective date raises a conflict error.
- The wrapper does not call user training code, schema apply, raw loaders, or
  deployment.
- The wrapper never calls package pointer or deploy functions.
- The wrapper accepts a plain dictionary from the upstream user task.

Integration-style tests:

- Existing lifecycle workflow can publish through the new helper and still write
  the same package/deployment lineage.
- Re-running with the same `export_id` returns the existing package instead of
  creating a duplicate.

Docs/readme contract tests should be updated so the recommended production DAG
pattern is the composable task pattern, while `build_pricing_model_dag(...)` is
documented as optional convenience/demo plumbing.

## Migration and Compatibility

No SQL migration is required for this API. It uses the tables and guards already
introduced by the rate-package lifecycle work.

Existing code remains compatible:

- `build_pricing_model_dag(...)` keeps working.
- `run_pipeline_no_airflow.py` keeps working.
- `ModelPublisher.publish_training_export(...)` keeps working.
- The generic deploy DAG keeps working and remains the intended deployment path.

The new helper is additive. After it lands, docs and scaffold comments can steer
new model authors toward custom DAG composition plus the final publish task.

## Open Follow-Up After V1

After the clean publish helper is stable, consider:

- disk-backed manifest creation from CSV/parquet artifacts;
- a dataframe/native Python rate-table publish API that bypasses workbook
  parsing;
- scaffold generation of a `tasks.py` example showing custom TaskFlow tasks;
- optional review/report task helpers between publish and deploy.
