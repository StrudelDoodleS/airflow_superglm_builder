# Model-Frame Dataset Manifest Design

## Problem

The current custom DAG/scaffold direction still assumes a dataset manifest can be
created by reading `DatasetSpec.manifest_sql`. That is too SQL-first for real
model builds.

A production model DAG may read source data from SQL, create targets or derived
features in Python, filter rows, select final model features, and never write the
final model-ready frame back to SQL Server. In that workflow, the original SQL
source does not contain the final engineered feature columns. Creating
`DATASET_MANIFEST` and `DATASET_COLUMN` from the original source query would
record the wrong row set and the wrong columns.

The manifest should describe the exact final model frame used for validation,
training, and rating-table export. Source provenance remains important, but for
v1 it should stay simple: track the logical source and source-data as-of date
using the fields the schema already has.

## Decision

For v1, do not add a raw source view requirement and do not add new SQL columns.

Add a frame-backed manifest API that writes to the existing tables:

- `pricing.DATASET_MANIFEST`
- `pricing.DATASET_COLUMN`
- `pricing.CV_SPLIT_SET`
- `pricing.CV_FOLD`

`DATASET_MANIFEST.source_system` records the logical source/provenance label,
such as `policy_admin`, `pricing_mart`, or `actuarial_workbench`.
`DATASET_MANIFEST.data_as_of_date` records the as-at/as-of date for the source
data used to build the model frame. `DATASET_COLUMN`, `row_count`, row-order
hashes, and split metadata describe the final post-ETL model frame, not the raw
source table.

This avoids pretending engineered features exist in the source SQL and avoids
forcing every model build to materialize a model-ready table back to SQL Server.

## Approaches Considered

The recommended approach is frame-backed manifest creation with source as-of
metadata. It fits the custom DAG shape: user code owns extraction, transforms,
feature engineering, and model fitting; the library records the final frame
metadata and publishes the package.

Forcing users to persist a model-ready SQL table would keep the current
`manifest_sql` path, but it creates unnecessary writeback requirements and makes
the scaffold hard to use when features only exist in Python.

Adding raw source view/query metadata to the SQL schema would add audit detail,
but it does not solve the actual problem: the source query still does not
contain engineered features. It also creates migration/re-seed churn before the
model build problem is fixed.

## Definitions

**Source data** is what the model task read before feature engineering. For v1,
the catalog tracks this with `source_system` and `data_as_of_date`.

**Model frame** is the final pandas `DataFrame` after source reads, joins,
filters, target construction, feature engineering, and feature selection. This
is the frame used for validation/training and rating-table export.

**Dataset manifest** is the SQL catalog receipt for the model frame. It records
row count, PK columns, target/weight columns, data as-of date, column metadata,
and optional validation split metadata.

**Rate package** is the exported pricing/rating structure. Its feature, term,
level, cell, and compiled tables remain the source of truth for what was
published into the rate package.

## Public API

Add a small frame manifest spec:

```python
@dataclass(frozen=True)
class ModelFrameManifestSpec:
    dataset_name: str
    source_system: str
    data_as_of_date: date | datetime | str
    pk_columns: tuple[str, ...]
    target_column: str | None
    weight_column: str | None = None
```

Add a frame-backed writer:

```python
def create_model_frame_manifest_with_split(
    engine: Engine,
    *,
    frame: pd.DataFrame,
    spec: ModelFrameManifestSpec,
    manifest_id: str | None = None,
    validation_split: ValidationSplitConfig = ValidationSplitConfig.kfold(),
    validation_split_artifact_root: Path | None = None,
    created_by: str = "airflow",
) -> DatasetManifestResult:
    ...
```

The function creates a manifest ID when one is not supplied, writes
`DATASET_MANIFEST`, writes final-frame `DATASET_COLUMN`, and writes validation
split metadata using the existing split logic. If `validation_split.materialize`
is true, it writes the compressed `.npz` split index artifact exactly as the
current SQL-backed manifest path does.

The existing SQL-backed API stays available:

```python
def create_dataset_manifest_with_split(..., dataset: DatasetSpec, ...):
    frame = pd.read_sql_query(text(dataset.manifest_sql), engine)
    return create_model_frame_manifest_with_split(..., frame=frame, ...)
```

That keeps old factory/demo flows working while giving custom DAGs the correct
default path.

## Custom DAG Flow

The production-style DAG should be:

```text
register model
-> prepare source data
-> train/validate/export and create model-frame manifest
-> publish completed build
-> deploy separately after review
```

The DAG code stays small:

```python
registered = register_pricing_model_task(model_config=MODEL_CONFIG)()
prepared = prepare_source_data_task()()
completed = train_validate_export_task()(prepared)
published = publish_completed_model_build_task(model_config=MODEL_CONFIG)(completed)

registered >> prepared >> completed >> published
```

Inside the model-owned training/export task:

```python
raw = read_prepared_source(prepared)
frame = build_final_model_frame(raw)

manifest = create_model_frame_manifest_with_split(
    runtime.get_engine(),
    frame=frame,
    spec=ModelFrameManifestSpec(
        dataset_name="claim_frequency_model_frame",
        source_system="policy_admin",
        data_as_of_date=prepared["data_as_of_date"],
        pk_columns=("policy_id",),
        target_column=MODEL_CONFIG.target_name,
        weight_column="earned_exposure",
    ),
    validation_split=MODEL_CONFIG.validation_split,
    validation_split_artifact_root=runtime.settings.validation_split_artifact_root,
    created_by="airflow",
)

model, metrics, model_path = fit_model(frame)
workbook_path = export_rating_tables(model, frame)

return CompletedModelBuild(
    rating_workbook_path=str(workbook_path),
    model_version=model_version,
    effective_from=effective_from,
    created_by="airflow",
    export_id=export_id,
    model_artifact_path=str(model_path),
    metrics=metrics,
    manifest_id=manifest.manifest_id,
    split_set_id=manifest.split_set_id,
).to_dict()
```

The publish task then uses the supplied `manifest_id` and `split_set_id`. It does
not need a `DatasetSpec` and it should not create a manifest implicitly in the
recommended custom DAG path.

## Data As-Of Handling

The data as-of date is required for the frame manifest spec. It should come from
the model task or upstream prepared payload, not from TOML by default.

Examples:

```python
data_as_of_date = prepared["data_as_of_date"]
data_as_of_date = frame["snapshot_date"].max().date()
data_as_of_date = date.today()  # acceptable for demos/manual runs only
```

Optional helper functions can make repetitive cases nicer:

```python
data_as_of_date = data_as_of_from_column(frame, "snapshot_date")
data_as_of_date = data_as_of_from_payload(prepared, "data_as_of_date")
```

These helpers are convenience functions only. They should not introduce a TOML
DSL for source SQL, server/database routing, or feature lists.

## Model Config and Old Factory Boundary

`ModelBuildConfig` / `model.toml` remains stable housekeeping for model registry
and publish defaults:

- `model_key`
- `model_name`
- `target_name`
- `model_type`
- `deployment_slot`
- validation split defaults
- default package status

The custom DAG path should not require `ModelSpec`, `DatasetSpec`,
`TRAINING_SQL`, or `feature_columns`. Those are factory-era conveniences and may
remain for backwards compatibility, but they should not appear as required
pieces in the default custom scaffold.

New scaffolded custom models should make this clear:

- `spec.py` loads `MODEL_CONFIG` from TOML only.
- `data.py` contains optional source-read/prep helpers.
- `modeling.py` builds the final model frame, creates the frame manifest, fits
  the model, exports rating tables, and returns `CompletedModelBuild.to_dict()`.
- the DAG imports reusable registry/publish tasks plus model-owned source and
  train/export tasks.

## Error Handling

`create_model_frame_manifest_with_split(...)` should raise clear errors when:

- the frame is empty;
- required PK columns are missing;
- target or weight columns are missing when supplied;
- PK columns contain nulls;
- PK column values are duplicated;
- `data_as_of_date` is blank or not date-like;
- `dataset_name` or `source_system` is blank;
- `validation_split.materialize=True` and no artifact root is supplied.

The function should not validate the original SQL source or connection logic.
At this point the model task has already built the final frame.

## Testing

Add focused tests for the new frame-backed manifest writer:

- writes `DATASET_MANIFEST` from an in-memory frame;
- writes `data_as_of_date` from the supplied spec instead of `date.today()`;
- writes `DATASET_COLUMN` for engineered features that do not exist in source
  SQL;
- writes target and weight column roles correctly;
- writes replayable k-fold split metadata;
- writes materialized train/test split `.npz` metadata when configured;
- rejects missing PK columns;
- rejects missing target/weight columns when configured;
- rejects null or duplicate PK values;
- rejects invalid `data_as_of_date`;
- leaves the existing SQL-backed `create_dataset_manifest_with_split(...)`
  working by delegating to the frame-backed writer.

Add scaffold/demo tests:

- default custom scaffold does not import `ModelSpec`;
- default custom scaffold does not require `DatasetSpec` or `TRAINING_SQL`;
- default custom DAG does not call `create_prepared_dataset_manifest_task(...)`;
- default custom train/export task returns `CompletedModelBuild` with
  `manifest_id` and `split_set_id`;
- `publish_completed_model_build_task(...)` still publishes from supplied
  manifest IDs and does not deploy.

Run the normal no-Docker checks after implementation:

```bash
rtk uv run pytest -q
rtk uv run ruff check ...
rtk uv run ruff format --check ...
rtk uv run python scripts/no_docker_services.py menu --dry-run
rtk uv run pytest -q tests/test_sql_server_syntax.py
rtk uv run sqlfluff parse --dialect tsql db/migrations
```

## Migration Impact

No SQL migration is required for v1.

The existing schema already has the necessary fields:

- `DATASET_MANIFEST.source_system`
- `DATASET_MANIFEST.data_as_of_date`
- `DATASET_MANIFEST.row_count`
- `DATASET_MANIFEST.pk_columns_json`
- `DATASET_COLUMN`
- `CV_SPLIT_SET`
- `CV_FOLD`

The implementation changes how manifest rows are created, not the table shape.
That means no re-seed is required solely for this fix.

Future richer audit fields, such as source database, source object, source SQL
hash, or observation-period dates, can be added later if the team needs them.
They are not required to unblock model builds.

## Implementation Notes

Most of the existing logic in `pricing_pipeline.data.manifest` can be reused.
The code already knows how to:

- build column metadata from a pandas frame;
- compute row-order hashes from PK columns;
- build replayable and materialized split metadata;
- write compressed split index artifacts.

The implementation should extract the common write path from
`create_dataset_manifest_with_split(...)` into the new frame-backed function.
The SQL-backed function should become a thin adapter that reads `manifest_sql`
into a frame and passes it to the new function.

The currently open custom scaffold PR should stay unmerged until its default
custom DAG aligns with this spec. The previous scaffold shape still creates the
manifest before training/export from a SQL `DatasetSpec`, which is the issue this
spec fixes.
