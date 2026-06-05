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

This design is about build audit metadata, not runtime scoring.

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

## Non-Goals

This design does not generate runtime scoring SQL.

It does not create SQL views, stored procedures, table-valued functions, or
application feature code for engineered model inputs. A published package may
depend on banded or derived features, such as values produced by `pd.cut()` in
the training task or by equivalent SQL in the scoring path. The scoring layer is
responsible for supplying feature values compatible with the published package.

This design records the final model frame and publishes the rating package. It
does not attempt to make every Python feature-engineering step executable in SQL
Server.

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

`DATASET_COLUMN` records all final model-frame columns, including PK, target,
and weight columns. The rate package tables remain the source of truth for
published rating inputs. PK, target, and weight columns should not become rating
features unless the model author explicitly includes them, which should normally
be treated as an export-layer error.

## Audit Coverage

For v1, the build audit trail is:

```text
DATASET_MANIFEST
  manifest_id, dataset_name, source_system, data_as_of_date, row_count,
  pk_columns_json, target_column, weight_column, created_ts, created_by

DATASET_COLUMN
  final model-frame columns, roles, pandas dtype, null counts, distinct counts

CV_SPLIT_SET / CV_FOLD
  split config, row-order hash, fold counts, optional materialized split artifact

MODEL_RUN / related lineage tables
  model_id, model_version, manifest_id, split_set_id, export_id,
  rate_package_id, workbook/model artifact paths, Airflow IDs when available

PRICING_RATE_PACKAGE and normalized package tables
  actual published rating structure
```

This answers:

- what model was built;
- which final frame it trained/exported from;
- what rows and columns were present;
- what split was used;
- what workbook was published;
- what SQL rate package was created;
- whether a rerun reused an existing package through the export ID.

It does not answer how production scoring computes every engineered feature.
That is a separate scoring integration project.

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

Use the same boundary-hardening style as `CompletedModelBuild`: required strings
must be non-empty, optional strings normalize cleanly, `data_as_of_date`
normalizes to a date, and `pk_columns` must be a non-empty tuple of unique
non-empty strings.

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

`DatasetSpec.manifest_sql` is a compatibility path for SQL-backed final model
frames and old factory/demo flows. It is not required for new custom DAGs and
should not be used merely to re-read raw source data for manifesting.

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
frame = frame.sort_values("policy_id").reset_index(drop=True)

split_indices = validation_split_indices(frame, MODEL_CONFIG.validation_split)
model, metrics, model_path = fit_model(frame, split_indices=split_indices)
workbook_path = export_rating_tables(model, frame)
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

A frame-backed manifest is an audit receipt for the frame supplied by the model
task. It does not persist the full frame unless the model task chooses to write a
frame artifact separately. Reproducibility depends on the source data, model
code, artifact paths, and recorded metadata being sufficient for the team's
audit standard.

## Split, Row Order, and Retry Semantics

The manifest writer treats the supplied frame order as the canonical model-frame
order. Model tasks should sort the final frame by PK or another deterministic
business key before calling `create_model_frame_manifest_with_split(...)`, unless
the training method deliberately requires a different order. The writer should
not silently sort the frame, because that could change what the model trained on.

If validation metrics are reported, the recorded split metadata must describe
the split actually used to produce those metrics. Model code should either use
the same deterministic split helper/config that the manifest writer records, or
pass/reuse its own split metadata when that API exists. Do not record default
k-fold metadata merely because a default split config exists.

A frame manifest is append-only unless a `manifest_id` is explicitly supplied.
For idempotent reruns of the same export, model tasks may derive a stable
manifest ID from the export/run key or look up and reuse an existing manifest.
V1 does not require manifest de-duplication; package idempotency remains enforced
by `source_export_id`.

A manifest may exist even if a later fit, export, or publish step fails.
Successful published builds are tied to the manifest through
`CompletedModelBuild.manifest_id` and model-run lineage.

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
- the frame has duplicate column names;
- the frame has blank or whitespace-only column names;
- `pk_columns` is empty;
- `pk_columns` contains duplicates;
- required PK columns are missing;
- target or weight columns are missing when supplied;
- PK columns contain nulls;
- PK column values are duplicated;
- `data_as_of_date` is blank or not date-like;
- `dataset_name` or `source_system` is blank;
- `validation_split.stratify_column` is supplied but missing from the frame;
- k-fold `n_splits` is greater than the frame row count;
- `validation_split.materialize=True` and no artifact root is supplied.

The function should not validate the original SQL source or connection logic.
At this point the model task has already built the final frame.

The DataFrame index is not part of row identity. A non-unique index is allowed
and ignored; PK columns define row identity and row-order hashing.

## Testing

Add focused tests for the new frame-backed manifest writer:

- writes `DATASET_MANIFEST` from an in-memory frame;
- writes `data_as_of_date` from the supplied spec instead of `date.today()`;
- writes `DATASET_COLUMN` for engineered features that do not exist in source
  SQL;
- writes target and weight column roles correctly;
- writes replayable k-fold split metadata;
- writes materialized train/test split `.npz` metadata when configured;
- records the supplied frame order consistently in the row-order hash;
- rejects duplicate final-frame column names;
- rejects blank final-frame column names;
- rejects empty `pk_columns`;
- rejects duplicate `pk_columns`;
- rejects missing PK columns;
- rejects missing target/weight columns when configured;
- rejects null or duplicate PK values;
- rejects missing stratify columns when configured;
- rejects k-fold splits where `n_splits > row_count`;
- rejects invalid `data_as_of_date`;
- leaves the existing SQL-backed `create_dataset_manifest_with_split(...)`
  working by delegating to the frame-backed writer.

Add scaffold/demo tests:

- default custom scaffold does not import `ModelSpec`;
- default custom scaffold does not require `DatasetSpec` or `TRAINING_SQL`;
- default custom DAG does not call `create_prepared_dataset_manifest_task(...)`;
- default custom train/export task returns `CompletedModelBuild` with
  `manifest_id` and `split_set_id`;
- custom train/export example creates the manifest from the final engineered
  frame, not the prepared raw frame;
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

Default custom scaffold examples should align with this spec before being
promoted as the recommended production pattern.
