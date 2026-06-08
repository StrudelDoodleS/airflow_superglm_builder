# Compact CV Split Artifact Design

## Problem

When `validation_split.materialize = true`, the current split artifact writes
explicit train/test row-position arrays for every fold:

```text
fold_1_train_idx
fold_1_test_idx
fold_2_train_idx
fold_2_test_idx
...
fold_k_train_idx
fold_k_test_idx
```

This format is simple and general, but it repeats row positions. For ordinary
exclusive validation assignments, such as k-fold or holdout, each row belongs to
exactly one test fold or one holdout role. Storing every train array is
therefore redundant.

For k-fold, knowing each row's test fold is enough:

```text
fold 1 test rows = rows where test_fold == 1
fold 1 train rows = rows where test_fold != 1
```

For holdout, knowing whether each row is test/holdout is enough:

```text
test rows = rows where is_test is true
train rows = rows where is_test is false
```

The goal is to reduce `.npz` artifact size without putting row-level split
assignments into SQL Server.

## Decision

Add a compact split artifact format for exclusive row-assignment splits.

The compact artifact records:

- artifact format metadata;
- the configured primary key columns for the model frame;
- a row identity hash derived from those primary key columns;
- one compact assignment array.

The artifact still uses row positions relative to the final model frame for
fast reconstruction, but the stored row identity makes the artifact auditable
and tied to whatever PK columns the model declared.

For k-fold-like splits:

```text
split_format = "fold_assignment_v1"
pk_columns = ["policy_id"]                 # or ["policy_id", "snapshot_month"]
row_key_hash = [...]
test_fold = [...]
```

For holdout-like splits:

```text
split_format = "holdout_assignment_v1"
pk_columns = ["policy_id"]                 # or composite PK columns
row_key_hash = [...]
is_test = [...]
```

The PK columns are not hard-coded. They come from
`ModelFrameManifestSpec.pk_columns` or the equivalent `DatasetSpec.pk_columns`
for the SQL-backed compatibility path.

## Non-Goals

This design does not store row-level split assignments in SQL Server.

It does not change `CV_SPLIT_SET`, `CV_FOLD`, `DATASET_MANIFEST`, or
`DATASET_COLUMN`.

It does not try to support every possible split strategy with the compact
format. Splits where a row can appear in test multiple times, or where folds are
not exclusive assignments, should keep the existing explicit train/test arrays.

It does not make split artifacts a source-data store. The artifact is still a
small validation-index artifact keyed to the final model-frame row identity.

## Artifact Formats

### Legacy Explicit Format

The current format remains supported:

```text
fold_1_train_idx
fold_1_test_idx
fold_2_train_idx
fold_2_test_idx
...
```

This format is general and can represent arbitrary split lists.

### Fold Assignment Format

Use this for exclusive multi-fold splits:

```text
split_format = "fold_assignment_v1"
pk_columns = np.array([...])
row_key_hash = np.array([...])
test_fold = np.array([...])
```

Rules:

- `test_fold` length must equal `row_count`.
- `row_key_hash` length must equal `row_count`.
- fold numbers are 1-based and must be in `1..fold_count`.
- every fold number in `1..fold_count` must have at least one test row.
- train rows for fold `n` are `test_fold != n`.
- test rows for fold `n` are `test_fold == n`.

### Holdout Assignment Format

Use this for one-fold train/test splits:

```text
split_format = "holdout_assignment_v1"
pk_columns = np.array([...])
row_key_hash = np.array([...])
is_test = np.array([...])
```

Rules:

- `is_test` length must equal `row_count`.
- `row_key_hash` length must equal `row_count`.
- `is_test` must be boolean or safely coercible to boolean.
- there must be at least one train row and at least one test row.
- train rows are `~is_test`.
- test rows are `is_test`.

## Row Identity

The artifact should use the model frame's configured PK columns, not a fixed
column name.

For a single PK:

```text
pk_columns = ["policy_id"]
```

For composite PKs:

```text
pk_columns = ["policy_id", "snapshot_month"]
```

The compact artifact should store a canonical `row_key_hash` array rather than
raw PK values by default.

Reasons:

- PK values may be sensitive.
- composite keys can contain mixed types.
- object arrays in `.npz` are awkward and can require pickle loading.
- the manifest already records `pk_columns_json`; the artifact only needs a
  durable row identity check.

The row key hash should be deterministic from the configured PK columns. It
should use the same canonical JSON-style value normalization as the row-order
hash path, but produce one hash per row instead of one digest for the whole
frame.

V1 should still require the current frame row order to match the artifact order.
That means loading compact artifacts verifies:

```text
current row_key_hash array == artifact row_key_hash array
current row_count == artifact assignment length
current row_order_sha256 == CV_SPLIT_SET.row_order_sha256
```

Future work could support reordering by mapping `row_key_hash -> row position`,
but v1 should not silently reorder. The current row-order contract is safer and
easier to reason about.

## Split Methods

Use compact fold assignment for methods where each row has exactly one test
fold:

- `kfold`
- `column_kfold`
- future `stratified_kfold`
- future `group_kfold`

Use compact holdout assignment for methods where each row is train or test once:

- `train_test_split`
- `column_holdout`

Keep the legacy explicit format for methods where a row can be in test multiple
times or where folds are arbitrary:

- repeated k-fold;
- repeated shuffle split;
- bootstrap validation;
- user-supplied arbitrary split lists.

## Public API

Keep the existing high-level APIs:

```python
write_validation_split_npz(...)
load_materialized_cv_folds(...)
materialize_cv_folds(...)
load_cv_folds(...)
```

The public behavior remains:

```python
dict[int, tuple[np.ndarray, np.ndarray]]
```

New internal helpers should isolate artifact-format details:

```python
def write_split_artifact_npz(
    frame: pd.DataFrame,
    *,
    validation_split: ValidationSplitConfig,
    pk_columns: tuple[str, ...],
    output_path: Path,
) -> str:
    ...

def load_split_artifact_npz(
    split_set: CVSplitSet,
    frame: pd.DataFrame | None = None,
    *,
    pk_columns: tuple[str, ...] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    ...
```

`frame` is required for compact artifacts because the loader must verify row
identity. Legacy explicit artifacts can still load without a frame.

If the existing `load_materialized_cv_folds(split_set)` API cannot verify compact
artifacts without a frame, it should:

- keep supporting legacy explicit artifacts without a frame;
- raise a clear error for compact artifacts when no frame is supplied;
- have `load_cv_folds(...)` pass the loaded frame into the materialized loader.

That keeps backwards compatibility while making compact artifacts safe.

## SQL and DDL Impact

No SQL DDL is required.

The existing SQL tables already store the artifact pointer and audit metadata:

```text
CV_SPLIT_SET.artifact_uri
CV_SPLIT_SET.artifact_sha256
CV_SPLIT_SET.splitter_params_json
CV_SPLIT_SET.row_order_sha256
CV_SPLIT_SET.row_count
CV_SPLIT_SET.fold_count
CV_FOLD.n_train
CV_FOLD.n_test
DATASET_MANIFEST.pk_columns_json
```

The artifact format is encoded inside the `.npz` file via `split_format`.

No SQL Server re-seed is required. Existing databases remain valid.

Existing `.npz` artifacts remain valid because the loader continues to support
the legacy explicit format.

## Data Flow

During manifest creation:

```text
final model frame
-> validation_split_indices(...)
-> compact assignment array when split is exclusive
-> .npz artifact with row_key_hash and assignment
-> CV_SPLIT_SET artifact_uri/artifact_sha256
-> CV_FOLD fold counts
```

During materialized load:

```text
CV_SPLIT_SET
-> artifact_uri
-> load .npz
-> if legacy explicit format: read fold_N_train_idx/fold_N_test_idx
-> if compact format: verify frame row identity and reconstruct folds
```

During later materialization of a replayable split:

```text
CV_SPLIT_SET replay params + final frame
-> replay_cv_folds(...)
-> compact .npz artifact where supported
-> update CV_SPLIT_SET split_mode/artifact_uri/artifact_sha256
```

## Error Handling

Raise clear `ValueError` messages when:

- compact artifact is loaded without the frame needed for identity validation;
- `split_format` is unknown;
- required arrays are missing;
- `row_key_hash` length does not match `row_count`;
- assignment array length does not match `row_count`;
- artifact `pk_columns` does not match the supplied/manifest PK columns;
- current frame row identity does not match artifact `row_key_hash`;
- `test_fold` contains fold numbers outside `1..fold_count`;
- a fold has no train or no test rows;
- holdout artifact has no train or no test rows;
- artifact hash does not match `CV_SPLIT_SET.artifact_sha256`.

Unknown artifacts should fail closed. If no `split_format` key exists, treat the
artifact as legacy explicit format.

## Offline Validation

This change should be validated without requiring SQL Server.

Add an offline SQLite-style test path that:

1. creates the existing manifest/CV tables in an in-memory or temporary SQLite
   database;
2. builds a small final model frame with configured PK columns;
3. creates a manifest with `materialize=True`;
4. confirms the `.npz` artifact uses the compact format;
5. reloads folds through the same public loader used by no-Docker/direct flows;
6. confirms reconstructed fold counts match `CV_FOLD`.

SQLite is only the offline stand-in. It does not replace SQL Server syntax
validation. Keep existing SQL Server migration parse/syntax checks.

## Testing

Add focused unit tests:

- writes compact k-fold artifacts;
- loads compact k-fold artifacts;
- writes compact train/test holdout artifacts;
- loads compact train/test holdout artifacts;
- writes compact `column_kfold` artifacts;
- writes compact `column_holdout` artifacts;
- legacy explicit artifacts still load;
- artifact SHA checks still run;
- compact artifact rejects missing frame;
- compact artifact rejects wrong PK columns;
- compact artifact rejects row identity mismatch;
- compact artifact rejects invalid fold assignment values;
- compact artifact rejects holdout assignment with no train or no test rows;
- composite PK row identity is stable and checked;
- non-default DataFrame indexes are ignored.

Add integration/offline tests:

- frame-backed manifest with `materialize=True` writes compact artifact and SQL
  metadata;
- `load_cv_folds(...)` reconstructs materialized compact folds from a dataset
  loader;
- `materialize_cv_folds(...)` converts replayable splits into compact artifacts
  where supported;
- no-Docker dry-run still works.

Run standard verification:

```bash
rtk uv run pytest -q
rtk uv run ruff check .
rtk uv run ruff format --check <touched files>
rtk uv run python scripts/no_docker_services.py menu --dry-run
rtk uv run pytest -q tests/test_sql_server_syntax.py
rtk uv run sqlfluff parse --dialect tsql db/migrations
rtk git diff --check
```

## Compatibility

The loader must support both artifact styles:

```text
new compact format:
  split_format present

legacy explicit format:
  no split_format key
  fold_N_train_idx / fold_N_test_idx arrays
```

No existing SQL rows need modification.

No old `.npz` files need rewriting.

If a compact artifact cannot be safely loaded because the frame is unavailable,
the code should raise a clear error telling the caller to provide the final model
frame or use the replay path.

## Implementation Notes

The current code has two materialization paths:

- `pricing_pipeline.data.manifest.write_validation_split_npz(...)`
- `pricing_pipeline.data.cv_splits.materialize_cv_folds(...)`

They should use the same artifact writer so behavior does not diverge.

The current loader:

- `pricing_pipeline.data.cv_splits.load_materialized_cv_folds(...)`

should become format-aware and support legacy explicit artifacts plus compact
assignment artifacts.

The row identity helper should live near `compute_row_order_sha256(...)` so the
same canonical PK normalization is used by both whole-frame row-order hashes and
per-row artifact identity hashes.

The implementation should not add a new config option unless needed. Compact
format should be the default for eligible split methods. Legacy explicit format
remains the fallback for unsupported split shapes and old artifacts.

