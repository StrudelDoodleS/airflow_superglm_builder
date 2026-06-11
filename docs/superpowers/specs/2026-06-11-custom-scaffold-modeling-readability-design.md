# Custom Scaffold Modeling Readability

## Status

Draft design for PR #8. This is a scaffold readability cleanup, not a runtime
architecture change.

## Problem

The default custom pricing model scaffold now has the right high-level shape:

```text
register model
prepare source data
train/export/create frame manifest
publish completed build
```

That shape should stay. It matches the frame-backed manifest design and keeps
production DAGs composable: model teams own source access, final-frame
construction, fitting, validation, and rating export; shared library code owns
SQL catalogue and publish lifecycle plumbing.

The remaining problem is readability. Generated `data.py` is fairly easy to
approach: it has one obvious `prepare_source_data(...)` stub and an optional
`sql/source_data.sql` file. Generated `modeling.py` is technically correct, but
it presents several `def` statements without making ownership obvious.

For a new model author, a function definition is just a function definition.
The scaffold does not clearly say:

```text
edit these functions
usually leave this lifecycle recipe alone
```

That makes the user-facing scaffold feel more like framework internals than a
guided model-building worksheet.

## Reported Issue

Generated `modeling.py` mixes:

```text
editable model stubs
shared lifecycle imports
version/export-id logic
effective/data-as-of parsing
frame sorting
split creation
manifest writing
completed-build payload construction
```

The bottom `train_validate_export_model(...)` function is especially dense. It
exists largely because the SQL catalogue and publish lifecycle need consistent
metadata:

```text
export_id
model_version
effective_from
data_as_of_date
manifest_id
split_set_id
rating_workbook_path
model_artifact_path
metrics
created_by
```

Those are real audit and publish facts, but the scaffold currently exposes them
in a way that can make users unsure what they should change.

## Design Principle

The user should supply modelling facts. The library should translate those
facts into SQL catalogue rows.

For PR #8, do not move the entire model build recipe into a shared helper. The
recipe should remain visible because it shows the important custom flow:

```text
prepared payload
-> source frame
-> final model frame
-> split indices
-> fit/export
-> frame manifest
-> completed-build payload
```

Instead, make the generated file communicate ownership:

```text
model-specific edit points first
standard lifecycle recipe second
```

The result should be easier to read without reintroducing a DAG factory or a
hidden all-in-one lifecycle abstraction.

## Goals

- Make generated `modeling.py` read like a guided worksheet.
- Put model-owned edit points at the top of the file.
- Mark the bottom lifecycle recipe as "usually leave this alone".
- Document the prepared-payload and return contracts in the generated stubs.
- Keep the current custom DAG shape unchanged.
- Keep frame-backed manifest creation visible in the recipe.
- Avoid SQL DDL changes and database re-seeds.

## Non-Goals

- Do not create a new DAG factory.
- Do not move `train_validate_export_model(...)` fully into shared code.
- Do not add `ModelSpec`, `DatasetSpec`, `manifest_sql`, `TRAINING_SQL`, or
  `build_pricing_model_dag(...)` back into the default custom scaffold.
- Do not split the default custom scaffold into more files for PR #8.
- Do not generate runtime scoring SQL, views, stored procedures, or application
  feature code.
- Do not change how packages are published or deployed.

## Desired Generated File Shape

Generated `modeling.py` should keep one file, but structure it into obvious
sections:

```python
"""Model-owned build logic for this pricing model.

Edit the functions in the first section. The final recipe function wires those
pieces into the shared manifest/publish contract.
"""

# Model-local config/constants.
from pricing_models.my_model.data import ...
from pricing_models.my_model.spec import MODEL_CONFIG

# Shared lifecycle helpers. Most model authors do not need to edit these imports.
from pricing_pipeline.data.manifest import ...
from pricing_pipeline.orchestration.completed_build_helpers import ...
from pricing_pipeline.publishing.model_versions import ...


# ---------------------------------------------------------------------------
# Edit These Model-Specific Functions
# ---------------------------------------------------------------------------

def read_prepared_source(prepared: Mapping[str, Any]) -> pd.DataFrame:
    ...


def build_final_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    ...


def fit_validate_export_rating_tables(...) -> tuple[str | Path, str | Path | None, dict[str, float]]:
    ...


# ---------------------------------------------------------------------------
# Standard Build Recipe - Usually Leave This Alone
# ---------------------------------------------------------------------------

def train_validate_export_model(...):
    ...
```

Keep the current function names for PR #8. The readability win should come from
section headers, docstrings, and return-contract comments, not from renaming
public scaffold hooks. Renames can be considered later if there is still user
confusion after this cleanup.

## Model-Owned Edit Points

### `read_prepared_source(...)`

Purpose:

```text
Load the source/prepared data identified by data.py's payload.
```

Input:

```text
prepared
  The merged payload from prepare_source_data_task(...). It includes standard
  run metadata plus whatever data.py returned.
```

Standard wrapper fields available in `prepared`:

```text
run_key
output_dir
effective_from
data_as_of_date
```

Model-owned fields commonly returned by `data.py`:

```text
source_data_path
training_frame_path
staging_table
source_row_count
extract_id
```

The scaffold should be explicit that these model-owned keys are examples, not
framework requirements. The only requirement is that `read_prepared_source(...)`
knows how to use them.

Example stub text:

```python
def read_prepared_source(prepared: Mapping[str, Any]) -> pd.DataFrame:
    """Load the prepared source frame for this run.

    data.py decides the handoff shape. For example, if prepare_source_data(...)
    returned {"source_data_path": ".../source.parquet"}, read that file here.
    Do not pass large DataFrames through Airflow/XCom.
    """
    raise NotImplementedError(...)
```

### `build_final_model_frame(...)`

Purpose:

```text
Create the final pandas model frame used for validation, training, rating-table
export, and frame-backed manifest creation.
```

This is where model authors should put:

```text
target construction
pd.cut/binning
feature engineering
filters
row exclusions
final feature/PK/target/weight column selection
```

The returned frame must contain:

```text
PK_COLUMNS
MODEL_CONFIG.target_name
WEIGHT_COLUMN when configured
all columns needed by fitting/export
any source split column used by validation_split
```

The scaffold should keep the warning that source split columns are validation
metadata, not automatically rating features.

### `fit_validate_export_rating_tables(...)`

Purpose:

```text
Fit/validate the model, export the rating workbook, optionally persist the model
object, and return small artifact metadata.
```

Return contract:

```python
return rating_workbook_path, model_artifact_path, metrics
```

Where:

```text
rating_workbook_path
  Required. Path to the workbook that the publish task stages into SQL.

model_artifact_path
  Optional. Path to a model pickle/joblib or other model object artifact.

metrics
  Optional small numeric metrics, such as row count, deviance, Gini, exposure
  sum, or validation loss. Values must be finite numbers.
```

The scaffold should show that `split_indices` are supplied and should be used
for validation metrics when metrics are reported.

## Standard Build Recipe

`train_validate_export_model(...)` should remain visible, but it should be
introduced as lifecycle glue:

```python
def train_validate_export_model(...):
    """Standard custom-model lifecycle recipe.

    Start by customizing the functions above. Edit this recipe only when your
    model needs a different build flow. The recipe resolves stable publish
    metadata, calls the model-owned functions, creates the frame-backed
    manifest, and returns the CompletedModelBuild payload consumed by the
    publish task.
    """
```

Its behavior should remain the same:

```text
1. Resolve run_key from prepared payload.
2. Build stable export_id from model_key + run_key.
3. Resolve model_version for that export_id.
4. Normalize effective_from and data_as_of_date.
5. Load prepared source data.
6. Build and sort final model frame by PK columns.
7. Compute validation split indices from MODEL_CONFIG.validation_split.
8. Fit/validate/export rating artifacts.
9. Create frame-backed manifest and split metadata.
10. Return completed_model_build_payload(...).
```

The recipe is DDL-aware in the sense that it assembles facts required by the SQL
catalogue. It should not require model authors to understand the underlying
tables to use it.

Add a short comment near the frame sort:

```python
# The manifest and split artifacts use this frame order, so keep ordering
# deterministic and aligned with PK_COLUMNS unless the model deliberately needs
# a different order.
frame = frame.sort_values(list(PK_COLUMNS)).reset_index(drop=True)
```

## Documentation Updates

Update README scaffold guidance to describe the generated `modeling.py` as:

```text
Edit:
  read_prepared_source
  build_final_model_frame
  fit_validate_export_rating_tables

Usually leave alone:
  train_validate_export_model
```

Also document the `prepare_source_data(...)` handoff clearly:

```text
data.py returns a small dict of paths, table names, IDs, or metadata.
airflow_tasks.py merges in run_key, output_dir, effective_from, data_as_of_date.
modeling.py receives that merged dict as prepared.
```

This should directly address confusion about example keys such as
`source_data_path`: they are model-owned handoff names, not framework-required
fields.

## Testing

Update scaffold tests to assert that generated custom `modeling.py` contains:

```text
Edit These Model-Specific Functions
Standard Build Recipe - Usually Leave This Alone
prepared-payload docstring text
rating_workbook_path / model_artifact_path / metrics return contract
source split column warning
```

Keep existing negative assertions:

```text
no MODEL_SPEC in default custom scaffold
no DatasetSpec
no manifest_sql
no TRAINING_SQL
no build_pricing_model_dag
no create_prepared_dataset_manifest_task
```

Run at minimum:

```bash
rtk uv run pytest -q tests/test_scaffold_pricing_model.py
rtk uv run ruff check scripts/scaffold_pricing_model.py tests/test_scaffold_pricing_model.py
rtk uv run ruff format --check scripts/scaffold_pricing_model.py tests/test_scaffold_pricing_model.py
```

Also include the existing generated-scaffold compile/sanity check if available.
Generated multi-line templates are easy to break with indentation or quoting
changes.

The PR should not rename generated hook functions. If implementation discovers a
strong reason to rename them, pause and update this design before proceeding.

## DDL / Re-Seed

No DDL change is required.

No database re-seed is required.

This work changes generated scaffold text and documentation only. It does not
change the frame-backed manifest tables, validation split tables, package
publish tables, or lineage tables.

## Acceptance Criteria

- New custom scaffold users can immediately identify which functions they need
  to edit.
- The standard lifecycle recipe remains visible but clearly marked as
  usually-left-alone.
- The prepared payload contract is documented in generated code and README.
- The rating export return contract is documented in generated code.
- The generated custom scaffold still compiles after generation.
- Existing custom DAG shape remains unchanged:

```text
register -> prepare -> train/export/create-manifest -> publish
```

- No factory-era defaults return to the custom scaffold.
- No SQL DDL, migration, or re-seed is introduced.
