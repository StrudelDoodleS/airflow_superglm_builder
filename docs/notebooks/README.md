# Notebook workflow and functions

This is the analyst-facing reference. The notebooks contain data and model
decisions; `pricing_pipeline.notebook` handles identifiers, evidence, SQL
writes, artifacts, publication, and deployment guards.

## Workflow boundaries

| Notebook | Reads | May write | Must not do |
|---|---|---|---|
| `01_data_ingestion.ipynb` | Source data | Verified model-frame artifact | Fit or publish a model |
| `02_model_training.ipynb` | Exact model frame; optional grouping artifact | Manifest, split evidence, run, metrics, candidate, package | Deploy |
| `03_model_editor.ipynb` | Published SQL candidate and bundle | `EDITOR_EDIT` child run/package | Open a draft or deploy |
| `04_model_deployment.ipynb` | Published SQL candidate and current champion | Deployment history/current pointer | Fit or edit |
| `99_scratch_work.ipynb` | Any exploratory source; published `RAW` for grouping work | Ignored local grouping artifact only | Build, publish, or deploy |

Accepted scratch data work moves to notebook 01. Accepted model choices move to
notebook 02. Scratch cells are excluded from model-source identity.

## Scaffold configuration

Copy `pricing_scaffold.example.toml` to `pricing_scaffold.toml` at the scaffold
root:

```toml
[notebook_defaults]
database_mode = "remote"
runtime_module = "work_runtime.database"
expected_remote_database = "PricingAudit"
```

```bash
uv run python scripts/scaffold_pricing_model.py \
  --model-name CLAIM_FREQUENCY \
  --target-name claim_count
```

Precedence is command line, explicit `--config`, auto-discovered
`<root>/pricing_scaffold.toml`, then built-in local defaults. Unknown sections
or keys fail fast. `ALLOW_REMOTE_WRITES` cannot be set in TOML.

## Connection guard

Generated notebooks expose four obvious settings:

```python
DATABASE_MODE = "local"  # or "remote"
RUNTIME_MODULE = None  # e.g. "work_runtime.database"
EXPECTED_REMOTE_DATABASE = ""
ALLOW_REMOTE_WRITES = False
```

`connect(...)` creates persistent SQLite databases in local mode. In remote
mode it imports the private runtime module, runs `SELECT DB_NAME()`, rejects a
database-name mismatch, and keeps mutation disabled until
`ALLOW_REMOTE_WRITES = True`.

The private runtime module supplies connectivity without putting secrets in the
repo:

```python
def get_engine(database=None):
    ...

def get_schema_names():
    return {
        "pricing": "python_pricing",
        "pricing_staging": "python_pricing_stg",
        "mlops": "python_mlops",
    }
```

## Model specification

`PricingModelSpec` is the one visible declaration shared by ingestion and
training:

```python
MODEL = PricingModelSpec(
    name="CLAIM_FREQUENCY",
    label="Claim frequency",
    target="claim_count",
    model_type="superglm_poisson",
    deployment_slot="CLAIM_FREQUENCY_UAT",
    features=("driver_age", "vehicle_age", "region"),
    dataset_name="claim_frequency_model_frame",
    source_system="pricing_sql",
    pk_columns=("policy_id",),
    offset_column="term_offset",
    offset_source_column="term",
    offset_label="log(term / 12)",
    sample_weight_column="model_weight",
    export_weight_column="rating_table_weight",
    data_as_of_column="data_as_of",
    validation=ValidationSplitConfig.kfold(
        n_splits=5,
        random_state=42,
        shuffle=True,
    ),
)
```

The frame must contain every declared column. Structural roles cannot overlap.
The offset is passed to SuperGLM as stored; the pipeline does not log it.
Sample weight and rating-table export weight are independent.

## Public notebook functions

Import these from `pricing_pipeline.notebook`.

| Function | Use | Main result or guard |
|---|---|---|
| `connect(...)` | Open local SQLite or guarded remote SQL | `NotebookContext` |
| `save_model_frame(frame, path, replace=False)` | Atomically hand off notebook 01 output | Joblib artifact plus JSON receipt |
| `inspect_model_frame(path)` | Read frame evidence without loading the frame | `ModelFrameArtifact` |
| `load_model_frame(path)` | Verify byte and frame hashes, then load | `pandas.DataFrame` |
| `register_model(pricing, spec, source_root=...)` | Create or validate stable model identity | `RegisteredModel` |
| `build_candidate(pricing, model=..., frame=..., superglm_model=..., model_kind=...)` | Fit and derive all evidence | `BuiltCandidate`; inspect `.metrics` |
| `publish_candidate(pricing, candidate)` | Publish the completed candidate | IDs, paths, status, `deduplicated` |
| `load_registered_model(...)` | Resolve one active SQL model by name/label | Review-only `RegisteredModel` |
| `list_candidate_versions(...)` | List published packages newest first | Friendly or technical DataFrame |
| `open_candidate(...)` | Verify and load one exact published package | `Candidate` with bundle and champion snapshot |
| `publish_edits(...)` | Save and publish an editor session | `EDITOR_EDIT` child publication |
| `deploy_package(...)` | Deploy exactly the reviewed candidate | Deployment record; stale champion fails |

`register_model`, `build_candidate`, `publish_candidate`, `publish_edits`, and
`deploy_package` call the context write guard. Editor and deployment functions
require remote mode.

## Data-as-at and manifest identity

`data_as_of` is the date through which source data is complete. Keep a constant,
non-null date column in the governed frame and set `data_as_of_column`. An
explicit `data_as_of=` may be used instead; if both exist, they must match.

The manifest records the date and column name, dataset/source names, primary
keys, column roles, row count, ordered-frame SHA-256, dtypes, statistics, and
runtime hash metadata. Changing data or data-as-at creates a new manifest.
Changing validation configuration or exact split indices creates a new split
set under the same manifest.

## Raw and routine grouping flow

Until SuperGLM provides a public grouping export API, the workbench owns one
isolated compatibility bridge to its private grouping object:

1. Publish the untouched `RAW` candidate in notebook 02.
2. Open that published RAW candidate in notebook 99.
3. Use `EditorSession` to collapse any levels across any categorical features.
4. Call `export_level_groupings(candidate, editor_session=..., path=...)`.
5. Notebook 02 calls `load_level_groupings(...)` and
   `apply_level_groupings(...)`, then fits `ROUTINE_EDIT`.

The ignored Joblib artifact stores the actual `dict[str, LevelGrouping]` Python
objects. Its JSON sidecar is readable integrity/provenance evidence, not a
hand-edited grouping config. Loading checks SuperGLM/Python versions, model,
source package, manifest, frame hash, data-as-at, feature names, levels, and the
group partition. Missing or no-op groupings skip the routine-edit build.
Grouping artifacts are deliberately tied to the exact SuperGLM version; after
an upgrade, reopen the RAW candidate in notebook 99 and export them again.

Grouping is Python model behaviour. SQL receives completed relativities and
evidence; it does not execute grouping rules.

## Publication and duplicate handling

Immediately before SQL staging, Python fingerprints final rating semantics:
base rate, terms, levels, group mappings, metadata, and relativities. Numbers
are canonicalized to 10 decimal places and row order is ignored.

The lookup key is:

```text
model_id + manifest_id + model_kind + model_equivalence_sha256
```

An equivalent successful build reuses the existing run/package and returns
`deduplicated=True`; it does not create staging rows. A different manifest or
model kind remains distinct. A different requested effective date raises
instead of silently discarding release intent.

## Artifact locations

Generated notebooks keep ignored local handoffs below the model directory.
New build folders use compact run keys and short digest components to remain
usable in Windows Explorer. Full identities remain inside receipts, bundles,
and SQL.
