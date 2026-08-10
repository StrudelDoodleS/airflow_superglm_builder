# SuperGLM pricing workbench

This repository provides a notebook-first workflow for ingesting model data,
fitting SuperGLM candidates, recording audit evidence, making optional market
edits, and explicitly deploying an immutable SQL rating package.

The analyst owns visible data and modeling decisions. The library owns generated
identifiers, dataset manifests, validation evidence, artifact hashes, model and
package versions, lineage, semantic duplicate prevention, and deployment
concurrency checks. Airflow is not part of the current approved workflow.

## Notebook workflow

Create a model package:

```bash
uv run python scripts/scaffold_pricing_model.py \
  --model-name CLAIM_FREQUENCY \
  --target-name claim_count \
  --model-label "Claim frequency"
```

The scaffold follows the strict `xx_name_name2.ipynb` naming rule:

```text
pricing_models/claim_frequency/
├── __init__.py
├── 01_data_ingestion.ipynb
├── 02_model_training.ipynb
├── 03_model_editor.ipynb
├── 04_model_deployment.ipynb
└── 99_scratch_work.ipynb
```

The normal path is:

1. `01_data_ingestion.ipynb` reads source data, performs accepted transforms,
   records the data-as-at stamp, and saves one verified model-frame handoff.
2. `02_model_training.ipynb` loads that exact frame, fits and publishes an
   untouched `RAW` model, then optionally loads editor-exported level groupings
   and fits a `ROUTINE_EDIT` model.
3. `03_model_editor.ipynb` is optional. Select a SQL model by name or label,
   display its package versions, choose one or default to latest, open a live
   `EditorSession`, preview it, and publish an `EDITOR_EDIT` child package.
4. `04_model_deployment.ipynb` lists only SQL packages in `PUBLISHED` state,
   opens the exact selected package for review, and explicitly deploys it.
5. `99_scratch_work.ipynb` is for disposable source and feature exploration.
   It can also open a published `RAW` candidate in SuperGLM's editor and export
   its categorical collapses for notebook 02. It cannot replace the governed
   frame, build, publish, or deploy, and its contents do not change the
   fitted-model source checksum.

Only the ingestion/training notebooks contribute notebook source to a fitted
candidate checksum; editor/deployment actions have their own durable evidence.

The reference implementation is under `pricing_models/mtpl_frequency/`.
There is no generated training module, DAG factory, TOML model factory, or
analyst-facing metadata form. The notebooks are the model definition.

## Data-as-at is a dataset version

`data_as_of` is the date through which source data is complete. It is not a
model-fit, publication, effective, or deployment date.

Every build must obtain it from either:

- a configured `PricingModelSpec.data_as_of_column` containing exactly one
  non-null date across the final frame; or
- the explicit `data_as_of=` argument to `build_candidate`.

If both are supplied, they must match. The generated workflow uses a retained,
constant `data_as_of` column so the stamp travels with the notebook handoff.

The manifest records `data_as_of_date`, `data_as_of_column`, dataset/source
names, primary keys, every declared column role, row count, exact ordered-frame
SHA-256, dtypes, column statistics, and runtime hash metadata. Its
`manifest_signature_sha256` binds that dataset snapshot, including the
data-as-at stamp. Validation configuration and exact fold indices have their
own deterministic `split_set_id`; changing validation strategy reuses the
dataset manifest, while changing data or data-as-at creates a new version.

## Visible model definition

The training notebook declares one `PricingModelSpec`:

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

Transforms remain normal Python:

```python
frame["term_offset"] = np.log(frame["term"] / 12.0)
frame["log_density"] = np.log1p(frame["density"])
```

`offset_column` is passed to fitting as stored; the pipeline never logs it.
`offset_source_column` retains upstream rating levels. Fit weight and export
weight are independent and never fall back to one another. SQL scoring expects
the final transformed feature columns and does not reconstruct Python features.

Call `register_model`, `build_candidate`, inspect `candidate.metrics`, and call
`publish_candidate` only when the evidence is acceptable. Save the notebook
first: the model-source checksum reads notebook source from disk while ignoring
execution counts and outputs.

## Temporary editor grouping handoff

Until SuperGLM exposes a public grouping-export API, one compatibility module
owns the private `feature_spec._grouping` access. Notebook code never touches
that attribute directly.

The analyst workflow is:

1. publish the untouched `RAW` candidate from notebook 02;
2. open that exact published RAW package in notebook 99;
3. select levels and use **Collapse and refit** across any categorical features;
4. run `export_level_groupings(...)` once; and
5. rerun notebook 02, which calls `load_level_groupings(...)` and
   `apply_level_groupings(...)` before fitting `ROUTINE_EDIT`.

The ignored `.local/routine_groupings.joblib` file contains the actual
`dict[str, LevelGrouping]` Python objects. Its generated JSON sidecar is readable
integrity and provenance evidence, not an analyst-authored configuration. The
loader verifies the artifact bytes, exact SuperGLM version, Python runtime,
model name, source package and manifest evidence, ordered model-frame checksum,
data-as-at date, feature names, level membership, and group partition. An absent
artifact or an artifact with no real collapse automatically skips
`ROUTINE_EDIT`; there is no manual enable flag.

Only categorical level-collapse decisions are exported by this handoff. Other
experimental coefficient or curve edits made in notebook 99 remain scratch
work and do not leak into the routine-edit fit.

Grouping is Python model behaviour. SQL receives the completed grouped model
output and grouping evidence; it does not interpret or apply grouping rules.

This private bridge is intentionally replaceable. When SuperGLM publishes its
public export/load API, only the compatibility module should change—the scratch
and training notebook calls remain stable.

## Duplicate-model prevention before SQL staging

Immediately before publication, Python parses the completed rating workbook and
its signed publication receipt without writing staging rows. It computes
`model_equivalence_sha256` from final rating semantics:

- base rate, terms, levels, group mappings, term metadata, and relativities;
- numeric values canonicalized to 10 decimal places;
- row ordering made irrelevant; and
- export ID, model version, file path, effective dates, actor/timestamps, and
  other publication-only identity excluded.

Python then performs a read-only lookup using:

```text
model_id + manifest_id + model_kind + model_equivalence_sha256
```

If a successful equivalent model already exists, publication returns its
existing package and model run with `deduplicated=True`. It does not write
`pricing_stg.STG_RATING_EXPORT`, a rate package, or a model run. The provisional
model-version reservation is released.

If an equivalent build is submitted with a different `effective_from`, the
pipeline raises before staging; preserving that release intent requires a
future model-build/rate-package separation rather than silently reusing it.

`RAW`, `ROUTINE_EDIT`, and `EDITOR_EDIT` are deliberately separate semantic
classes, even if two happen to produce the same numbers. A different dataset
manifest—especially a different data-as-at version—is also never silently
merged.

Staging recomputes the same fingerprint, and a filtered SQL unique index guards
the same key. Those are consistency and concurrent-writer backstops; the normal
redundancy decision happens in Python before SQL staging.

## Automatic evidence

Building and publishing records:

- stable SQL model ID and generated trained-model version;
- manifest signature, data-as-at, exact frame checksum, schema and column roles;
- validation configuration, exact folds and materialized split checksum;
- model source checksum and runtime/SuperGLM versions;
- candidate bundle path, format, size and checksum;
- workbook and publication-receipt checksums;
- model kind and semantic equivalence checksum;
- model-run, fold, and scoped metrics;
- immutable package version/status and editor parent lineage; and
- deployment slot, prior champion, selected package, reason, and deployer.

The SQL database is the audit source of truth. `mlflow_run_id` remains optional,
but the current notebook workflow does not create or log MLflow runs.
Analysts never type a model or package version.

New filesystem artifacts use compact run keys and digest-based manifest/split
components so deeply nested `state/` paths remain usable in Windows Explorer.
The shortened folder names are locators only: full model, export, manifest, and
split identities remain in candidate bundles, receipts, and SQL. Previously
recorded long artifact paths remain readable; only newly created paths use the
compact convention.

## Local SQLite mode

Generated notebooks begin with:

```python
DATABASE_MODE = "local"
RUNTIME_MODULE = None
EXPECTED_REMOTE_DATABASE = ""
ALLOW_REMOTE_WRITES = False
```

`connect(mode="local", ...)` creates persistent SQLite databases under the
model's ignored `.local/` directory. Local mode records real manifests, splits,
model runs, metrics, packages, equivalence checks, and artifacts. Packages use
`LOCAL_AUDIT`; local mode does not deploy a live package or run the editor.

## Guarded SQL Server mode

Keep work connectivity in a private importable module. Do not commit
server names, tokens, passwords, or copied connection code.

```python
# work_runtime/database.py
def get_engine(database=None):
    ...

def get_schema_names():
    return {
        "pricing": "python_pricing",
        "pricing_staging": "python_pricing_stg",
        "mlops": "python_mlops",
    }
```

After confirming the target, set:

```python
DATABASE_MODE = "remote"
RUNTIME_MODULE = "work_runtime.database"
EXPECTED_REMOTE_DATABASE = "PricingAudit"
ALLOW_REMOTE_WRITES = True
```

Remote connection executes `SELECT DB_NAME()` and refuses writes unless it
matches `EXPECTED_REMOTE_DATABASE`. Candidate and split artifact roots must be
durable and readable by every later review/publication process.

Apply versioned migrations separately from model notebooks:

```bash
uv run python scripts/apply_schema.py --runtime-module work_runtime.database
```

## Editor and deployment

`open_candidate` verifies package state, one successful model run, manifest and
split lineage, artifact checksum, runtime compatibility, and bundle identity.

The optional editor remains visible:

```python
reviewed = open_candidate(pricing, model=model, package_version=3)
editor_session = EditorSession.from_model(
    reviewed.bundle.fitted_model,
    train_data=(
        reviewed.bundle.X,
        reviewed.bundle.y,
        reviewed.bundle.sample_weight,
        reviewed.bundle.offset,
    ),
    cv_report=reviewed.bundle.cv_report,
)
display(editor_session.widget())
edited_model = editor_session.to_model()
edited = publish_edits(
    pricing,
    candidate=reviewed,
    editor_session=editor_session,
    reason=EDIT_REASON,
)
```

`deploy_package` accepts only a `Candidate` returned by `open_candidate`. It
carries the champion snapshot visible during review, so deployment fails rather
than overwriting a champion that changed concurrently.

## Audit views

- `pricing.V_FINAL_MODEL_RELATIVITY` exposes each final package relativity with
  model ID/kind/equivalence digest, package identity, full manifest identity,
  data-as-at, frame evidence, validation split ID, term and level.
- `pricing.V_MODEL_CANDIDATE_RELATIVITY` exposes published candidates;
  `pricing.V_PUBLISHED_MODEL_RELATIVITY` remains its compatibility alias.
- `pricing.V_CURRENT_DEPLOYED_RELATIVITY` exposes only the currently deployed
  package per model/slot, with deployment identity and timestamps.
- `pricing.V_MODEL_VALIDATION_SPLIT` and
  `pricing.V_MODEL_VALIDATION_SUMMARY` expose validation evidence.
- `pricing.V_MODEL_LINEAGE_REDUNDANCY_CHECK` detects missing, duplicated, or
  mismatched manifest/split links.

Filtered unique indexes enforce manifest and semantic-model deduplication.
Separate grouped “redundancy” views are intentionally omitted because those
indexes make duplicate rows impossible rather than merely reportable.

Edited packages keep parent IDs but do not inherit validation metrics they did
not produce. Future continuous spline SQL export requires a separate exact
scoring contract.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Never commit model-local `.local/` databases, notebook outputs, credentials, or
private work modules.
