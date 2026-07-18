# SuperGLM pricing workbench

This repository provides a notebook-first workflow for fitting and validating a SuperGLM,
publishing immutable SQL rating packages, making optional edits, and deploying one reviewed package.

The analyst owns the visible model, features, transforms, split strategy, and
scoring choices. The library owns generated identifiers, dataset evidence,
checksums, package lineage, retries, and SQL writes. Airflow is not part of the
approved workflow; durable orchestration can call the same functions later.

## Start a pricing model

```bash
uv run python scripts/scaffold_pricing_model.py --model-name CLAIM_FREQUENCY \
  --target-name claim_count --model-label "Claim frequency"
```

This creates `pricing_models/claim_frequency/pricing_model.ipynb` beside a small `__init__.py`.
The bundled reference is `pricing_models/mtpl_frequency/pricing_model.ipynb`.

Work through the notebook in order:

1. Select local or remote database mode in the first cell.
2. Load data and perform visible feature transforms in Python.
3. Declare `PricingModelSpec`, the validation strategy, `FEATURES`, and the ordinary `SuperGLM` object.
4. Call `register_model` and `build_candidate`.
5. Inspect `candidate.validation_metrics` before package publication.
6. Call `publish_candidate` to create or reuse the immutable baseline package.
7. Optionally call `open_candidate`, edit through a visible `EditorSession`, and call `publish_edits`.
8. Optionally call `deploy_package` for the exact package reviewed.

There is no generated training module, DAG factory, TOML model factory, or analyst-facing
metadata form. The notebook is the model definition. Analysts never type a model or package version.

## Keep model decisions visible

The standard scaffold visibly requests all three held-out metrics:

```python
from superglm import Categorical, Numeric, Spline, SuperGLM

SCORING = ("deviance", "nll", "gini")
FEATURES = {
    "area": Categorical(),
    "driver_age": Spline(),
    "vehicle_age": Numeric(),
}
superglm_model = SuperGLM(family="tweedie", features=FEATURES)
```

`PricingModelSpec.scoring` receives `SCORING`; change it deliberately if a model
needs fewer metrics. Feature and offset transforms remain ordinary Python:

```python
frame["log_density"] = np.log1p(frame["density"])
frame["term_offset"] = np.log(frame["term"] / 12.0)

MODEL = PricingModelSpec(
    name="CLAIM_FREQUENCY",
    label="Claim frequency",
    target="claim_count",
    model_type="superglm_poisson",
    deployment_slot="CLAIM_FREQUENCY_UAT",
    features=tuple(FEATURES),
    dataset_name="claim_frequency_model_frame",
    source_system="pricing_sql",
    pk_columns=("policy_id",),
    offset_column="term_offset",
    offset_source_column="term",
    offset_label="log(term / 12)",
    sample_weight_column="model_weight",
    export_weight_column="rating_table_weight",
    data_as_of_column="data_as_of",
    validation=ValidationSplitConfig.kfold(n_splits=5, random_state=42, shuffle=True),
    scoring=SCORING,
)
```

`offset_column` is passed to fitting exactly as stored; the pipeline never logs
it. `offset_source_column` retains upstream values for export, so a model fitted
with `log(term / 12)` can still publish levels 12 and 36. Sample weight affects
fitting, while export weight affects rating-table aggregation. These inputs do
not fall back to one another.

SQL scoring expects the final transformed feature columns. It does not recreate
`log1p(density)` from raw density, so production preparation must apply the same visible
transform. Save the notebook before building because the model source checksum comes from disk.

`data_as_of` is the date through which the input data is complete. It can be explicit or
derived from one constant-valued frame column; it is not a deployment date.

Supported split methods are generated KFold, generated train/test split, column
KFold, and column holdout. Each public validation row represents held-out rows:
KFold and column KFold produce one row per split; train/test and column holdout
produce one. The generic split schema can later represent repeated methods such
as group shuffle, but group shuffle is not implemented now.

## Build, inspect, publish, then deploy

```python
candidate = build_candidate(
    pricing, model=model, frame=frame,
    superglm_model=superglm_model, data_as_of=DATA_AS_OF,
)
display(candidate.validation_metrics)

published = publish_candidate(pricing, candidate)
```

`build_candidate` fits the final model and creates held-out validation evidence.
The displayed DataFrame has one row per validation split with training and
validation counts plus the requested metrics. Inspect it before
`publish_candidate` writes the immutable package and its audit evidence.
Publication does not change a live deployment.

For a market or underwriting edit, publish the baseline first, then use the
public SuperGLM editor directly:

```python
from superglm.editor import EditorSession

reviewed = open_candidate(pricing, model=model, package_version=published.package_version)
editor_session = EditorSession.from_model(
    reviewed.bundle.fitted_model,
    train_data=(reviewed.bundle.X, reviewed.bundle.y,
                reviewed.bundle.sample_weight, reviewed.bundle.offset),
    cv_report=reviewed.bundle.cv_report,
)
display(editor_session.widget())
```

Keep preview and publication in separate cells:

```python
edited_model = editor_session.to_model()  # in-memory preview; no writes

edited = publish_edits(
    pricing, candidate=reviewed,
    editor_session=editor_session, reason=EDIT_REASON,
)
```

Rerun `to_model()` after further widget changes. `publish_edits` verifies that
the session belongs to the opened parent, saves and replays its session JSON,
and creates an immutable child; it does not hide or own the `EditorSession`.
Deployment stays a separate, remote-only cell:

```python
deployment = deploy_package(pricing, package=reviewed, reason=DEPLOYMENT_REASON)
```

`deploy_package` requires a `Candidate` returned by `open_candidate` and uses the
champion snapshot seen during review. It fails rather than overwriting a champion
that changed concurrently.

## Identity, retry, and edit lineage

| Identifier | Meaning |
| --- | --- |
| `model_id` | Stable SQL identity for the registered pricing model. |
| `model_version` | One independently fitted root baseline, such as `v5`. |
| `package_version` | Every immutable package within that root, including edits. |
| `model_run_id` | Technical build/publication record linked to the evidence. |

A material change to data, model source/configuration, features, split geometry,
scoring, or runtime creates a new root fingerprint and `model_version`. An
identical rerun reuses the canonical root package, model run, and versions after
checking the stored evidence; timestamps and attempt paths do not manufacture a
new version.

An edit keeps the root `model_version`, receives a new `package_version`, and
records the direct edited package parent. Root validation evidence is `DIRECT`.
An edited child (including a chain of edits) points to its root validation source
and is `INHERITED_FROM_PARENT`; it is not described as newly cross-validated.
Its final relativity rows show the edited model while its validation rows show the
baseline evidence. New features, transforms, spline definitions, family, or fit
settings require another root build.

The framework automatically records the data-as-of date, primary-key columns,
ordered frame metadata and SHA-256, validation method and membership, model source checksum,
candidate bundle and runtime identity, model-run and fold metrics (presented publicly as
validation splits), curve status, package checksums, edited package parent/reason, and deployment
history. The SQL database is the audit source of truth for these records.
The current notebook workflow does not create or log MLflow runs.

## Analyst SQL views

| View | Row grain and meaning |
| --- | --- |
| `pricing.V_FINAL_MODEL_RELATIVITY` | One package final-model term level/cell. `model_fit_scope='PACKAGE_FINAL_MODEL'`; edited packages show their actual edits, never split estimators. |
| `pricing.V_MODEL_VALIDATION_SPLIT` | One package/model run and held-out validation split, including counts, lineage, deviance, NLL, and Gini. |
| `pricing.V_MODEL_VALIDATION_SUMMARY` | One package/model run with split count, validation coverage, curve status, and mean/population-SD metrics. |
| `pricing.V_MODEL_VALIDATION_SPLIT_RELATIVITY` | One package, validation split, term, and curve point from the model fitted on that split's training rows. `model_fit_scope='VALIDATION_TRAINING_SPLIT_MODEL'`. |
| `pricing.V_CURRENT_DATASET_VALIDATION_SPLIT` | One latest dataset manifest and split-geometry row; it deliberately contains no model metrics. |

Deviance, negative log-likelihood (`nll`), and Gini are the exact held-out values
returned by the pinned SuperGLM runtime, not locally reimplemented statistics.
The summary takes the arithmetic mean and population standard deviation across split values;
one split has SD zero. A metric intentionally omitted from `SCORING` is `NULL` in the views.

Validation curves cover comparable one-dimensional categorical/ordered,
numeric, polynomial, and spline main effects. Split estimators remain transient;
SQL stores normalized points on a shared domain and common deterministic
reference. `eta_contribution` is the reference-relative link-scale value. For a
log link, `relativity = exp(eta_contribution)`; for every non-log link,
`relativity` is `NULL` and the link contribution remains available.

Interactions are multidimensional surfaces and offsets are external additions,
so neither is represented as a validation curve. Unsupported or malformed curve
sets record `UNAVAILABLE` with a reason and zero points; valid metrics and the
final package can still publish without partial curve evidence.

## Local and remote databases

The generated notebook starts locally:

```python
DATABASE_MODE = "local"
RUNTIME_MODULE = None
EXPECTED_REMOTE_DATABASE = ""
ALLOW_REMOTE_WRITES = False
```

Local mode creates and upgrades persistent SQLite databases under the model's ignored `.local/`
directory. It publishes audit rows with `LOCAL_AUDIT` status and can query all five views.
`V_FINAL_MODEL_RELATIVITY` is empty locally because SQLite publication
does not compile local rating tables. Local mode also rejects `open_candidate`, editor
publication, and deployment immediately; it does not deploy a live package. Deleting `.local/`
is a destructive local reseed: reconnect to recreate
the current schema and rerun the notebook to repopulate evidence.

For remote SQL Server, keep connectivity in a private importable runtime module.
Do not commit
server names, tokens, passwords, copied connection code, or work modules.

```python
# work_runtime/database.py -- private work code; interface only
def get_engine(database=None):
    ...  # return the SQLAlchemy engine that already works at work

def get_runtime_settings():
    return {
        "pricing_database": "PricingAudit",
        "workbench_artifact_root": "/mnt/approved_backed_up_pricing/workbench",
        "validation_split_artifact_root": "/mnt/approved_backed_up_pricing/splits",
        "mlflow_enabled": False,
        "skip_database_create": True,
    }
```

Use the most durable approved, backed-up filesystem available and an absolute
path readable by every process that opens or publishes a candidate. Then enable
remote writes only after confirming the target:

```python
DATABASE_MODE = "remote"
RUNTIME_MODULE = "work_runtime.database"
EXPECTED_REMOTE_DATABASE = "PricingAudit"
ALLOW_REMOTE_WRITES = True
```

Remote connection runs `SELECT DB_NAME()` and refuses writes unless it exactly
matches `EXPECTED_REMOTE_DATABASE`. Apply forward SQL Server migrations without
putting credentials in the command:

```bash
uv run python scripts/apply_schema.py --runtime-module work_runtime.database
```

The command records migration checksums, skips already-applied files, and rejects
changed migration history. Use `scripts/reset_remote_pricing_schema.py` only for
an explicitly disposable database; it defaults to a guarded dry run and requires
the expected database plus explicit execution and confirmation flags.

## Artifact durability boundary

Candidate joblib bundles and editor session JSON remain under
`WORKBENCH_ARTIFACT_ROOT`. SQL stores their paths, sizes, hashes, formats, runtime
identity, lineage, and final normalized rating tables; SQL is not an object store.

If that filesystem is lost, deployed SQL rating tables and lineage remain, but
the exact Python candidate cannot be safely reopened, an unpublished editor
session cannot be resumed, and a retry requiring missing bytes fails
verification. A repository checkout, transient WSL directory, or model `.local/`
folder is not a durable remote artifact root. Do not place a SQLite or MLflow
database on a OneDrive/SharePoint-synced path; file syncing does not provide safe
concurrent database writes.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Never commit model-local `.local/` databases, notebook outputs, credentials, or
work connection modules.
