# Scaffolded Candidate Workbench Prototype Design

## Purpose

Prototype a simple analyst workflow that proves a newly scaffolded pricing model can:

1. run as a scheduled Airflow model build;
2. produce an immutable, non-live SQL candidate and a reopenable fitted SuperGLM artifact;
3. be selected from a Jupyter notebook without exposing lineage identifiers;
4. open in the existing SuperGLM editor with its retained evaluation context;
5. save an auditable edit session and publish an edited child candidate; and
6. continue to use the existing deployment DAG for any live pointer change.

The prototype tests the user experience and the artifact boundary. It does not build a
multi-user application or solve shared enterprise artifact storage.

## Design Principles

- The default output of `scripts/scaffold_pricing_model.py` is the product blueprint.
  Demo packages such as MTPL remain regression examples, not the analyst contract.
- `scripts/scaffold_pricing_model.py` is the only supported model/DAG authoring path for
  this prototype. The legacy generic DAG builder remains untouched and unused.
- Analysts own source access, frame construction, SuperGLM configuration, and validation
  decisions. Shared code owns versions, manifests, artifact hashes, receipts, publication,
  and Airflow handoffs.
- Airflow does not wait for a human. Scheduled fitting and editor-derived publication are
  separate DAG runs.
- SQL stores audit and lookup metadata. A serialized artifact stores the fitted Python
  object. SQL metadata is not used to reconstruct a SuperGLM model.
- Successful scheduled packages are candidates, not live models. The existing deployment
  pointer determines the champion.
- The prototype uses ordinary pandas transforms. It introduces no transform expression
  language or general inverse-transform framework.

## Prototype Scope

### In scope

- Improve the default custom scaffold so its normal SuperGLM fit/CV/export path works
  without every analyst rebuilding lifecycle plumbing.
- Use the exact folds evaluated by `superglm.cross_validate()` for the existing split
  manifest and artifact.
- Persist the run-level and fold-level CV metrics already carried by the build result,
  including their evaluation scope, instead of leaving the production metric tables
  empty.
- Persist a model artifact path and SHA-256 against the successful model run.
- Provide read-only candidate history and candidate loading utilities for a notebook.
- Provide a generic notebook showing candidate selection and the real SuperGLM editor.
- Save the editor session JSON and edited fitted model artifact.
- Publish an edited workbook as a child of the selected package while preserving the
  original dataset and split lineage.
- Exercise the existing manual deployment DAG after child publication.
- Demonstrate one explicit model-local `LogDensity` to raw `Density` source-axis review
  workbook in the generated reference model, while keeping the operational export in
  prepared-feature space.

### Out of scope

- A custom web application, Airflow UI plugin, or notebook-owned workflow database.
- Shared object storage, MLflow as a required service, or cross-Cloud-PC artifact access.
- Automatic champion promotion or a general champion/challenger policy engine.
- Arbitrary transform serialization, transform parsing, or automatic inversion.
- Raw-feature SQL input transforms. The current SQL scorer has no binding from `Density`
  to model input `LogDensity`.
- Editing interaction terms; the current SuperGLM editor supports one-dimensional main
  effects.
- Removing TOML. The prototype will reveal whether the small housekeeping file earns its
  place; changing discovery is a separate decision.
- Production authorization or two-person approval rules.
- Refactoring, extending, or routing new models through the legacy generic DAG builder.

### Runtime shape

The prototype targets the current no-Docker setup. Airflow scheduler/webserver, Jupyter,
and the model code run as ordinary processes in the same Windows Cloud PC environment and
share one configured artifact root. SQL Server remains the durable audit store. The
SuperGLM editor dependencies are installed in that Python environment through the
upstream `editor` extra; the design introduces no editor container or new long-running
application. The notebook calls the existing Airflow webserver API behind a helper, so an
analyst neither starts services nor uses an Airflow CLI.

### New components

The prototype adds a small, explicit set of components rather than a generic service
framework:

- `pricing_pipeline/modeling/standard_superglm.py`: `ModelInputs` and the shared standard
  fit/CV/export runner;
- `pricing_pipeline/workbench/`: `Workbench.from_runtime()`, candidate history/loading,
  verified bundle handling, editor launch, and Airflow submission status;
- `pricing_pipeline/publishing/editor_candidate.py`:
  `publish_editor_candidate()` for workbook-derived child publication;
- `dags/pricing_publish_editor_candidate.py`: the manual publication DAG;
- `tutorials/scaffolded_candidate_workbench.ipynb`: the analyst walkthrough; and
- the next SQL migration: model-run artifact columns, one-run-per-package enforcement,
  metric persistence support, and a non-live package scorer.

`Settings` gains `workbench_artifact_root`, `airflow_api_url`, and a redacted
`airflow_api_token` (or equivalent custom-runtime credential provider) for prototype
Airflow API authentication. The Python environment adds the pinned SuperGLM editor
extra plus direct `joblib` and `httpx` dependencies; the helper does not
rely on scikit-learn or Airflow transitive packages. `scripts/scaffold_pricing_model.py`
generates imports and hooks against these public components. All names in this subsection
are proposed APIs; they do not exist in the current repository yet.

## Analyst Experience

### Metadata the analyst does not enter

The analyst declares modeling semantics once in code; each run derives the audit record:

| Audit field | Automatic source |
| --- | --- |
| Model version | Existing `resolve_model_version_for_export()` SQL allocator, idempotent by export ID |
| Package version and parent | Existing package writer and selected parent candidate |
| Export/run identity | Airflow DAG/run context |
| Data as-of and row PKs | Prepared-source result plus the model's declared PK columns |
| Dataset and column facts | Exact sorted final frame used by CV and fitting |
| Sample-weight/offset presence | The actual `ModelInputs` passed to SuperGLM |
| CV split lineage | Fold indices returned by the actual `cross_validate()` call |
| Metrics and telemetry | SuperGLM CV result and fitted model telemetry |
| Code/artifact integrity | Shared source and artifact hashers |
| Analyst identity and timestamps | Airflow/notebook runtime identity and SQL timestamps |

The model author still makes real decisions such as what “data as-of” means, which
columns are PKs, and whether an offset is part of the model. They encode each decision
once in the model package; they do not copy IDs, versions, hashes, or presence flags into
every run.

### Create and finish a model

The analyst starts with the existing scaffold command. The generated package keeps the
current file boundaries, but `modeling.py` presents a smaller editable section:

```python
FIT_MODE = "fit_reml"
CV_SCORING = ("deviance",)


def read_prepared_source(prepared) -> pd.DataFrame:
    return pd.read_parquet(prepared["source_data_path"])


def build_final_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.loc[raw["Exposure"] > 0].copy()
    if (frame["Density"] <= 0).any():
        raise ValueError("Density must be positive before the log transform")
    frame["LogDensity"] = np.log(frame["Density"])
    return frame


def build_training_inputs(frame: pd.DataFrame) -> ModelInputs:
    exposure = frame["Exposure"].astype(float).rename("Exposure")
    return ModelInputs(
        X=frame.loc[:, FEATURE_COLUMNS].copy(),
        y=frame[TARGET_COLUMN].to_numpy(dtype=float),
        sample_weight=exposure,
        offset=None,
        export_weight=exposure,
    )


def build_model() -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "LogDensity": Spline(n_knots=8),
            "Area": Categorical(),
        },
    )


def validation_splitter(frame: pd.DataFrame):
    return KFold(n_splits=5, shuffle=True, random_state=42)
```

`ModelInputs` is a small value object matching the real SuperGLM boundaries:

```python
@dataclass(frozen=True)
class ModelInputs:
    X: pd.DataFrame
    y: np.ndarray
    sample_weight: pd.Series | np.ndarray | None = None
    sample_weight_name: str | None = None
    offset: pd.Series | np.ndarray | None = None
    export_weight: pd.Series | np.ndarray | None = None
    export_weight_name: str | None = None
```

It is not a recipe engine. Omitted weight and offset values remain `None`, and shared code
records those facts automatically. The generated no-offset path also creates
`OffsetExportContract(handling="NONE")` without asking the analyst to declare it. A model
that supplies an offset must provide its existing model-owned export options and
`OffsetExportContract`; shared code passes the offset through fit, CV, editor context, and
export but does not guess whether it represents an exported factor or exposure already
applied upstream. Weight names are inferred from a named pandas Series; an analyst using a
bare array supplies its name once in `ModelInputs` so the receipt can distinguish fit and
portfolio/export weighting.

Shared receipt/package metadata records `fit_sample_weight_used`,
`fit_sample_weight_name`, `export_weight_used`, `export_weight_name`, and
`fit_used_offset`, alongside the existing offset contract. These values come from the
actual fit/export arguments; absent values are recorded as false/null rather than
`UNKNOWN`.

The generated standard recipe calls one shared runner with these functions. Advanced
models may replace the generated fit/export function, but the default is complete and
uses the public SuperGLM API. The analyst chooses the splitter, fit mode, and requested
scores in Python; shared code executes the repeated fold loop, forces `return_oof=True`,
and performs the audit writes. The shown Python splitter is paired with
`method = "custom"` and `materialize = true` in the generated TOML. If a model instead
selects a built-in TOML split method, shared code constructs that splitter and does not
call the Python override, preventing two split definitions from diverging.

### Review weekly candidates

The generated DAG remains explicit and can be scheduled by editing its `schedule` value.
Each successful run publishes an immutable package but does not move a deployment pointer.

The generic notebook lives at `tutorials/scaffolded_candidate_workbench.ipynb` and starts
with:

```python
from pricing_pipeline.workbench import Workbench

workbench = Workbench.from_runtime()
history = workbench.candidates("MY_MODEL")
display(history)

candidate = workbench.open(
    "MY_MODEL",
    package_version=int(history.iloc[0]["Package"]),
)
candidate.editor()
```

The visible history is deliberately small, for example:

| Package | Fitted | Data through | Parent | State | Baseline pooled CV deviance | Editor train Δ | Editor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `13` | 10 Jul 2026 | 30 Jun 2026 | - | Candidate | 0.482 | - | Ready |
| `12` | 03 Jul 2026 | 23 Jun 2026 | `11` | Edited candidate | parent: 0.482 | +0.009 | Ready |
| `10` | 26 Jun 2026 | 16 Jun 2026 | - | Champion in production | 0.487 | - | Ready |

The analyst selects a friendly package version, not a SQL identifier. Artifact IDs,
manifest IDs, split IDs, and hashes are excluded from the default frame; they are
available through an explicit `workbench.candidates("MY_MODEL", technical=True)` call for
support and audit work. Package versions remain the existing integer sequence. Champion
state is evaluated in the model's configured deployment slot by default, or in an
explicit `deployment_slot=` supplied to `candidates()`.

Legacy/manual packages may appear with `Editor = Unavailable` and a concise reason. An
editor-openable row must resolve to exactly one successful `MODEL_RUN` with a verified
candidate artifact; zero or multiple matches raise `CandidateLineageError` rather than
silently choosing a run.

### Submit edits

After editing, the notebook calls:

```python
submission = candidate.submit_edits(
    reason="Market calibration of sparse high-age relativities",
)
```

For the prototype, this helper saves a local submission bundle and triggers the manual
`pricing_publish_editor_candidate` DAG through Airflow's API. The notebook does not write
package rows itself. The bundle contains:

- parent rate package ID;
- editor session JSON path and hash;
- authoritative edited model path and hash created from the live editor session;
- model run, dataset manifest, and split set references inherited from the parent; and
- analyst reason and claimed identity supplied by the authenticated Airflow trigger when
  available, otherwise a clearly labelled prototype audit identity rather than an
  authorization guarantee.

The publication function re-exports the edited model, stages the workbook, creates an
immutable child package, and writes editor lineage into revision metadata. It never
deploys the package. `submission.status()` returns a friendly queued/running/published or
failed result. Once published,
`submission.request_deployment(reason="Approved market calibration")` triggers the
existing deployment DAG with the resolved child package and configured slot, so the
analyst does not copy a package ID into Airflow.

## Shared Standard Runner

The shared runner performs the monotonous lifecycle once:

1. load and deterministically sort the final frame by declared PK columns;
2. hash the model package's sorted `.py`, `.sql`, and `.toml` source files;
3. build `ModelInputs`;
4. obtain the model-owned splitter;
5. call `superglm.cross_validate()` with `sample_weight`, `offset`, the chosen fit mode,
   `return_oof=True`, and `error_score="raise"`;
6. reject non-converged folds, duplicate test-row membership, and invalid indices; record
   OOF coverage explicitly, then retain `CrossValidationResult.fold_indices` as the
   authoritative audited folds;
7. fit a fresh model on the full intended training frame using the same fit mode;
8. capture `model.training_telemetry()` and adapt CV evidence into a JSON-safe primitive
   dictionary containing fold, mean, pooled, and standard-deviation scores plus explicit
   scope labels;
9. export and hash the canonical rating workbook using `export_weight` when supplied,
   otherwise the fit sample weight, and build/hash the matching publication receipt,
   automatically using the no-offset contract when `offset is None`;
10. invoke the model's optional `write_review_workbook()` hook to create analyst-only
    presentation artifacts that are never used for staging;
11. create the existing dataset/split manifest from the exact final frame and fold
    indices;
12. serialize the candidate bundle with the fitted model, model inputs, compact CV
    evidence, `manifest_id`, `split_set_id`, declared PKs, row-order hash, and ordered-PK
    fingerprint, then hash the completed artifact; and
13. return an extended completed-build payload with candidate artifact metadata and
    run-level/per-fold metrics carrying `cv` scope.

This requires deliberate extensions to `CompletedModelBuild`, `ModelExportResult`,
`publish_completed_model_build()`, `publish_model_export()`, and `record_model_run()`.
The new payload fields are candidate artifact path/hash/format/size, Python version,
SuperGLM version, model-source SHA-256, and compact metric records. The current code's
`model_artifact_path` and `metrics` fields are not treated as sufficient merely because
they already exist: the production publication path must actually persist the new values.

After package publication assigns `model_run_id`, the production lineage writer upserts
those metrics into `mlops.MODEL_RUN_METRIC` and `pricing.CV_FOLD_METRIC`. Metric persistence
is part of the same idempotent run handoff; analysts do not enter metric names or scopes.
Candidate history uses `pooled_scores["deviance"]`, not an ambiguous fold mean. Uncovered
rows are permitted for forward/temporal validation but are reported as partial OOF
coverage; the generated K-fold reference covers each row exactly once. Repeated-CV folds
with duplicate test membership require a custom runner because SuperGLM's single OOF
array would overwrite predictions.

The scaffold continues to support a custom split callback. A tiny frozen splitter adapter
exposes existing positional `(train_idx, test_idx)` pairs through the `.split()` interface
expected by SuperGLM; this prevents the audit split and evaluated split from diverging.

## Artifact and SQL Boundary

The single-host prototype writes artifacts beneath the existing state root:

```text
state/workbench_artifacts/<model_name>/<export_id>/
  candidate_bundle.joblib
  rating_tables.xlsx
  publication_receipt.json
  rating_tables_review.xlsx  # optional model-local review artifact
  editor_session.json        # editor-derived submission only
  edited_model.joblib        # editor-derived submission only
```

`candidate_bundle.joblib` is one trusted dictionary containing the fitted model, `X`, `y`,
optional fit sample weight, optional offset, export weight, compact CV report, and the
resolved offset export contract. It also binds the manifest/split IDs, declared PKs,
row-order hash, ordered-PK fingerprint, model-source SHA-256, and optional review-artifact
path/hash. The compact CV report retains fold metrics, fold indices, and OOF predictions
but not fitted fold estimators. This is intentionally self-contained for the single-host
prototype. It is not the final storage format for large or sensitive work datasets.

SQL stores artifact location and integrity, not a reconstruction of the Python object.
A new migration adds direct nullable columns to `pricing.MODEL_RUN` for the candidate
artifact path, SHA-256, serialization format, byte size, Python version, and SuperGLM
version, plus the automatically calculated model-source SHA-256. Editor artifacts are
recorded in the child package's revision metadata. The prototype does not add a general
artifact table or artifact service.

The source hash uses normalized relative paths plus file bytes for the model package's
sorted `.py`, `.sql`, and `.toml` files, excluding caches and generated artifacts. A
filtered unique index on non-null `MODEL_RUN.rate_package_id` enforces at most one run per
package after a migration preflight rejects any existing duplicates. Legacy/manual
packages with no model run remain valid, but are not editor-openable.

The migration also adds `pricing.PREDICT_RATE_PACKAGE`, a package-ID variant factored from
the current deployed-package scorer. It accepts the same prepared-feature JSON and
exposure but resolves the explicitly selected DRAFT or PUBLISHED package instead of a live
deployment pointer, querying package tables rather than current-deployment views. Editor
publication uses it for pre-publication Python/SQL parity; it does not change deployment
state and it does not add raw-feature transforms.

`Workbench.open()`:

1. resolves exactly one successful package/model-run lineage or raises
   `CandidateLineageError`;
2. reads only an artifact produced by this pipeline;
3. verifies path policy, format, byte size, and SHA-256 before trusted `joblib`
   deserialization;
4. verifies the recorded Python and SuperGLM versions against the running environment;
5. obtains evaluation data and compact CV evidence from the verified candidate bundle;
   and
6. creates `EditorSession.from_model()` with compact CV evidence.

The concrete session call is:

```python
session = EditorSession.from_model(
    model,
    train_data=(X, y, sample_weight, offset),
    cv_report=cv_report_json,
)
widget = session.widget()
```

`cv_report_json` contains primitives only; the SuperGLM `CrossValidationResult` object is
never passed directly to the iframe. `Candidate.editor()` retains this exact live session
and widget for `submit_edits()` and exposes `close_editor()` to stop the localhost editor
server. Creating a second session at submission time would discard the analyst's edits.
This localhost iframe behaviour is why the same-host Cloud-PC runtime is an explicit
prototype assumption.

If the artifact is missing, tampered with, or incompatible, the candidate remains
inspectable from SQL but the editor is unavailable with a precise error. A production
follow-up must choose between reproducible as-of reloading and a separately governed frame
artifact before using this with large or sensitive work datasets.

## Editor-Derived Publication

The existing cell-only manual revision API is not sufficient because the SuperGLM editor
can alter curve coefficients and perform structural refits. The prototype adds a focused
editor-derived export path:

1. load and verify the parent artifact;
2. in the notebook's live editor process, call `session.save(json_path)` for the operation
   audit, materialize with
   `session.to_model(X=X, y=y, sample_weight=sample_weight, offset=offset)`, and serialize
   that current edited fitted model as the authoritative artifact;
3. in the publication DAG, verify both hashes and load the edited model rather than trying
   to recreate structural refits from JSON alone;
4. export through the same model-owned publication path used by the parent, including its
   optional analyst review workbook;
5. build a new publication receipt from the edited fitted object;
6. write/hash a complete candidate bundle containing the edited fitted model and inherited
   evaluation/manifest/split context; and
7. call the new `publish_editor_candidate()` path with the bundle metadata and
   `parent_rate_package_id` set to the selected package.

In one publication transaction, that API creates the immutable child, creates a new
`pricing.MODEL_RUN` for the editor-publication DAG pointing to the child and edited
candidate bundle, copies the parent run's `MODEL_RUN_DATASET` and
`MODEL_RUN_SPLIT_SET` associations, writes scoped comparison metrics, and stores
edit-session path/hash, parent run ID, submission ID, and analyst reason in revision
metadata. It never tries to attach two packages to the original run.

`publish_editor_candidate()` validates that the parent exists, belongs to the same
model, and is `PUBLISHED`. Under the existing package lock it creates a DRAFT child,
stages/compiles the edited workbook, validates it with the non-live package scorer, assigns
the next integer package version, records the parent and revision metadata, and finalizes
the child as `PUBLISHED` in one transaction. This is a distinct API because the current
training writer hardcodes null parent/revision fields and the current manual revision path
can edit cells only. An existing submission is idempotently reusable only when parent ID,
editor-session hash, edited-model hash, receipt hash, model version, and effective dates
all match. Its export ID and artifact directory are deterministically derived from the
submission ID, so an Airflow retry addresses the same files and SQL identity.

Pinned SuperGLM session JSON does not contain all evaluation data or every refitted model
state. It is therefore an audit of operations, not the authoritative replay artifact for
collapse/refit/distribution edits. `edited_model.joblib` is authoritative; publication
fails if it is missing, incompatible with the parent baseline, or not exportable. A later
upstream session-persistence improvement may make those operations fully replayable
without changing this audit boundary.

Publishable operations are those represented by the session's current in-force model and
accepted by the normal exporter/receipt checks, including coefficient edits and a
materialized categorical-collapse refit. The editor's fixed-offset refit is a separate
diagnostic and is not substituted for the model returned by the explicit `to_model()`
call above.

The child receives a new package version but keeps the parent's trained `model_version`
and package effective dates; the edit is a governed derivation of that fit, not a claim
that a new training run occurred. The eventual deployment's effective timestamp remains a
separate deployment decision.

No separate run-kind column is needed in the prototype: `dag_id` distinguishes scheduled
training from editor publication, while the parent package and parent run references make
the derivation explicit. Consequently, the original fit run remains immutable and every
editor-ready package shown in candidate history resolves to exactly one model run and one
verified candidate artifact.

An edited candidate is compared with both its raw parent and the current deployment. A
predictive metric change is evidence, not an automatic pass/fail verdict. The child's
history never presents the parent's CV score as though the edited model were
cross-validated: it labels that value as the parent baseline. Before/after metrics
recalculated against retained editor data are stored with their exact scope. The generated
prototype has training data only, so the notebook says that plainly and labels the
comparison `editor_training`; it does not call it CV. A custom model may later supply an
untouched validation/test frame through SuperGLM's existing editor-session API without
changing metric semantics.

Parent-versus-edit comparison uses the parent bundle's identical rows, sample weights,
offsets, and PK order. Champion comparison uses a verified champion artifact when one is
available; otherwise the new package-specific SQL scorer evaluates the current deployed
package on the same prepared-feature sample and the notebook labels any unavailable
evidence. Named summaries are persisted on the derived run with scopes such as
`editor_training_parent` and `editor_training_champion`, with fuller details in revision
metadata. Hard failures are limited to integrity errors such as invalid artifacts, export
errors, missing terms, or Python/package-specific SQL scoring mismatch on a deterministic
PK sample.

## Transform Handling

Analysts continue to write transforms directly in `build_final_model_frame()` using
pandas or NumPy. The prototype records the model package source hash and final-frame
manifest; it does not parse Python expressions.

The generated reference model provides one explicit module-level hook for its
`LogDensity = log(Density)` spline. Its signature is
`write_review_workbook(*, fitted_model, inputs: ModelInputs, output_path: Path) -> Path |
None`. This is a scaffold/model-module hook called by the shared runner, not a method
dynamically attached to `SuperGLM`. Its ordinary model-local Python implementation:

1. requires the already-validated positive-density domain;
2. obtains the same numeric `LogDensity` discretization-impact inputs used for the
   canonical curve instead of reparsing rounded Excel interval strings;
3. writes a separate review table with prepared endpoints plus raw `Density` endpoints
   obtained through `exp`, leaving relativities/log coefficients unchanged;
4. labels the sheet `PRESENTATION ONLY` and states that operational scoring still consumes
   `LogDensity`; and
5. records the review artifact path and hash in the candidate bundle's technical payload.

The raw representative is explicitly `exp(log_representative)`, a geometric axis point;
it is not described as an exposure-weighted raw mean. Tests verify transformed intervals
and relativities against the canonical block before the review workbook is exposed.

The canonical `rating_tables.xlsx` and publication receipt are not rewritten, and only
they may enter SQL staging. This is necessary because current receipt validation requires
workbook term names to match, compiled bands have no source/prepared binding, and SQL does
not calculate `LOG(Density)`. The raw parent and every edited child regenerate the same
review view. There is no registry of transforms, parser, or automatic inverse selection.
Models without this hook publish only their prepared feature axis; unsupported or
non-invertible transforms do the same.

If the real upstream scorer must supply raw `Density`, that becomes a separate small
design: add an allow-listed input binding to receipt term metadata and implement the
matching SQL transform. The prototype does not guess this boundary from a column name.
Generalization is considered only after the actual input contract is known and at least
two real models demonstrate the same need.

## Error Handling

- Scaffolded default builds fail with model-focused messages for missing final-frame
  columns, invalid split indices, absent model artifacts, and SuperGLM export failures.
- Optional weight and offset values are never represented as `UNKNOWN` for a new build.
  They are recorded as present or absent from the actual fit call.
- A non-`None` offset without a model-owned export contract fails before workbook staging;
  the shared runner never invents offset semantics from a column name.
- CV uses `error_score="raise"`; fold exceptions, non-convergence, invalid/duplicate test
  membership, and misleading OOF coverage are surfaced before publication.
- Candidate loading never falls back to an unverified artifact path.
- Editor submission validates that session/model baseline term shapes, levels, grids, and
  recorded parent hash match the selected fitted model.
- Editor publication is idempotent by a submission ID plus the full parent/session/model/
  receipt/version/effective-date compatibility tuple.
- A DRAFT child that fails package-specific SQL parity is rolled back and never appears as
  a PUBLISHED candidate.
- Deployment continues to require an explicit reason and uses the existing transactional
  deployment lock and history.

## Verification

The prototype is complete when automated tests demonstrate:

1. the default scaffold generates the reduced editable interface and explicit DAG;
2. a freshly scaffolded temporary model can execute the standard runner on a small frame;
3. the exact folds returned by SuperGLM CV are persisted to the split manifest/artifact;
4. fold failures/non-convergence fail the run, and OOF coverage is validated and reported;
5. `None` sample weight/offset states require no analyst metadata, while named fit/export
   weights are recorded accurately in receipt/package metadata;
6. run-level and fold-level CV metrics are persisted with stable names, pooled/fold
   semantics, and `cv` scope;
7. a successful build persists the extended hash-verified candidate artifact fields,
   manifest/PK binding, and model-source hash;
8. the filtered unique index rejects multiple model runs for one non-null package;
9. candidate history resolves integer package versions in the configured deployment slot
   without exposing model IDs;
10. `Workbench.open()` loads the fitted artifact, passes a primitive CV report, retains the
    real editor session/widget, and closes its local server;
11. editor-session JSON preserves its supported operation audit while the edited model
    artifact remains authoritative for publication;
12. an edited session publishes idempotently through `publish_editor_candidate()` as an
    immutable child package and a distinct derived model
    run with inherited dataset/split associations, a verified edited artifact, and editor
    revision metadata;
13. edited metrics retain their true evaluation scope and do not masquerade as fresh CV;
14. the non-live package scorer blocks a Python/SQL mismatch before child publication;
15. the `LogDensity` review workbook exposes correctly exponentiated raw `Density` values
    without changing the canonical workbook/receipt used for staging; and
16. `submission.request_deployment()` deploys that child through the existing deployment
    DAG without requiring the analyst to copy an ID.

One end-to-end smoke path will use a package generated by the scaffold during the test or
prototype setup. MTPL remains regression coverage only and is not the source of the
scaffold contract.

## Follow-up Decision Gate

After the prototype is demonstrated, decide whether to continue based on three questions:

- Is the notebook interaction simple enough for analysts without a custom application?
- Are retained model artifacts acceptably sized and governed, or must evaluation frames
  be reloaded/materialized separately?
- Is a shared network location available, or should trusted compact artifacts be stored
  directly in SQL Server for cross-Cloud-PC access?

Only after those answers should the design consider removing TOML, central artifact
storage, generalized source-axis transforms, or automated challenger promotion.
