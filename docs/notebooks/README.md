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
| `04_manual_adjustment.ipynb` | Deployed or exact published package | Replayable policy plus `MANUAL_EDIT` child; optional explicit deployment | Silently skip missing levels |
| `05_model_deployment.ipynb` | Published SQL candidate and current champion | Deployment history/current pointer | Fit or edit |
| `99_scratch_work.ipynb` | Any exploratory source; published `RAW` for grouping work | Ignored local grouping artifact only | Build, publish, or deploy |

Accepted scratch data work moves to notebook 01. Accepted model choices move to
notebook 02. Scratch cells are excluded from model-source identity.

## Optional scratch benchmarks

Notebook 99 includes two deliberately disposable benchmarks:

- `unconstrained_superglm_features(...)` keeps raw categorical levels and uses
  unconstrained, data-driven splines with REML-estimated lambdas. It applies no
  grouping or monotonic/shape decision. `superglm_edf_table(...)` shows the
  effective degrees of freedom used by each smooth; ordered-categorical special
  levels are reported separately.
- `fit_boosted_blend(...)` fits CatBoost, LightGBM, and XGBoost out of fold,
  learns non-negative weights summing to one from held-out unit deviance,
  then refits the three learners on all scratch rows. With exposure, it fits an
  offset-equivalent rate and `predict_expected(...)` returns the aggregate
  response. The tree fit keeps credibility weight separate: its effective
  rate-scale weight is `sample_weight * exposure ** (2 - tweedie_power)`
  (`sample_weight * exposure` for Poisson).

The notebook passes its fitted unconstrained GAM as `reference_superglm`. For a
compound Tweedie target, the helper reads the power from that fitted model and
fixes the exact value in CatBoost, LightGBM, XGBoost, and the OOF blend deviance.
Per-tree objective or variance-power overrides are rejected, so a hyperparameter
search cannot silently change the distribution contract. Code without a fitted
reference may instead pass one explicit `tweedie_power`.

Set the GAM once in the modelling cell, for example
`SCRATCH_FAMILY = Tweedie(p=1.6)`. The blend cell needs no second power setting.

Install the optional tree libraries once, then restart the notebook kernel:

```bash
uv sync --extra scratch
```

These helpers return only in-memory Python objects. They have no SQL,
publication, or deployment path; an accepted feature decision must still move
into notebook 02 and the governed candidate workflow.

## Underwriter HTML review

After scoring one common review sample, the standalone report runner can compare
frequency, severity, or burn-cost predictions without publishing anything:

```bash
cp docs/notebooks/underwriter_report.example.toml state/underwriter_report.toml
uv run python scripts/build_underwriter_report.py \
  --config state/underwriter_report.toml \
  --allow-local-input
```

The report accepts multiple named prediction columns plus actual and sample-
weight columns. Optional fitted SuperGLM objects supply native main-effect
importance, relativity curves, confidence intervals, EDF, and exposure context;
optional rating workbooks supply exported relativities when an object is not
available. The generated HTML is self-contained, contains aggregate chart data
only, and uses a locally owned stylesheet for its read-only app tabs, context
bar, inspector, metrics, and relativity charts.

When `[superglm_objects]` is enabled, add `--allow-trusted-model-load` and load
only artifacts you trust: Joblib/pickle deserialisation can execute code.

The model-neutral library path needs only scored columns and provides metrics,
weighted prediction KDEs, model movement, Lorenz/gains curves, and double lift:

```python
from pricing_pipeline.reporting import build_scored_model_report

result = build_scored_model_report(
    scored,
    actual="actual",
    predictions={"Current": "pred_current", "Challenger": "pred_new"},
    sample_weight="exposure",
    features=["region", "age"],
    output_path="state/model_review.html",
)
```

Adapters add model-native importance, fitted effects, exact likelihood, and
later interaction evidence without changing the scored-data report core.

Set `problem_type` to `frequency`, `severity`, or `burn_cost`. Frequency and
burn-cost reports expect exposure in `sample_weight`; severity expects claim
count. That setting drives the response, volume, Lorenz/gains and density-axis
labels. Prediction KDEs share one axis and use a scrollable multi-model picker,
so selecting another model never silently changes the comparison scale.

Double-lift selectors choose the numerator and denominator used to rank rows.
Within every weighted bin, actual and predictions are calculated independently
as `sum(weight * value) / sum(weight)`, and the report shows raw exposure plus
exposure share for every bin. A scrollable checkbox picker controls which model
curves are displayed. The rebase selector can then divide all plotted series by
actual or any model in that bin; it changes only the display, never the bins or
underlying aggregates. Prediction distributions are business-weighted Gaussian
KDEs rather than histograms. The same tab has an aggregate model-movement view:
an exposure-weighted rank-migration heatmap shows reranking, while a log-scale
prediction heatmap shows local level changes that similar marginal KDEs can
hide. Cells below `minimum_cell_size` are removed before HTML serialization and
their combined exposure share is disclosed. Lorenz/gains uses the same
multi-model picker and shows equality plus the sample's tie-aware
perfect-ordering curve. Plotly is embedded inside the offline file, so line
charts and heatmaps provide hover, pan, zoom, autoscale, and reset without a CDN.

Double lift also reports quantitative evidence. Common-power row deviance is
decomposed by bin, while bounded line agreement reports clipped weighted Lin
concordance between each displayed model line and actuals: 1 is an exact match
and 0 means no positive agreement. Binned calibration D² remains available as
technical evidence. When both predictions
have training-fitted likelihood metadata, exact held-out NLL is the primary
pairwise score. For compound Tweedie models this is SuperGLM's adaptive full
density calculation: exact Wright--Bessel/series work in its bounded numerical
region and the library's guarded saddlepoint fallback outside it. A supplied
SuperGLM object provides its own fitted `p` and `phi`; otherwise put them under
`[model_likelihoods."Display name"]` in TOML.
The report never estimates either value on review outcomes. With no metadata it
falls back explicitly to deviance. `comparison_unit` can name a policy/cluster
column for the paired interval; its values are used only in memory and never
embedded in HTML. `minimum_cell_size` defaults to 20 distinct comparison units:
double-lift bins are deterministically coarsened until every displayed cell
meets it, and the report refuses samples that cannot satisfy the threshold.
Identifier columns used as `comparison_unit` must not also appear in `features`;
the builder rejects that overlap so identifier levels cannot leak through a
relativity table.

Predictions must already include any model offset and remain on the response
scale. When fitted SuperGLM evidence is collected for a current or holdout
portfolio, pass the exactly row-aligned log offset with `offset=` or configure
the optional `[columns].offset` column. It is used only to verify that the
fitted object produced the supplied predictions: the report does not apply the
offset again or serialize its values. If it is omitted, a retained training
offset is accepted only when the fitted rows and weights are provably aligned.

For SuperGLM-enriched evidence, the compatibility facade keeps the established
signature. Supplied likelihood metadata validates a fitted object's values and
never overrides them:

```python
from pricing_pipeline.reporting import (
    ModelLikelihoodSpec,
    UnderwriterReportOptions,
    build_underwriter_report,
)

report = build_underwriter_report(
    scored_frame,
    actual="actual_response",
    predictions={"Current": "prediction_current", "Challenger": "prediction_new"},
    sample_weight="business_weight",
    features=["feature_a", "feature_b"],
    superglm_models={"Current": fitted_model},
    rating_workbooks={"Challenger": challenger_workbook},
    model_likelihoods={
        "Challenger": ModelLikelihoodSpec(tweedie_power=1.5, dispersion=0.72),
    },
    offset="report_time_offset",  # optional, aligned fitted-evidence binding only
    comparison_unit="policy_id",  # optional; one unit per row when omitted
    output_path="state/underwriter_report/model_review.html",
    options=UnderwriterReportOptions(
        problem_type="burn_cost",
        tweedie_power=1.5,
        movement_bins=10,
        interaction_points=80,
        comparison_bootstrap_replicates=200,
        minimum_cell_size=20,
    ),
)
```

Treat optimiser-selected blend weights as a technical predictive upper bound,
not a pricing recommendation. A production GAM/GBM blend should apply its
governed GAM floor (often 40–50% where that is the business standard) and be
chosen using feature-tail calibration, double lift, sparse-support behaviour,
and repeated-snapshot stability alongside average deviance.

For a full held-out comparison against a local parquet, copy
`scratch_blend_diagnostics.example.toml` to ignored `state/`, fill in the exact
local path and column names, and run:

```bash
uv run python scripts/run_scratch_blend_diagnostics.py \
  --config state/scratch_blend_diagnostics.toml \
  --allow-local-input
```

The 60/20/20 split fits on train, chooses only the technical GAM/GBM weight on
validation, and reserves test for diagnosis. The red comparator is an OOF-
weighted CatBoost/LightGBM/XGBoost blend with fixed, untuned learner settings.
Double-lift bins are balanced by the declared sample weight and ordered by
`boosted blend / GAM`. Pairwise heatmaps first remove both one-way effects from
that log ratio, exposing the interaction the boosted blend added; their second
panel shows held-out boosted-blend-minus-GAM Tweedie deviance. A positive red
cell means the blend is worse there. Always read that beside its support
bar/count: sparse red cells are warnings, not discoveries. The weighted Lorenz
plot reports each model's Gini on the same untouched test rows.

The double-lift presentation follows the CAS RPM model-lift handout: each
curve is indexed to its own portfolio average. The CSV also retains the raw
ratio-of-sums values. Gini is treated only as a ranking statistic, never as a
calibration measure.

The runner fixes one declared Tweedie power in all four models. With aggregate
response `y`, exposure `e`, and credibility weight `w`, trees fit `y / e` with
weight `w * e ** (2 - p)`, which is offset-equivalent Tweedie fitting. Outputs
are aggregate CSV/PNG evidence only under ignored `state/`; no rows, models, or
predictions are retained or published. Every calibration point is a ratio of
sums: `sum(sample_weight * response) / sum(sample_weight)`, with predictions
aggregated using the identical denominator.

## Baseline epochs and monitoring

Treat the editor as an optional genesis/refresh gate, not a weekly modelling
step:

```text
baseline epoch
  ingest -> fit RAW/ROUTINE_EDIT -> optional editor -> publish -> deploy
                                                          |
                                                   immutable fit contract
                                                          |
monitoring
  ingest a new dated snapshot -> static/frozen/lambda/adaptive comparisons
                              -> SQL evidence only; never auto-deploy
```

The deployed run starts the epoch. Its exact edited model is authoritative, so
the contract includes editor-created groupings, categorical levels and bases,
special levels, monotonic/shape constraints, basis type and dimension, fitted
knots, and fitted REML lambdas. A proper refresh goes through the baseline lane
again, is deployed deliberately, and starts a new contract and comparison
epoch.

Keep these as two conceptual lanes even if the notebooks remain in one model
directory. If they are split into physical subdirectories, use `baseline/` and
`monitoring/`; do not call the second lane `deployment`, because its variants
are diagnostic observations rather than candidate packages.

The four supported monitoring presets are deliberately limited:

| Variant | Coefficients | REML lambdas | Data-driven knots | Always fixed |
|---|---|---|---|---|
| `STATIC_SCORE` | Deployed | Deployed | Deployed | Groupings, levels, specials, constraints, basis type/dimension |
| `FROZEN_REFIT` | Refit | Deployed | Deployed | Same |
| `REESTIMATE_LAMBDA` | Refit | Refit | Deployed | Same |
| `FULL_ADAPTIVE` | Refit | Refit | Refit | Same |

This gives one clean coefficient-drift view, one smoothing-response view, and
one adaptive challenger. Arbitrary switch combinations are intentionally not
supported because most do not have a stable business interpretation.

## Scaffold configuration

Copy `pricing_scaffold.example.toml` to `pricing_scaffold.toml` at the scaffold
root:

```toml
[notebook_defaults]
database_mode = "remote"
runtime_module = "work_runtime.database"
expected_remote_database = "PricingAudit"

[manual_edit_defaults]
source_selector = "deployed"
carry_forward = true
```

```bash
uv run python scripts/scaffold_pricing_model.py \
  --model-name CLAIM_FREQUENCY \
  --target-name claim_count
```

Precedence is command line, explicit `--config`, auto-discovered
`<root>/pricing_scaffold.toml`, then built-in local defaults. Unknown sections
or keys fail fast. `ALLOW_REMOTE_WRITES` cannot be set in TOML.

`source_selector = "deployed"` makes notebook 04 open the package currently
deployed in the model's configured slot. An exact `PACKAGE_VERSION` in the
notebook overrides it. `carry_forward` is recorded in the canonical policy;
it never causes an implicit publication or deployment.

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
| `open_deployed_candidate(...)` | Resolve and open the exact package deployed in the configured slot | `Candidate` carrying baseline run/deployment evidence |
| `publish_edits(...)` | Save and publish an editor session | `EDITOR_EDIT` child publication |
| `ManualAdjustmentPolicy.from_rows(...)` | Define relative level/range multipliers | Canonical replayable policy and SHA-256 |
| `apply_manual_adjustment_policy(...)` | Apply the policy to one clean candidate | `ManualEditReview` with rules, edited model, and portfolio impact |
| `manual_adjustment_policy_from_candidate(...)` | Recover and verify a policy from a published manual child | `ManualAdjustmentPolicy` |
| `publish_manual_adjustment(...)` | Reapply the canonical policy and publish it | `MANUAL_EDIT` child publication |
| `deploy_package(...)` | Deploy exactly the reviewed candidate | Deployment record; stale champion fails |
| `build_model_fit_contract(...)` | Freeze the deployed model's structural and smoothing evidence | Immutable canonical JSON and SHA-256 |
| `run_monitoring_fit(...)` | Score or refit one controlled monitoring preset from a verified deployed `Candidate` | Terms, lambdas, comparable relativities, explicitly weighted metrics, frame/config/result digests |
| `persist_monitoring_fit(...)` | Write a completed observation after lineage checks | Deduplicated monitoring-run receipt |

A monitoring notebook can open the champion once, prepare the new manifest's
feature frame in the same column order, and run the presets explicitly:

```python
from pricing_pipeline.notebook import MonitoringVariant, run_monitoring_fit

baseline = open_deployed_candidate(pricing, model=model)
results = {
    variant: run_monitoring_fit(
        baseline,
        X_new,
        y_new,
        variant=variant,
        sample_weight=weight_new,
        offset=offset_new,
        model_frame=frame_new,
        target_column=model.spec.target,
        offset_column=model.spec.offset_column,
    )
    for variant in MonitoringVariant
}
```

Persist only after all requested fits have succeeded. Pass the new snapshot's
`manifest_id`, `baseline.model_run_id`, and
`baseline.technical["current_deployment_id"]`. Exact retries deduplicate; a
different data-as-at/manifest creates a new observation.

For persisted evidence, `run_monitoring_fit` accepts the deployed `Candidate`,
re-queries its current SQL lineage, and reloads the exact artifact from its
stored path, byte count, runtime metadata, and SHA-256. A raw fitted `SuperGLM`
is supported only for local simulation and its result cannot be persisted. The
ordered `model_frame` must hash to the supplied observation manifest at
persistence time. If the deployed fit contract declares a sample-weight or
offset input, the new snapshot must supply it; an offset is rejected when the
deployed model was fitted without one.

The post-fit guard verifies the fitted object, not just its input config. For
protected quantities it requires exact equality: structure and governed
level/grouping metadata, knot and boundary arrays, final lambdas, fixed-lambda
policy, and every fixed-lambda REML history step. `FROZEN_REFIT` also requires
SuperGLM's `fixed_lambdas` termination reason. A mismatch raises before a
result can be persisted. SQL stores the canonical evidence JSON and its SHA-256
with `invariant_status = 'VERIFIED'`, plus the exact ordered-frame, fit-config,
and complete-result digests. Metric names state their weighting explicitly,
for example `sample_weighted_mean_prediction`.

To reproduce a synthetic 60% baseline followed by four 10% arrivals, including
out-of-time scoring and drift figures:

```bash
uv run python scripts/simulate_model_monitoring.py
```

Outputs go to ignored local state under `state/monitoring_simulation/`.

`register_model`, `build_candidate`, `publish_candidate`, `publish_edits`,
`publish_manual_adjustment`, and `deploy_package` call the context write guard.
Editor/manual publication and deployment require remote mode.

## Manual business adjustments

Notebook 04 shows every published package with model kind, fit time,
data-as-at, parent and deployment state. It opens the current deployment by
default. Each policy rule names a feature, either exact levels or a numeric
range, a positive multiplier, and a reason. Python converts the multiplier to
the model's log-link shift; missing levels, overlapping rules, invalid factors,
and invalid preview predictions fail before publication.

The resulting package is an immutable `MANUAL_EDIT` child. Its SQL revision
metadata contains the canonical policy payload and digest as well as the
normal editor artifact evidence. Publication and deployment are separate
cells. The optional deployment cell defaults to off.

`carry_forward = true` means that the relative policy is intended to be
replayed against a later clean candidate. Apply it to that new base model, not
to the previous `MANUAL_EDIT`, so an uplift does not compound. Current weekly
monitoring rows are evidence-only and are not automatically adjusted or
deployable; policy replay matters only when a new candidate is being prepared.
Set `POLICY_SOURCE_PACKAGE_VERSION` to an earlier `MANUAL_EDIT` package to load
and verify its policy from SQL rather than typing its rules again. Replay is
refused when that policy recorded `carry_forward = false`; the trusted publisher
also reloads the parent and replays the canonical policy before accepting the
submitted model.

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
