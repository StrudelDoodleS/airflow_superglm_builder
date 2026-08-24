# Script command index

Script paths stay flat because notebooks, shell launchers, tests, and work
runbooks call them directly. This index groups them by job without breaking
those stable entry points.

## Notebook workspace

| Script | Purpose |
|---|---|
| `scaffold_pricing_model.py` | Create the six-notebook model workspace; reads optional scaffold TOML defaults. |

## SQL schema and inspection

| Script | Purpose |
|---|---|
| `apply_schema.py` | Guard target database and apply only missing versioned migrations. |
| `reset_remote_pricing_schema.py` | Dry-run or destructively reset runtime-owned schemas and replay migrations. |
| `generate_db_diagrams.py` | Generate a catalog-derived static ERD site. |
| `render_schema_diagrams.py` | Render committed Mermaid diagrams and optionally preview them with Chafa Kitty output. |
| `inspect_rating_package.py` | Print one package, its terms, and sample cells. |
| `render_schema_sql.py` | Internal renderer used to substitute configured schema names; not a CLI. |
| `pricing_db.py` | Shared runtime/engine loader used by scripts; not a CLI. |

## Demo data

| Script | Purpose |
|---|---|
| `load_fremtpl_raw.py` | Load the freMTPL demo table; `--replace` truncates/reloads it. |
| `reset_pricing_experiments.py` | Delete local experiment history; requires `--yes`. |

## Monitoring diagnostics

| Script | Purpose |
|---|---|
| `simulate_model_monitoring.py` | Generate a synthetic 60% baseline plus four 10% arrivals; verify monitoring invariants and render feature, lambda, knot, relativity, and out-of-time performance figures. |

Run it with `uv run python scripts/simulate_model_monitoring.py`. Generated
CSVs, invariant evidence, and PNGs stay in ignored local state under
`state/monitoring_simulation/`.

## Underwriter model review

| Script | Purpose |
|---|---|
| `build_underwriter_report.py` | Build one self-contained HTML report from scored predictions, actuals, business weight, and optional SuperGLM objects/rating workbooks. |
| `build_underwriter_report_demo.py` | Fit three contrasting model vintages on public freMTPL data and render the local report preview. |
| `portable_underwriter_report.py` | Copyable prediction-only report; runs outside this repository with PEP 723/uv dependencies. |
| `export_portable_underwriter_report.py` | Regenerate or freshness-check the portable file from the canonical generic report runtime. |

For use in another project, copy only
[`portable_underwriter_report.py`](portable_underwriter_report.py). The short
[portable guide](../docs/notebooks/portable_underwriter_report.md) covers its
direct Python and TOML interfaces.

Copy `docs/notebooks/underwriter_report.example.toml` into ignored `state/`,
set the local paths and generic column mapping, then run:

```bash
uv run python scripts/build_underwriter_report.py \
  --config state/underwriter_report.toml \
  --allow-local-input
```

Prediction-only and workbook-only configurations do not load Joblib or
SuperGLM. If `[superglm_objects]` is configured, load only trusted artifacts and
add `--allow-trusted-model-load`; Joblib deserialisation can execute code. The HTML has
tabs for metrics, top main effects, relativities, shared-axis multi-model
prediction KDEs, Lorenz/gains curves, and configurable double lift with raw
volume by bin. Double lift reports deviance decomposition, binned calibration
D², a paired interval, and exact held-out NLL when each model has training-
fitted likelihood metadata. Fitted SuperGLM objects provide that metadata
automatically; TOML can provide it for prediction-only models. Its locally
owned stylesheet provides app tabs, context chips, inspector, metric strip,
chart geometry, confidence bands, hollow points, and yellow exposure
conventions. A configurable minimum cell size (20 distinct comparison units by
default) coarsens double-lift bins and rejects undersized report samples, so it
embeds aggregate evidence rather than row-level records. It needs no
browser-side package or internet connection.

For a richer public preview with three model vintages on one freMTPL holdout
(old 75% data, refreshed data, then refreshed data plus one feature), run:

```bash
uv run python scripts/build_underwriter_report_demo.py
```

It uses every fetched public row by default, retaining a 75/25 train/holdout
split, and writes `state/report_smoke/model_review.html`. Pass `--rows` only for
a quicker smoke run. This preview is illustrative; the governed report runner
above remains the entry point for local work data.

For fitted SuperGLM objects, top-feature ranking is the model's weighted
variance of each main-effect contribution on the link scale. This answers
“which features move the fitted rate most?” without expensive refitting. It is
not causal or incremental drop-one importance. Workbook-only ranking uses a
clearly labelled export-weighted log-relativity proxy.

## Scratch model diagnostics

| Script | Purpose |
|---|---|
| `run_scratch_blend_diagnostics.py` | Fit an unconstrained GAM and Tweedie CatBoost/LightGBM/XGBoost blend against one local parquet, then show where tree interactions help or fail out of sample. |

Keep the TOML and parquet local. Copy the generic example from
`docs/notebooks/scratch_blend_diagnostics.example.toml` into ignored `state/`,
replace its placeholder path and column names, then run:

```bash
uv sync --extra scratch
uv run python scripts/run_scratch_blend_diagnostics.py \
  --config state/scratch_blend_diagnostics.toml \
  --allow-local-input
```

The script writes only aggregate CSVs, a short report, and PNG diagnostics
under the configured ignored `state/` directory. It does not write SQL or save
row-level predictions.

## Local development services

| Script | Purpose |
|---|---|
| `bootstrap_no_docker.sh` | Install dependencies and create local state folders. |
| `no_docker_services.py` | List/start/stop the host-process service menu. |
| `start_no_docker_stack.sh` | Shell launcher for selected host services. |
| `start_mlflow_local.py` | Start optional local MLflow. Notebook publication does not require it. |
| `smoke_check.py` | Check that the installed SuperGLM exposes rating export. |

Use `uv run python <script> --help` before mutations. The destructive commands
are `reset_remote_pricing_schema.py`, `reset_pricing_experiments.py`, and
`load_fremtpl_raw.py --replace`; each has an explicit confirmation or flag.

See the [SQL runbook](../docs/sql/README.md) for the migration/reset decision and
the [notebook guide](../docs/notebooks/README.md) for scaffold usage.
