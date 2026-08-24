# SuperGLM pricing workbench

This repository is a notebook-first path from model data to a reviewed,
immutable SQL rating package. Airflow is not required for the current workflow.

## Create a model workspace

```bash
uv sync
cp pricing_scaffold.example.toml pricing_scaffold.toml
uv run python scripts/scaffold_pricing_model.py \
  --model-name CLAIM_FREQUENCY \
  --target-name claim_count \
  --model-label "Claim frequency"
```

The scaffold creates:

```text
pricing_models/claim_frequency/
├── 01_data_ingestion.ipynb
├── 02_model_training.ipynb
├── 03_model_editor.ipynb
├── 04_manual_adjustment.ipynb
├── 05_model_deployment.ipynb
└── 99_scratch_work.ipynb
```

Notebook names follow `xx_name_name2.ipynb`.

| Notebook | Purpose |
|---|---|
| `01_data_ingestion.ipynb` | Build the governed model frame and record its data-as-at date. |
| `02_model_training.ipynb` | Fit and publish `RAW`, then optionally `ROUTINE_EDIT`, candidates. |
| `03_model_editor.ipynb` | Optionally edit a selected published package and publish an `EDITOR_EDIT`. |
| `04_manual_adjustment.ipynb` | Apply replayable business factors and publish a `MANUAL_EDIT`; deployment is optional and explicit. |
| `05_model_deployment.ipynb` | Review and deploy one selected published package. |
| `99_scratch_work.ipynb` | Disposable data, feature, model, and grouping experiments. It cannot publish or deploy. |

The reference workflow is in
[`pricing_models/mtpl_frequency`](pricing_models/mtpl_frequency).

The editor is an optional baseline/refresh gate. Once that selected package is
deployed, controlled weekly monitoring can compare static scoring, a
coefficient-only frozen refit, a fixed-knot lambda refit, and a fully adaptive
refit. Monitoring rows are evidence only and cannot be deployed as packages.

## Scaffold defaults

`pricing_scaffold.toml` at `--root` is discovered automatically. An explicit
`--config` wins; explicit command-line options win over the file.

```toml
[notebook_defaults]
database_mode = "remote"
runtime_module = "work_runtime.database"
expected_remote_database = "PricingAudit"

[manual_edit_defaults]
source_selector = "deployed"
carry_forward = true
```

Only the five keys shown above are accepted. Do not put credentials in this file.
`ALLOW_REMOTE_WRITES` is deliberately not configurable and every generated
notebook starts with it set to `False`.

## Important rules

- Data-as-at is part of dataset identity, not a fit or deployment timestamp.
- Grouping happens in Python. SQL stores the completed model and its evidence.
- An equivalent successful model is detected in Python before SQL staging.
- `RAW`, `ROUTINE_EDIT`, `EDITOR_EDIT`, and `MANUAL_EDIT` are distinct model kinds.
- Monitoring always freezes groupings, levels, specials, basis shape, and
  monotonic constraints from the exact deployed baseline.
- Every monitoring fit is checked again after fitting. Protected lambdas,
  lambda history, knot locations/boundaries, and model structure must verify
  exactly before SQL persistence is allowed.
- Local mode uses persistent SQLite audit databases; editor/manual publication
  and deployment require guarded remote mode.
- Save notebooks before building: source cells are part of model evidence;
  outputs and execution counts are not.

## Guides

- [Notebook workflow and function reference](docs/notebooks/README.md)
- [SQL schema, relationships, triggers, views, and migration runbook](docs/sql/README.md)
- [Script command index](scripts/README.md)

For an underwriter-facing comparison of already-scored models, use
`scripts/build_underwriter_report.py` with the generic
`docs/notebooks/underwriter_report.example.toml`. It produces one offline HTML
file and does not write models or diagnostics to SQL.

The migration chain in [`db/migrations`](db/migrations) is the authoritative SQL
Server schema. The standalone `docs/pricing_useful_tables*.sql` files are
conceptual extracts, not a replacement for migrations.

## Work database setup

For an existing database, apply only missing migrations:

```bash
uv run python scripts/apply_schema.py \
  --runtime-module work_runtime.database \
  --expected-database PricingAudit
```

Use the reset command only for a disposable schema. It is dry-run by default;
the destructive command and checks are in the [SQL runbook](docs/sql/README.md).

## Verify

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Do not commit model-local `.local/` state, notebook outputs, credentials, or
private work runtime modules.
