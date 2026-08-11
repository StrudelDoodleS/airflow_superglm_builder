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
├── 04_model_deployment.ipynb
└── 99_scratch_work.ipynb
```

Notebook names follow `xx_name_name2.ipynb`.

| Notebook | Purpose |
|---|---|
| `01_data_ingestion.ipynb` | Build the governed model frame and record its data-as-at date. |
| `02_model_training.ipynb` | Fit and publish `RAW`, then optionally `ROUTINE_EDIT`, candidates. |
| `03_model_editor.ipynb` | Optionally edit a selected published package and publish an `EDITOR_EDIT`. |
| `04_model_deployment.ipynb` | Review and deploy one selected published package. |
| `99_scratch_work.ipynb` | Disposable data, feature, model, and grouping experiments. It cannot publish or deploy. |

The reference workflow is in
[`pricing_models/mtpl_frequency`](pricing_models/mtpl_frequency).

## Scaffold defaults

`pricing_scaffold.toml` at `--root` is discovered automatically. An explicit
`--config` wins; explicit command-line options win over the file.

```toml
[notebook_defaults]
database_mode = "remote"
runtime_module = "work_runtime.database"
expected_remote_database = "PricingAudit"
```

Only those three keys are accepted. Do not put credentials in this file.
`ALLOW_REMOTE_WRITES` is deliberately not configurable and every generated
notebook starts with it set to `False`.

## Important rules

- Data-as-at is part of dataset identity, not a fit or deployment timestamp.
- Grouping happens in Python. SQL stores the completed model and its evidence.
- An equivalent successful model is detected in Python before SQL staging.
- `RAW`, `ROUTINE_EDIT`, and `EDITOR_EDIT` are distinct model kinds.
- Local mode uses persistent SQLite audit databases; editor publication and
  deployment require guarded remote mode.
- Save notebooks before building: source cells are part of model evidence;
  outputs and execution counts are not.

## Guides

- [Notebook workflow and function reference](docs/notebooks/README.md)
- [SQL schema, relationships, triggers, views, and migration runbook](docs/sql/README.md)
- [Script command index](scripts/README.md)

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
