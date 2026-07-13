# Local and Remote Pricing Notebook Template

## Goal

Provide a real `.ipynb` template that an analyst can scaffold for a pricing model and run either against a self-contained local SQLite audit store or the existing private work SQL runtime. The notebook should expose model decisions while hiding audit plumbing and must never contain private connection details.

## Analyst workflow

The first code cell contains the important controls:

```python
DATABASE_MODE = "local"  # "local" or "remote"
EXPECTED_REMOTE_DATABASE = ""  # required in remote mode
ALLOW_REMOTE_WRITES = False

DATA_AS_OF = None
RUN_EDITOR = False
DEPLOY = False
```

The connection cell creates a notebook context and displays a clear destination summary before any model work:

```python
pricing = connect(
    mode=DATABASE_MODE,
    local_root=MODEL_DIR / ".local",
    expected_remote_database=EXPECTED_REMOTE_DATABASE,
    allow_remote_writes=ALLOW_REMOTE_WRITES,
)
pricing.destination
```

The remaining visible cells are ordinary notebook work: load a frame, define features and transforms, fit the model, build and publish a candidate, and optionally open the editor or deploy. Source data loading remains analyst-owned, so it can use an existing private SQL helper, pandas, parquet, or another source without coupling that code to the audit destination.

## Local mode

- Create and reuse the existing three-schema SQLite representation beneath `MODEL_DIR / ".local"`.
- Apply the offline SQLite DDL only when the store is new or incomplete.
- Never reset or delete a local store as a side effect of running the pricing notebook.
- Point generated artifacts at local writable directories rather than Docker `/opt` paths.
- Do not commit empty or populated SQLite database files.

This gives a newly scaffolded notebook a working audit store without Docker, Airflow, credentials, or database setup.

## Remote mode

- Load the existing generic private runtime through `PRICING_RUNTIME_MODULE`.
- Keep all server names, credentials, schema names, and private imports outside the repository.
- Require `EXPECTED_REMOTE_DATABASE` and verify it against `SELECT DB_NAME()` after connecting.
- Never create, migrate, seed, or reset the remote database from a pricing-model notebook.
- Permit connection and reads with `ALLOW_REMOTE_WRITES = False`, but reject every mutating operation with an actionable error. This includes model registration, candidate building, publication, editor persistence, and deployment.
- Allow mutations only when the verified database matches and `ALLOW_REMOTE_WRITES = True`.

The SQL DDL remains a separate administrative notebook or setup process.

## API shape

`pricing_pipeline.notebook.connect` gains explicit local and remote modes and returns the existing notebook context extended with:

- the selected mode;
- whether writes are allowed;
- a safe human-readable destination;
- a central write guard used by all mutating notebook helpers.

Existing callers that do not choose a mode retain their current runtime-provider behavior for compatibility. The notebook template always chooses explicitly.

## Scaffolded notebook

`scripts/scaffold_pricing_model.py` creates `pricing_model.ipynb` in each new pricing-model directory. It is valid notebook JSON rather than a generated `.py` surrogate and contains no executed outputs.

The notebook is intentionally small:

1. imports and model directory;
2. global analyst controls;
3. connection and visible destination;
4. analyst-owned data-loading cell;
5. feature, transform, and model definition;
6. one candidate build call;
7. metrics and audit summary;
8. explicit publish, edit, and deploy cells.

The framework derives dataset fingerprints, primary-key tracking, split metadata, model revisions, transform metadata, offsets, weights, metrics, and audit rows wherever the available inputs make that possible. Analysts supply genuine modelling decisions; they do not manually assemble audit records.

## Verification

Tests will demonstrate that:

- local mode creates a usable schema and preserves prior data on rerun;
- local artifacts use writable model-local paths;
- remote mode rejects a missing or mismatched database confirmation;
- remote mutations are blocked until explicitly enabled;
- an allowed remote context uses the private runtime provider without running DDL;
- all notebook mutation entry points use the write guard;
- scaffolding produces a valid, output-free `.ipynb` whose code cells compile;
- no connection details or SQLite database files are added to version control.

