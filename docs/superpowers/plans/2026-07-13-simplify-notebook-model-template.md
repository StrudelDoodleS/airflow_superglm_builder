# Simplify Notebook Model Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the verbose analyst notebook call with one Python model specification, safely resolve data-as-of lineage, and stop assigning undeployed candidates a fabricated effective date.

**Architecture:** Add a backward-compatible `PricingModelSpec` to the synchronous notebook API. The spec owns stable model, input, dataset, validation, and standard exposure-offset decisions; `build_candidate()` derives the low-level `ModelInputs` and audit arguments. Keep legacy/DAG callers working by making effective dates optional end-to-end, with a forward SQL migration that permits `NULL` while deployment timestamps remain authoritative.

**Tech Stack:** Python 3.14, pandas, NumPy, Pydantic, SuperGLM 0.11, SQLAlchemy, SQL Server DDL, pytest, Jupyter notebook JSON.

---

### Task 1: Add the simple Python model specification

**Files:**
- Modify: `pricing_pipeline/notebook.py`
- Modify: `tests/test_notebook_workflow.py`

- [ ] **Step 1: Write failing tests for `PricingModelSpec` and spec registration**

Add tests that create:

```python
spec = api.PricingModelSpec(
    name="CLAIM_FREQUENCY",
    label="Claim frequency",
    target="claim_count",
    model_type="superglm_poisson",
    deployment_slot="PRODUCTION",
    features=("age", "region"),
    dataset_name="claim_frequency_frame",
    source_system="pricing_sql",
    pk_columns=("policy_id",),
    exposure_column="exposure",
    validation=ValidationSplitConfig.kfold(
        n_splits=2,
        random_state=7,
        materialize=True,
    ),
)
```

Assert normalized immutable fields, reject blank/duplicate feature and PK names,
reject overlap between target/PK/features/exposure, and assert:

```python
model = api.register_model(
    context,
    spec,
    source_root=source_root,
    created_by="analyst@example.test",
)
assert model.spec is spec
assert model.config.model_name == spec.name
assert model.config.validation_split == spec.validation
```

Retain the existing keyword-oriented registration test to prove backward
compatibility.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
rtk proxy uv run pytest tests/test_notebook_workflow.py -k 'pricing_model_spec or register_model' -q
```

Expected: failures because `PricingModelSpec` and positional spec registration do
not exist.

- [ ] **Step 3: Implement the immutable spec and registration adapter**

In `pricing_pipeline/notebook.py`, add:

```python
@dataclass(frozen=True)
class PricingModelSpec:
    name: str
    label: str
    target: str
    model_type: str
    deployment_slot: str
    features: tuple[str, ...]
    dataset_name: str
    source_system: str
    pk_columns: tuple[str, ...]
    validation: ValidationSplitConfig = ValidationSplitConfig.kfold()
    exposure_column: str | None = None
    sample_weight_column: str | None = None
    data_as_of_column: str | None = None
    scoring: tuple[str, ...] = ("deviance",)
    fit_mode: str = "fit_reml"
```

Normalize text and tuple fields in `__post_init__`, reject duplicates and role
overlap, and keep `source_system` explicit. Add `spec: PricingModelSpec | None`
to `RegisteredModel`.

Change registration to:

```python
def register_model(
    pricing: NotebookContext,
    spec: PricingModelSpec | None = None,
    *,
    name: str | None = None,
    label: str | None = None,
    target: str | None = None,
    model_type: str | None = None,
    deployment_slot: str = "PRODUCTION",
    validation_split: ValidationSplitConfig = ValidationSplitConfig.kfold(),
    source_root: str | Path,
    package_status: str = "PUBLISHED",
    created_by: str | None = None,
) -> RegisteredModel:
```

When `spec` is supplied, derive the `ModelBuildConfig` from it and reject
simultaneous legacy identity arguments. When absent, retain existing behavior.
Export `PricingModelSpec` from `__all__`.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run:

```bash
rtk proxy uv run pytest tests/test_notebook_workflow.py -k 'pricing_model_spec or register_model' -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the spec API**

```bash
rtk git add pricing_pipeline/notebook.py tests/test_notebook_workflow.py
rtk git commit -m "feat: add simple notebook pricing model spec"
```

### Task 2: Derive candidate inputs and data-as-of lineage

**Files:**
- Modify: `pricing_pipeline/notebook.py`
- Modify: `tests/test_notebook_workflow.py`

- [ ] **Step 1: Write failing tests for the four-argument analyst path**

Create a frame containing PK, target, features, exposure, and a dedicated
snapshot column. Register a `PricingModelSpec` and call:

```python
candidate = api.build_candidate(
    context,
    model=model,
    frame=frame,
    model_factory=lambda: object(),
    data_as_of="2026-06-30",
    run_key="notebook-run-1",
    created_by="analyst@example.test",
)
```

Capture `run_standard_superglm_build()` and assert:

```python
assert list(inputs.X.columns) == ["age", "region"]
assert inputs.y.name == "claim_count"
assert np.allclose(inputs.offset, np.log(frame["exposure"]))
assert inputs.export_weight.name == "exposure"
assert manifest_spec.dataset_name == "claim_frequency_frame"
assert manifest_spec.source_system == "pricing_sql"
assert manifest_spec.data_as_of_date == date(2026, 6, 30)
assert manifest_spec.pk_columns == ("policy_id",)
assert offset_contract.handling == "EXPORTED_FACTOR"
assert offset_export_options["offset_source"].equals(inputs.export_weight)
```

Add failure cases for missing features, null/non-positive exposure, neither
data-as-of source, ambiguous/null `data_as_of_column`, and explicit/column date
disagreement. Add a passing test where a single-valued snapshot column supplies
the date without a `data_as_of=` argument.

- [ ] **Step 2: Run the candidate tests and verify they fail**

Run:

```bash
rtk proxy uv run pytest tests/test_notebook_workflow.py -k 'build_candidate or data_as_of or exposure' -q
```

Expected: new simple-path and validation tests fail.

- [ ] **Step 3: Implement safe derivation while retaining advanced overrides**

Make `X`, `y`, `scoring`, `dataset_name`, `source_system`, and `pk_columns`
optional keyword arguments to `build_candidate()`. Resolve omitted values from
`model.spec`; preserve explicit values for the existing advanced API.

Add date resolution equivalent to:

```python
def _resolve_data_as_of(
    frame: pd.DataFrame,
    *,
    explicit: date | datetime | str | None,
    column: str | None,
) -> date:
    column_value = None
    if column is not None:
        if column not in frame:
            raise ValueError(f"data-as-of column is missing from model frame: {column}")
        if frame[column].isna().any():
            raise ValueError(f"data-as-of column {column!r} contains null values")
        values = {_normalise_notebook_date(value, "data_as_of") for value in frame[column]}
        if len(values) != 1:
            raise ValueError(f"data-as-of column {column!r} must contain exactly one date")
        column_value = values.pop()
    explicit_value = (
        None if explicit is None else _normalise_notebook_date(explicit, "data_as_of")
    )
    if explicit_value is not None and column_value is not None and explicit_value != column_value:
        raise ValueError("explicit data_as_of does not match the configured data-as-of column")
    resolved = explicit_value or column_value
    if resolved is None:
        raise ValueError("provide data_as_of or configure PricingModelSpec.data_as_of_column")
    return resolved
```

For a configured exposure column, validate numeric finite positive values and
derive the offset/export settings only when the corresponding advanced override
was not supplied:

```python
exposure = frame[spec.exposure_column].astype(float)
offset = np.log(exposure)
export_weight = exposure
export_weight_name = spec.exposure_column
weight_column = spec.exposure_column
offset_contract = OffsetExportContract(
    handling="EXPORTED_FACTOR",
    source_factor_name=spec.exposure_column,
    published_factor_name=spec.exposure_column,
    source_name=spec.exposure_column,
    label=f"log({spec.exposure_column})",
)
offset_export_options = {
    "offset_source": exposure,
    "offset_name": spec.exposure_column,
    "offset_kind": "auto",
}
```

Continue to pass every derived vector through the canonical PK identity alignment
already present in `build_candidate()`.

- [ ] **Step 4: Run the notebook workflow tests and verify they pass**

Run:

```bash
rtk proxy uv run pytest tests/test_notebook_workflow.py -q
```

Expected: all notebook workflow tests pass, including the legacy verbose call.

- [ ] **Step 5: Commit candidate derivation**

```bash
rtk git add pricing_pipeline/notebook.py tests/test_notebook_workflow.py
rtk git commit -m "feat: derive notebook candidate audit inputs"
```

### Task 3: Permit candidates without an effective date

**Files:**
- Create: `db/migrations/V026__nullable_candidate_effective_date.sql`
- Modify: `db/offline_sqlite/pricing.sql`
- Modify: `db/offline_sqlite/pricing_stg.sql`
- Modify: `pricing_pipeline/models/spec.py`
- Modify: `pricing_pipeline/modeling/standard_superglm.py`
- Modify: `pricing_pipeline/orchestration/completed_build_helpers.py`
- Modify: `pricing_pipeline/orchestration/publish_completed_build.py`
- Modify: `pricing_pipeline/publishing/staging.py`
- Modify: `pricing_pipeline/publishing/editor_candidate.py`
- Modify: `tests/test_migrations.py`
- Modify: `tests/test_publish_completed_build.py`
- Modify: `tests/test_standard_superglm.py`
- Modify: `tests/test_editor_candidate_publisher.py`

- [ ] **Step 1: Write failing nullable-date contract tests**

Add migration assertions for both SQL Server columns:

```python
sql = migration("V026__nullable_candidate_effective_date.sql")
assert "ALTER TABLE pricing.PRICING_RATE_PACKAGE" in sql
assert "ALTER COLUMN effective_from_date DATE NULL" in sql
assert "ALTER TABLE pricing_stg.STG_RATING_EXPORT" in sql
```

Add tests proving:

```python
build = CompletedModelBuild(
    rating_workbook_path="rating.xlsx",
    model_version="v1",
    effective_from=None,
)
assert build.effective_from is None
```

and that completed publication passes `effective_from=None` to
`ModelExportResult`/staging. Add an editor parent-row test with
`effective_from_date=None` and assert `ParentCandidate.effective_from is None`.

- [ ] **Step 2: Run nullable-date tests and verify they fail**

Run:

```bash
rtk proxy uv run pytest tests/test_migrations.py tests/test_publish_completed_build.py tests/test_standard_superglm.py tests/test_editor_candidate_publisher.py -k 'effective_from or nullable_candidate' -q
```

Expected: failures from required Pydantic/dataclass fields, stringifying `None`,
and the absent migration.

- [ ] **Step 3: Add the forward migration and nullable offline schema**

Create `V026__nullable_candidate_effective_date.sql`:

```sql
IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'effective_from_date') IS NOT NULL
BEGIN
    ALTER TABLE pricing.PRICING_RATE_PACKAGE
        ALTER COLUMN effective_from_date DATE NULL;
END;
GO

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'effective_from_date') IS NOT NULL
BEGIN
    ALTER TABLE pricing_stg.STG_RATING_EXPORT
        ALTER COLUMN effective_from_date DATE NULL;
END;
GO
```

Change the corresponding SQLite columns, including offline `MODEL_RUN`, from
`TEXT NOT NULL` to nullable `TEXT`.

- [ ] **Step 4: Thread `None` through completed builds and publication**

Apply these compatible type changes:

```python
class CompletedModelBuild(BaseModel):
    effective_from: str | None = None

@field_validator("effective_from", mode="before")
def _effective_from_date_text(cls, value):
    return None if value is None else _normalise_effective_from(value)
```

Use `str | None` in `completed_model_build_payload()`,
`run_standard_superglm_build()`, `ModelExportResult`, and
`stage_rating_export()`. In `publish_completed_model_build()`, pass
`build.effective_from` without `_required_text()`.

Change editor loading from:

```python
effective_from=str(row["effective_from_date"])
```

to:

```python
effective_from=(
    None
    if row.get("effective_from_date") is None
    else str(row["effective_from_date"])
)
```

Keep `effective_from_for_run()` unchanged for legacy scheduled/Airflow callers;
they may still deliberately supply a proposed date.

- [ ] **Step 5: Run the affected suites and verify they pass**

Run:

```bash
rtk proxy uv run pytest tests/test_migrations.py tests/test_publish_completed_build.py tests/test_standard_superglm.py tests/test_editor_candidate_publisher.py tests/test_rating_export.py tests/test_package_writer.py -q
```

Expected: all selected tests pass; existing non-null dates remain normalized and
idempotency comparisons treat `None` as a stable value.

- [ ] **Step 6: Commit nullable candidate dates**

```bash
rtk git add db/migrations/V026__nullable_candidate_effective_date.sql db/offline_sqlite/pricing.sql db/offline_sqlite/pricing_stg.sql pricing_pipeline/models/spec.py pricing_pipeline/modeling/standard_superglm.py pricing_pipeline/orchestration/completed_build_helpers.py pricing_pipeline/orchestration/publish_completed_build.py pricing_pipeline/publishing/staging.py pricing_pipeline/publishing/editor_candidate.py tests/test_migrations.py tests/test_publish_completed_build.py tests/test_standard_superglm.py tests/test_editor_candidate_publisher.py
rtk git commit -m "fix: separate candidate publication from deployment dates"
```

### Task 4: Rewrite the actual MTPL notebook template

**Files:**
- Modify: `pricing_models/mtpl_frequency/pricing_model.ipynb`
- Modify: `tests/test_pricing_model_notebooks.py`

- [ ] **Step 1: Write failing notebook-structure assertions**

Parse the notebook and assert:

- the first code cell contains imports only;
- the second code cell contains `MODEL = PricingModelSpec(` and `DATA_AS_OF`;
- the third code cell contains `RUN_EDITOR = False` and `DEPLOY = False`;
- no cell contains `EFFECTIVE_FROM`;
- the build cell contains the five analyst-facing arguments and does not contain
  `X=X`, `y=y`, `dataset_name=`, `source_system=`, `pk_columns=`,
  `weight_column=`, `offset_contract=`, or `offset_export_options=`.

- [ ] **Step 2: Run the notebook contract tests and verify they fail**

Run:

```bash
rtk proxy uv run pytest tests/test_pricing_model_notebooks.py -q
```

Expected: failures because settings and flags are currently scattered and the
build call remains verbose.

- [ ] **Step 3: Reorder and simplify the notebook**

Make the first cells title, imports, analyst settings, and optional actions. The
MTPL source has no governed fold column, so use:

```python
validation=ValidationSplitConfig.kfold(
    n_splits=5,
    random_state=42,
    shuffle=True,
    materialize=True,
)
```

Use `PricingModelSpec`, register it with:

```python
model = register_model(
    pricing,
    MODEL,
    source_root=MODEL_DIR,
)
```

Retain `Exposure` and transformed columns in `frame`, remove separate notebook
`X`, `y`, `offset`, offset-contract, and export-option construction, and use:

```python
candidate = build_candidate(
    pricing,
    model=model,
    frame=frame,
    model_factory=make_model,
    data_as_of=DATA_AS_OF,
)
```

Keep SQL, feature transforms, model definition, publication, optional editor,
and optional deployment visible. Do not add TOML or Airflow calls.

- [ ] **Step 4: Run notebook and helper tests**

Run:

```bash
rtk proxy jq empty pricing_models/mtpl_frequency/pricing_model.ipynb
rtk proxy uv run pytest tests/test_pricing_model_notebooks.py tests/test_notebook_workflow.py -q
```

Expected: valid notebook JSON and all selected tests pass.

- [ ] **Step 5: Commit the notebook template**

```bash
rtk git add pricing_models/mtpl_frequency/pricing_model.ipynb tests/test_pricing_model_notebooks.py
rtk git commit -m "feat: simplify pricing model notebook workflow"
```

### Task 5: Verify SQL publication and regression safety

**Files:**
- Modify only if verification exposes a defect in files already listed above.

- [ ] **Step 1: Run static and full-suite checks**

Run:

```bash
rtk ruff check .
rtk git diff --check
rtk proxy jq empty pricing_models/mtpl_frequency/pricing_model.ipynb
rtk proxy uv run pytest tests/ -q
```

Expected: Ruff reports no issues, diff check and JSON validation are silent, and
the complete pytest suite passes.

- [ ] **Step 2: Apply migrations twice to an isolated SQL Server database**

Create a disposable `PricingLabNotebookCheck` database through the configured
runtime, apply `scripts/apply_schema.py`, then run it a second time.

Expected: V001-V026 apply once; the second run skips every current migration.

- [ ] **Step 3: Execute the actual notebook code cells**

Seed deterministic MTPL-shaped source rows, execute every code cell sequentially
in one namespace with `RUN_EDITOR=False` and `DEPLOY=False`, and query SQL.

Expected:

```text
MODEL_RUN.run_status = SUCCESS
PRICING_RATE_PACKAGE.package_status = PUBLISHED
PRICING_RATE_PACKAGE.effective_from_date = NULL
PRICING_MODEL_DEPLOYMENT row count = 0
PRICING_RATE_CELL row count > 0
PRICING_COMPILED_RATE_CELL row count > 0
```

- [ ] **Step 4: Remove the disposable verification database**

Drop only `PricingLabNotebookCheck` and leave the existing `PricingLab` database
untouched.

- [ ] **Step 5: Record final repository state**

Run:

```bash
rtk git status --short
rtk git log --oneline -5
```

Expected: only intentionally pre-existing unrelated changes, if any, remain
uncommitted; the implementation commits are visible on the feature branch.
