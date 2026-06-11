# Custom Scaffold Modeling Readability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make generated custom `modeling.py` clearly show which functions model authors should edit and which lifecycle recipe they usually leave alone.

**Architecture:** This is scaffold text and documentation only. Update `scripts/scaffold_pricing_model.py` templates and README guidance; do not change runtime behavior, function names, DAG shape, SQL DDL, or publish logic.

**Tech Stack:** Python scaffold generator, pytest scaffold tests, ruff.

---

### Task 1: Lock In Scaffold Signposting Tests

**Files:**
- Modify: `tests/test_scaffold_pricing_model.py`

- [ ] **Step 1: Write the failing test assertions**

In `test_scaffold_pricing_model_writes_model_package_and_dag`, extend the
`modeling` assertions with:

```python
assert "Edit These Model-Specific Functions" in modeling
assert "Standard Build Recipe - Usually Leave This Alone" in modeling
assert "data.py decides the handoff shape" in modeling
assert "Do not pass large DataFrames through Airflow/XCom" in modeling
assert "rating_workbook_path, model_artifact_path, metrics" in modeling
assert "Start by customizing the functions above" in modeling
assert "The manifest and split artifacts use this frame order" in modeling
```

Keep the existing assertions that `read_prepared_source`,
`build_final_model_frame`, `fit_validate_export_rating_tables`, and
`train_validate_export_model` still exist.

- [ ] **Step 2: Verify the test fails**

Run:

```bash
rtk uv run pytest -q tests/test_scaffold_pricing_model.py::test_scaffold_pricing_model_writes_model_package_and_dag
```

Expected: FAIL because the generated `modeling.py` does not yet contain the new
section headers/docstrings.

### Task 2: Update Generated Modeling Template

**Files:**
- Modify: `scripts/scaffold_pricing_model.py`

- [ ] **Step 1: Update `_custom_modeling_template(...)` docstring and sections**

Change the generated module docstring to:

```python
"""Model-owned build logic for this pricing model.

Edit the functions in the first section. The final recipe function wires those
pieces into the shared manifest/publish contract.
"""
```

Group imports with comments:

```python
# Model-local config/constants.
...

# Shared lifecycle helpers. Most model authors do not need to edit these imports.
...
```

Add the section headers:

```python
# ---------------------------------------------------------------------------
# Edit These Model-Specific Functions
# ---------------------------------------------------------------------------
```

and:

```python
# ---------------------------------------------------------------------------
# Standard Build Recipe - Usually Leave This Alone
# ---------------------------------------------------------------------------
```

- [ ] **Step 2: Add edit-point docstrings**

Add docstrings to:

```python
def read_prepared_source(prepared: Mapping[str, Any]) -> pd.DataFrame:
    """Load the prepared source frame for this run.

    data.py decides the handoff shape. For example, if prepare_source_data(...)
    returned {"source_data_path": ".../source.parquet"}, read that file here.
    Do not pass large DataFrames through Airflow/XCom.
    """
```

```python
def build_final_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Create the final frame used for validation, training, export, and manifesting."""
```

```python
def fit_validate_export_rating_tables(...):
    """Fit/validate the model and export the rating workbook.

    Return:
        rating_workbook_path, model_artifact_path, metrics
    """
```

- [ ] **Step 3: Add lifecycle recipe docstring and frame-order comment**

At the start of `train_validate_export_model(...)`, add:

```python
"""Standard custom-model lifecycle recipe.

Start by customizing the functions above. Edit this recipe only when your model
needs a different build flow. The recipe resolves stable publish metadata,
calls the model-owned functions, creates the frame-backed manifest, and returns
the CompletedModelBuild payload consumed by the publish task.
"""
```

Before sorting the final frame, add:

```python
# The manifest and split artifacts use this frame order, so keep ordering
# deterministic and aligned with PK_COLUMNS unless the model deliberately needs
# a different order.
```

- [ ] **Step 4: Verify the focused scaffold test passes**

Run:

```bash
rtk uv run pytest -q tests/test_scaffold_pricing_model.py::test_scaffold_pricing_model_writes_model_package_and_dag
```

Expected: PASS.

### Task 3: Update README Guidance

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Clarify generated `modeling.py` ownership**

In the custom scaffold bullet list, update the `modeling.py` bullet to say it
contains three model-owned edit points:

```text
read_prepared_source
build_final_model_frame
fit_validate_export_rating_tables
```

and one standard recipe:

```text
train_validate_export_model
```

- [ ] **Step 2: Clarify prepared payload handoff**

Add a short paragraph:

```text
`prepare_source_data(...)` returns a small dictionary of paths, table names, IDs,
or metadata. The Airflow wrapper merges in `run_key`, `output_dir`,
`effective_from`, and `data_as_of_date`; `modeling.py` receives that merged
dictionary as `prepared`. Keys such as `source_data_path` are model-owned
examples, not framework-required field names.
```

### Task 4: Verification And Commit

**Files:**
- Verify: `scripts/scaffold_pricing_model.py`
- Verify: `tests/test_scaffold_pricing_model.py`
- Verify: `README.md`

- [ ] **Step 1: Run focused tests**

```bash
rtk uv run pytest -q tests/test_scaffold_pricing_model.py
```

Expected: all scaffold tests pass.

- [ ] **Step 2: Run lint/format checks**

```bash
rtk uv run ruff check scripts/scaffold_pricing_model.py tests/test_scaffold_pricing_model.py
rtk uv run ruff format --check scripts/scaffold_pricing_model.py tests/test_scaffold_pricing_model.py
```

Expected: both commands exit 0.

- [ ] **Step 3: Run generated scaffold compile sanity**

Generate a temporary scaffold and compile generated files:

```bash
tmpdir="$(mktemp -d)"
rtk uv run python scripts/scaffold_pricing_model.py --root "$tmpdir" --model-key CHECK_MODEL --model-label "Check model" --target-name target
rtk uv run python -m py_compile \
  "$tmpdir/pricing_models/check_model/data.py" \
  "$tmpdir/pricing_models/check_model/modeling.py" \
  "$tmpdir/pricing_models/check_model/airflow_tasks.py" \
  "$tmpdir/dags/pricing_check_model.py"
```

Expected: commands exit 0.

- [ ] **Step 4: Check diff cleanliness**

```bash
rtk git diff --check
```

Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
rtk git add README.md scripts/scaffold_pricing_model.py tests/test_scaffold_pricing_model.py docs/superpowers/plans/2026-06-11-custom-scaffold-modeling-readability.md
rtk git commit -m "docs: clarify custom scaffold modeling hooks"
```
