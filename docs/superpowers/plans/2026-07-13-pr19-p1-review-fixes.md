# PR 19 P1 Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all five P1 findings on PR 19 so reviewed champion evidence, publication artifacts, scheduled lineage, numeric terms, and categorical interactions are correct and retry-safe.

**Architecture:** Persist the editor's reviewed champion snapshot in immutable package revision metadata and make deployment consume that snapshot. Serialize editor attempts with a same-host file lock and unique directories, move scheduled `MODEL_RUN` recording into the package transaction, then extend the existing receipt/staging/SQL path for numeric main effects and two-way categorical interactions. Keep the changes in current domain modules; add no service, transform language, or generic workflow framework.

**Tech Stack:** Python 3.14, Pydantic, SQLAlchemy, pandas/openpyxl, SuperGLM, SQL Server T-SQL, pytest, Ruff, Git.

---

## File Map

- Modify `pricing_pipeline/publishing/editor_candidate.py`: champion snapshot,
  existing-publication resolution, editor attempt lock/directories, and safe retry flow.
- Modify `pricing_pipeline/workbench/core.py`: read and validate reviewed champion metadata
  from the committed child package.
- Modify `pricing_pipeline/workbench/submission.py`: retain reviewed evidence after status
  resolution and use it for deployment compare-and-swap.
- Modify `pricing_pipeline/publishing/publisher.py`: pass the scheduled lineage writer into
  package publication.
- Modify `pricing_pipeline/orchestration/pipeline.py`: resolve completed exports before
  staging and record scheduled lineage on the package transaction connection.
- Modify `pricing_pipeline/orchestration/publish_completed_build.py`: return canonical
  existing lineage and safely discard only redundant retry-local artifact directories.
- Modify `pricing_pipeline/publishing/superglm_metadata.py`: emit supported interaction
  metadata and reject unsupported interaction classes.
- Modify `pricing_pipeline/publishing/staging.py`: derive interaction parents from the
  receipt, bound main-effect blocks, parse interaction matrices, and validate numeric
  per-unit rows.
- Modify `db/migrations/V025__package_specific_scoring.sql`: score numeric main effects
  and categorical interactions before generic cell matching.
- Modify focused tests in `tests/test_candidate_editor.py`,
  `tests/test_candidate_workbench.py`, `tests/test_editor_candidate_publisher.py`,
  `tests/test_model_publisher.py`, `tests/test_rating_export.py`,
  `tests/test_publish_completed_build.py`, `tests/test_superglm_metadata.py`, and
  `tests/test_migrations.py`.

### Task 1: Persist and enforce reviewed champion evidence

**Files:**
- Modify: `pricing_pipeline/publishing/editor_candidate.py:31-260,469-610`
- Modify: `pricing_pipeline/workbench/core.py:179-230`
- Modify: `pricing_pipeline/workbench/submission.py:31-205`
- Test: `tests/test_editor_candidate_publisher.py:220-475`
- Test: `tests/test_candidate_workbench.py:183-230`
- Test: `tests/test_candidate_editor.py:568-650`

- [ ] **Step 1: Write failing champion snapshot tests**

Add tests that make `_load_champion_bundle()` return a snapshot containing the same
`deployment.rate_package_id` selected with the artifact, and distinguish no champion
from an unavailable champion:

```python
snapshot = _load_champion_bundle(
    Engine(rows=[{"rate_package_id": 107, **artifact_fields}]),
    model_id=17,
    deployment_slot="HOME_FREQ_UAT",
    allowed_root=tmp_path,
    parent_bundle=parent,
)

assert snapshot.status == "COMPARED"
assert snapshot.rate_package_id == 107
assert snapshot.bundle.manifest_id == "champion-manifest"

empty = _load_champion_bundle(
    Engine(rows=[]),
    model_id=17,
    deployment_slot="HOME_FREQ_UAT",
    allowed_root=tmp_path,
    parent_bundle=parent,
)
assert empty.status == "NO_CHAMPION"
assert empty.rate_package_id is None
```

Add an export metadata assertion:

```python
metadata = json.loads(exported.revision_metadata_json)
assert metadata["champion_comparison"] == {
    "available": True,
    "deployment_slot": "HOME_FREQ_UAT",
    "rate_package_id": 107,
    "reason": None,
    "status": "COMPARED",
}
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
rtk proxy uv run pytest tests/test_editor_candidate_publisher.py -k 'champion' -q
```

Expected: failures because the lookup returns a tuple and does not select
`deployment.rate_package_id`.

- [ ] **Step 3: Implement the immutable champion snapshot**

In `editor_candidate.py`, introduce the concrete value object and use it on
`ParentCandidate`:

```python
@dataclass(frozen=True)
class ChampionSnapshot:
    deployment_slot: str
    rate_package_id: int | None
    bundle: CandidateBundle | None
    unavailable_reason: str | None

    @property
    def status(self) -> str:
        if self.rate_package_id is None:
            return "NO_CHAMPION"
        if self.bundle is None:
            return "UNAVAILABLE"
        return "COMPARED"

    def revision_metadata(self) -> dict[str, Any]:
        return {
            "available": self.status == "COMPARED",
            "deployment_slot": self.deployment_slot,
            "rate_package_id": self.rate_package_id,
            "reason": self.unavailable_reason,
            "status": self.status,
        }
```

Select `deployment.rate_package_id` in the champion query. Raise
`EditorSubmissionError` when more than one active deployment row resolves because no
single reviewed identity exists. Return `ChampionSnapshot` for all valid outcomes and
use `parent.champion.bundle` for metrics plus `parent.champion.revision_metadata()` in
the revision JSON.

- [ ] **Step 4: Write failing workbench/deployment tests**

Change the successful publication fixture to include canonical revision JSON and assert
that resolution exposes it:

```python
revision_metadata = {
    "champion_comparison": {
        "status": "COMPARED",
        "deployment_slot": "HOME_FREQ_UAT",
        "rate_package_id": 107,
        "reason": None,
    }
}
assert resolved["reviewed_champion_rate_package_id"] == 107
assert resolved["reviewed_champion_status"] == "COMPARED"
```

Update the notebook submission test so a fake current-champion helper would return 109,
but the request still sends reviewed package 107 without calling the helper:

```python
candidate.workbench.resolve_editor_publication = lambda submission: {
    "rate_package_id": 108,
    "package_version": 8,
    "model_run_id": 908,
    "reviewed_champion_status": "COMPARED",
    "reviewed_champion_rate_package_id": 107,
    "reviewed_deployment_slot": "HOME_FREQ_UAT",
    "reviewed_champion_reason": None,
}
candidate.workbench.current_champion_rate_package_id = lambda *args, **kwargs: pytest.fail(
    "deployment must not refresh champion evidence"
)
```

Add rejection tests for `UNAVAILABLE`, malformed metadata, and an explicit deployment
slot different from the reviewed slot.

- [ ] **Step 5: Run the workbench tests and confirm RED**

Run:

```bash
rtk proxy uv run pytest tests/test_candidate_workbench.py tests/test_candidate_editor.py -k 'publication or deployment' -q
```

Expected: failures because revision JSON is not selected/parsed and deployment refreshes
the champion.

- [ ] **Step 6: Parse and consume reviewed evidence**

Select `rp.revision_metadata_json` in `Workbench.resolve_editor_publication()`. Validate
that it is a JSON object with a `champion_comparison` object whose status, package ID,
slot, and reason obey the three-state contract. Return normalized fields in the result.

Add private publication fields to `EditorSubmission`:

```python
reviewed_champion_status: str | None = field(default=None, repr=False)
reviewed_champion_rate_package_id: int | None = field(default=None, repr=False)
reviewed_deployment_slot: str | None = field(default=None, repr=False)
reviewed_champion_reason: str | None = field(default=None, repr=False)
```

Populate them in `status()`. In `request_deployment()`, reject `UNAVAILABLE`, require a
package ID for `COMPARED`, require null for `NO_CHAMPION`, require the requested slot to
equal `reviewed_deployment_slot`, and pass the persisted ID directly as
`expected_current_rate_package_id`.

- [ ] **Step 7: Run Task 1 tests and confirm GREEN**

Run:

```bash
rtk proxy uv run pytest tests/test_editor_candidate_publisher.py tests/test_candidate_workbench.py tests/test_candidate_editor.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
rtk git add pricing_pipeline/publishing/editor_candidate.py pricing_pipeline/workbench/core.py pricing_pipeline/workbench/submission.py tests/test_editor_candidate_publisher.py tests/test_candidate_workbench.py tests/test_candidate_editor.py
rtk git commit -m "fix: deploy against reviewed champion evidence"
```

### Task 2: Make editor publication attempts immutable

**Files:**
- Modify: `pricing_pipeline/publishing/editor_candidate.py:1-100,469-610,780-875`
- Test: `tests/test_editor_candidate_publisher.py:1-220`

- [ ] **Step 1: Write failing retry and failure-injection tests**

Build a submission under a temporary workbench root and assert that two exported
attempts have different final paths:

```python
first = export_edited_model(
    parent,
    submission,
    allowed_root=tmp_path,
    write_dir=tmp_path / "published/.staging/attempt-a",
    published_dir=tmp_path / "published/attempts/attempt-a",
)
second = export_edited_model(
    parent,
    submission,
    allowed_root=tmp_path,
    write_dir=tmp_path / "published/.staging/attempt-b",
    published_dir=tmp_path / "published/attempts/attempt-b",
)
assert first.candidate_artifact_path != second.candidate_artifact_path
```

Add orchestration tests proving:

```python
# Existing complete SQL publication returns before export.
monkeypatch.setattr(editor_candidate, "_resolve_existing_editor_publication", lambda *a, **k: existing)
monkeypatch.setattr(editor_candidate, "export_edited_model", lambda *a, **k: pytest.fail("no write"))

# A package publication failure removes the new final directory only.
with pytest.raises(RuntimeError, match="injected"):
    publish_editor_submission(...)
assert not attempt.final_dir.exists()
assert committed_attempt.read_bytes() == committed_bytes
```

Also assert an existing candidate artifact is verified by path/hash/size before reuse and
that a package without exactly one successful run raises a repair-required error.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
rtk proxy uv run pytest tests/test_editor_candidate_publisher.py -k 'attempt or retry or existing_publication' -q
```

Expected: failures because editor output uses one deterministic `published/` directory
and existing SQL state is checked only after files are overwritten.

- [ ] **Step 3: Add attempt paths and same-host lock**

Add this focused internal value object and lock context:

```python
@dataclass(frozen=True)
class EditorPublicationAttempt:
    staging_dir: Path
    final_dir: Path


@contextmanager
def _editor_publication_lock(submission_dir: Path):
    lock_path = submission_dir / "publication.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
```

Create each attempt with `uuid4().hex` under
`published/.staging/<id>` and `published/attempts/<id>`. Both resolved directories must
remain beneath the verified submission/workbench roots.

- [ ] **Step 4: Rebase persisted paths before promotion**

Extend `export_edited_model()` with required `write_dir` and `published_dir` keyword
arguments. Write bytes only beneath `write_dir`, but construct all persisted paths from
`published_dir`. When the review hook returns staging metadata, retain its hash/size and
replace only its path before serializing the candidate bundle:

```python
persisted_review = {
    "path": str(published_dir / Path(review_artifact.path).name),
    "sha256": review_artifact.sha256,
    "size_bytes": review_artifact.size_bytes,
}
artifact = save_candidate_bundle(edited_bundle, write_dir / "candidate_bundle.joblib")
persisted_artifact = replace(
    artifact,
    path=str(published_dir / "candidate_bundle.joblib"),
)
```

After every file is created and hashed, rename the staging directory to its unique final
directory with `os.rename`; no file is rewritten after promotion.

- [ ] **Step 5: Resolve SQL state before any write**

Add `_resolve_existing_editor_publication()` using a package `LEFT JOIN MODEL_RUN` query
for exact model, parent package, and deterministic editor export ID. Its behavior is:

```python
if not rows:
    return None
if len(rows) != 1 or rows[0]["model_run_id"] is None:
    raise EditorSubmissionError("editor publication requires lineage repair")
if rows[0]["package_status"] != "PUBLISHED" or rows[0]["run_status"] != "SUCCESS":
    raise EditorSubmissionError("editor publication requires lineage repair")
load_candidate_bundle(...committed metadata..., allowed_root=allowed_root)
return EditorPublicationResult(..., was_existing=True)
```

Under the publication lock, call this resolver before loading the parent/champion or
exporting. When it returns a result, return immediately. When no package exists, remove
orphan directories for this submission, build/promote one attempt, stage, publish, and
retain it only after SQL success. Normal failures remove the current staging/final
attempt with `shutil.rmtree`; they never remove any directory returned by the SQL query.

- [ ] **Step 6: Run Task 2 tests and confirm GREEN**

Run:

```bash
rtk proxy uv run pytest tests/test_editor_candidate_publisher.py -q
```

Expected: all tests pass, including injected retry/failure cases.

- [ ] **Step 7: Commit Task 2**

```bash
rtk git add pricing_pipeline/publishing/editor_candidate.py tests/test_editor_candidate_publisher.py
rtk git commit -m "fix: publish editor artifacts immutably"
```

### Task 3: Commit scheduled packages and lineage atomically

**Files:**
- Modify: `pricing_pipeline/publishing/publisher.py:130-177`
- Modify: `pricing_pipeline/orchestration/pipeline.py:1-175`
- Modify: `pricing_pipeline/orchestration/publish_completed_build.py:564-700`
- Test: `tests/test_model_publisher.py:230-350`
- Test: `tests/test_rating_export.py:2280-2410`
- Test: `tests/test_publish_completed_build.py:529-910`

- [ ] **Step 1: Write failing atomic callback tests**

Change the publisher delegation test to provide and capture a lineage writer:

```python
writer = object()
result = ModelPublisher(engine, config()).publish_training_export(
    export,
    package_lineage_writer=writer,
)
assert publish_call["package_lineage_writer"] is writer
```

Change the pipeline test so the fake publisher invokes the callback with its live
connection and package ID, then assert `record_model_run_on_connection()` receives that
same connection:

```python
def publish_training_export(self, export, *, package_lineage_writer):
    package_lineage_writer(transaction_connection, 123)
    return publish_result

assert lineage_calls[0][0] is transaction_connection
assert lineage_calls[0][1]["rate_package_id"] == 123
assert standalone_record_calls == []
```

- [ ] **Step 2: Run atomicity tests and confirm RED**

Run:

```bash
rtk proxy uv run pytest tests/test_model_publisher.py tests/test_rating_export.py -k 'publish_training_export or publish_model_export' -q
```

Expected: failures because the publisher has no callback parameter and pipeline lineage
uses a second transaction.

- [ ] **Step 3: Move lineage into the package transaction**

Add a keyword-only callback to `ModelPublisher.publish_training_export()` and pass it to
`publish_rating_package()`:

```python
def publish_training_export(
    self,
    export: ModelExportResult | dict,
    *,
    package_lineage_writer=None,
) -> PublishResult:
    ...
    result = publish_rating_package(
        self.engine,
        export_id=export_result.export_id,
        created_by=export_result.created_by,
        package_status=self.config.default_package_status,
        package_lineage_writer=package_lineage_writer,
    )
```

In `publish_model_export()`, construct the complete lineage mapping before publication,
close over `model_run_id`, and call `record_model_run_on_connection()` from the callback.
Remove the standalone `record_model_run()` call. The callback first checks for an
existing successful run attached to the resolved package and returns it without updating
immutable paths; otherwise it records the new run.

- [ ] **Step 4: Write failing completed-export preflight tests**

Add tests for a canonical existing package/run resolver:

```python
existing = ExistingPublishedRun(
    model_id=17,
    model_name="HOME_FREQ",
    model_version="20260713.1",
    export_id="home-freq__scheduled-1",
    rate_package_id=107,
    package_version=7,
    package_status="PUBLISHED",
    model_run_id=907,
    manifest_id="original-manifest",
    split_set_id="original-split",
    rating_workbook_path=str(original / "rating_tables.xlsx"),
    publication_receipt_path=str(original / "publication_receipt.json"),
    publication_receipt_sha256="a" * 64,
)
```

Assert `publish_model_export()` returns this result before calling staging/publisher.
Assert a package row with null/failed/multiple run lineage raises a repair-required
integrity error. In `publish_completed_model_build()`, assert the returned manifest and
split set are the canonical values and a redundant new attempt directory beneath
`settings.workbench_artifact_root` is removed, while a path outside that root is retained.

- [ ] **Step 5: Run completed-export tests and confirm RED**

Run:

```bash
rtk proxy uv run pytest tests/test_publish_completed_build.py tests/test_rating_export.py -k 'existing or retry or repair' -q
```

Expected: the existing package conflicts on `source_file`, canonical manifest lineage is
not returned, or redundant artifacts remain.

- [ ] **Step 6: Add deterministic export preflight and safe cleanup**

In `pipeline.py`, add a frozen `ExistingPublishedRun` and a query joining
`PRICING_RATE_PACKAGE`, `PRICING_MODEL`, `MODEL_RUN`, and the training validation split
link by model ID and source export ID. Return `None` for no package; return exactly one
successful package/run; raise for incomplete or ambiguous state. Verify a present
candidate artifact with its committed metadata before reuse.

Run this resolver after registered-model validation and before staging. Return a result
dictionary containing canonical `manifest_id`, `split_set_id`, `model_run_id`, workbook,
receipt, package fields, and `was_existing=True`.

In `publish_completed_build.py`, prefer canonical fields from the publication result:

```python
result_manifest_id = str(publish_result.get("manifest_id") or manifest_id)
result_split_set_id = publish_result.get("split_set_id", split_set_id)
```

When `was_existing` is true, identify the common retry-local artifact parent from the
incoming workbook/receipt/candidate paths. Remove it only if it resolves under
`settings.workbench_artifact_root`, differs from every canonical referenced parent, and
contains the incoming build paths. Leave SQL manifest/split rows intact.

- [ ] **Step 7: Run Task 3 tests and confirm GREEN**

Run:

```bash
rtk proxy uv run pytest tests/test_model_publisher.py tests/test_rating_export.py tests/test_publish_completed_build.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
rtk git add pricing_pipeline/publishing/publisher.py pricing_pipeline/orchestration/pipeline.py pricing_pipeline/orchestration/publish_completed_build.py tests/test_model_publisher.py tests/test_rating_export.py tests/test_publish_completed_build.py
rtk git commit -m "fix: publish scheduled lineage atomically"
```

### Task 4: Export and stage two-way categorical interactions

**Files:**
- Modify: `pricing_pipeline/publishing/superglm_metadata.py:1-40,430-525`
- Modify: `pricing_pipeline/publishing/staging.py:40-360,470-540`
- Test: `tests/test_superglm_metadata.py`
- Test: `tests/test_rating_export.py:640-720,960-1030`

- [ ] **Step 1: Write failing real-SuperGLM receipt tests**

Fit a small model containing two `Categorical` main effects plus
`CategoricalInteraction("territory", "age_band")`, then assert:

```python
metadata = receipt.model_dump(mode="json")["term_metadata"]
interaction = metadata["territory_age_band"]
assert interaction["feature_kind"] == "categorical_interaction"
assert interaction["parent_names"] == ["territory", "age_band"]
assert interaction["input_column_names"] == ["territory", "age_band"]
assert interaction["interaction_order"] == 2
```

Parameterize `NumericInteraction`, `PolynomialInteraction`, and `TensorInteraction` model
metadata doubles and assert `build_superglm_publication_receipt()` raises a value error
naming the unsupported interaction term and categorical-only support.

- [ ] **Step 2: Run receipt tests and confirm RED**

Run:

```bash
rtk proxy uv run pytest tests/test_superglm_metadata.py -k 'interaction' -q
```

Expected: the receipt contains only `_specs` main effects and silently omits
`_interaction_specs`.

- [ ] **Step 3: Add interaction metadata extraction**

Import the four known SuperGLM interaction classes. Add ordered iteration over
`_interaction_specs` using `_interaction_order`, falling back to mapping order only when
the order attribute is absent. For `CategoricalInteraction`, emit:

```python
{
    "feature_kind": "categorical_interaction",
    "superglm_class": "CategoricalInteraction",
    "source_term_name": source_name,
    "published_term_name": published_name,
    "parent_names": list(spec.parent_names),
    "input_column_names": [overrides.get(name, clean_identifier(name)) for name in spec.parent_names],
    "interaction_order": 2,
    "declared": {},
    "effective": {"encoding": "categorical_cross_product"},
    "fitted": {},
}
```

Apply the same canonical-name collision checks as main effects. Raise before receipt
creation for every other interaction class or arity.

- [ ] **Step 4: Write failing workbook staging tests**

Use `export_rating_tables()` from the same fitted categorical-interaction model and its
receipt. Call `stage_rating_export()` with `insert_staging_frames` monkeypatched to
capture frames. Assert:

```python
interaction_rows = rate_df[rate_df.term_type == "CATEGORICAL_INTERACTION"]
assert len(interaction_rows) == len(territory_levels) * len(age_band_levels)
levels = level_df[level_df.row_id.isin(interaction_rows.row_id)]
assert set(levels.position_no) == {1, 2}
assert set(levels.feature_name) == {"territory", "age_band"}
assert "territory:age_band" not in main_effect_levels
```

Also assert malformed/ragged/non-positive matrices fail with the interaction term name.

- [ ] **Step 5: Run staging tests and confirm RED**

Run:

```bash
rtk proxy uv run pytest tests/test_rating_export.py -k 'interaction' -q
```

Expected: main-block parsing reaches the interaction header and tries to convert a
categorical column header to float.

- [ ] **Step 6: Parse receipt-declared interaction matrices**

Add `_receipt_interaction_features()` and set `args.interaction_features_json` before
`build_staging_frames()`. Extend `_receipt_term_type()` to map
`categorical_interaction` to `CATEGORICAL_INTERACTION`.

In `build_staging_frames()`:

1. locate every declared interaction title row by canonical/source term identity;
2. set the earliest title row as the hard stop for main-effect data;
3. require numeric, finite, positive multipliers within bounded main blocks;
4. find the matrix header whose first populated cell matches parent 1;
5. read parent-2 levels across columns and parent-1 levels down rows;
6. emit one rate row and two ordered level rows for each matrix cell; and
7. reject missing headers, duplicates, ragged cells, or invalid relativities.

Use the diagnostic key:

```python
cell_key = (
    f"{term_name}="
    f"{parents[0]}={left_level}|{parents[1]}={top_level}"
)
```

The normalized level rows, not this display key, are the scorer contract.

- [ ] **Step 7: Run Task 4 tests and confirm GREEN**

Run:

```bash
rtk proxy uv run pytest tests/test_superglm_metadata.py tests/test_rating_export.py -q
```

Expected: all tests pass, including the real exported interaction workbook.

- [ ] **Step 8: Commit Task 4**

```bash
rtk git add pricing_pipeline/publishing/superglm_metadata.py pricing_pipeline/publishing/staging.py tests/test_superglm_metadata.py tests/test_rating_export.py
rtk git commit -m "feat: stage categorical interaction relativities"
```

### Task 5: Correct numeric and interaction SQL scoring

**Files:**
- Modify: `db/migrations/V025__package_specific_scoring.sql:90-170`
- Modify: `pricing_pipeline/publishing/staging.py:280-390`
- Test: `tests/test_migrations.py:181-200`
- Test: `tests/test_rating_export.py:640-720`

- [ ] **Step 1: Write failing numeric staging and SQL contract tests**

Assert a real fitted numeric main effect stages one numeric `per_unit` component with its
receipt-derived input mapping. Add migration assertions requiring normalized mappings
and branch exclusions:

```python
assert "cell.term_type = 'NUMERIC_MAIN'" in sql
assert "TRY_CONVERT(FLOAT" in sql
assert "raw_value *" in sql
assert "PRICING_TERM_FEATURE" in sql
assert "PRICING_RATE_CELL_LEVEL" in sql
assert "cell.term_type = 'CATEGORICAL_INTERACTION'" in sql
assert "cell.term_type NOT IN ('NUMERIC_MAIN', 'CATEGORICAL_INTERACTION')" in sql
```

Use more specific normalized SQL fragments if formatting differs; the assertions must
prove numeric multiplication and all-component interaction matching, not only table-name
presence.

- [ ] **Step 2: Run scorer contract tests and confirm RED**

Run:

```bash
rtk proxy uv run pytest tests/test_migrations.py tests/test_rating_export.py -k 'numeric_main or package_specific_scorer' -q
```

Expected: V025 contains only band and exact term-name cell matching.

- [ ] **Step 3: Validate numeric per-unit staging**

After receipt types are applied, validate every `NUMERIC_MAIN` term has exactly one rate
row, one level row at position 1, `level_code == "per_unit"`, numeric feature type, and a
finite log coefficient. Raise a term-specific staging error on any violation.

- [ ] **Step 4: Add the `NUMERIC_MAIN` scorer branch**

Before generic cell matching, join each numeric term to its position-1
`PRICING_TERM_FEATURE`, read that row's `input_column_name` from JSON, and convert it to
float. Insert exactly one matched row using:

```sql
CAST(raw.raw_value * CAST(cell.log_coefficient AS FLOAT) AS FLOAT)
```

Set the breakdown multiplier to `EXP(raw_value * beta)`. Filter the compiled cell through
its position-1 level row with `level_code = 'per_unit'`. Null or failed conversion inserts
nothing so the existing error 50003 fires.

- [ ] **Step 5: Add the categorical-interaction scorer branch**

Before generic cell matching, consider only interaction cells for which no declared
term-feature/component pair is unmatched:

```sql
NOT EXISTS (
    SELECT 1
    FROM pricing.PRICING_TERM_FEATURE AS tf
    WHERE tf.term_id = cell.term_id
      AND NOT EXISTS (
          SELECT 1
          FROM pricing.PRICING_RATE_CELL_LEVEL AS rcl
          JOIN pricing.PRICING_FEATURE_LEVEL AS fl
            ON fl.feature_level_id = rcl.feature_level_id
          WHERE rcl.cell_id = cell.cell_id
            AND rcl.position_no = tf.position_no
            AND fl.level_code = JSON_VALUE(
                @features_json,
                CONCAT('$.', tf.input_column_name)
            )
      )
)
```

Add count equality checks so extra/missing cell levels cannot partially match. Ensure one
cell per interaction term is inserted. Exclude `NUMERIC_MAIN` and
`CATEGORICAL_INTERACTION` from the later generic exact-cell branch.

- [ ] **Step 6: Run Task 5 tests and confirm GREEN**

Run:

```bash
rtk proxy uv run pytest tests/test_migrations.py tests/test_rating_export.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 5**

```bash
rtk git add db/migrations/V025__package_specific_scoring.sql pricing_pipeline/publishing/staging.py tests/test_migrations.py tests/test_rating_export.py
rtk git commit -m "fix: score numeric and interaction terms"
```

### Task 6: Integrate, verify, and prepare the PR update

**Files:**
- Modify only files required by failures found during integration.

- [ ] **Step 1: Run the five focused regression groups**

```bash
rtk proxy uv run pytest tests/test_candidate_editor.py tests/test_candidate_workbench.py tests/test_editor_candidate_publisher.py tests/test_model_publisher.py tests/test_publish_completed_build.py tests/test_superglm_metadata.py tests/test_rating_export.py tests/test_migrations.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the full suite**

```bash
rtk proxy uv run pytest tests
```

Expected: all tests pass with zero failures or errors.

- [ ] **Step 3: Run static and whitespace verification**

```bash
rtk ruff check .
rtk git diff --check origin/feature/scaffolded-candidate-workbench...HEAD
rtk git status -sb
```

Expected: Ruff and diff checks are clean; status contains only intentional commits and
is ahead of the remote branch.

- [ ] **Step 4: Review the complete branch diff against the five findings**

```bash
rtk git diff --stat origin/feature/scaffolded-candidate-workbench...HEAD
rtk git log --oneline origin/feature/scaffolded-candidate-workbench..HEAD
```

Expected: the diff maps directly to champion evidence, immutable editor attempts,
atomic scheduled lineage, interaction staging/scoring, and numeric scoring.

- [ ] **Step 5: Commit any integration-only correction**

If integration required a source/test correction, stage only those files and use:

```bash
rtk git commit -m "fix: integrate PR review corrections"
```

If no integration correction exists, do not create an empty commit.

- [ ] **Step 6: Push the verified branch**

```bash
rtk git push origin feature/scaffolded-candidate-workbench
```

Expected: the draft PR branch updates successfully. Report verification evidence and
the five addressed thread URLs; do not post replies or resolve GitHub threads without a
separate explicit instruction.
