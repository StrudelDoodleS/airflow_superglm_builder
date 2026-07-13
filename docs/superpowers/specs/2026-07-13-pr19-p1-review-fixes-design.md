# PR 19 P1 Review Fixes Design

## Status

This design covers the five unresolved P1 review findings on draft PR 19. The
remediation scope was approved in conversation on 2026-07-13. Implementation remains
gated on review of this written specification.

## Context

PR 19 introduces the scaffolded candidate workbench: a scheduled SuperGLM build creates
an immutable SQL candidate and reopenable Python artifact, an analyst edits the model in
Jupyter, Airflow publishes the edited child candidate, and a separate deployment DAG can
move the live champion pointer.

The review found five correctness gaps at the boundaries between those steps:

1. deployment currently compares against the champion at button-click time instead of
   the champion whose performance evidence the analyst reviewed;
2. an editor retry can overwrite candidate files already referenced by a committed
   `MODEL_RUN`;
3. scheduled package publication and `MODEL_RUN` lineage commit in separate SQL
   transactions;
4. package scoring does not match categorical interactions through their component
   feature mappings; and
5. package scoring treats a continuous main-effect coefficient as a categorical cell
   instead of multiplying it by the raw numeric input.

These are release-blocking because they can either deploy a decision supported by stale
evidence, corrupt immutable lineage, leave a half-published run, or return an incorrect
score.

## Goals

- Make the exact champion used for editor comparison durable and enforce it as the
  deployment compare-and-swap expectation.
- Make editor publication artifacts immutable across retries and concurrent requests.
- Commit a scheduled rate package and its successful model-run lineage atomically.
- Stage and score SuperGLM continuous main effects with their real per-unit semantics.
- Stage and score categorical-by-categorical SuperGLM interactions through their
  component features.
- Keep the analyst experience unchanged: analysts select, edit, publish, and request
  deployment without entering package IDs, model versions, hashes, or lineage fields.
- Preserve the current trusted, single-host Windows/WSL deployment boundary and avoid a
  new service or general expression framework.

## Non-goals

- Automatic champion promotion or a generic challenger-selection policy.
- A distributed lock, shared object store, or multi-host notebook service.
- A transform language or automatic inversion of arbitrary feature transforms.
- Continuous, spline, tensor-product, or higher-order interaction export and scoring.
- Reconstructing fitted SuperGLM Python objects from SQL rows.
- Removing the legacy factory code. That is a separately requested cleanup and will be
  handled after these PR review blockers.

## Invariants

The implementation must preserve these invariants:

1. A committed `MODEL_RUN` points only to files whose bytes can no longer change.
2. One deterministic `export_id` identifies one immutable package/run result.
3. A successful publication exposes either both the rate package and successful model
   run or neither of them.
4. A deployment request is authorized by the evidence published with the edited
   candidate, not by a fresh read that silently changes the decision basis.
5. Every required rating term matches exactly once, or the scorer raises the existing
   missing-term error.
6. SQL remains the durable audit/index layer; candidate bundles remain the durable
   fitted-object layer.

## 1. Durable Champion Evidence

### Problem

`export_edited_model()` calculates champion comparison metrics while loading the parent
candidate. The resulting revision metadata currently records only whether comparison
was available. Later, `EditorSubmission.request_deployment()` reads the current champion
again and sends that new ID as `expected_current_rate_package_id`.

If package 107 supplied the reviewed comparison and package 109 became champion before
the analyst clicked deploy, the current code sends 109. The SQL compare-and-swap then
succeeds even though the analyst never reviewed evidence against 109.

### Champion snapshot

The champion lookup used by editor publication will return a value object containing:

- `deployment_slot`;
- the active `rate_package_id`, if one exists;
- the verified champion `CandidateBundle`, if it can be loaded; and
- a stable unavailability reason when it cannot be loaded.

The package ID comes from the same deployment row as the artifact fields. It must not be
looked up in a second query. This keeps the identity and comparison artifact from
describing different champions.

Editor revision metadata will store a `champion_comparison` object with this contract:

```json
{
  "status": "COMPARED",
  "deployment_slot": "PRODUCTION",
  "rate_package_id": 107,
  "reason": null
}
```

`status` has exactly three values:

- `COMPARED`: an active champion existed, its artifact was verified, and comparison
  metrics were calculated. `rate_package_id` is required.
- `NO_CHAMPION`: no active champion existed in the deployment slot when the editor
  publication loaded its comparison evidence. `rate_package_id` is null.
- `UNAVAILABLE`: an active champion existed but its fitted artifact could not be
  verified or compared. `rate_package_id` is required and `reason` is required.

The existing boolean `available` may be retained in the JSON for compatibility with
current readers, but `status` and `rate_package_id` are the authoritative deployment
fields. `COMPARED` maps to `available=true`; the other states map to `available=false`.

### Deployment behavior

After the publication DAG succeeds, `Workbench.resolve_editor_publication()` will read
and validate `revision_metadata_json` from the committed child package along with the
package and model-run IDs. `EditorSubmission.status()` will retain the resolved champion
snapshot on the in-memory submission object.

`request_deployment()` will behave as follows:

| Published comparison status | Deployment expectation | Result |
| --- | --- | --- |
| `COMPARED` | reviewed `rate_package_id` | trigger deployment DAG |
| `NO_CHAMPION` | SQL null / no active champion | trigger deployment DAG |
| `UNAVAILABLE` | none | reject before triggering Airflow |
| missing or malformed | none | reject as invalid publication lineage |

The persisted deployment slot is also part of the reviewed evidence. The request uses
that slot by default, and an explicit `deployment_slot` argument that differs after
normalization is rejected. Evidence from one slot can never authorize deployment to
another slot.

The request path will no longer call `current_champion_rate_package_id()`. That helper
may remain for read-only UI display, but it cannot supply deployment concurrency
control. The deployment DAG and stored procedure retain their existing compare-and-swap
check. Therefore, if the champion changes after comparison, the deployment fails safely
instead of silently changing the comparison basis.

The deployment request identity hash will include the persisted expectation. Repeating
the same request remains idempotent, while a different reason or slot produces a
different request identity as it does today.

## 2. Immutable Editor Publication Attempts

### Problem

Every editor publication currently writes to a deterministic `published/` directory.
On retry, `candidate_bundle.joblib`, workbooks, and the receipt can be overwritten before
SQL publication commits. If SQL then rejects the retry, the already committed
`MODEL_RUN` still contains the old hash and size but its path now contains new bytes.

### Directory layout

Each publication attempt will use a unique immutable directory beneath the submission
directory:

```text
<submission-parent>/
  publication.lock
  published/
    attempts/
      <attempt-id>/
        rating_tables.xlsx
        publication_receipt.json
        rating_tables_review.xlsx       # only when the model hook emits it
        candidate_bundle.joblib
    .staging/
      <attempt-id>/                      # exists only while files are being built
```

`attempt-id` is a generated opaque token. It is not the deterministic `export_id` and
is not analyst-supplied. All paths persisted to SQL point to the final
`published/attempts/<attempt-id>/` directory.

### Lock and publication sequence

Editor publication will use the same trusted-local filesystem lock primitive used by
submission recovery. The lock covers one submission and is held for the following
sequence:

1. Resolve the deterministic editor `export_id`.
2. Query SQL for an existing published package and successful model run for that exact
   model, parent package, and export ID.
3. If exactly one complete result exists, verify its referenced candidate artifact
   against the committed path, hash, size, format, and runtime metadata, then return its
   canonical SQL lineage without touching any artifact file. Missing or changed bytes
   are an integrity error, not a reason to republish under the same export ID.
4. If a package exists without exactly one successful run, fail with a lineage-repair
   error; do not create or overwrite artifacts.
5. Remove only abandoned staging/attempt directories proven not to be referenced by a
   committed model run for this submission. If the SQL reference check cannot complete,
   cleanup fails closed and leaves the directory in place.
6. Build every artifact in a new `.staging/<attempt-id>/` directory.
7. Hash and validate all files, then atomically rename that directory to
   `attempts/<attempt-id>/` on the same filesystem.
8. Stage the workbook and execute package publication plus model-run lineage in the
   existing single SQL transaction.
9. On a normal SQL failure, remove only this unpublished final attempt directory. Never
   remove an attempt referenced by SQL.
10. On success, retain the final directory permanently and return the committed IDs.

An abrupt process death can leave an unreferenced staging or final attempt. A later retry
cleans it only after checking SQL under the submission lock. A death after SQL commit is
safe: the next retry sees the complete result in step 3 and returns it without changing
the committed bytes.

The bundle's embedded review-artifact path, export result, and revision metadata must use
the final path, not the temporary staging path. The attempt's final location is therefore
calculated before export. File hashes are calculated from the staging files, all embedded
paths are explicitly rebased to the corresponding final locations before the bundle is
serialized, and the directory is then renamed without rewriting its contents.

### Concurrency boundary

The lock deliberately solves same-host notebook/Airflow concurrency, which is the
deployment model for this repository. SQL uniqueness and idempotency remain the final
defence if two different hosts are introduced accidentally, but distributed publication
is outside this change.

## 3. Atomic Scheduled Package and Model-Run Publication

### Problem

`publish_model_export()` currently calls `publish_training_export()`, commits the rate
package, and then calls `record_model_run()` in a second transaction. Failure between
those calls leaves a package with no successful run. A cleared retry may build a new
manifest directory under the same export ID; its different workbook `source_file` then
conflicts with the already committed package before lineage can recover.

### Transaction boundary

Scheduled publication will use the same transaction shape already used by editor
publication:

```text
stage export
  -> publish_rating_package(package_lineage_writer=...)
       SQL transaction:
         validate and insert/reuse package rows
         record MODEL_RUN, metrics, and fold metrics on the same connection
       commit
```

`publish_model_export()` will construct all lineage arguments before publication and
pass a callback into `ModelPublisher.publish_training_export()`. That callback receives
the package writer's live connection and resolved package ID, then calls
`record_model_run_on_connection()`. The outer standalone `record_model_run()` call will
be removed from this path.

If model-run insertion, metric insertion, or fold-metric insertion fails, the package
transaction rolls back. A retry then has no half-published package to conflict with.

### Existing-result preflight

Before replacing staging rows or comparing a new filesystem path, scheduled publication
will resolve the deterministic export ID in SQL:

- Exactly one published package plus exactly one successful model run returns the
  canonical existing result immediately. The returned manifest, split set, workbook
  path, package ID, version, status, and receipt identity come from SQL, not from the
  redundant retry payload.
- A package without one successful model run is treated as legacy/inconsistent state and
  raises an explicit repair error. New code cannot create this state.
- Multiple packages/runs, a failed run, or conflicting model identity raises an
  integrity error rather than guessing.

The early return occurs before staging mutation. If upstream training already created a
new retry-local artifact directory, the orchestration layer may remove that redundant
directory only when it is beneath the configured workbench root and differs from every
path referenced by the canonical run. Existing SQL manifest/split evidence remains
durable; this change does not introduce broad deletion of audit rows.

The same `export_id` remains immutable identity. A caller cannot use a cleared retry to
replace a successful export with changed data or files. It receives the existing result
or an explicit identity conflict.

### Crash outcomes

| Failure point | Durable result | Retry behavior |
| --- | --- | --- |
| before package transaction | no package/run | publish normally |
| during package or lineage insert | no package/run after rollback | publish normally |
| after transaction commit, before task response | package and successful run | return existing result |
| legacy package exists without run | inconsistent package only | stop with repair error |

## 4. Continuous `NUMERIC_MAIN` Scoring

### Required semantics

The SuperGLM exporter represents a continuous linear main effect as one `per_unit` log
coefficient. For input `x` and coefficient `beta`, its contribution is:

```text
log contribution = beta * x
multiplier       = exp(beta * x)
```

It is not an exact categorical lookup for a cell named `term=x`.

### Staging contract

A staged `NUMERIC_MAIN` term must contain:

- exactly one `PRICING_TERM_FEATURE` mapping at position 1;
- exactly one rate cell representing `per_unit`;
- a finite stored log coefficient; and
- a numeric feature value type.

Malformed continuous terms fail package validation/publication rather than producing an
ambiguous scorer result.

### Stored-procedure matching

`V025__package_specific_scoring.sql` will add a `NUMERIC_MAIN` branch before generic
categorical cell matching. For each such term it will:

1. obtain the JSON input using `PRICING_TERM_FEATURE.input_column_name`;
2. convert the value with `TRY_CONVERT(FLOAT, ...)`;
3. find the single `per_unit` compiled cell;
4. insert one `@matched` row with `log_coefficient = raw_value * beta`; and
5. report the model feature/input column and raw input in the score breakdown.

A missing, null, non-numeric, NaN-like, or otherwise unconvertible value produces no
match. The existing final term-count check then throws error 50003. Zero is valid and
contributes a multiplier of one. Negative values are valid unless rejected by upstream
model semantics.

Generic exact-cell matching will exclude `NUMERIC_MAIN`, preventing a term from matching
twice.

## 5. Categorical Interaction Staging and Scoring

### Supported scope

This change supports categorical-by-categorical SuperGLM interactions exported as a
two-dimensional relativity matrix. Both parent features and their order are preserved.

The receipt builder will inspect SuperGLM interaction metadata in addition to main-effect
metadata. It will emit, for every supported interaction:

- the canonical published term name;
- `feature_kind = "categorical_interaction"`;
- the ordered parent feature names from `parent_names`;
- the corresponding ordered SQL input-column names; and
- the interaction order/arity, which must equal two for this release.

If the fitted model contains a numeric, spline, tensor, polynomial, mixed-type,
higher-order, or otherwise unsupported interaction, receipt creation fails before any
workbook or package is published. The exception names the term and supported contract.
Silently omitting an interaction is forbidden.

### Workbook parsing

The staging parser will use the verified receipt as the semantic source of term identity
and parent order. It will no longer read every main-effect block to the bottom of the
sheet.

Main-effect parsing will consume only the contiguous rows belonging to that main block.
Receipt-declared interaction title rows establish a hard end boundary for every main
block; an earlier blank boundary also ends the block. Inside those bounds, a non-numeric
multiplier is a staging error rather than a silent terminator. This prevents interaction
matrix headers such as `old` and `young` from being interpreted as main-effect
multipliers without hiding a malformed main-effect row.

For each receipt-declared interaction, the parser will locate the exact interaction
title in the exporter sheet, then locate its matrix header. The matrix contract is:

- the first header cell identifies parent feature 1;
- the remaining populated header cells are levels of parent feature 2;
- each following populated row begins with a level of parent feature 1; and
- each matrix value is the relativity for that ordered pair.

Parsing stops at a blank matrix boundary or the next declared term. Missing headers,
duplicate levels, non-finite/non-positive relativities, ragged matrices, an unexpected
parent name, or a duplicate pair raises a staging error with the interaction term name.

Each matrix value creates:

- one rate-cell row with `term_type = CATEGORICAL_INTERACTION`;
- two ordered cell-level rows, positions 1 and 2;
- one ordered term-feature mapping for each parent; and
- `log_coefficient = log(multiplier)`.

The cell key is deterministic and diagnostic, for example:

```text
territory:age_band=territory=urban|age_band=young
```

SQL matching does not parse this display key. It relies on the normalized component
rows. Reference combinations with relativity `1.0` remain explicit cells and contribute
zero on the log scale.

### Stored-procedure matching

`V025__package_specific_scoring.sql` will add a categorical-interaction branch before
generic cell matching. A compiled interaction cell matches only when every ordered
component satisfies all of the following:

- the cell level belongs to the same term and rate cell;
- its position matches the corresponding `PRICING_TERM_FEATURE.position_no`;
- the term-feature row supplies the JSON `input_column_name`; and
- the JSON value exactly equals that component's `level_code`.

The branch also verifies that the count of matched components equals the declared term
feature count and cell level count. This prevents partial matches. It inserts at most one
`@matched` row per interaction term; zero or multiple matching cells is an integrity
failure, never an arbitrary choice.

Generic exact-cell matching will exclude `CATEGORICAL_INTERACTION`. The breakdown will
identify the interaction term and a canonical representation of its component inputs.

## Metadata and API Changes

The changes are deliberately small and domain-specific:

- `ParentCandidate` gains the champion package snapshot, not just an optional bundle and
  reason.
- `EditorExport.revision_metadata_json` carries the durable comparison contract.
- `Workbench.resolve_editor_publication()` returns the parsed reviewed-champion
  expectation together with package/run lineage.
- `EditorSubmission` stores that resolved expectation after successful status polling.
- `ModelPublisher.publish_training_export()` accepts the package-lineage writer needed
  to share the package transaction.
- the SuperGLM receipt term metadata gains supported categorical interaction entries;
  no generic transform or expression abstraction is introduced.

No analyst enters or edits these fields. They are derived from the deployment row,
fitted SuperGLM object, receipt, Airflow context, and SQL-assigned identities.

## Error Handling

Errors will be actionable and preserve durable state:

- stale reviewed champion: deployment DAG compare-and-swap failure naming the expected
  and current package IDs;
- champion exists but comparison unavailable: notebook-side deployment rejection naming
  the recorded reason;
- malformed revision metadata: candidate-lineage error before Airflow is triggered;
- existing package without successful run: explicit repair-required publication error;
- editor SQL failure: only the current unpublished attempt is removed;
- unsupported interaction: export/receipt error naming the interaction and supported
  categorical-by-categorical scope;
- malformed interaction matrix: staging error naming the matrix defect and term;
- missing/non-numeric continuous input or unmatched interaction component: scorer error
  50003 through the existing required-term check.

No failure path overwrites or removes an artifact referenced by a committed model run.

## Migration Strategy

The scoring procedure is introduced by `V025__package_specific_scoring.sql` on this
unmerged feature branch. Both scoring corrections will therefore be made in V025 itself
rather than adding a follow-up migration that assumes the broken procedure was deployed.
Tests will treat V025 as the authoritative fresh-install definition. Any developer
database that already applied the draft V025 must be rebuilt or have that draft migration
reapplied according to the repository's existing local migration workflow.

No schema column is required for champion evidence because the immutable revision JSON
already belongs to the child package. Interaction scoring uses the existing normalized
`PRICING_TERM_FEATURE` and `PRICING_RATE_CELL_LEVEL` tables.

## Test Strategy

Each review finding will first receive a failing regression test that demonstrates the
reported behavior, followed by the smallest implementation that makes it pass.

### Champion evidence tests

- Publication records the exact champion package ID used to calculate metrics.
- `NO_CHAMPION`, `COMPARED`, and `UNAVAILABLE` metadata validate correctly.
- Deployment uses the recorded ID even when a later current-champion query would return
  another ID.
- `UNAVAILABLE` and malformed/missing evidence cannot trigger Airflow.
- SQL compare-and-swap rejects a deployment after the reviewed champion changes.

### Editor artifact tests

- Two distinct attempts never share an artifact path.
- A successful retry returns before exporter/file writes.
- Injected SQL failure removes only the current unpublished attempt.
- A retry after failure creates new paths and leaves committed bytes/hash/size unchanged.
- Concurrent same-submission publication is serialized and resolves one package/run.
- Crash-recovery cleanup refuses to delete any SQL-referenced directory.

### Scheduled atomicity tests

- Injected model-run insertion failure rolls back the package.
- Metrics/fold-metric failure also rolls back the package.
- A retry after rollback succeeds with a new attempt path.
- A complete existing package/run returns canonical lineage before staging replacement.
- A package without a successful run raises the repair-required error.
- A changed payload cannot replace a successful export ID.

### Numeric scoring tests

- A real fitted SuperGLM numeric main effect stages as one `per_unit` coefficient.
- V025 uses the mapped input column and multiplies beta by positive, zero, and negative
  raw values.
- Missing and non-numeric inputs fail the required-term check.
- `NUMERIC_MAIN` is excluded from generic exact-cell matching.

### Interaction tests

- A real fitted categorical interaction receipt records both ordered parents.
- Main-effect staging stops before the interaction section.
- The exported interaction matrix produces the expected cells, two ordered levels per
  cell, and two term-feature mappings.
- The scorer matches the same pair through component input columns even though the term
  name is not a JSON feature key.
- An input pair absent from the exported matrix fails the required-term check; this
  change does not invent a fallback combination.
- Unsupported interaction types fail before publication.

Where the repository's SQL integration harness is available, scorer tests will execute
the stored procedure. Text/contract tests will additionally assert the migration uses
the normalized mapping tables and excludes numeric/interaction terms from the generic
branch.

Focused tests will run after each change. Final verification will run the full test
suite, Ruff, and Git whitespace checks using the repository's configured commands.

## Review-Thread Acceptance Map

| Review finding | Acceptance condition |
| --- | --- |
| stale champion evidence | deployed expectation is the package persisted with the reviewed comparison; any later champion change fails CAS |
| editor bundle overwrite | every committed run references a unique immutable attempt directory; retries return or create a new directory |
| scheduled half-publication | package, model run, metrics, and fold metrics share one SQL commit/rollback boundary |
| interaction term-name lookup | categorical interactions stage normalized component mappings and score through those mappings |
| numeric exact-cell lookup | numeric main effects apply the per-unit coefficient to the raw mapped input |

All five threads are ready to resolve only after their regression tests pass, full
verification succeeds, and the updated branch is pushed. GitHub replies and thread
resolution remain a separate explicit review action after implementation.
