# PR 19 Publication Hardening Design

## Status

The user approved addressing all four unresolved PR 19 review findings on
2026-07-13. This specification records the accepted behavior before implementation.

## Goal

Make local and remote notebook publication safe under concurrent execution and make a
successful remote retry reusable only when every immutable item of audit evidence is
identical.

## Invariants

1. A model version is reserved once per `(model_id, export_id)` before fitting and is
   unique within a model, even when two remote notebooks start concurrently.
2. A package transaction can write lineage only for the exact staging export that it
   copied into the package.
3. Reusing an existing successful export requires exact agreement with all supplied
   immutable package, run, dataset, split, metric, fold, receipt, and candidate evidence.
4. Importing and using the notebook helpers works on native Windows as well as Linux/WSL.

## Remote model-version reservation

Add `pricing.PRICING_MODEL_VERSION_RESERVATION`, keyed by model and export with an
additional unique model/version constraint. The remote resolver will take an update/range
lock on the model row, reuse a published or reserved version for the same export, scan
root-package and reservation versions, and insert the next `vN` in one transaction.
Existing root packages are backfilled into the reservation table without allowing
duplicate historical rows to break the migration. SQLite keeps its existing reservation
table and transaction-based allocation.

## Staging-to-lineage binding

SQL Server staging replacement and package loading will both acquire the same
transaction-owned `sp_getapplock` resource derived from `export_id`. Package publication
will additionally compare the locked staging header with the export evidence supplied by
the caller: model identity/version, effective dates, resolved workbook path, and receipt
hash. A mismatch is an integrity error before package rows or lineage are written. The
lock helper is a no-op for SQLite and test doubles because SQLite publication is already
protected by the local publication lock.

## Existing-run retry validation

The existing-run resolver will read the package/run row plus all dataset, validation
split, aggregate metric, and fold-metric rows within one connection. It will compare them
to the incoming `ModelExportResult` after stable path/date/float normalization. Any
missing, extra, or changed evidence raises `PublishedRunIntegrityError`; only exact
evidence returns `was_existing=True`. The persisted candidate bundle is still verified
against its committed path, hash, size, format, and runtime metadata before reuse.

## Cross-platform local lock

Add one small standard-library file-lock context manager. It opens a sentinel file and
uses a lazily imported backend: `fcntl.flock` on POSIX and one-byte `msvcrt.locking` on
Windows. Offline publication, editor publication, and submission-root recovery all use
this helper, so none of those import `fcntl` at module load. No service, dependency, or
distributed-lock abstraction is introduced.

## Error handling and testing

- SQL application-lock acquisition failures name the export and abort the transaction.
- Reservation conflicts roll back; deterministic same-export retries return the stored
  reservation.
- Staging-header and retry-evidence conflicts report the mismatched field or collection.
- Unit tests first reproduce each review failure, including two pre-publication remote
  reservations and a simulated Windows lock backend.
- Migration contract tests cover the new table, constraints, and historical backfill.
- Focused suites, Ruff, compilation, whitespace checks, and the full test suite gate the
  PR update.

## Non-goals

- Distributed filesystem locking for notebook artifacts.
- Replacing SQL Server with MLflow or another service.
- Changing the analyst-facing notebook API or exposing reservation/audit fields.
- Repairing previously inconsistent packages automatically.

## Re-review addendum: immutable publication handoff

The second Codex review identified three remaining races in the gap between staging and
package publication. The accepted refinement keeps the analyst API unchanged:

- Staging computes one canonical SHA-256 over the export, rate-cell, cell-level, and
  term-metadata frames. SQL stores it on both the staging header and the package copied
  from that staging generation. The caller carries the digest internally between the two
  transactions, so replaced staging and an existing package built from different cells
  are both rejected. Opening an existing local SQLite store applies the two additive
  digest-column upgrades in place, preserving its data.
- Every newly inserted root package must find the `(model_id, export_id)` reservation
  under update/range locks and its staged `model_version` must equal the reserved value.
  Child editor packages continue to inherit the parent version and do not allocate a new
  trained-model version.
- Finding an existing package is validation-only. The package writer never invokes the
  lineage writer on that path; scheduled and editor callers resolve and validate the
  winner's complete durable lineage, then return it or raise an integrity error. A retry
  always restages first so the rate-cell digest is checked before an existing result can
  be reused.
