# SuperGLM Publication Receipt Design

## Problem

The rating package stores operational multipliers, but it does not store enough
metadata to explain how each SuperGLM term was produced.

This is already visible for offsets. A model fitted with:

```python
offset = np.log(term_months / 12)
```

can export a deployable `TermMonths` factor:

```text
TermMonths=12 -> 1.0
TermMonths=36 -> 3.0
```

That works operationally, but the package must also record that `TermMonths` is
an offset factor with fixed log coefficient `1.0`, and that the source transform
was `log(TermMonths / 12)`.

The same audit gap exists for other SuperGLM feature types:

- categorical terms have a base-level rule and a fitted base level;
- ordered categoricals may use a spline or step basis;
- splines have kind, knot strategy, fitted knots, constraints, and boundaries;
- polynomial terms have degree and fitted scaling bounds;
- interactions have parent terms and interaction type.

The workbook and normalized rate cells remain the scoring truth. The new
metadata is a publication receipt that explains those operational cells.

## Current Context

The fitted SuperGLM object exists inside the model-owned training/export
function. That function currently writes the rating workbook and model pickle,
then returns a small `CompletedModelBuild` payload through Airflow/XCom.

Publishing does not receive the fitted object. It receives paths and metadata,
then stages the workbook and writes the package.

That means metadata extraction must happen during training, while the fitted
model is still in memory. Publishing should not unpickle the model just to
extract metadata. Unpickling would couple publishing to the exact SuperGLM and
Python version and would make pickle loading part of the trusted publication
boundary.

The current staging path reads the workbook and infers term type from visible
cells. That is enough to stage multipliers, but not enough to recover the
SuperGLM feature specification that produced those multipliers.

## Goals

- Write a versioned SuperGLM publication receipt during training/export.
- Carry that receipt across Airflow as a path plus SHA-256 hash, not as a large
  dictionary in XCom.
- Stage and publish package-level metadata, offset metadata, and term metadata.
- Record metadata for categorical, ordered categorical, spline, polynomial,
  numeric, and offset terms.
- Record interaction metadata only when interaction terms are operationally
  staged in the workbook.
- Record both declared/requested feature settings, effective settings after
  model defaults, and fitted/auditable state when available.
- Mark exported offset factors distinctly from ordinary categorical terms.
- Include the publication receipt hash in package idempotency checks.
- Preserve metadata through manual revisions.
- Keep SQL scoring unchanged. Multipliers remain the operational truth.
- Support both SQL Server and offline SQLite DDL.

## Non-Goals

- Do not reconstruct arbitrary Python constructor calls from metadata.
- Do not make SQL scoring evaluate splines, bases, penalties, or constraints.
- Do not store row-level training data or model frames in SQL.
- Do not require every model family to use this metadata contract. This design
  is explicitly SuperGLM-specific.
- Do not put every detail into `term_type`. `term_type` remains a coarse
  operational label.
- Do not implement interaction workbook staging in this PR unless explicitly
  included in the implementation plan.
- Do not add `term_metadata_json` to compiled cells in v1. Compiled cells can
  join to `PRICING_TERM`.

## Decision

Add a single validated publication receipt sidecar:

```text
rating_tables.xlsx
superglm_publication_receipt.json
superglm_model.pkl
```

The model-owned training/export function writes the receipt and returns:

```python
publication_receipt_path: str | None
publication_receipt_sha256: str | None
```

Add these fields to:

```text
CompletedModelBuild
ModelExportResult
MODEL_RUN lineage
```

Publishing stages both the workbook and the receipt, then writes the package
atomically.

## Publication Receipt Contract

Use one validated receipt instead of loosely related dictionaries:

```python
class OffsetExportContract(BaseModel):
    handling: Literal[
        "NONE",
        "EXPORTED_FACTOR",
        "ALREADY_APPLIED_SQL_EXPOSURE",
    ]
    source_factor_name: str | None = None
    published_factor_name: str | None = None
    source_name: str | None = None
    label: str | None = None


class SuperGLMPublicationReceipt(BaseModel):
    schema_name: Literal["superglm_publication_receipt"]
    schema_version: Literal[1]
    metadata_origin: Literal["SUPERGLM_FITTED_MODEL"]
    superglm_version: str
    extractor_version: str
    package_metadata: dict[str, Any]
    term_metadata: dict[str, dict[str, Any]]
    offset_contract: OffsetExportContract
```

The package metadata must include at least this model envelope:

```json
{
  "model": {
    "family": "poisson",
    "link": "log",
    "fit_used_offset": true
  }
}
```

The full receipt is stored on the package as `publication_receipt_json`. The
per-term `term_metadata_json` rows are query-friendly denormalizations of the
same receipt.

The canonical receipt hash must be computed from the exact bytes written to the
sidecar:

```python
def canonical_receipt_bytes(receipt: SuperGLMPublicationReceipt) -> bytes:
    return json.dumps(
        receipt.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


canonical = canonical_receipt_bytes(receipt)
path.write_bytes(canonical)
digest = hashlib.sha256(canonical).hexdigest()
```

`allow_nan=False` is required so NaN and infinity fail before publication.

The publisher must:

```text
1. read the sidecar file once;
2. validate it as strict JSON and as SuperGLMPublicationReceipt;
3. recreate canonical receipt bytes;
4. verify raw file bytes equal canonical receipt bytes;
5. calculate SHA-256 from those canonical bytes;
6. compare against a validated 64-character lowercase hex digest.
```

This avoids formatting, trailing-newline, dictionary-order, and permissive NaN
differences.

## Airflow And Publish Data Flow

Preferred custom model flow:

```text
fit SuperGLM
export workbook using the offset contract
extract and normalize publication receipt from fitted model
write superglm_publication_receipt.json
log receipt sidecar as an MLflow artifact when MLflow is enabled
return workbook path + receipt path + receipt hash
stage workbook and receipt
publish package in one transaction
```

The completed-build payload must carry only small fields:

```text
rating_workbook_path
model_artifact_path
publication_receipt_path
publication_receipt_sha256
model_version
effective_from
export_id
manifest_id
split_set_id
metrics
```

The receipt file is the metadata boundary. Publishing validates the hash before
trusting the file.

## Offset Contract Controls Workbook Export

Offset handling is not only a publish-time concern. It must control workbook
generation.

For an exported offset factor:

```python
export_rating_tables(
    fitted,
    X,
    y,
    exposure,
    output_path=workbook_path,
    offset=offset,
    offset_source=frame["TermMonths"],
    offset_name=offset_contract.source_factor_name,
)
```

The offset source does not need to be part of `X`. For example, `Exposure` may
create `log(Exposure)` while being deliberately excluded from the feature frame.

Only names and transform labels enter the receipt. The source series itself is
not persisted.

For external offset handling:

```python
offset_contract = OffsetExportContract(
    handling="ALREADY_APPLIED_SQL_EXPOSURE",
    source_name="Exposure",
    label="log(Exposure)",
)
```

SuperGLM needs an export option to suppress the offset workbook block, for
example:

```python
include_offset: bool = True
```

or:

```python
offset_export: Literal["auto", "factor", "none"] = "auto"
```

Without that SuperGLM capability, `ALREADY_APPLIED_SQL_EXPOSURE` cannot be
implemented safely because the workbook may still contain an offset block that
would be staged and double counted.

SuperGLM's current source-aware offset export also has
`offset_max_exact_levels=20` by default. Model-owned export code must pass a
higher explicit value when an exported offset source legitimately has more than
20 distinct tariff levels.

## Offset Validation

Supported handling values:

```text
NONE
EXPORTED_FACTOR
ALREADY_APPLIED_SQL_EXPOSURE
```

Cross-field rules:

```text
NONE
  source_factor_name, published_factor_name, source_name, and label must be null

EXPORTED_FACTOR
  source_factor_name, published_factor_name, source_name, and label are required

ALREADY_APPLIED_SQL_EXPOSURE
  source_factor_name and published_factor_name must be null
  source_name and label are required
```

Validation rules:

```text
model fitted without offset + NONE
  valid

model fitted with offset + EXPORTED_FACTOR
  valid only if the declared factor exists in staged terms

model fitted with offset + ALREADY_APPLIED_SQL_EXPOSURE
  valid only if no offset factor is staged

model fitted with offset + NONE or missing handling
  error

model fitted without offset + EXPORTED_FACTOR
  error

declared offset factor exists but package says NONE
  error

external offset handling plus staged offset factor
  error, because it risks double counting
```

For an exported transformed offset:

```text
PRICING_RATE_PACKAGE.offset_handling     = EXPORTED_FACTOR
PRICING_RATE_PACKAGE.offset_factor_name  = TermMonths
PRICING_RATE_PACKAGE.offset_source_name  = TermMonths
PRICING_RATE_PACKAGE.offset_label        = log(TermMonths / 12)

PRICING_TERM.term_name                   = TermMonths
PRICING_TERM.term_type                   = OFFSET_FACTOR
```

Offset term metadata:

```json
{
  "feature_kind": "offset",
  "source_term_name": "TermMonths",
  "published_term_name": "TermMonths",
  "offset_handling": "EXPORTED_FACTOR",
  "fixed_log_coefficient": 1.0,
  "coefficient_source": "offset",
  "source_factor_name": "TermMonths",
  "published_factor_name": "TermMonths",
  "offset_source_name": "TermMonths",
  "offset_label": "log(TermMonths / 12)"
}
```

The rate cells remain normal multiplier cells:

```text
TermMonths=12 -> 1.0
TermMonths=36 -> 3.0
```

## SQL Scoring Offset Semantics

The current SQL prediction procedure calculates:

```text
prediction = base_rate * @exposure * EXP(SUM(log_coefficient))
```

Therefore the caller must apply offset/exposure consistently with the package
metadata:

```text
offset_handling = EXPORTED_FACTOR
  The offset is already represented by an `OFFSET_FACTOR` term in
  EXP(SUM(log_coefficient)).

  Call SQL scoring with @exposure = 1.0 for that offset component.
  Example: if TermMonths=36 exported multiplier 3.0, do not also pass
  @exposure = 3.0 for the same term exposure.

offset_handling = ALREADY_APPLIED_SQL_EXPOSURE
  No offset factor is staged.

  Pass EXP(offset) through @exposure.
  Example: offset = log(TermMonths / 12), TermMonths=36, pass @exposure = 3.0.

offset_handling = NONE
  No model offset exists.

  Pass normal policy exposure semantics through @exposure as required by the
  product/rating calculation.
```

This is a caller contract for v1. The stored procedure does not infer offset
handling internally unless a later scoring API explicitly adds that dispatch.

## DDL Changes

Add package-level metadata to `PRICING_RATE_PACKAGE`:

```text
publication_receipt_json    nullable
publication_receipt_sha256  nullable
package_metadata_json       nullable
revision_metadata_json      nullable
offset_handling             not null
offset_factor_name          nullable
offset_source_name          nullable
offset_label                nullable
metadata_origin             nullable
```

Allowed final `offset_handling` values:

```text
NONE
EXPORTED_FACTOR
ALREADY_APPLIED_SQL_EXPOSURE
UNKNOWN
```

`UNKNOWN` is for legacy/workbook-only compatibility. New custom DAG publishes
should not use `UNKNOWN`.

Add staging columns to `pricing_stg.STG_RATING_EXPORT`:

```text
publication_receipt_json    nullable
publication_receipt_sha256  nullable
package_metadata_json       nullable
offset_handling             nullable
offset_factor_name          nullable
offset_source_name          nullable
offset_label                nullable
metadata_origin             nullable
```

Add term metadata staging:

```text
pricing_stg.STG_TERM_METADATA
  export_id                 not null
  term_name                 not null
  term_metadata_json         not null
  primary key (export_id, term_name)
  foreign key (export_id) references pricing_stg.STG_RATING_EXPORT(export_id)
```

Add term metadata to final package terms:

```text
pricing.PRICING_TERM
  term_metadata_json         nullable
```

Add receipt lineage to `pricing.MODEL_RUN`:

```text
publication_receipt_path     nullable
publication_receipt_sha256   nullable
```

For SQL Server JSON fields:

```sql
CHECK (
    publication_receipt_json IS NULL
    OR ISJSON(publication_receipt_json) = 1
);

CHECK (
    package_metadata_json IS NULL
    OR ISJSON(package_metadata_json) = 1
);

CHECK (
    revision_metadata_json IS NULL
    OR ISJSON(revision_metadata_json) = 1
);

CHECK (
    term_metadata_json IS NULL
    OR ISJSON(term_metadata_json) = 1
);

CHECK (
    offset_handling IN (
        'NONE',
        'EXPORTED_FACTOR',
        'ALREADY_APPLIED_SQL_EXPOSURE',
        'UNKNOWN'
    )
);
```

Migration order for existing package rows:

```text
1. add offset_handling nullable
2. backfill existing rows to UNKNOWN
3. alter offset_handling to NOT NULL
4. add enum and JSON constraints
```

`publication_receipt_sha256` must be either null or a 64-character lowercase
hex digest.

Use a new migration rather than editing original core/staging migrations. Mirror
equivalent columns/tables in offline SQLite DDL. Existing SQLite files created
with `CREATE TABLE IF NOT EXISTS` will not gain columns, so smoke runs must
reset local SQLite files or apply an explicit local migration.

Staging replacement must delete `STG_TERM_METADATA` before deleting its parent
`STG_RATING_EXPORT` row.

## Term Types

`term_type` describes the operational scoring shape, not the full model origin.
The receipt metadata explains the model origin.

Keep `term_type` coarse:

```text
CATEGORICAL_MAIN
ORDERED_CATEGORICAL_MAIN
DISCRETIZED_SPLINE_1D
NUMERIC_MAIN
NUMERIC_BANDED_1D
OFFSET_FACTOR
CATEGORICAL_INTERACTION
NUMERIC_INTERACTION
SPLINE_INTERACTION
POLYNOMIAL_INTERACTION
```

`OFFSET_FACTOR` is a real term type because it affects package interpretation:
the factor is operational, but it came from a fitted offset with fixed
coefficient `1.0`.

Detailed SuperGLM settings go in `term_metadata_json`.

Receipt-to-type mapping for staged main effects:

| Receipt metadata | Workbook/staging shape | `term_type` |
| --- | --- | --- |
| `feature_kind = categorical` | level lookup | `CATEGORICAL_MAIN` |
| `feature_kind = ordered_categorical` | level lookup | `ORDERED_CATEGORICAL_MAIN` |
| `feature_kind = spline` | interval lookup | `DISCRETIZED_SPLINE_1D` |
| `feature_kind = polynomial` | interval lookup | `NUMERIC_BANDED_1D` |
| `feature_kind = numeric` | per-unit row | `NUMERIC_MAIN` |
| `feature_kind = offset` | exported offset lookup | `OFFSET_FACTOR` |

`ORDERED_CATEGORICAL_MAIN` must score the same way as `CATEGORICAL_MAIN`; it
exists to preserve the operational distinction that the level lookup came from
an ordered categorical term. `NUMERIC_BANDED_1D` remains in the compiled 1D band
path. Do not use `POLYNOMIAL_MAIN` for polynomial rating tables unless the
compiled-band and scoring dispatch are updated to support it.

## Interaction Scope

SuperGLM can export interactions as two-dimensional matrices lower in the
workbook, while the current staging parser only reads three-column main-effect
blocks from the configured term/header rows.

V1 metadata must be persisted only for terms that are operationally staged.
Interaction receipts are deferred unless the implementation also adds
interaction workbook staging.

If interaction staging is included, use this coarse mapping:

| SuperGLM class | Operational `term_type` |
| --- | --- |
| `CategoricalInteraction` | `CATEGORICAL_INTERACTION` |
| `NumericInteraction` | `NUMERIC_INTERACTION` |
| `NumericCategorical` | `NUMERIC_INTERACTION` |
| `SplineCategorical` | `SPLINE_INTERACTION` |
| `TensorInteraction` | `SPLINE_INTERACTION` |
| `PolynomialInteraction` | `POLYNOMIAL_INTERACTION` |
| `PolynomialCategorical` | `POLYNOMIAL_INTERACTION` |

If interaction metadata is extracted but no matching staged interaction term
exists, publication must not fail in v1. It should omit that metadata from
`term_metadata`. If this information is useful, list it in a package-level
`non_operational_interactions` array instead. Do not write
`STG_TERM_METADATA` rows for terms that do not exist operationally.

## SuperGLM Metadata Extractor

Add a SuperGLM-specific extractor module, for example:

```text
pricing_pipeline/publishing/superglm_metadata.py
```

Public shape:

```python
def build_superglm_publication_receipt(
    model,
    *,
    offset_contract: OffsetExportContract,
    source_to_published_names: Mapping[str, str] | None = None,
) -> SuperGLMPublicationReceipt:
    ...
```

The extractor may inspect SuperGLM internals because this metadata is explicitly
SuperGLM-specific:

```text
model._specs
model._feature_order
model._interaction_specs
model._interaction_order
model._fit_used_offset
```

The extractor must be defensive:

- omit optional fields that are absent;
- serialize numpy/pandas values into JSON-safe Python values;
- store class names rather than raw Python objects;
- reject non-finite values;
- normalize grouping metadata instead of serializing grouping objects;
- avoid failing publication for unknown future SuperGLM fields unless the field
  is required for offset validation.

## Naming

The staging parser normalizes workbook term names with `clean_identifier()`.
The receipt must preserve both names:

```json
{
  "source_term_name": "Term Months",
  "published_term_name": "Term_Months"
}
```

Move the canonicalizer into a shared naming module used by both receipt
extraction and staging. The receipt writer must use the same canonicalizer as
staging and reject name collisions, such as two source names that both become
`Term_Months`.

For offsets:

```text
source_factor_name
  workbook block title passed to SuperGLM export

published_factor_name
  canonical term identifier stored in PRICING_TERM.term_name
```

`PRICING_RATE_PACKAGE.offset_factor_name` must equal
`PRICING_TERM.term_name`, so it stores `published_factor_name`, not the raw
source/export display name.

## Metadata Shapes

Each term receipt should use:

```json
{
  "feature_kind": "...",
  "superglm_class": "...",
  "source_term_name": "...",
  "published_term_name": "...",
  "declared": {},
  "effective": {},
  "fitted": {}
}
```

`declared` is what the model author requested.

`effective` is after model defaults or SuperGLM normalization are applied.

`fitted` is state learned from data during fitting.

### Categorical

```json
{
  "feature_kind": "categorical",
  "superglm_class": "Categorical",
  "source_term_name": "Area",
  "published_term_name": "Area",
  "declared": {
    "base": "most_exposed",
    "grouping": null
  },
  "effective": {},
  "fitted": {
    "levels": ["A", "B", "C"],
    "base_level": "A",
    "non_base_levels": ["B", "C"]
  }
}
```

For grouping, store normalized maps:

```json
{
  "grouping": {
    "original_to_group": {"A1": "A", "A2": "A"},
    "group_to_originals": {"A": ["A1", "A2"]},
    "grouped_levels": ["A"]
  }
}
```

### Ordered Categorical

```json
{
  "feature_kind": "ordered_categorical",
  "superglm_class": "OrderedCategorical",
  "source_term_name": "VehicleBand",
  "published_term_name": "VehicleBand",
  "declared": {
    "basis": "spline",
    "kind": "ps",
    "base": "most_exposed",
    "n_knots_requested": 5,
    "degree": 3,
    "select": false,
    "penalty": "ssp",
    "ordered_levels": ["low", "medium", "high"],
    "level_to_value": {
      "low": 0.0,
      "medium": 0.5,
      "high": 1.0
    }
  },
  "effective": {
    "n_knots_effective": 2
  },
  "fitted": {
    "base_level": "medium",
    "non_base_levels": ["low", "high"],
    "n_levels": 3
  },
  "spline": {
    "declared": {
      "class_name": "PSpline",
      "n_knots_requested": 5,
      "degree": 3,
      "penalty": "ssp",
      "select": false
    },
    "effective": {
      "n_knots_effective": 2
    },
    "fitted": {
      "class_name": "PSpline",
      "boundary": [0.0, 1.0],
      "knots": [0.5]
    }
  }
}
```

A supplied `Spline(...)` object is copied by SuperGLM and its knot count may be
clamped to the number of ordered levels. Store requested and effective values
separately.

### Spline

```json
{
  "feature_kind": "spline",
  "superglm_class": "PSpline",
  "source_term_name": "DrivAge",
  "published_term_name": "DrivAge",
  "declared": {
    "kind": "ps",
    "n_knots": 10,
    "degree": 3,
    "knot_strategy": "quantile",
    "penalty": "ssp",
    "select": true,
    "discrete": true,
    "n_bins": 256,
    "extrapolation": "clip",
    "constraint_kind": "increasing",
    "constraint_mode": "postfit",
    "m": [2],
    "knot_alpha": 0.2,
    "explicit_knots": null,
    "explicit_boundary": null,
    "lambda_policy": null
  },
  "effective": {
    "kind": "ps",
    "class_name": "PSpline",
    "discrete": true,
    "n_bins": 256
  },
  "fitted": {
    "class_name": "PSpline",
    "boundary": [18.0, 90.0],
    "knots": [25.0, 35.0, 45.0, 55.0],
    "raw_basis_count": 8,
    "coefficient_width": 7,
    "knot_strategy_actual": "quantile"
  }
}
```

Derive `kind` from concrete class when the original factory kind is not
available:

```text
PSpline -> ps
BSplineSmooth -> bs
NaturalSpline -> ns
CubicRegressionSpline -> cr
CardinalCRSpline -> cr_cardinal
```

Do not store both `constraint_*` and `monotone_*` aliases. Store canonical
constraint fields. Current constraint kinds are:

```text
increasing
decreasing
convex
concave
```

Current constraint modes are:

```text
fit
postfit
```

### Polynomial

```json
{
  "feature_kind": "polynomial",
  "superglm_class": "Polynomial",
  "source_term_name": "Age",
  "published_term_name": "Age",
  "declared": {
    "degree": 3
  },
  "effective": {},
  "fitted": {
    "lower_bound": 0.0,
    "upper_bound": 100.0
  }
}
```

### Numeric

`Numeric` has no constructor-level unit setting. Do not claim it declared a
unit.

```json
{
  "feature_kind": "numeric",
  "superglm_class": "Numeric",
  "source_term_name": "LogDensity",
  "published_term_name": "LogDensity",
  "declared": {},
  "effective": {
    "encoding": "identity"
  },
  "fitted": {}
}
```

## Staging And Publishing

Extend staging so a receipt path/hash can be supplied alongside the workbook:

```python
stage_rating_export(
    engine,
    workbook_path=...,
    export_id=...,
    ...,
    publication_receipt_path=...,
    publication_receipt_sha256=...,
    metadata_mode="REQUIRE_SUPERGLM_RECEIPT",
)
```

Supported metadata modes:

```text
REQUIRE_SUPERGLM_RECEIPT
  receipt path and hash are required;
  every operational staged main-effect term must have metadata;
  unmatched operational metadata is rejected.

ALLOW_WORKBOOK_ONLY
  receipt may be absent;
  offset_handling='UNKNOWN' is permitted;
  term_metadata_json may be null or minimal.
```

Staging:

- validates the receipt hash;
- reads the receipt;
- writes full `publication_receipt_json`, package metadata, and offset fields
  to `STG_RATING_EXPORT`;
- writes term metadata to `STG_TERM_METADATA`;
- validates term metadata names against staged workbook terms;
- marks the declared offset factor as `OFFSET_FACTOR`.

Publishing:

- includes `publication_receipt_sha256` in idempotency conflict checks;
- writes full `publication_receipt_json`, package metadata, and offset fields
  to `PRICING_RATE_PACKAGE`;
- writes term metadata to `PRICING_TERM`;
- rejects incompatible offset states before final package write.

The offline SQLite publisher is a separate implementation path and must be
updated explicitly. It does not use the SQL Server `package_writer`. It must
support:

- receipt hash conflict checking;
- full receipt/package metadata and offset fields;
- `STG_TERM_METADATA` loading;
- `PRICING_TERM.term_metadata_json`;
- `OFFSET_FACTOR` validation;
- manual row inserts that mirror the SQL Server publish semantics.

## Idempotency

The existing `(model_id, source_export_id)` idempotency check must include the
receipt hash.

If the same `export_id` is retried with different package/term/offset metadata,
publishing must report a conflict instead of silently returning the old package.

Add `publication_receipt_sha256` to the existing export conflict comparison.

## Manual Revisions

Manual child packages inherit immutable model-origin metadata:

- copy `publication_receipt_json`;
- copy `publication_receipt_sha256`;
- copy package metadata and offset fields;
- copy `PRICING_TERM.term_metadata_json`;
- keep the original receipt metadata as model-origin metadata;
- rate cells remain the current operational values.

Manual revision information is recorded separately:

```text
PRICING_RATE_PACKAGE.revision_metadata_json
```

Example:

```json
{
  "revision_kind": "MANUAL",
  "parent_rate_package_id": 123,
  "reason": "underwriter adjustment"
}
```

Do not mutate `publication_receipt_json` for manual child packages. If the
receipt JSON changes, the copied `publication_receipt_sha256` no longer matches.

Manual edits to `OFFSET_FACTOR` cells must be rejected. A term marked as an
offset factor represents a fixed-coefficient transformed offset, not a manual
actuarial adjustment factor.

The manual-revision implementation uses explicit package and term column lists.
Those lists must be updated to copy the new package metadata fields and
`PRICING_TERM.term_metadata_json`.

## Workbook-Only Compatibility

If only a workbook is available and no fitted SuperGLM object is available:

```text
publish still works
term_metadata_json may be NULL or minimal
offset_handling may be UNKNOWN only if explicitly allowed
```

The explicit switch is:

```python
metadata_mode: Literal[
    "REQUIRE_SUPERGLM_RECEIPT",
    "ALLOW_WORKBOOK_ONLY",
] = "REQUIRE_SUPERGLM_RECEIPT"
```

New custom DAGs should not silently publish `UNKNOWN`. They should use the
default `REQUIRE_SUPERGLM_RECEIPT` mode and provide the receipt extracted from
the fitted SuperGLM model.

`CompletedModelBuild` and `ModelExportResult` should add receipt fields as
optional `None` defaults so legacy payloads still deserialize. Publication only
allows those fields to be absent when `metadata_mode = "ALLOW_WORKBOOK_ONLY"`.

## Error Handling

Raise clear errors for:

- receipt hash mismatch;
- invalid receipt digest format;
- missing receipt in `REQUIRE_SUPERGLM_RECEIPT` mode;
- non-canonical or non-finite JSON values;
- model fitted with offset but no offset contract;
- `EXPORTED_FACTOR` contract but declared factor missing from staged terms;
- `ALREADY_APPLIED_SQL_EXPOSURE` contract but an offset factor is staged;
- model not fitted with offset but offset handling is not `NONE`;
- staged metadata term names that do not match staged workbook terms;
- canonicalized term-name collisions;
- same `export_id` with changed receipt hash;
- manual revision attempting to edit an `OFFSET_FACTOR` cell.

Unknown future SuperGLM feature classes should not crash metadata extraction if
they are not required for publication. They should produce a minimal receipt:

```json
{
  "feature_kind": "unknown",
  "superglm_class": "SomeFutureFeature"
}
```

## Testing

Add unit tests for metadata extraction:

- receipt contains `schema_name`, `schema_version`, `metadata_origin`,
  `superglm_version`, and `extractor_version`;
- canonical receipt writer writes exactly `canonical_receipt_bytes(...)`;
- receipt hash validates only against canonical bytes;
- receipt validation rejects a semantically equivalent but non-canonical JSON
  file whose raw bytes differ from `canonical_receipt_bytes(parsed_receipt)`;
- `Categorical` captures declared base, grouping maps, levels, and fitted base
  level;
- `OrderedCategorical` captures order/value map and nested spline metadata;
- `OrderedCategorical(basis="step")` emits no nested spline metadata;
- `Spline(kind="ps")` and `PSpline(...)` produce comparable normalized metadata;
- spline metadata captures fitted knots, fitted boundary, degree, penalty,
  `m`, `knot_alpha`, explicit knots/boundary, lambda policy, constraints,
  discrete/n_bins, extrapolation, raw basis count, and coefficient width;
- model-level spline defaults are resolved into `effective`;
- `Polynomial` captures degree and fitted bounds;
- `Numeric` captures `encoding = identity` and no fake unit declaration;
- interaction metadata is omitted from `term_metadata` unless interaction
  staging is implemented;
- numpy/pandas normalization rejects non-finite values.

Add staging/publishing tests:

- completed-build/XCom round-trip carries receipt path and hash;
- `CompletedModelBuild` and `ModelExportResult` accept absent receipt fields for
  legacy payloads;
- `REQUIRE_SUPERGLM_RECEIPT` rejects absent receipt fields;
- `ALLOW_WORKBOOK_ONLY` permits absent receipt fields and writes `UNKNOWN`;
- full `publication_receipt_json` survives staging and publishing;
- package metadata survives staging and publishing;
- `EXPORTED_FACTOR` drives the actual workbook block name;
- `EXPORTED_FACTOR` marks `PRICING_TERM.term_type = OFFSET_FACTOR`;
- fitted offset without offset contract fails;
- `ALREADY_APPLIED_SQL_EXPOSURE` suppresses the workbook offset block;
- external offset handling plus staged offset factor fails;
- same `export_id` plus changed receipt hash fails;
- term-name normalization and collision detection;
- staging replacement deletes term metadata in FK-safe order;
- workbook-only legacy path can publish with `UNKNOWN` only when explicitly
  allowed.
- publication receipt sidecar is logged as an MLflow artifact when MLflow is
  enabled.

Add manual revision tests:

- manual revision copies publication receipt JSON/hash, package metadata, offset
  fields, and term metadata;
- manual revision writes separate `revision_metadata_json`;
- manual revision rejects edits to `OFFSET_FACTOR` cells.

Add offline SQLite smoke coverage:

- transformed offset `log(TermMonths / 12)` publishes cells `12 -> 1` and
  `36 -> 3`;
- offline publisher performs receipt hash conflict checking;
- `PRICING_RATE_PACKAGE` records full receipt JSON, offset fields, and receipt
  hash;
- `PRICING_TERM` stores `OFFSET_FACTOR` and `term_metadata_json`;
- categorical, ordered categorical, and spline examples persist metadata.
- SQL scoring for `EXPORTED_FACTOR` uses the exported offset term with
  `@exposure = 1.0`;
- SQL scoring for `ALREADY_APPLIED_SQL_EXPOSURE` has no offset term and passes
  `EXP(offset)` through `@exposure`;
- both scoring paths produce equivalent predictions for the same transformed
  offset.

## DDL And Reseed

This design requires DDL changes.

For the current development remote SQL Server schemas, a destructive reset and
reseed is acceptable because the tables are empty/currently disposable.

For future historical environments, migrate existing packages with:

```text
offset_handling = UNKNOWN
publication_receipt_sha256 = NULL
package_metadata_json = NULL
term_metadata_json = NULL
```

## Open Follow-Ups

This design does not require `PRICING_COMPILED_RATE_CELL.term_metadata_json`.
If PowerBI/reporting strongly prefers a no-join compiled table later, add that
as a separate denormalization change.

This design does not make SQL Server evaluate spline formulas. If a later
project wants raw-feature SQL scoring using SuperGLM spline bases, that is a
separate runtime scoring project.

Interaction receipts become first-class once interaction workbook staging is
implemented.
