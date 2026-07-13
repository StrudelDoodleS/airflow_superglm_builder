# Simple Notebook Pricing Model Design

**Date:** 2026-07-13
**Status:** Approved in conversation; awaiting written-spec review

## Purpose

Make a pricing-model notebook read like an analyst workflow rather than a call
into deployment plumbing. The analyst should define the model, prepare the
model frame, choose validation, fit, review, and publish. Generated identifiers,
row identity, offset export metadata, package versions, and SQL audit writes
remain library responsibilities.

This design also removes the false requirement that an undeployed candidate
must have a business effective date.

## Notebook Experience

The first executable cells are:

1. imports;
2. one **Analyst settings** cell containing the important stable model and data
   decisions;
3. one **Optional actions** cell containing editor and deployment flags, all
   disabled by default.

The SQL query, feature transforms, and `make_model()` remain separate visible
cells. Feature transforms remain ordinary Python; this design does not add a
transform DSL.

The settings cell uses one small Python object:

```python
MODEL = PricingModelSpec(
    name="MTPL_FREQ",
    label="Motor frequency",
    target="ClaimNb",
    model_type="superglm_poisson",
    deployment_slot="MTPL_FREQ_UAT",
    features=(
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "LogDensity",
        "Area",
        "VehPower",
        "VehBrand",
        "VehGas",
        "Region",
    ),
    dataset_name="freMTPL2freq_model_frame",
    source_system="freMTPL_raw_sql",
    pk_columns=("IDpol",),
    exposure_column="Exposure",
    validation=ValidationSplitConfig.column_kfold(
        column="cv_fold",
        materialize=True,
    ),
)

DATA_AS_OF = date(2026, 6, 30)
```

The MTPL demonstration may continue to use ordinary generated K-folds until
its source table contains a fold column. The example above documents the
supported source-column pattern.

The build cell becomes:

```python
candidate = build_candidate(
    pricing,
    model=model,
    frame=frame,
    model_factory=make_model,
    data_as_of=DATA_AS_OF,
)
```

There is no `effective_from` argument.

## `PricingModelSpec`

`PricingModelSpec` is the only new analyst-facing configuration object. It is
Python, not TOML, and contains stable decisions already present in the verbose
notebook calls:

- SQL model identity: name, label, target, model type, and deployment slot;
- model inputs: ordered feature columns and optional exposure/sample-weight
  columns;
- dataset lineage: dataset name, source system, and primary-key columns;
- validation strategy;
- optional fit mode and scoring overrides, with the current standard defaults.

The object may optionally declare `data_as_of_column`. It does not contain the
run-specific explicit data-as-of value.

`register_model()` accepts the spec and returns the existing registered model
handle. Existing lower-level call shapes remain supported for non-notebook code
during migration.

## Candidate Input Derivation

For the simple notebook path, `build_candidate()` derives:

- `X` from `MODEL.features`;
- `y` from `MODEL.target`;
- canonical row identity from `MODEL.pk_columns`;
- dataset manifest metadata from the model spec and resolved data-as-of value;
- validation indices from `MODEL.validation`;
- default scoring and fit mode from the spec;
- sample weights when a sample-weight column is configured.

When `exposure_column` is configured, the standard frequency-model convention
is explicit and automatic:

- require finite, strictly positive exposure;
- fit with `log(exposure)` as the offset;
- use exposure as the rating-export averaging weight;
- record exposure as the manifest weight column;
- construct the `EXPORTED_FACTOR` offset publication contract;
- pass the source-aware offset options required by the SuperGLM Excel exporter.

Models that do not use this convention leave `exposure_column` unset. Advanced
nonstandard offsets may continue to use the lower-level API; this design does
not introduce a general offset-transform abstraction.

## Data-As-Of Resolution

Data as of is the source-data cutoff or snapshot vintage, not the notebook run
date. It does not itself filter the source query.

Resolution follows these rules:

1. An explicit `data_as_of=` value is accepted.
2. If `MODEL.data_as_of_column` is configured, the column must be present,
   non-null, and resolve to exactly one normalized date across the frame.
3. If both are supplied, they must agree.
4. If neither is available, candidate building fails with a direct message.

The helper never derives data as of from the maximum event, transaction, policy,
or claim date. Such a maximum is not proof of source-system completeness.

## Effective-Date Lifecycle

An undeployed candidate has no deployment date. Candidate publication therefore
does not require or invent one.

- `pricing.PRICING_RATE_PACKAGE.effective_from_date` becomes nullable and means
  only an optional proposed business-effective date for legacy or external
  publishers.
- Notebook candidate publication writes it as `NULL`.
- Editor child packages preserve the parent's optional value without inventing
  one.
- Existing populated dates are retained.
- `pricing.PRICING_MODEL_DEPLOYMENT.effective_from_ts` remains the authoritative
  activation timestamp. The current deployment operation populates it at the
  moment the champion pointer changes.
- Future scheduled deployment is out of scope and can later extend the
  deployment operation with an explicit timestamp.

The staging schema and completed-build payload become nullable-compatible. No
sentinel dates and no publication-date defaults are permitted.

## SQL Migration

A new forward migration alters candidate/package and staging effective-date
columns to allow `NULL`. Existing base migrations remain historically stable;
new installations reach the same final schema by applying the forward
migration.

Views continue to expose the optional package date and, where deployment state
is relevant, the authoritative deployment timestamp. Package-specific scoring
continues to select by package or deployment pointer and does not depend on the
optional package date.

## Validation and Errors

Candidate building fails before fitting when:

- required frame, target, feature, PK, exposure, weight, split, or data-as-of
  columns are missing;
- PK values are null or duplicated;
- configured source-column splits contain invalid assignments;
- exposure is null, non-finite, or non-positive;
- explicit and column-derived data-as-of values disagree.

Analysts do not provide generated IDs, model versions, fold artifacts, hashes,
or offset publication objects.

## Verification

Tests cover:

- `PricingModelSpec` validation and backward-compatible registration;
- automatic `X`, `y`, row identity, validation, exposure offset, and lineage
  derivation;
- explicit and column-derived data-as-of resolution, including disagreement and
  ambiguous-column failures;
- nullable effective-date migration and publication;
- editor revisions of packages with no proposed effective date;
- notebook cell order and the absence of the verbose plumbing arguments;
- the complete existing test suite and lint checks.

The final verification executes the actual notebook cells against an isolated
SQL Server database using the pinned SuperGLM version, confirms a successful
model run and populated rating tables, and confirms the candidate package date
is `NULL` while no deployment row exists.

## Out of Scope

- scheduled or future-dated deployment;
- a feature-transform DSL;
- automatic inference from arbitrary event-date columns;
- an MLflow persistence redesign;
- replacement of the existing advanced build API;
- renewed Airflow automation work.
