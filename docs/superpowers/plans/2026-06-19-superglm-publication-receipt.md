# SuperGLM Publication Receipt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist a versioned SuperGLM publication receipt with each published rate package, including offset contract metadata, SuperGLM term metadata, canonical receipt hashing, and SQL/offline/manual-revision parity.

**Architecture:** Write a canonical JSON sidecar during model training/export while the fitted SuperGLM object is still in memory. Carry only the receipt path and hash through Airflow/XCom, stage the receipt alongside the workbook, then publish package-level metadata and per-term metadata into SQL. Keep multipliers as the scoring truth; metadata explains how those multipliers were produced.

**Tech Stack:** Python, Pydantic v2, SuperGLM 0.10.x, pandas, SQLAlchemy, SQL Server T-SQL migrations, offline SQLite DDL, pytest, ruff.

---

## File Map

- Create `pricing_pipeline/publishing/naming.py`
  - Owns shared term-name canonicalization currently duplicated in staging.
- Create `pricing_pipeline/publishing/superglm_publication_receipt.py`
  - Owns `OffsetExportContract`, `SuperGLMPublicationReceipt`, JSON normalization, canonical bytes, SHA-256 validation, and receipt file IO.
- Create `pricing_pipeline/publishing/superglm_metadata.py`
  - Extracts SuperGLM package and term metadata from a fitted model.
- Modify `pricing_pipeline/publishing/rating_export.py`
  - Forwards offset export controls to SuperGLM and optionally logs receipt artifacts through MLflow.
- Modify `pricing_pipeline/orchestration/publish_completed_build.py`
  - Adds receipt path/hash to `CompletedModelBuild`, publish result, and model-run lineage.
- Modify `pricing_pipeline/models/spec.py`
  - Adds receipt path/hash to `ModelExportResult`.
- Modify `pricing_pipeline/publishing/staging.py`
  - Stages receipt metadata and `STG_TERM_METADATA`.
- Modify `pricing_pipeline/publishing/package_writer.py`
  - Publishes receipt metadata, offset fields, term metadata, and hash conflict checks.
- Modify `pricing_pipeline/publishing/manual_revision.py`
  - Copies immutable metadata and rejects offset-factor edits.
- Modify `scripts/run_mtpl_frequency_offline_sqlite.py`
  - Mirrors metadata publish behavior for offline SQLite.
- Modify `db/migrations/*.sql`
  - Adds SQL Server DDL for receipt/package/term metadata.
- Modify `db/offline_sqlite/*.sql`
  - Adds equivalent offline SQLite structures.
- Add tests across:
  - `tests/test_superglm_publication_receipt.py`
  - `tests/test_superglm_metadata.py`
  - `tests/test_rating_export.py`
  - `tests/test_publish_completed_build.py`
  - `tests/test_package_writer.py`
  - `tests/test_manual_revision.py`
  - `tests/test_mtpl_offline_sqlite_runner.py`
  - `tests/test_migrations.py`

---

### Task 1: Add SQL And SQLite DDL

**Files:**
- Create: `db/migrations/V022__superglm_publication_receipt_metadata.sql`
- Modify: `db/offline_sqlite/pricing.sql`
- Modify: `db/offline_sqlite/pricing_stg.sql`
- Test: `tests/test_migrations.py`
- Test: `tests/test_mtpl_offline_sqlite_runner.py`

- [ ] **Step 1: Write failing migration assertions**

Add assertions in `tests/test_migrations.py`:

```python
def test_superglm_publication_receipt_migration_adds_metadata_columns():
    migration = Path("db/migrations/V022__superglm_publication_receipt_metadata.sql").read_text(
        encoding="utf-8",
    )

    assert "publication_receipt_json" in migration
    assert "publication_receipt_sha256" in migration
    assert "package_metadata_json" in migration
    assert "revision_metadata_json" in migration
    assert "offset_handling" in migration
    assert "STG_TERM_METADATA" in migration
    assert "term_metadata_json" in migration
    assert "ISJSON(publication_receipt_json)" in migration
    assert "ALREADY_APPLIED_SQL_EXPOSURE" in migration
```

Add offline DDL assertions in `tests/test_mtpl_offline_sqlite_runner.py`:

```python
def test_offline_sqlite_ddl_contains_publication_receipt_metadata():
    pricing_sql = Path("db/offline_sqlite/pricing.sql").read_text(encoding="utf-8")
    staging_sql = Path("db/offline_sqlite/pricing_stg.sql").read_text(encoding="utf-8")

    assert "publication_receipt_json" in pricing_sql
    assert "publication_receipt_sha256" in pricing_sql
    assert "revision_metadata_json" in pricing_sql
    assert "term_metadata_json" in pricing_sql
    assert "publication_receipt_json" in staging_sql
    assert "STG_TERM_METADATA" in staging_sql
```

- [ ] **Step 2: Run tests red**

Run:

```bash
rtk uv run pytest -q tests/test_migrations.py::test_superglm_publication_receipt_migration_adds_metadata_columns tests/test_mtpl_offline_sqlite_runner.py::test_offline_sqlite_ddl_contains_publication_receipt_metadata
```

Expected: fail because the migration and SQLite DDL do not yet include the new fields.

- [ ] **Step 3: Add SQL Server migration**

Create `db/migrations/V022__superglm_publication_receipt_metadata.sql`:

```sql
IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'publication_receipt_json') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD publication_receipt_json NVARCHAR(MAX) NULL;

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'publication_receipt_sha256') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD publication_receipt_sha256 CHAR(64) NULL;

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'package_metadata_json') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD package_metadata_json NVARCHAR(MAX) NULL;

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'revision_metadata_json') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD revision_metadata_json NVARCHAR(MAX) NULL;

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'offset_handling') IS NULL
BEGIN
    ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD offset_handling NVARCHAR(64) NULL;
    EXEC('UPDATE pricing.PRICING_RATE_PACKAGE SET offset_handling = ''UNKNOWN'' WHERE offset_handling IS NULL;');
    ALTER TABLE pricing.PRICING_RATE_PACKAGE ALTER COLUMN offset_handling NVARCHAR(64) NOT NULL;
END;

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'offset_factor_name') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD offset_factor_name NVARCHAR(256) NULL;

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'offset_source_name') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD offset_source_name NVARCHAR(256) NULL;

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'offset_label') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD offset_label NVARCHAR(1024) NULL;

IF COL_LENGTH('pricing.PRICING_RATE_PACKAGE', 'metadata_origin') IS NULL
    ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD metadata_origin NVARCHAR(128) NULL;

IF COL_LENGTH('pricing.PRICING_TERM', 'term_metadata_json') IS NULL
    ALTER TABLE pricing.PRICING_TERM ADD term_metadata_json NVARCHAR(MAX) NULL;

IF COL_LENGTH('pricing.MODEL_RUN', 'publication_receipt_path') IS NULL
    ALTER TABLE pricing.MODEL_RUN ADD publication_receipt_path NVARCHAR(1024) NULL;

IF COL_LENGTH('pricing.MODEL_RUN', 'publication_receipt_sha256') IS NULL
    ALTER TABLE pricing.MODEL_RUN ADD publication_receipt_sha256 CHAR(64) NULL;

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'publication_receipt_json') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT ADD publication_receipt_json NVARCHAR(MAX) NULL;

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'publication_receipt_sha256') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT ADD publication_receipt_sha256 CHAR(64) NULL;

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'package_metadata_json') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT ADD package_metadata_json NVARCHAR(MAX) NULL;

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'offset_handling') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT ADD offset_handling NVARCHAR(64) NULL;

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'offset_factor_name') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT ADD offset_factor_name NVARCHAR(256) NULL;

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'offset_source_name') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT ADD offset_source_name NVARCHAR(256) NULL;

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'offset_label') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT ADD offset_label NVARCHAR(1024) NULL;

IF COL_LENGTH('pricing_stg.STG_RATING_EXPORT', 'metadata_origin') IS NULL
    ALTER TABLE pricing_stg.STG_RATING_EXPORT ADD metadata_origin NVARCHAR(128) NULL;

IF OBJECT_ID('pricing_stg.STG_TERM_METADATA', 'U') IS NULL
CREATE TABLE pricing_stg.STG_TERM_METADATA (
    export_id NVARCHAR(128) NOT NULL,
    term_name NVARCHAR(256) NOT NULL,
    term_metadata_json NVARCHAR(MAX) NOT NULL,
    CONSTRAINT PK_STG_TERM_METADATA PRIMARY KEY (export_id, term_name),
    CONSTRAINT FK_STG_TERM_METADATA_EXPORT FOREIGN KEY (export_id)
        REFERENCES pricing_stg.STG_RATING_EXPORT(export_id),
    CONSTRAINT CK_STG_TERM_METADATA_JSON CHECK (ISJSON(term_metadata_json) = 1)
);

IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name = 'CK_PRICING_RATE_PACKAGE_OFFSET_HANDLING')
ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD CONSTRAINT CK_PRICING_RATE_PACKAGE_OFFSET_HANDLING
CHECK (offset_handling IN ('NONE', 'EXPORTED_FACTOR', 'ALREADY_APPLIED_SQL_EXPOSURE', 'UNKNOWN'));

IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name = 'CK_PRICING_RATE_PACKAGE_RECEIPT_JSON')
ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD CONSTRAINT CK_PRICING_RATE_PACKAGE_RECEIPT_JSON
CHECK (publication_receipt_json IS NULL OR ISJSON(publication_receipt_json) = 1);

IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name = 'CK_PRICING_RATE_PACKAGE_METADATA_JSON')
ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD CONSTRAINT CK_PRICING_RATE_PACKAGE_METADATA_JSON
CHECK (package_metadata_json IS NULL OR ISJSON(package_metadata_json) = 1);

IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name = 'CK_PRICING_RATE_PACKAGE_REVISION_METADATA_JSON')
ALTER TABLE pricing.PRICING_RATE_PACKAGE ADD CONSTRAINT CK_PRICING_RATE_PACKAGE_REVISION_METADATA_JSON
CHECK (revision_metadata_json IS NULL OR ISJSON(revision_metadata_json) = 1);

IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name = 'CK_PRICING_TERM_METADATA_JSON')
ALTER TABLE pricing.PRICING_TERM ADD CONSTRAINT CK_PRICING_TERM_METADATA_JSON
CHECK (term_metadata_json IS NULL OR ISJSON(term_metadata_json) = 1);
```

- [ ] **Step 4: Update offline SQLite DDL**

In `db/offline_sqlite/pricing.sql`, add the new columns to:

```sql
CREATE TABLE IF NOT EXISTS pricing.PRICING_RATE_PACKAGE (
    ...
    publication_receipt_json TEXT,
    publication_receipt_sha256 TEXT,
    package_metadata_json TEXT,
    revision_metadata_json TEXT,
    offset_handling TEXT NOT NULL DEFAULT 'UNKNOWN',
    offset_factor_name TEXT,
    offset_source_name TEXT,
    offset_label TEXT,
    metadata_origin TEXT,
    ...
);
```

Add `term_metadata_json TEXT` to `pricing.PRICING_TERM`.

Add receipt fields to `pricing.MODEL_RUN`:

```sql
publication_receipt_path TEXT,
publication_receipt_sha256 TEXT,
```

In `db/offline_sqlite/pricing_stg.sql`, add the same staging columns to `STG_RATING_EXPORT` and create:

```sql
CREATE TABLE IF NOT EXISTS pricing_stg.STG_TERM_METADATA (
    export_id TEXT NOT NULL,
    term_name TEXT NOT NULL,
    term_metadata_json TEXT NOT NULL,
    PRIMARY KEY (export_id, term_name),
    FOREIGN KEY (export_id) REFERENCES STG_RATING_EXPORT(export_id)
);
```

- [ ] **Step 5: Run DDL tests green**

Run:

```bash
rtk uv run pytest -q tests/test_migrations.py::test_superglm_publication_receipt_migration_adds_metadata_columns tests/test_mtpl_offline_sqlite_runner.py::test_offline_sqlite_ddl_contains_publication_receipt_metadata
```

Expected: pass.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add db/migrations/V022__superglm_publication_receipt_metadata.sql db/offline_sqlite/pricing.sql db/offline_sqlite/pricing_stg.sql tests/test_migrations.py tests/test_mtpl_offline_sqlite_runner.py
rtk git commit -m "Add publication receipt DDL"
```

---

### Task 2: Add Naming And Receipt Contract Modules

**Files:**
- Create: `pricing_pipeline/publishing/naming.py`
- Create: `pricing_pipeline/publishing/superglm_publication_receipt.py`
- Test: `tests/test_superglm_publication_receipt.py`

- [ ] **Step 1: Write failing receipt tests**

Create `tests/test_superglm_publication_receipt.py`:

```python
from __future__ import annotations

import json

import pytest

from pricing_pipeline.publishing.naming import clean_identifier
from pricing_pipeline.publishing.superglm_publication_receipt import (
    OffsetExportContract,
    SuperGLMPublicationReceipt,
    canonical_receipt_bytes,
    load_publication_receipt,
    publication_receipt_sha256,
    write_publication_receipt,
)


def _receipt() -> SuperGLMPublicationReceipt:
    return SuperGLMPublicationReceipt(
        schema_name="superglm_publication_receipt",
        schema_version=1,
        metadata_origin="SUPERGLM_FITTED_MODEL",
        superglm_version="0.10.0",
        extractor_version="1",
        package_metadata={"model": {"family": "poisson", "link": "log", "fit_used_offset": True}},
        term_metadata={
            "TermMonths": {
                "feature_kind": "offset",
                "source_term_name": "Term Months",
                "published_term_name": "TermMonths",
            }
        },
        offset_contract=OffsetExportContract(
            handling="EXPORTED_FACTOR",
            source_factor_name="Term Months",
            published_factor_name="TermMonths",
            source_name="TermMonths",
            label="log(TermMonths / 12)",
        ),
    )


def test_clean_identifier_matches_staging_normalization():
    assert clean_identifier("Term Months") == "Term_Months"
    assert clean_identifier("  A/B + C  ") == "A_B_C"
    assert clean_identifier("") == "unknown"


def test_offset_contract_cross_field_validation():
    OffsetExportContract(handling="NONE")
    OffsetExportContract(
        handling="EXPORTED_FACTOR",
        source_factor_name="Term Months",
        published_factor_name="Term_Months",
        source_name="TermMonths",
        label="log(TermMonths / 12)",
    )
    OffsetExportContract(
        handling="ALREADY_APPLIED_SQL_EXPOSURE",
        source_name="Exposure",
        label="log(Exposure)",
    )

    with pytest.raises(ValueError, match="must be null"):
        OffsetExportContract(handling="NONE", source_name="Exposure")
    with pytest.raises(ValueError, match="required"):
        OffsetExportContract(handling="EXPORTED_FACTOR", source_name="TermMonths")
    with pytest.raises(ValueError, match="must be null"):
        OffsetExportContract(
            handling="ALREADY_APPLIED_SQL_EXPOSURE",
            source_factor_name="Exposure",
            source_name="Exposure",
            label="log(Exposure)",
        )


def test_canonical_receipt_hash_uses_exact_canonical_bytes(tmp_path):
    receipt = _receipt()
    expected = json.dumps(
        receipt.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    assert canonical_receipt_bytes(receipt) == expected
    path = tmp_path / "superglm_publication_receipt.json"
    digest = write_publication_receipt(receipt, path)

    assert path.read_bytes() == expected
    assert digest == publication_receipt_sha256(receipt)
    assert load_publication_receipt(path, expected_sha256=digest) == receipt


def test_receipt_loader_rejects_noncanonical_equivalent_json(tmp_path):
    receipt = _receipt()
    canonical_digest = publication_receipt_sha256(receipt)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt.model_dump(mode="json"), indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="not canonical"):
        load_publication_receipt(path, expected_sha256=canonical_digest)
```

- [ ] **Step 2: Run tests red**

Run:

```bash
rtk uv run pytest -q tests/test_superglm_publication_receipt.py
```

Expected: fail because the modules do not exist.

- [ ] **Step 3: Add shared naming module**

Create `pricing_pipeline/publishing/naming.py`:

```python
from __future__ import annotations

import re


def clean_identifier(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"
```

- [ ] **Step 4: Add receipt contract module**

Create `pricing_pipeline/publishing/superglm_publication_receipt.py` with:

```python
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class OffsetExportContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    handling: Literal["NONE", "EXPORTED_FACTOR", "ALREADY_APPLIED_SQL_EXPOSURE"]
    source_factor_name: str | None = None
    published_factor_name: str | None = None
    source_name: str | None = None
    label: str | None = None

    @field_validator("source_factor_name", "published_factor_name", "source_name", "label", mode="before")
    @classmethod
    def _optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @model_validator(mode="after")
    def _validate_fields(self) -> "OffsetExportContract":
        names = {
            "source_factor_name": self.source_factor_name,
            "published_factor_name": self.published_factor_name,
            "source_name": self.source_name,
            "label": self.label,
        }
        if self.handling == "NONE":
            present = [key for key, value in names.items() if value is not None]
            if present:
                raise ValueError("NONE offset handling requires all offset fields to be null")
        elif self.handling == "EXPORTED_FACTOR":
            missing = [key for key, value in names.items() if value is None]
            if missing:
                raise ValueError("EXPORTED_FACTOR requires " + ", ".join(missing))
        elif self.handling == "ALREADY_APPLIED_SQL_EXPOSURE":
            if self.source_factor_name is not None or self.published_factor_name is not None:
                raise ValueError(
                    "ALREADY_APPLIED_SQL_EXPOSURE factor names must be null"
                )
            missing = [key for key in ("source_name", "label") if names[key] is None]
            if missing:
                raise ValueError(
                    "ALREADY_APPLIED_SQL_EXPOSURE requires " + ", ".join(missing)
                )
        return self


class SuperGLMPublicationReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_name: Literal["superglm_publication_receipt"]
    schema_version: Literal[1]
    metadata_origin: Literal["SUPERGLM_FITTED_MODEL"]
    superglm_version: str
    extractor_version: str
    package_metadata: dict[str, Any]
    term_metadata: dict[str, dict[str, Any]]
    offset_contract: OffsetExportContract

    @field_validator("superglm_version", "extractor_version", mode="before")
    @classmethod
    def _required_text(cls, value: Any) -> str:
        if value is None or not str(value).strip():
            raise ValueError("is required")
        return str(value).strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("publication receipt contains non-finite float")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def canonical_receipt_bytes(receipt: SuperGLMPublicationReceipt) -> bytes:
    payload = _json_safe(receipt.model_dump(mode="json"))
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def publication_receipt_sha256(receipt: SuperGLMPublicationReceipt) -> str:
    return hashlib.sha256(canonical_receipt_bytes(receipt)).hexdigest()


def _validate_digest(value: str) -> str:
    digest = str(value).strip()
    if len(digest) != 64 or digest.lower() != digest or not all(
        char in "0123456789abcdef" for char in digest
    ):
        raise ValueError("publication receipt digest must be 64 lowercase hex characters")
    return digest


def write_publication_receipt(receipt: SuperGLMPublicationReceipt, path: str | Path) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canonical = canonical_receipt_bytes(receipt)
    out.write_bytes(canonical)
    return hashlib.sha256(canonical).hexdigest()


def load_publication_receipt(
    path: str | Path,
    *,
    expected_sha256: str,
) -> SuperGLMPublicationReceipt:
    expected = _validate_digest(expected_sha256)
    raw = Path(path).read_bytes()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("publication receipt is not valid JSON") from exc

    receipt = SuperGLMPublicationReceipt.model_validate(parsed)
    canonical = canonical_receipt_bytes(receipt)
    if raw != canonical:
        raise ValueError("publication receipt file is not canonical")
    actual = hashlib.sha256(canonical).hexdigest()
    if actual != expected:
        raise ValueError("publication receipt SHA-256 mismatch")
    return receipt
```

- [ ] **Step 5: Run tests green**

Run:

```bash
rtk uv run pytest -q tests/test_superglm_publication_receipt.py
```

Expected: pass.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add pricing_pipeline/publishing/naming.py pricing_pipeline/publishing/superglm_publication_receipt.py tests/test_superglm_publication_receipt.py
rtk git commit -m "Add SuperGLM publication receipt contract"
```

---

### Task 3: Extract SuperGLM Term Metadata

**Files:**
- Create: `pricing_pipeline/publishing/superglm_metadata.py`
- Test: `tests/test_superglm_metadata.py`

- [ ] **Step 1: Write failing extractor tests**

Create `tests/test_superglm_metadata.py` with tests that fit tiny SuperGLM models:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from superglm import Categorical, Numeric, OrderedCategorical, Polynomial, Spline, SuperGLM
from superglm.features.spline import PSpline

from pricing_pipeline.publishing.superglm_metadata import build_superglm_publication_receipt
from pricing_pipeline.publishing.superglm_publication_receipt import OffsetExportContract


def _fit_model(features):
    n = 90
    rng = np.random.default_rng(20260619)
    X = pd.DataFrame(
        {
            "cat": np.array(["A", "B", "C"])[np.arange(n) % 3],
            "ord": np.array(["low", "medium", "high"])[np.arange(n) % 3],
            "age": np.linspace(18.0, 90.0, n),
            "poly": np.linspace(0.0, 10.0, n),
            "num": rng.normal(size=n),
        }
    )
    y = rng.poisson(np.exp(-2.0 + 0.01 * X["age"]))
    model = SuperGLM(
        family="poisson",
        features=features,
        selection_penalty=0.0,
        discrete=True,
        n_bins=32,
        retain_fit_state=False,
    )
    fitted = model.fit(X, y, sample_weight=np.ones(n))
    return fitted or model


def test_extracts_categorical_ordered_spline_polynomial_and_numeric_metadata():
    model = _fit_model(
        {
            "cat": Categorical(base="most_exposed"),
            "ord": OrderedCategorical(order=["low", "medium", "high"], basis="spline", n_knots=5),
            "age": Spline(kind="ps", n_knots=4, knot_strategy="quantile", discrete=True, n_bins=16),
            "poly": Polynomial(degree=2),
            "num": Numeric(),
        }
    )
    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    assert receipt.schema_name == "superglm_publication_receipt"
    assert receipt.schema_version == 1
    assert receipt.metadata_origin == "SUPERGLM_FITTED_MODEL"
    assert receipt.package_metadata["model"]["family"] == "poisson"
    assert receipt.package_metadata["model"]["fit_used_offset"] is False

    cat = receipt.term_metadata["cat"]
    assert cat["feature_kind"] == "categorical"
    assert cat["declared"]["base"] == "most_exposed"
    assert sorted(cat["fitted"]["levels"]) == ["A", "B", "C"]
    assert cat["fitted"]["base_level"] in {"A", "B", "C"}

    ordered = receipt.term_metadata["ord"]
    assert ordered["feature_kind"] == "ordered_categorical"
    assert ordered["declared"]["ordered_levels"] == ["low", "medium", "high"]
    assert ordered["declared"]["n_knots_requested"] == 5
    assert ordered["effective"]["n_knots_effective"] == 2
    assert ordered["spline"]["fitted"]["class_name"] == "PSpline"

    spline = receipt.term_metadata["age"]
    assert spline["feature_kind"] == "spline"
    assert spline["declared"]["kind"] == "ps"
    assert spline["declared"]["knot_strategy"] == "quantile"
    assert spline["fitted"]["boundary"] == [18.0, 90.0]
    assert spline["fitted"]["raw_basis_count"] > 0

    poly = receipt.term_metadata["poly"]
    assert poly["feature_kind"] == "polynomial"
    assert poly["declared"]["degree"] == 2
    assert poly["fitted"]["lower_bound"] == 0.0
    assert poly["fitted"]["upper_bound"] == 10.0

    numeric = receipt.term_metadata["num"]
    assert numeric["feature_kind"] == "numeric"
    assert numeric["declared"] == {}
    assert numeric["effective"]["encoding"] == "identity"


def test_spline_factory_and_direct_pspline_normalize_to_same_kind():
    model = _fit_model(
        {
            "age": Spline(kind="ps", n_knots=4),
            "poly": PSpline(n_knots=4),
        }
    )
    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    assert receipt.term_metadata["age"]["effective"]["kind"] == "ps"
    assert receipt.term_metadata["poly"]["effective"]["kind"] == "ps"
    assert receipt.term_metadata["age"]["fitted"]["class_name"] == "PSpline"
    assert receipt.term_metadata["poly"]["fitted"]["class_name"] == "PSpline"


def test_ordered_categorical_step_has_no_nested_spline():
    model = _fit_model(
        {
            "ord": OrderedCategorical(
                order=["low", "medium", "high"],
                basis="step",
                base="first",
            )
        }
    )
    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    metadata = receipt.term_metadata["ord"]
    assert metadata["feature_kind"] == "ordered_categorical"
    assert metadata["declared"]["basis"] == "step"
    assert "spline" not in metadata
```

- [ ] **Step 2: Run tests red**

Run:

```bash
rtk uv run pytest -q tests/test_superglm_metadata.py
```

Expected: fail because the extractor does not exist.

- [ ] **Step 3: Implement extractor skeleton**

Create `pricing_pipeline/publishing/superglm_metadata.py` with JSON-safe helpers:

```python
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import superglm
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.polynomial import Polynomial
from superglm.features.spline import (
    BSplineSmooth,
    CardinalCRSpline,
    CubicRegressionSpline,
    NaturalSpline,
    PSpline,
    _SplineBase,
)

from pricing_pipeline.publishing.naming import clean_identifier
from pricing_pipeline.publishing.superglm_publication_receipt import (
    OffsetExportContract,
    SuperGLMPublicationReceipt,
)

EXTRACTOR_VERSION = "1"

_SPLINE_KIND_BY_CLASS = {
    PSpline: "ps",
    BSplineSmooth: "bs",
    NaturalSpline: "ns",
    CubicRegressionSpline: "cr",
    CardinalCRSpline: "cr_cardinal",
}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("non-finite value in SuperGLM metadata")
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, tuple | list | set):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return repr(value)


def _spline_kind(spec: _SplineBase) -> str:
    for klass, kind in _SPLINE_KIND_BY_CLASS.items():
        if isinstance(spec, klass):
            return kind
    return type(spec).__name__
```

- [ ] **Step 4: Implement per-feature metadata functions**

Add functions:

```python
def _categorical_metadata(name: str, spec: Categorical) -> dict[str, Any]:
    return {
        "feature_kind": "categorical",
        "superglm_class": type(spec).__name__,
        "source_term_name": name,
        "published_term_name": clean_identifier(name),
        "declared": {
            "base": spec.base,
            "grouping": _grouping_metadata(getattr(spec, "_grouping", None)),
        },
        "effective": {},
        "fitted": {
            "levels": _json_value(getattr(spec, "_levels", [])),
            "base_level": getattr(spec, "_base_level", None) or None,
            "non_base_levels": _json_value(getattr(spec, "_non_base", [])),
        },
    }


def _spline_metadata(name: str, spec: _SplineBase) -> dict[str, Any]:
    fitted_knots = spec.fitted_knots
    fitted_boundary = spec.fitted_boundary
    r_inv = getattr(spec, "_R_inv", None)
    return {
        "feature_kind": "spline",
        "superglm_class": type(spec).__name__,
        "source_term_name": name,
        "published_term_name": clean_identifier(name),
        "declared": {
            "kind": _spline_kind(spec),
            "n_knots": spec.n_knots,
            "degree": spec.degree,
            "knot_strategy": getattr(spec, "knot_strategy", None),
            "penalty": getattr(spec, "penalty", None),
            "select": getattr(spec, "select", None),
            "discrete": getattr(spec, "discrete", None),
            "n_bins": getattr(spec, "n_bins", None),
            "extrapolation": getattr(spec, "extrapolation", None),
            "constraint_kind": getattr(spec, "constraint_kind", None),
            "constraint_mode": getattr(spec, "constraint_mode", None),
            "m": _json_value(getattr(spec, "_m_orders", None)),
            "knot_alpha": getattr(spec, "knot_alpha", None),
            "explicit_knots": _json_value(getattr(spec, "_explicit_knots", None)),
            "explicit_boundary": _json_value(getattr(spec, "_explicit_boundary", None)),
            "lambda_policy": _json_value(getattr(spec, "_lambda_policy", None)),
        },
        "effective": {
            "kind": _spline_kind(spec),
            "class_name": type(spec).__name__,
            "discrete": getattr(spec, "discrete", None),
            "n_bins": getattr(spec, "n_bins", None),
        },
        "fitted": {
            "class_name": type(spec).__name__,
            "boundary": _json_value(fitted_boundary),
            "knots": _json_value(fitted_knots),
            "raw_basis_count": int(getattr(spec, "_n_basis", 0)),
            "coefficient_width": None if r_inv is None else int(r_inv.shape[1]),
            "knot_strategy_actual": getattr(spec, "_knot_strategy_actual", None),
        },
    }
```

Implement `_ordered_categorical_metadata`, `_polynomial_metadata`, `_numeric_metadata`, and `_grouping_metadata` following the spec examples.

- [ ] **Step 5: Implement receipt builder**

Add:

```python
def build_superglm_publication_receipt(
    model,
    *,
    offset_contract: OffsetExportContract,
    source_to_published_names: Mapping[str, str] | None = None,
) -> SuperGLMPublicationReceipt:
    term_metadata: dict[str, dict[str, Any]] = {}
    for name in getattr(model, "_feature_order", []):
        spec = model._specs[name]
        if isinstance(spec, Categorical):
            metadata = _categorical_metadata(name, spec)
        elif isinstance(spec, OrderedCategorical):
            metadata = _ordered_categorical_metadata(name, spec)
        elif isinstance(spec, _SplineBase):
            metadata = _spline_metadata(name, spec)
        elif isinstance(spec, Polynomial):
            metadata = _polynomial_metadata(name, spec)
        elif isinstance(spec, Numeric):
            metadata = _numeric_metadata(name, spec)
        else:
            metadata = {
                "feature_kind": "unknown",
                "superglm_class": type(spec).__name__,
                "source_term_name": name,
                "published_term_name": clean_identifier(name),
            }
        published = (source_to_published_names or {}).get(name, metadata["published_term_name"])
        metadata["published_term_name"] = published
        if published in term_metadata:
            raise ValueError(f"canonical term name collision: {published!r}")
        term_metadata[published] = _json_value(metadata)

    return SuperGLMPublicationReceipt(
        schema_name="superglm_publication_receipt",
        schema_version=1,
        metadata_origin="SUPERGLM_FITTED_MODEL",
        superglm_version=getattr(superglm, "__version__", "unknown"),
        extractor_version=EXTRACTOR_VERSION,
        package_metadata={
            "model": {
                "family": str(getattr(model, "family", "")),
                "link": type(getattr(model, "_link", None)).__name__,
                "fit_used_offset": bool(getattr(model, "_fit_used_offset", False)),
            }
        },
        term_metadata=term_metadata,
        offset_contract=offset_contract,
    )
```

- [ ] **Step 6: Run tests green**

Run:

```bash
rtk uv run pytest -q tests/test_superglm_metadata.py tests/test_superglm_publication_receipt.py
```

Expected: pass.

- [ ] **Step 7: Commit**

Run:

```bash
rtk git add pricing_pipeline/publishing/superglm_metadata.py tests/test_superglm_metadata.py
rtk git commit -m "Extract SuperGLM publication metadata"
```

---

### Task 4: Carry Receipt Paths Through Completed Build Contracts

**Files:**
- Modify: `pricing_pipeline/orchestration/publish_completed_build.py`
- Modify: `pricing_pipeline/models/spec.py`
- Modify: `pricing_pipeline/orchestration/pipeline.py`
- Test: `tests/test_publish_completed_build.py`

- [ ] **Step 1: Write failing contract tests**

Add tests:

```python
def test_completed_model_build_accepts_publication_receipt_fields():
    build = CompletedModelBuild(
        rating_workbook_path="/tmp/rating.xlsx",
        model_version="v1",
        effective_from="2026-06-19",
        publication_receipt_path="/tmp/superglm_publication_receipt.json",
        publication_receipt_sha256="a" * 64,
    )

    assert build.publication_receipt_path == "/tmp/superglm_publication_receipt.json"
    assert build.publication_receipt_sha256 == "a" * 64
    assert build.to_dict()["publication_receipt_sha256"] == "a" * 64


def test_completed_model_build_rejects_bad_receipt_hash():
    with pytest.raises(CompletedModelBuildError, match="publication_receipt_sha256"):
        CompletedModelBuild(
            rating_workbook_path="/tmp/rating.xlsx",
            model_version="v1",
            effective_from="2026-06-19",
            publication_receipt_sha256="not-a-hash",
        )
```

- [ ] **Step 2: Run tests red**

Run:

```bash
rtk uv run pytest -q tests/test_publish_completed_build.py -k publication_receipt
```

Expected: fail because the fields are unknown.

- [ ] **Step 3: Add fields to Pydantic model**

In `CompletedModelBuild`, add:

```python
publication_receipt_path: str | None = None
publication_receipt_sha256: str | None = None
```

Include `publication_receipt_path` in the optional text validator.

Add validator:

```python
@field_validator("publication_receipt_sha256", mode="before")
@classmethod
def _optional_sha256(cls, value: Any) -> str | None:
    if value is None:
        return None
    digest = str(value).strip()
    if len(digest) != 64 or digest.lower() != digest or not all(
        char in "0123456789abcdef" for char in digest
    ):
        raise ValueError("must be a 64-character lowercase hex SHA-256 digest")
    return digest
```

- [ ] **Step 4: Add fields to dataclasses**

In `pricing_pipeline/models/spec.py`, add to `ModelExportResult`:

```python
publication_receipt_path: str | None = None
publication_receipt_sha256: str | None = None
```

In `CompletedModelPublishResult`, add:

```python
publication_receipt_path: str | None = None
publication_receipt_sha256: str | None = None
```

Update `publish_completed_model_build(...)` and `publish_model_export(...)` construction to pass those fields through.

- [ ] **Step 5: Run tests green**

Run:

```bash
rtk uv run pytest -q tests/test_publish_completed_build.py tests/test_rate_package_lifecycle_workflow.py
```

Expected: pass.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add pricing_pipeline/orchestration/publish_completed_build.py pricing_pipeline/models/spec.py pricing_pipeline/orchestration/pipeline.py tests/test_publish_completed_build.py
rtk git commit -m "Carry publication receipt through completed build"
```

---

### Task 5: Stage Receipt Metadata

**Files:**
- Modify: `pricing_pipeline/publishing/staging.py`
- Modify: `tests/test_rating_export.py`

- [ ] **Step 1: Write failing staging tests**

Add tests that create a temporary workbook and receipt:

```python
def test_stage_rating_export_writes_receipt_and_term_metadata(engine, tmp_path):
    workbook = tmp_path / "rating.xlsx"
    write_minimal_rating_workbook(workbook, term_name="TermMonths", levels=[("12", 1.0), ("36", 3.0)])
    receipt = SuperGLMPublicationReceipt(
        schema_name="superglm_publication_receipt",
        schema_version=1,
        metadata_origin="SUPERGLM_FITTED_MODEL",
        superglm_version="0.10.0",
        extractor_version="1",
        package_metadata={"model": {"family": "poisson", "link": "log", "fit_used_offset": True}},
        term_metadata={
            "TermMonths": {
                "feature_kind": "offset",
                "source_term_name": "TermMonths",
                "published_term_name": "TermMonths",
            }
        },
        offset_contract=OffsetExportContract(
            handling="EXPORTED_FACTOR",
            source_factor_name="TermMonths",
            published_factor_name="TermMonths",
            source_name="TermMonths",
            label="log(TermMonths / 12)",
        ),
    )
    receipt_path = tmp_path / "superglm_publication_receipt.json"
    digest = write_publication_receipt(receipt, receipt_path)

    stage_rating_export(
        engine,
        workbook_path=workbook,
        export_id="export_1",
        model_name="offset_model",
        model_version="v1",
        effective_from="2026-06-19",
        publication_receipt_path=receipt_path,
        publication_receipt_sha256=digest,
        metadata_mode="REQUIRE_SUPERGLM_RECEIPT",
        replace=True,
        model_id=1,
    )

    with engine.begin() as con:
        export = con.execute(text("SELECT * FROM pricing_stg.STG_RATING_EXPORT WHERE export_id='export_1'")).mappings().one()
        term = con.execute(text("SELECT * FROM pricing_stg.STG_RATE_CELL WHERE export_id='export_1' AND term_name='TermMonths'")).mappings().first()
        metadata = con.execute(text("SELECT * FROM pricing_stg.STG_TERM_METADATA WHERE export_id='export_1' AND term_name='TermMonths'")).mappings().one()

    assert export["publication_receipt_sha256"] == digest
    assert export["offset_handling"] == "EXPORTED_FACTOR"
    assert export["offset_factor_name"] == "TermMonths"
    assert term["term_type"] == "OFFSET_FACTOR"
    assert json.loads(metadata["term_metadata_json"])["feature_kind"] == "offset"
```

- [ ] **Step 2: Run staging tests red**

Run:

```bash
rtk uv run pytest -q tests/test_rating_export.py -k receipt
```

Expected: fail because `stage_rating_export` does not accept receipt args.

- [ ] **Step 3: Move `clean_identifier` use to shared module**

In `staging.py`, replace local `clean_identifier` implementation with:

```python
from pricing_pipeline.publishing.naming import clean_identifier
```

Remove the local function.

- [ ] **Step 4: Extend staging args**

Add parameters to `stage_rating_export(...)`:

```python
publication_receipt_path: str | Path | None = None
publication_receipt_sha256: str | None = None
metadata_mode: Literal["REQUIRE_SUPERGLM_RECEIPT", "ALLOW_WORKBOOK_ONLY"] = "REQUIRE_SUPERGLM_RECEIPT"
```

When receipt path/hash are supplied, call:

```python
receipt = load_publication_receipt(
    publication_receipt_path,
    expected_sha256=publication_receipt_sha256,
)
```

When `metadata_mode == "REQUIRE_SUPERGLM_RECEIPT"` and receipt is missing, raise:

```python
ValueError("publication receipt is required")
```

- [ ] **Step 5: Write `STG_TERM_METADATA` rows**

Add a `term_metadata_df` built from `receipt.term_metadata`:

```python
term_metadata_df = pd.DataFrame(
    [
        {
            "export_id": args.export_id,
            "term_name": term_name,
            "term_metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        }
        for term_name, metadata in receipt.term_metadata.items()
    ]
)
```

For `EXPORTED_FACTOR`, use `receipt.offset_contract.published_factor_name` to override matching `rate_df["term_type"]` to `OFFSET_FACTOR`.

Add receipt/package fields to `export_df`:

```python
export_df["publication_receipt_json"] = canonical_receipt_bytes(receipt).decode("utf-8")
export_df["publication_receipt_sha256"] = args.publication_receipt_sha256
export_df["package_metadata_json"] = json.dumps(receipt.package_metadata, sort_keys=True, separators=(",", ":"))
export_df["offset_handling"] = receipt.offset_contract.handling
export_df["offset_factor_name"] = receipt.offset_contract.published_factor_name
export_df["offset_source_name"] = receipt.offset_contract.source_name
export_df["offset_label"] = receipt.offset_contract.label
export_df["metadata_origin"] = receipt.metadata_origin
```

- [ ] **Step 6: Delete staging metadata before parent row**

In replacement mode, delete:

```sql
DELETE FROM pricing_stg.STG_TERM_METADATA WHERE export_id = :export_id
```

before deleting `STG_RATING_EXPORT`.

- [ ] **Step 7: Insert staging metadata**

After `STG_RATING_EXPORT`, `STG_RATE_CELL`, and `STG_CELL_LEVEL`, insert `term_metadata_df` into `STG_TERM_METADATA` when non-empty.

- [ ] **Step 8: Run staging tests green**

Run:

```bash
rtk uv run pytest -q tests/test_rating_export.py -k receipt
```

Expected: pass.

- [ ] **Step 9: Commit**

Run:

```bash
rtk git add pricing_pipeline/publishing/staging.py tests/test_rating_export.py
rtk git commit -m "Stage SuperGLM publication receipt metadata"
```

---

### Task 6: Publish Receipt Metadata And Idempotency

**Files:**
- Modify: `pricing_pipeline/publishing/package_writer.py`
- Test: `tests/test_package_writer.py`

- [ ] **Step 1: Write failing package writer tests**

Add tests for SQL generated by package writer:

```python
def test_package_writer_compares_publication_receipt_hash_on_existing_export():
    existing = {
        "model_version": "v1",
        "effective_from_date": "2026-06-19",
        "effective_to_date": None,
        "source_file": "/tmp/rating.xlsx",
        "publication_receipt_sha256": "a" * 64,
    }
    staged = {
        "model_version": "v1",
        "effective_from_date": "2026-06-19",
        "effective_to_date": None,
        "source_file": "/tmp/rating.xlsx",
        "publication_receipt_sha256": "b" * 64,
    }

    assert "publication_receipt_sha256" in _existing_export_conflicts(existing, staged)[0]
```

Add a publish SQL assertion:

```python
def test_package_writer_inserts_receipt_metadata_columns(fake_engine):
    load_staging_to_rating_package(fake_engine, argparse.Namespace(export_id="export_1", package_status="PUBLISHED"))
    sql = "\n".join(fake_engine.sql)
    assert "publication_receipt_json" in sql
    assert "publication_receipt_sha256" in sql
    assert "package_metadata_json" in sql
    assert "term_metadata_json" in sql
    assert "STG_TERM_METADATA" in sql
```

- [ ] **Step 2: Run tests red**

Run:

```bash
rtk uv run pytest -q tests/test_package_writer.py -k "receipt or existing_export"
```

Expected: fail because package writer does not compare or insert receipt metadata.

- [ ] **Step 3: Include metadata in staging meta query**

In `package_writer.py`, extend the `STG_RATING_EXPORT` select:

```sql
publication_receipt_json,
publication_receipt_sha256,
package_metadata_json,
offset_handling,
offset_factor_name,
offset_source_name,
offset_label,
metadata_origin
```

- [ ] **Step 4: Include hash in idempotency conflict check**

Add `"publication_receipt_sha256"` to `_existing_export_conflicts(...)`.

- [ ] **Step 5: Insert package metadata columns**

Extend the `PRICING_RATE_PACKAGE` insert column list and parameter dict:

```sql
publication_receipt_json,
publication_receipt_sha256,
package_metadata_json,
revision_metadata_json,
offset_handling,
offset_factor_name,
offset_source_name,
offset_label,
metadata_origin,
```

Use `NULL` for `revision_metadata_json` on initial publish.

- [ ] **Step 6: Write term metadata into `PRICING_TERM`**

When inserting terms from staged rows, left join or look up `STG_TERM_METADATA` by `(export_id, term_name)` and insert `term_metadata_json` into `PRICING_TERM`.

For `OFFSET_FACTOR`, fail if the package offset contract does not match:

```python
if meta["offset_handling"] == "EXPORTED_FACTOR" and not meta["offset_factor_name"]:
    raise ValueError("EXPORTED_FACTOR requires offset_factor_name")
```

Also fail if `offset_handling = ALREADY_APPLIED_SQL_EXPOSURE` and any staged `STG_RATE_CELL.term_type = 'OFFSET_FACTOR'`.

- [ ] **Step 7: Run package writer tests green**

Run:

```bash
rtk uv run pytest -q tests/test_package_writer.py
```

Expected: pass.

- [ ] **Step 8: Commit**

Run:

```bash
rtk git add pricing_pipeline/publishing/package_writer.py tests/test_package_writer.py
rtk git commit -m "Publish SuperGLM receipt metadata"
```

---

### Task 7: Update Offline SQLite Publisher Parity

**Files:**
- Modify: `scripts/run_mtpl_frequency_offline_sqlite.py`
- Test: `tests/test_mtpl_offline_sqlite_runner.py`

- [ ] **Step 1: Write failing offline parity test**

Add:

```python
def test_offline_publisher_persists_receipt_metadata(tmp_path):
    result = run_mtpl_frequency_offline_sqlite.run_mtpl_frequency_offline_sqlite(
        db_root=tmp_path / "mtpl",
        row_count=40,
        synthetic_source=True,
        effective_from="2026-06-19",
        reset=True,
    )

    pricing_db = Path(result["db_paths"]["pricing"])
    with sqlite3.connect(pricing_db) as con:
        package = con.execute(
            "SELECT publication_receipt_sha256, offset_handling FROM PRICING_RATE_PACKAGE"
        ).fetchone()
        terms = con.execute(
            "SELECT COUNT(*) FROM PRICING_TERM WHERE term_metadata_json IS NOT NULL"
        ).fetchone()

    assert package[0] is not None
    assert package[1] in {"NONE", "EXPORTED_FACTOR", "ALREADY_APPLIED_SQL_EXPOSURE"}
    assert terms[0] > 0
```

- [ ] **Step 2: Run offline test red**

Run:

```bash
rtk uv run pytest -q tests/test_mtpl_offline_sqlite_runner.py -k receipt
```

Expected: fail until the offline runner writes a receipt and publisher copies metadata.

- [ ] **Step 3: Update offline runner training path**

In `run_mtpl_frequency_offline_sqlite`, after fitting/exporting the model, build a receipt with `OffsetExportContract` and write it with `write_publication_receipt(...)`. Add `publication_receipt_path` and `publication_receipt_sha256` to `completed_build`.

- [ ] **Step 4: Update offline publish helper**

In `publish_offline_rating_package(...)`:

- read metadata fields from `pricing_stg.STG_RATING_EXPORT`;
- include `publication_receipt_sha256` in existing package conflict checks;
- insert package metadata/offset fields into `PRICING_RATE_PACKAGE`;
- load `pricing_stg.STG_TERM_METADATA`;
- insert `PRICING_TERM.term_metadata_json`;
- reject `ALREADY_APPLIED_SQL_EXPOSURE` if any staged term has `term_type = 'OFFSET_FACTOR'`.

- [ ] **Step 5: Run offline tests green**

Run:

```bash
rtk uv run pytest -q tests/test_mtpl_offline_sqlite_runner.py
```

Expected: pass.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add scripts/run_mtpl_frequency_offline_sqlite.py tests/test_mtpl_offline_sqlite_runner.py
rtk git commit -m "Mirror receipt metadata in offline SQLite publisher"
```

---

### Task 8: Manual Revision Metadata Semantics

**Files:**
- Modify: `pricing_pipeline/publishing/manual_revision.py`
- Test: `tests/test_manual_revision.py`

- [ ] **Step 1: Write failing manual revision tests**

Add tests:

```python
def test_manual_revision_copies_receipt_and_term_metadata():
    # Use existing test fixture that creates parent package rows.
    # Add publication_receipt_json/hash/package_metadata_json/term_metadata_json
    # to the mocked parent rows.
    result = _write_manual_revision(...)

    assert inserted_package["publication_receipt_sha256"] == "a" * 64
    assert json.loads(inserted_package["revision_metadata_json"])["revision_kind"] == "MANUAL"
    assert copied_term["term_metadata_json"] == parent_term["term_metadata_json"]


def test_manual_revision_rejects_offset_factor_edits():
    parent_term["term_type"] = "OFFSET_FACTOR"
    edited_rate_cells = pd.DataFrame([...])

    with pytest.raises(ManualRevisionError, match="OFFSET_FACTOR"):
        _write_manual_revision(...)
```

- [ ] **Step 2: Run manual tests red**

Run:

```bash
rtk uv run pytest -q tests/test_manual_revision.py -k "metadata or OFFSET_FACTOR"
```

Expected: fail because metadata is not copied and offset edits are not rejected.

- [ ] **Step 3: Reject offset edits**

Before writing temp edit rows, query edited cell IDs joined to parent terms:

```sql
SELECT t.term_name
FROM pricing.PRICING_RATE_CELL AS rc
JOIN pricing.PRICING_TERM AS t ON t.term_id = rc.term_id
WHERE rc.cell_id IN (...)
  AND t.term_type = 'OFFSET_FACTOR'
```

If any rows return, raise:

```python
ManualRevisionError("manual revisions cannot edit OFFSET_FACTOR cells")
```

- [ ] **Step 4: Copy package metadata**

Extend the manual package insert columns:

```sql
publication_receipt_json,
publication_receipt_sha256,
package_metadata_json,
revision_metadata_json,
offset_handling,
offset_factor_name,
offset_source_name,
offset_label,
metadata_origin,
```

Set `revision_metadata_json` to deterministic JSON:

```python
json.dumps(
    {
        "revision_kind": "MANUAL",
        "parent_rate_package_id": parent_rate_package_id,
        "reason": reason,
    },
    sort_keys=True,
    separators=(",", ":"),
)
```

- [ ] **Step 5: Copy term metadata**

Extend the `MERGE pricing.PRICING_TERM` source/insert lists with `term_metadata_json`.

- [ ] **Step 6: Run manual tests green**

Run:

```bash
rtk uv run pytest -q tests/test_manual_revision.py
```

Expected: pass.

- [ ] **Step 7: Commit**

Run:

```bash
rtk git add pricing_pipeline/publishing/manual_revision.py tests/test_manual_revision.py
rtk git commit -m "Preserve receipt metadata in manual revisions"
```

---

### Task 9: Offset Export And SQL Scoring Semantics

**Files:**
- Modify: `pricing_pipeline/publishing/rating_export.py`
- Test: `tests/test_rating_export.py`
- Test: `tests/test_sql_server_syntax.py` or existing prediction-proc tests

- [ ] **Step 1: Write failing offset export tests**

Add tests proving:

```python
def test_exported_offset_factor_passes_offset_kwargs_to_superglm_export(fake_model, tmp_path):
    output = tmp_path / "rating.xlsx"
    export_rating_tables(
        fake_model,
        pd.DataFrame({"Region": ["A"]}),
        np.array([1.0]),
        np.array([1.0]),
        output_path=output,
        offset=np.array([np.log(3.0)]),
        offset_source=pd.Series([36], name="TermMonths"),
        offset_name="TermMonths",
        offset_max_exact_levels=50,
    )

    assert fake_model.export_kwargs["offset_name"] == "TermMonths"
    assert fake_model.export_kwargs["offset_max_exact_levels"] == 50
```

Add scoring semantic tests around the proc SQL text:

```python
def test_prediction_proc_exposure_contract_is_documented():
    proc = Path("db/migrations/V021__unify_model_name.sql").read_text(encoding="utf-8")
    assert "@base_rate * @exposure * EXP(SUM(log_coefficient))" in proc
```

- [ ] **Step 2: Run tests red**

Run:

```bash
rtk uv run pytest -q tests/test_rating_export.py -k offset
```

Expected: fail if the helper does not forward all offset kwargs from PR #15/current main.

- [ ] **Step 3: Extend export helper**

Ensure `export_rating_tables(...)` accepts and forwards:

```python
offset=None
offset_source=None
offset_name: str | None = None
offset_kind: str | None = None
offset_max_exact_levels: int | None = None
include_offset: bool | None = None
n_bins: int = 150
```

Only forward kwargs when not `None`.

- [ ] **Step 4: Add end-to-end scoring tests**

Add an offline scoring test that publishes two equivalent packages:

```text
EXPORTED_FACTOR:
  TermMonths=36 factor exists, score with exposure=1.0

ALREADY_APPLIED_SQL_EXPOSURE:
  no TermMonths factor, score with exposure=3.0
```

Assert both predictions are equal within tolerance.

- [ ] **Step 5: Run tests green**

Run:

```bash
rtk uv run pytest -q tests/test_rating_export.py tests/test_sql_server_syntax.py
```

Expected: pass.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add pricing_pipeline/publishing/rating_export.py tests/test_rating_export.py tests/test_sql_server_syntax.py
rtk git commit -m "Wire offset export and scoring semantics"
```

---

### Task 10: Final Integration Verification

**Files:**
- Modify as needed after previous tasks.
- Test all touched areas.

- [ ] **Step 1: Run focused suite**

Run:

```bash
rtk uv run pytest -q \
  tests/test_superglm_publication_receipt.py \
  tests/test_superglm_metadata.py \
  tests/test_rating_export.py \
  tests/test_package_writer.py \
  tests/test_manual_revision.py \
  tests/test_publish_completed_build.py \
  tests/test_mtpl_offline_sqlite_runner.py \
  tests/test_migrations.py \
  tests/test_sql_server_syntax.py
```

Expected: pass.

- [ ] **Step 2: Run full suite**

Run:

```bash
rtk uv run pytest -q
```

Expected: pass.

- [ ] **Step 3: Run lint and format**

Run:

```bash
rtk uv run ruff check pricing_pipeline scripts tests
rtk uv run ruff format --check pricing_pipeline scripts tests
rtk git diff --check
```

Expected: all pass.

- [ ] **Step 4: Run SQL checks**

Run:

```bash
rtk uv run pytest -q tests/test_sql_server_syntax.py
rtk uv run sqlfluff parse --dialect tsql db/migrations
```

Expected: pass.

- [ ] **Step 5: Run offline smoke**

Run:

```bash
rtk uv run python scripts/run_mtpl_frequency_offline_sqlite.py \
  --reset \
  --synthetic-source \
  --row-count 120 \
  --effective-from 2026-06-19 \
  --db-root state/offline/mtpl_frequency_receipt_smoke
```

Expected:

```text
pricing.sqlite exists
pricing_stg.sqlite exists
mlops.sqlite exists
PRICING_RATE_PACKAGE has publication_receipt_sha256
PRICING_TERM has non-null term_metadata_json rows
```

- [ ] **Step 6: Commit any final fixes**

If verification required small fixes, inspect the changed files and commit only the files touched by those fixes:

```bash
rtk git status --short
rtk git add pricing_pipeline scripts tests db
rtk git commit -m "Verify SuperGLM receipt integration"
```

- [ ] **Step 7: Open PR**

Run:

```bash
rtk git push -u origin feature/superglm-publication-receipt
rtk gh pr create --base main --head feature/superglm-publication-receipt --title "Add SuperGLM publication receipt metadata" --body-file docs/superpowers/plans/2026-06-19-superglm-publication-receipt.md
```

Expected: PR opens against `main`.
