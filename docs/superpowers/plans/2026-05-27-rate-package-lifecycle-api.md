# Rate Package Lifecycle API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe Airflow-oriented rate-package lifecycle API with stable model config, strict registry validation, candidate package publishing, a reusable deploy DAG, and controlled manual package revisions.

**Architecture:** Add typed lifecycle contracts and a thin `ModelPublisher` facade while keeping implementation logic in focused modules under `pricing_pipeline.publishing`. Move script-owned publish logic into package modules and leave scripts as CLI shims. Split package creation from deployment so production build DAGs create immutable `PUBLISHED` candidates and a generic deploy DAG moves live slots.

**Tech Stack:** Python 3.14, `tomllib`, dataclasses, SQLAlchemy text SQL, pandas, Apache Airflow 3.2.1, SQL Server migrations, pytest.

---

## File Structure

- Create `pricing_pipeline/models/config.py`
  - Owns `ModelBuildConfig`, TOML loading, and config validation.
- Create `pricing_models/mtpl_frequency/model.toml`
  - Stable model housekeeping config for the existing MTPL frequency model.
- Modify `pricing_models/mtpl_frequency/spec.py`
  - Load `model.toml` and keep executable training pieces in Python.
- Modify `pricing_pipeline/publishing/model_registry.py`
  - Add strict lookup, validation, and explicit registration functions.
  - Keep `ensure_pricing_model(...)` as a compatibility wrapper for scripts/tests that still need bootstrap behavior.
- Create `pricing_pipeline/publishing/lifecycle.py`
  - Owns public dataclasses: `RatePackageSelector`, `PublishResult`, `DeploymentResult`, `RatePackageSnapshot`, `RatePackageRevisionResult`, and `PredictionComparison`.
- Create `pricing_pipeline/publishing/publisher.py`
  - Owns `ModelPublisher`, the primary Airflow/operator-facing API.
- Create `pricing_pipeline/publishing/staging.py`
  - Package-native home for staging workbook functions currently in `scripts/load_superglm_excel_to_staging.py`.
- Create `pricing_pipeline/publishing/package_writer.py`
  - Package-native home for package creation functions currently in `scripts/load_staging_to_rating_package.py`.
- Create `pricing_pipeline/publishing/deployment.py`
  - Owns deploy selector resolution and deployment history writes.
- Create `pricing_pipeline/publishing/manual_revision.py`
  - Owns package snapshot loading, edit diffing, validation, and child-package revision creation.
- Create `pricing_pipeline/publishing/prediction_compare.py`
  - Owns advisory prediction comparison for package edits.
- Modify `pricing_pipeline/publishing/rating_package.py`
  - Turn it into a compatibility re-export module that imports package-native functions directly.
- Modify `scripts/load_superglm_excel_to_staging.py`
  - Keep CLI parser, delegate implementation to `pricing_pipeline.publishing.staging`.
- Modify `scripts/load_staging_to_rating_package.py`
  - Keep CLI parser, delegate implementation to `pricing_pipeline.publishing.package_writer`.
- Modify `pricing_pipeline/orchestration/pipeline.py`
  - Use `ModelPublisher` for registry validation and package publish.
  - Stop deploying from the build publish step.
- Modify `pricing_pipeline/orchestration/dag_factory.py`
  - Load model config and pass it into train/publish tasks.
- Create `dags/pricing_deploy_rate_package.py`
  - Generic manually triggered deploy DAG.
- Create migration `db/migrations/V016__rate_package_version_and_deploy_guards.sql`
  - Add unique `(model_id, package_version)` index.
- Modify docs/README where current behavior is described.
- Add or extend tests:
  - `tests/test_model_config.py`
  - `tests/test_model_registry.py`
  - `tests/test_model_publisher.py`
  - `tests/test_package_writer.py`
  - `tests/test_deploy_rate_package_dag.py`
  - `tests/test_manual_revision.py`
  - `tests/test_prediction_compare.py`
  - Existing `tests/test_rating_export.py`, `tests/test_migrations.py`, `tests/test_runtime_contract.py`, and `tests/test_dag_import.py`

---

### Task 1: Add Stable Model Config Loading

**Files:**
- Create: `pricing_pipeline/models/config.py`
- Create: `pricing_models/mtpl_frequency/model.toml`
- Modify: `pricing_models/mtpl_frequency/spec.py`
- Test: `tests/test_model_config.py`
- Test: `tests/test_model_layout.py`

- [ ] **Step 1: Write failing tests for TOML config loading**

Add `tests/test_model_config.py`:

```python
from pathlib import Path

import pytest

from pricing_pipeline.models.config import ModelBuildConfig, load_model_build_config


def test_load_model_build_config_reads_stable_metadata(tmp_path: Path):
    path = tmp_path / "model.toml"
    path.write_text(
        "\n".join(
            [
                'model_key = "MTPL_FREQ"',
                'model_label = "Motor frequency"',
                'target_name = "ClaimNb"',
                'model_type = "superglm_poisson"',
                'deployment_slot = "MTPL_FREQ_UAT"',
                'default_package_status = "PUBLISHED"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_model_build_config(path)

    assert config == ModelBuildConfig(
        model_key="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot="MTPL_FREQ_UAT",
        default_package_status="PUBLISHED",
    )


def test_load_model_build_config_rejects_missing_required_field(tmp_path: Path):
    path = tmp_path / "model.toml"
    path.write_text(
        "\n".join(
            [
                'model_key = "MTPL_FREQ"',
                'target_name = "ClaimNb"',
                'model_type = "superglm_poisson"',
                'deployment_slot = "MTPL_FREQ_UAT"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_label"):
        load_model_build_config(path)


def test_load_model_build_config_rejects_non_published_default_status(tmp_path: Path):
    path = tmp_path / "model.toml"
    path.write_text(
        "\n".join(
            [
                'model_key = "MTPL_FREQ"',
                'model_label = "Motor frequency"',
                'target_name = "ClaimNb"',
                'model_type = "superglm_poisson"',
                'deployment_slot = "MTPL_FREQ_UAT"',
                'default_package_status = "DRAFT"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="default_package_status"):
        load_model_build_config(path)
```

Extend `tests/test_model_layout.py`:

```python
from pricing_models.mtpl_frequency.spec import MODEL_CONFIG, MODEL_SPEC


def test_mtpl_frequency_model_config_matches_spec_identity():
    assert MODEL_CONFIG.model_key == MODEL_SPEC.model_key
    assert MODEL_CONFIG.model_label == MODEL_SPEC.model_label
    assert MODEL_CONFIG.target_name == MODEL_SPEC.target_name
    assert MODEL_CONFIG.model_type == MODEL_SPEC.model_type
    assert MODEL_CONFIG.deployment_slot == MODEL_SPEC.deployment_slot
    assert MODEL_CONFIG.default_package_status == "PUBLISHED"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk uv run pytest tests/test_model_config.py tests/test_model_layout.py -q
```

Expected: failure because `pricing_pipeline.models.config` and `MODEL_CONFIG` do not exist.

- [ ] **Step 3: Implement config dataclass and loader**

Create `pricing_pipeline/models/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class ModelBuildConfig:
    model_key: str
    model_label: str
    target_name: str
    model_type: str
    deployment_slot: str
    default_package_status: str = "PUBLISHED"


_REQUIRED_FIELDS = (
    "model_key",
    "model_label",
    "target_name",
    "model_type",
    "deployment_slot",
    "default_package_status",
)


def _require_non_empty_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model config field {field!r} must be a non-empty string")
    return value.strip()


def load_model_build_config(path: str | Path) -> ModelBuildConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    for field in _REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(f"model config missing required field {field!r}")

    config = ModelBuildConfig(
        model_key=_require_non_empty_string(data, "model_key"),
        model_label=_require_non_empty_string(data, "model_label"),
        target_name=_require_non_empty_string(data, "target_name"),
        model_type=_require_non_empty_string(data, "model_type"),
        deployment_slot=_require_non_empty_string(data, "deployment_slot"),
        default_package_status=_require_non_empty_string(data, "default_package_status"),
    )
    if config.default_package_status != "PUBLISHED":
        raise ValueError("default_package_status must be 'PUBLISHED' for production builds")
    return config
```

- [ ] **Step 4: Add the MTPL config file**

Create `pricing_models/mtpl_frequency/model.toml`:

```toml
model_key = "MTPL_FREQ"
model_label = "Motor frequency"
target_name = "ClaimNb"
model_type = "superglm_poisson"
deployment_slot = "MTPL_FREQ_UAT"
default_package_status = "PUBLISHED"
```

- [ ] **Step 5: Wire config into the MTPL spec module**

Modify `pricing_models/mtpl_frequency/spec.py`:

```python
from pathlib import Path

from pricing_pipeline.models.config import load_model_build_config
```

Then define `MODEL_CONFIG` before `MODEL_SPEC`:

```python
MODEL_CONFIG = load_model_build_config(Path(__file__).with_name("model.toml"))
```

Update `MODEL_SPEC` fields to read stable metadata from config:

```python
MODEL_SPEC = ModelSpec(
    model_key=MODEL_CONFIG.model_key,
    model_label=MODEL_CONFIG.model_label,
    target_name=MODEL_CONFIG.target_name,
    model_type=MODEL_CONFIG.model_type,
    experiment_name="pricing-mtpl-frequency",
    deployment_slot=MODEL_CONFIG.deployment_slot,
    dataset=FREMTPL_DATASET_SPEC,
    training_sql=TRAINING_SQL,
    feature_columns=tuple(FEATURE_COLUMNS),
    build_model=build_model,
    build_training_frame=build_training_frame,
    package_status=MODEL_CONFIG.default_package_status,
)
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
rtk uv run pytest tests/test_model_config.py tests/test_model_layout.py -q
```

Expected: pass.

Commit:

```bash
rtk git add pricing_pipeline/models/config.py pricing_models/mtpl_frequency/model.toml pricing_models/mtpl_frequency/spec.py tests/test_model_config.py tests/test_model_layout.py
rtk git commit -m "feat: add stable model build config"
```

---

### Task 2: Add Strict Model Registry Validation

**Files:**
- Modify: `pricing_pipeline/publishing/model_registry.py`
- Test: `tests/test_model_registry.py`
- Update: `tests/test_rating_export.py`

- [ ] **Step 1: Write failing tests for registry lookup and strict validation**

Create `tests/test_model_registry.py`:

```python
from types import SimpleNamespace

import pytest

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.model_registry import (
    ModelRegistryError,
    PricingModelRecord,
    get_pricing_model,
    register_pricing_model,
    validate_registered_model,
)


class FakeScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one(self):
        return self.value

    def scalar_one_or_none(self):
        return self.value


class FakeMappingsResult:
    def __init__(self, row):
        self.row = row

    def one_or_none(self):
        return self.row


class FakeResult:
    def __init__(self, row=None, scalar=None):
        self.row = row
        self.scalar = scalar

    def mappings(self):
        return FakeMappingsResult(self.row)

    def scalar_one(self):
        return self.scalar


class FakeConnection:
    def __init__(self, row=None, scalar=17):
        self.row = row
        self.scalar = scalar
        self.events = []

    def execute(self, statement, params=None):
        sql = str(statement)
        self.events.append((sql, params))
        if "SELECT model_id" in sql and "FROM pricing.PRICING_MODEL" in sql:
            return FakeResult(row=self.row, scalar=self.scalar)
        return FakeResult(scalar=self.scalar)


def config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_key="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot="MTPL_FREQ_UAT",
        default_package_status="PUBLISHED",
    )


def test_get_pricing_model_returns_record_for_existing_model():
    con = FakeConnection(
        {
            "model_id": 17,
            "model_key": "MTPL_FREQ",
            "model_label": "Motor frequency",
            "target_name": "ClaimNb",
            "model_type": "superglm_poisson",
            "model_status": "ACTIVE",
        }
    )

    record = get_pricing_model(con, "MTPL_FREQ")

    assert record == PricingModelRecord(
        model_id=17,
        model_key="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        model_status="ACTIVE",
    )


def test_validate_registered_model_fails_when_model_missing():
    con = FakeConnection(None)

    with pytest.raises(ModelRegistryError, match="not registered"):
        validate_registered_model(con, config())


def test_validate_registered_model_fails_on_metadata_mismatch():
    con = FakeConnection(
        {
            "model_id": 17,
            "model_key": "MTPL_FREQ",
            "model_label": "Motor frequency",
            "target_name": "LossCost",
            "model_type": "superglm_poisson",
            "model_status": "ACTIVE",
        }
    )

    with pytest.raises(ModelRegistryError, match="target_name"):
        validate_registered_model(con, config())


def test_register_pricing_model_inserts_without_updating_existing_rows():
    con = FakeConnection(scalar=17)

    model_id = register_pricing_model(con, config(), created_by="airflow")

    assert model_id == 17
    sql = con.events[0][0]
    assert "INSERT INTO pricing.PRICING_MODEL" in sql
    assert "MERGE" not in sql
    assert "UPDATE SET" not in sql
```

- [ ] **Step 2: Run registry tests to verify failure**

Run:

```bash
rtk uv run pytest tests/test_model_registry.py -q
```

Expected: failure because new registry objects do not exist.

- [ ] **Step 3: Implement registry records and strict validation**

Modify `pricing_pipeline/publishing/model_registry.py`.

Add imports:

```python
from dataclasses import dataclass

from pricing_pipeline.models.config import ModelBuildConfig
```

Add records and error:

```python
class ModelRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class PricingModelRecord:
    model_id: int
    model_key: str
    model_label: str | None
    target_name: str
    model_type: str
    model_status: str
```

Add strict functions above `ensure_pricing_model(...)`:

```python
def get_pricing_model(con: _Executable, model_key: str) -> PricingModelRecord | None:
    row = (
        con.execute(
            text(
                """
                SELECT
                    model_id,
                    model_key,
                    model_label,
                    target_name,
                    model_type,
                    model_status
                FROM pricing.PRICING_MODEL
                WHERE model_key = :model_key
                """
            ),
            {"model_key": model_key},
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        return None
    return PricingModelRecord(
        model_id=int(row["model_id"]),
        model_key=str(row["model_key"]),
        model_label=row["model_label"],
        target_name=str(row["target_name"]),
        model_type=str(row["model_type"]),
        model_status=str(row["model_status"]),
    )


def validate_registered_model(con: _Executable, config: ModelBuildConfig) -> PricingModelRecord:
    record = get_pricing_model(con, config.model_key)
    if record is None:
        raise ModelRegistryError(
            f"model_key {config.model_key!r} is not registered; run explicit model registration first"
        )

    mismatches: list[str] = []
    if record.model_label != config.model_label:
        mismatches.append(f"model_label db={record.model_label!r} config={config.model_label!r}")
    if record.target_name != config.target_name:
        mismatches.append(f"target_name db={record.target_name!r} config={config.target_name!r}")
    if record.model_type != config.model_type:
        mismatches.append(f"model_type db={record.model_type!r} config={config.model_type!r}")
    if record.model_status != "ACTIVE":
        mismatches.append(f"model_status db={record.model_status!r} expected='ACTIVE'")

    if mismatches:
        raise ModelRegistryError(
            f"registered model {config.model_key!r} does not match config: "
            + "; ".join(mismatches)
        )
    return record


def register_pricing_model(con: _Executable, config: ModelBuildConfig, *, created_by: str) -> int:
    con.execute(
        text(
            """
            INSERT INTO pricing.PRICING_MODEL (
                model_key,
                model_label,
                target_name,
                model_type,
                model_status,
                created_by
            )
            SELECT
                :model_key,
                :model_label,
                :target_name,
                :model_type,
                'ACTIVE',
                :created_by
            WHERE NOT EXISTS (
                SELECT 1
                FROM pricing.PRICING_MODEL WITH (UPDLOCK, HOLDLOCK)
                WHERE model_key = :model_key
            );
            """
        ),
        {
            "model_key": config.model_key,
            "model_label": config.model_label,
            "target_name": config.target_name,
            "model_type": config.model_type,
            "created_by": created_by,
        },
    )
    return int(
        con.execute(
            text(
                """
                SELECT model_id
                FROM pricing.PRICING_MODEL
                WHERE model_key = :model_key
                """
            ),
            {"model_key": config.model_key},
        ).scalar_one()
    )
```

Keep `ensure_pricing_model(...)` unchanged for compatibility in this task.

- [ ] **Step 4: Run focused tests and update old tests if SQL string expectations changed**

Run:

```bash
rtk uv run pytest tests/test_model_registry.py tests/test_rating_export.py::test_ensure_pricing_model_merges_by_model_key_and_returns_model_id -q
```

Expected: pass. If `test_ensure_pricing_model_merges_by_model_key_and_returns_model_id` fails, preserve the old `ensure_pricing_model(...)` behavior exactly and keep strict validation separate.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add pricing_pipeline/publishing/model_registry.py tests/test_model_registry.py tests/test_rating_export.py
rtk git commit -m "feat: add strict model registry validation"
```

---

### Task 3: Add Lifecycle Contracts and ModelPublisher Facade

**Files:**
- Create: `pricing_pipeline/publishing/lifecycle.py`
- Create: `pricing_pipeline/publishing/publisher.py`
- Test: `tests/test_model_publisher.py`

- [ ] **Step 1: Write failing tests for `ModelPublisher` construction and delegation**

Create `tests/test_model_publisher.py`:

```python
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.lifecycle import RatePackageSelector
from pricing_pipeline.publishing.publisher import ModelPublisher


def config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_key="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot="MTPL_FREQ_UAT",
        default_package_status="PUBLISHED",
    )


def test_model_publisher_stores_engine_and_config():
    engine = object()
    publisher = ModelPublisher(engine, config())

    assert publisher.engine is engine
    assert publisher.config.model_key == "MTPL_FREQ"


def test_rate_package_selector_requires_one_selector():
    assert RatePackageSelector(rate_package_id=123).rate_package_id == 123
    assert RatePackageSelector(package_version=7).package_version == 7


def test_model_publisher_validate_registered_model_delegates(monkeypatch):
    calls = []
    engine = object()
    publisher = ModelPublisher(engine, config())

    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.validate_model_on_engine",
        lambda engine_arg, config_arg: calls.append((engine_arg, config_arg)) or 17,
    )

    assert publisher.validate_registered_model() == 17
    assert calls == [(engine, config())]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk uv run pytest tests/test_model_publisher.py -q
```

Expected: failure because lifecycle and publisher modules do not exist.

- [ ] **Step 3: Add lifecycle dataclasses**

Create `pricing_pipeline/publishing/lifecycle.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RatePackageSelector:
    rate_package_id: int | None = None
    package_version: int | None = None

    def __post_init__(self) -> None:
        selected = [
            self.rate_package_id is not None,
            self.package_version is not None,
        ]
        if sum(selected) != 1:
            raise ValueError("exactly one rate package selector is required")


@dataclass(frozen=True)
class PublishResult:
    mlflow_run_id: str
    export_id: str
    rate_package_id: int
    package_version: int
    rating_workbook_path: str


@dataclass(frozen=True)
class DeploymentResult:
    model_id: int
    deployment_slot: str
    previous_rate_package_id: int | None
    rate_package_id: int
    package_version: int
    deployed_by: str
    deployment_reason: str


@dataclass(frozen=True)
class RatePackageSnapshot:
    metadata: dict[str, Any]
    terms: pd.DataFrame
    rate_cells: pd.DataFrame
    cell_levels: pd.DataFrame
    compiled_rate_cells: pd.DataFrame
    compiled_1d_bands: pd.DataFrame


@dataclass(frozen=True)
class RatePackageRevisionResult:
    rate_package_id: int
    package_version: int
    parent_rate_package_id: int
    changed_rate_cell_count: int
    base_rate_changed: bool
    diff_summary: pd.DataFrame


@dataclass(frozen=True)
class PredictionComparison:
    summary: dict[str, float]
    changed_rows: pd.DataFrame
```

- [ ] **Step 4: Add publisher facade skeleton**

Create `pricing_pipeline/publishing/publisher.py`:

```python
from __future__ import annotations

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.lifecycle import RatePackageSelector
from pricing_pipeline.publishing.model_registry import validate_registered_model


def validate_model_on_engine(engine, config: ModelBuildConfig) -> int:
    with engine.begin() as con:
        return validate_registered_model(con, config).model_id


class ModelPublisher:
    def __init__(self, engine, config: ModelBuildConfig):
        self.engine = engine
        self.config = config

    def validate_registered_model(self) -> int:
        return validate_model_on_engine(self.engine, self.config)

    def load_rate_package(self, selector: RatePackageSelector):
        from pricing_pipeline.publishing.manual_revision import load_rate_package_snapshot

        return load_rate_package_snapshot(self.engine, self.config, selector)
```

Only include methods with implemented delegates in this task. The publish,
deploy, revision, and comparison methods are added in the tasks that introduce
their backing modules.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk uv run pytest tests/test_model_publisher.py -q
```

Expected: pass.

Commit:

```bash
rtk git add pricing_pipeline/publishing/lifecycle.py pricing_pipeline/publishing/publisher.py tests/test_model_publisher.py
rtk git commit -m "feat: add model publisher facade"
```

---

### Task 4: Extract Staging Logic From Script to Package Module

**Files:**
- Create: `pricing_pipeline/publishing/staging.py`
- Modify: `scripts/load_superglm_excel_to_staging.py`
- Modify: `pricing_pipeline/publishing/rating_package.py`
- Test: `tests/test_rating_export.py`

- [ ] **Step 1: Write contract test that package module exposes staging functions**

Add to `tests/test_rating_export.py`:

```python
def test_package_staging_module_exposes_excel_staging_functions():
    from pricing_pipeline.publishing import staging

    assert callable(staging.build_staging_frames)
    assert callable(staging.insert_staging_frames)
    assert callable(staging.stage_rating_export)
```

- [ ] **Step 2: Run focused test to verify failure**

Run:

```bash
rtk uv run pytest tests/test_rating_export.py::test_package_staging_module_exposes_excel_staging_functions -q
```

Expected: failure because `pricing_pipeline.publishing.staging` does not exist.

- [ ] **Step 3: Move staging implementation into package module**

Create `pricing_pipeline/publishing/staging.py` by moving these functions and constants from `scripts/load_superglm_excel_to_staging.py`:

- `INTERVAL_RE`
- `RANGE_RE`
- `cell_to_zero_index`
- `clean_text`
- `clean_identifier`
- `parse_interval`
- `find_blocks`
- `infer_term_type`
- `split_interaction_level`
- `build_staging_frames`
- `insert_staging_frames`
- `stage_rating_export`

Keep imports needed by those functions:

```python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from pricing_pipeline.publishing.model_registry import ensure_pricing_model
```

Do not move `parse_args()`, `main()`, `ROOT`, `sys.path` mutation, or `get_engine`.

- [ ] **Step 4: Turn script into a CLI shim**

Modify `scripts/load_superglm_excel_to_staging.py` so it imports implementation functions:

```python
from pricing_pipeline.publishing.staging import (
    build_staging_frames,
    insert_staging_frames,
    stage_rating_export,
)
```

Keep `parse_args()` and `main()` in the script. Remove duplicate implementation functions from the script after confirming tests import from the package module.

- [ ] **Step 5: Update compatibility wrapper**

Modify `pricing_pipeline/publishing/rating_package.py` to import directly:

```python
from pricing_pipeline.publishing.staging import stage_rating_export
```

Leave `publish_rating_package` pointing to the current script until package writer extraction is complete in Task 5.

- [ ] **Step 6: Run staging-related tests and commit**

Run:

```bash
rtk uv run pytest tests/test_rating_export.py -q
```

Expected: pass.

Commit:

```bash
rtk git add pricing_pipeline/publishing/staging.py scripts/load_superglm_excel_to_staging.py pricing_pipeline/publishing/rating_package.py tests/test_rating_export.py
rtk git commit -m "refactor: move rating export staging into package"
```

---

### Task 5: Extract Package Writer and Return Package Metadata

**Files:**
- Create: `pricing_pipeline/publishing/package_writer.py`
- Modify: `scripts/load_staging_to_rating_package.py`
- Modify: `pricing_pipeline/publishing/rating_package.py`
- Modify: `pricing_pipeline/publishing/publisher.py`
- Test: `tests/test_package_writer.py`
- Update: `tests/test_rating_export.py`

- [ ] **Step 1: Write tests for package writer result shape and no-deploy default**

Create `tests/test_package_writer.py`:

```python
from types import SimpleNamespace

from pricing_pipeline.publishing.lifecycle import PublishResult
from pricing_pipeline.publishing.package_writer import publish_rating_package


def test_publish_rating_package_builds_args_without_deployment_pointer(monkeypatch):
    captured = []

    def fake_load(engine, args):
        captured.append((engine, args))
        args.package_version = 3
        return 42

    monkeypatch.setattr(
        "pricing_pipeline.publishing.package_writer.load_staging_to_rating_package",
        fake_load,
    )
    engine = object()

    result = publish_rating_package(
        engine,
        export_id="export-1",
        created_by="airflow",
        package_status="PUBLISHED",
    )

    assert result == PublishResult(
        mlflow_run_id="",
        export_id="export-1",
        rate_package_id=42,
        package_version=3,
        rating_workbook_path="",
    )
    args = captured[0][1]
    assert args.export_id == "export-1"
    assert args.created_by == "airflow"
    assert args.package_status == "PUBLISHED"
    assert args.set_pointer is None
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
rtk uv run pytest tests/test_package_writer.py -q
```

Expected: failure because `package_writer` does not exist.

- [ ] **Step 3: Move package writer implementation into package module**

Create `pricing_pipeline/publishing/package_writer.py` by moving `load_staging_to_rating_package(...)` from `scripts/load_staging_to_rating_package.py`.

Keep imports:

```python
from __future__ import annotations

import argparse

from sqlalchemy import text

from pricing_pipeline.publishing.lifecycle import PublishResult
from pricing_pipeline.publishing.model_registry import ensure_pricing_model
```

Add package-native wrapper:

```python
def publish_rating_package(
    engine,
    *,
    export_id: str,
    created_by: str = "python",
    package_status: str = "PUBLISHED",
) -> PublishResult:
    args = argparse.Namespace(
        export_id=export_id,
        created_by=created_by,
        package_status=package_status,
        set_pointer=None,
    )
    rate_package_id = load_staging_to_rating_package(engine, args)
    return PublishResult(
        mlflow_run_id="",
        export_id=export_id,
        rate_package_id=int(rate_package_id),
        package_version=int(args.package_version),
        rating_workbook_path="",
    )
```

Keep pointer/deployment SQL inside `load_staging_to_rating_package(...)` temporarily for script compatibility, but package-native `publish_rating_package(...)` must always set `set_pointer=None`.

- [ ] **Step 4: Turn script into CLI shim with explicit legacy pointer behavior**

Modify `scripts/load_staging_to_rating_package.py`:

```python
from pricing_pipeline.publishing.package_writer import (
    load_staging_to_rating_package,
)
```

Keep `parse_args()` and `main()` in the script. Add a script-level compatibility wrapper:

```python
def publish_rating_package(
    engine,
    *,
    export_id: str,
    pointer_name: str | None,
    created_by: str = "python",
    package_status: str = "DRAFT",
) -> int:
    args = argparse.Namespace(
        export_id=export_id,
        created_by=created_by,
        package_status=package_status,
        set_pointer=pointer_name,
    )
    return load_staging_to_rating_package(engine, args)
```

This preserves existing tests and demos that import from the script.

- [ ] **Step 5: Update `rating_package.py` to direct imports**

Modify `pricing_pipeline/publishing/rating_package.py`:

```python
from pricing_pipeline.publishing.package_writer import publish_rating_package
from pricing_pipeline.publishing.staging import stage_rating_export
```

Remove `_ensure_scripts_path()` and dynamic `import_module(...)`.

- [ ] **Step 6: Add publisher method for training export publish**

Modify `pricing_pipeline/publishing/publisher.py`:

```python
from pricing_pipeline.models.spec import ModelExportResult
from pricing_pipeline.publishing.lifecycle import PublishResult
from pricing_pipeline.publishing.package_writer import publish_rating_package
from pricing_pipeline.publishing.staging import stage_rating_export
```

Add method:

```python
    def publish_training_export(self, export: ModelExportResult | dict) -> PublishResult:
        export_result = ModelExportResult.from_mapping(export)
        stage_rating_export(
            self.engine,
            workbook_path=Path(export_result.rating_workbook_path),
            export_id=export_result.export_id,
            model_name=self.config.model_key,
            model_version=export_result.model_version,
            target_name=self.config.target_name,
            model_type=self.config.model_type,
            effective_from=export_result.effective_from,
            created_by=export_result.created_by,
            replace=True,
        )
        result = publish_rating_package(
            self.engine,
            export_id=export_result.export_id,
            created_by=export_result.created_by,
            package_status=self.config.default_package_status,
        )
        return PublishResult(
            mlflow_run_id=export_result.mlflow_run_id,
            export_id=result.export_id,
            rate_package_id=result.rate_package_id,
            package_version=result.package_version,
            rating_workbook_path=export_result.rating_workbook_path,
        )
```

Add `from pathlib import Path`.

- [ ] **Step 7: Run package writer and rating export tests**

Run:

```bash
rtk uv run pytest tests/test_package_writer.py tests/test_rating_export.py -q
```

Expected: pass after updating old wrapper expectations where `pricing_pipeline.publishing.rating_package.publish_rating_package(...)` now returns `PublishResult`.

- [ ] **Step 8: Commit**

Run:

```bash
rtk git add pricing_pipeline/publishing/package_writer.py scripts/load_staging_to_rating_package.py pricing_pipeline/publishing/rating_package.py pricing_pipeline/publishing/publisher.py tests/test_package_writer.py tests/test_rating_export.py
rtk git commit -m "refactor: move rating package writer into package"
```

---

### Task 6: Add Package Version Constraint and Atomic Allocation

**Files:**
- Create: `db/migrations/V016__rate_package_version_and_deploy_guards.sql`
- Modify: `pricing_pipeline/publishing/package_writer.py`
- Test: `tests/test_migrations.py`
- Test: `tests/test_package_writer.py`

- [ ] **Step 1: Write migration tests**

Add to `tests/test_migrations.py`:

```python
def test_rate_package_version_guard_migration_adds_unique_model_version_index():
    migration = Path("db/migrations/V016__rate_package_version_and_deploy_guards.sql").read_text(
        encoding="utf-8"
    )

    assert "UX_PRICING_RATE_PACKAGE_MODEL_VERSION" in migration
    assert "PRICING_RATE_PACKAGE(model_id, package_version)" in migration
    assert "WHERE model_id IS NOT NULL" in migration


def test_package_writer_allocates_version_under_lock():
    writer = Path("pricing_pipeline/publishing/package_writer.py").read_text(encoding="utf-8")

    assert "WITH (UPDLOCK, HOLDLOCK)" in writer
    assert "MAX(package_version)" in writer
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk uv run pytest tests/test_migrations.py::test_rate_package_version_guard_migration_adds_unique_model_version_index tests/test_migrations.py::test_package_writer_allocates_version_under_lock -q
```

Expected: failure because migration and locking SQL do not exist.

- [ ] **Step 3: Add migration**

Create `db/migrations/V016__rate_package_version_and_deploy_guards.sql`:

```sql
IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_PRICING_RATE_PACKAGE_MODEL_VERSION'
      AND object_id = OBJECT_ID('pricing.PRICING_RATE_PACKAGE')
)
CREATE UNIQUE INDEX UX_PRICING_RATE_PACKAGE_MODEL_VERSION
ON pricing.PRICING_RATE_PACKAGE(model_id, package_version)
WHERE model_id IS NOT NULL;
GO
```

- [ ] **Step 4: Update package version SQL**

In `pricing_pipeline/publishing/package_writer.py`, replace:

```sql
SELECT ISNULL(MAX(package_version), 0) + 1
FROM pricing.PRICING_RATE_PACKAGE
WHERE model_id = :model_id
```

with:

```sql
SELECT ISNULL(MAX(package_version), 0) + 1
FROM pricing.PRICING_RATE_PACKAGE WITH (UPDLOCK, HOLDLOCK)
WHERE model_id = :model_id
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk uv run pytest tests/test_migrations.py tests/test_package_writer.py -q
```

Expected: pass.

Commit:

```bash
rtk git add db/migrations/V016__rate_package_version_and_deploy_guards.sql pricing_pipeline/publishing/package_writer.py tests/test_migrations.py
rtk git commit -m "feat: guard rate package version allocation"
```

---

### Task 7: Update Training Publish Flow to Create Candidates Only

**Files:**
- Modify: `pricing_pipeline/orchestration/pipeline.py`
- Modify: `pricing_pipeline/orchestration/dag_factory.py`
- Modify: `pricing_pipeline/models/spec.py`
- Modify: `dags/pricing_mtpl_frequency.py`
- Modify: `dags/pricing_superglm_pipeline.py`
- Test: `tests/test_rating_export.py`
- Test: `tests/test_dag_import.py`

- [ ] **Step 1: Write tests for candidate publish behavior**

Update `tests/test_rating_export.py` in the existing pipeline flow test:

```python
publish_call = next(event for event in calls if event[0] == "publish_training_export")
assert publish_call[2]["export_id"] == export_id
assert publish_call[2]["deployment_slot"] == "MTPL_FREQ_UAT"
```

Replace assertions that expect `publish_rating_package(... pointer_name="MTPL_FREQ_UAT" ...)` with assertions that a `ModelPublisher` is created and `publish_training_export(...)` is called.

Add a focused test:

```python
def test_publish_model_export_returns_candidate_without_deploying(monkeypatch, tmp_path: Path):
    calls = []

    class FakePublisher:
        def __init__(self, engine, config):
            calls.append(("publisher_init", engine, config))

        def validate_registered_model(self):
            calls.append(("validate_registered_model",))
            return 17

        def publish_training_export(self, export):
            calls.append(("publish_training_export", export))
            return SimpleNamespace(
                mlflow_run_id="mlflow-run-1",
                export_id=export["export_id"],
                rate_package_id=123,
                package_version=4,
                rating_workbook_path=export["rating_workbook_path"],
            )

    monkeypatch.setattr(pipeline, "ModelPublisher", FakePublisher)
    engine = object()
    export = {
        "model_id": 17,
        "model_key": "MTPL_FREQ",
        "model_version": "20260527",
        "model_type": "superglm_poisson",
        "target_name": "ClaimNb",
        "deployment_slot": "MTPL_FREQ_UAT",
        "manifest_id": "manifest-1",
        "dag_id": "pricing_dag",
        "airflow_run_id": "manual__1",
        "mlflow_run_id": "mlflow-run-1",
        "split_set_id": None,
        "export_id": "export-1",
        "rating_workbook_path": str(tmp_path / "rating_tables.xlsx"),
        "effective_from": "2026-05-27",
        "created_by": "airflow",
        "package_status": "PUBLISHED",
    }

    result = pipeline.publish_model_export(engine, export)

    assert result["rate_package_id"] == "123"
    assert result["package_version"] == "4"
    assert ("validate_registered_model",) in calls
```

- [ ] **Step 2: Run focused tests to verify failure**

Run:

```bash
rtk uv run pytest tests/test_rating_export.py -q
```

Expected: failure because pipeline still calls old staging and publish functions directly.

- [ ] **Step 3: Add config to `ModelSpec` or pass it beside the spec**

Prefer passing config explicitly through DAG factory instead of embedding it deeply in `ModelSpec`.

Modify `pricing_pipeline/orchestration/dag_factory.py` signature:

```python
from pricing_pipeline.models.config import ModelBuildConfig


def build_pricing_model_dag(
    *,
    dag_id: str,
    spec: ModelSpec,
    model_config: ModelBuildConfig,
    schedule=None,
    tags: list[str] | None = None,
):
```

Update DAG files:

```python
from pricing_models.mtpl_frequency.spec import MODEL_CONFIG, MODEL_SPEC
```

Call:

```python
pricing_mtpl_frequency = build_pricing_model_dag(
    dag_id="pricing_mtpl_frequency",
    spec=MODEL_SPEC,
    model_config=MODEL_CONFIG,
    tags=["pricing", "mtpl", "frequency", "mlflow"],
)
```

Make the same change in `dags/pricing_superglm_pipeline.py`.

- [ ] **Step 4: Update publish pipeline to use `ModelPublisher`**

Modify `pricing_pipeline/orchestration/pipeline.py` imports:

```python
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.publisher import ModelPublisher
```

Change `publish_model_export` signature:

```python
def publish_model_export(
    engine,
    export: ModelExportResult | dict,
    *,
    model_config: ModelBuildConfig,
) -> dict[str, str]:
```

Implementation:

```python
    export_result = ModelExportResult.from_mapping(export)
    publisher = ModelPublisher(engine, model_config)
    publisher.validate_registered_model()
    publish_result = publisher.publish_training_export(export_result)
    record_model_run(
        engine,
        dag_id=export_result.dag_id,
        airflow_run_id=export_result.airflow_run_id,
        mlflow_run_id=export_result.mlflow_run_id,
        manifest_id=export_result.manifest_id,
        split_set_id=export_result.split_set_id,
        export_id=export_result.export_id,
        model_id=export_result.model_id,
        model_name=export_result.model_key,
        model_version=export_result.model_version,
        rate_package_id=publish_result.rate_package_id,
        rating_workbook_path=publish_result.rating_workbook_path,
        run_status="SUCCESS",
        created_by=export_result.created_by,
    )
    return {
        "mlflow_run_id": publish_result.mlflow_run_id,
        "export_id": publish_result.export_id,
        "rate_package_id": str(publish_result.rate_package_id),
        "package_version": str(publish_result.package_version),
        "rating_workbook_path": publish_result.rating_workbook_path,
    }
```

Update `run_training_export_publish(...)` to require `model_config` and pass it through.

- [ ] **Step 5: Update DAG factory publish task**

In `pricing_pipeline/orchestration/dag_factory.py`:

```python
        @task
        def publish_export(export: dict[str, Any]) -> dict[str, str]:
            return publish_model_export(
                get_engine(settings_from_env()),
                export,
                model_config=model_config,
            )
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
rtk uv run pytest tests/test_rating_export.py tests/test_dag_import.py -q
```

Expected: pass.

Commit:

```bash
rtk git add pricing_pipeline/orchestration/pipeline.py pricing_pipeline/orchestration/dag_factory.py dags/pricing_mtpl_frequency.py dags/pricing_superglm_pipeline.py tests/test_rating_export.py tests/test_dag_import.py
rtk git commit -m "feat: publish candidate packages without deploying"
```

---

### Task 8: Add Deployment API

**Files:**
- Create: `pricing_pipeline/publishing/deployment.py`
- Modify: `pricing_pipeline/publishing/publisher.py`
- Test: `tests/test_deployment.py`

- [ ] **Step 1: Write tests for deploy validation and SQL shape**

Create `tests/test_deployment.py`:

```python
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.deployment import deploy_rate_package


class FakeScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class FakeMappingsResult:
    def __init__(self, row):
        self.row = row

    def one(self):
        return self.row

    def one_or_none(self):
        return self.row


class FakeResult:
    def __init__(self, row=None, scalar=None):
        self.row = row
        self.scalar = scalar

    def mappings(self):
        return FakeMappingsResult(self.row)

    def scalar_one_or_none(self):
        return self.scalar


class FakeConnection:
    def __init__(self):
        self.events = []

    def execute(self, statement, params=None):
        sql = str(statement)
        self.events.append((sql, params))
        if "FROM pricing.PRICING_RATE_PACKAGE" in sql:
            return FakeResult(
                {
                    "model_id": 17,
                    "rate_package_id": 123,
                    "package_version": 4,
                    "package_status": "PUBLISHED",
                }
            )
        if "FROM pricing.PRICING_MODEL_DEPLOYMENT" in sql:
            return FakeResult(scalar=122)
        return FakeResult()


class FakeBegin:
    def __init__(self, con):
        self.con = con

    def __enter__(self):
        return self.con

    def __exit__(self, *args):
        return None


class FakeEngine:
    def __init__(self, con):
        self.con = con

    def begin(self):
        return FakeBegin(self.con)


def config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_key="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot="MTPL_FREQ_UAT",
        default_package_status="PUBLISHED",
    )


def test_deploy_rate_package_closes_current_and_inserts_new_deployment():
    con = FakeConnection()
    result = deploy_rate_package(
        FakeEngine(con),
        config(),
        rate_package_id=123,
        package_version=None,
        deployment_slot=None,
        deployment_reason="Reviewed and approved",
        deployed_by="mhick",
        model_id=17,
    )

    assert result.previous_rate_package_id == 122
    assert result.rate_package_id == 123
    assert result.deployment_slot == "MTPL_FREQ_UAT"
    sql_text = "\n".join(event[0] for event in con.events)
    assert "UPDATE pricing.PRICING_MODEL_DEPLOYMENT" in sql_text
    assert "INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT" in sql_text
    assert "deployment_note" in sql_text
    assert "MERGE pricing.PRICING_PACKAGE_POINTER" in sql_text
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk uv run pytest tests/test_deployment.py -q
```

Expected: failure because deployment module does not exist.

- [ ] **Step 3: Implement deployment API**

Create `pricing_pipeline/publishing/deployment.py`:

```python
from __future__ import annotations

from sqlalchemy import text

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.lifecycle import DeploymentResult


class DeploymentError(RuntimeError):
    pass


def deploy_rate_package(
    engine,
    config: ModelBuildConfig,
    *,
    rate_package_id: int | None,
    package_version: int | None,
    deployment_slot: str | None,
    deployment_reason: str,
    deployed_by: str,
    model_id: int,
) -> DeploymentResult:
    slot = deployment_slot or config.deployment_slot
    if not deployment_reason.strip():
        raise DeploymentError("deployment_reason is required")
    if not deployed_by.strip():
        raise DeploymentError("deployed_by is required")
    if (rate_package_id is None) == (package_version is None):
        raise DeploymentError("provide exactly one of rate_package_id or package_version")

    if rate_package_id is not None:
        package_filter = "rate_package_id = :rate_package_id"
        package_params = {"rate_package_id": rate_package_id, "model_id": model_id}
    else:
        package_filter = "model_id = :model_id AND package_version = :package_version"
        package_params = {"package_version": package_version, "model_id": model_id}

    with engine.begin() as con:
        package = (
            con.execute(
                text(
                    f"""
                    SELECT
                        model_id,
                        rate_package_id,
                        package_version,
                        package_status
                    FROM pricing.PRICING_RATE_PACKAGE
                    WHERE {package_filter}
                    """
                ),
                package_params,
            )
            .mappings()
            .one()
        )
        if int(package["model_id"]) != int(model_id):
            raise DeploymentError("package belongs to another model")
        if package["package_status"] != "PUBLISHED":
            raise DeploymentError("only PUBLISHED packages can be deployed")

        previous_rate_package_id = con.execute(
            text(
                """
                SELECT rate_package_id
                FROM pricing.PRICING_MODEL_DEPLOYMENT
                WHERE model_id = :model_id
                  AND deployment_slot = :deployment_slot
                  AND effective_to_ts IS NULL
                """
            ),
            {"model_id": model_id, "deployment_slot": slot},
        ).scalar_one_or_none()
        resolved_rate_package_id = int(package["rate_package_id"])
        if previous_rate_package_id == resolved_rate_package_id:
            raise DeploymentError("package is already current in the requested slot")

        con.execute(
            text(
                """
                UPDATE pricing.PRICING_MODEL_DEPLOYMENT
                SET effective_to_ts = SYSUTCDATETIME()
                WHERE model_id = :model_id
                  AND deployment_slot = :deployment_slot
                  AND effective_to_ts IS NULL;
                """
            ),
            {"model_id": model_id, "deployment_slot": slot},
        )
        con.execute(
            text(
                """
                INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT (
                    model_id,
                    rate_package_id,
                    deployment_slot,
                    deployed_by,
                    deployment_note
                )
                VALUES (
                    :model_id,
                    :rate_package_id,
                    :deployment_slot,
                    :deployed_by,
                    :deployment_note
                );
                """
            ),
            {
                "model_id": model_id,
                "rate_package_id": resolved_rate_package_id,
                "deployment_slot": slot,
                "deployed_by": deployed_by,
                "deployment_note": deployment_reason,
            },
        )
        con.execute(
            text(
                """
                MERGE pricing.PRICING_PACKAGE_POINTER AS tgt
                USING (
                    SELECT
                        :model_id AS model_id,
                        :pointer_name AS pointer_name,
                        :rate_package_id AS rate_package_id,
                        :updated_by AS updated_by
                ) AS src
                ON tgt.model_id = src.model_id
                   AND tgt.pointer_name = src.pointer_name
                WHEN MATCHED THEN
                    UPDATE SET
                        rate_package_id = src.rate_package_id,
                        updated_ts = SYSUTCDATETIME(),
                        updated_by = src.updated_by
                WHEN NOT MATCHED THEN
                    INSERT (model_id, pointer_name, rate_package_id, updated_by)
                    VALUES (src.model_id, src.pointer_name, src.rate_package_id, src.updated_by);
                """
            ),
            {
                "model_id": model_id,
                "pointer_name": slot,
                "rate_package_id": resolved_rate_package_id,
                "updated_by": deployed_by,
            },
        )
    return DeploymentResult(
        model_id=model_id,
        deployment_slot=slot,
        previous_rate_package_id=previous_rate_package_id,
        rate_package_id=resolved_rate_package_id,
        package_version=int(package["package_version"]),
        deployed_by=deployed_by,
        deployment_reason=deployment_reason,
    )
```

- [ ] **Step 4: Add publisher deploy method**

Modify `pricing_pipeline/publishing/publisher.py`:

```python
    def deploy(
        self,
        *,
        rate_package_id: int | None = None,
        package_version: int | None = None,
        deployment_reason: str,
        deployed_by: str,
        deployment_slot: str | None = None,
    ):
        from pricing_pipeline.publishing.deployment import deploy_rate_package

        model_id = self.validate_registered_model()
        return deploy_rate_package(
            self.engine,
            self.config,
            rate_package_id=rate_package_id,
            package_version=package_version,
            deployment_slot=deployment_slot,
            deployment_reason=deployment_reason,
            deployed_by=deployed_by,
            model_id=model_id,
        )
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk uv run pytest tests/test_deployment.py tests/test_model_publisher.py -q
```

Expected: pass.

Commit:

```bash
rtk git add pricing_pipeline/publishing/deployment.py pricing_pipeline/publishing/publisher.py tests/test_deployment.py
rtk git commit -m "feat: add rate package deployment api"
```

---

### Task 9: Add Generic Deploy DAG

**Files:**
- Create: `dags/pricing_deploy_rate_package.py`
- Test: `tests/test_deploy_rate_package_dag.py`
- Test: `tests/test_dag_import.py`

- [ ] **Step 1: Write DAG tests**

Create `tests/test_deploy_rate_package_dag.py`:

```python
import importlib


def test_deploy_rate_package_dag_imports():
    module = importlib.import_module("dags.pricing_deploy_rate_package")

    assert module.pricing_deploy_rate_package.dag_id == "pricing_deploy_rate_package"


def test_deploy_rate_package_dag_exposes_param_names():
    module = importlib.import_module("dags.pricing_deploy_rate_package")
    dag = module.pricing_deploy_rate_package

    for name in [
        "model_key",
        "rate_package_id",
        "package_version",
        "deployment_slot",
        "deployment_reason",
        "deployed_by",
    ]:
        assert name in dag.params
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk uv run pytest tests/test_deploy_rate_package_dag.py -q
```

Expected: failure because deploy DAG does not exist.

- [ ] **Step 3: Implement deploy DAG**

Create `dags/pricing_deploy_rate_package.py`:

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow.sdk import dag, get_current_context, task

from pricing_models.registry import get_model_spec
from pricing_pipeline.infra.db import get_engine
from pricing_pipeline.models.config import load_model_build_config
from pricing_pipeline.publishing.publisher import ModelPublisher


def _config_path_for_model_key(model_key: str) -> Path:
    spec = get_model_spec(model_key)
    module = __import__(spec.__module__, fromlist=["__file__"])
    return Path(module.__file__).with_name("model.toml")


@dag(
    dag_id="pricing_deploy_rate_package",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["pricing", "deploy"],
    params={
        "model_key": "",
        "rate_package_id": None,
        "package_version": None,
        "deployment_slot": None,
        "deployment_reason": "",
        "deployed_by": "",
    },
)
def _deploy_rate_package_dag():
    @task
    def deploy_package() -> dict[str, str]:
        context = get_current_context()
        params = context["params"]
        model_key = str(params["model_key"]).strip()
        if not model_key:
            raise ValueError("model_key is required")
        rate_package_id = params.get("rate_package_id")
        package_version = params.get("package_version")
        if (rate_package_id is None) == (package_version is None):
            raise ValueError("provide exactly one of rate_package_id or package_version")

        config = load_model_build_config(_config_path_for_model_key(model_key))
        publisher = ModelPublisher(get_engine(), config)
        result = publisher.deploy(
            rate_package_id=None if rate_package_id is None else int(rate_package_id),
            package_version=None if package_version is None else int(package_version),
            deployment_slot=params.get("deployment_slot"),
            deployment_reason=str(params["deployment_reason"]),
            deployed_by=str(params["deployed_by"]),
        )
        return {
            "model_key": config.model_key,
            "deployment_slot": result.deployment_slot,
            "previous_rate_package_id": str(result.previous_rate_package_id),
            "rate_package_id": str(result.rate_package_id),
            "package_version": str(result.package_version),
            "deployed_by": result.deployed_by,
            "deployment_reason": result.deployment_reason,
        }

    deploy_package()


pricing_deploy_rate_package = _deploy_rate_package_dag()
```

- [ ] **Step 4: Run DAG tests and commit**

Run:

```bash
rtk uv run pytest tests/test_deploy_rate_package_dag.py tests/test_dag_import.py -q
```

Expected: pass.

Commit:

```bash
rtk git add dags/pricing_deploy_rate_package.py tests/test_deploy_rate_package_dag.py tests/test_dag_import.py
rtk git commit -m "feat: add generic rate package deploy dag"
```

---

### Task 10: Add Package Snapshot Loading

**Files:**
- Create: `pricing_pipeline/publishing/manual_revision.py`
- Modify: `pricing_pipeline/publishing/publisher.py`
- Test: `tests/test_manual_revision.py`

- [ ] **Step 1: Write tests for selector SQL and snapshot shape**

Create `tests/test_manual_revision.py`:

```python
import pandas as pd

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.lifecycle import RatePackageSelector, RatePackageSnapshot
from pricing_pipeline.publishing.manual_revision import load_rate_package_snapshot


class FakeEngine:
    pass


def config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_key="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot="MTPL_FREQ_UAT",
        default_package_status="PUBLISHED",
    )


def test_load_rate_package_snapshot_uses_rate_package_id_selector(monkeypatch):
    calls = []

    def fake_read_sql_query(sql, engine, params=None):
        calls.append((sql, engine, params))
        if "PRICING_RATE_PACKAGE" in str(sql):
            return pd.DataFrame(
                [
                    {
                        "model_id": 17,
                        "rate_package_id": 123,
                        "package_version": 4,
                        "package_status": "PUBLISHED",
                    }
                ]
            )
        return pd.DataFrame()

    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision.pd.read_sql_query",
        fake_read_sql_query,
    )

    snapshot = load_rate_package_snapshot(
        FakeEngine(),
        config(),
        RatePackageSelector(rate_package_id=123),
    )

    assert isinstance(snapshot, RatePackageSnapshot)
    assert snapshot.metadata["rate_package_id"] == 123
    assert calls[0][2] == {"model_key": "MTPL_FREQ", "rate_package_id": 123}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk uv run pytest tests/test_manual_revision.py -q
```

Expected: failure because `manual_revision` does not exist.

- [ ] **Step 3: Implement snapshot loader**

Create `pricing_pipeline/publishing/manual_revision.py`:

```python
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.lifecycle import RatePackageSelector, RatePackageSnapshot


class ManualRevisionError(RuntimeError):
    pass


def _package_where_clause(selector: RatePackageSelector) -> tuple[str, dict[str, int]]:
    if selector.rate_package_id is not None:
        return "rp.rate_package_id = :rate_package_id", {
            "rate_package_id": selector.rate_package_id
        }
    if selector.package_version is not None:
        return "rp.package_version = :package_version", {
            "package_version": selector.package_version
        }
    raise ManualRevisionError("unsupported selector")


def load_rate_package_snapshot(
    engine,
    config: ModelBuildConfig,
    selector: RatePackageSelector,
) -> RatePackageSnapshot:
    where_sql, selector_params = _package_where_clause(selector)
    params = {"model_key": config.model_key, **selector_params}
    metadata_df = pd.read_sql_query(
        text(
            f"""
            SELECT
                rp.model_id,
                rp.rate_package_id,
                rp.parent_rate_package_id,
                rp.model_name,
                rp.model_version,
                rp.package_version,
                rp.base_rate,
                rp.effective_from_date,
                rp.effective_to_date,
                rp.package_status,
                rp.created_ts,
                rp.created_by
            FROM pricing.PRICING_RATE_PACKAGE rp
            JOIN pricing.PRICING_MODEL m
              ON m.model_id = rp.model_id
            WHERE m.model_key = :model_key
              AND {where_sql}
            """
        ),
        engine,
        params=params,
    )
    if len(metadata_df) != 1:
        raise ManualRevisionError("rate package selector must resolve to exactly one package")
    metadata = metadata_df.iloc[0].to_dict()
    rate_package_id = int(metadata["rate_package_id"])

    terms = pd.read_sql_query(
        text("SELECT * FROM pricing.PRICING_TERM WHERE rate_package_id = :rate_package_id"),
        engine,
        params={"rate_package_id": rate_package_id},
    )
    rate_cells = pd.read_sql_query(
        text(
            """
            SELECT rc.*
            FROM pricing.PRICING_RATE_CELL rc
            JOIN pricing.PRICING_TERM t
              ON t.term_id = rc.term_id
            WHERE t.rate_package_id = :rate_package_id
            """
        ),
        engine,
        params={"rate_package_id": rate_package_id},
    )
    cell_levels = pd.read_sql_query(
        text(
            """
            SELECT rcl.*
            FROM pricing.PRICING_RATE_CELL_LEVEL rcl
            JOIN pricing.PRICING_RATE_CELL rc
              ON rc.cell_id = rcl.cell_id
            JOIN pricing.PRICING_TERM t
              ON t.term_id = rc.term_id
            WHERE t.rate_package_id = :rate_package_id
            """
        ),
        engine,
        params={"rate_package_id": rate_package_id},
    )
    compiled_rate_cells = pd.read_sql_query(
        text(
            "SELECT * FROM pricing.PRICING_COMPILED_RATE_CELL WHERE rate_package_id = :rate_package_id"
        ),
        engine,
        params={"rate_package_id": rate_package_id},
    )
    compiled_1d_bands = pd.read_sql_query(
        text(
            "SELECT * FROM pricing.PRICING_COMPILED_1D_RATE_BAND WHERE rate_package_id = :rate_package_id"
        ),
        engine,
        params={"rate_package_id": rate_package_id},
    )
    return RatePackageSnapshot(
        metadata=metadata,
        terms=terms,
        rate_cells=rate_cells,
        cell_levels=cell_levels,
        compiled_rate_cells=compiled_rate_cells,
        compiled_1d_bands=compiled_1d_bands,
    )
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
rtk uv run pytest tests/test_manual_revision.py tests/test_model_publisher.py -q
```

Expected: pass.

Commit:

```bash
rtk git add pricing_pipeline/publishing/manual_revision.py pricing_pipeline/publishing/publisher.py tests/test_manual_revision.py
rtk git commit -m "feat: load rate package snapshots"
```

---

### Task 11: Add Manual Revision Diff Validation

**Files:**
- Modify: `pricing_pipeline/publishing/manual_revision.py`
- Test: `tests/test_manual_revision.py`

- [ ] **Step 1: Add tests for editable diff rules**

Append to `tests/test_manual_revision.py`:

```python
import pytest

from pricing_pipeline.publishing.manual_revision import diff_rate_cell_edits, validate_rate_cell_edits


def base_rate_cells() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cell_id": 1,
                "term_id": 10,
                "cell_key_text": "VehAge=[0, 1)",
                "multiplier": 1.0,
                "log_coefficient": 0.0,
            },
            {
                "cell_id": 2,
                "term_id": 10,
                "cell_key_text": "VehAge=[1, 2)",
                "multiplier": 1.2,
                "log_coefficient": 0.1823215568,
            },
        ]
    )


def test_diff_rate_cell_edits_detects_multiplier_change():
    original = base_rate_cells()
    edited = original.copy()
    edited.loc[1, "multiplier"] = 1.25

    diff = diff_rate_cell_edits(original, edited)

    assert diff[["cell_id", "old_multiplier", "new_multiplier"]].to_dict("records") == [
        {"cell_id": 2, "old_multiplier": 1.2, "new_multiplier": 1.25}
    ]


def test_validate_rate_cell_edits_rejects_empty_diff():
    original = base_rate_cells()

    with pytest.raises(ValueError, match="no manual rate cell changes"):
        validate_rate_cell_edits(original, original.copy())


def test_validate_rate_cell_edits_rejects_identity_column_change():
    original = base_rate_cells()
    edited = original.copy()
    edited.loc[0, "cell_key_text"] = "changed"

    with pytest.raises(ValueError, match="forbidden column"):
        validate_rate_cell_edits(original, edited)


def test_validate_rate_cell_edits_rejects_non_positive_multiplier():
    original = base_rate_cells()
    edited = original.copy()
    edited.loc[0, "multiplier"] = 0.0

    with pytest.raises(ValueError, match="positive finite"):
        validate_rate_cell_edits(original, edited)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk uv run pytest tests/test_manual_revision.py -q
```

Expected: failure because diff functions do not exist.

- [ ] **Step 3: Implement diff and validation**

Add to `pricing_pipeline/publishing/manual_revision.py`:

```python
import numpy as np
```

Add functions:

```python
_IDENTITY_COLUMNS = {
    "cell_id",
    "term_id",
    "cell_key_text",
    "cell_key_digest",
    "is_reference",
    "is_default",
}


def diff_rate_cell_edits(original: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
    merged = original.merge(
        edited,
        on="cell_id",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    changed = merged[
        ~np.isclose(
            merged["multiplier_old"].astype(float),
            merged["multiplier_new"].astype(float),
        )
    ].copy()
    if changed.empty:
        return pd.DataFrame(
            columns=["cell_id", "old_multiplier", "new_multiplier", "old_log_coefficient"]
        )
    return pd.DataFrame(
        {
            "cell_id": changed["cell_id"],
            "old_multiplier": changed["multiplier_old"].astype(float),
            "new_multiplier": changed["multiplier_new"].astype(float),
            "old_log_coefficient": changed["log_coefficient_old"].astype(float),
        }
    )


def validate_rate_cell_edits(original: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
    if set(original["cell_id"]) != set(edited["cell_id"]):
        raise ValueError("edited rate cells must contain the same cell_id values")
    original_by_id = original.set_index("cell_id").sort_index()
    edited_by_id = edited.set_index("cell_id").sort_index()

    for column in _IDENTITY_COLUMNS - {"cell_id"}:
        if column in original_by_id.columns and column in edited_by_id.columns:
            if not original_by_id[column].equals(edited_by_id[column]):
                raise ValueError(f"forbidden column edited: {column}")

    multipliers = edited_by_id["multiplier"].astype(float)
    if not np.isfinite(multipliers).all() or (multipliers <= 0).any():
        raise ValueError("multipliers must be positive finite numbers")

    diff = diff_rate_cell_edits(original, edited)
    if diff.empty:
        raise ValueError("no manual rate cell changes detected")
    return diff
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
rtk uv run pytest tests/test_manual_revision.py -q
```

Expected: pass.

Commit:

```bash
rtk git add pricing_pipeline/publishing/manual_revision.py tests/test_manual_revision.py
rtk git commit -m "feat: validate manual rate cell edits"
```

---

### Task 12: Add Manual Revision Creation API

**Files:**
- Modify: `pricing_pipeline/publishing/manual_revision.py`
- Modify: `pricing_pipeline/publishing/publisher.py`
- Test: `tests/test_manual_revision.py`

- [ ] **Step 1: Write tests for revision API validation**

Append to `tests/test_manual_revision.py`:

```python
from pricing_pipeline.publishing.lifecycle import RatePackageSnapshot
from pricing_pipeline.publishing.manual_revision import create_manual_revision


def snapshot_for_revision() -> RatePackageSnapshot:
    return RatePackageSnapshot(
        metadata={
            "model_id": 17,
            "rate_package_id": 123,
            "package_version": 4,
            "package_status": "PUBLISHED",
            "base_rate": 0.12,
        },
        terms=pd.DataFrame(),
        rate_cells=base_rate_cells(),
        cell_levels=pd.DataFrame(),
        compiled_rate_cells=pd.DataFrame(),
        compiled_1d_bands=pd.DataFrame(),
    )


def test_create_manual_revision_requires_reason():
    snapshot = snapshot_for_revision()
    edited = snapshot.rate_cells.copy()
    edited.loc[0, "multiplier"] = 1.1

    with pytest.raises(ValueError, match="reason"):
        create_manual_revision(
            object(),
            config(),
            parent=snapshot,
            edited_rate_cells=edited,
            reason="",
            created_by="mhick",
        )


def test_create_manual_revision_returns_revision_result(monkeypatch):
    snapshot = snapshot_for_revision()
    edited = snapshot.rate_cells.copy()
    edited.loc[0, "multiplier"] = 1.1

    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision._write_manual_revision",
        lambda *args, **kwargs: (124, 5),
    )

    result = create_manual_revision(
        object(),
        config(),
        parent=snapshot,
        edited_rate_cells=edited,
        reason="Temporary adjustment",
        created_by="mhick",
    )

    assert result.rate_package_id == 124
    assert result.package_version == 5
    assert result.parent_rate_package_id == 123
    assert result.changed_rate_cell_count == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
rtk uv run pytest tests/test_manual_revision.py -q
```

Expected: failure because `create_manual_revision` does not exist.

- [ ] **Step 3: Implement public revision API and writer seam**

Add to `pricing_pipeline/publishing/manual_revision.py`:

```python
from pricing_pipeline.publishing.lifecycle import RatePackageRevisionResult
```

Add functions:

```python
def _write_manual_revision(
    engine,
    config: ModelBuildConfig,
    *,
    parent: RatePackageSnapshot,
    edited_rate_cells: pd.DataFrame,
    diff: pd.DataFrame,
    reason: str,
    created_by: str,
) -> tuple[int, int]:
    raise NotImplementedError("manual revision SQL writer is implemented in the next step")


def create_manual_revision(
    engine,
    config: ModelBuildConfig,
    *,
    parent: RatePackageSnapshot,
    edited_rate_cells: pd.DataFrame,
    reason: str,
    created_by: str,
) -> RatePackageRevisionResult:
    if not reason.strip():
        raise ValueError("manual revision reason is required")
    if not created_by.strip():
        raise ValueError("manual revision created_by is required")
    if parent.metadata["package_status"] != "PUBLISHED":
        raise ValueError("manual revisions require a PUBLISHED parent package")

    diff = validate_rate_cell_edits(parent.rate_cells, edited_rate_cells)
    rate_package_id, package_version = _write_manual_revision(
        engine,
        config,
        parent=parent,
        edited_rate_cells=edited_rate_cells,
        diff=diff,
        reason=reason,
        created_by=created_by,
    )
    return RatePackageRevisionResult(
        rate_package_id=int(rate_package_id),
        package_version=int(package_version),
        parent_rate_package_id=int(parent.metadata["rate_package_id"]),
        changed_rate_cell_count=len(diff),
        base_rate_changed=False,
        diff_summary=diff,
    )
```

- [ ] **Step 4: Add publisher method**

Modify `pricing_pipeline/publishing/publisher.py`:

```python
    def create_manual_revision(
        self,
        *,
        parent,
        edited_rate_cells,
        reason: str,
        created_by: str,
    ):
        from pricing_pipeline.publishing.manual_revision import create_manual_revision

        self.validate_registered_model()
        return create_manual_revision(
            self.engine,
            self.config,
            parent=parent,
            edited_rate_cells=edited_rate_cells,
            reason=reason,
            created_by=created_by,
        )
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk uv run pytest tests/test_manual_revision.py tests/test_model_publisher.py -q
```

Expected: pass because the writer is monkeypatched in the public API test.

Commit:

```bash
rtk git add pricing_pipeline/publishing/manual_revision.py pricing_pipeline/publishing/publisher.py tests/test_manual_revision.py
rtk git commit -m "feat: add manual revision api"
```

---

### Task 13: Implement Manual Revision SQL Writer

**Files:**
- Modify: `pricing_pipeline/publishing/manual_revision.py`
- Test: `tests/test_manual_revision.py`
- Test: `tests/test_runtime_contract.py`

- [ ] **Step 1: Add SQL writer contract tests**

Append to `tests/test_runtime_contract.py`:

```python
def test_manual_revision_writer_creates_child_package_and_finalizes_published():
    writer = Path("pricing_pipeline/publishing/manual_revision.py").read_text(
        encoding="utf-8"
    )

    assert "parent_rate_package_id" in writer
    assert "package_status" in writer
    assert "PUBLISHED" in writer
    assert "WITH (UPDLOCK, HOLDLOCK)" in writer
    assert "PRICING_RATE_PACKAGE" in writer
    assert "PRICING_RATE_CELL" in writer
    assert "PRICING_COMPILED_RATE_CELL" in writer
```

- [ ] **Step 2: Run contract test to verify failure**

Run:

```bash
rtk uv run pytest tests/test_runtime_contract.py::test_manual_revision_writer_creates_child_package_and_finalizes_published -q
```

Expected: failure because writer is still a stub.

- [ ] **Step 3: Replace writer stub with SQL implementation**

In `pricing_pipeline/publishing/manual_revision.py`, replace `_write_manual_revision(...)`.

Implementation requirements:

- Open `engine.begin()` once.
- Allocate `package_version` using `WITH (UPDLOCK, HOLDLOCK)`.
- Insert a child row into `pricing.PRICING_RATE_PACKAGE` with:
  - `parent_rate_package_id = parent.metadata["rate_package_id"]`
  - `model_id = parent.metadata["model_id"]`
  - copied `model_name`, `model_version`, `effective_from_date`, `effective_to_date`
  - copied `base_rate`
  - initial `package_status = "DRAFT"`
  - `created_by = created_by`
- Copy terms from parent to child and keep an old-term to new-term mapping.
- Copy rate cells using edited multiplier/log coefficient values for changed cells.
- Copy rate cell level mappings using old-cell to new-cell mapping.
- Copy compiled flat cells using edited values.
- Copy compiled 1D bands using edited values where the source row maps to changed cells.
- Finalize child package to `PUBLISHED`.
- Return `(rate_package_id, package_version)`.

Use temporary mapping tables inside the transaction:

```sql
DECLARE @term_map TABLE (old_term_id BIGINT NOT NULL, new_term_id BIGINT NOT NULL);
DECLARE @cell_map TABLE (old_cell_id BIGINT NOT NULL, new_cell_id BIGINT NOT NULL);
```

Use pandas only to build parameter rows for changed `cell_id -> multiplier` values:

```python
changed_rows = [
    {
        "cell_id": int(row.cell_id),
        "multiplier": float(row.new_multiplier),
        "log_coefficient": float(np.log(row.new_multiplier)),
    }
    for row in diff.itertuples(index=False)
]
```

Persist the changed rows into a transaction-local temp table:

```sql
CREATE TABLE #manual_rate_cell_edits (
    cell_id BIGINT NOT NULL PRIMARY KEY,
    multiplier DECIMAL(19,10) NOT NULL,
    log_coefficient DECIMAL(19,12) NOT NULL
);
```

Insert rows with `con.execute(text("INSERT INTO #manual_rate_cell_edits ..."), changed_rows)`.

For unchanged rows, use original values. For changed rows, use `COALESCE(edit.multiplier, rc.multiplier)` and `COALESCE(edit.log_coefficient, rc.log_coefficient)`.

- [ ] **Step 4: Run manual revision tests**

Run:

```bash
rtk uv run pytest tests/test_manual_revision.py tests/test_runtime_contract.py::test_manual_revision_writer_creates_child_package_and_finalizes_published -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add pricing_pipeline/publishing/manual_revision.py tests/test_manual_revision.py tests/test_runtime_contract.py
rtk git commit -m "feat: write manual rate package revisions"
```

---

### Task 14: Add Prediction Comparison Helper

**Files:**
- Create: `pricing_pipeline/publishing/prediction_compare.py`
- Modify: `pricing_pipeline/publishing/publisher.py`
- Test: `tests/test_prediction_compare.py`

- [ ] **Step 1: Write tests for comparison summary**

Create `tests/test_prediction_compare.py`:

```python
import pandas as pd

from pricing_pipeline.publishing.prediction_compare import compare_prediction_vectors


def test_compare_prediction_vectors_reports_summary_and_top_changes():
    before = pd.Series([100.0, 200.0, 300.0], name="before")
    after = pd.Series([110.0, 190.0, 330.0], name="after")

    comparison = compare_prediction_vectors(before, after, top_n=2)

    assert comparison.summary == {
        "row_count": 3.0,
        "mean_absolute_change": 16.666666666666668,
        "max_absolute_change": 30.0,
        "mean_relative_change": 0.06666666666666667,
        "max_relative_change": 0.1,
    }
    assert comparison.changed_rows["absolute_change"].tolist() == [30.0, 10.0]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
rtk uv run pytest tests/test_prediction_compare.py -q
```

Expected: failure because module does not exist.

- [ ] **Step 3: Implement vector comparison helper**

Create `pricing_pipeline/publishing/prediction_compare.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from pricing_pipeline.publishing.lifecycle import PredictionComparison


def compare_prediction_vectors(
    before: pd.Series,
    after: pd.Series,
    *,
    top_n: int = 25,
) -> PredictionComparison:
    before_values = before.astype(float).reset_index(drop=True)
    after_values = after.astype(float).reset_index(drop=True)
    if len(before_values) != len(after_values):
        raise ValueError("before and after predictions must have the same length")
    absolute_change = (after_values - before_values).abs()
    relative_change = absolute_change / before_values.replace(0.0, np.nan).abs()
    relative_change = relative_change.fillna(0.0)
    changed_rows = pd.DataFrame(
        {
            "row_index": range(len(before_values)),
            "before": before_values,
            "after": after_values,
            "absolute_change": absolute_change,
            "relative_change": relative_change,
        }
    ).sort_values(["absolute_change", "row_index"], ascending=[False, True])
    return PredictionComparison(
        summary={
            "row_count": float(len(before_values)),
            "mean_absolute_change": float(absolute_change.mean()),
            "max_absolute_change": float(absolute_change.max()),
            "mean_relative_change": float(relative_change.mean()),
            "max_relative_change": float(relative_change.max()),
        },
        changed_rows=changed_rows.head(top_n).reset_index(drop=True),
    )
```

- [ ] **Step 4: Add publisher convenience method**

Modify `pricing_pipeline/publishing/publisher.py`:

```python
    def compare_prediction_vectors(self, before, after, *, top_n: int = 25):
        from pricing_pipeline.publishing.prediction_compare import compare_prediction_vectors

        return compare_prediction_vectors(before, after, top_n=top_n)
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk uv run pytest tests/test_prediction_compare.py tests/test_model_publisher.py -q
```

Expected: pass.

Commit:

```bash
rtk git add pricing_pipeline/publishing/prediction_compare.py pricing_pipeline/publishing/publisher.py tests/test_prediction_compare.py
rtk git commit -m "feat: compare package prediction changes"
```

---

### Task 15: Update Docs and Runtime Contracts

**Files:**
- Modify: `README.md`
- Modify: `tests/test_readme_contract.py`
- Modify: `tests/test_runtime_contract.py`

- [ ] **Step 1: Add README contract tests**

Extend `tests/test_readme_contract.py`:

```python
def test_readme_documents_rate_package_lifecycle_api():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "model.toml" in readme
    assert "ModelPublisher" in readme
    assert "pricing_deploy_rate_package" in readme
    assert "PUBLISHED" in readme
    assert "manual revision" in readme.lower()
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
rtk uv run pytest tests/test_readme_contract.py -q
```

Expected: failure because README does not document the new lifecycle API.

- [ ] **Step 3: Update README**

Add a section after "Pricing Model History":

```markdown
## Rate Package Lifecycle

Production model builds use stable model metadata from `model.toml` and the
`ModelPublisher` API. The config file records SQL housekeeping identity such as
`model_key`, `target_name`, `model_type`, and the default deployment slot. SQL
Server owns `model_id`.

Airflow build DAGs create immutable `PUBLISHED` package candidates. They do not
move live deployment pointers by default. Deployments are handled by the generic
`pricing_deploy_rate_package` DAG, which accepts a reviewed `rate_package_id`,
validates that it belongs to the configured model, and moves the deployment slot
with an audit reason.

Manual rate changes are implemented as manual revisions: load an existing
package, edit constrained rate-cell DataFrames, create a child package with
`parent_rate_package_id`, and deploy that child through the same deploy DAG.
Published package rows are never edited directly.
```

- [ ] **Step 4: Run contract tests**

Run:

```bash
rtk uv run pytest tests/test_readme_contract.py tests/test_runtime_contract.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add README.md tests/test_readme_contract.py tests/test_runtime_contract.py
rtk git commit -m "docs: document rate package lifecycle"
```

---

### Task 16: Final Verification

**Files:**
- No new files.
- Validate complete task set.

- [ ] **Step 1: Run ruff**

Run:

```bash
rtk uv run ruff check .
```

Expected: pass.

- [ ] **Step 2: Run focused lifecycle tests**

Run:

```bash
rtk uv run pytest tests/test_model_config.py tests/test_model_registry.py tests/test_model_publisher.py tests/test_package_writer.py tests/test_deployment.py tests/test_deploy_rate_package_dag.py tests/test_manual_revision.py tests/test_prediction_compare.py -q
```

Expected: pass.

- [ ] **Step 3: Run broader regression tests**

Run:

```bash
rtk uv run pytest tests/test_rating_export.py tests/test_migrations.py tests/test_runtime_contract.py tests/test_dag_import.py tests/test_model_layout.py tests/test_readme_contract.py -q
```

Expected: pass.

- [ ] **Step 4: Run full test suite**

Run:

```bash
rtk uv run pytest tests/ -q
```

Expected: pass.

- [ ] **Step 5: Inspect git status**

Run:

```bash
rtk git status --short
```

Expected: only unrelated pre-existing untracked files may remain, such as `.claude/`.

- [ ] **Step 6: Commit final verification note only if docs changed during verification**

If verification required docs/test command updates, commit those exact files:

```bash
rtk git add README.md tests/test_readme_contract.py
rtk git commit -m "docs: clarify lifecycle verification"
```

If no files changed during verification, do not create an empty commit.

---

## Self-Review Notes

Spec coverage:

- Stable model config is covered by Task 1.
- Strict registry validation is covered by Task 2.
- Hybrid API and `ModelPublisher` are covered by Task 3.
- Script-to-package refactor is covered by Tasks 4 and 5.
- Published candidate package behavior is covered by Tasks 5 and 7.
- Version uniqueness and locking are covered by Task 6.
- Separate deploy DAG is covered by Tasks 8 and 9.
- Manual package revisions are covered by Tasks 10 through 13.
- Prediction comparison is covered by Task 14.
- Docs and runtime contracts are covered by Task 15.
- Verification is covered by Task 16.

Type consistency:

- `ModelBuildConfig` is introduced before all consumers.
- `RatePackageSelector`, `PublishResult`, `DeploymentResult`, `RatePackageSnapshot`,
  `RatePackageRevisionResult`, and `PredictionComparison` are introduced before
  publisher methods use them.
- `ModelPublisher` methods are added only after their backing modules exist.

Scope:

- The plan implements v1 exact selectors. It does not implement temporal
  `as_of` selection, relative versions, UI editing, or embedded Airflow HITL.
