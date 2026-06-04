# Completed Model Build Publish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean Python helper and Airflow TaskFlow wrapper that publish a user-completed model build into SQL without using the all-in-one DAG factory.

**Architecture:** Add `pricing_pipeline/orchestration/publish_completed_build.py` as a thin adapter around existing registry, manifest, publisher, and lineage primitives. User DAGs own ingestion/training/export and pass a small completed-build dictionary to the library publish task. The helper never trains, raw-loads, applies schema, deploys, moves package pointers, or creates model registry rows.

**Tech Stack:** Python dataclasses, SQLAlchemy engine, Airflow TaskFlow API, existing pricing pipeline lifecycle modules, pytest, ruff.

---

## File Structure

- Create `pricing_pipeline/orchestration/publish_completed_build.py`
  - Owns `CompletedModelBuild`, `CompletedModelPublishResult`, `CompletedModelBuildError`, `publish_completed_model_build(...)`, and `publish_completed_model_build_task(...)`.
  - Depends on existing `ModelBuildConfig`, `DatasetSpec`, `ModelExportResult`, `Settings`, `create_dataset_manifest_with_split(...)`, `new_manifest_id(...)`, `build_export_id(...)`, `publish_model_export(...)`, and `validate_model_on_engine(...)`.
- Create `tests/test_publish_completed_build.py`
  - Unit tests for dataclass coercion, defaults, validation, manifest behavior, publisher delegation, and Airflow wrapper behavior.
- Modify `README.md`
  - Document custom DAG composition with the final publish task as the recommended serious workflow.
- Modify `tests/test_readme_contract.py`
  - Add contract text checks for the new helper and the “custom tasks + final publish task” workflow.

---

### Task 1: Completed Build Contracts

**Files:**
- Create: `pricing_pipeline/orchestration/publish_completed_build.py`
- Test: `tests/test_publish_completed_build.py`

- [ ] **Step 1: Write failing tests for completed-build coercion**

Add to `tests/test_publish_completed_build.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from pricing_pipeline.orchestration.publish_completed_build import (
    CompletedModelBuild,
    CompletedModelBuildError,
    CompletedModelPublishResult,
)


def test_completed_model_build_round_trips_plain_dict(tmp_path):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")

    build = CompletedModelBuild(
        rating_workbook_path=str(workbook),
        model_version="20260603",
        effective_from="2026-06-03",
        mlflow_run_id=None,
        metrics={"deviance": 12.5},
    )

    payload = build.to_dict()
    assert payload["rating_workbook_path"] == str(workbook)
    assert payload["metrics"] == {"deviance": 12.5}
    assert CompletedModelBuild.from_mapping(payload) == build
    assert CompletedModelBuild.from_mapping(build) is build


def test_completed_model_build_rejects_unknown_mapping_keys():
    with pytest.raises(CompletedModelBuildError, match="unknown completed build field"):
        CompletedModelBuild.from_mapping(
            {
                "rating_workbook_path": "rating.xlsx",
                "model_version": "20260603",
                "effective_from": "2026-06-03",
                "unexpected": "value",
            }
        )


def test_completed_model_publish_result_to_dict():
    result = CompletedModelPublishResult(
        model_id=17,
        model_key="CLAIM_FREQ",
        model_version="20260603",
        manifest_id="manifest-1",
        split_set_id=None,
        export_id="export-1",
        rate_package_id=42,
        package_version=3,
        package_status="PUBLISHED",
        rating_workbook_path="/tmp/rating.xlsx",
        mlflow_run_id=None,
        was_existing=False,
    )

    assert result.to_dict()["package_status"] == "PUBLISHED"
    assert result.to_dict()["was_existing"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk uv run pytest tests/test_publish_completed_build.py -q
```

Expected: fail with `ModuleNotFoundError` or missing names from `pricing_pipeline.orchestration.publish_completed_build`.

- [ ] **Step 3: Implement minimal contracts**

Create `pricing_pipeline/orchestration/publish_completed_build.py` with:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping


class CompletedModelBuildError(ValueError):
    """Raised when a completed-build payload cannot be published."""


@dataclass(frozen=True)
class CompletedModelBuild:
    rating_workbook_path: str
    model_version: str
    effective_from: str
    created_by: str | None = None
    export_id: str | None = None
    dag_id: str | None = None
    airflow_run_id: str | None = None
    mlflow_run_id: str | None = None
    manifest_id: str | None = None
    split_set_id: str | None = None
    model_artifact_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        value: "CompletedModelBuild | Mapping[str, Any]",
    ) -> "CompletedModelBuild":
        if isinstance(value, cls):
            return value
        data = dict(value)
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise CompletedModelBuildError(
                "unknown completed build field(s): " + ", ".join(unknown)
            )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompletedModelPublishResult:
    model_id: int
    model_key: str
    model_version: str
    manifest_id: str
    split_set_id: str | None
    export_id: str
    rate_package_id: int
    package_version: int
    package_status: str
    rating_workbook_path: str
    mlflow_run_id: str | None = None
    was_existing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
rtk uv run pytest tests/test_publish_completed_build.py -q
```

Expected: pass for the contract tests.

---

### Task 2: Pure Python Publish Helper

**Files:**
- Modify: `pricing_pipeline/orchestration/publish_completed_build.py`
- Test: `tests/test_publish_completed_build.py`

- [ ] **Step 1: Write failing tests for helper defaults and delegation**

Add tests:

```python
from dataclasses import dataclass

from pricing_pipeline.data.manifest import DatasetManifestResult
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.config import ModelBuildConfig, ValidationSplitConfig
from pricing_pipeline.models.spec import DatasetSpec, ModelExportResult
from pricing_pipeline.publishing.lifecycle import PublishResult
from pricing_pipeline.orchestration.publish_completed_build import (
    publish_completed_model_build,
)


def _config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_key="CLAIM_FREQ",
        model_label="Claim frequency",
        target_name="claim_count",
        model_type="superglm_poisson",
        deployment_slot="CLAIM_FREQ_CURRENT",
        default_package_status="PUBLISHED",
        validation_split=ValidationSplitConfig.none(),
    )


def _dataset() -> DatasetSpec:
    return DatasetSpec(
        dataset_name="claim_freq_training",
        source_system="azure_sql",
        manifest_sql="SELECT * FROM work.claim_freq_training",
        pk_columns=("policy_id",),
        target_column="claim_count",
        weight_column="earned_exposure",
    )


def _settings(tmp_path) -> Settings:
    return Settings(
        pricing_database="PricingLab",
        mlflow_tracking_uri="",
        mlflow_enabled=False,
        rating_export_root=tmp_path / "rating_exports",
        validation_split_artifact_root=tmp_path / "validation_splits",
    )


def test_publish_completed_model_build_creates_manifest_and_delegates(
    tmp_path,
    monkeypatch,
):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_model_on_engine",
        lambda engine, config: calls.append(("validate", engine, config)) or 17,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.new_manifest_id",
        lambda dataset_name: f"{dataset_name}_manifest",
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.create_dataset_manifest_with_split",
        lambda engine, **kwargs: calls.append(("manifest", engine, kwargs))
        or DatasetManifestResult(
            manifest_id="claim_freq_training_manifest",
            split_set_id="split-1",
            split_artifact_uri=None,
        ),
    )

    def fake_publish(engine, export, *, model_config):
        calls.append(("publish", engine, export, model_config))
        assert isinstance(export, ModelExportResult)
        return {
            "mlflow_run_id": "mlflow-1",
            "export_id": "export-1",
            "rate_package_id": "42",
            "package_version": "7",
            "rating_workbook_path": str(workbook),
        }

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        fake_publish,
    )

    result = publish_completed_model_build(
        object(),
        settings=_settings(tmp_path),
        model_config=_config(),
        dataset=_dataset(),
        completed_build={
            "rating_workbook_path": str(workbook),
            "model_version": "20260603",
            "effective_from": "2026-06-03",
            "export_id": "export-1",
        },
        created_by="airflow",
    )

    assert result.model_id == 17
    assert result.manifest_id == "claim_freq_training_manifest"
    assert result.split_set_id == "split-1"
    assert result.package_status == "PUBLISHED"
    assert result.mlflow_run_id == "mlflow-1"
    assert calls[0][0] == "validate"
    assert calls[1][0] == "manifest"
    assert calls[2][0] == "publish"
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
rtk uv run pytest tests/test_publish_completed_build.py::test_publish_completed_model_build_creates_manifest_and_delegates -q
```

Expected: fail because `publish_completed_model_build` does not exist yet.

- [ ] **Step 3: Implement minimal helper**

In `pricing_pipeline/orchestration/publish_completed_build.py`, add imports and helper code:

```python
from pathlib import Path

from pricing_pipeline.data.manifest import (
    create_dataset_manifest_with_split,
    new_manifest_id,
)
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import DatasetSpec, ModelExportResult
from pricing_pipeline.orchestration.pipeline import publish_model_export
from pricing_pipeline.publishing.publisher import validate_model_on_engine
from pricing_pipeline.publishing.rating_export import build_export_id


_DEFAULT_PYTHON_DAG_ID = "python_publish_completed_model_build"


def _required_text(value: str | None, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise CompletedModelBuildError(f"{field_name} is required")
    return str(value).strip()


def _existing_workbook(path_value: str) -> str:
    path = Path(_required_text(path_value, "rating_workbook_path"))
    if not path.exists():
        raise CompletedModelBuildError(
            f"rating_workbook_path does not exist: {path.as_posix()}"
        )
    return str(path)
```

Implement `publish_completed_model_build(...)` so it:

- coerces `CompletedModelBuild.from_mapping(...)`;
- validates `rating_workbook_path`, `model_version`, `effective_from`, `created_by`;
- validates model registry with `validate_model_on_engine(...)`;
- derives `dag_id`, `airflow_run_id`, `export_id`;
- creates manifest when no `manifest_id` was supplied;
- builds `ModelExportResult`;
- calls `publish_model_export(...)`;
- returns `CompletedModelPublishResult`.

- [ ] **Step 4: Run targeted helper tests**

Run:

```bash
rtk uv run pytest tests/test_publish_completed_build.py -q
```

Expected: pass for contract and helper delegation tests.

---

### Task 3: Manifest Reuse, Validation, and Idempotency Surface

**Files:**
- Modify: `pricing_pipeline/orchestration/publish_completed_build.py`
- Test: `tests/test_publish_completed_build.py`

- [ ] **Step 1: Write failing tests for manifest reuse and validation**

Add tests:

```python
def test_publish_completed_model_build_reuses_supplied_manifest_without_dataset(
    tmp_path,
    monkeypatch,
):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.validate_model_on_engine",
        lambda engine, config: 17,
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.create_dataset_manifest_with_split",
        lambda *args, **kwargs: calls.append("manifest"),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_model_export",
        lambda engine, export, *, model_config: {
            "mlflow_run_id": "",
            "export_id": export.export_id,
            "rate_package_id": "42",
            "package_version": "7",
            "rating_workbook_path": export.rating_workbook_path,
        },
    )

    result = publish_completed_model_build(
        object(),
        settings=_settings(tmp_path),
        model_config=_config(),
        dataset=None,
        completed_build={
            "rating_workbook_path": str(workbook),
            "model_version": "20260603",
            "effective_from": "2026-06-03",
            "export_id": "export-1",
            "manifest_id": "manifest-existing",
            "split_set_id": None,
            "created_by": "airflow",
        },
    )

    assert result.manifest_id == "manifest-existing"
    assert calls == []


def test_publish_completed_model_build_requires_dataset_without_manifest(tmp_path):
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_text("fake workbook", encoding="utf-8")

    with pytest.raises(CompletedModelBuildError, match="dataset is required"):
        publish_completed_model_build(
            object(),
            settings=_settings(tmp_path),
            model_config=_config(),
            dataset=None,
            completed_build={
                "rating_workbook_path": str(workbook),
                "model_version": "20260603",
                "effective_from": "2026-06-03",
                "export_id": "export-1",
                "created_by": "airflow",
            },
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk uv run pytest tests/test_publish_completed_build.py -q
```

Expected: fail until manifest reuse and dataset-required behavior are implemented.

- [ ] **Step 3: Implement manifest branch behavior**

Update `publish_completed_model_build(...)`:

- if `build.manifest_id` is supplied, reuse it and avoid manifest creation;
- if absent and `dataset is None`, raise `CompletedModelBuildError("dataset is required when manifest_id is not supplied")`;
- otherwise call `create_dataset_manifest_with_split(...)`.

- [ ] **Step 4: Run targeted tests**

Run:

```bash
rtk uv run pytest tests/test_publish_completed_build.py -q
```

Expected: pass.

---

### Task 4: Airflow TaskFlow Wrapper

**Files:**
- Modify: `pricing_pipeline/orchestration/publish_completed_build.py`
- Test: `tests/test_publish_completed_build.py`

- [ ] **Step 1: Write failing tests for task wrapper delegation**

Add test:

```python
def test_publish_completed_model_build_task_fills_airflow_context(
    tmp_path,
    monkeypatch,
):
    calls = []
    config = _config()
    dataset = _dataset()

    @dataclass(frozen=True)
    class FakeDag:
        dag_id: str

    class FakeRuntime:
        settings = _settings(tmp_path)

        def get_engine(self):
            return "engine"

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.runtime_from_dag_config",
        lambda runtime_module=None: FakeRuntime(),
    )
    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.get_current_context",
        lambda: {
            "dag": FakeDag("claim_freq_build"),
            "run_id": "manual__20260603",
        },
    )

    def fake_publish(engine, **kwargs):
        calls.append((engine, kwargs))
        return CompletedModelPublishResult(
            model_id=17,
            model_key="CLAIM_FREQ",
            model_version="20260603",
            manifest_id="manifest-1",
            split_set_id=None,
            export_id="CLAIM_FREQ__manual__20260603",
            rate_package_id=42,
            package_version=7,
            package_status="PUBLISHED",
            rating_workbook_path=kwargs["completed_build"]["rating_workbook_path"],
        )

    monkeypatch.setattr(
        "pricing_pipeline.orchestration.publish_completed_build.publish_completed_model_build",
        fake_publish,
    )

    from pricing_pipeline.orchestration.publish_completed_build import (
        publish_completed_model_build_task,
    )

    task_callable = publish_completed_model_build_task(
        model_config=config,
        dataset=dataset,
        created_by="airflow",
    )
    result = task_callable.function(
        {
            "rating_workbook_path": str(tmp_path / "rating_tables.xlsx"),
            "model_version": "20260603",
            "effective_from": "2026-06-03",
        }
    )

    assert result["model_key"] == "CLAIM_FREQ"
    assert calls[0][0] == "engine"
    completed = calls[0][1]["completed_build"]
    assert completed["dag_id"] == "claim_freq_build"
    assert completed["airflow_run_id"] == "manual__20260603"
    assert completed["created_by"] == "airflow"
```

- [ ] **Step 2: Run wrapper test to verify it fails**

Run:

```bash
rtk uv run pytest tests/test_publish_completed_build.py::test_publish_completed_model_build_task_fills_airflow_context -q
```

Expected: fail because wrapper does not exist yet.

- [ ] **Step 3: Implement TaskFlow wrapper**

Add imports:

```python
from airflow.sdk import get_current_context, task

from pricing_pipeline.orchestration.dag_factory import runtime_from_dag_config
```

Implement:

```python
def publish_completed_model_build_task(
    *,
    model_config: ModelBuildConfig,
    dataset: DatasetSpec,
    runtime_module: str | None = None,
    created_by: str = "airflow",
    task_id: str = "publish_completed_model_build",
):
    @task(task_id=task_id)
    def _publish(completed_build: Mapping[str, Any]) -> dict[str, Any]:
        context = get_current_context()
        dag = context.get("dag")
        payload = CompletedModelBuild.from_mapping(completed_build).to_dict()
        payload.setdefault("dag_id", getattr(dag, "dag_id", None))
        payload.setdefault("airflow_run_id", context.get("run_id"))
        payload.setdefault("created_by", created_by)
        runtime = runtime_from_dag_config(runtime_module)
        result = publish_completed_model_build(
            runtime.get_engine(),
            settings=runtime.settings,
            model_config=model_config,
            dataset=dataset,
            completed_build=payload,
            created_by=created_by,
        )
        return result.to_dict()

    return _publish
```

- [ ] **Step 4: Run wrapper tests**

Run:

```bash
rtk uv run pytest tests/test_publish_completed_build.py -q
```

Expected: pass.

---

### Task 5: Documentation and Contracts

**Files:**
- Modify: `README.md`
- Modify: `tests/test_readme_contract.py`

- [ ] **Step 1: Write failing README contract test**

Add assertions in `tests/test_readme_contract.py`:

```python
def test_readme_documents_completed_build_publish_task():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "publish_completed_model_build_task" in readme
    assert "custom DAG" in readme
    assert "build_pricing_model_dag" in readme
    assert "does not deploy" in readme
```

- [ ] **Step 2: Run README contract test to verify it fails**

Run:

```bash
rtk uv run pytest tests/test_readme_contract.py -q
```

Expected: fail until README documents the new helper.

- [ ] **Step 3: Update README**

In the “Adding Models” section, add a short “Custom DAG publish task” subsection showing:

```python
from pricing_models.claim_freq.spec import DATASET_SPEC, MODEL_CONFIG
from pricing_models.claim_freq.tasks import prepare_training_data, train_and_export_rates
from pricing_pipeline.orchestration.publish_completed_build import (
    publish_completed_model_build_task,
)


@dag(dag_id="claim_freq_build", schedule=None, catchup=False)
def claim_freq_build():
    data = prepare_training_data()
    build = train_and_export_rates(data)

    publish_completed_model_build_task(
        model_config=MODEL_CONFIG,
        dataset=DATASET_SPEC,
    )(build)
```

Include the text:

```text
This final publish task writes manifest/split metadata, model-run lineage, and
rate package rows to SQL. It creates a deployable package candidate but does not
deploy or move any deployment slot pointer.
```

- [ ] **Step 4: Run README tests**

Run:

```bash
rtk uv run pytest tests/test_readme_contract.py -q
```

Expected: pass.

---

### Task 6: Full Verification and Commit

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
rtk uv run pytest tests/test_publish_completed_build.py tests/test_readme_contract.py -q
```

Expected: pass.

- [ ] **Step 2: Run lint**

Run:

```bash
rtk uv run ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 3: Run full test suite**

Run:

```bash
rtk uv run pytest -q
```

Expected: pass.

- [ ] **Step 4: Inspect diff**

Run:

```bash
rtk git diff --stat
rtk git diff -- pricing_pipeline/orchestration/publish_completed_build.py tests/test_publish_completed_build.py README.md tests/test_readme_contract.py
```

Expected: only the planned helper, tests, and README changes are present.

- [ ] **Step 5: Commit and push**

Run:

```bash
rtk git add pricing_pipeline/orchestration/publish_completed_build.py tests/test_publish_completed_build.py README.md tests/test_readme_contract.py docs/superpowers/plans/2026-06-03-completed-model-build-publish.md
rtk git commit -m "feat: publish completed model builds"
rtk git push origin feature/rate-package-lifecycle-api
```

Expected: branch pushed with the completed helper.

---

## Self-Review

Spec coverage:

- Completed-build dataclasses are covered by Task 1.
- Pure helper, defaults, manifest creation/reuse, package status, and lineage delegation are covered by Tasks 2 and 3.
- Airflow TaskFlow wrapper is covered by Task 4.
- README/scaffold guidance is covered by Task 5.
- Verification and push are covered by Task 6.

Placeholder scan:

- No `TBD`, `TODO`, or unspecified implementation steps remain.

Type consistency:

- `CompletedModelBuild`, `CompletedModelPublishResult`, `publish_completed_model_build`, and `publish_completed_model_build_task` use the same signatures across tasks.
