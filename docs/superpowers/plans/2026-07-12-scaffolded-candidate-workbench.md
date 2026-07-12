# Scaffolded Candidate Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the custom scaffold produce scheduled, reopenable SuperGLM candidates that an analyst can edit from one notebook and publish as auditable child packages through explicit Airflow DAGs.

**Architecture:** `scripts/scaffold_pricing_model.py` remains the sole supported authoring path and generates an explicit TaskFlow DAG. A shared standard runner owns CV, fitting, manifests, rating export, candidate bundles, and completed-build metadata; a thin `Workbench` owns candidate lookup, verified loading, the live SuperGLM editor session, and Airflow REST calls. Scheduled training, editor-child publication, and deployment remain separate DAG runs, and `pricing_pipeline/orchestration/dag_factory.py` is not modified or imported by the new path.

**Tech Stack:** Python 3.14, pandas, NumPy, scikit-learn splitters, SuperGLM 0.11 editor API, joblib, Pydantic 2, SQLAlchemy, SQL Server migrations, Airflow 3.2 TaskFlow/API, httpx, openpyxl, pytest.

---

## File map

- `pricing_pipeline/modeling/standard_superglm.py`: model inputs, audited CV, full fit, export, manifest, and completed-build assembly.
- `pricing_pipeline/workbench/artifacts.py`: versioned candidate bundle serialization and integrity verification.
- `pricing_pipeline/workbench/core.py`: friendly history, candidate loading, and live editor lifecycle.
- `pricing_pipeline/workbench/airflow.py`: small Airflow 3 REST client.
- `pricing_pipeline/workbench/submission.py`: immutable editor submission bundles and status/deployment handles.
- `pricing_pipeline/publishing/editor_candidate.py`: verify an editor submission, export the edited model, and publish a child package/run.
- `dags/pricing_publish_editor_candidate.py`: explicit manual editor-publication DAG.
- `scripts/scaffold_pricing_model.py`: generate the analyst-owned hooks and explicit scheduled DAG using the standard runner.
- `db/migrations/V024__candidate_model_artifacts.sql`: candidate artifact lineage and one-run-per-package guard.
- `db/migrations/V025__package_specific_scoring.sql`: score an unpublished package by ID for parity checks.
- `tutorials/scaffolded_candidate_workbench.ipynb`: generic analyst walkthrough.
- Existing `dag_factory.py`: deliberately untouched.

### Task 1: Pin the editor runtime and add no-Docker workbench settings

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `uv.lock`
- Modify: `pricing_pipeline/infra/config.py`
- Modify: `pricing_pipeline/infra/runtime.py`
- Modify: `.env.nodocker.example`
- Test: `tests/test_runtime_contract.py`
- Test: `tests/test_runtime_provider.py`

- [ ] **Step 1: Write failing dependency and settings tests**

```python
def test_workbench_runtime_dependencies_are_direct_and_editor_capable():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    pin = (
        "superglm[editor] @ git+https://github.com/StrudelDoodleS/superglm.git@"
        "1072f7792cf255899fa6ba93579efd49a25ccdb4"
    )
    assert pin in pyproject
    assert pin in requirements
    assert '"joblib"' in pyproject
    assert '"httpx"' in pyproject


def test_settings_load_workbench_and_airflow_api_values(tmp_path):
    settings = Settings.from_env(
        {
            "WORKBENCH_ARTIFACT_ROOT": str(tmp_path / "candidates"),
            "AIRFLOW_API_URL": "http://127.0.0.1:8080/api/v2",
            "AIRFLOW_API_TOKEN": "unit-token",
        }
    )
    assert settings.workbench_artifact_root == tmp_path / "candidates"
    assert settings.airflow_api_url == "http://127.0.0.1:8080/api/v2"
    assert settings.airflow_api_token == "unit-token"
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `rtk proxy .venv/bin/pytest tests/test_runtime_contract.py tests/test_runtime_provider.py -q`

Expected: FAIL because the editor extra and workbench settings are absent.

- [ ] **Step 3: Add dependencies and settings**

Add these `Settings` fields and environment mappings:

```python
workbench_artifact_root: Path = Path("state/workbench_artifacts")
airflow_api_url: str = "http://127.0.0.1:8080/api/v2"
airflow_api_token: str | None = None
```

Teach `runtime_from_module()` to accept the same three keys, document their no-Docker values in `.env.nodocker.example`, replace the SuperGLM pin with the editor extra at commit `1072f7792cf255899fa6ba93579efd49a25ccdb4`, add direct `joblib` and `httpx` dependencies, and run `rtk uv lock` followed by `rtk uv sync --frozen`.

- [ ] **Step 4: Run the focused tests**

Run: `rtk proxy .venv/bin/pytest tests/test_runtime_contract.py tests/test_runtime_provider.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add pyproject.toml requirements.txt uv.lock pricing_pipeline/infra/config.py pricing_pipeline/infra/runtime.py .env.nodocker.example tests/test_runtime_contract.py tests/test_runtime_provider.py
rtk git commit -m "feat: add candidate workbench runtime settings"
```

### Task 2: Add a versioned, verified candidate bundle

**Files:**
- Create: `pricing_pipeline/workbench/__init__.py`
- Create: `pricing_pipeline/workbench/artifacts.py`
- Test: `tests/test_candidate_artifacts.py`

- [ ] **Step 1: Write failing round-trip and integrity tests**

```python
def test_candidate_bundle_round_trip_verifies_hash_and_lineage(tmp_path):
    bundle = CandidateBundle(
        fitted_model={"coef": [0.1]},
        X=pd.DataFrame({"age": [20.0, 30.0]}),
        y=np.array([0.0, 1.0]),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={"scope": "cv", "pooled_scores": {"deviance": 0.4}},
        manifest_id="manifest-1",
        split_set_id="split-1",
        pk_columns=("policy_id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "NONE"},
    )
    metadata = save_candidate_bundle(bundle, tmp_path / "candidate_bundle.joblib")
    loaded = load_candidate_bundle(
        metadata.path,
        expected_sha256=metadata.sha256,
        expected_size_bytes=metadata.size_bytes,
        allowed_root=tmp_path,
    )
    assert loaded.manifest_id == "manifest-1"
    assert loaded.X.equals(bundle.X)


def test_candidate_bundle_rejects_tampering(tmp_path):
    path = tmp_path / "candidate_bundle.joblib"
    metadata = save_candidate_bundle(minimal_bundle(), path)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(CandidateArtifactError, match="SHA-256"):
        load_candidate_bundle(
            path,
            expected_sha256=metadata.sha256,
            expected_size_bytes=path.stat().st_size,
            allowed_root=tmp_path,
        )
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `rtk proxy .venv/bin/pytest tests/test_candidate_artifacts.py -q`

Expected: FAIL because `pricing_pipeline.workbench.artifacts` does not exist.

- [ ] **Step 3: Implement the bundle contract and atomic persistence**

Define these public types:

```python
BUNDLE_FORMAT = "superglm-candidate-joblib-v1"


@dataclass(frozen=True)
class CandidateBundle:
    fitted_model: Any
    X: pd.DataFrame
    y: np.ndarray
    sample_weight: pd.Series | np.ndarray | None
    offset: pd.Series | np.ndarray | None
    export_weight: pd.Series | np.ndarray | None
    cv_report: dict[str, Any]
    manifest_id: str
    split_set_id: str | None
    pk_columns: tuple[str, ...]
    row_order_sha256: str
    model_source_sha256: str
    offset_contract: dict[str, Any]
    review_artifact: dict[str, Any] | None = None


@dataclass(frozen=True)
class CandidateArtifactMetadata:
    path: str
    sha256: str
    format: str
    size_bytes: int
    python_version: str
    superglm_version: str
```

`save_candidate_bundle()` must write a temporary sibling, atomically replace the target, hash the completed bytes, and return metadata. `load_candidate_bundle()` must reject paths outside `allowed_root`, verify format/size/hash before `joblib.load()`, and fail on incompatible Python major/minor or SuperGLM versions.

- [ ] **Step 4: Run the artifact tests**

Run: `rtk proxy .venv/bin/pytest tests/test_candidate_artifacts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/workbench/__init__.py pricing_pipeline/workbench/artifacts.py tests/test_candidate_artifacts.py
rtk git commit -m "feat: add verified candidate bundles"
```

### Task 3: Implement audited SuperGLM CV primitives

**Files:**
- Create: `pricing_pipeline/modeling/__init__.py`
- Create: `pricing_pipeline/modeling/standard_superglm.py`
- Test: `tests/test_standard_superglm.py`

- [ ] **Step 1: Write failing tests for inputs, folds, OOF coverage, and JSON evidence**

```python
def test_precomputed_splitter_replays_exact_folds():
    folds = [(np.array([0, 1]), np.array([2])), (np.array([1, 2]), np.array([0]))]
    splitter = PrecomputedSplitter(folds, row_count=3)
    replayed = list(splitter.split(pd.DataFrame(index=range(3))))
    assert [pair[1].tolist() for pair in replayed] == [[2], [0]]


def test_precomputed_splitter_rejects_duplicate_test_membership():
    folds = [(np.array([0]), np.array([1])), (np.array([2]), np.array([1]))]
    with pytest.raises(StandardSuperGLMError, match="duplicate test-row"):
        PrecomputedSplitter(folds, row_count=3)


def test_cv_report_adapter_returns_json_primitives():
    report, metrics, fold_metrics = cv_result_to_records(fake_cv_result())
    json.dumps(report)
    assert report["scope"] == "cv"
    assert metrics["cv_pooled_deviance"] == pytest.approx(0.42)
    assert fold_metrics[0].metric_name == "deviance"
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `rtk proxy .venv/bin/pytest tests/test_standard_superglm.py -q`

Expected: FAIL because the standard modeling module does not exist.

- [ ] **Step 3: Implement `ModelInputs`, `PrecomputedSplitter`, and CV adaptation**

```python
@dataclass(frozen=True)
class ModelInputs:
    X: pd.DataFrame
    y: np.ndarray
    sample_weight: pd.Series | np.ndarray | None = None
    sample_weight_name: str | None = None
    offset: pd.Series | np.ndarray | None = None
    export_weight: pd.Series | np.ndarray | None = None
    export_weight_name: str | None = None


@dataclass(frozen=True)
class FoldMetric:
    fold_no: int
    metric_name: str
    metric_value: float
```

`run_cross_validation()` must call SuperGLM with `return_estimators=True`, `return_oof=True`, and `error_score="raise"`; reject non-converged estimators and duplicate test membership; record partial OOF coverage; discard fold estimators after extracting convergence; and convert DataFrames, dataclasses, NumPy scalars, and arrays to JSON primitives. Stable run metrics use `cv_pooled_<name>`, `cv_mean_<name>`, and `cv_std_<name>` names.

- [ ] **Step 4: Run the focused tests**

Run: `rtk proxy .venv/bin/pytest tests/test_standard_superglm.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/modeling/__init__.py pricing_pipeline/modeling/standard_superglm.py tests/test_standard_superglm.py
rtk git commit -m "feat: add audited SuperGLM cross validation"
```

### Task 4: Build the shared full-fit, manifest, export, and bundle runner

**Files:**
- Modify: `pricing_pipeline/modeling/standard_superglm.py`
- Modify: `pricing_pipeline/orchestration/completed_build_helpers.py`
- Modify: `pricing_pipeline/orchestration/publish_completed_build.py`
- Modify: `pricing_pipeline/publishing/superglm_metadata.py`
- Test: `tests/test_standard_superglm.py`
- Test: `tests/test_completed_build_helpers.py`
- Test: `tests/test_publish_completed_build.py`
- Test: `tests/test_superglm_metadata.py`

- [ ] **Step 1: Write a failing standard-runner integration test**

```python
def test_standard_runner_uses_cv_folds_for_manifest_and_returns_candidate_metadata(
    tmp_path, monkeypatch
):
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3, 4],
            "target": [0.0, 1.0, 0.0, 1.0],
            "age": [20.0, 30.0, 40.0, 50.0],
        }
    )
    captured = {}
    monkeypatch.setattr(
        standard_superglm,
        "create_model_frame_manifest_with_split",
        lambda engine, **kwargs: captured.update(kwargs)
        or SimpleNamespace(manifest_id="manifest-1", split_set_id="split-1"),
    )
    result = run_standard_superglm_build(
        object(),
        frame=frame,
        inputs=ModelInputs(X=frame[["age"]], y=frame["target"].to_numpy()),
        model_factory=FakeSuperGLM,
        split_indices=kfold_indices(len(frame)),
        fit_mode="fit",
        scoring=("deviance",),
        output_dir=tmp_path,
        model_name="TEST_FREQ",
        model_version="v1",
        export_id="export-1",
        effective_from="2026-07-12",
        manifest_spec=ModelFrameManifestSpec(
            dataset_name="test_frame",
            source_system="pytest",
            data_as_of_date="2026-06-30",
            pk_columns=("policy_id",),
            target_column="target",
        ),
        validation_split=ValidationSplitConfig.custom(materialize=True),
        split_artifact_root=tmp_path / "splits",
        model_source_root=tmp_path,
        created_by="pytest",
    )
    assert captured["split_indices"] == result.fold_indices
    assert Path(result.completed_build["candidate_artifact_path"]).exists()
    assert result.completed_build["manifest_id"] == "manifest-1"
```

- [ ] **Step 2: Run the integration test and confirm failure**

Run: `rtk proxy .venv/bin/pytest tests/test_standard_superglm.py::test_standard_runner_uses_cv_folds_for_manifest_and_returns_candidate_metadata -q`

Expected: FAIL because `run_standard_superglm_build()` is absent.

- [ ] **Step 3: Implement `run_standard_superglm_build()`**

The function must perform this exact order:

```python
evidence = run_cross_validation(
    model_factory(),
    inputs,
    split_indices=split_indices,
    fit_mode=fit_mode,
    scoring=scoring,
)
fitted = fit_full_model(model_factory(), inputs, fit_mode=fit_mode)
workbook_path, receipt_path, receipt_sha256 = export_fitted_model(
    fitted,
    inputs,
    output_dir=output_dir,
    offset_contract=offset_contract,
)
manifest = create_model_frame_manifest_with_split(
    engine,
    frame=frame,
    spec=manifest_spec,
    validation_split=validation_split,
    validation_split_artifact_root=split_artifact_root,
    split_indices=evidence.fold_indices,
    created_by=created_by,
)
artifact_metadata = save_candidate_bundle(
    build_candidate_bundle(fitted, inputs, evidence, manifest, frame, model_source_root),
    Path(output_dir) / "candidate_bundle.joblib",
)
```

Use `OffsetExportContract(handling="NONE")` automatically when `inputs.offset is None`. For a present offset, require an explicit contract and exporter options. Hash normalized relative paths plus bytes for sorted model-local `.py`, `.sql`, and `.toml` files. Return `StandardBuildResult(completed_build, fold_indices, cv_report)` and extend `completed_model_build_payload()` with the candidate metadata fields without asking callers to enter them.

Extend `CompletedModelBuild` in the same task with the seven candidate artifact fields, `metric_scopes: dict[str, str]`, and validated fold-metric dictionaries. Candidate artifact fields are either all absent or all present, SHA values are lowercase 64-character hex, sizes are positive integers, metric scopes cover every emitted metric, and every fold metric contains `fold_no`, `metric_name`, and finite `metric_value`.

Extend `build_superglm_publication_receipt()` with explicit optional fit/export weight names and write `fit_sample_weight_used`, `fit_sample_weight_name`, `export_weight_used`, `export_weight_name`, and existing `fit_used_offset` values into package metadata. Infer names from pandas Series; require the corresponding `ModelInputs` name for a bare non-null array. Record false/null for absent weights rather than `UNKNOWN`.

- [ ] **Step 4: Run the focused runner/helper tests**

Run: `rtk proxy .venv/bin/pytest tests/test_standard_superglm.py tests/test_completed_build_helpers.py tests/test_publish_completed_build.py tests/test_superglm_metadata.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/modeling/standard_superglm.py pricing_pipeline/orchestration/completed_build_helpers.py pricing_pipeline/orchestration/publish_completed_build.py pricing_pipeline/publishing/superglm_metadata.py tests/test_standard_superglm.py tests/test_completed_build_helpers.py tests/test_publish_completed_build.py tests/test_superglm_metadata.py
rtk git commit -m "feat: add standard scaffolded SuperGLM runner"
```

### Task 5: Persist candidate metadata and CV metrics in model-run lineage

**Files:**
- Create: `db/migrations/V024__candidate_model_artifacts.sql`
- Modify: `pricing_pipeline/models/spec.py`
- Modify: `pricing_pipeline/orchestration/publish_completed_build.py`
- Modify: `pricing_pipeline/orchestration/pipeline.py`
- Modify: `pricing_pipeline/publishing/lineage.py`
- Modify: `db/offline_sqlite/pricing.sql`
- Modify: `db/offline_sqlite/mlops.sql`
- Test: `tests/test_migrations.py`
- Test: `tests/test_publish_completed_build.py`
- Test: `tests/test_model_layout.py`

- [ ] **Step 1: Write failing schema and payload propagation tests**

```python
def test_candidate_artifact_migration_extends_model_run_and_guards_package_identity():
    sql = Path("db/migrations/V024__candidate_model_artifacts.sql").read_text(
        encoding="utf-8"
    )
    for column in (
        "candidate_artifact_path",
        "candidate_artifact_sha256",
        "candidate_artifact_format",
        "candidate_artifact_size_bytes",
        "candidate_python_version",
        "candidate_superglm_version",
        "model_source_sha256",
    ):
        assert column in sql
    assert "UX_MODEL_RUN_RATE_PACKAGE" in sql
    assert "WHERE rate_package_id IS NOT NULL" in sql


def test_model_export_carries_candidate_metadata_and_scoped_metrics():
    export = ModelExportResult(
        model_id=17,
        model_name="HOME_FREQ",
        model_version="v1",
        model_type="superglm_poisson",
        target_name="claim_count",
        deployment_slot="HOME_FREQ_UAT",
        manifest_id="manifest-1",
        dag_id="pricing_home_freq",
        airflow_run_id="scheduled__20260712",
        mlflow_run_id="",
        split_set_id="split-1",
        export_id="export-1",
        rating_workbook_path="rating.xlsx",
        effective_from="2026-07-12",
        created_by="airflow",
        candidate_artifact_path="candidate.joblib",
        candidate_artifact_sha256="a" * 64,
        candidate_artifact_format="superglm-candidate-joblib-v1",
        candidate_artifact_size_bytes=123,
        candidate_python_version="3.14",
        candidate_superglm_version="0.11.0",
        model_source_sha256="b" * 64,
        metrics={"cv_pooled_deviance": 0.42},
        metric_scopes={"cv_pooled_deviance": "cv"},
        fold_metrics=(
            {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.40},
        ),
    )
    assert export.candidate_artifact_size_bytes == 123
    assert export.metric_scopes["cv_pooled_deviance"] == "cv"
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `rtk proxy .venv/bin/pytest tests/test_migrations.py tests/test_publish_completed_build.py tests/test_model_layout.py -q`

Expected: FAIL because the migration and `ModelExportResult` propagation do not exist.

- [ ] **Step 3: Add lineage fields and idempotent metric upserts**

Add the seven nullable artifact columns to `pricing.MODEL_RUN`, SHA/size checks, a duplicate-package preflight using `THROW`, and the filtered unique index. Mirror the columns in offline SQLite.

Extend `ModelExportResult` with:

```python
candidate_artifact_path: str | None = None
candidate_artifact_sha256: str | None = None
candidate_artifact_format: str | None = None
candidate_artifact_size_bytes: int | None = None
candidate_python_version: str | None = None
candidate_superglm_version: str | None = None
model_source_sha256: str | None = None
metrics: dict[str, float] = field(default_factory=dict)
metric_scopes: dict[str, str] = field(default_factory=dict)
fold_metrics: tuple[dict[str, int | str | float], ...] = ()
```

Carry the already-validated completed-build fields through `publish_completed_model_build()` into `ModelExportResult`, then through `publish_model_export()` into `record_model_run()`. Inside the existing lineage transaction, MERGE run metrics into `mlops.MODEL_RUN_METRIC` and fold metrics into `pricing.CV_FOLD_METRIC`; retries update the same names instead of inserting duplicates.

- [ ] **Step 4: Run focused lineage tests**

Run: `rtk proxy .venv/bin/pytest tests/test_migrations.py tests/test_publish_completed_build.py tests/test_model_layout.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add db/migrations/V024__candidate_model_artifacts.sql db/offline_sqlite/pricing.sql db/offline_sqlite/mlops.sql pricing_pipeline/models/spec.py pricing_pipeline/orchestration/publish_completed_build.py pricing_pipeline/orchestration/pipeline.py pricing_pipeline/publishing/lineage.py tests/test_migrations.py tests/test_publish_completed_build.py tests/test_model_layout.py
rtk git commit -m "feat: persist candidate artifacts and CV metrics"
```

### Task 6: Make the custom scaffold generate the complete standard recipe

**Files:**
- Modify: `scripts/scaffold_pricing_model.py`
- Modify: `tests/test_scaffold_pricing_model.py`
- Create: `tests/test_scaffolded_standard_model.py`

- [ ] **Step 1: Replace the old scaffold assertions with failing standard-runner assertions**

```python
def test_custom_scaffold_uses_standard_runner_and_never_dag_factory(tmp_path):
    scaffold_pricing_model(
        ScaffoldOptions(
            model_name="HOME_FREQ",
            model_label="Home frequency",
            target_name="claim_count",
            root=tmp_path,
        )
    )
    modeling = (tmp_path / "pricing_models/home_freq/modeling.py").read_text(
        encoding="utf-8"
    )
    dag = (tmp_path / "dags/pricing_home_freq.py").read_text(encoding="utf-8")
    assert "ModelInputs" in modeling
    assert "def build_training_inputs" in modeling
    assert "def build_model" in modeling
    assert "def validation_splitter" in modeling
    assert "run_standard_superglm_build" in modeling
    assert "fit_validate_export_rating_tables" not in modeling
    assert "build_pricing_model_dag" not in dag
    assert "orchestration.dag_factory" not in dag
```

Add a temporary-package test that imports generated `modeling.py`, replaces only `read_prepared_source`, `build_final_model_frame`, `build_training_inputs`, `build_model`, and `validation_splitter` with fixture functions, and proves `train_validate_export_model()` returns a completed build containing a real candidate bundle.

- [ ] **Step 2: Run scaffold tests and confirm failure**

Run: `rtk proxy .venv/bin/pytest tests/test_scaffold_pricing_model.py tests/test_scaffolded_standard_model.py -q`

Expected: FAIL because the generated scaffold still contains the unimplemented fit/export loop.

- [ ] **Step 3: Rewrite only the custom scaffold template**

Generate these analyst-owned hooks:

```python
FIT_MODE = "fit_reml"
CV_SCORING = ("deviance",)


def read_prepared_source(prepared: Mapping[str, Any]) -> pd.DataFrame:
    raise NotImplementedError("Return the prepared source DataFrame")


def build_final_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.copy()


def build_training_inputs(frame: pd.DataFrame) -> ModelInputs:
    raise NotImplementedError("Select X, y, optional weights, and optional offset")


def build_model():
    raise NotImplementedError("Configure and return a SuperGLM model")


def validation_splitter(frame: pd.DataFrame):
    return validation_split_indices(frame, MODEL_CONFIG.validation_split)


def write_review_workbook(*, fitted_model, inputs, output_path):
    return None
```

The generated `train_validate_export_model()` sorts by declared PKs, resolves version/effective/as-of values, converts the splitter output to positional pairs, and delegates all lifecycle work to `run_standard_superglm_build()`. Keep the explicit generated TaskFlow DAG. Do not edit `_factory_*` templates or `pricing_pipeline/orchestration/dag_factory.py`.

- [ ] **Step 4: Run scaffold tests**

Run: `rtk proxy .venv/bin/pytest tests/test_scaffold_pricing_model.py tests/test_scaffolded_standard_model.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add scripts/scaffold_pricing_model.py tests/test_scaffold_pricing_model.py tests/test_scaffolded_standard_model.py
rtk git commit -m "feat: generate complete scaffolded model builds"
```

### Task 7: Add friendly candidate history and verified loading

**Files:**
- Create: `pricing_pipeline/workbench/core.py`
- Modify: `pricing_pipeline/workbench/__init__.py`
- Test: `tests/test_candidate_workbench.py`

- [ ] **Step 1: Write failing history and loading tests**

```python
def test_candidates_returns_friendly_columns_and_hides_lineage_ids(monkeypatch):
    workbench = Workbench(engine=object(), settings=test_settings())
    monkeypatch.setattr(workbench, "_candidate_rows", lambda model_name, slot: candidate_rows())
    history = workbench.candidates("HOME_FREQ")
    assert list(history.columns) == [
        "Package",
        "Fitted",
        "Data through",
        "Parent",
        "State",
        "Baseline pooled CV deviance",
        "Editor train delta",
        "Editor",
    ]
    assert "model_run_id" not in history.columns
    assert history.iloc[0]["State"] == "Champion in HOME_FREQ_UAT"


def test_open_resolves_one_successful_run_and_verifies_bundle(tmp_path, monkeypatch):
    metadata = save_candidate_bundle(minimal_bundle(), tmp_path / "candidate.joblib")
    workbench = Workbench(engine=object(), settings=test_settings(tmp_path))
    monkeypatch.setattr(
        workbench,
        "_resolve_candidate",
        lambda model_name, package_version: candidate_record(metadata),
    )
    candidate = workbench.open("HOME_FREQ", package_version=7)
    assert candidate.package_version == 7
    assert candidate.bundle.manifest_id == "manifest-1"


def test_open_rejects_ambiguous_run_lineage(monkeypatch):
    workbench = Workbench(engine=object(), settings=test_settings())
    monkeypatch.setattr(workbench, "_resolve_candidate_rows", lambda model, version: [{}, {}])
    with pytest.raises(CandidateLineageError, match="exactly one successful MODEL_RUN"):
        workbench.open("HOME_FREQ", package_version=7)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `rtk proxy .venv/bin/pytest tests/test_candidate_workbench.py -q`

Expected: FAIL because `Workbench` and `Candidate` do not exist.

- [ ] **Step 3: Implement `Workbench.from_runtime()`, history, and loading**

`Workbench.from_runtime(runtime_module=None)` must call `runtime_from_env_or_module()`, retain its configured engine/settings, and never require notebook cells to construct SQLAlchemy or Airflow clients. The history query joins model, package, model run, dataset manifest, run metrics, parent package, and current deployment. Default history omits IDs/hashes; `technical=True` returns them. `open()` selects by model name plus integer package version, requires exactly one successful artifact-backed model run, validates the artifact metadata, and returns a `Candidate` carrying the bundle and parent/run identity.

- [ ] **Step 4: Run candidate tests**

Run: `rtk proxy .venv/bin/pytest tests/test_candidate_workbench.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/workbench/core.py pricing_pipeline/workbench/__init__.py tests/test_candidate_workbench.py
rtk git commit -m "feat: add candidate history and verified loading"
```

### Task 8: Retain the live SuperGLM editor and create immutable submissions

**Files:**
- Create: `pricing_pipeline/workbench/airflow.py`
- Create: `pricing_pipeline/workbench/submission.py`
- Modify: `pricing_pipeline/workbench/core.py`
- Modify: `pricing_pipeline/workbench/__init__.py`
- Test: `tests/test_candidate_editor.py`
- Test: `tests/test_airflow_api_client.py`

- [ ] **Step 1: Write failing editor/session and HTTP tests**

```python
def test_candidate_retains_live_editor_session_until_submission(tmp_path):
    session = FakeEditorSession()
    candidate = candidate_with_bundle(tmp_path, session_factory=lambda **kwargs: session)
    widget = candidate.editor()
    assert widget is session.widget_value
    assert candidate.editor_session is session
    submission = candidate.submit_edits(reason="Sparse-age market calibration")
    assert session.saved_json_path.exists()
    assert submission.parent_rate_package_id == candidate.rate_package_id
    assert Path(submission.edited_model_path).exists()


def test_airflow_client_triggers_editor_dag_with_submission_path():
    def handler(request):
        assert request.method == "POST"
        assert request.url.path.endswith(
            "/dags/pricing_publish_editor_candidate/dagRuns"
        )
        return httpx.Response(
            200,
            json={"dag_run_id": "manual__submission-1", "state": "queued"},
        )

    client = AirflowClient(
        "http://127.0.0.1:8080/api/v2",
        token="token",
        transport=httpx.MockTransport(handler),
    )
    result = client.trigger_dag(
        "pricing_publish_editor_candidate",
        run_id="manual__submission-1",
        conf={"submission_path": "state/submission.json", "submission_sha256": "a" * 64},
    )
    assert result.state == "queued"
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `rtk proxy .venv/bin/pytest tests/test_candidate_editor.py tests/test_airflow_api_client.py -q`

Expected: FAIL because editor/session/submission and Airflow client classes are absent.

- [ ] **Step 3: Implement the live-session and submission boundary**

`Candidate.editor()` must create and retain exactly one session:

```python
self.editor_session = EditorSession.from_model(
    self.bundle.fitted_model,
    train_data=(
        self.bundle.X,
        self.bundle.y,
        self.bundle.sample_weight,
        self.bundle.offset,
    ),
    cv_report=self.bundle.cv_report,
)
self.editor_widget = self.editor_session.widget()
return self.editor_widget
```

`submit_edits()` must require a non-empty reason and live session, call `session.save(json_path)`, call `session.to_model(X=X, y=y, sample_weight=sample_weight, offset=offset)`, atomically joblib-serialize/hash the edited model, create canonical `EditorSubmission` JSON, derive `submission_id` from parent package plus session/model hashes, and trigger the explicit editor DAG. `close_editor()` must stop the widget server when supported. `AirflowClient` uses bearer auth, explicit timeouts, `raise_for_status()`, and Airflow 3 `/api/v2` endpoints for trigger/status.

- [ ] **Step 4: Run editor/client tests**

Run: `rtk proxy .venv/bin/pytest tests/test_candidate_editor.py tests/test_airflow_api_client.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/workbench/airflow.py pricing_pipeline/workbench/submission.py pricing_pipeline/workbench/core.py pricing_pipeline/workbench/__init__.py tests/test_candidate_editor.py tests/test_airflow_api_client.py
rtk git commit -m "feat: add notebook editor submissions"
```

### Task 9: Publish edited workbooks as idempotent child packages

**Files:**
- Create: `pricing_pipeline/publishing/editor_candidate.py`
- Modify: `pricing_pipeline/publishing/package_writer.py`
- Modify: `pricing_pipeline/publishing/publisher.py`
- Modify: `pricing_pipeline/publishing/lineage.py`
- Create: `db/migrations/V025__package_specific_scoring.sql`
- Test: `tests/test_editor_candidate_publisher.py`
- Test: `tests/test_package_writer.py`
- Test: `tests/test_migrations.py`

- [ ] **Step 1: Write failing child-publication and package-scorer tests**

```python
def test_editor_publisher_creates_child_and_derived_run(monkeypatch, tmp_path):
    submission = valid_editor_submission(tmp_path)
    calls = []
    monkeypatch.setattr(editor_candidate, "load_verified_submission", lambda path, digest: submission)
    monkeypatch.setattr(editor_candidate, "load_parent_candidate", lambda engine, submission: parent_candidate())
    monkeypatch.setattr(editor_candidate, "export_edited_model", lambda parent, submission: edited_export(tmp_path))
    monkeypatch.setattr(
        editor_candidate,
        "publish_rating_package",
        lambda engine, **kwargs: calls.append(("publish", kwargs)) or child_publish_result(),
    )
    monkeypatch.setattr(
        editor_candidate,
        "record_derived_model_run",
        lambda engine, **kwargs: calls.append(("lineage", kwargs)) or 91,
    )
    result = publish_editor_submission(
        object(),
        settings=test_settings(tmp_path),
        submission_path=submission.path,
        submission_sha256=submission.sha256,
        dag_id="pricing_publish_editor_candidate",
        airflow_run_id="manual__submission-1",
        created_by="analyst@example.test",
    )
    assert result.parent_rate_package_id == submission.parent_rate_package_id
    assert calls[0][1]["parent_rate_package_id"] == submission.parent_rate_package_id
    assert calls[1][1]["rate_package_id"] == result.rate_package_id


def test_package_specific_scorer_does_not_resolve_live_pointer():
    sql = Path("db/migrations/V025__package_specific_scoring.sql").read_text(
        encoding="utf-8"
    )
    assert "CREATE OR ALTER PROCEDURE pricing.PREDICT_RATE_PACKAGE" in sql
    assert "@rate_package_id BIGINT" in sql
    assert "PRICING_PACKAGE_POINTER" not in sql
    assert "V_CURRENT_RATE_PACKAGE" not in sql
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `rtk proxy .venv/bin/pytest tests/test_editor_candidate_publisher.py tests/test_package_writer.py tests/test_migrations.py -q`

Expected: FAIL because editor publication and package-specific scoring are absent.

- [ ] **Step 3: Extend the existing package writer and implement editor publication**

Add optional arguments to the existing writer, defaulting to the training behavior:

```python
def publish_rating_package(
    engine,
    *,
    export_id: str,
    created_by: str = "python",
    package_status: str = "PUBLISHED",
    parent_rate_package_id: int | None = None,
    revision_metadata_json: str | None = None,
) -> PublishResult:
```

Inside the existing lock/transaction, a non-null parent must exist, belong to the staged model, have `PUBLISHED` status, and have the same trained model version/effective dates. Insert the parent and canonical revision JSON instead of hardcoded nulls. Existing-export idempotency must also compare parent ID and revision metadata.

`publish_editor_submission()` must verify submission/session/model/baseline hashes, load the authoritative edited model plus parent bundle, export with the parent's inputs/weights/offset contract, rebuild and hash the receipt, stage it, publish a child carrying the parent's trained model version/effective dates, save a complete edited candidate bundle, record a derived model run with inherited manifest/split links, and persist stable `editor_training_parent_*` comparison metrics. When the configured champion exists, compare the same prepared rows/weights/offset using its verified artifact or the package-specific scorer and persist `editor_training_champion_*`; otherwise store a clear unavailable reason in revision metadata. Retries use a deterministic export ID from `submission_id` and accept an existing child only when every compatibility hash/date/version matches.

Create `pricing.PREDICT_RATE_PACKAGE` by extracting the current scorer's matching logic to take `@rate_package_id`, prepared-feature JSON, exposure, and breakdown flag. It must query the selected package's compiled cells/bands directly and work for DRAFT or PUBLISHED packages. Run a deterministic bounded PK sample through Python and this procedure before final child status; mismatch rolls back publication.

- [ ] **Step 4: Run editor publisher/package tests**

Run: `rtk proxy .venv/bin/pytest tests/test_editor_candidate_publisher.py tests/test_package_writer.py tests/test_migrations.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/publishing/editor_candidate.py pricing_pipeline/publishing/package_writer.py pricing_pipeline/publishing/publisher.py pricing_pipeline/publishing/lineage.py db/migrations/V025__package_specific_scoring.sql tests/test_editor_candidate_publisher.py tests/test_package_writer.py tests/test_migrations.py
rtk git commit -m "feat: publish editor-derived candidate packages"
```

### Task 10: Add the explicit editor-publication DAG and deployment handoff

**Files:**
- Create: `dags/pricing_publish_editor_candidate.py`
- Modify: `pricing_pipeline/workbench/submission.py`
- Test: `tests/test_editor_candidate_dag.py`
- Test: `tests/test_candidate_editor.py`

- [ ] **Step 1: Write failing DAG and deployment-handoff tests**

```python
def test_editor_candidate_dag_is_explicit_and_manual():
    source = Path("dags/pricing_publish_editor_candidate.py").read_text(encoding="utf-8")
    assert 'dag_id="pricing_publish_editor_candidate"' in source
    assert "schedule=None" in source
    assert "publish_editor_submission" in source
    assert "build_pricing_model_dag" not in source
    assert "dag_factory" not in source


def test_submission_requests_existing_deploy_dag(airflow_client, published_submission):
    published_submission.request_deployment(
        reason="Approved market calibration",
        deployment_slot="HOME_FREQ_UAT",
    )
    assert airflow_client.triggered[-1].dag_id == "pricing_deploy_rate_package"
    assert airflow_client.triggered[-1].conf == {
        "model_name": "HOME_FREQ",
        "package_version": 8,
        "deployment_slot": "HOME_FREQ_UAT",
        "deployment_reason": "Approved market calibration",
    }
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `rtk proxy .venv/bin/pytest tests/test_editor_candidate_dag.py tests/test_candidate_editor.py -q`

Expected: FAIL because the DAG and deployment handle do not exist.

- [ ] **Step 3: Implement the explicit TaskFlow DAG and friendly handle**

The DAG must read `submission_path` and `submission_sha256` from `dag_run.conf`, obtain runtime/engine through `runtime_from_env_or_module()`, obtain `dag_id`/`run_id` from Airflow context, call `publish_editor_submission()`, and return only small publication metadata. It must never wait for a human and must not import the legacy DAG factory.

`Submission.status()` polls the Airflow run and, when successful, exposes model name, integer child package version, state, and Airflow link. `request_deployment()` requires published state and a non-empty reason, then triggers the existing `pricing_deploy_rate_package` DAG with resolved friendly values.

- [ ] **Step 4: Run DAG/submission tests**

Run: `rtk proxy .venv/bin/pytest tests/test_editor_candidate_dag.py tests/test_candidate_editor.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add dags/pricing_publish_editor_candidate.py pricing_pipeline/workbench/submission.py tests/test_editor_candidate_dag.py tests/test_candidate_editor.py
rtk git commit -m "feat: add editor candidate publication DAG"
```

### Task 11: Demonstrate the raw-axis review artifact without changing SQL scoring

**Files:**
- Modify: `pricing_pipeline/modeling/standard_superglm.py`
- Modify: `scripts/scaffold_pricing_model.py`
- Test: `tests/test_standard_superglm.py`
- Test: `tests/test_scaffolded_standard_model.py`

- [ ] **Step 1: Write a failing model-local review-hook test**

```python
def test_model_local_log_density_review_is_separate_from_canonical_export(tmp_path):
    canonical = pd.DataFrame(
        {
            "log_lower": [0.0, 1.0],
            "log_upper": [1.0, 2.0],
            "log_representative": [0.5, 1.5],
            "relativity": [0.8, 1.2],
        }
    )

    def write_review_workbook(*, fitted_model, inputs, output_path):
        review = canonical.assign(
            density_lower=np.exp(canonical["log_lower"]),
            density_upper=np.exp(canonical["log_upper"]),
            density_representative=np.exp(canonical["log_representative"]),
        )
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            review.to_excel(writer, sheet_name="PRESENTATION ONLY", index=False)
        return output_path

    result = call_review_hook(
        write_review_workbook,
        fitted_model=object(),
        inputs=minimal_inputs(),
        output_path=tmp_path / "rating_tables_review.xlsx",
    )
    review = pd.read_excel(result.path, sheet_name="PRESENTATION ONLY")
    assert review["density_lower"].tolist() == pytest.approx([1.0, np.e])
    assert result.sha256
    assert canonical["log_lower"].tolist() == [0.0, 1.0]
```

- [ ] **Step 2: Run the review-hook tests and confirm failure**

Run: `rtk proxy .venv/bin/pytest tests/test_standard_superglm.py tests/test_scaffolded_standard_model.py -q`

Expected: FAIL because the standard runner does not invoke/hash a model-local review hook.

- [ ] **Step 3: Invoke and record an optional model-local hook**

After canonical workbook/receipt export and before candidate-bundle serialization, call the supplied module-level `write_review_workbook` hook. Require any returned path to stay inside the run output directory, compute its SHA-256/size, and store only that technical metadata in `CandidateBundle.review_artifact`. Never pass the review workbook to staging. Keep the generated default hook returning `None`; the concrete log/exp implementation belongs in the analyst model module and remains ordinary pandas/NumPy code.

- [ ] **Step 4: Run review/scaffold tests**

Run: `rtk proxy .venv/bin/pytest tests/test_standard_superglm.py tests/test_scaffolded_standard_model.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add pricing_pipeline/modeling/standard_superglm.py scripts/scaffold_pricing_model.py tests/test_standard_superglm.py tests/test_scaffolded_standard_model.py
rtk git commit -m "feat: add model-local review workbooks"
```

### Task 12: Add the generic notebook, end-to-end smoke path, and operator documentation

**Files:**
- Create: `tutorials/scaffolded_candidate_workbench.ipynb`
- Modify: `README.md`
- Modify: `tests/test_tutorials.py`
- Create: `tests/test_scaffolded_candidate_workflow.py`
- Modify: `tests/test_dag_import.py`

- [ ] **Step 1: Write failing notebook and end-to-end contract tests**

```python
def test_candidate_workbench_notebook_uses_runtime_facade_and_no_sql_ids():
    notebook = json.loads(
        Path("tutorials/scaffolded_candidate_workbench.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        cell_source
        for cell in notebook["cells"]
        for cell_source in cell.get("source", [])
    )
    assert "Workbench.from_runtime()" in source
    assert 'workbench.candidates("MY_MODEL")' in source
    assert "candidate.editor()" in source
    assert "candidate.submit_edits(" in source
    assert "rate_package_id=" not in source
    assert "create_engine(" not in source


def test_scaffolded_workflow_builds_opens_edits_and_submits(tmp_path):
    result = run_scaffolded_workflow_smoke(tmp_path)
    assert result.scheduled_candidate.bundle_verified is True
    assert result.editor_session_opened is True
    assert result.submission.dag_id == "pricing_publish_editor_candidate"
    assert result.submission.parent_package_version == 1
```

- [ ] **Step 2: Run notebook/workflow tests and confirm failure**

Run: `rtk proxy .venv/bin/pytest tests/test_tutorials.py tests/test_scaffolded_candidate_workflow.py tests/test_dag_import.py -q`

Expected: FAIL because the notebook and vertical smoke fixture do not exist.

- [ ] **Step 3: Add the notebook and concise scaffold-first documentation**

The notebook contains four executable sections: connect via `Workbench.from_runtime()`, display friendly candidate history, open one integer package version in the editor, and submit a reason with status/deployment examples. It contains no engine construction, SQL IDs, hashes, or Airflow credentials.

README documentation must state:

```text
scripts/scaffold_pricing_model.py is the supported model-authoring path.
The legacy generic DAG builder is not used by the candidate workbench workflow.
Scheduled training, editor publication, and deployment are separate DAG runs.
SQL stores audit/lookup metadata; verified joblib bundles store fitted Python objects.
The raw-axis review workbook is presentation-only and never staged for SQL scoring.
```

The smoke fixture scaffolds a temporary model package, supplies a four-row deterministic model/frame and fake SQL publisher, runs the real standard runner, reloads the verified candidate through `Workbench`, opens a fake editor session, and asserts the Airflow submission payload. It must not use MTPL as the scaffold contract.

- [ ] **Step 4: Run focused and full verification**

Run: `rtk proxy .venv/bin/pytest tests/test_tutorials.py tests/test_scaffolded_candidate_workflow.py tests/test_dag_import.py -q`

Expected: PASS.

Run: `rtk proxy .venv/bin/pytest tests/ -q`

Expected: PASS at 100% with no collection errors.

Run: `rtk ruff check pricing_pipeline scripts dags tests`

Expected: PASS with no diagnostics.

- [ ] **Step 5: Commit**

```bash
rtk git add tutorials/scaffolded_candidate_workbench.ipynb README.md tests/test_tutorials.py tests/test_scaffolded_candidate_workflow.py tests/test_dag_import.py
rtk git commit -m "docs: add scaffolded candidate workbench workflow"
```

## Final acceptance checklist

- [ ] A default custom scaffold imports no legacy DAG factory and contains no analyst-owned CV loop.
- [ ] The exact SuperGLM CV folds become the stored split artifact and failed/non-converged folds cannot silently publish.
- [ ] A successful scheduled run stores a verified candidate bundle and all SQL lineage without manual versions, hashes, or presence flags.
- [ ] Candidate history uses friendly integer package versions and deployment-slot state while hiding plumbing identifiers.
- [ ] The notebook retains the live editor session and the edited model artifact is authoritative over session JSON.
- [ ] Editor submission triggers a separate manual DAG, creates an immutable child package/run, and never deploys automatically.
- [ ] Deployment uses the existing deploy DAG through a friendly submission handle.
- [ ] Raw-axis Excel output remains a presentation artifact; canonical prepared-feature export is the only staged workbook.
- [ ] The complete suite, Ruff, and DAG import checks pass in the isolated worktree.
