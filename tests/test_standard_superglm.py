from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline.data.manifest import ModelFrameManifestSpec
from pricing_pipeline.models.config import ValidationSplitConfig


class _FakeModel:
    def __init__(self):
        self.fit_X = None
        self.fit_y = None
        self.fit_sample_weight = None
        self.fit_offset = None

    def fit_reml(self, X, y, sample_weight=None, offset=None):
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        self.fit_sample_weight = sample_weight
        self.fit_offset = offset
        return self

    def training_telemetry(self):
        return {"converged": True, "n_iter": 4}


def _api():
    try:
        module = importlib.import_module("pricing_pipeline.modeling.standard_superglm")
        return module
    except ModuleNotFoundError as exc:
        pytest.fail(f"standard SuperGLM API is not implemented: {exc}")


def _folds():
    return [
        (np.array([0, 1]), np.array([2])),
        (np.array([1, 2]), np.array([0])),
    ]


def _cv_result(*, converged=(True, True), oof_predictions=None):
    return SimpleNamespace(
        fold_scores=pd.DataFrame(
            {
                "fold": [0, 1],
                "n_train": [2, 2],
                "n_test": [1, 1],
                "fit_time_s": [0.1, 0.2],
                "score_time_s": [0.01, 0.02],
                "converged": list(converged),
                "n_iter": [3, 4],
                "effective_df": [1.5, 1.7],
                "deviance": [0.4, 0.5],
            }
        ),
        mean_scores={"deviance": np.float64(0.45)},
        pooled_scores={"deviance": np.float64(0.42)},
        std_scores={"deviance": np.float64(0.05)},
        fold_indices=_folds(),
        curve_similarity=None,
        oof_predictions=(
            np.array([0.25, np.nan, 0.75]) if oof_predictions is None else oof_predictions
        ),
        estimators=None,
    )


def test_precomputed_splitter_replays_exact_folds():
    api = _api()
    splitter = api.PrecomputedSplitter(_folds(), row_count=3)

    replayed = list(splitter.split(pd.DataFrame(index=range(3))))

    assert [pair[0].tolist() for pair in replayed] == [[0, 1], [1, 2]]
    assert [pair[1].tolist() for pair in replayed] == [[2], [0]]
    assert splitter.oof_coverage == pytest.approx(2 / 3)


def test_precomputed_splitter_rejects_duplicate_test_membership():
    api = _api()
    folds = [
        (np.array([0]), np.array([1])),
        (np.array([2]), np.array([1])),
    ]

    with pytest.raises(api.StandardSuperGLMError, match="duplicate test-row"):
        api.PrecomputedSplitter(folds, row_count=3)


def test_precomputed_splitter_rejects_out_of_range_indices():
    api = _api()
    folds = [(np.array([0, 1]), np.array([3]))]

    with pytest.raises(api.StandardSuperGLMError, match="outside row range"):
        api.PrecomputedSplitter(folds, row_count=3)


def test_cv_report_adapter_returns_json_primitives_and_stable_metrics():
    api = _api()

    report, metrics, fold_metrics = api.cv_result_to_records(
        _cv_result(),
        oof_coverage=2 / 3,
    )

    json.dumps(report, allow_nan=False)
    assert report["scope"] == "cv"
    assert report["oof_coverage"] == pytest.approx(2 / 3)
    assert report["oof_predictions"][0] == pytest.approx(0.25)
    assert report["oof_predictions"][1] is None
    assert report["oof_predictions"][2] == pytest.approx(0.75)
    assert metrics == {
        "cv_mean_deviance": pytest.approx(0.45),
        "cv_pooled_deviance": pytest.approx(0.42),
        "cv_std_deviance": pytest.approx(0.05),
        "cv_oof_coverage": pytest.approx(2 / 3),
    }
    assert [(item.fold_no, item.metric_name, item.metric_value) for item in fold_metrics] == [
        (1, "deviance", pytest.approx(0.4)),
        (2, "deviance", pytest.approx(0.5)),
    ]


def test_run_cross_validation_passes_strict_superglm_options():
    api = _api()
    captured = {}

    def fake_cross_validate(model, X, y, **kwargs):
        captured.update({"model": model, "X": X, "y": y, **kwargs})
        return _cv_result()

    inputs = api.ModelInputs(
        X=pd.DataFrame({"age": [20.0, 30.0, 40.0]}),
        y=np.array([0.0, 1.0, 0.0]),
    )
    evidence = api.run_cross_validation(
        object(),
        inputs,
        split_indices=_folds(),
        fit_mode="fit_reml",
        scoring=("deviance",),
        cross_validate_fn=fake_cross_validate,
    )

    assert captured["error_score"] == "raise"
    assert captured["return_oof"] is True
    assert captured["return_estimators"] is False
    assert captured["fit_mode"] == "fit_reml"
    assert evidence.metrics["cv_pooled_deviance"] == pytest.approx(0.42)
    assert evidence.fold_indices[0][1].tolist() == [2]


def test_run_cross_validation_rejects_non_converged_fold():
    api = _api()
    inputs = api.ModelInputs(
        X=pd.DataFrame({"age": [20.0, 30.0, 40.0]}),
        y=np.array([0.0, 1.0, 0.0]),
    )

    with pytest.raises(api.StandardSuperGLMError, match="fold 2 did not converge"):
        api.run_cross_validation(
            object(),
            inputs,
            split_indices=_folds(),
            fit_mode="fit",
            scoring=("deviance",),
            cross_validate_fn=lambda *args, **kwargs: _cv_result(converged=(True, False)),
        )


def test_standard_runner_requires_explicit_canonical_row_ids(tmp_path):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )

    with pytest.raises(api.StandardSuperGLMError, match="requires ModelInputs.row_ids"):
        api.run_standard_superglm_build(
            object(),
            frame=frame,
            inputs=api.ModelInputs(
                X=frame[["age"]],
                y=frame["target"].to_numpy(),
            ),
            model_factory=_FakeModel,
            split_indices=_folds(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            output_dir=tmp_path / "run",
            model_id=17,
            model_name="HOME_FREQ",
            model_version="v1",
            model_type="superglm_poisson",
            target_name="target",
            deployment_slot="HOME_FREQ_CURRENT",
            export_id="export-1",
            effective_from="2026-07-12",
            manifest_spec=ModelFrameManifestSpec(
                dataset_name="home_freq_frame",
                source_system="pytest",
                data_as_of_date="2026-06-30",
                pk_columns=("policy_id",),
                target_column="target",
            ),
            validation_split=ValidationSplitConfig(
                method="custom",
                n_splits=None,
                random_state=None,
                shuffle=False,
                materialize=True,
            ),
            split_artifact_root=tmp_path / "splits",
            model_source_root=tmp_path / "source",
            created_by="pytest",
            cross_validate_fn=lambda *args, **kwargs: pytest.fail(
                "CV must not run before canonical-row validation"
            ),
        )


def test_standard_runner_rejects_model_source_drift_during_training(
    tmp_path,
    monkeypatch,
):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )
    source_hashes = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(api, "hash_model_source", lambda root: next(source_hashes))
    monkeypatch.setattr(
        api,
        "create_model_frame_manifest_with_split",
        lambda *args, **kwargs: pytest.fail(
            "source drift must fail before audit evidence is persisted"
        ),
    )

    with pytest.raises(api.StandardSuperGLMError, match="model source changed"):
        api.run_standard_superglm_build(
            object(),
            frame=frame,
            inputs=_identity_bound_inputs(api, frame),
            model_factory=_FakeModel,
            split_indices=_folds(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            output_dir=tmp_path / "run",
            model_id=17,
            model_name="HOME_FREQ",
            model_version="v1",
            model_type="superglm_poisson",
            target_name="target",
            deployment_slot="HOME_FREQ_CURRENT",
            export_id="export-1",
            effective_from=None,
            manifest_spec=ModelFrameManifestSpec(
                dataset_name="home_freq_frame",
                source_system="pytest",
                data_as_of_date="2026-06-30",
                pk_columns=("policy_id",),
                target_column="target",
            ),
            validation_split=ValidationSplitConfig(
                method="custom",
                n_splits=None,
                random_state=None,
                shuffle=False,
                materialize=True,
            ),
            split_artifact_root=tmp_path / "splits",
            model_source_root=tmp_path / "source",
            created_by="pytest",
            cross_validate_fn=lambda *args, **kwargs: _cv_result(),
        )


@pytest.mark.parametrize(
    ("case", "input_builder", "match"),
    [
        (
            "filtered",
            lambda frame: (
                frame.iloc[:2][["age"]].copy(),
                frame.iloc[:2][["policy_id"]].copy(),
            ),
            "row count",
        ),
        (
            "reordered",
            lambda frame: (
                frame.iloc[::-1][["age"]].copy(),
                frame.iloc[::-1][["policy_id"]].copy(),
            ),
            "index/order",
        ),
        (
            "reset-index",
            lambda frame: (
                frame[["age"]].reset_index(drop=True),
                frame[["policy_id"]].reset_index(drop=True),
            ),
            "index/order",
        ),
        (
            "wrong-pk",
            lambda frame: (
                frame[["age"]].copy(),
                frame[["policy_id"]].rename(columns={"policy_id": "account_id"}),
            ),
            "primary-key columns",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_standard_runner_rejects_inputs_not_aligned_to_canonical_frame(
    tmp_path,
    case,
    input_builder,
    match,
):
    del case
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        },
        index=[10, 11, 12],
    )
    X, row_ids = input_builder(frame)

    with pytest.raises(api.StandardSuperGLMError, match=match):
        api.run_standard_superglm_build(
            object(),
            frame=frame,
            inputs=api.ModelInputs(
                X=X,
                y=np.zeros(len(X)),
                row_ids=row_ids,
            ),
            model_factory=_FakeModel,
            split_indices=_folds(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            output_dir=tmp_path / "run",
            model_id=17,
            model_name="HOME_FREQ",
            model_version="v1",
            model_type="superglm_poisson",
            target_name="target",
            deployment_slot="HOME_FREQ_CURRENT",
            export_id="export-1",
            effective_from="2026-07-12",
            manifest_spec=ModelFrameManifestSpec(
                dataset_name="home_freq_frame",
                source_system="pytest",
                data_as_of_date="2026-06-30",
                pk_columns=("policy_id",),
                target_column="target",
            ),
            validation_split=ValidationSplitConfig(
                method="custom",
                n_splits=None,
                random_state=None,
                shuffle=False,
                materialize=True,
            ),
            split_artifact_root=tmp_path / "splits",
            model_source_root=tmp_path / "source",
            created_by="pytest",
            cross_validate_fn=lambda *args, **kwargs: pytest.fail(
                "CV must not run before canonical-row validation"
            ),
        )


@pytest.mark.parametrize(
    ("pk_values", "match"),
    [
        ([1, None, 3], "null"),
        ([1, 1, 3], "duplicate"),
    ],
)
def test_standard_runner_rejects_missing_or_duplicate_row_identity_before_cv(
    tmp_path,
    pk_values,
    match,
):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": pk_values,
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )

    with pytest.raises(api.StandardSuperGLMError, match=match):
        api.run_standard_superglm_build(
            object(),
            frame=frame,
            inputs=_identity_bound_inputs(api, frame),
            model_factory=_FakeModel,
            split_indices=_folds(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            output_dir=tmp_path / "run",
            model_id=17,
            model_name="HOME_FREQ",
            model_version="v1",
            model_type="superglm_poisson",
            target_name="target",
            deployment_slot="HOME_FREQ_CURRENT",
            export_id="export-1",
            effective_from="2026-07-12",
            manifest_spec=ModelFrameManifestSpec(
                dataset_name="home_freq_frame",
                source_system="pytest",
                data_as_of_date="2026-06-30",
                pk_columns=("policy_id",),
                target_column="target",
            ),
            validation_split=ValidationSplitConfig(
                method="custom",
                n_splits=None,
                random_state=None,
                shuffle=False,
                materialize=True,
            ),
            split_artifact_root=tmp_path / "splits",
            model_source_root=tmp_path / "source",
            created_by="pytest",
            cross_validate_fn=lambda *args, **kwargs: pytest.fail(
                "CV must not run before canonical-row validation"
            ),
        )


def _identity_bound_inputs(api, frame, **overrides):
    row_ids = frame[["policy_id"]].copy()
    identity = pd.Index(row_ids["policy_id"].to_numpy(copy=True), name="policy_id")
    X = frame[["age"]].copy()
    X.index = identity
    values = {
        "X": X,
        "y": pd.Series(
            frame["target"].to_numpy(copy=True),
            index=identity,
            name="target",
        ),
        "row_ids": row_ids,
    }
    values.update(overrides)
    return api.ModelInputs(**values)


def test_canonical_validation_rejects_reversed_then_reset_feature_frame():
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [0, 1, 2],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )
    inputs = _identity_bound_inputs(
        api,
        frame,
        X=frame.iloc[::-1][["age"]].reset_index(drop=True),
    )

    with pytest.raises(api.StandardSuperGLMError, match="ModelInputs.X.*identity index"):
        api._validate_canonical_row_ids(frame, inputs, pk_columns=("policy_id",))


def test_canonical_validation_rejects_reordered_target_series():
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [0, 1, 2],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )
    inputs = _identity_bound_inputs(api, frame)
    reordered_y = inputs.y.iloc[::-1]
    inputs = _identity_bound_inputs(api, frame, y=reordered_y)

    with pytest.raises(api.StandardSuperGLMError, match="ModelInputs.y.*identity index"):
        api._validate_canonical_row_ids(frame, inputs, pk_columns=("policy_id",))


@pytest.mark.parametrize("field_name", ["sample_weight", "offset", "export_weight"])
def test_canonical_validation_rejects_reordered_optional_row_inputs(field_name):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [0, 1, 2],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )
    identity = pd.Index([2, 1, 0], name="policy_id")
    reordered = pd.Series([1.0, 1.0, 1.0], index=identity, name=field_name)
    inputs = _identity_bound_inputs(api, frame, **{field_name: reordered})

    with pytest.raises(
        api.StandardSuperGLMError,
        match=rf"ModelInputs.{field_name}.*identity index",
    ):
        api._validate_canonical_row_ids(frame, inputs, pk_columns=("policy_id",))


@pytest.mark.parametrize(
    "manifest_id",
    ("../escape", "nested/manifest", "manifest id", ".", ""),
)
def test_manifest_attempt_directory_rejects_unsafe_path_components(
    tmp_path,
    manifest_id,
):
    api = _api()

    with pytest.raises(api.StandardSuperGLMError, match="safe path component"):
        api._manifest_attempt_directory(tmp_path / "run", manifest_id)


def test_standard_runner_removes_partial_attempt_but_keeps_manifest_evidence(
    tmp_path,
    monkeypatch,
):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
        }
    )
    split_evidence = tmp_path / "splits" / "manifest-failure-split.npz"
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "model.py").write_text("MODEL = 'HOME_FREQ'\n", encoding="utf-8")

    def fake_manifest(engine, **kwargs):
        del engine, kwargs
        split_evidence.parent.mkdir(parents=True)
        split_evidence.write_bytes(b"durable split evidence")
        return SimpleNamespace(
            manifest_id="manifest-failure",
            split_set_id="manifest-failure-split",
            split_artifact_uri=str(split_evidence),
        )

    def failing_export(model, X, y, exposure, output_path, **kwargs):
        del model, X, y, exposure, kwargs
        Path(output_path).write_bytes(b"partial workbook")
        raise RuntimeError("artifact export failed")

    monkeypatch.setattr(api, "create_model_frame_manifest_with_split", fake_manifest)
    monkeypatch.setattr(api, "export_rating_tables", failing_export)

    with pytest.raises(RuntimeError, match="artifact export failed"):
        api.run_standard_superglm_build(
            object(),
            frame=frame,
            inputs=_identity_bound_inputs(api, frame),
            model_factory=_FakeModel,
            split_indices=_folds(),
            fit_mode="fit_reml",
            scoring=("deviance",),
            cross_validate_fn=lambda *args, **kwargs: _cv_result(),
            output_dir=tmp_path / "run",
            model_id=17,
            model_name="HOME_FREQ",
            model_version="v1",
            model_type="superglm_poisson",
            target_name="target",
            deployment_slot="HOME_FREQ_CURRENT",
            export_id="export-1",
            effective_from="2026-07-12",
            manifest_spec=ModelFrameManifestSpec(
                dataset_name="home_freq_frame",
                source_system="pytest",
                data_as_of_date="2026-06-30",
                pk_columns=("policy_id",),
                target_column="target",
            ),
            validation_split=ValidationSplitConfig(
                method="custom",
                n_splits=None,
                random_state=None,
                shuffle=False,
                materialize=True,
            ),
            split_artifact_root=tmp_path / "splits",
            model_source_root=source_root,
            created_by="pytest",
        )

    assert not (tmp_path / "run" / "manifest-failure").exists()
    assert split_evidence.read_bytes() == b"durable split evidence"


def test_standard_runner_uses_cv_folds_for_manifest_and_returns_candidate_metadata(
    tmp_path,
    monkeypatch,
):
    api = _api()
    frame = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "target": [0.0, 1.0, 0.0],
            "age": [20.0, 30.0, 40.0],
            "Exposure": [2.0, 4.0, 3.0],
        }
    )
    source_root = tmp_path / "pricing_models" / "home_freq"
    (source_root / "sql").mkdir(parents=True)
    (source_root / "modeling.py").write_text("FIT_MODE = 'fit_reml'\n", encoding="utf-8")
    (source_root / "model.toml").write_text('model_name = "HOME_FREQ"\n', encoding="utf-8")
    (source_root / "sql" / "source.sql").write_text("SELECT 1;\n", encoding="utf-8")
    captured = {}
    models = []

    def model_factory():
        model = _FakeModel()
        models.append(model)
        return model

    def fake_export(model, X, y, exposure, output_path, **kwargs):
        captured["export_weight"] = exposure
        captured["export_options"] = kwargs
        Path(output_path).write_bytes(b"canonical workbook")
        return Path(output_path)

    manifest_ids = iter(("manifest-1", "manifest-2"))

    def fake_manifest(engine, **kwargs):
        captured["manifest"] = kwargs
        manifest_id = next(manifest_ids)
        return SimpleNamespace(
            manifest_id=manifest_id,
            split_set_id=f"{manifest_id}-split",
            split_artifact_uri=str(tmp_path / "splits" / f"{manifest_id}-split.npz"),
        )

    def fake_receipt_writer(receipt, path):
        Path(path).write_bytes(b"canonical receipt")
        return "c" * 64

    monkeypatch.setattr(api, "export_rating_tables", fake_export)
    monkeypatch.setattr(api, "create_model_frame_manifest_with_split", fake_manifest)
    monkeypatch.setattr(api, "build_superglm_publication_receipt", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "write_publication_receipt", fake_receipt_writer)

    base_inputs = _identity_bound_inputs(api, frame)
    exposure = pd.Series(
        frame["Exposure"].to_numpy(copy=True),
        index=base_inputs.X.index,
        name="Exposure",
    )
    inputs = _identity_bound_inputs(
        api,
        frame,
        offset=np.log(exposure),
        export_weight=exposure,
        export_weight_name="Exposure",
    )
    from pricing_pipeline.publishing.superglm_publication_receipt import (
        OffsetExportContract,
    )

    build_kwargs = {
        "frame": frame,
        "inputs": inputs,
        "model_factory": model_factory,
        "split_indices": _folds(),
        "fit_mode": "fit_reml",
        "scoring": ("deviance",),
        "cross_validate_fn": lambda *args, **kwargs: _cv_result(),
        "output_dir": tmp_path / "run",
        "model_id": 17,
        "model_name": "HOME_FREQ",
        "model_version": "v1",
        "model_type": "superglm_poisson",
        "target_name": "target",
        "deployment_slot": "HOME_FREQ_CURRENT",
        "export_id": "export-1",
        "effective_from": "2026-07-12",
        "manifest_spec": ModelFrameManifestSpec(
            dataset_name="home_freq_frame",
            source_system="pytest",
            data_as_of_date="2026-06-30",
            pk_columns=("policy_id",),
            target_column="target",
        ),
        "validation_split": ValidationSplitConfig(
            method="custom",
            n_splits=None,
            random_state=None,
            shuffle=False,
            materialize=True,
        ),
        "split_artifact_root": tmp_path / "splits",
        "model_source_root": source_root,
        "created_by": "pytest",
        "offset_contract": OffsetExportContract(
            handling="EXPORTED_FACTOR",
            source_factor_name="Exposure",
            published_factor_name="Exposure",
            source_name="Exposure",
            label="log(Exposure)",
        ),
    }
    result = api.run_standard_superglm_build(object(), **build_kwargs)

    from pricing_pipeline.orchestration.publish_completed_build import CompletedModelBuild
    from pricing_pipeline.workbench.artifacts import load_candidate_bundle

    assert isinstance(result.completed_build, CompletedModelBuild)
    bundle = load_candidate_bundle(
        result.completed_build.candidate_artifact_path,
        expected_sha256=result.completed_build.candidate_artifact_sha256,
        expected_size_bytes=result.completed_build.candidate_artifact_size_bytes,
        expected_format=result.completed_build.candidate_artifact_format,
        expected_python_version=result.completed_build.candidate_python_version,
        expected_superglm_version=result.completed_build.candidate_superglm_version,
        allowed_root=tmp_path / "run",
    )
    assert bundle.model_name == "HOME_FREQ"
    assert bundle.model_version == "v1"
    assert bundle.export_id == "export-1"
    first_paths = {
        "workbook": Path(result.completed_build.rating_workbook_path),
        "receipt": Path(result.completed_build.publication_receipt_path),
        "candidate": Path(result.completed_build.candidate_artifact_path),
    }
    first_bytes = {name: path.read_bytes() for name, path in first_paths.items()}
    second_result = api.run_standard_superglm_build(object(), **build_kwargs)

    assert [test.tolist() for _, test in captured["manifest"]["split_indices"]] == [
        [2],
        [0],
    ]
    assert models[1].fit_X.equals(inputs.X)
    np.testing.assert_allclose(captured["export_weight"], exposure)
    np.testing.assert_allclose(captured["export_options"]["offset"], np.log(exposure))
    np.testing.assert_allclose(captured["export_options"]["offset_source"], exposure)
    assert captured["export_options"]["offset_name"] == "Exposure"
    assert captured["export_options"]["offset_kind"] == "auto"
    assert result.completed_build.manifest_id == "manifest-1"
    assert result.completed_build.split_set_id == "manifest-1-split"
    assert second_result.completed_build.manifest_id == "manifest-2"
    second_paths = {
        "workbook": Path(second_result.completed_build.rating_workbook_path),
        "receipt": Path(second_result.completed_build.publication_receipt_path),
        "candidate": Path(second_result.completed_build.candidate_artifact_path),
    }
    assert {path.parent for path in first_paths.values()} == {
        (tmp_path / "run" / "manifest-1").resolve()
    }
    assert {path.parent for path in second_paths.values()} == {
        (tmp_path / "run" / "manifest-2").resolve()
    }
    assert set(first_paths.values()).isdisjoint(second_paths.values())
    assert {name: path.read_bytes() for name, path in first_paths.items()} == first_bytes
    assert Path(result.completed_build.candidate_artifact_path).exists()
    assert result.completed_build.candidate_artifact_sha256
    assert result.completed_build.model_source_sha256
    assert result.completed_build.rating_workbook_sha256 == api.hash_file_sha256(
        first_paths["workbook"]
    )
    assert result.metrics["cv_pooled_deviance"] == pytest.approx(0.42)
    assert bundle.cv_report["model_name"] == "HOME_FREQ"
    assert bundle.cv_report["fit_mode"] == "fit_reml"
    assert bundle.cv_report["scoring"] == ["deviance"]


def test_model_source_hash_tracks_notebook_source_but_ignores_execution_output(tmp_path):
    from pricing_pipeline.modeling.standard_superglm import hash_model_source

    notebook_path = tmp_path / "pricing_model.ipynb"
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["MODEL_NAME = 'HOME_FREQ'\n"],
            }
        ],
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    original = hash_model_source(tmp_path)

    checkpoint_dir = tmp_path / ".ipynb_checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "pricing_model-checkpoint.ipynb"
    checkpoint.write_text(
        json.dumps(
            {
                **notebook,
                "cells": [
                    {
                        **notebook["cells"][0],
                        "source": ["MODEL_NAME = 'STALE_CHECKPOINT'\n"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint_only_change = hash_model_source(tmp_path)

    notebook["cells"][0]["execution_count"] = 7
    notebook["cells"][0]["outputs"] = [{"output_type": "stream", "text": ["trained\n"]}]
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    output_only_change = hash_model_source(tmp_path)

    notebook["cells"][0]["source"] = ["MODEL_NAME = 'HOME_SEVERITY'\n"]
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    source_change = hash_model_source(tmp_path)

    assert checkpoint_only_change == original
    assert output_only_change == original
    assert source_change != original
