from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _artifact_api():
    try:
        module = importlib.import_module("pricing_pipeline.workbench.artifacts")
        return (
            module.CandidateArtifactError,
            module.CandidateBundle,
            module.load_candidate_bundle,
            module.save_candidate_bundle,
        )
    except (ModuleNotFoundError, AttributeError) as exc:
        pytest.fail(f"candidate artifact API is not implemented: {exc}")


def _minimal_bundle():
    _, CandidateBundle, _, _ = _artifact_api()
    return CandidateBundle(
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


def _load(path: Path, metadata, *, allowed_root: Path):
    _, _, load_candidate_bundle, _ = _artifact_api()
    return load_candidate_bundle(
        path,
        expected_sha256=metadata.sha256,
        expected_size_bytes=metadata.size_bytes,
        expected_format=metadata.format,
        expected_python_version=metadata.python_version,
        expected_superglm_version=metadata.superglm_version,
        allowed_root=allowed_root,
    )


def test_candidate_bundle_round_trip_verifies_hash_and_lineage(tmp_path):
    _, _, _, save_candidate_bundle = _artifact_api()
    bundle = _minimal_bundle()

    metadata = save_candidate_bundle(bundle, tmp_path / "candidate_bundle.joblib")
    loaded = _load(Path(metadata.path), metadata, allowed_root=tmp_path)

    assert metadata.format == "superglm-candidate-joblib-v1"
    assert loaded.manifest_id == "manifest-1"
    assert loaded.split_set_id == "split-1"
    assert loaded.pk_columns == ("policy_id",)
    assert loaded.X.equals(bundle.X)
    assert np.array_equal(loaded.y, bundle.y)


def test_candidate_bundle_rejects_same_size_tampering(tmp_path):
    CandidateArtifactError, _, _, save_candidate_bundle = _artifact_api()
    path = tmp_path / "candidate_bundle.joblib"
    metadata = save_candidate_bundle(_minimal_bundle(), path)
    tampered = bytearray(path.read_bytes())
    tampered[-1] ^= 1
    path.write_bytes(tampered)

    with pytest.raises(CandidateArtifactError, match="SHA-256"):
        _load(path, metadata, allowed_root=tmp_path)


def test_candidate_bundle_rejects_path_outside_allowed_root(tmp_path):
    CandidateArtifactError, _, _, save_candidate_bundle = _artifact_api()
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    metadata = save_candidate_bundle(_minimal_bundle(), outside_root / "candidate.joblib")

    with pytest.raises(CandidateArtifactError, match="outside configured artifact root"):
        _load(Path(metadata.path), metadata, allowed_root=allowed_root)


def test_candidate_bundle_rejects_incompatible_python_before_deserializing(
    tmp_path,
    monkeypatch,
):
    CandidateArtifactError, _, load_candidate_bundle, save_candidate_bundle = _artifact_api()
    metadata = save_candidate_bundle(_minimal_bundle(), tmp_path / "candidate.joblib")
    deserialized = False

    def fail_if_loaded(path):
        nonlocal deserialized
        deserialized = True
        raise AssertionError(f"joblib.load must not be called for {path}")

    monkeypatch.setattr("joblib.load", fail_if_loaded)

    with pytest.raises(CandidateArtifactError, match="Python version"):
        load_candidate_bundle(
            metadata.path,
            expected_sha256=metadata.sha256,
            expected_size_bytes=metadata.size_bytes,
            expected_format=metadata.format,
            expected_python_version="2.7.18",
            expected_superglm_version=metadata.superglm_version,
            allowed_root=tmp_path,
        )

    assert deserialized is False
