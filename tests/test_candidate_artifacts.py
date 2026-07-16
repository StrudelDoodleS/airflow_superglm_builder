from __future__ import annotations

import importlib
from dataclasses import replace
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
        offset_source=None,
        export_weight=None,
        cv_report={"scope": "cv", "pooled_scores": {"deviance": 0.4}},
        model_name="HOME_FREQ",
        model_version="v1",
        export_id="export-1",
        manifest_id="manifest-1",
        split_set_id="split-1",
        pk_columns=("policy_id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        model_frame_sha256="c" * 64,
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
    from pricing_pipeline.publishing.superglm_publication_receipt import (
        OffsetExportContract,
    )

    _, _, _, save_candidate_bundle = _artifact_api()
    bundle = _minimal_bundle()

    metadata = save_candidate_bundle(bundle, tmp_path / "candidate_bundle.joblib")
    loaded = _load(Path(metadata.path), metadata, allowed_root=tmp_path)

    assert metadata.format == "superglm-candidate-joblib-v2"
    assert loaded.model_name == "HOME_FREQ"
    assert loaded.model_version == "v1"
    assert loaded.export_id == "export-1"
    assert loaded.manifest_id == "manifest-1"
    assert loaded.split_set_id == "split-1"
    assert loaded.pk_columns == ("policy_id",)
    assert loaded.model_frame_sha256 == "c" * 64
    assert loaded.X.equals(bundle.X)
    assert np.array_equal(loaded.y, bundle.y)
    assert loaded.offset_contract == OffsetExportContract(handling="NONE")
    assert not hasattr(loaded, "offset_export_options")


@pytest.mark.parametrize("digest", ["", "A" * 64, "a" * 63, "g" * 64])
def test_candidate_bundle_rejects_invalid_model_frame_sha256(digest):
    CandidateArtifactError, _, _, _ = _artifact_api()

    with pytest.raises(CandidateArtifactError, match="model_frame_sha256"):
        replace(_minimal_bundle(), model_frame_sha256=digest)


@pytest.mark.parametrize(
    ("offset_source", "offset_source_name", "message"),
    [
        (None, "Term", "EXPORTED_FACTOR requires offset_source"),
        (np.array([12.0, 36.0]), None, "EXPORTED_FACTOR requires offset_source_name"),
        (np.array([12.0]), "Term", "offset_source length 1 does not match X row count 2"),
        (
            np.array([12.0, float("inf")]),
            "Term",
            "offset_source contains non-finite numeric values",
        ),
    ],
)
def test_exported_offset_bundle_rejects_missing_or_invalid_source(
    offset_source,
    offset_source_name,
    message,
):
    CandidateArtifactError, _, _, _ = _artifact_api()
    from pricing_pipeline.publishing.superglm_publication_receipt import (
        OffsetExportContract,
    )

    with pytest.raises(CandidateArtifactError, match=message):
        replace(
            _minimal_bundle(),
            offset=np.log(np.array([1.0, 3.0])),
            offset_source=offset_source,
            offset_source_name=offset_source_name,
            offset_contract=OffsetExportContract(
                handling="EXPORTED_FACTOR",
                source_factor_name="Term",
                published_factor_name="Term",
                source_name="Term",
                label="log(Term / 12)",
            ),
        )


def test_exported_offset_bundle_rejects_source_name_that_conflicts_with_contract():
    CandidateArtifactError, _, _, _ = _artifact_api()
    from pricing_pipeline.publishing.superglm_publication_receipt import (
        OffsetExportContract,
    )

    with pytest.raises(CandidateArtifactError, match="offset_source_name.*source_name"):
        replace(
            _minimal_bundle(),
            offset=np.log(np.array([1.0, 3.0])),
            offset_source=np.array([12.0, 36.0]),
            offset_source_name="OtherTerm",
            offset_contract=OffsetExportContract(
                handling="EXPORTED_FACTOR",
                source_factor_name="Term",
                published_factor_name="Term",
                source_name="Term",
                label="log(Term / 12)",
            ),
        )


@pytest.mark.parametrize("field_name", ["model_name", "model_version", "export_id"])
def test_candidate_bundle_rejects_missing_model_identity(tmp_path, field_name):
    CandidateArtifactError, _, _, save_candidate_bundle = _artifact_api()
    bundle = replace(_minimal_bundle(), **{field_name: " "})

    with pytest.raises(CandidateArtifactError, match=field_name):
        save_candidate_bundle(bundle, tmp_path / "candidate_bundle.joblib")


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


def test_candidate_bundle_deserializes_the_verified_snapshot_when_path_is_replaced(
    tmp_path,
    monkeypatch,
):
    _, _, _, save_candidate_bundle = _artifact_api()
    trusted_path = tmp_path / "candidate.joblib"
    replacement_path = tmp_path / "replacement.joblib"
    trusted_metadata = save_candidate_bundle(_minimal_bundle(), trusted_path)
    replacement_bundle = replace(_minimal_bundle(), manifest_id="manifest-replaced")
    save_candidate_bundle(replacement_bundle, replacement_path)
    real_joblib_load = importlib.import_module("joblib").load

    def replace_path_then_deserialize(source):
        replacement_path.replace(trusted_path)
        return real_joblib_load(source)

    monkeypatch.setattr(
        "pricing_pipeline.workbench.artifacts.joblib.load",
        replace_path_then_deserialize,
    )

    loaded = _load(trusted_path, trusted_metadata, allowed_root=tmp_path)

    assert loaded.manifest_id == "manifest-1"
