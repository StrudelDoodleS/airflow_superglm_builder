from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from superglm import Categorical, Numeric, collapse_levels
from superglm.features.grouping import LevelGrouping

from pricing_pipeline.data.manifest import model_frame_evidence
from pricing_pipeline.modeling import level_grouping_artifact as grouping_api
from pricing_pipeline.workbench.core import Candidate


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy_id": range(1, 9),
            "region": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "brand": ["B1", "B2", "B3", "B4", "B1", "B2", "B3", "B4"],
            "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "data_as_of": ["2026-06-30"] * 8,
        }
    )


def _editor_session(frame: pd.DataFrame, *, reference_model=None):
    region = collapse_levels(
        frame["region"],
        groups={"B_OR_C": ["B", "C"]},
    )
    brand = collapse_levels(
        frame["brand"],
        groups={
            "B1_OR_B2": ["B1", "B2"],
            "B3_OR_B4": ["B3", "B4"],
        },
    )
    model = SimpleNamespace(
        features={
            "region": Categorical(grouping=region),
            "brand": Categorical(grouping=brand),
            "x": Numeric(),
        }
    )
    return SimpleNamespace(model=model, reference_model=reference_model)


def _save(frame: pd.DataFrame, path: Path, *, session=None, replace=False):
    frame_sha256, _ = model_frame_evidence(frame)
    return grouping_api.save_editor_level_groupings(
        _editor_session(frame) if session is None else session,
        path,
        source_model_name="HOME_FREQ",
        source_package_version=7,
        source_manifest_id="manifest-20260630",
        source_model_frame_sha256=frame_sha256,
        source_data_as_of_date="2026-06-30",
        replace=replace,
    )


def test_actual_level_grouping_objects_round_trip_for_multiple_features_and_groups(
    tmp_path: Path,
):
    frame = _frame()
    path = tmp_path / "routine_groupings.joblib"

    saved = _save(frame, path)
    loaded = grouping_api.load_level_groupings(
        path,
        frame=frame,
        expected_model_name="HOME_FREQ",
        expected_data_as_of_date="2026-06-30",
        expected_manifest_id="manifest-20260630",
        allowed_root=tmp_path,
    )

    assert grouping_api.inspect_level_groupings(path) == saved
    assert saved.feature_names == ("brand", "region")
    assert saved.collapsed_group_count == 3
    assert set(loaded) == {"brand", "region"}
    assert all(type(grouping) is LevelGrouping for grouping in loaded.values())
    assert loaded["brand"].group_to_originals == {
        "B1_OR_B2": ["B1", "B2"],
        "B3_OR_B4": ["B3", "B4"],
    }

    raw_features = {
        "region": Categorical(base="first"),
        "brand": Categorical(base="B2"),
        "x": Numeric(),
    }
    routine_features = grouping_api.apply_level_groupings(raw_features, loaded)
    assert raw_features["region"]._grouping is None
    assert routine_features["region"] is not raw_features["region"]
    assert routine_features["region"]._grouping == loaded["region"]
    assert routine_features["region"].base == "first"
    assert routine_features["brand"].base == "B1_OR_B2"


def test_extracts_multiple_groups_from_a_real_editor_session():
    import numpy as np
    from superglm import SuperGLM
    from superglm.editor import EditorSession

    frame = pd.DataFrame(
        {
            "region": np.repeat(["A", "B", "C", "D"], 20),
            "x": np.tile(np.linspace(-1.0, 1.0, 20), 4),
        }
    )
    means = frame["region"].map({"A": 1.0, "B": 2.0, "C": 2.0, "D": 4.0})
    y = np.random.default_rng(20260810).poisson(means * np.exp(0.2 * frame["x"]))
    model = SuperGLM(
        features={"region": Categorical(base="first"), "x": Numeric()},
        selection_penalty=0.0,
    ).fit(frame, y)
    session = EditorSession.from_model(model, train_data=(frame, y))
    session.select_levels("region", ["B", "C"])
    session.replace_with_collapsed_levels("region", method="fit")
    session.select_levels("region", ["A", "D"])
    session.replace_with_collapsed_levels("region", method="fit")

    groupings = grouping_api.extract_editor_level_groupings(session)

    assert type(groupings["region"]) is LevelGrouping
    assert groupings["region"].group_to_originals == {
        "A+D": ["A", "D"],
        "B+C": ["B", "C"],
    }


def test_identity_only_editor_grouping_exports_an_empty_skip_artifact(tmp_path: Path):
    frame = _frame()
    identity = collapse_levels(frame["region"])
    session = SimpleNamespace(
        model=SimpleNamespace(features={"region": Categorical(grouping=identity)})
    )
    path = tmp_path / "routine_groupings.joblib"

    saved = _save(frame, path, session=session)
    loaded = grouping_api.load_level_groupings(
        path,
        frame=frame,
        expected_model_name="HOME_FREQ",
        expected_data_as_of_date="2026-06-30",
        allowed_root=tmp_path,
    )

    assert saved.feature_names == ()
    assert saved.collapsed_group_count == 0
    assert loaded == {}


def test_grouping_save_is_idempotent_and_requires_replace_for_changed_decision(
    tmp_path: Path,
):
    frame = _frame()
    path = tmp_path / "routine_groupings.joblib"
    first = _save(frame, path)
    retry = _save(frame.copy(), path)
    changed = collapse_levels(frame["region"], groups={"A_OR_D": ["A", "D"]})
    changed_session = SimpleNamespace(
        model=SimpleNamespace(features={"region": Categorical(grouping=changed)})
    )

    assert retry == first
    with pytest.raises(FileExistsError, match="replace=True"):
        _save(frame, path, session=changed_session)

    replaced = _save(frame, path, session=changed_session, replace=True)
    assert replaced.grouping_sha256 != first.grouping_sha256


def test_grouping_save_requires_replace_after_superglm_runtime_changes(
    tmp_path: Path,
    monkeypatch,
):
    frame = _frame()
    path = tmp_path / "routine_groupings.joblib"
    _save(frame, path)
    monkeypatch.setattr(grouping_api, "_superglm_version", lambda: "future-public-api")

    with pytest.raises(FileExistsError, match="replace=True"):
        _save(frame, path)

    replaced = _save(frame, path, replace=True)
    assert replaced.superglm_version == "future-public-api"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda frame: frame.assign(data_as_of="2026-07-31"), "ordered model frame"),
        (
            lambda frame: frame.assign(region=["A", "B", "C", "NEW", "A", "B", "C", "NEW"]),
            "ordered model frame",
        ),
    ],
)
def test_grouping_load_rejects_a_different_frame_before_application(
    tmp_path: Path,
    change,
    message: str,
):
    frame = _frame()
    path = tmp_path / "routine_groupings.joblib"
    _save(frame, path)

    with pytest.raises(grouping_api.LevelGroupingArtifactError, match=message):
        grouping_api.load_level_groupings(
            path,
            frame=change(frame.copy()),
            expected_model_name="HOME_FREQ",
            expected_data_as_of_date="2026-06-30",
            allowed_root=tmp_path,
        )


def test_grouping_load_rejects_wrong_model_data_as_at_manifest_and_root(tmp_path: Path):
    frame = _frame()
    path = tmp_path / "inside" / "routine_groupings.joblib"
    _save(frame, path)

    common = {
        "frame": frame,
        "expected_model_name": "HOME_FREQ",
        "expected_data_as_of_date": "2026-06-30",
        "expected_manifest_id": "manifest-20260630",
        "allowed_root": tmp_path,
    }
    with pytest.raises(grouping_api.LevelGroupingArtifactError, match="different model"):
        grouping_api.load_level_groupings(path, **{**common, "expected_model_name": "OTHER"})
    with pytest.raises(grouping_api.LevelGroupingArtifactError, match="data-as-at"):
        grouping_api.load_level_groupings(
            path,
            **{**common, "expected_data_as_of_date": "2026-07-31"},
        )
    with pytest.raises(grouping_api.LevelGroupingArtifactError, match="dataset manifest"):
        grouping_api.load_level_groupings(
            path,
            **{**common, "expected_manifest_id": "manifest-other"},
        )
    with pytest.raises(grouping_api.LevelGroupingArtifactError, match="outside allowed root"):
        grouping_api.load_level_groupings(
            path,
            **{**common, "allowed_root": tmp_path / "elsewhere"},
        )


def test_tampered_grouping_bytes_are_rejected_before_joblib_deserialization(
    tmp_path: Path,
    monkeypatch,
):
    frame = _frame()
    path = tmp_path / "routine_groupings.joblib"
    _save(frame, path)
    path.write_bytes(path.read_bytes() + b"tampered")

    def unexpected_load(_source):
        raise AssertionError("joblib.load must not run before integrity checks")

    monkeypatch.setattr(grouping_api.joblib, "load", unexpected_load)
    with pytest.raises(grouping_api.LevelGroupingArtifactError, match="size"):
        grouping_api.load_level_groupings(
            path,
            frame=frame,
            expected_model_name="HOME_FREQ",
            expected_data_as_of_date="2026-06-30",
            allowed_root=tmp_path,
        )


def test_tampered_readable_grouping_evidence_is_rejected_without_unpickling(
    tmp_path: Path,
):
    frame = _frame()
    path = tmp_path / "routine_groupings.joblib"
    _save(frame, path)
    metadata_path = path.with_suffix(".joblib.json")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["groupings"]["region"]["original_to_group"]["A"] = "B_OR_C"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(grouping_api.LevelGroupingArtifactError, match="inconsistent"):
        grouping_api.inspect_level_groupings(path)


def test_malformed_non_partition_grouping_is_never_exported(tmp_path: Path):
    frame = _frame()
    malformed = LevelGrouping(
        original_to_group={"A": "AB", "B": "AB"},
        group_to_originals={"AB": ["A", "B"], "C": ["B"]},
        all_original_levels=["A", "B"],
        grouped_levels=["AB", "C"],
    )
    session = SimpleNamespace(
        model=SimpleNamespace(features={"region": Categorical(grouping=malformed)})
    )

    with pytest.raises(grouping_api.LevelGroupingArtifactError, match="inconsistent"):
        _save(frame, tmp_path / "routine_groupings.joblib", session=session)


def test_notebook_api_binds_export_to_selected_raw_candidate_and_registered_model(
    tmp_path: Path,
):
    from pricing_pipeline import notebook as notebook_api

    frame = _frame()
    frame_sha256, _ = model_frame_evidence(frame)
    source_model = object()
    session = _editor_session(frame, reference_model=source_model)
    candidate = Candidate(
        workbench=SimpleNamespace(),
        model_name="HOME_FREQ",
        package_version=7,
        rate_package_id=17,
        parent_rate_package_id=None,
        model_run_id=27,
        bundle=SimpleNamespace(
            fitted_model=source_model,
            model_frame_sha256=frame_sha256,
            manifest_id="manifest-20260630",
        ),
        technical={
            "model_kind": "RAW",
            "data_as_of_date": "2026-06-30",
        },
    )
    path = tmp_path / "routine_groupings.joblib"

    exported = notebook_api.export_level_groupings(
        candidate,
        editor_session=session,
        path=path,
    )
    spec = notebook_api.PricingModelSpec(
        name="HOME_FREQ",
        label="Home frequency",
        target="target",
        model_type="superglm_poisson",
        deployment_slot="HOME_FREQ_UAT",
        features=("region", "brand", "x"),
        dataset_name="home_frequency_frame",
        source_system="test",
        pk_columns=("policy_id",),
        data_as_of_column="data_as_of",
    )
    registered = notebook_api.RegisteredModel(
        model_id=1,
        config=SimpleNamespace(model_name="HOME_FREQ"),
        source_root=tmp_path,
        spec=spec,
    )
    loaded = notebook_api.load_level_groupings(path, frame=frame, model=registered)

    assert exported.source_manifest_id == "manifest-20260630"
    assert set(loaded) == {"brand", "region"}

    candidate.technical["model_kind"] = "EDITOR_EDIT"
    with pytest.raises(ValueError, match="RAW candidate"):
        notebook_api.export_level_groupings(
            candidate,
            editor_session=session,
            path=tmp_path / "wrong.joblib",
        )
