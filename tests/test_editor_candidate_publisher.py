from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.spec import ApprovedModelBuild
from pricing_pipeline.publishing.lifecycle import PublishResult

EDITOR_CONFIG = SimpleNamespace(
    model_name="HOME_FREQ",
    deployment_slot="HOME_FREQ_UAT",
    target_name="claim_count",
    model_type="superglm_poisson",
)


def _editor_build(tmp_path, *, workbook_path=None, **overrides) -> ApprovedModelBuild:
    workbook_path = workbook_path or tmp_path / "rating_tables.xlsx"
    values = {
        "model_id": 17,
        "model_name": "HOME_FREQ",
        "model_version": "v4",
        "model_type": "superglm_poisson",
        "model_kind": "EDITOR_EDIT",
        "target_name": "claim_count",
        "deployment_slot": "HOME_FREQ_UAT",
        "manifest_id": "manifest-1",
        "split_set_id": "split-1",
        "export_id": "editor__submission_1",
        "rating_workbook_path": str(workbook_path),
        "rating_workbook_sha256": "a" * 64,
        "created_by": "analyst@example.test",
        "publication_receipt_path": str(tmp_path / "publication_receipt.json"),
        "publication_receipt_sha256": "c" * 64,
        "candidate_artifact_path": str(tmp_path / "candidate_bundle.joblib"),
        "candidate_artifact_sha256": "d" * 64,
        "candidate_artifact_format": "superglm-candidate-joblib-v2",
        "candidate_artifact_size_bytes": 321,
        "candidate_python_version": "3.14.4",
        "candidate_superglm_version": "0.11.0",
        "model_source_sha256": "b" * 64,
        "model_frame_sha256": "f" * 64,
        "metrics": {"editor_training_deviance_delta": 0.009},
        "metric_scopes": {"editor_training_deviance_delta": "editor_training_parent"},
    }
    values.update(overrides)
    return ApprovedModelBuild(**values)


def test_editor_export_carries_only_completed_build_and_editor_publication_values():
    from dataclasses import fields

    from pricing_pipeline.publishing.editor_candidate import EditorExport

    assert {field.name for field in fields(EditorExport)} == {
        "completed_build",
        "publication_receipt",
        "revision_metadata",
        "edited_model",
        "bundle",
    }


def test_editor_publisher_creates_child_and_derived_run(monkeypatch, tmp_path):
    from pricing_pipeline.publishing import editor_candidate

    workbook_path = tmp_path / "rating_tables.xlsx"
    workbook_path.write_bytes(b"editor workbook")

    submission = SimpleNamespace(
        path=str(tmp_path / "submission.json"),
        sha256="a" * 64,
        submission_id="submission-1",
        model_name="HOME_FREQ",
        deployment_slot="HOME_FREQ_UAT",
        source_package_version=7,
        parent_rate_package_id=107,
        parent_model_run_id=907,
        manifest_id="manifest-1",
        split_set_id="split-1",
        reason="Market calibration",
        claimed_identity="prototype-local-not-authenticated",
        model_source_sha256="b" * 64,
    )
    parent = SimpleNamespace(
        model_id=17,
        model_name="HOME_FREQ",
        model_version="v4",
        effective_from=None,
        effective_to=None,
        config=SimpleNamespace(
            model_name="HOME_FREQ",
            target_name="claim_count",
            model_type="superglm_poisson",
        ),
    )
    build = _editor_build(
        tmp_path,
        workbook_path=workbook_path,
        rating_workbook_sha256=editor_candidate.sha256_file(workbook_path),
    )
    fingerprinted_build = build.model_copy(update={"model_equivalence_sha256": "f" * 64})
    exported = SimpleNamespace(
        completed_build=build,
        publication_receipt=object(),
        revision_metadata={
            "claimed_identity": "prototype-local-not-authenticated",
            "kind": "SUPERGLM_EDITOR",
        },
        edited_model=object(),
        bundle=object(),
    )
    calls = []
    allowed_roots = []
    publish_connection = object()
    publication_is_active = False
    monkeypatch.setattr(
        editor_candidate,
        "load_verified_submission",
        lambda path, digest, **kwargs: submission,
    )

    def fake_load_parent_candidate(
        engine,
        loaded_submission,
        *,
        allowed_root,
        model_config,
    ):
        allowed_roots.append(("parent", allowed_root))
        assert model_config is EDITOR_CONFIG
        return parent

    def fake_export_edited_model(
        loaded_parent,
        loaded_submission,
        *,
        created_by,
        allowed_root,
        write_dir,
        published_dir,
    ):
        allowed_roots.append(("edited", allowed_root))
        assert created_by == "analyst@example.test"
        return exported

    monkeypatch.setattr(editor_candidate, "load_parent_candidate", fake_load_parent_candidate)
    monkeypatch.setattr(editor_candidate, "export_edited_model", fake_export_edited_model)
    monkeypatch.setattr(
        editor_candidate,
        "ensure_model_equivalence",
        lambda completed_build, **kwargs: fingerprinted_build,
    )
    monkeypatch.setattr(
        editor_candidate,
        "find_equivalent_publication",
        lambda engine, *, build: None,
    )
    monkeypatch.setattr(
        editor_candidate,
        "_resolve_existing_editor_publication",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        editor_candidate,
        "stage_rating_export",
        lambda engine, **kwargs: calls.append(("stage", kwargs["created_by"])) or "e" * 64,
    )

    def fake_publish_rating_package(engine, **kwargs):
        nonlocal publication_is_active
        calls.append(("publish", kwargs))
        publication_is_active = True
        try:
            lineage_writer = kwargs.get("package_lineage_writer")
            model_run_id = (
                None if lineage_writer is None else lineage_writer(publish_connection, 108)
            )
        finally:
            publication_is_active = False
        return PublishResult(
            mlflow_run_id="",
            export_id=build.export_id,
            rate_package_id=108,
            package_version=8,
            rating_workbook_path=build.rating_workbook_path,
            model_run_id=model_run_id,
        )

    def fake_record_model_run(engine, *, connection, **kwargs):
        assert engine is None
        assert publication_is_active, "lineage must execute inside package publication"
        assert connection is publish_connection
        calls.append(("lineage", kwargs))
        return 908

    monkeypatch.setattr(
        editor_candidate,
        "publish_rating_package",
        fake_publish_rating_package,
    )
    monkeypatch.setattr(
        editor_candidate,
        "record_model_run",
        fake_record_model_run,
    )

    result = editor_candidate.publish_editor_submission(
        object(),
        settings=Settings(workbench_artifact_root=tmp_path),
        submission_path=submission.path,
        submission_sha256=submission.sha256,
        dag_id="pricing_publish_editor_candidate",
        airflow_run_id="manual__submission-1",
        created_by="analyst@example.test",
        model_config=EDITOR_CONFIG,
    )

    assert result.parent_rate_package_id == submission.parent_rate_package_id
    assert result.rate_package_id == 108
    assert result.package_version == 8
    assert result.model_run_id == 908
    assert [name for name, _value in calls] == ["stage", "publish", "lineage"]
    publish_kwargs = calls[1][1]
    assert publish_kwargs["parent_rate_package_id"] == submission.parent_rate_package_id
    assert publish_kwargs["revision_metadata"] == {
        "claimed_identity": "prototype-local-not-authenticated",
        "kind": "SUPERGLM_EDITOR",
        "published_by": "analyst@example.test",
    }
    assert "package_status" not in publish_kwargs
    assert callable(publish_kwargs["package_lineage_writer"])
    assert publish_kwargs["expected_staged_metadata"] == {
        "export_id": build.export_id,
        "model_id": parent.model_id,
        "model_name": parent.model_name,
        "model_version": parent.model_version,
        "effective_from_date": None,
        "effective_to_date": None,
        "source_file": str(Path(build.rating_workbook_path).resolve()),
        "publication_receipt_sha256": build.publication_receipt_sha256,
        "staging_content_sha256": "e" * 64,
        "model_equivalence_sha256": "f" * 64,
    }
    lineage_kwargs = calls[2][1]
    assert lineage_kwargs["rate_package_id"] == result.rate_package_id
    assert lineage_kwargs["parent_model_run_id"] == submission.parent_model_run_id
    assert lineage_kwargs["build"] is fingerprinted_build
    assert build.manifest_id == submission.manifest_id
    assert build.split_set_id == submission.split_set_id
    assert build.rating_workbook_sha256 == editor_candidate.sha256_file(workbook_path)
    assert build.candidate_artifact_sha256 == "d" * 64
    assert allowed_roots == [("parent", tmp_path), ("edited", tmp_path)]


def test_existing_editor_publication_returns_before_artifact_write(monkeypatch, tmp_path):
    from pricing_pipeline.publishing import editor_candidate

    submission_path = tmp_path / "submission" / "submission.json"
    submission_path.parent.mkdir()
    submission = SimpleNamespace(
        path=str(submission_path),
        sha256="a" * 64,
        submission_id="submission-1",
        model_name="HOME_FREQ",
        deployment_slot="HOME_FREQ_UAT",
        parent_rate_package_id=107,
    )
    existing = editor_candidate.EditorPublicationResult(
        submission_id="submission-1",
        model_name="HOME_FREQ",
        parent_rate_package_id=107,
        rate_package_id=108,
        package_version=8,
        model_run_id=908,
        package_status="PUBLISHED",
        was_existing=True,
    )
    monkeypatch.setattr(
        editor_candidate,
        "load_verified_submission",
        lambda *args, **kwargs: submission,
    )
    monkeypatch.setattr(
        editor_candidate,
        "_resolve_existing_editor_publication",
        lambda *args, **kwargs: existing,
        raising=False,
    )
    monkeypatch.setattr(
        editor_candidate,
        "load_parent_candidate",
        lambda *args, **kwargs: pytest.fail("existing publication must return first"),
    )
    monkeypatch.setattr(
        editor_candidate,
        "export_edited_model",
        lambda *args, **kwargs: pytest.fail("existing publication must not write files"),
    )

    result = editor_candidate.publish_editor_submission(
        object(),
        settings=Settings(workbench_artifact_root=tmp_path),
        submission_path=str(submission_path),
        submission_sha256="a" * 64,
        dag_id="pricing_publish_editor_candidate",
        airflow_run_id="manual__submission-1",
        created_by="analyst@example.test",
        model_config=EDITOR_CONFIG,
    )

    assert result == existing


@pytest.mark.parametrize(
    ("stored_carry_forward", "stored_parent_ids", "is_compatible"),
    [
        pytest.param(True, (107, 907), False, id="different-policy"),
        pytest.param(False, (106, 906), False, id="different-parent"),
        pytest.param(False, (107, 907), True, id="same-policy-and-parent"),
    ],
)
def test_manual_equivalence_requires_matching_immutable_policy_lineage(
    monkeypatch,
    tmp_path,
    stored_carry_forward,
    stored_parent_ids,
    is_compatible,
):
    from dataclasses import replace

    from pricing_pipeline.modeling.manual_adjustment import (
        ManualAdjustmentPolicy,
        ManualAdjustmentRule,
    )
    from pricing_pipeline.publishing import editor_candidate

    staging_dir = tmp_path / ".staging" / "attempt"
    final_dir = tmp_path / "attempts" / "attempt"
    staging_dir.mkdir(parents=True)
    final_dir.parent.mkdir(parents=True)
    workbook = staging_dir / "rating_tables.xlsx"
    workbook.write_bytes(b"manual rating workbook")

    requested_policy = ManualAdjustmentPolicy(
        name="market adjustment",
        version=1,
        reason="Approved market response",
        carry_forward=False,
        rules=(
            ManualAdjustmentRule.multiply_levels(
                "segment",
                ["B"],
                1.05,
                reason="Selected segment uplift",
            ),
        ),
    )
    stored_policy = replace(requested_policy, carry_forward=stored_carry_forward)
    stored_parent_rate_package_id, stored_parent_model_run_id = stored_parent_ids

    def edit_metadata(policy):
        return {
            "manual_adjustment_policy": policy.to_payload(),
            "manual_adjustment_policy_sha256": policy.sha256,
        }

    requested_edit_metadata = edit_metadata(requested_policy)
    submission = SimpleNamespace(
        submission_id="submission-new-policy",
        model_kind="MANUAL_EDIT",
        parent_rate_package_id=107,
        parent_model_run_id=907,
        edit_metadata=requested_edit_metadata,
    )
    parent = SimpleNamespace(effective_to=None)
    build = _editor_build(
        tmp_path,
        workbook_path=final_dir / workbook.name,
        rating_workbook_sha256=editor_candidate.sha256_file(workbook),
        model_kind="MANUAL_EDIT",
    )
    fingerprinted_build = build.model_copy(update={"model_equivalence_sha256": "f" * 64})
    exported = SimpleNamespace(
        completed_build=build,
        revision_metadata={
            "kind": "SUPERGLM_MANUAL_EDIT",
            "parent_rate_package_id": submission.parent_rate_package_id,
            "parent_model_run_id": submission.parent_model_run_id,
            "edit_metadata": requested_edit_metadata,
        },
    )
    equivalent = SimpleNamespace(
        model_name="HOME_FREQ",
        parent_rate_package_id=stored_parent_rate_package_id,
        rate_package_id=108,
        package_version=8,
        model_run_id=908,
        package_status="PUBLISHED",
    )
    stored_row = {
        "parent_rate_package_id": stored_parent_rate_package_id,
        "parent_model_run_id": stored_parent_model_run_id,
        "revision_metadata_json": json.dumps(
            {
                "kind": "SUPERGLM_MANUAL_EDIT",
                "parent_rate_package_id": stored_parent_rate_package_id,
                "parent_model_run_id": stored_parent_model_run_id,
                "edit_metadata": edit_metadata(stored_policy),
            }
        ),
    }

    class Result:
        def mappings(self):
            return self

        def one_or_none(self):
            return stored_row

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, statement, params):
            assert params == {
                "rate_package_id": equivalent.rate_package_id,
                "model_run_id": equivalent.model_run_id,
            }
            return Result()

    class Engine:
        def connect(self):
            return Connection()

    monkeypatch.setattr(
        editor_candidate,
        "schema_names_from_connectable",
        lambda engine: SimpleNamespace(pricing="pricing"),
    )
    monkeypatch.setattr(
        editor_candidate,
        "load_parent_candidate",
        lambda *args, **kwargs: parent,
    )
    monkeypatch.setattr(
        editor_candidate,
        "export_edited_model",
        lambda *args, **kwargs: exported,
    )
    monkeypatch.setattr(
        editor_candidate,
        "ensure_model_equivalence",
        lambda completed_build, **kwargs: fingerprinted_build,
    )
    monkeypatch.setattr(
        editor_candidate,
        "find_equivalent_publication",
        lambda engine, *, build: equivalent,
    )
    monkeypatch.setattr(
        editor_candidate,
        "release_unused_model_version_reservation",
        lambda *args, **kwargs: None,
    )

    def publish():
        return editor_candidate._publish_new_editor_submission(
            Engine(),
            submission=submission,
            allowed_root=tmp_path,
            attempt=editor_candidate.EditorPublicationAttempt(staging_dir, final_dir),
            dag_id="pricing_publish_editor_candidate",
            airflow_run_id="manual__submission-new-policy",
            created_by="publisher@example.test",
            model_config=EDITOR_CONFIG,
        )

    if is_compatible:
        result = publish()
        assert result.rate_package_id == equivalent.rate_package_id
        assert result.deduplicated is True
    else:
        with pytest.raises(
            editor_candidate.EditorSubmissionError,
            match="equivalent MANUAL_EDIT package has incompatible immutable policy lineage",
        ):
            publish()


def test_editor_retry_rejects_submission_slot_mismatch_before_existing_lookup(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline.publishing import editor_candidate

    submission_path = tmp_path / "submission" / "submission.json"
    submission_path.parent.mkdir()
    submission = SimpleNamespace(
        path=str(submission_path),
        deployment_slot="HOME_FREQ_PRODUCTION",
    )
    monkeypatch.setattr(
        editor_candidate,
        "load_verified_submission",
        lambda *args, **kwargs: submission,
    )
    monkeypatch.setattr(
        editor_candidate,
        "_resolve_existing_editor_publication",
        lambda *args, **kwargs: pytest.fail("slot mismatch reached existing-publication lookup"),
    )

    with pytest.raises(editor_candidate.EditorSubmissionError, match="deployment_slot"):
        editor_candidate.publish_editor_submission(
            object(),
            settings=Settings(workbench_artifact_root=tmp_path),
            submission_path=str(submission_path),
            submission_sha256="a" * 64,
            dag_id="pricing_publish_editor_candidate",
            airflow_run_id="manual__submission-1",
            created_by="analyst@example.test",
            model_config=EDITOR_CONFIG,
        )


def test_existing_editor_publication_verifies_committed_candidate_bytes(
    monkeypatch,
    tmp_path,
):
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing import editor_candidate
    from pricing_pipeline.workbench.artifacts import CandidateBundle, save_candidate_bundle
    from pricing_pipeline.workbench.submission import EditorSubmissionError

    bundle = CandidateBundle(
        fitted_model={"model": "edited"},
        X=pd.DataFrame({"x": [1.0]}),
        y=np.array([0.0]),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={},
        model_name="HOME_FREQ",
        model_version="20260603",
        export_id="editor__submission_1",
        manifest_id="manifest-1",
        split_set_id="split-1",
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "NONE"},
        model_frame_sha256="f" * 64,
    )
    artifact = save_candidate_bundle(bundle, tmp_path / "candidate_bundle.joblib")
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_bytes(b"editor workbook")
    row = {
        "model_name": "HOME_FREQ",
        "model_version": "20260603",
        "export_id": "editor__submission_1",
        "rate_package_id": 108,
        "package_version": 8,
        "package_status": "PUBLISHED",
        "parent_rate_package_id": 107,
        "model_run_id": 908,
        "parent_model_run_id": 907,
        "run_status": "SUCCESS",
        "rating_workbook_path": str(workbook),
        "rating_workbook_sha256": editor_candidate.sha256_file(workbook),
        "candidate_artifact_path": artifact.path,
        "candidate_artifact_sha256": artifact.sha256,
        "candidate_artifact_format": artifact.format,
        "candidate_artifact_size_bytes": artifact.size_bytes,
        "candidate_python_version": artifact.python_version,
        "candidate_superglm_version": artifact.superglm_version,
        "manifest_id": bundle.manifest_id,
        "split_set_id": bundle.split_set_id,
        "model_source_sha256": bundle.model_source_sha256,
        "model_frame_sha256": bundle.model_frame_sha256,
    }

    class Rows:
        def mappings(self):
            return self

        def all(self):
            return [row]

    class Connection:
        def execute(self, statement, params):
            assert "manifest.model_frame_sha256" in str(statement)
            assert "DATASET_MANIFEST AS manifest" in str(statement)
            return Rows()

    class Begin:
        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class Engine:
        def begin(self):
            return Begin()

    monkeypatch.setattr(
        editor_candidate,
        "schema_names_from_connectable",
        lambda engine: SimpleNamespace(pricing="pricing", mlops="mlops"),
    )
    submission = SimpleNamespace(
        submission_id="submission-1",
        model_name="HOME_FREQ",
        model_kind="EDITOR_EDIT",
        path=str(tmp_path / "submission.json"),
        sha256="a" * 64,
        parent_rate_package_id=107,
        parent_model_run_id=907,
        manifest_id=bundle.manifest_id,
        split_set_id=bundle.split_set_id,
        model_source_sha256=bundle.model_source_sha256,
        edit_metadata=None,
    )
    row["revision_metadata_json"] = json.dumps(
        {
            "kind": "SUPERGLM_EDITOR",
            "submission_id": submission.submission_id,
            "submission_path": submission.path,
            "submission_sha256": submission.sha256,
            "parent_rate_package_id": submission.parent_rate_package_id,
            "parent_model_run_id": submission.parent_model_run_id,
        }
    )

    result = editor_candidate._resolve_existing_editor_publication(
        Engine(),
        submission,
        allowed_root=tmp_path,
    )

    assert result is not None
    assert result.model_run_id == 908
    assert result.was_existing is True

    for field_name in ("model_version", "export_id"):
        original = row[field_name]
        row[field_name] = f"wrong-{field_name}"
        with pytest.raises(EditorSubmissionError, match=field_name):
            editor_candidate._resolve_existing_editor_publication(
                Engine(),
                submission,
                allowed_root=tmp_path,
            )
        row[field_name] = original

    row["parent_model_run_id"] = 999
    with pytest.raises(EditorSubmissionError, match="parent_model_run_id"):
        editor_candidate._resolve_existing_editor_publication(
            Engine(),
            submission,
            allowed_root=tmp_path,
        )
    row["parent_model_run_id"] = submission.parent_model_run_id

    workbook.write_bytes(b"overwritten")
    with pytest.raises(EditorSubmissionError, match="rating workbook"):
        editor_candidate._resolve_existing_editor_publication(
            Engine(),
            submission,
            allowed_root=tmp_path,
        )
    workbook.write_bytes(b"editor workbook")

    Path(artifact.path).write_bytes(b"overwritten")
    with pytest.raises(EditorSubmissionError, match="failed verification"):
        editor_candidate._resolve_existing_editor_publication(
            Engine(),
            submission,
            allowed_root=tmp_path,
        )


@pytest.mark.parametrize("changed_identity", ["submission-sha", "manual-policy"])
def test_existing_manual_publication_rejects_changed_signed_submission_before_artifact_load(
    monkeypatch,
    tmp_path,
    changed_identity,
):
    from dataclasses import replace

    from pricing_pipeline.modeling.manual_adjustment import (
        ManualAdjustmentPolicy,
        ManualAdjustmentRule,
    )
    from pricing_pipeline.publishing import editor_candidate

    policy = ManualAdjustmentPolicy(
        name="market adjustment",
        version=1,
        reason="Approved market response",
        carry_forward=True,
        rules=(
            ManualAdjustmentRule.multiply_levels(
                "segment",
                ["B"],
                1.05,
                reason="Selected segment uplift",
            ),
        ),
    )

    def edit_metadata(value):
        return {
            "manual_adjustment_policy": value.to_payload(),
            "manual_adjustment_policy_sha256": value.sha256,
        }

    stored_edit_metadata = edit_metadata(policy)
    submission = SimpleNamespace(
        submission_id="submission-1",
        model_name="HOME_FREQ",
        model_kind="MANUAL_EDIT",
        path=str(tmp_path / "submission.json"),
        sha256="a" * 64,
        parent_rate_package_id=107,
        parent_model_run_id=907,
        manifest_id="manifest-1",
        split_set_id="split-1",
        model_source_sha256="b" * 64,
        edit_metadata=stored_edit_metadata,
    )
    revision_metadata = {
        "kind": "SUPERGLM_MANUAL_EDIT",
        "submission_id": submission.submission_id,
        "submission_path": submission.path,
        "submission_sha256": submission.sha256,
        "parent_rate_package_id": submission.parent_rate_package_id,
        "parent_model_run_id": submission.parent_model_run_id,
        "edit_metadata": stored_edit_metadata,
    }
    if changed_identity == "submission-sha":
        submission.sha256 = "e" * 64
    else:
        submission.edit_metadata = edit_metadata(replace(policy, carry_forward=False))

    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_bytes(b"manual workbook")
    row = {
        "model_name": submission.model_name,
        "model_version": "20260603",
        "export_id": "manual__submission_1",
        "rate_package_id": 108,
        "package_version": 8,
        "package_status": "PUBLISHED",
        "parent_rate_package_id": 107,
        "model_run_id": 908,
        "parent_model_run_id": 907,
        "run_status": "SUCCESS",
        "model_kind": "MANUAL_EDIT",
        "rating_workbook_path": str(workbook),
        "rating_workbook_sha256": editor_candidate.sha256_file(workbook),
        "candidate_artifact_path": str(tmp_path / "candidate.joblib"),
        "candidate_artifact_sha256": "d" * 64,
        "candidate_artifact_format": "superglm-candidate-joblib-v2",
        "candidate_artifact_size_bytes": 321,
        "candidate_python_version": "3.14.4",
        "candidate_superglm_version": "0.13.0",
        "manifest_id": submission.manifest_id,
        "split_set_id": submission.split_set_id,
        "model_source_sha256": submission.model_source_sha256,
        "model_frame_sha256": "f" * 64,
        "revision_metadata_json": json.dumps(revision_metadata),
    }

    class Rows:
        def mappings(self):
            return self

        def all(self):
            return [row]

    class Connection:
        def execute(self, statement, params):
            return Rows()

    class Begin:
        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class Engine:
        def begin(self):
            return Begin()

    monkeypatch.setattr(
        editor_candidate,
        "schema_names_from_connectable",
        lambda engine: SimpleNamespace(pricing="pricing", mlops="mlops"),
    )
    monkeypatch.setattr(
        editor_candidate,
        "load_candidate_bundle",
        lambda *args, **kwargs: pytest.fail(
            "changed signed submission reached committed artifact loading"
        ),
    )

    with pytest.raises(
        editor_candidate.EditorSubmissionError,
        match="existing editor publication signed submission metadata does not match",
    ):
        editor_candidate._resolve_existing_editor_publication(
            Engine(),
            submission,
            allowed_root=tmp_path,
        )


@pytest.mark.parametrize("lineage_owner", ["sql", "bundle"])
@pytest.mark.parametrize(
    ("field_name", "different_value"),
    [
        ("manifest_id", "different-manifest"),
        ("split_set_id", "different-split"),
        ("model_source_sha256", "c" * 64),
        ("model_frame_sha256", None),
        ("model_frame_sha256", "e" * 64),
    ],
)
def test_existing_editor_publication_rejects_mismatched_lineage(
    monkeypatch,
    tmp_path,
    lineage_owner,
    field_name,
    different_value,
):
    from pricing_pipeline.publishing import editor_candidate
    from pricing_pipeline.workbench.submission import EditorSubmissionError

    expected = {
        "manifest_id": "manifest-1",
        "split_set_id": "split-1",
        "model_source_sha256": "b" * 64,
        "model_frame_sha256": "f" * 64,
    }
    identity = {
        "model_name": "HOME_FREQ",
        "model_version": "20260603",
        "export_id": "editor__submission_1",
    }
    sql_lineage = {**expected, **identity}
    bundle_lineage = {**expected, **identity}
    if lineage_owner == "sql":
        sql_lineage[field_name] = different_value
    else:
        bundle_lineage[field_name] = different_value
    workbook = tmp_path / "rating_tables.xlsx"
    workbook.write_bytes(b"editor workbook")

    row = {
        "model_name": "HOME_FREQ",
        "rate_package_id": 108,
        "package_version": 8,
        "package_status": "PUBLISHED",
        "parent_rate_package_id": 107,
        "model_run_id": 908,
        "parent_model_run_id": 907,
        "run_status": "SUCCESS",
        "rating_workbook_path": str(workbook),
        "rating_workbook_sha256": editor_candidate.sha256_file(workbook),
        "candidate_artifact_path": str(tmp_path / "candidate.joblib"),
        "candidate_artifact_sha256": "d" * 64,
        "candidate_artifact_format": "superglm-candidate-joblib-v2",
        "candidate_artifact_size_bytes": 321,
        "candidate_python_version": "3.14.4",
        "candidate_superglm_version": "0.11.0",
        **sql_lineage,
    }

    class Rows:
        def mappings(self):
            return self

        def all(self):
            return [row]

    class Connection:
        def execute(self, statement, params):
            return Rows()

    class Begin:
        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class Engine:
        def begin(self):
            return Begin()

    monkeypatch.setattr(
        editor_candidate,
        "schema_names_from_connectable",
        lambda engine: SimpleNamespace(pricing="pricing", mlops="mlops"),
    )
    monkeypatch.setattr(
        editor_candidate,
        "load_candidate_bundle",
        lambda *args, **kwargs: SimpleNamespace(**bundle_lineage),
    )
    submission = SimpleNamespace(
        submission_id="submission-1",
        model_name="HOME_FREQ",
        model_kind="EDITOR_EDIT",
        path=str(tmp_path / "submission.json"),
        sha256="a" * 64,
        parent_rate_package_id=107,
        parent_model_run_id=907,
        edit_metadata=None,
        **expected,
    )
    row["revision_metadata_json"] = json.dumps(
        {
            "kind": "SUPERGLM_EDITOR",
            "submission_id": submission.submission_id,
            "submission_path": submission.path,
            "submission_sha256": submission.sha256,
            "parent_rate_package_id": submission.parent_rate_package_id,
            "parent_model_run_id": submission.parent_model_run_id,
        }
    )

    expected_layer = (
        "model_frame_sha256"
        if field_name == "model_frame_sha256"
        else "SQL lineage"
        if lineage_owner == "sql"
        else "bundle lineage"
    )
    with pytest.raises(EditorSubmissionError, match=expected_layer):
        editor_candidate._resolve_existing_editor_publication(
            Engine(),
            submission,
            allowed_root=tmp_path,
        )


def test_failed_editor_publication_removes_only_its_unique_attempt(monkeypatch, tmp_path):
    from pricing_pipeline.publishing import editor_candidate

    submission_path = tmp_path / "submission" / "submission.json"
    submission_path.parent.mkdir()
    submission = SimpleNamespace(
        path=str(submission_path),
        sha256="a" * 64,
        submission_id="submission-1",
        model_name="HOME_FREQ",
        deployment_slot="HOME_FREQ_UAT",
        source_package_version=7,
        parent_rate_package_id=107,
        parent_model_run_id=907,
        manifest_id="manifest-1",
        split_set_id="split-1",
        model_source_sha256="b" * 64,
    )
    parent = SimpleNamespace(
        model_id=17,
        model_name="HOME_FREQ",
        model_version="v4",
        effective_from=None,
        effective_to=None,
        config=SimpleNamespace(
            target_name="ClaimCount",
            model_type="Poisson",
        ),
    )
    committed_artifact = tmp_path / "committed" / "candidate_bundle.joblib"
    committed_artifact.parent.mkdir()
    committed_artifact.write_bytes(b"committed bytes")
    attempt_paths = []

    def fake_export(
        loaded_parent,
        loaded_submission,
        *,
        created_by,
        allowed_root,
        write_dir,
        published_dir,
    ):
        write_dir = Path(write_dir)
        published_dir = Path(published_dir)
        candidate_path = write_dir / "candidate_bundle.joblib"
        candidate_path.write_bytes(b"retry bytes")
        workbook_path = write_dir / "rating_tables.xlsx"
        workbook_path.write_bytes(b"rating workbook")
        attempt_paths.append((write_dir, published_dir))
        build = _editor_build(
            tmp_path,
            workbook_path=published_dir / "rating_tables.xlsx",
            rating_workbook_sha256=editor_candidate.sha256_file(workbook_path),
            created_by=created_by,
            publication_receipt_path=str(published_dir / "publication_receipt.json"),
            candidate_artifact_path=str(published_dir / "candidate_bundle.joblib"),
            candidate_artifact_size_bytes=11,
            metrics={},
            metric_scopes={},
        )
        return SimpleNamespace(
            completed_build=build,
            publication_receipt=object(),
            revision_metadata={"kind": "SUPERGLM_EDITOR"},
            edited_model=object(),
            bundle=object(),
        )

    monkeypatch.setattr(
        editor_candidate,
        "load_verified_submission",
        lambda *args, **kwargs: submission,
    )
    monkeypatch.setattr(
        editor_candidate,
        "_resolve_existing_editor_publication",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        editor_candidate,
        "load_parent_candidate",
        lambda *args, **kwargs: parent,
    )
    monkeypatch.setattr(editor_candidate, "export_edited_model", fake_export)
    monkeypatch.setattr(
        editor_candidate,
        "ensure_model_equivalence",
        lambda completed_build, **kwargs: completed_build.model_copy(
            update={"model_equivalence_sha256": "f" * 64}
        ),
    )
    monkeypatch.setattr(
        editor_candidate,
        "find_equivalent_publication",
        lambda engine, *, build: None,
    )
    monkeypatch.setattr(editor_candidate, "stage_rating_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        editor_candidate,
        "publish_rating_package",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected SQL failure")),
    )

    for _attempt_no in range(2):
        with pytest.raises(RuntimeError, match="injected SQL failure"):
            editor_candidate.publish_editor_submission(
                object(),
                settings=Settings(workbench_artifact_root=tmp_path),
                submission_path=str(submission_path),
                submission_sha256="a" * 64,
                dag_id="pricing_publish_editor_candidate",
                airflow_run_id="manual__submission-1",
                created_by="analyst@example.test",
                model_config=EDITOR_CONFIG,
            )

    assert len(attempt_paths) == 2
    assert attempt_paths[0] != attempt_paths[1]
    for write_dir, published_dir in attempt_paths:
        assert not write_dir.exists()
        assert not published_dir.exists()
    assert committed_artifact.read_bytes() == b"committed bytes"


def test_editor_publication_rejects_workbook_mutated_during_staging(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline.publishing import editor_candidate

    staging_dir = tmp_path / ".staging" / "attempt"
    final_dir = tmp_path / "attempts" / "attempt"
    staging_dir.mkdir(parents=True)
    final_dir.parent.mkdir(parents=True)
    submission = SimpleNamespace(submission_id="submission-1")
    parent = SimpleNamespace(
        model_id=17,
        model_name="HOME_FREQ",
        model_version="v4",
        effective_from=None,
        effective_to=None,
        config=SimpleNamespace(target_name="ClaimNb", model_type="Poisson"),
    )

    def export_model(*args, **kwargs):
        workbook = staging_dir / "rating_tables.xlsx"
        workbook.write_bytes(b"original")
        return SimpleNamespace(
            completed_build=_editor_build(
                tmp_path,
                workbook_path=final_dir / "rating_tables.xlsx",
                rating_workbook_sha256=editor_candidate.sha256_file(workbook),
                created_by=kwargs["created_by"],
            ),
        )

    def mutate_workbook(*args, **kwargs):
        (final_dir / "rating_tables.xlsx").write_bytes(b"mutated during staging")
        return "a" * 64

    monkeypatch.setattr(editor_candidate, "load_parent_candidate", lambda *args, **kwargs: parent)
    monkeypatch.setattr(editor_candidate, "export_edited_model", export_model)
    monkeypatch.setattr(
        editor_candidate,
        "ensure_model_equivalence",
        lambda completed_build, **kwargs: completed_build.model_copy(
            update={"model_equivalence_sha256": "f" * 64}
        ),
    )
    monkeypatch.setattr(
        editor_candidate,
        "find_equivalent_publication",
        lambda engine, *, build: None,
    )
    monkeypatch.setattr(editor_candidate, "stage_rating_export", mutate_workbook)
    monkeypatch.setattr(
        editor_candidate,
        "publish_rating_package",
        lambda *args, **kwargs: pytest.fail("mutated workbook reached package publication"),
    )

    with pytest.raises(editor_candidate.EditorSubmissionError, match="changed during staging"):
        editor_candidate._publish_new_editor_submission(
            object(),
            submission=submission,
            allowed_root=tmp_path,
            attempt=editor_candidate.EditorPublicationAttempt(staging_dir, final_dir),
            dag_id="pricing_publish_editor_candidate",
            airflow_run_id="manual__submission-1",
            created_by="publisher@example.test",
            model_config=EDITOR_CONFIG,
        )


def test_editor_export_writes_staging_bytes_but_persists_final_attempt_paths(
    monkeypatch,
    tmp_path,
):
    import joblib
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing import editor_candidate
    from pricing_pipeline.publishing.superglm_publication_receipt import (
        OffsetExportContract,
    )
    from pricing_pipeline.workbench.artifacts import CandidateBundle

    submission_dir = tmp_path / "submission"
    write_dir = submission_dir / "published" / ".staging" / "attempt-a"
    final_dir = submission_dir / "published" / "attempts" / "attempt-a"
    write_dir.mkdir(parents=True)
    term = pd.Series([36.0], name="Term")
    rating_weight = pd.Series([2.0], name="RatingWeight")
    bundle = CandidateBundle(
        fitted_model={"model": "parent"},
        X=pd.DataFrame({"x": [1.0]}),
        y=np.array([0.0]),
        sample_weight=None,
        offset=np.log(term / 12.0),
        offset_source=term,
        export_weight=rating_weight,
        cv_report={
            "mean_scores": {"deviance": 0.48},
            "pooled_scores": {"deviance": 0.47},
            "std_scores": {"deviance": 0.03},
            "oof_coverage": 1.0,
        },
        model_name="HOME_FREQ",
        model_version="20260603",
        export_id="parent-export",
        manifest_id="manifest-1",
        split_set_id="split-1",
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        model_frame_sha256="f" * 64,
        offset_contract=OffsetExportContract(
            handling="EXPORTED_FACTOR",
            source_factor_name="Term",
            published_factor_name="Term",
            source_name="Term",
            label="log(Term / 12)",
        ),
        offset_source_name="Term",
        export_weight_name="RatingWeight",
    )
    parent = SimpleNamespace(
        model_id=17,
        model_name="HOME_FREQ",
        model_version="20260603",
        effective_from=None,
        config=SimpleNamespace(
            model_type="superglm_poisson",
            target_name="claim_count",
        ),
        bundle=bundle,
        champion=editor_candidate.ChampionSnapshot(
            deployment_slot="HOME_FREQ_UAT",
            rate_package_id=None,
            bundle=None,
            unavailable_reason="no champion is deployed in HOME_FREQ_UAT",
        ),
    )
    submission = SimpleNamespace(
        format="superglm-editor-submission-v2",
        submission_id="submission-1",
        deployment_slot="HOME_FREQ_UAT",
        manifest_id="manifest-1",
        split_set_id="split-1",
        model_source_sha256="b" * 64,
        reason="Market calibration",
        claimed_identity="analyst@example.test",
        parent_rate_package_id=107,
        parent_model_run_id=907,
        path=str(submission_dir / "submission.json"),
        sha256="c" * 64,
        editor_session_path=str(submission_dir / "editor_session.json"),
        editor_session_sha256="d" * 64,
        editor_session_size_bytes=10,
        edited_model_path=str(submission_dir / "edited_model.joblib"),
        edited_model_sha256="e" * 64,
        edited_model_size_bytes=11,
        edited_model_format="superglm-edited-model-joblib-v1",
        edited_model_python_version="3.14.4",
        edited_model_superglm_version="0.13.0",
        baseline_candidate_sha256="f" * 64,
    )

    monkeypatch.setattr(
        editor_candidate,
        "_load_edited_model",
        lambda *args, **kwargs: {"model": "edited"},
    )

    captured_export = {}

    def fake_export_rating_tables(*args, output_path, **kwargs):
        captured_export["weight"] = args[3]
        captured_export["options"] = kwargs
        Path(output_path).write_bytes(b"workbook")

    def fake_write_receipt(receipt, path):
        Path(path).write_bytes(b"receipt")
        return "1" * 64

    monkeypatch.setattr(editor_candidate, "export_rating_tables", fake_export_rating_tables)
    monkeypatch.setattr(
        editor_candidate,
        "build_superglm_publication_receipt",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(editor_candidate, "write_publication_receipt", fake_write_receipt)
    monkeypatch.setattr(
        editor_candidate,
        "training_comparison_metrics",
        lambda *args, **kwargs: ({}, {}),
    )

    exported = editor_candidate.export_edited_model(
        parent,
        submission,
        created_by="publisher@example.test",
        allowed_root=tmp_path,
        write_dir=write_dir,
        published_dir=final_dir,
    )

    build = exported.completed_build
    assert Path(build.rating_workbook_path) == final_dir / "rating_tables.xlsx"
    assert build.rating_workbook_sha256 == editor_candidate.sha256_file(
        write_dir / "rating_tables.xlsx"
    )
    assert Path(build.publication_receipt_path) == final_dir / "publication_receipt.json"
    assert Path(build.candidate_artifact_path) == final_dir / "candidate_bundle.joblib"
    assert build.created_by == "publisher@example.test"
    assert build.manifest_id == submission.manifest_id
    assert build.split_set_id == submission.split_set_id
    assert build.model_source_sha256 == submission.model_source_sha256
    assert build.model_frame_sha256 == bundle.model_frame_sha256
    assert build.metrics == {}
    assert build.metric_scopes == {}
    assert exported.revision_metadata["baseline_cv_metrics"] == {
        "cv_mean_deviance": 0.48,
        "cv_pooled_deviance": 0.47,
        "cv_std_deviance": 0.03,
        "cv_oof_coverage": 1.0,
    }
    assert (write_dir / "rating_tables.xlsx").read_bytes() == b"workbook"
    assert (write_dir / "candidate_bundle.joblib").is_file()
    assert not final_dir.exists()
    envelope = joblib.load(write_dir / "candidate_bundle.joblib")
    assert envelope["bundle"].model_name == "HOME_FREQ"
    assert envelope["bundle"].model_version == "20260603"
    assert envelope["bundle"].export_id == "editor__submission_1"
    assert envelope["bundle"].cv_report == {}
    pd.testing.assert_series_equal(captured_export["weight"], rating_weight)
    np.testing.assert_allclose(captured_export["options"]["offset"], np.log(term / 12.0))
    pd.testing.assert_series_equal(captured_export["options"]["offset_source"], term)
    assert captured_export["options"]["offset_name"] == "Term"
    assert captured_export["options"]["offset_kind"] == "auto"
    assert exported.revision_metadata["claimed_identity"] == "analyst@example.test"
    assert exported.revision_metadata["edited_model_path"] == submission.edited_model_path
    assert exported.revision_metadata["edited_model_sha256"] == submission.edited_model_sha256
    assert (
        exported.revision_metadata["edited_model_size_bytes"] == submission.edited_model_size_bytes
    )
    assert exported.revision_metadata["edited_model_format"] == submission.edited_model_format
    assert (
        exported.revision_metadata["edited_model_python_version"]
        == submission.edited_model_python_version
    )
    assert (
        exported.revision_metadata["edited_model_superglm_version"]
        == submission.edited_model_superglm_version
    )
    assert "published_by" not in exported.revision_metadata


def test_editor_publication_lock_serializes_the_same_submission(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from pricing_pipeline.publishing.editor_candidate import _editor_publication_lock

    first_acquired = Event()
    release_first = Event()
    second_started = Event()
    second_acquired = Event()

    def first_worker():
        with _editor_publication_lock(tmp_path):
            first_acquired.set()
            assert release_first.wait(timeout=2)

    def second_worker():
        second_started.set()
        with _editor_publication_lock(tmp_path):
            second_acquired.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_worker)
        assert first_acquired.wait(timeout=2)
        second = executor.submit(second_worker)
        assert second_started.wait(timeout=2)
        assert not second_acquired.wait(timeout=0.05)
        release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert second_acquired.is_set()


def test_parent_candidate_rejects_submission_deployment_slot_mismatch_before_sql(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline.publishing import editor_candidate

    submission = SimpleNamespace(
        model_name="HOME_FREQ",
        deployment_slot="HOME_FREQ_PRODUCTION",
    )
    monkeypatch.setattr(
        editor_candidate,
        "schema_names_from_connectable",
        lambda engine: pytest.fail("deployment-slot mismatch reached SQL"),
    )

    with pytest.raises(editor_candidate.EditorSubmissionError, match="deployment_slot"):
        editor_candidate.load_parent_candidate(
            object(),
            submission,
            allowed_root=tmp_path,
            model_config=EDITOR_CONFIG,
        )


@pytest.mark.parametrize(
    "submission_relative_path",
    [
        "submission.json",
        "models/HOME_FREQ/editor/submissions/deep/submission.json",
    ],
)
@pytest.mark.parametrize("effective_from_date", ["2026-01-01", None])
def test_parent_candidate_uses_exact_configured_root_and_unambiguous_split_link(
    monkeypatch,
    tmp_path,
    submission_relative_path,
    effective_from_date,
):
    from pricing_pipeline.publishing import editor_candidate

    configured_root = tmp_path / "configured-workbench"
    candidate_path = configured_root / "models/HOME_FREQ/runs/deep/candidate.joblib"
    submission = SimpleNamespace(
        path=str(configured_root / submission_relative_path),
        model_name="HOME_FREQ",
        deployment_slot="HOME_FREQ_UAT",
        source_package_version=7,
        parent_rate_package_id=107,
        parent_model_run_id=907,
        manifest_id="manifest-1",
        split_set_id="split-1",
        baseline_candidate_path=str(candidate_path),
        baseline_candidate_sha256="a" * 64,
        model_source_sha256="b" * 64,
    )
    row = {
        "model_id": 17,
        "model_name": submission.model_name,
        "model_version": "v4",
        "run_model_version": "v4",
        "export_id": "parent-export",
        "package_version": submission.source_package_version,
        "rate_package_id": submission.parent_rate_package_id,
        "effective_from_date": effective_from_date,
        "effective_to_date": None,
        "model_run_id": submission.parent_model_run_id,
        "run_status": "SUCCESS",
        "manifest_id": submission.manifest_id,
        "split_set_id": submission.split_set_id,
        "candidate_artifact_path": submission.baseline_candidate_path,
        "candidate_artifact_sha256": submission.baseline_candidate_sha256,
        "candidate_artifact_format": "superglm-candidate-joblib-v2",
        "candidate_artifact_size_bytes": 321,
        "candidate_python_version": "3.14.4",
        "candidate_superglm_version": "0.11.0",
        "model_source_sha256": submission.model_source_sha256,
        "model_frame_sha256": "d" * 64,
    }

    class Rows:
        def mappings(self):
            return self

        def all(self):
            return [row]

    class Connection:
        def __init__(self):
            self.statements = []

        def execute(self, statement, params):
            self.statements.append((str(statement), params))
            return Rows()

    class Begin:
        def __init__(self, connection):
            self.connection = connection

        def __enter__(self):
            return self.connection

        def __exit__(self, exc_type, exc, tb):
            return False

    class Engine:
        def __init__(self):
            self.connection = Connection()

        def begin(self):
            return Begin(self.connection)

    bundle = SimpleNamespace(
        model_name="HOME_FREQ",
        model_version="v4",
        export_id="parent-export",
        manifest_id="manifest-1",
        split_set_id="split-1",
        model_source_sha256="b" * 64,
        model_frame_sha256="d" * 64,
    )
    load_calls = []
    champion_calls = []

    def fake_load_candidate_bundle(path, **kwargs):
        load_calls.append((path, kwargs))
        return bundle

    def fake_load_champion_bundle(engine, **kwargs):
        champion_calls.append(kwargs)
        return None, "no champion"

    monkeypatch.setattr(
        editor_candidate,
        "schema_names_from_connectable",
        lambda engine: SimpleNamespace(pricing="pricing", mlops="mlops"),
    )
    monkeypatch.setattr(editor_candidate, "load_candidate_bundle", fake_load_candidate_bundle)
    monkeypatch.setattr(editor_candidate, "_load_champion_bundle", fake_load_champion_bundle)
    injected_config = EDITOR_CONFIG
    engine = Engine()

    parent = editor_candidate.load_parent_candidate(
        engine,
        submission,
        allowed_root=configured_root,
        model_config=injected_config,
    )

    assert parent.bundle is bundle
    assert parent.config is injected_config
    assert parent.effective_from == effective_from_date
    assert load_calls[0][0] == str(candidate_path)
    assert load_calls[0][1]["allowed_root"] == configured_root
    assert champion_calls[0]["allowed_root"] == configured_root
    assert champion_calls[0]["deployment_slot"] == submission.deployment_slot
    statement = engine.connection.statements[0][0]
    assert "split_link.manifest_id = mr.manifest_id" in statement
    assert "split_link.dataset_role = 'training'" in statement
    assert "split_link.split_role = 'validation'" in statement
    assert "mr.model_version AS run_model_version" in statement
    assert "mr.export_id" in statement
    assert "manifest.model_frame_sha256" in statement
    assert "DATASET_MANIFEST AS manifest" in statement

    for sql_digest, bundle_digest in (
        (None, "d" * 64),
        ("not-a-digest", "d" * 64),
        ("d" * 64, None),
        ("d" * 64, "e" * 64),
    ):
        row["model_frame_sha256"] = sql_digest
        bundle.model_frame_sha256 = bundle_digest
        with pytest.raises(editor_candidate.EditorSubmissionError, match="model_frame_sha256"):
            editor_candidate.load_parent_candidate(
                engine,
                submission,
                allowed_root=configured_root,
                model_config=injected_config,
            )
    row["model_frame_sha256"] = "d" * 64
    bundle.model_frame_sha256 = "d" * 64

    for field_name in ("model_name", "model_version", "export_id"):
        original = getattr(bundle, field_name)
        setattr(bundle, field_name, f"wrong-{field_name}")
        with pytest.raises(editor_candidate.EditorSubmissionError, match=field_name):
            editor_candidate.load_parent_candidate(
                engine,
                submission,
                allowed_root=configured_root,
                model_config=injected_config,
            )
        setattr(bundle, field_name, original)


@pytest.mark.parametrize(
    "submission_relative_path",
    ["submission.json", "crafted/deep/layout/submission.json"],
)
def test_editor_session_root_cannot_be_widened_by_submission_path(
    tmp_path,
    submission_relative_path,
):
    from pricing_pipeline.publishing import editor_candidate
    from pricing_pipeline.workbench.submission import EditorSubmissionError

    configured_root = tmp_path / "configured-workbench"
    outside_session = tmp_path / "outside" / "editor_session.json"
    parent = SimpleNamespace(bundle=SimpleNamespace(fitted_model=object()))
    submission = SimpleNamespace(
        path=str(tmp_path / submission_relative_path),
        editor_session_path=str(outside_session),
    )

    with pytest.raises(EditorSubmissionError, match="outside artifact root"):
        editor_candidate._load_edited_model(
            parent,
            submission,
            allowed_root=configured_root,
        )


def test_editor_session_replays_against_verified_parent_model(
    monkeypatch,
    tmp_path,
):
    from superglm.editor import EditorSession

    from pricing_pipeline.publishing import editor_candidate
    from pricing_pipeline.workbench.submission import sha256_file

    configured_root = tmp_path / "configured-workbench"
    session_path = configured_root / "models/HOME_FREQ/editor/deep/session.json"
    session_path.parent.mkdir(parents=True)
    session_path.write_bytes(b'{"edits": []}\n')
    fitted_model = object()
    edited_model = object()
    bundle = SimpleNamespace(
        fitted_model=fitted_model,
        X=object(),
        y=object(),
        sample_weight=object(),
        offset=object(),
    )
    parent = SimpleNamespace(bundle=bundle)
    submission = SimpleNamespace(
        format="superglm-editor-submission-v1",
        path=str(configured_root / "submission.json"),
        editor_session_path=str(session_path),
        editor_session_size_bytes=session_path.stat().st_size,
        editor_session_sha256=sha256_file(session_path),
    )
    replay = SimpleNamespace(to_model=lambda **kwargs: (edited_model, kwargs))
    monkeypatch.setattr(
        EditorSession,
        "load",
        staticmethod(lambda path, *, model: replay),
    )

    loaded, replayed_inputs = editor_candidate._load_edited_model(
        parent,
        submission,
        allowed_root=configured_root,
    )

    assert loaded is edited_model
    assert replayed_inputs == {
        "X": bundle.X,
        "y": bundle.y,
        "sample_weight": bundle.sample_weight,
        "offset": bundle.offset,
    }


def test_v2_submission_loads_final_model_without_replaying_session(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd
    from superglm.editor import EditorSession

    from pricing_pipeline.publishing import editor_candidate

    class Model:
        def __init__(self, features, beta):
            self.features = features
            self._result = SimpleNamespace(beta=np.asarray(beta))

        def predict(self, X, offset=None):
            return np.ones(len(X))

    parent_model = Model({"region": object(), "x": object()}, [0.0, 0.1, 0.2, 0.3])
    edited_model = Model({"x": object(), "region": object()}, [0.0, 0.2, 0.3])
    parent = SimpleNamespace(
        bundle=SimpleNamespace(
            fitted_model=parent_model,
            X=pd.DataFrame({"region": ["A", "B"], "x": [1.0, 2.0]}),
            offset=None,
        )
    )
    submission = SimpleNamespace(
        format="superglm-editor-submission-v2",
        edited_model_path=str(tmp_path / "edited_model.joblib"),
        edited_model_sha256="a" * 64,
        edited_model_size_bytes=123,
        edited_model_format="superglm-edited-model-joblib-v1",
        edited_model_python_version="3.14.4",
        edited_model_superglm_version="0.13.0",
    )
    received = {}

    def fake_load(path, **metadata):
        received["path"] = path
        received.update(metadata)
        return edited_model

    monkeypatch.setattr(editor_candidate, "load_edited_model", fake_load, raising=False)
    monkeypatch.setattr(
        EditorSession,
        "load",
        staticmethod(lambda *args, **kwargs: pytest.fail("v2 session was replayed")),
    )

    loaded = editor_candidate._load_edited_model(
        parent,
        submission,
        allowed_root=tmp_path,
    )

    assert loaded is edited_model
    assert len(loaded._result.beta) < len(parent_model._result.beta)
    assert received == {
        "path": submission.edited_model_path,
        "expected_sha256": submission.edited_model_sha256,
        "expected_size_bytes": submission.edited_model_size_bytes,
        "expected_format": submission.edited_model_format,
        "expected_python_version": submission.edited_model_python_version,
        "expected_superglm_version": submission.edited_model_superglm_version,
        "allowed_root": tmp_path,
    }


@pytest.mark.parametrize("metadata_kind", ["missing", "bad-schema", "bad-sha"])
def test_manual_submission_rejects_missing_or_malformed_policy_before_model_load(
    monkeypatch,
    tmp_path,
    metadata_kind,
):
    from pricing_pipeline.modeling.manual_adjustment import (
        ManualAdjustmentPolicy,
        ManualAdjustmentRule,
    )
    from pricing_pipeline.publishing import editor_candidate

    policy = ManualAdjustmentPolicy(
        name="market adjustment",
        version=1,
        reason="Approved market response",
        rules=(
            ManualAdjustmentRule.multiply_levels(
                "segment",
                ["B"],
                1.05,
                reason="Selected segment uplift",
            ),
        ),
    )
    edit_metadata = None
    if metadata_kind != "missing":
        policy_payload = policy.to_payload()
        if metadata_kind == "bad-schema":
            policy_payload["rules"] = "not-a-list"
        edit_metadata = {
            "manual_adjustment_policy": policy_payload,
            "manual_adjustment_policy_sha256": (
                policy.sha256 if metadata_kind == "bad-schema" else "0" * 64
            ),
        }
    submission = SimpleNamespace(
        format="superglm-editor-submission-v2",
        model_kind="MANUAL_EDIT",
        edit_metadata=edit_metadata,
        edited_model_path=str(tmp_path / "edited_model.joblib"),
        edited_model_sha256="a" * 64,
        edited_model_size_bytes=123,
        edited_model_format="superglm-edited-model-joblib-v1",
        edited_model_python_version="3.14.4",
        edited_model_superglm_version="0.13.0",
    )
    parent = SimpleNamespace(bundle=SimpleNamespace(fitted_model=object()))
    monkeypatch.setattr(
        editor_candidate,
        "load_edited_model",
        lambda *args, **kwargs: pytest.fail(
            "invalid MANUAL_EDIT policy reached trusted object loading"
        ),
    )

    with pytest.raises(
        editor_candidate.EditorSubmissionError,
        match="MANUAL_EDIT submission has invalid manual adjustment policy",
    ):
        editor_candidate._load_edited_model(
            parent,
            submission,
            allowed_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("model_variant", "error_match"),
    [
        pytest.param("trusted-parent", None, id="trusted-parent"),
        pytest.param(
            "mutated-parent",
            "does not match trusted manual adjustment policy replay",
            id="mutated-parent",
        ),
        pytest.param(
            "off-frame-structure",
            "publication receipt does not match trusted manual adjustment policy replay",
            id="off-frame-structure",
        ),
        pytest.param(
            "unspanned-coefficient",
            "normalized fitted runtime state does not match trusted manual adjustment policy replay",
            id="unspanned-coefficient",
        ),
    ],
)
def test_manual_submission_must_match_policy_replayed_on_trusted_parent(
    monkeypatch,
    tmp_path,
    model_variant,
    error_match,
):
    import math

    import numpy as np
    import pandas as pd
    from superglm import Categorical, Numeric, SuperGLM
    from superglm.editor import EditorSession

    from pricing_pipeline.modeling.manual_adjustment import (
        ManualAdjustmentPolicy,
        ManualAdjustmentRule,
    )
    from pricing_pipeline.publishing import editor_candidate

    x = (
        np.zeros(60)
        if model_variant == "unspanned-coefficient"
        else np.tile(np.linspace(-1.0, 1.0, 20), 3)
    )
    frame = pd.DataFrame(
        {
            "segment": np.repeat(["A", "B", "C"], 20),
            "hidden": np.tile(["H", "I"], 30),
            "x": x,
        }
    )
    target = np.array(
        [
            {"A": 1.0, "B": 2.0, "C": 3.0}[segment] * np.exp(0.1 * x)
            for segment, x in zip(frame["segment"], frame["x"], strict=True)
        ]
    )
    trusted_parent_model = SuperGLM(
        features={
            "segment": Categorical(base="first"),
            "hidden": Categorical(base="first", levels=["H", "I"]),
            "x": Numeric(),
        },
        selection_penalty=0.0,
    ).fit(frame, target)
    train_data = (frame, target, None, None)
    submission_parent_model = trusted_parent_model
    if model_variant == "mutated-parent":
        mutated_parent_session = EditorSession.from_model(
            trusted_parent_model,
            train_data=train_data,
            cv_report={},
        )
        mutated_parent_session.select_levels("segment", ["A"])
        mutated_parent_session.shift("segment", math.log(1.2))
        submission_parent_model = mutated_parent_session.to_model()
    elif model_variant == "off-frame-structure":
        submission_parent_model = SuperGLM(
            features={
                "segment": Categorical(base="first"),
                "hidden": Categorical(base="first", levels=["H", "I", "OFF"]),
                "x": Numeric(),
            },
            selection_penalty=0.0,
        ).fit(frame, target)

    policy = ManualAdjustmentPolicy(
        name="market adjustment",
        version=1,
        reason="Approved market response",
        rules=(
            ManualAdjustmentRule.multiply_levels(
                "segment",
                ["B"],
                1.05,
                reason="Selected segment uplift",
            ),
        ),
    )
    submitted_session = EditorSession.from_model(
        submission_parent_model,
        train_data=train_data,
        cv_report={},
    )
    for rule in policy.rules:
        rule.apply(submitted_session)
    submitted_model = submitted_session.to_model()
    if model_variant == "unspanned-coefficient":
        changed_beta = np.array(submitted_model._result.beta, copy=True)
        changed_beta[-1] += 3.0
        object.__setattr__(submitted_model._result, "beta", changed_beta)
        trusted_frame_prediction = submitted_session.to_model().predict(frame)
        np.testing.assert_allclose(submitted_model.predict(frame), trusted_frame_prediction)
        off_frame = pd.DataFrame({"segment": ["A"], "hidden": ["H"], "x": [2.0]})
        assert not np.allclose(
            submitted_model.predict(off_frame),
            submitted_session.to_model().predict(off_frame),
        )

    parent = SimpleNamespace(
        bundle=SimpleNamespace(
            fitted_model=trusted_parent_model,
            X=frame,
            y=target,
            sample_weight=None,
            offset=None,
            cv_report={},
            offset_contract=editor_candidate.OffsetExportContract(handling="NONE"),
            fit_sample_weight_name=None,
            export_weight_name=None,
        )
    )
    submission = SimpleNamespace(
        format="superglm-editor-submission-v2",
        model_kind="MANUAL_EDIT",
        edit_metadata={
            "manual_adjustment_policy": policy.to_payload(),
            "manual_adjustment_policy_sha256": policy.sha256,
        },
        edited_model_path=str(tmp_path / "edited_model.joblib"),
        edited_model_sha256="a" * 64,
        edited_model_size_bytes=123,
        edited_model_format="superglm-edited-model-joblib-v1",
        edited_model_python_version="3.14.4",
        edited_model_superglm_version="0.13.0",
    )
    monkeypatch.setattr(
        editor_candidate,
        "load_edited_model",
        lambda *args, **kwargs: submitted_model,
    )

    def load():
        return editor_candidate._load_edited_model(
            parent,
            submission,
            allowed_root=tmp_path,
        )

    if error_match is not None:
        with pytest.raises(
            editor_candidate.EditorSubmissionError,
            match=error_match,
        ):
            load()
    else:
        assert load() is submitted_model


@pytest.mark.parametrize("mutate_spline", [False, True], ids=["loaded-replay", "mutated-r-inv"])
def test_manual_replay_compares_complete_normalized_fitted_runtime_state(mutate_spline):
    import io

    import joblib
    import numpy as np
    import pandas as pd
    from superglm import Categorical, NaturalSpline, SuperGLM

    from pricing_pipeline.modeling.manual_adjustment import (
        ManualAdjustmentPolicy,
        ManualAdjustmentRule,
        replay_manual_adjustment_policy,
    )
    from pricing_pipeline.publishing import editor_candidate

    x = np.tile([-1.0, 0.0, 1.0], 30)
    segment = np.repeat(["A", "B", "C"], 30)
    frame = pd.DataFrame({"segment": segment, "x": x})
    target = np.exp(0.2 * x) * np.array(
        [{"A": 1.0, "B": 2.0, "C": 3.0}[value] for value in segment]
    )
    parent_model = SuperGLM(
        features={
            "segment": Categorical(base="first"),
            "x": NaturalSpline(n_knots=8, extrapolation="extend"),
        },
        selection_penalty=0.0,
    ).fit(frame, target)
    bundle = SimpleNamespace(
        fitted_model=parent_model,
        X=frame,
        y=target,
        sample_weight=None,
        offset=None,
        cv_report={},
        offset_contract=editor_candidate.OffsetExportContract(handling="NONE"),
        fit_sample_weight_name=None,
        export_weight_name=None,
    )
    policy = ManualAdjustmentPolicy(
        name="market adjustment",
        version=1,
        reason="Approved market response",
        rules=(
            ManualAdjustmentRule.multiply_levels(
                "segment",
                ["B"],
                1.05,
                reason="Selected segment uplift",
            ),
        ),
    )
    _, submitted_model = replay_manual_adjustment_policy(bundle, policy)
    artifact = io.BytesIO()
    joblib.dump(submitted_model, artifact, protocol=5)
    artifact.seek(0)
    submitted_model = joblib.load(artifact)

    if mutate_spline:
        spline = submitted_model._specs["x"]
        plan = next(
            entry for entry in submitted_model._prediction_plan["features"] if entry["name"] == "x"
        )
        spline_beta = np.asarray(submitted_model._result.beta)[plan["beta_idx"]]
        raw_basis = spline._basis_matrix(frame["x"].to_numpy()).toarray()
        null_vector = np.linalg.svd(raw_basis, full_matrices=True)[2][-1]
        assert np.max(np.abs(raw_basis @ null_vector)) < 1e-12
        beta_position = int(np.argmax(np.abs(spline_beta)))
        changed_r_inv = np.array(spline._R_inv, copy=True)
        changed_r_inv[:, beta_position] += null_vector / spline_beta[beta_position]
        spline.set_reparametrisation(changed_r_inv)

        _, trusted_model = replay_manual_adjustment_policy(bundle, policy)
        np.testing.assert_allclose(
            submitted_model.predict(frame),
            trusted_model.predict(frame),
        )
        grid = np.linspace(-1.0, 1.0, 201)
        grid_delta = spline._basis_matrix(grid).toarray() @ null_vector
        off_frame_x = float(grid[int(np.argmax(np.abs(grid_delta)))])
        off_frame = pd.DataFrame({"segment": ["A"], "x": [off_frame_x]})
        assert not np.allclose(
            submitted_model.predict(off_frame),
            trusted_model.predict(off_frame),
        )

    parent = SimpleNamespace(bundle=bundle)
    if mutate_spline:
        with pytest.raises(
            editor_candidate.EditorSubmissionError,
            match="normalized fitted runtime state does not match trusted manual adjustment policy replay",
        ):
            editor_candidate._require_manual_policy_replay(
                parent,
                submitted_model,
                policy,
            )
    else:
        editor_candidate._require_manual_policy_replay(
            parent,
            submitted_model,
            policy,
        )


@pytest.mark.parametrize(
    "edited_features",
    [
        {"region": object()},
        {"region": object(), "x": object(), "new_feature": object()},
    ],
)
def test_v2_submission_rejects_changed_feature_names(
    monkeypatch,
    tmp_path,
    edited_features,
):
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing import editor_candidate

    parent_model = SimpleNamespace(features={"region": object(), "x": object()})
    edited_model = SimpleNamespace(
        features=edited_features,
        _result=object(),
        predict=lambda X, offset=None: np.ones(len(X)),
    )
    parent = SimpleNamespace(
        bundle=SimpleNamespace(
            fitted_model=parent_model,
            X=pd.DataFrame({"region": ["A"], "x": [1.0]}),
            offset=None,
        )
    )
    submission = SimpleNamespace(
        format="superglm-editor-submission-v2",
        edited_model_path=str(tmp_path / "edited_model.joblib"),
        edited_model_sha256="a" * 64,
        edited_model_size_bytes=123,
        edited_model_format="superglm-edited-model-joblib-v1",
        edited_model_python_version="3.14.4",
        edited_model_superglm_version="0.13.0",
    )
    monkeypatch.setattr(
        editor_candidate,
        "load_edited_model",
        lambda *args, **kwargs: edited_model,
        raising=False,
    )

    with pytest.raises(editor_candidate.EditorSubmissionError, match="feature names"):
        editor_candidate._load_edited_model(parent, submission, allowed_root=tmp_path)


@pytest.mark.parametrize(
    ("result", "prediction", "message"),
    [
        (None, [1.0], "not fitted"),
        (object(), None, "no callable predict"),
        (object(), [float("inf")], "invalid training predictions"),
    ],
)
def test_v2_submission_rejects_unusable_final_model(
    monkeypatch,
    tmp_path,
    result,
    prediction,
    message,
):
    import pandas as pd

    from pricing_pipeline.publishing import editor_candidate

    model = SimpleNamespace(
        features={"x": object()},
        _result=result,
        predict=None if prediction is None else lambda X, offset=None: prediction,
    )
    parent = SimpleNamespace(
        bundle=SimpleNamespace(
            fitted_model=SimpleNamespace(features={"x": object()}),
            X=pd.DataFrame({"x": [1.0]}),
            offset=None,
        )
    )
    submission = SimpleNamespace(
        format="superglm-editor-submission-v2",
        edited_model_path=str(tmp_path / "edited_model.joblib"),
        edited_model_sha256="a" * 64,
        edited_model_size_bytes=123,
        edited_model_format="superglm-edited-model-joblib-v1",
        edited_model_python_version="3.14.4",
        edited_model_superglm_version="0.13.0",
    )
    monkeypatch.setattr(
        editor_candidate,
        "load_edited_model",
        lambda *args, **kwargs: model,
        raising=False,
    )

    with pytest.raises(editor_candidate.EditorSubmissionError, match=message):
        editor_candidate._load_edited_model(parent, submission, allowed_root=tmp_path)


def test_collapsed_editor_model_publishes(tmp_path):
    import joblib
    import numpy as np
    import pandas as pd
    from superglm import Categorical, Numeric, SuperGLM
    from superglm.editor import EditorSession

    from pricing_pipeline.publishing import editor_candidate, staging
    from pricing_pipeline.workbench.artifacts import CandidateBundle
    from pricing_pipeline.workbench.submission import save_editor_submission

    region = np.repeat(["A", "B", "C", "D"], 20)
    x = np.tile(np.linspace(-1.0, 1.0, 20), 4)
    mean = np.array([{"A": 1.0, "B": 2.0, "C": 2.0, "D": 4.0}[value] for value in region])
    y = np.random.default_rng(20260722).poisson(mean * np.exp(0.2 * x))
    frame = pd.DataFrame({"region": region, "x": x})
    parent_model = SuperGLM(
        features={"region": Categorical(base="first"), "x": Numeric()},
        selection_penalty=0.0,
    ).fit(frame, y)
    bundle = CandidateBundle(
        fitted_model=parent_model,
        X=frame,
        y=y,
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={},
        model_name="HOME_FREQ",
        model_version="v1",
        export_id="export-1",
        manifest_id="manifest-1",
        split_set_id=None,
        pk_columns=("policy_id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        model_frame_sha256="d" * 64,
        offset_contract={"handling": "NONE"},
    )
    candidate = SimpleNamespace(
        workbench=SimpleNamespace(
            settings=Settings(workbench_artifact_root=tmp_path),
            model_config=SimpleNamespace(deployment_slot="HOME_FREQ_UAT"),
        ),
        model_name="HOME_FREQ",
        package_version=1,
        rate_package_id=101,
        model_run_id=201,
        bundle=bundle,
        technical={"candidate_artifact_sha256": "c" * 64},
    )
    session = EditorSession.from_model(parent_model, train_data=(frame, y))
    session.select_levels("region", ["B", "C"])
    session.replace_with_collapsed_levels("region", method="fit")
    submission = save_editor_submission(
        candidate,
        editor_session=session,
        reason="Combine equivalent market regions",
        claimed_identity="analyst@example.test",
    )

    parent = editor_candidate.ParentCandidate(
        model_id=17,
        model_name="HOME_FREQ",
        model_version="v1",
        package_version=1,
        rate_package_id=101,
        model_run_id=201,
        effective_from=None,
        effective_to=None,
        config=SimpleNamespace(
            model_type="superglm_poisson",
            target_name="claim_count",
        ),
        bundle=bundle,
        champion=editor_candidate.ChampionSnapshot(
            deployment_slot="HOME_FREQ_UAT",
            rate_package_id=None,
            bundle=None,
            unavailable_reason="no champion is deployed",
        ),
    )
    write_dir = tmp_path / "publication-staging"
    write_dir.mkdir()
    exported = editor_candidate.export_edited_model(
        parent,
        submission,
        created_by="publisher@example.test",
        allowed_root=tmp_path,
        write_dir=write_dir,
        published_dir=tmp_path / "published",
    )
    loaded = exported.edited_model

    assert len(loaded.result.beta) < len(parent_model.result.beta)
    assert set(loaded.features) == set(parent_model.features)

    _, rates, _ = staging.build_staging_frames(
        staging.StagingExport(
            workbook_path=write_dir / "rating_tables.xlsx",
            export_id="edited-export",
            model_name="HOME_FREQ",
            target_name="claim_count",
            model_type="superglm_poisson",
            model_version="v1",
            effective_from=None,
            effective_to=None,
            interaction_features={},
            created_by="publisher@example.test",
            replace=False,
            model_id=17,
        )
    )
    region_rows = rates.loc[rates["term_name"] == "region"]
    relativities = dict(
        zip(
            region_rows["cell_key_text"].str.removeprefix("region="),
            region_rows["multiplier"],
            strict=True,
        )
    )
    assert list(relativities) == ["A", "B", "C", "D"]
    assert relativities["B"] == pytest.approx(relativities["C"])

    child_bundle = joblib.load(write_dir / "candidate_bundle.joblib")["bundle"]
    assert len(child_bundle.fitted_model.result.beta) < len(parent_model.result.beta)
    np.testing.assert_allclose(
        child_bundle.fitted_model.predict(frame),
        loaded.predict(frame),
    )


def test_training_comparison_metrics_are_stable_and_scoped():
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import training_comparison_metrics
    from pricing_pipeline.workbench.artifacts import CandidateBundle

    class Model:
        def __init__(self, prediction):
            self.prediction = np.asarray(prediction, dtype=float)
            self._distribution = SimpleNamespace(
                deviance_unit=lambda y, mu: (np.asarray(y) - np.asarray(mu)) ** 2
            )

        def predict(self, X, offset=None):
            assert len(X) == 3
            return self.prediction

    bundle = CandidateBundle(
        fitted_model=Model([1.0, 2.0, 3.0]),
        X=pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        y=np.array([1.0, 1.0, 4.0]),
        sample_weight=np.array([1.0, 2.0, 1.0]),
        offset=None,
        export_weight=None,
        cv_report={},
        model_name="HOME_FREQ",
        model_version="20260603",
        export_id="export-1",
        manifest_id="manifest-1",
        split_set_id="split-1",
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "NONE"},
    )

    metrics, scopes = training_comparison_metrics(
        bundle.fitted_model,
        Model([1.1, 1.8, 3.2]),
        bundle,
        comparison_name="parent",
    )

    assert metrics["editor_training_parent_mean_absolute_prediction_delta"] == pytest.approx(0.175)
    assert metrics["editor_training_parent_max_absolute_prediction_delta"] == pytest.approx(0.2)
    assert metrics["editor_training_deviance_delta"] == pytest.approx(-0.2675)
    assert set(scopes.values()) == {"editor_training_parent"}


def test_champion_comparison_scores_parent_rows_even_when_training_rows_differ(tmp_path):
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import _load_champion_bundle
    from pricing_pipeline.workbench.artifacts import CandidateBundle, save_candidate_bundle

    parent = CandidateBundle(
        fitted_model={"model": "parent"},
        X=pd.DataFrame({"x": [1.0, 2.0]}),
        y=np.array([0.0, 1.0]),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={},
        model_name="HOME_FREQ",
        model_version="20260603",
        export_id="parent-export",
        manifest_id="parent-manifest",
        split_set_id="parent-split",
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={
            "handling": "NONE",
            "source_factor_name": None,
            "published_factor_name": None,
            "source_name": None,
            "label": None,
        },
    )
    champion = CandidateBundle(
        fitted_model={"model": "champion"},
        X=pd.DataFrame({"x": [8.0, 9.0, 10.0]}),
        y=np.array([1.0, 0.0, 1.0]),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={},
        model_name="HOME_FREQ",
        model_version="20260603",
        export_id="champion-export",
        manifest_id="champion-manifest",
        split_set_id="champion-split",
        pk_columns=("id",),
        row_order_sha256="c" * 64,
        model_source_sha256="d" * 64,
        offset_contract={"handling": "NONE"},
    )
    artifact = save_candidate_bundle(champion, tmp_path / "champion.joblib")

    class Rows:
        def mappings(self):
            return self

        def all(self):
            return [
                {
                    "rate_package_id": 107,
                    "run_status": "SUCCESS",
                    "candidate_artifact_path": artifact.path,
                    "candidate_artifact_sha256": artifact.sha256,
                    "candidate_artifact_format": artifact.format,
                    "candidate_artifact_size_bytes": artifact.size_bytes,
                    "candidate_python_version": artifact.python_version,
                    "candidate_superglm_version": artifact.superglm_version,
                }
            ]

    class Connection:
        def execute(self, statement, params):
            return Rows()

    class Begin:
        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class Engine:
        def begin(self):
            return Begin()

    snapshot = _load_champion_bundle(
        Engine(),
        model_id=17,
        deployment_slot="HOME_FREQ_UAT",
        allowed_root=tmp_path,
        parent_bundle=parent,
    )

    assert snapshot.status == "COMPARED"
    assert snapshot.rate_package_id == 107
    assert snapshot.unavailable_reason is None
    assert snapshot.bundle is not None
    assert snapshot.bundle.manifest_id == "champion-manifest"


@pytest.mark.parametrize(
    ("champion_offset_contract", "expected_reason"),
    [
        (
            {
                "handling": "EXPORTED_FACTOR",
                "source_factor_name": "Duration",
                "published_factor_name": "Duration",
                "source_name": "Duration",
                "label": "log(Duration)",
            },
            "the deployed champion uses a different offset contract",
        ),
        (
            {"handling": "NONE", "source_name": "invalid"},
            "the deployed champion has an invalid offset contract",
        ),
    ],
)
def test_champion_comparison_rejects_incompatible_offset_contract(
    monkeypatch,
    tmp_path,
    champion_offset_contract,
    expected_reason,
):
    import pandas as pd

    from pricing_pipeline.publishing import editor_candidate

    parent = SimpleNamespace(
        X=pd.DataFrame({"x": [1.0]}),
        offset_contract={
            "handling": "EXPORTED_FACTOR",
            "source_factor_name": "Exposure",
            "published_factor_name": "Exposure",
            "source_name": "Exposure",
            "label": "log(Exposure)",
        },
    )
    champion = SimpleNamespace(
        X=pd.DataFrame({"x": [8.0]}),
        offset_contract=champion_offset_contract,
    )
    row = {
        "rate_package_id": 107,
        "run_status": "SUCCESS",
        "candidate_artifact_path": str(tmp_path / "champion.joblib"),
        "candidate_artifact_sha256": "a" * 64,
        "candidate_artifact_format": "superglm-candidate-joblib-v2",
        "candidate_artifact_size_bytes": 321,
        "candidate_python_version": "3.14.4",
        "candidate_superglm_version": "0.11.0",
    }

    class Rows:
        def mappings(self):
            return self

        def all(self):
            return [row]

    class Connection:
        def execute(self, statement, params):
            return Rows()

    class Begin:
        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class Engine:
        def begin(self):
            return Begin()

    monkeypatch.setattr(
        editor_candidate,
        "schema_names_from_connectable",
        lambda engine: SimpleNamespace(pricing="pricing"),
    )
    monkeypatch.setattr(
        editor_candidate,
        "load_candidate_bundle",
        lambda *args, **kwargs: champion,
    )

    snapshot = editor_candidate._load_champion_bundle(
        Engine(),
        model_id=17,
        deployment_slot="HOME_FREQ_UAT",
        allowed_root=tmp_path,
        parent_bundle=parent,
    )

    assert snapshot.status == "UNAVAILABLE"
    assert snapshot.bundle is None
    assert snapshot.unavailable_reason == expected_reason


@pytest.mark.parametrize(
    ("rows", "expected_status", "expected_rate_package_id"),
    [
        ([], "NO_CHAMPION", None),
        (
            [{"rate_package_id": 107, "run_status": "FAILED"}],
            "UNAVAILABLE",
            107,
        ),
    ],
)
def test_champion_snapshot_distinguishes_absent_and_unavailable_champion(
    rows,
    expected_status,
    expected_rate_package_id,
    tmp_path,
):
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import _load_champion_bundle
    from pricing_pipeline.workbench.artifacts import CandidateBundle

    class Rows:
        def mappings(self):
            return self

        def all(self):
            return rows

    class Connection:
        def execute(self, statement, params):
            return Rows()

    class Begin:
        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class Engine:
        def begin(self):
            return Begin()

    parent = CandidateBundle(
        fitted_model=object(),
        X=pd.DataFrame({"x": [1.0]}),
        y=np.array([0.0]),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={},
        model_name="HOME_FREQ",
        model_version="20260603",
        export_id="parent-export",
        manifest_id="parent-manifest",
        split_set_id=None,
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "NONE"},
    )

    snapshot = _load_champion_bundle(
        Engine(),
        model_id=17,
        deployment_slot="HOME_FREQ_UAT",
        allowed_root=tmp_path,
        parent_bundle=parent,
    )

    assert snapshot.status == expected_status
    assert snapshot.rate_package_id == expected_rate_package_id
    assert snapshot.revision_metadata()["deployment_slot"] == "HOME_FREQ_UAT"
    assert snapshot.revision_metadata()["available"] is (expected_status == "COMPARED")


def test_package_specific_parity_uses_bounded_rows_and_explicit_package_id():
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import verify_package_sql_parity
    from pricing_pipeline.workbench.artifacts import CandidateBundle

    class Model:
        def predict(self, X, offset=None):
            return np.asarray(X["x"], dtype=float) * 2.0

    class Result:
        def __init__(self, prediction):
            self.prediction = prediction

        def mappings(self):
            return self

        def one(self):
            return {"prediction": self.prediction}

    class Connection:
        def __init__(self):
            self.calls = []

        def execute(self, statement, params):
            self.calls.append((str(statement), params))
            return Result(params["x_prediction"])

    bundle = CandidateBundle(
        fitted_model=Model(),
        X=pd.DataFrame({"x": np.arange(100, dtype=float)}),
        y=np.ones(100),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={},
        model_name="HOME_FREQ",
        model_version="20260603",
        export_id="export-1",
        manifest_id="manifest-1",
        split_set_id=None,
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "NONE"},
    )
    connection = Connection()

    verify_package_sql_parity(
        connection,
        rate_package_id=108,
        edited_model=bundle.fitted_model,
        bundle=bundle,
        sample_size=5,
        execute_params_hook=lambda params, expected: {
            **params,
            "x_prediction": expected,
        },
    )

    assert len(connection.calls) == 5
    assert all(params["rate_package_id"] == 108 for _sql, params in connection.calls)
    assert all("PREDICT_RATE_PACKAGE" in sql for sql, _params in connection.calls)


def test_package_sql_parity_uses_published_feature_names():
    import numpy as np
    import pandas as pd
    from superglm import Numeric, SuperGLM

    from pricing_pipeline.publishing.editor_candidate import verify_package_sql_parity
    from pricing_pipeline.publishing.superglm_metadata import (
        build_superglm_publication_receipt,
    )
    from pricing_pipeline.publishing.superglm_publication_receipt import (
        OffsetExportContract,
    )
    from pricing_pipeline.workbench.artifacts import CandidateBundle

    class Result:
        def __init__(self, prediction):
            self.prediction = prediction

        def mappings(self):
            return self

        def one(self):
            return {"prediction": self.prediction}

    class Connection:
        def __init__(self):
            self.payloads = []

        def execute(self, _statement, params):
            self.payloads.append(json.loads(params["features_json"]))
            return Result(params["x_prediction"])

    training_x = pd.DataFrame({"a/b": np.linspace(0.5, 3.0, 30)})
    training_y = np.random.default_rng(20260714).poisson(1.5, size=len(training_x))
    model = SuperGLM(
        features={"a/b": Numeric()},
        selection_penalty=0.0,
    ).fit(training_x, training_y)
    bundle = CandidateBundle(
        fitted_model=model,
        X=pd.DataFrame({"a/b": [1.5]}),
        y=np.ones(1),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={},
        model_name="HOME_FREQ",
        model_version="20260603",
        export_id="export-1",
        manifest_id="manifest-1",
        split_set_id=None,
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "NONE"},
    )
    connection = Connection()
    receipt = build_superglm_publication_receipt(
        model,
        offset_contract=OffsetExportContract(handling="NONE"),
    )

    verify_package_sql_parity(
        connection,
        rate_package_id=108,
        edited_model=model,
        bundle=bundle,
        publication_receipt=receipt,
        execute_params_hook=lambda params, expected: {
            **params,
            "x_prediction": expected,
        },
    )

    assert connection.payloads == [{"a_b": 1.5}]


class _ParityResult:
    def __init__(self, prediction):
        self.prediction = prediction

    def mappings(self):
        return self

    def one(self):
        return {"prediction": self.prediction}


class _ParityConnection:
    def __init__(self, *, allow_execute=True):
        self.allow_execute = allow_execute
        self.calls = []

    def execute(self, statement, params):
        if not self.allow_execute:
            raise AssertionError("SQL must not execute for an invalid offset source")
        self.calls.append(params)
        return _ParityResult(params["x_prediction"])


def _offset_parity_bundle(*, handling, offset_source=None):
    import numpy as np
    import pandas as pd

    from pricing_pipeline.workbench.artifacts import CandidateBundle

    raw_source = np.array([12.0, 36.0])
    fitted_offset = np.log(raw_source / 12.0)

    class Model:
        def predict(self, X, offset=None):
            np.testing.assert_allclose(offset, fitted_offset)
            return np.asarray(X["x"], dtype=float) + np.exp(offset)

    if handling == "EXPORTED_FACTOR":
        contract = {
            "handling": handling,
            "source_factor_name": "Term",
            "published_factor_name": "Term",
            "source_name": "Term",
            "label": "log(Term / 12)",
        }
        if offset_source is None:
            offset_source = pd.Series(raw_source, name="Term")
        offset_source_name = "Term"
    else:
        contract = {
            "handling": handling,
            "source_name": "Term",
            "label": "log(Term / 12)",
        }
        offset_source_name = None

    return CandidateBundle(
        fitted_model=Model(),
        X=pd.DataFrame({"x": [1.0, 3.0]}),
        y=np.ones(2),
        sample_weight=None,
        offset=fitted_offset,
        offset_source=offset_source,
        export_weight=None,
        cv_report={},
        model_name="HOME_FREQ",
        model_version="20260603",
        export_id="export-1",
        manifest_id="manifest-1",
        split_set_id=None,
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract=contract,
        offset_source_name=offset_source_name,
    ), raw_source


@pytest.mark.parametrize("weight_kind", ["series", "array"])
def test_package_sql_parity_uses_bound_offset_source_for_exported_factor(weight_kind):
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import verify_package_sql_parity

    raw_source = pd.Series([12.0, 36.0], name="Term")
    offset_sources = {
        "series": raw_source,
        "array": raw_source.to_numpy(),
    }
    bundle, expected_source = _offset_parity_bundle(
        handling="EXPORTED_FACTOR",
        offset_source=offset_sources[weight_kind],
    )
    connection = _ParityConnection()

    verify_package_sql_parity(
        connection,
        rate_package_id=108,
        edited_model=bundle.fitted_model,
        bundle=bundle,
        sample_size=2,
        execute_params_hook=lambda params, expected: {
            **params,
            "x_prediction": expected,
        },
    )

    assert [
        json.loads(params["features_json"])["Term"] for params in connection.calls
    ] == expected_source.tolist()


def test_package_sql_parity_preserves_bound_categorical_offset_source_positionally():
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import verify_package_sql_parity

    published_levels = pd.Series(
        ["basic", "premium"],
        index=pd.Index([101, 303]),
        name="Term",
        dtype="category",
    )
    bundle, _raw_exposure = _offset_parity_bundle(
        handling="EXPORTED_FACTOR",
        offset_source=published_levels,
    )
    connection = _ParityConnection()

    verify_package_sql_parity(
        connection,
        rate_package_id=108,
        edited_model=bundle.fitted_model,
        bundle=bundle,
        sample_size=2,
        execute_params_hook=lambda params, expected: {
            **params,
            "x_prediction": expected,
        },
    )

    assert [
        json.loads(params["features_json"])["Term"] for params in connection.calls
    ] == published_levels.tolist()


def test_package_sql_parity_applies_fitted_offset_as_sql_exposure():
    import numpy as np

    from pricing_pipeline.publishing.editor_candidate import verify_package_sql_parity

    bundle, _raw_source = _offset_parity_bundle(handling="ALREADY_APPLIED_SQL_EXPOSURE")
    connection = _ParityConnection()

    verify_package_sql_parity(
        connection,
        rate_package_id=108,
        edited_model=bundle.fitted_model,
        bundle=bundle,
        sample_size=2,
        execute_params_hook=lambda params, expected: {
            **params,
            "x_prediction": expected,
        },
    )

    np.testing.assert_allclose(
        [params["exposure"] for params in connection.calls],
        np.exp(bundle.offset),
    )


def test_parent_cv_metrics_are_labeled_as_revision_baseline():
    import numpy as np
    import pandas as pd

    from pricing_pipeline.publishing.editor_candidate import parent_cv_metrics
    from pricing_pipeline.workbench.artifacts import CandidateBundle

    bundle = CandidateBundle(
        fitted_model=object(),
        X=pd.DataFrame({"x": [1.0]}),
        y=np.array([0.0]),
        sample_weight=None,
        offset=None,
        export_weight=None,
        cv_report={
            "mean_scores": {"deviance": 0.48},
            "pooled_scores": {"deviance": 0.47},
            "std_scores": {"deviance": 0.03},
            "oof_coverage": 1.0,
        },
        model_name="HOME_FREQ",
        model_version="20260603",
        export_id="export-1",
        manifest_id="manifest-1",
        split_set_id="split-1",
        pk_columns=("id",),
        row_order_sha256="a" * 64,
        model_source_sha256="b" * 64,
        offset_contract={"handling": "NONE"},
    )

    metrics = parent_cv_metrics(bundle)

    assert metrics == {
        "cv_mean_deviance": 0.48,
        "cv_pooled_deviance": 0.47,
        "cv_std_deviance": 0.03,
        "cv_oof_coverage": 1.0,
    }
