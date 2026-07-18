from __future__ import annotations

import math
from inspect import signature

import pytest

from pricing_pipeline.models.spec import ApprovedModelBuild
from pricing_pipeline.publishing.lineage import record_model_run


_UNSET = object()


class _Result:
    def __init__(self, scalar=None, row=None, rows=()):
        self.scalar = scalar
        self.row = row
        self.rows = list(rows)

    def mappings(self):
        return self

    def one_or_none(self):
        return self.row

    def all(self):
        return self.rows

    def scalar_one(self):
        return 501

    def scalar_one_or_none(self):
        return self.scalar


class _Connection:
    def __init__(
        self,
        *,
        parent_matches=True,
        parent_validation_source_model_run_id=_UNSET,
        package_overrides=None,
        fold_rows=None,
    ):
        self.events = []
        self.parent_matches = parent_matches
        self.parent_validation_source_model_run_id = parent_validation_source_model_run_id
        self.package_overrides = dict(package_overrides or {})
        self.fold_rows = list(
            fold_rows
            or (
                {
                    "manifest_id": "manifest-2",
                    "fold_no": 1,
                    "n_train": 80,
                    "n_test": 20,
                },
            )
        )

    def execute(self, statement, params=None):
        sql = str(statement)
        self.events.append((sql, params))
        if "child_package.parent_rate_package_id" in sql:
            if not self.parent_matches:
                return _Result(scalar=None)
            validation_source = self.parent_validation_source_model_run_id
            if validation_source is _UNSET or validation_source is None:
                validation_source = params["parent_model_run_id"]
            return _Result(scalar=validation_source)
        if "FROM pricing.PRICING_RATE_PACKAGE AS package" in sql:
            return _Result(
                row={
                    "model_id": 17,
                    "model_version": "v2",
                    "source_export_id": "export-2",
                    "parent_rate_package_id": None,
                    "build_fingerprint_sha256": "0" * 64,
                    **self.package_overrides,
                }
            )
        if "FROM pricing.CV_SPLIT_SET AS split_set" in sql:
            return _Result(rows=self.fold_rows)
        return _Result()


def _approved_build() -> ApprovedModelBuild:
    return ApprovedModelBuild(
        model_id=17,
        model_name="HOME_FREQ",
        model_version="v2",
        model_type="superglm_poisson",
        target_name="claim_count",
        deployment_slot="HOME_FREQ_UAT",
        manifest_id="manifest-2",
        split_set_id="split-2",
        export_id="export-2",
        rating_workbook_path="/tmp/attempt-2/rating.xlsx",
        rating_workbook_sha256="a" * 64,
        created_by="airflow",
        mlflow_run_id="mlflow-2",
        publication_receipt_path="/tmp/attempt-2/publication_receipt.json",
        publication_receipt_sha256="b" * 64,
        candidate_artifact_path="/tmp/attempt-2/candidate.joblib",
        candidate_artifact_sha256="c" * 64,
        candidate_artifact_format="superglm-candidate-joblib-v3",
        candidate_artifact_size_bytes=321,
        candidate_python_version="3.14.4",
        candidate_superglm_version="0.12.0",
        candidate_superglm_git_sha="f" * 40,
        build_fingerprint_sha256="0" * 64,
        builder_source_sha256="1" * 64,
        materialized_split_sha256="2" * 64,
        runtime_sha256="3" * 64,
        candidate_superglm_sha256="4" * 64,
        row_order_sha256="5" * 64,
        model_source_sha256="d" * 64,
        model_frame_sha256="e" * 64,
        metrics={"deviance": 0.42},
        metric_scopes={"deviance": "cv"},
        validation_splits=(
            {
                "validation_split_no": 1,
                "n_train": 80,
                "n_validation": 20,
                "metrics": {"deviance": 0.4},
            },
        ),
        validation_curve_status="COMPLETE",
        validation_curve_points=(
            {
                "validation_split_no": 1,
                "term_name": "age",
                "point_no": 1,
                "point_kind": "NUMERIC",
                "x_numeric": 0.0,
                "level_text": None,
                "eta_contribution": 0.0,
                "relativity": 1.0,
                "support_value": 10.0,
                "reference_value": 0.0,
                "reference_level": None,
            },
            {
                "validation_split_no": 1,
                "term_name": "age",
                "point_no": 2,
                "point_kind": "NUMERIC",
                "x_numeric": 1.0,
                "level_text": None,
                "eta_contribution": 0.1,
                "relativity": math.exp(0.1),
                "support_value": 5.0,
                "reference_value": 0.0,
                "reference_level": None,
            },
        ),
    )


def test_record_model_run_derives_audit_evidence_from_approved_build():
    connection = _Connection()
    build = _approved_build()

    model_run_id = record_model_run(
        None,
        build=build,
        dag_id="dag",
        airflow_run_id="scheduled__2026-07-12",
        rate_package_id=43,
        connection=connection,
    )

    assert model_run_id == 501
    model_run_merge = next(
        event for event in connection.events if "MERGE pricing.MODEL_RUN" in event[0]
    )
    assert model_run_merge[1]["manifest_id"] == build.manifest_id
    assert model_run_merge[1]["split_set_id"] == build.split_set_id
    assert model_run_merge[1]["rating_workbook_sha256"] == build.rating_workbook_sha256
    assert model_run_merge[1]["candidate_superglm_git_sha"] == build.candidate_superglm_git_sha
    assert model_run_merge[1]["builder_source_sha256"] == build.builder_source_sha256
    assert model_run_merge[1]["materialized_split_sha256"] == build.materialized_split_sha256
    assert model_run_merge[1]["runtime_sha256"] == build.runtime_sha256
    assert model_run_merge[1]["candidate_superglm_sha256"] == build.candidate_superglm_sha256
    assert model_run_merge[1]["validation_curve_status"] == "COMPLETE"
    assert "candidate_superglm_git_sha = :candidate_superglm_git_sha" in model_run_merge[0]
    assert model_run_merge[1]["run_status"] == "SUCCESS"
    assert any("MERGE mlops.MODEL_RUN_METRIC" in sql for sql, _ in connection.events)
    assert any("MERGE pricing.CV_FOLD_METRIC" in sql for sql, _ in connection.events)
    curve_inserts = [
        event
        for event in connection.events
        if "INSERT INTO pricing.CV_SPLIT_CURVE_POINT" in event[0]
    ]
    assert len(curve_inserts) == 1
    assert isinstance(curve_inserts[0][1], list)
    assert len(curve_inserts[0][1]) == 2
    root_source_update = next(
        event
        for event in connection.events
        if "validation_source_model_run_id = :model_run_id" in event[0]
    )
    assert root_source_update[1]["model_run_id"] == 501
    assert list(signature(record_model_run).parameters) == [
        "engine",
        "build",
        "dag_id",
        "airflow_run_id",
        "rate_package_id",
        "parent_model_run_id",
        "connection",
    ]


@pytest.mark.parametrize(
    ("case", "parent_model_run_id", "stored_parent_source", "expected_source"),
    [
        ("direct_edit", 409, 409, 409),
        ("edit_of_edit", 777, 409, 409),
        ("legacy_parent", 777, None, 777),
    ],
)
def test_record_model_run_derives_editor_validation_source_from_parent_chain(
    case,
    parent_model_run_id,
    stored_parent_source,
    expected_source,
):
    connection = _Connection(
        parent_validation_source_model_run_id=stored_parent_source,
    )

    record_model_run(
        None,
        connection=connection,
        build=_approved_build(),
        dag_id="dag",
        airflow_run_id=f"manual__{case}",
        rate_package_id=43,
        parent_model_run_id=parent_model_run_id,
    )

    parent_source_query = next(
        event for event in connection.events if "child_package.parent_rate_package_id" in event[0]
    )
    assert "COALESCE(" in parent_source_query[0]
    model_run_merge = next(
        event for event in connection.events if "MERGE pricing.MODEL_RUN" in event[0]
    )
    assert model_run_merge[1]["validation_source_model_run_id"] == expected_source
    assert "validation_source_model_run_id = :validation_source_model_run_id" in model_run_merge[0]


@pytest.mark.parametrize("parent_model_run_id", [409, None])
def test_record_model_run_replaces_complete_mutable_lineage_snapshot(
    parent_model_run_id,
):
    connection = _Connection()

    record_model_run(
        None,
        connection=connection,
        build=_approved_build(),
        dag_id="dag",
        airflow_run_id="scheduled__2026-07-12",
        rate_package_id=43,
        parent_model_run_id=parent_model_run_id,
    )

    split_cleanup = next(event for event in connection.events if "DELETE split_link" in event[0])
    dataset_cleanup = next(
        event for event in connection.events if "DELETE dataset_link" in event[0]
    )
    assert "parent_split.model_run_id = :parent_model_run_id" in split_cleanup[0]
    assert "parent_split.manifest_id = split_link.manifest_id" in split_cleanup[0]
    assert "parent_split.split_set_id = split_link.split_set_id" in split_cleanup[0]
    assert "parent_split.dataset_role = split_link.dataset_role" in split_cleanup[0]
    assert "parent_split.split_role = split_link.split_role" in split_cleanup[0]
    assert "parent_dataset.model_run_id = :parent_model_run_id" in dataset_cleanup[0]
    assert "parent_dataset.manifest_id = dataset_link.manifest_id" in dataset_cleanup[0]
    assert "parent_dataset.dataset_role = dataset_link.dataset_role" in dataset_cleanup[0]
    assert split_cleanup[1]["parent_model_run_id"] == parent_model_run_id
    assert dataset_cleanup[1]["parent_model_run_id"] == parent_model_run_id
    model_run_merge = next(
        event for event in connection.events if "MERGE pricing.MODEL_RUN" in event[0]
    )
    assert "rating_workbook_sha256 = :rating_workbook_sha256" in model_run_merge[0]
    assert model_run_merge[1]["rating_workbook_sha256"] == "a" * 64


def test_record_model_run_rejects_parent_run_from_another_parent_package():
    connection = _Connection(parent_matches=False)

    with pytest.raises(
        RuntimeError,
        match="parent_model_run_id does not match the package parent",
    ):
        record_model_run(
            None,
            connection=connection,
            build=_approved_build(),
            dag_id="dag",
            airflow_run_id="scheduled__2026-07-12",
            rate_package_id=43,
            parent_model_run_id=409,
        )


def test_record_model_run_rejects_package_ownership_mismatch_before_model_run_write():
    connection = _Connection(package_overrides={"model_id": 99})

    with pytest.raises(RuntimeError, match="rate package ownership.*model_id"):
        record_model_run(
            None,
            connection=connection,
            build=_approved_build(),
            dag_id="dag",
            airflow_run_id="scheduled__2026-07-12",
            rate_package_id=43,
        )

    assert not any("MERGE pricing.MODEL_RUN" in sql for sql, _ in connection.events)


def test_record_model_run_rejects_cv_fold_geometry_mismatch_before_model_run_write():
    connection = _Connection(
        fold_rows=(
            {
                "manifest_id": "manifest-2",
                "fold_no": 1,
                "n_train": 80,
                "n_test": 21,
            },
        )
    )

    with pytest.raises(RuntimeError, match="CV_FOLD geometry"):
        record_model_run(
            None,
            connection=connection,
            build=_approved_build(),
            dag_id="dag",
            airflow_run_id="scheduled__2026-07-12",
            rate_package_id=43,
        )

    assert not any("MERGE pricing.MODEL_RUN" in sql for sql, _ in connection.events)


def test_record_model_run_persists_unavailable_curve_as_status_with_zero_points():
    connection = _Connection()
    build = _approved_build().model_copy(
        update={
            "validation_curve_status": "UNAVAILABLE",
            "validation_curve_reason": "upstream curve comparison failed",
            "validation_curve_points": (),
        }
    )

    record_model_run(
        None,
        connection=connection,
        build=build,
        dag_id="dag",
        airflow_run_id="scheduled__2026-07-12",
        rate_package_id=43,
    )

    model_run_merge = next(
        event for event in connection.events if "MERGE pricing.MODEL_RUN" in event[0]
    )
    assert model_run_merge[1]["validation_curve_status"] == "UNAVAILABLE"
    assert model_run_merge[1]["validation_curve_reason"] == ("upstream curve comparison failed")
    assert not any(
        "INSERT INTO pricing.CV_SPLIT_CURVE_POINT" in sql for sql, _ in connection.events
    )
