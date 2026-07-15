from __future__ import annotations

from inspect import signature

import pytest

from pricing_pipeline.models.spec import ApprovedModelBuild
from pricing_pipeline.publishing.lineage import record_model_run


class _Result:
    def __init__(self, scalar=None):
        self.scalar = scalar

    def mappings(self):
        return self

    def one_or_none(self):
        return None

    def scalar_one(self):
        return 501

    def scalar_one_or_none(self):
        return self.scalar


class _Connection:
    def __init__(self, *, parent_matches=True):
        self.events = []
        self.parent_matches = parent_matches

    def execute(self, statement, params=None):
        sql = str(statement)
        self.events.append((sql, params))
        if "child_package.parent_rate_package_id" in sql:
            return _Result(scalar=1 if self.parent_matches else None)
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
        candidate_artifact_format="superglm-candidate-joblib-v1",
        candidate_artifact_size_bytes=321,
        candidate_python_version="3.14.4",
        candidate_superglm_version="0.11.0",
        model_source_sha256="d" * 64,
        model_frame_sha256="e" * 64,
        metrics={"deviance": 0.42},
        metric_scopes={"deviance": "cv"},
        fold_metrics=(
            {"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},
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
    assert model_run_merge[1]["run_status"] == "SUCCESS"
    assert any("MERGE mlops.MODEL_RUN_METRIC" in sql for sql, _ in connection.events)
    assert any("MERGE pricing.CV_FOLD_METRIC" in sql for sql, _ in connection.events)
    assert list(signature(record_model_run).parameters) == [
        "engine",
        "build",
        "dag_id",
        "airflow_run_id",
        "rate_package_id",
        "parent_model_run_id",
        "connection",
    ]


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
