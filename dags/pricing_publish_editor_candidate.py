from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from airflow.sdk import dag, get_current_context, task

from pricing_pipeline.infra.runtime import runtime_from_env_or_module
from pricing_pipeline.publishing.editor_candidate import publish_editor_submission


def _required_conf(conf: Mapping[str, Any], name: str) -> str:
    value = conf.get(name)
    cleaned = "" if value is None else str(value).strip()
    if not cleaned:
        raise ValueError(f"dag_run.conf.{name} is required")
    return cleaned


def publish_editor_candidate_from_context(context: Mapping[str, Any]) -> dict[str, Any]:
    dag_run = context["dag_run"]
    conf = dict(dag_run.conf or {})
    submission_path = _required_conf(conf, "submission_path")
    submission_sha256 = _required_conf(conf, "submission_sha256")
    if len(submission_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in submission_sha256
    ):
        raise ValueError("dag_run.conf.submission_sha256 must be 64 lowercase hex characters")

    runtime = runtime_from_env_or_module(env=os.environ)
    dag_obj = context.get("dag")
    dag_id = str(getattr(dag_obj, "dag_id", None) or getattr(dag_run, "dag_id"))
    identity_candidates = (
        context.get("triggering_user_name"),
        getattr(dag_run, "triggering_user_name", None),
    )
    triggering_identity = next(
        (
            str(value).strip()
            for value in identity_candidates
            if value is not None and str(value).strip()
        ),
        None,
    )
    if triggering_identity is None:
        raise ValueError("an authenticated Airflow trigger identity is required")
    result = publish_editor_submission(
        runtime.get_engine(),
        settings=runtime.settings,
        submission_path=submission_path,
        submission_sha256=submission_sha256,
        dag_id=dag_id,
        airflow_run_id=str(dag_run.run_id),
        created_by=triggering_identity,
    )
    return {
        "submission_id": result.submission_id,
        "model_name": result.model_name,
        "parent_rate_package_id": result.parent_rate_package_id,
        "rate_package_id": result.rate_package_id,
        "package_version": result.package_version,
        "model_run_id": result.model_run_id,
        "package_status": result.package_status,
        "was_existing": result.was_existing,
    }


@dag(
    dag_id="pricing_publish_editor_candidate",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["pricing", "superglm", "editor"],
)
def _pricing_publish_editor_candidate():
    @task
    def publish_submission() -> dict[str, Any]:
        return publish_editor_candidate_from_context(get_current_context())

    publish_submission()


pricing_publish_editor_candidate = _pricing_publish_editor_candidate()
