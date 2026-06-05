from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pricing_pipeline.orchestration.completed_build_helpers import effective_from_for_run
from pricing_pipeline.orchestration.run_context import run_key_for_value


def context_logical_date(context: Mapping[str, Any]) -> object | None:
    value = context.get("logical_date")
    if value is not None:
        return value

    dag_run = context.get("dag_run")
    if dag_run is None:
        return None
    return (
        getattr(dag_run, "logical_date", None)
        or getattr(dag_run, "run_after", None)
        or getattr(dag_run, "execution_date", None)
    )


def task_run_metadata(
    context: Mapping[str, Any],
    *,
    output_root: str | Path,
) -> dict[str, str]:
    logical_date = context_logical_date(context)
    run_value = context.get("run_id") or logical_date or "manual"
    run_key = run_key_for_value(run_value)
    effective_from = effective_from_for_run(logical_date)

    return {
        "run_key": run_key,
        "output_dir": str(Path(output_root) / run_key),
        "effective_from": effective_from,
        "data_as_of_date": effective_from,
    }


def merge_prepared_payload_metadata(
    metadata: Mapping[str, str],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if "run_key" in payload and str(payload["run_key"]) != metadata["run_key"]:
        raise ValueError("prepare_source_data returned a run_key that differs from task metadata")

    return {
        **metadata,
        **payload,
        "run_key": metadata["run_key"],
        "output_dir": payload.get("output_dir", metadata["output_dir"]),
        "effective_from": payload.get("effective_from", metadata["effective_from"]),
        "data_as_of_date": payload.get("data_as_of_date", metadata["data_as_of_date"]),
    }
