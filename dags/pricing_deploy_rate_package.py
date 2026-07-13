from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime

from airflow.sdk import dag, get_current_context, task

from pricing_models.registry import get_model_config
from pricing_pipeline.infra.runtime import runtime_from_env_or_module
from pricing_pipeline.publishing.publisher import ModelPublisher


_PARAMS = {
    "model_name": "",
    "rate_package_id": None,
    "package_version": None,
    "expected_current_rate_package_id": None,
    "deployment_slot": None,
    "deployment_reason": "",
    "deployed_by": "",
}


def get_engine():
    return runtime_from_env_or_module(env=os.environ).get_engine()


def _optional_value(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _optional_int(value: object, field_name: str) -> int | None:
    value = _optional_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, str):
        cleaned = value.strip()
        unsigned = cleaned[1:] if cleaned[:1] in {"+", "-"} else cleaned
        if unsigned.isdigit():
            return int(cleaned)
    raise ValueError(f"{field_name} must be an integer")


def _optional_text(value: object) -> str | None:
    value = _optional_value(value)
    if value is None:
        return None
    return str(value).strip()


def _text_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _string_id(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def deployment_params_from_context(context: Mapping[str, object]) -> dict[str, object]:
    params = dict(context.get("params") or {})
    dag_run = context.get("dag_run")
    triggering_user_name = getattr(dag_run, "triggering_user_name", None)
    if triggering_user_name and str(triggering_user_name).strip():
        params["deployed_by"] = str(triggering_user_name).strip()
    return params


def deploy_rate_package_from_params(params: Mapping[str, object]) -> dict[str, str]:
    model_name = str(params.get("model_name", "")).strip()
    if not model_name:
        raise ValueError("model_name is required")

    rate_package_id = _optional_int(params.get("rate_package_id"), "rate_package_id")
    package_version = _optional_int(params.get("package_version"), "package_version")
    if (rate_package_id is None) == (package_version is None):
        raise ValueError("provide exactly one of rate_package_id or package_version")
    if "expected_current_rate_package_id" not in params:
        raise ValueError("expected_current_rate_package_id is required")
    expected_current_rate_package_id = _optional_int(
        params.get("expected_current_rate_package_id"),
        "expected_current_rate_package_id",
    )

    config = get_model_config(model_name)
    result = ModelPublisher(get_engine(), config).deploy(
        rate_package_id=rate_package_id,
        package_version=package_version,
        expected_current_rate_package_id=expected_current_rate_package_id,
        deployment_slot=_optional_text(params.get("deployment_slot")),
        deployment_reason=_text_value(params.get("deployment_reason")),
        deployed_by=_text_value(params.get("deployed_by")),
    )

    return {
        "model_name": config.model_name,
        "deployment_slot": result.deployment_slot,
        "previous_rate_package_id": _string_id(result.previous_rate_package_id),
        "rate_package_id": str(result.rate_package_id),
        "package_version": str(result.package_version),
        "deployed_by": result.deployed_by,
        "deployment_reason": result.deployment_reason,
    }


@dag(
    dag_id="pricing_deploy_rate_package",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["pricing", "deploy"],
    params=_PARAMS,
)
def _pricing_deploy_rate_package():
    @task
    def deploy_package() -> dict[str, str]:
        return deploy_rate_package_from_params(
            deployment_params_from_context(get_current_context())
        )

    deploy_package()


pricing_deploy_rate_package = _pricing_deploy_rate_package()
