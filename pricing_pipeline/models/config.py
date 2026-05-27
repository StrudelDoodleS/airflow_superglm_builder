from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class ModelBuildConfig:
    model_key: str
    model_label: str
    target_name: str
    model_type: str
    deployment_slot: str
    default_package_status: str = "PUBLISHED"


_REQUIRED_FIELDS = (
    "model_key",
    "model_label",
    "target_name",
    "model_type",
    "deployment_slot",
    "default_package_status",
)


def _require_non_empty_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model config field {field!r} must be a non-empty string")
    return value.strip()


def load_model_build_config(path: str | Path) -> ModelBuildConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    for field in _REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(f"model config missing required field {field!r}")

    config = ModelBuildConfig(
        model_key=_require_non_empty_string(data, "model_key"),
        model_label=_require_non_empty_string(data, "model_label"),
        target_name=_require_non_empty_string(data, "target_name"),
        model_type=_require_non_empty_string(data, "model_type"),
        deployment_slot=_require_non_empty_string(data, "deployment_slot"),
        default_package_status=_require_non_empty_string(
            data,
            "default_package_status",
        ),
    )
    if config.default_package_status != "PUBLISHED":
        raise ValueError("default_package_status must be 'PUBLISHED' for production builds")
    return config
