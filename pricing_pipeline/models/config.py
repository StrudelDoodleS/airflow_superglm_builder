from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class ValidationSplitConfig:
    method: str = "kfold"
    n_splits: int | None = 5
    test_size: float | None = None
    random_state: int | None = 42
    shuffle: bool = True
    stratify_column: str | None = None
    materialize: bool = False

    @classmethod
    def kfold(
        cls,
        *,
        n_splits: int = 5,
        random_state: int | None = 42,
        shuffle: bool = True,
        materialize: bool = False,
    ) -> "ValidationSplitConfig":
        return cls(
            method="kfold",
            n_splits=n_splits,
            random_state=random_state,
            shuffle=shuffle,
            materialize=materialize,
        )

    @classmethod
    def train_test_split(
        cls,
        *,
        test_size: float = 0.2,
        random_state: int | None = 42,
        shuffle: bool = True,
        stratify_column: str | None = None,
        materialize: bool = False,
    ) -> "ValidationSplitConfig":
        return cls(
            method="train_test_split",
            n_splits=None,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify_column=stratify_column,
            materialize=materialize,
        )

    @classmethod
    def none(cls) -> "ValidationSplitConfig":
        return cls(
            method="none",
            n_splits=None,
            test_size=None,
            random_state=None,
            shuffle=False,
            materialize=False,
        )


@dataclass(frozen=True)
class ModelBuildConfig:
    model_key: str
    model_label: str
    target_name: str
    model_type: str
    deployment_slot: str
    default_package_status: str = "PUBLISHED"
    validation_split: ValidationSplitConfig = field(default_factory=ValidationSplitConfig.kfold)


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


def _optional_string(data: dict[str, Any], field: str) -> str | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model config field {field!r} must be a non-empty string or null")
    return value.strip()


def _bool_value(data: dict[str, Any], field: str, default: bool) -> bool:
    value = data.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"model config field {field!r} must be true or false")
    return value


def _int_value(data: dict[str, Any], field: str, default: int | None) -> int | None:
    value = data.get(field, default)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"model config field {field!r} must be an integer")
    return value


def _float_value(data: dict[str, Any], field: str, default: float | None) -> float | None:
    value = data.get(field, default)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ValueError(f"model config field {field!r} must be numeric")
    return float(value)


def _validation_split_config(data: dict[str, Any]) -> ValidationSplitConfig:
    split = data.get("validation_split", {})
    if not split:
        return ValidationSplitConfig.kfold()
    if not isinstance(split, dict):
        raise ValueError("model config field 'validation_split' must be a table")

    method = _require_non_empty_string(split, "method")
    if method == "none":
        return ValidationSplitConfig.none()
    if method == "kfold":
        n_splits = _int_value(split, "n_splits", 5)
        if n_splits is None or n_splits < 2:
            raise ValueError("validation_split.n_splits must be at least 2")
        return ValidationSplitConfig.kfold(
            n_splits=n_splits,
            random_state=_int_value(split, "random_state", 42),
            shuffle=_bool_value(split, "shuffle", True),
            materialize=_bool_value(split, "materialize", False),
        )
    if method == "train_test_split":
        test_size = _float_value(split, "test_size", 0.2)
        if test_size is None or not 0 < test_size < 1:
            raise ValueError("validation_split.test_size must be between 0 and 1")
        return ValidationSplitConfig.train_test_split(
            test_size=test_size,
            random_state=_int_value(split, "random_state", 42),
            shuffle=_bool_value(split, "shuffle", True),
            stratify_column=_optional_string(split, "stratify_column"),
            materialize=_bool_value(split, "materialize", False),
        )

    raise ValueError(
        "validation_split.method must be one of: kfold, train_test_split, none"
    )


def load_model_build_config(path: str | Path) -> ModelBuildConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    for field_name in _REQUIRED_FIELDS:
        if field_name not in data:
            raise ValueError(f"model config missing required field {field_name!r}")

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
        validation_split=_validation_split_config(data),
    )
    if config.default_package_status != "PUBLISHED":
        raise ValueError("default_package_status must be 'PUBLISHED' for production builds")
    return config
