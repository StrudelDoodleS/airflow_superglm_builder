from __future__ import annotations

from enum import StrEnum
from typing import Any


class ModelKind(StrEnum):
    """Lifecycle meaning of one fitted/published model run."""

    RAW = "RAW"
    ROUTINE_EDIT = "ROUTINE_EDIT"
    EDITOR_EDIT = "EDITOR_EDIT"


def normalise_model_kind(value: Any) -> str:
    cleaned = str(value or "").strip().upper()
    try:
        return ModelKind(cleaned).value
    except ValueError as exc:
        allowed = ", ".join(kind.value for kind in ModelKind)
        raise ValueError(f"model_kind must be one of: {allowed}") from exc


__all__ = ["ModelKind", "normalise_model_kind"]
