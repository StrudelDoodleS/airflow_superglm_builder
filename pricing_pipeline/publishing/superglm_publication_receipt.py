from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class OffsetExportContract(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    handling: Literal["NONE", "EXPORTED_FACTOR", "ALREADY_APPLIED_SQL_EXPOSURE"]
    source_factor_name: str | None = None
    published_factor_name: str | None = None
    source_name: str | None = None
    label: str | None = None

    @field_validator(
        "source_factor_name",
        "published_factor_name",
        "source_name",
        "label",
        mode="before",
    )
    @classmethod
    def _optional_non_empty_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("must be a string")
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def _validate_handling_fields(self) -> "OffsetExportContract":
        factor_fields = ("source_factor_name", "published_factor_name")
        offset_fields = (*factor_fields, "source_name", "label")

        if self.handling == "NONE":
            present = [field for field in offset_fields if getattr(self, field) is not None]
            if present:
                raise ValueError(
                    "offset fields must be null when handling is NONE: " + ", ".join(present)
                )
            return self

        if self.handling == "EXPORTED_FACTOR":
            missing = [field for field in offset_fields if getattr(self, field) is None]
            if missing:
                raise ValueError(
                    "offset fields are required when handling is EXPORTED_FACTOR: "
                    + ", ".join(missing)
                )
            return self

        present_factors = [field for field in factor_fields if getattr(self, field) is not None]
        if present_factors:
            raise ValueError(
                "factor fields must be null when handling is ALREADY_APPLIED_SQL_EXPOSURE: "
                + ", ".join(present_factors)
            )
        missing = [field for field in ("source_name", "label") if getattr(self, field) is None]
        if missing:
            raise ValueError(
                "offset fields are required when handling is ALREADY_APPLIED_SQL_EXPOSURE: "
                + ", ".join(missing)
            )
        return self


class SuperGLMPublicationReceipt(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    schema_name: Literal["superglm_publication_receipt"]
    schema_version: Literal[1]
    metadata_origin: Literal["SUPERGLM_FITTED_MODEL"]
    superglm_version: str
    extractor_version: str
    package_metadata: dict[str, Any]
    term_metadata: dict[str, dict[str, Any]]
    offset_contract: OffsetExportContract

    @field_validator("superglm_version", "extractor_version", mode="before")
    @classmethod
    def _required_non_empty_text(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("is required")
        return value.strip()


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("receipt contains a non-finite float")
    if isinstance(value, dict):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_non_finite(item)


def canonical_receipt_bytes(receipt: SuperGLMPublicationReceipt) -> bytes:
    data = receipt.model_dump(mode="json")
    _reject_non_finite(data)
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def publication_receipt_sha256(receipt: SuperGLMPublicationReceipt) -> str:
    return hashlib.sha256(canonical_receipt_bytes(receipt)).hexdigest()


def write_publication_receipt(receipt: SuperGLMPublicationReceipt, path: str | Path) -> str:
    canonical = canonical_receipt_bytes(receipt)
    digest = hashlib.sha256(canonical).hexdigest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical)
    return digest


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"publication receipt is not strict JSON: {value}")


def load_publication_receipt(
    path: str | Path,
    *,
    expected_sha256: str,
) -> SuperGLMPublicationReceipt:
    if _SHA256_HEX_RE.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 must be 64 lowercase hex characters")

    raw = Path(path).read_bytes()
    try:
        data = json.loads(raw, parse_constant=_reject_json_constant)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("publication receipt is not valid JSON") from exc

    receipt = SuperGLMPublicationReceipt.model_validate(data)
    canonical = canonical_receipt_bytes(receipt)
    if raw != canonical:
        raise ValueError("publication receipt is not canonical")

    digest = hashlib.sha256(canonical).hexdigest()
    if digest != expected_sha256:
        raise ValueError("publication receipt sha256 does not match expected_sha256")

    return receipt
