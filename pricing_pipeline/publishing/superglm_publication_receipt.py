from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _normalise_json_metadata_value(
    value: Any,
    path: str,
    *,
    allow_frozen: bool = False,
) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} metadata contains a non-finite float")
        return value
    if type(value) is list or (allow_frozen and type(value) is tuple):
        return tuple(
            _normalise_json_metadata_value(
                item,
                f"{path}[{index}]",
                allow_frozen=allow_frozen,
            )
            for index, item in enumerate(value)
        )
    if type(value) is dict or (allow_frozen and isinstance(value, Mapping)):
        normalised: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} metadata keys must be strings")
            normalised[key] = _normalise_json_metadata_value(
                item,
                f"{path}.{key}",
                allow_frozen=allow_frozen,
            )
        return MappingProxyType(normalised)
    raise ValueError(f"{path} metadata must contain only JSON-native values")


def _normalise_metadata_object(
    value: Any,
    field_name: str,
    *,
    allow_frozen: bool = False,
) -> Mapping[str, Any]:
    if type(value) is not dict and not (allow_frozen and isinstance(value, Mapping)):
        raise ValueError(f"{field_name} metadata must be a JSON object")
    return _normalise_json_metadata_value(value, field_name, allow_frozen=allow_frozen)


def _normalise_term_metadata(
    value: Any,
    *,
    allow_frozen: bool = False,
) -> Mapping[str, Mapping[str, Any]]:
    if type(value) is not dict and not (allow_frozen and isinstance(value, Mapping)):
        raise ValueError("term_metadata metadata must be a JSON object")

    terms: dict[str, Mapping[str, Any]] = {}
    for term_name, metadata in value.items():
        if not isinstance(term_name, str):
            raise ValueError("term_metadata metadata keys must be strings")
        if type(metadata) is not dict and not (allow_frozen and isinstance(metadata, Mapping)):
            raise ValueError(f"term_metadata.{term_name} metadata must be a JSON object")
        terms[term_name] = _normalise_json_metadata_value(
            metadata,
            f"term_metadata.{term_name}",
            allow_frozen=allow_frozen,
        )
    return MappingProxyType(terms)


def _json_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_metadata_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_metadata_value(item) for item in value]
    return value


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

    @model_validator(mode="before")
    @classmethod
    def _normalise_metadata_before_validation(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        if "package_metadata" in data:
            data["package_metadata"] = _normalise_metadata_object(
                data["package_metadata"],
                "package_metadata",
            )
        if "term_metadata" in data:
            data["term_metadata"] = _normalise_term_metadata(data["term_metadata"])
        return data

    @model_validator(mode="after")
    def _freeze_metadata_after_validation(self) -> "SuperGLMPublicationReceipt":
        object.__setattr__(
            self,
            "package_metadata",
            _normalise_metadata_object(
                self.package_metadata,
                "package_metadata",
                allow_frozen=True,
            ),
        )
        object.__setattr__(
            self,
            "term_metadata",
            _normalise_term_metadata(self.term_metadata, allow_frozen=True),
        )
        return self

    @field_serializer("package_metadata", "term_metadata")
    def _serialise_metadata(self, value: Any) -> Any:
        return _json_metadata_value(value)


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("receipt contains a non-finite float")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item)


def canonical_receipt_bytes(receipt: SuperGLMPublicationReceipt) -> bytes:
    _reject_non_finite(receipt.package_metadata)
    _reject_non_finite(receipt.term_metadata)
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
