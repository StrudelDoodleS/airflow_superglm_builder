from __future__ import annotations

import hashlib
import re


SQL_SERVER_IDENTIFIER_MAX_LENGTH = 128
DEFAULT_RUN_KEY_MAX_LENGTH = 99
DEFAULT_RUN_KEY_DIGEST_LENGTH = 10

_SAFE_RUN_KEY_PATTERN = re.compile(r"[^A-Za-z0-9_]+")


def run_key_for_value(
    value: object | None,
    *,
    max_length: int = DEFAULT_RUN_KEY_MAX_LENGTH,
    digest_length: int = DEFAULT_RUN_KEY_DIGEST_LENGTH,
) -> str:
    if max_length < digest_length + 2:
        raise ValueError("max_length must leave room for a readable prefix and digest")

    raw = "manual" if value is None else str(value).strip()
    compact = raw.replace("-", "").replace(":", "").replace("+", "")
    safe = _SAFE_RUN_KEY_PATTERN.sub("_", compact).strip("_").lower()
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:digest_length]
    suffix = f"_{digest}"
    prefix_length = max(max_length - len(suffix), 1)
    prefix = (safe or "manual")[:prefix_length].rstrip("_") or "manual"
    return f"{prefix}{suffix}"


def scoped_identifier(
    prefix: str,
    value: object | None,
    *,
    max_length: int = SQL_SERVER_IDENTIFIER_MAX_LENGTH,
    digest_length: int = DEFAULT_RUN_KEY_DIGEST_LENGTH,
) -> str:
    safe_prefix = prefix.strip()
    if not safe_prefix:
        raise ValueError("prefix is required")

    run_key_max_length = max_length - len(safe_prefix) - 1
    if run_key_max_length < digest_length + 2:
        raise ValueError("prefix is too long for the requested identifier length")

    return (
        f"{safe_prefix}_"
        f"{run_key_for_value(value, max_length=run_key_max_length, digest_length=digest_length)}"
    )
