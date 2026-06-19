from __future__ import annotations

import re


def clean_identifier(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"
