from __future__ import annotations

import re
from pathlib import Path


def build_export_id(model_name: str, run_id: str) -> str:
    raw = f"{model_name}__{run_id}".lower()
    raw = raw.replace("-", "").replace(":", "").replace("+", "")
    safe = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    return safe or "rating_export"


def export_rating_tables(
    model,
    X,
    y,
    export_weight,
    output_path: Path,
    *,
    offset=None,
    offset_source=None,
    offset_name: str | None = None,
    offset_kind: str | None = None,
    offset_max_exact_levels: int | None = None,
    n_bins: int = 150,
) -> Path:
    export_fn = getattr(model, "export_rating_tables", None)
    if not callable(export_fn):
        raise RuntimeError(
            "SuperGLM rating table export support is required. Install a SuperGLM "
            "version that includes PR #109 and exposes model.export_rating_tables()."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {"sample_weight": export_weight, "n_bins": n_bins}
    optional_export_kwargs = {
        "offset": offset,
        "offset_source": offset_source,
        "offset_name": offset_name,
        "offset_kind": offset_kind,
        "offset_max_exact_levels": offset_max_exact_levels,
    }
    export_kwargs.update(
        {key: value for key, value in optional_export_kwargs.items() if value is not None}
    )
    export_fn(output_path, X, y, **export_kwargs)
    return output_path
