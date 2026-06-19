from __future__ import annotations

import re
from pathlib import Path

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None

from pricing_pipeline.infra.mlflow_tracking import optional_mlflow_client

_DEFAULT_MLFLOW_CLIENT = object()


def build_export_id(model_name: str, run_id: str) -> str:
    raw = f"{model_name}__{run_id}".lower()
    raw = raw.replace("-", "").replace(":", "").replace("+", "")
    safe = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    return safe or "rating_export"


def build_rating_export_path(
    root: Path,
    model_name: str,
    logical_date: str,
    export_id: str,
) -> Path:
    return root / model_name / logical_date / export_id / "rating_tables.xlsx"


def export_rating_tables(
    model,
    X,
    y,
    exposure,
    output_path: Path,
    *,
    offset=None,
    offset_source=None,
    offset_name: str | None = None,
    offset_kind: str | None = None,
    offset_max_exact_levels: int | None = None,
    n_bins: int = 150,
    mlflow_client=_DEFAULT_MLFLOW_CLIENT,
) -> Path:
    export_fn = getattr(model, "export_rating_tables", None)
    if not callable(export_fn):
        raise RuntimeError(
            "SuperGLM rating table export support is required. Install a SuperGLM "
            "version that includes PR #109 and exposes model.export_rating_tables()."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {"sample_weight": exposure, "n_bins": n_bins}
    optional_export_kwargs = {
        "offset": offset,
        "offset_source": offset_source,
        "offset_name": offset_name,
        "offset_kind": offset_kind,
        "offset_max_exact_levels": offset_max_exact_levels,
    }
    export_kwargs.update(
        {
            key: value
            for key, value in optional_export_kwargs.items()
            if value is not None
        }
    )
    export_fn(output_path, X, y, **export_kwargs)
    resolved_mlflow_client = (
        mlflow if mlflow_client is _DEFAULT_MLFLOW_CLIENT else mlflow_client
    )
    optional_mlflow_client(resolved_mlflow_client).log_artifact(
        str(output_path),
        artifact_path="rating_tables",
    )
    return output_path
