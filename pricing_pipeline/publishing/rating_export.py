from __future__ import annotations

import re
from pathlib import Path

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None

from pricing_pipeline.infra.mlflow_tracking import optional_mlflow_client


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


def export_rating_tables(model, X, y, exposure, output_path: Path) -> Path:
    export_fn = getattr(model, "export_rating_tables", None)
    if not callable(export_fn):
        raise RuntimeError(
            "SuperGLM rating table export support is required. Install a SuperGLM "
            "version that includes PR #109 and exposes model.export_rating_tables()."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_fn(output_path, X, y, sample_weight=exposure, n_bins=150)
    optional_mlflow_client(mlflow).log_artifact(
        str(output_path),
        artifact_path="rating_tables",
    )
    return output_path
