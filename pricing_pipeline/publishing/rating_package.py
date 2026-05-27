from __future__ import annotations

from importlib import import_module
import os
import sys
from pathlib import Path

from pricing_pipeline.publishing.staging import stage_rating_export

__all__ = ["publish_rating_package", "stage_rating_export"]


def _ensure_scripts_path() -> None:
    candidates = []
    project_root = os.environ.get("PRICING_PROJECT_ROOT")
    if project_root:
        candidates.append(Path(project_root))
    candidates.extend(
        [
            Path("/opt/pricing"),
            Path(__file__).resolve().parents[1],
        ]
    )

    valid_candidates = [
        candidate for candidate in candidates if (candidate / "scripts").is_dir()
    ]
    for candidate in reversed(valid_candidates):
        if (candidate / "scripts").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_ensure_scripts_path()

_publish = import_module(
    "scripts.load_staging_to_rating_package"
).publish_rating_package


def publish_rating_package(
    engine,
    *,
    export_id: str,
    pointer_name: str | None,
    created_by: str = "python",
    package_status: str = "DRAFT",
) -> int:
    return _publish(
        engine,
        export_id=export_id,
        pointer_name=pointer_name,
        created_by=created_by,
        package_status=package_status,
    )
