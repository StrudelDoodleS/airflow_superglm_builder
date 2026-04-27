from __future__ import annotations

from importlib import import_module
import os
import sys
from pathlib import Path


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
_stage = import_module("scripts.load_superglm_excel_to_staging").stage_rating_export


def stage_rating_export(
    engine,
    *,
    workbook_path: Path,
    export_id: str,
    model_name: str,
    model_version: str | None,
    effective_from: str,
    effective_to: str | None = None,
    created_by: str = "python",
    replace: bool = False,
) -> None:
    _stage(
        engine,
        workbook_path=workbook_path,
        export_id=export_id,
        model_name=model_name,
        model_version=model_version,
        effective_from=effective_from,
        effective_to=effective_to,
        created_by=created_by,
        replace=replace,
    )


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
