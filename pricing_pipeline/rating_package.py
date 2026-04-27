from __future__ import annotations

from pathlib import Path

from scripts.load_staging_to_rating_package import publish_rating_package as _publish
from scripts.load_superglm_excel_to_staging import stage_rating_export as _stage


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
