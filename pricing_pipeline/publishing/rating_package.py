from __future__ import annotations

import argparse

from pricing_pipeline.publishing import package_writer
from pricing_pipeline.publishing.lifecycle import PublishResult
from pricing_pipeline.publishing.staging import stage_rating_export

__all__ = ["publish_rating_package", "stage_rating_export"]


def publish_rating_package(
    engine,
    *,
    export_id: str,
    pointer_name: str | None = None,
    created_by: str = "python",
    package_status: str = "DRAFT",
) -> PublishResult | int:
    if pointer_name is None:
        return package_writer.publish_rating_package(
            engine,
            export_id=export_id,
            created_by=created_by,
            package_status=package_status,
        )

    args = argparse.Namespace(
        export_id=export_id,
        created_by=created_by,
        package_status=package_status,
        set_pointer=pointer_name,
    )
    return int(package_writer.load_staging_to_rating_package(engine, args))
