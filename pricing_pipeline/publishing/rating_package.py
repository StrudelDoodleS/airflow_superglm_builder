from __future__ import annotations

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
    if pointer_name is not None:
        raise ValueError(
            "package publish no longer deploys rate packages; publish the package "
            "first, then deploy it with ModelPublisher.deploy or the deploy DAG"
        )
    return package_writer.publish_rating_package(
        engine,
        export_id=export_id,
        created_by=created_by,
        package_status=package_status,
    )
