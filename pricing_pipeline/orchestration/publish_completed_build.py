from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.infra.db import configure_engine
from pricing_pipeline.infra.schema import schema_names_from_connectable
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import (
    ApprovedModelBuild,
    ApprovedModelBuildError,
)
from pricing_pipeline.orchestration.pipeline import publish_model_export
from pricing_pipeline.publishing.model_registry import validate_registered_model
from pricing_pipeline.publishing.lifecycle import CompletedModelPublishResult
from pricing_pipeline.workbench.artifacts import load_completed_build_candidate_bundle


@dataclass(frozen=True)
class CandidateSQLLineage:
    manifest_id: str
    row_count: int
    pk_columns: tuple[str, ...]
    model_frame_sha256: str | None
    split_set_id: str | None
    split_row_order_sha256: str | None


def publish_completed_model_build(
    engine,
    *,
    settings: Settings,
    model_config: ModelBuildConfig,
    completed_build: ApprovedModelBuild,
) -> CompletedModelPublishResult:
    engine = configure_engine(engine, settings.schema_names)
    if not isinstance(completed_build, ApprovedModelBuild):
        raise TypeError("completed_build must be an ApprovedModelBuild")
    build = completed_build
    with engine.begin() as connection:
        model_id = validate_registered_model(connection, model_config).model_id
    candidate_sql_lineage = load_candidate_sql_lineage(
        engine,
        manifest_id=build.manifest_id,
        split_set_id=build.split_set_id,
    )
    _verify_candidate_artifact(
        build,
        sql_lineage=candidate_sql_lineage,
        allowed_root=settings.workbench_artifact_root,
    )
    publish_result = publish_model_export(
        engine,
        build,
        model_config=model_config,
        validated_model_id=model_id,
        allowed_artifact_root=settings.workbench_artifact_root,
    )

    _discard_redundant_completed_build_attempt(
        build,
        publish_result=publish_result,
        artifact_root=settings.workbench_artifact_root,
    )
    return publish_result


def load_candidate_sql_lineage(
    engine,
    *,
    manifest_id: str,
    split_set_id: str | None,
) -> CandidateSQLLineage:
    schemas = schema_names_from_connectable(engine)
    with engine.begin() as con:
        manifest_row = (
            con.execute(
                text(
                    f"""
                    SELECT manifest_id, row_count, pk_columns_json, model_frame_sha256
                    FROM {schemas.pricing}.DATASET_MANIFEST
                    WHERE manifest_id = :manifest_id
                    """
                ),
                {"manifest_id": manifest_id},
            )
            .mappings()
            .one_or_none()
        )
        if manifest_row is None:
            raise ApprovedModelBuildError(f"manifest_id {manifest_id!r} was not found")

        split_row = None
        if split_set_id is not None:
            split_row = (
                con.execute(
                    text(
                        f"""
                        SELECT
                            split_set_id,
                            manifest_id,
                            row_count,
                            row_order_sha256
                        FROM {schemas.pricing}.CV_SPLIT_SET
                        WHERE split_set_id = :split_set_id
                        """
                    ),
                    {"split_set_id": split_set_id},
                )
                .mappings()
                .one_or_none()
            )
        else:
            split_row = (
                con.execute(
                    text(
                        f"""
                        SELECT split_set_id
                        FROM {schemas.pricing}.CV_SPLIT_SET
                        WHERE manifest_id = :manifest_id
                        """
                    ),
                    {"manifest_id": manifest_id},
                )
                .mappings()
                .first()
            )

    try:
        raw_pk_columns = json.loads(str(manifest_row["pk_columns_json"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ApprovedModelBuildError(
            f"manifest_id {manifest_id!r} has invalid pk_columns_json"
        ) from exc
    if not isinstance(raw_pk_columns, list) or not all(
        isinstance(column, str) and column for column in raw_pk_columns
    ):
        raise ApprovedModelBuildError(
            f"manifest_id {manifest_id!r} has invalid pk_columns_json"
        )

    if split_set_id is not None:
        if split_row is None:
            raise ApprovedModelBuildError(f"split_set_id {split_set_id!r} was not found")
        if split_row["manifest_id"] != manifest_id:
            raise ApprovedModelBuildError(
                f"split_set_id {split_set_id!r} does not belong to manifest_id {manifest_id!r}"
            )
        if int(split_row["row_count"]) != int(manifest_row["row_count"]):
            raise ApprovedModelBuildError(
                f"split_set_id {split_set_id!r} row count does not match manifest_id "
                f"{manifest_id!r}"
            )
    elif split_row is not None:
        raise ApprovedModelBuildError(
            "candidate artifact omits split_set_id but SQL manifest owns split_set_id "
            f"{split_row['split_set_id']!r}"
        )

    return CandidateSQLLineage(
        manifest_id=manifest_id,
        row_count=int(manifest_row["row_count"]),
        pk_columns=tuple(raw_pk_columns),
        model_frame_sha256=(
            None
            if manifest_row["model_frame_sha256"] is None
            else str(manifest_row["model_frame_sha256"])
        ),
        split_set_id=split_set_id,
        split_row_order_sha256=(None if split_row is None else str(split_row["row_order_sha256"])),
    )


def _verify_candidate_artifact(
    build: ApprovedModelBuild,
    *,
    sql_lineage: CandidateSQLLineage,
    allowed_root: str | Path,
) -> None:
    try:
        bundle = load_completed_build_candidate_bundle(
            build,
            allowed_root=allowed_root,
        )
    except Exception as exc:
        raise ApprovedModelBuildError(f"candidate artifact verification failed: {exc}") from exc
    if build.model_frame_sha256 != sql_lineage.model_frame_sha256:
        raise ApprovedModelBuildError(
            "completed-build model_frame_sha256 does not match SQL manifest lineage: "
            f"expected={sql_lineage.model_frame_sha256!r}, "
            f"actual={build.model_frame_sha256!r}"
        )
    if bundle.pk_columns != sql_lineage.pk_columns:
        raise ApprovedModelBuildError(
            "candidate artifact pk_columns do not match SQL manifest lineage: "
            f"expected={sql_lineage.pk_columns!r}, actual={bundle.pk_columns!r}"
        )
    if len(bundle.X) != sql_lineage.row_count:
        raise ApprovedModelBuildError(
            "candidate artifact row count does not match SQL manifest lineage: "
            f"expected={sql_lineage.row_count}, actual={len(bundle.X)}"
        )
    if (
        sql_lineage.split_set_id is not None
        and bundle.row_order_sha256 != sql_lineage.split_row_order_sha256
    ):
        raise ApprovedModelBuildError(
            "candidate artifact row_order_sha256 does not match SQL split lineage: "
            f"expected={sql_lineage.split_row_order_sha256!r}, "
            f"actual={bundle.row_order_sha256!r}"
        )


def _discard_redundant_completed_build_attempt(
    build: ApprovedModelBuild,
    *,
    publish_result: CompletedModelPublishResult,
    artifact_root: str | Path,
) -> None:
    if not publish_result.was_existing:
        return
    root = Path(artifact_root).expanduser().resolve()
    incoming_values = [
        build.rating_workbook_path,
        build.publication_receipt_path,
        build.candidate_artifact_path,
    ]
    incoming_paths = [
        Path(value).expanduser().resolve() for value in incoming_values if value is not None
    ]
    if not incoming_paths:
        return
    attempt_dir = incoming_paths[0].parent
    if any(path.parent != attempt_dir for path in incoming_paths):
        return
    if not attempt_dir.is_relative_to(root):
        return
    relative_attempt = attempt_dir.relative_to(root)
    if len(relative_attempt.parts) < 3:
        return
    canonical_values = [
        publish_result.rating_workbook_path,
        publish_result.publication_receipt_path,
    ]
    canonical_paths = [
        Path(str(value)).expanduser().resolve()
        for value in canonical_values
        if value is not None and str(value).strip()
    ]
    if any(path == attempt_dir or path.is_relative_to(attempt_dir) for path in canonical_paths):
        return
    if attempt_dir.is_symlink() or not attempt_dir.is_dir():
        return
    shutil.rmtree(attempt_dir)
