from __future__ import annotations

import re

from sqlalchemy import text

from pricing_pipeline.infra.schema import schema_names_from_connectable


_VERSION_PATTERN = re.compile(r"^v([0-9]+)$")


def _required_text(value: str, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def existing_model_version_for_export(
    engine,
    *,
    model_key: str,
    export_id: str,
) -> str | None:
    model_key = _required_text(model_key, "model_key")
    export_id = _required_text(export_id, "export_id")
    schemas = schema_names_from_connectable(engine)

    with engine.begin() as con:
        version = con.execute(
            text(
                f"""
                SELECT TOP (1) rp.model_version
                FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp
                JOIN {schemas.pricing}.PRICING_MODEL AS pm
                  ON pm.model_id = rp.model_id
                WHERE pm.model_key = :model_key
                  AND rp.source_export_id = :export_id
                ORDER BY rp.rate_package_id DESC
                """
            ),
            {"model_key": model_key, "export_id": export_id},
        ).scalar()
    return None if version is None else str(version)


def next_trained_model_version(engine, *, model_key: str) -> str:
    model_key = _required_text(model_key, "model_key")
    schemas = schema_names_from_connectable(engine)

    with engine.begin() as con:
        versions = list(
            con.execute(
                text(
                    f"""
                    SELECT rp.model_version
                    FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp
                    JOIN {schemas.pricing}.PRICING_MODEL AS pm
                      ON pm.model_id = rp.model_id
                    WHERE pm.model_key = :model_key
                      AND rp.parent_rate_package_id IS NULL
                    """
                ),
                {"model_key": model_key},
            ).scalars()
        )

    version_numbers = []
    for version in versions:
        match = _VERSION_PATTERN.match(str(version))
        if match is not None:
            version_numbers.append(int(match.group(1)))
    return f"v{max(version_numbers, default=0) + 1}"


def resolve_model_version_for_export(
    engine,
    *,
    model_key: str,
    export_id: str,
) -> str:
    existing = existing_model_version_for_export(
        engine,
        model_key=model_key,
        export_id=export_id,
    )
    if existing is not None:
        return existing
    return next_trained_model_version(engine, model_key=model_key)
