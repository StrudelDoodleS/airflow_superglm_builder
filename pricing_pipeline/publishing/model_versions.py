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


def resolve_model_version_for_export(
    engine,
    *,
    model_name: str,
    export_id: str,
) -> str:
    model_name = _required_text(model_name, "model_name")
    export_id = _required_text(export_id, "export_id")
    schemas = schema_names_from_connectable(engine)

    with engine.begin() as con:
        model_id = con.execute(
            text(
                f"""
                SELECT pm.model_id
                FROM {schemas.pricing}.PRICING_MODEL AS pm WITH (UPDLOCK, HOLDLOCK)
                WHERE pm.model_name = :model_name
                """
            ),
            {"model_name": model_name},
        ).scalar_one_or_none()
        if model_id is None:
            raise ValueError(f"pricing model is not registered: {model_name}")

        existing_versions = list(
            con.execute(
                text(
                    f"""
                    SELECT existing.model_version
                    FROM (
                        SELECT rp.model_version
                        FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp
                        WHERE rp.model_id = :model_id
                          AND rp.source_export_id = :export_id
                        UNION
                        SELECT reservation.model_version
                        FROM {schemas.pricing}.PRICING_MODEL_VERSION_RESERVATION AS reservation
                        WHERE reservation.model_id = :model_id
                          AND reservation.export_id = :export_id
                    ) AS existing
                    """
                ),
                {"model_id": int(model_id), "export_id": export_id},
            ).scalars()
        )
        if len(existing_versions) > 1:
            raise RuntimeError(
                "published package and model-version reservation disagree for "
                f"model={model_name!r}, export_id={export_id!r}"
            )
        if existing_versions:
            return str(existing_versions[0])

        versions = list(
            con.execute(
                text(
                    f"""
                    SELECT rp.model_version
                    FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp
                    WHERE rp.model_id = :model_id
                      AND rp.parent_rate_package_id IS NULL
                    UNION ALL
                    SELECT reservation.model_version
                    FROM {schemas.pricing}.PRICING_MODEL_VERSION_RESERVATION AS reservation
                    WHERE reservation.model_id = :model_id
                    """
                ),
                {"model_id": int(model_id)},
            ).scalars()
        )
        version_numbers = []
        for version in versions:
            match = _VERSION_PATTERN.match(str(version))
            if match is not None:
                version_numbers.append(int(match.group(1)))
        reserved = f"v{max(version_numbers, default=0) + 1}"
        con.execute(
            text(
                f"""
                INSERT INTO {schemas.pricing}.PRICING_MODEL_VERSION_RESERVATION (
                    model_id,
                    export_id,
                    model_version
                ) VALUES (
                    :model_id,
                    :export_id,
                    :model_version
                )
                """
            ),
            {
                "model_id": int(model_id),
                "export_id": export_id,
                "model_version": reserved,
            },
        )
        return reserved
