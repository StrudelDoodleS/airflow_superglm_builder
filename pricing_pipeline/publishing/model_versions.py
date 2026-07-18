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


def _required_sha256(value: str, field_name: str) -> str:
    digest = _required_text(value, field_name)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


def resolve_model_version_for_export(
    engine,
    *,
    model_name: str,
    export_id: str,
    build_fingerprint_sha256: str,
    expected_database: str,
) -> str:
    model_name = _required_text(model_name, "model_name")
    export_id = _required_text(export_id, "export_id")
    build_fingerprint_sha256 = _required_sha256(
        build_fingerprint_sha256,
        "build_fingerprint_sha256",
    )
    schemas = schema_names_from_connectable(engine)

    with engine.begin() as con:
        _verify_expected_database(con, expected_database)
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

        canonical_version = con.execute(
            text(
                f"""
                SELECT rp.model_version
                FROM {schemas.pricing}.PRICING_RATE_PACKAGE AS rp
                    WITH (UPDLOCK, HOLDLOCK)
                WHERE rp.model_id = :model_id
                  AND rp.parent_rate_package_id IS NULL
                  AND rp.build_fingerprint_sha256 = :build_fingerprint_sha256
                """
            ),
            {
                "model_id": int(model_id),
                "build_fingerprint_sha256": build_fingerprint_sha256,
            },
        ).scalar_one_or_none()

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
        if canonical_version is not None:
            if existing_versions and str(existing_versions[0]) != str(canonical_version):
                raise RuntimeError(
                    "canonical root package and export reservation disagree for "
                    f"model={model_name!r}, export_id={export_id!r}, "
                    f"build_fingerprint_sha256={build_fingerprint_sha256!r}"
                )
            return str(canonical_version)
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


def _verify_expected_database(connection, expected_database: str) -> None:
    expected = _required_text(expected_database, "expected_database")
    actual_value = connection.execute(text("SELECT DB_NAME()")).scalar_one_or_none()
    actual = str(actual_value or "").strip()
    if not actual:
        raise RuntimeError("Remote connection did not report a database name")
    if actual.casefold() != expected.casefold():
        raise RuntimeError(
            "Remote database mismatch: "
            f"expected {expected!r}, connected to {actual!r}; "
            "model-version reservation transaction aborted before writes"
        )
