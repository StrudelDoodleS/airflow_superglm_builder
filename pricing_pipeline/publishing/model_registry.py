from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import text

from pricing_pipeline.models.config import ModelBuildConfig


class _Executable(Protocol):
    def execute(self, statement, params=None):
        ...


class _Beginable(Protocol):
    def begin(self):
        ...


class ModelRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class PricingModelRecord:
    model_id: int
    model_key: str
    model_label: str | None
    target_name: str
    model_type: str
    model_status: str


def get_pricing_model(con: _Executable, model_key: str) -> PricingModelRecord | None:
    row = (
        con.execute(
            text(
                """
                SELECT model_id,
                    model_key,
                    model_label,
                    target_name,
                    model_type,
                    model_status
                FROM pricing.PRICING_MODEL
                WHERE model_key = :model_key
                """
            ),
            {"model_key": model_key},
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        return None
    return PricingModelRecord(
        model_id=int(row["model_id"]),
        model_key=str(row["model_key"]),
        model_label=row["model_label"],
        target_name=str(row["target_name"]),
        model_type=str(row["model_type"]),
        model_status=str(row["model_status"]),
    )


def validate_registered_model(
    con: _Executable, config: ModelBuildConfig
) -> PricingModelRecord:
    record = get_pricing_model(con, config.model_key)
    if record is None:
        raise ModelRegistryError(
            f"model_key {config.model_key!r} is not registered; "
            "run explicit model registration first"
        )

    mismatches: list[str] = []
    if record.model_label != config.model_label:
        mismatches.append(
            f"model_label db={record.model_label!r} config={config.model_label!r}"
        )
    if record.target_name != config.target_name:
        mismatches.append(
            f"target_name db={record.target_name!r} config={config.target_name!r}"
        )
    if record.model_type != config.model_type:
        mismatches.append(
            f"model_type db={record.model_type!r} config={config.model_type!r}"
        )
    if record.model_status != "ACTIVE":
        mismatches.append(f"model_status db={record.model_status!r} expected='ACTIVE'")

    if mismatches:
        raise ModelRegistryError(
            f"registered model {config.model_key!r} does not match config: "
            + "; ".join(mismatches)
        )
    return record


def register_pricing_model(
    con: _Executable, config: ModelBuildConfig, *, created_by: str
) -> int:
    con.execute(
        text(
            """
            INSERT INTO pricing.PRICING_MODEL (
                model_key,
                model_label,
                target_name,
                model_type,
                model_status,
                created_by
            )
            SELECT
                :model_key,
                :model_label,
                :target_name,
                :model_type,
                'ACTIVE',
                :created_by
            WHERE NOT EXISTS (
                SELECT 1
                FROM pricing.PRICING_MODEL WITH (UPDLOCK, HOLDLOCK)
                WHERE model_key = :model_key
            );
            """
        ),
        {
            "model_key": config.model_key,
            "model_label": config.model_label,
            "target_name": config.target_name,
            "model_type": config.model_type,
            "created_by": created_by,
        },
    )
    return int(
        con.execute(
            text(
                """
                SELECT model_id
                FROM pricing.PRICING_MODEL
                WHERE model_key = :model_key
                """
            ),
            {"model_key": config.model_key},
        ).scalar_one()
    )


def _ensure_pricing_model_on_connection(
    con: _Executable,
    *,
    model_key: str,
    target_name: str,
    model_type: str,
    created_by: str,
    model_label: str | None = None,
    model_status: str = "ACTIVE",
) -> int:
    params = {
        "model_key": model_key,
        "model_label": model_label,
        "target_name": target_name,
        "model_type": model_type,
        "model_status": model_status,
        "created_by": created_by,
    }
    con.execute(
        text(
            """
                MERGE pricing.PRICING_MODEL WITH (HOLDLOCK) AS tgt
                USING (
                    SELECT
                        :model_key AS model_key,
                        :model_label AS model_label,
                        :target_name AS target_name,
                        :model_type AS model_type,
                        :model_status AS model_status,
                        :created_by AS created_by
                ) AS src
                ON tgt.model_key = src.model_key
                WHEN MATCHED THEN
                    UPDATE SET
                        model_label = COALESCE(src.model_label, tgt.model_label),
                        target_name = src.target_name,
                        model_type = src.model_type,
                        model_status = src.model_status
                WHEN NOT MATCHED THEN
                    INSERT (
                        model_key,
                        model_label,
                        target_name,
                        model_type,
                        model_status,
                        created_by
                    )
                    VALUES (
                        src.model_key,
                        src.model_label,
                        src.target_name,
                        src.model_type,
                        src.model_status,
                        src.created_by
                    );
                """
        ),
        params,
    )
    return int(
        con.execute(
            text(
                """
                SELECT model_id
                FROM pricing.PRICING_MODEL
                WHERE model_key = :model_key
                """
            ),
            {"model_key": model_key},
        ).scalar_one()
    )


def ensure_pricing_model(
    bind: _Executable | _Beginable,
    *,
    model_key: str,
    target_name: str,
    model_type: str,
    created_by: str,
    model_label: str | None = None,
    model_status: str = "ACTIVE",
) -> int:
    if hasattr(bind, "execute"):
        return _ensure_pricing_model_on_connection(
            bind,  # type: ignore[arg-type]
            model_key=model_key,
            target_name=target_name,
            model_type=model_type,
            created_by=created_by,
            model_label=model_label,
            model_status=model_status,
        )

    with bind.begin() as con:  # type: ignore[union-attr]
        return _ensure_pricing_model_on_connection(
            con,
            model_key=model_key,
            target_name=target_name,
            model_type=model_type,
            created_by=created_by,
            model_label=model_label,
            model_status=model_status,
        )
