from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import text

from pricing_pipeline.models.config import ModelBuildConfig


class _Executable(Protocol):
    def execute(self, statement, params=None): ...


class ModelRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class PricingModelRecord:
    model_id: int
    model_name: str
    model_label: str | None
    target_name: str
    model_type: str
    model_status: str


def get_pricing_model(con: _Executable, model_name: str) -> PricingModelRecord | None:
    row = (
        con.execute(
            text(
                """
                SELECT model_id,
                    model_name,
                    model_label,
                    target_name,
                    model_type,
                    model_status
                FROM pricing.PRICING_MODEL
                WHERE model_name = :model_name
                """
            ),
            {"model_name": model_name},
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        return None
    return PricingModelRecord(
        model_id=int(row["model_id"]),
        model_name=str(row["model_name"]),
        model_label=row["model_label"],
        target_name=str(row["target_name"]),
        model_type=str(row["model_type"]),
        model_status=str(row["model_status"]),
    )


def validate_registered_model(con: _Executable, config: ModelBuildConfig) -> PricingModelRecord:
    record = get_pricing_model(con, config.model_name)
    if record is None:
        raise ModelRegistryError(
            f"model_name {config.model_name!r} is not registered; "
            "run explicit model registration first"
        )

    mismatches: list[str] = []
    if record.model_label != config.model_label:
        mismatches.append(f"model_label db={record.model_label!r} config={config.model_label!r}")
    if record.target_name != config.target_name:
        mismatches.append(f"target_name db={record.target_name!r} config={config.target_name!r}")
    if record.model_type != config.model_type:
        mismatches.append(f"model_type db={record.model_type!r} config={config.model_type!r}")
    if record.model_status != "ACTIVE":
        mismatches.append(f"model_status db={record.model_status!r} expected='ACTIVE'")

    if mismatches:
        raise ModelRegistryError(
            f"registered model {config.model_name!r} does not match config: "
            + "; ".join(mismatches)
        )
    return record


def register_pricing_model(
    con: _Executable,
    config: ModelBuildConfig,
    *,
    created_by: str,
) -> PricingModelRecord:
    con.execute(
        text(
            """
            INSERT INTO pricing.PRICING_MODEL (
                model_name,
                model_label,
                target_name,
                model_type,
                model_status,
                created_by
            )
            SELECT
                :model_name,
                :model_label,
                :target_name,
                :model_type,
                'ACTIVE',
                :created_by
            WHERE NOT EXISTS (
                SELECT 1
                FROM pricing.PRICING_MODEL WITH (UPDLOCK, HOLDLOCK)
                WHERE model_name = :model_name
            );
            """
        ),
        {
            "model_name": config.model_name,
            "model_label": config.model_label,
            "target_name": config.target_name,
            "model_type": config.model_type,
            "created_by": created_by,
        },
    )
    return validate_registered_model(con, config)
