from __future__ import annotations

from typing import Protocol

from sqlalchemy import text


class _Executable(Protocol):
    def execute(self, statement, params=None):
        ...


class _Beginable(Protocol):
    def begin(self):
        ...


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
