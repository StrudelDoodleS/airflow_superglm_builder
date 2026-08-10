from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from superglm import Categorical, Numeric, SuperGLM

from pricing_pipeline.models.config import ValidationSplitConfig
from pricing_pipeline.notebook import (
    PricingModelSpec,
    build_candidate,
    connect,
    publish_candidate,
    register_model,
)
from pricing_pipeline.publishing.equivalence import (
    ModelEquivalenceError,
    ensure_model_equivalence,
    find_equivalent_publication,
)


def _superglm() -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        n_bins=32,
        features={
            "driver_age": Numeric(),
            "area": Categorical(),
        },
    )


def test_local_sqlite_reuses_semantically_identical_model_before_second_staging(
    tmp_path: Path,
):
    model_root = tmp_path / "pricing_models" / "local_equivalence"
    model_root.mkdir(parents=True)
    (model_root / "02_model_training.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": ["MODEL = 'LOCAL_EQUIVALENCE'\n"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    row_count = 90
    rng = np.random.default_rng(20260807)
    frame = pd.DataFrame(
        {
            "policy_id": np.arange(1, row_count + 1),
            "driver_age": rng.integers(18, 80, row_count).astype(float),
            "area": np.resize(["A", "B", "C"], row_count),
            "data_as_of": ["2026-06-30"] * row_count,
        }
    )
    eta = (
        -1.0
        + 0.006 * (frame["driver_age"] - 40)
        + frame["area"].map({"A": 0.0, "B": 0.2, "C": -0.1})
    )
    frame["claim_count"] = rng.poisson(np.exp(eta))

    pricing = connect(mode="local", local_root=model_root / ".local")
    model = register_model(
        pricing,
        PricingModelSpec(
            name="LOCAL_EQUIVALENCE",
            label="Local equivalence",
            target="claim_count",
            model_type="superglm_poisson",
            deployment_slot="LOCAL_EQUIVALENCE_UAT",
            features=("driver_age", "area"),
            dataset_name="local_equivalence_frame",
            source_system="unit_test",
            pk_columns=("policy_id",),
            data_as_of_column="data_as_of",
            validation=ValidationSplitConfig.kfold(
                n_splits=3,
                random_state=42,
                shuffle=True,
            ),
            fit_mode="fit",
        ),
        source_root=model_root,
        created_by="local-test",
    )

    def build(kind: str):
        return build_candidate(
            pricing,
            model=model,
            frame=frame.copy(),
            superglm_model=_superglm(),
            model_kind=kind,
            created_by="local-test",
        )

    first_raw_candidate = build("RAW")
    first_raw = publish_candidate(pricing, first_raw_candidate)

    fingerprinted_raw = ensure_model_equivalence(first_raw_candidate.completed_build)
    different_split = fingerprinted_raw.model_copy(
        update={"split_set_id": "different-validation-split"}
    )
    equivalent = find_equivalent_publication(
        pricing.engine,
        build=different_split,
    )
    assert equivalent is not None
    assert equivalent.model_run_id == first_raw.model_run_id
    assert equivalent.split_set_id == first_raw.split_set_id

    different_kind = fingerprinted_raw.model_copy(update={"model_kind": "EDITOR_EDIT"})
    assert (
        find_equivalent_publication(
            pricing.engine,
            build=different_kind,
        )
        is None
    )

    with pricing.engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE pricing.MODEL_RUN
                SET split_set_id = NULL
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": first_raw.model_run_id},
        )
    normalized_split_equivalent = find_equivalent_publication(
        pricing.engine,
        build=fingerprinted_raw,
    )
    assert normalized_split_equivalent is not None
    assert normalized_split_equivalent.split_set_id == first_raw.split_set_id
    with pricing.engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE pricing.MODEL_RUN
                SET split_set_id = :split_set_id
                WHERE model_run_id = :model_run_id
                """
            ),
            {
                "model_run_id": first_raw.model_run_id,
                "split_set_id": first_raw.split_set_id,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO mlops.MODEL_RUN_SPLIT_SET (
                    model_run_id,
                    manifest_id,
                    split_set_id,
                    dataset_role,
                    split_role
                )
                VALUES (
                    :model_run_id,
                    :manifest_id,
                    'alternate-validation-split',
                    'training',
                    'validation'
                )
                """
            ),
            {
                "model_run_id": first_raw.model_run_id,
                "manifest_id": first_raw.manifest_id,
            },
        )
    with pytest.raises(
        ModelEquivalenceError,
        match="multiple training/validation split links",
    ):
        find_equivalent_publication(
            pricing.engine,
            build=fingerprinted_raw,
        )
    with pricing.engine.begin() as connection:
        connection.execute(
            text(
                """
                DELETE FROM mlops.MODEL_RUN_SPLIT_SET
                WHERE model_run_id = :model_run_id
                  AND split_set_id = 'alternate-validation-split'
                  AND split_role = 'validation'
                """
            ),
            {"model_run_id": first_raw.model_run_id},
        )

    reissue_with_new_effective_date = ensure_model_equivalence(
        first_raw_candidate.completed_build.model_copy(update={"effective_from": "2026-07-01"})
    )
    assert (
        reissue_with_new_effective_date.model_equivalence_sha256
        == fingerprinted_raw.model_equivalence_sha256
    )
    with pytest.raises(
        ModelEquivalenceError,
        match="cannot preserve a new release intent",
    ):
        find_equivalent_publication(
            pricing.engine,
            build=reissue_with_new_effective_date,
        )

    duplicate_raw = publish_candidate(pricing, build("RAW"))
    routine = publish_candidate(pricing, build("ROUTINE_EDIT"))

    assert duplicate_raw.rate_package_id == first_raw.rate_package_id
    assert duplicate_raw.model_run_id == first_raw.model_run_id
    assert duplicate_raw.deduplicated is True
    assert duplicate_raw.manifest_id == first_raw.manifest_id
    assert duplicate_raw.model_equivalence_sha256 == first_raw.model_equivalence_sha256
    assert routine.rate_package_id != first_raw.rate_package_id
    assert routine.model_kind == "ROUTINE_EDIT"
    assert routine.deduplicated is False
    assert routine.manifest_id == first_raw.manifest_id
    assert routine.model_equivalence_sha256 == first_raw.model_equivalence_sha256

    with pricing.engine.connect() as connection:
        counts = {
            "manifests": connection.execute(
                text("SELECT COUNT(*) FROM pricing.DATASET_MANIFEST")
            ).scalar_one(),
            "packages": connection.execute(
                text("SELECT COUNT(*) FROM pricing.PRICING_RATE_PACKAGE")
            ).scalar_one(),
            "runs": connection.execute(text("SELECT COUNT(*) FROM pricing.MODEL_RUN")).scalar_one(),
            "staged": connection.execute(
                text("SELECT COUNT(*) FROM pricing_stg.STG_RATING_EXPORT")
            ).scalar_one(),
            "reservations": connection.execute(
                text("SELECT COUNT(*) FROM pricing.PRICING_MODEL_VERSION_RESERVATION")
            ).scalar_one(),
        }
        final_lineage = [
            dict(row)
            for row in connection.execute(
                text(
                    """
                    SELECT DISTINCT
                        model_kind,
                        model_equivalence_sha256,
                        manifest_signature_sha256,
                        dataset_name,
                        data_as_of_date,
                        data_as_of_column
                    FROM pricing.V_FINAL_MODEL_RELATIVITY
                    ORDER BY model_kind
                    """
                )
            )
            .mappings()
            .all()
        ]

    assert counts == {
        "manifests": 1,
        "packages": 2,
        "runs": 2,
        "staged": 2,
        "reservations": 2,
    }
    assert [row["model_kind"] for row in final_lineage] == [
        "RAW",
        "ROUTINE_EDIT",
    ]
    assert all(row["data_as_of_date"] == "2026-06-30" for row in final_lineage)
    assert all(row["data_as_of_column"] == "data_as_of" for row in final_lineage)
    assert all(len(row["manifest_signature_sha256"]) == 64 for row in final_lineage)
