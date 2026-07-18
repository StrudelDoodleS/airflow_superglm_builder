from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from superglm import Numeric, SuperGLM

from pricing_pipeline.models.spec import ApprovedModelBuildError


def test_real_local_notebook_build_publishes_audit_evidence_to_all_five_views(
    tmp_path,
):
    from pricing_pipeline import notebook as api
    from pricing_pipeline.models.config import ValidationSplitConfig

    model_root = tmp_path / "pricing_models" / "claim_frequency"
    model_root.mkdir(parents=True)
    (model_root / "pricing_model.py").write_text(
        "# Stable analyst-owned model source used by the integration test.\n",
        encoding="utf-8",
    )
    context = api.connect(mode="local", local_root=model_root / ".local")
    feature_values = np.linspace(18.0, 78.0, 40)
    frame = pd.DataFrame(
        {
            "policy_id": np.arange(1, 41),
            "claim_count": np.tile([0.0, 1.0, 0.0, 2.0, 1.0], 8),
            "age": feature_values,
        }
    )
    spec = api.PricingModelSpec(
        name="CLAIM_FREQUENCY",
        label="Claim frequency",
        target="claim_count",
        model_type="superglm_poisson",
        deployment_slot="CLAIM_FREQUENCY_UAT",
        features=("age",),
        dataset_name="claim_frequency_frame",
        source_system="pytest",
        pk_columns=("policy_id",),
        validation=ValidationSplitConfig.kfold(
            n_splits=2,
            random_state=42,
            shuffle=True,
        ),
    )
    model = api.register_model(
        context,
        spec,
        source_root=model_root,
        created_by="integration@example.test",
    )

    def superglm_model():
        return SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            n_bins=8,
            features={"age": Numeric()},
        )

    candidate = api.build_candidate(
        context,
        model=model,
        frame=frame,
        superglm_model=superglm_model(),
        data_as_of="2026-06-30",
        created_by="integration@example.test",
    )

    published = api.publish_candidate(context, candidate)
    retry_candidate = api.build_candidate(
        context,
        model=model,
        frame=frame,
        superglm_model=superglm_model(),
        data_as_of="2026-06-30",
        created_by="retrying.integration@example.test",
    )
    canonical_candidate_path = Path(candidate.completed_build.candidate_artifact_path)
    canonical_candidate_bytes = canonical_candidate_path.read_bytes()
    canonical_candidate_path.unlink()
    with pytest.raises(
        ApprovedModelBuildError,
        match="canonical candidate artifact verification failed",
    ):
        api.publish_candidate(context, retry_candidate)
    assert Path(retry_candidate.completed_build.candidate_artifact_path).is_file()
    canonical_candidate_path.write_bytes(canonical_candidate_bytes)
    canonical_candidate_path.write_bytes(b"corrupt canonical candidate")
    with pytest.raises(
        ApprovedModelBuildError,
        match="canonical candidate artifact verification failed",
    ):
        api.publish_candidate(context, retry_candidate)
    canonical_candidate_path.write_bytes(canonical_candidate_bytes)
    with context.engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE pricing.MODEL_RUN
                SET manifest_id = 'corrupted-canonical-manifest'
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": published.model_run_id},
        )
    with pytest.raises(ValueError, match="run_manifest_id"):
        api.publish_candidate(context, retry_candidate)
    with context.engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE pricing.MODEL_RUN
                SET manifest_id = :manifest_id
                WHERE model_run_id = :model_run_id
                """
            ),
            {
                "manifest_id": published.manifest_id,
                "model_run_id": published.model_run_id,
            },
        )
    retried = api.publish_candidate(context, retry_candidate)

    assert published.model_version == "v1"
    assert published.was_existing is False
    assert retry_candidate.completed_build.manifest_id != candidate.completed_build.manifest_id
    assert (
        retry_candidate.completed_build.build_fingerprint_sha256
        == candidate.completed_build.build_fingerprint_sha256
    )
    assert retried.was_existing is True
    assert retried.rate_package_id == published.rate_package_id
    assert retried.model_run_id == published.model_run_id
    assert retried.manifest_id == published.manifest_id
    assert retried.split_set_id == published.split_set_id
    assert canonical_candidate_path.is_file()
    assert not Path(retry_candidate.completed_build.candidate_artifact_path).parent.exists()
    with context.engine.connect() as connection:
        results = {
            "final": connection.execute(
                text(
                    """
                    SELECT *
                    FROM pricing.V_FINAL_MODEL_RELATIVITY
                    WHERE rate_package_id = :rate_package_id
                    """
                ),
                {"rate_package_id": published.rate_package_id},
            ).all(),
            "split": connection.execute(
                text(
                    """
                    SELECT *
                    FROM pricing.V_MODEL_VALIDATION_SPLIT
                    WHERE model_run_id = :model_run_id
                    """
                ),
                {"model_run_id": published.model_run_id},
            ).all(),
            "summary": connection.execute(
                text(
                    """
                    SELECT *
                    FROM pricing.V_MODEL_VALIDATION_SUMMARY
                    WHERE model_run_id = :model_run_id
                    """
                ),
                {"model_run_id": published.model_run_id},
            ).all(),
            "curve": connection.execute(
                text(
                    """
                    SELECT *
                    FROM pricing.V_MODEL_VALIDATION_SPLIT_RELATIVITY
                    WHERE model_run_id = :model_run_id
                    """
                ),
                {"model_run_id": published.model_run_id},
            ).all(),
            "current_split": connection.execute(
                text(
                    """
                        SELECT *
                        FROM pricing.V_CURRENT_DATASET_VALIDATION_SPLIT
                        WHERE dataset_name = :dataset_name
                        """
                ),
                {"dataset_name": spec.dataset_name},
            ).all(),
        }
        point_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM pricing.CV_SPLIT_CURVE_POINT
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": published.model_run_id},
        ).scalar_one()

    assert len(results["split"]) == 2
    assert len(results["summary"]) == 1
    assert len(results["current_split"]) == 2
    if candidate.completed_build.validation_curve_status == "COMPLETE":
        assert results["curve"]
        assert point_count == len(candidate.completed_build.validation_curve_points)
    else:
        assert candidate.completed_build.validation_curve_status == "UNAVAILABLE"
        assert results["curve"] == []
        assert point_count == 0
    # Local publication currently records audit evidence without compiling editable rate tables.
    assert results["final"] == []
