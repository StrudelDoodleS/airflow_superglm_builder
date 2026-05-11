from __future__ import annotations

from pricing_pipeline.data.datasets import FREMTPL_DATASET_SPEC
from pricing_pipeline.models.spec import ModelSpec
from pricing_models.mtpl_frequency.training import (
    FEATURE_COLUMNS,
    TRAINING_SQL,
    build_model,
    build_training_frame,
)


MODEL_SPEC = ModelSpec(
    model_key="MTPL_FREQ",
    model_label="Motor frequency",
    target_name="ClaimNb",
    model_type="superglm_poisson",
    experiment_name="pricing-mtpl-frequency",
    deployment_slot="MTPL_FREQ_UAT",
    dataset=FREMTPL_DATASET_SPEC,
    training_sql=TRAINING_SQL,
    feature_columns=tuple(FEATURE_COLUMNS),
    build_model=build_model,
    build_training_frame=build_training_frame,
)
