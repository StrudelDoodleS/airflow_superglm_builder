from __future__ import annotations

from pathlib import Path

from pricing_pipeline.data.datasets import FREMTPL_DATASET_SPEC
from pricing_pipeline.models.config import load_model_build_config
from pricing_pipeline.models.spec import ModelSpec
from pricing_models.mtpl_frequency.training import (
    FEATURE_COLUMNS,
    TRAINING_SQL,
    build_model,
    build_training_frame,
)


MODEL_CONFIG = load_model_build_config(Path(__file__).with_name("model.toml"))

MODEL_SPEC = ModelSpec(
    model_key=MODEL_CONFIG.model_key,
    model_label=MODEL_CONFIG.model_label,
    target_name=MODEL_CONFIG.target_name,
    model_type=MODEL_CONFIG.model_type,
    experiment_name="pricing-mtpl-frequency",
    deployment_slot=MODEL_CONFIG.deployment_slot,
    dataset=FREMTPL_DATASET_SPEC,
    training_sql=TRAINING_SQL,
    feature_columns=tuple(FEATURE_COLUMNS),
    build_model=build_model,
    build_training_frame=build_training_frame,
    package_status=MODEL_CONFIG.default_package_status,
)
