from __future__ import annotations

from pricing_models.mtpl_frequency.spec import MODEL_CONFIG, MODEL_SPEC
from pricing_pipeline.orchestration.dag_factory import (
    build_pricing_model_dag,
    context_date_iso as _context_date_iso,  # noqa: F401
    schema_dir_from_env,
)

SCHEMA_DIR = schema_dir_from_env()

pricing_superglm_pipeline = build_pricing_model_dag(
    dag_id="pricing_superglm_pipeline",
    spec=MODEL_SPEC,
    model_config=MODEL_CONFIG,
    tags=["pricing", "superglm", "mlflow"],
)
