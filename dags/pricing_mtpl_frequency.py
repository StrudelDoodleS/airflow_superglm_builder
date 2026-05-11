from __future__ import annotations

from pricing_models.mtpl_frequency.spec import MODEL_SPEC
from pricing_pipeline.orchestration.dag_factory import (
    build_pricing_model_dag,
    context_date_iso as _context_date_iso,  # noqa: F401
    schema_dir_from_env,
)

SCHEMA_DIR = schema_dir_from_env()

pricing_mtpl_frequency = build_pricing_model_dag(
    dag_id="pricing_mtpl_frequency",
    spec=MODEL_SPEC,
    tags=["pricing", "mtpl", "frequency", "mlflow"],
)
