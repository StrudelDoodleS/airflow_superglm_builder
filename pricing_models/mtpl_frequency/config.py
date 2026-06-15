from __future__ import annotations

from pathlib import Path

from pricing_pipeline.models.config import load_model_build_config


MODEL_CONFIG = load_model_build_config(Path(__file__).with_name("model.toml"))
