from __future__ import annotations

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.lifecycle import RatePackageSelector
from pricing_pipeline.publishing.model_registry import validate_registered_model


def validate_model_on_engine(engine, config: ModelBuildConfig) -> int:
    with engine.begin() as con:
        return validate_registered_model(con, config).model_id


class ModelPublisher:
    def __init__(self, engine, config: ModelBuildConfig):
        self.engine = engine
        self.config = config

    def validate_registered_model(self) -> int:
        return validate_model_on_engine(self.engine, self.config)

    def load_rate_package(self, selector: RatePackageSelector):
        from pricing_pipeline.publishing.manual_revision import load_rate_package_snapshot

        return load_rate_package_snapshot(self.engine, self.config, selector)
