from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.lifecycle import RatePackageSelector
from pricing_pipeline.publishing.publisher import ModelPublisher


def config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_key="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot="MTPL_FREQ_UAT",
        default_package_status="PUBLISHED",
    )


def test_model_publisher_stores_engine_and_config():
    engine = object()
    publisher = ModelPublisher(engine, config())

    assert publisher.engine is engine
    assert publisher.config.model_key == "MTPL_FREQ"


def test_rate_package_selector_requires_one_selector():
    assert RatePackageSelector(rate_package_id=123).rate_package_id == 123
    assert RatePackageSelector(package_version=7).package_version == 7


def test_model_publisher_validate_registered_model_delegates(monkeypatch):
    calls = []
    engine = object()
    publisher = ModelPublisher(engine, config())

    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.validate_model_on_engine",
        lambda engine_arg, config_arg: calls.append((engine_arg, config_arg)) or 17,
    )

    assert publisher.validate_registered_model() == 17
    assert calls == [(engine, config())]
