import pytest

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.model_registry import (
    ModelRegistryError,
    PricingModelRecord,
    get_pricing_model,
    register_pricing_model,
    validate_registered_model,
)


class FakeScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one(self):
        return self.value

    def scalar_one_or_none(self):
        return self.value


class FakeMappingsResult:
    def __init__(self, row):
        self.row = row

    def one_or_none(self):
        return self.row


class FakeResult:
    def __init__(self, row=None, scalar=None):
        self.row = row
        self.scalar = scalar

    def mappings(self):
        return FakeMappingsResult(self.row)

    def scalar_one(self):
        return self.scalar


class FakeConnection:
    def __init__(self, row=None, scalar=17):
        self.row = row
        self.scalar = scalar
        self.events = []

    def execute(self, statement, params=None):
        sql = str(statement)
        self.events.append((sql, params))
        if "SELECT model_id" in sql and "FROM pricing.PRICING_MODEL" in sql:
            return FakeResult(row=self.row, scalar=self.scalar)
        return FakeResult(scalar=self.scalar)


def config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_name="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot="MTPL_FREQ_UAT",
    )


def test_get_pricing_model_returns_record_for_existing_model():
    con = FakeConnection(
        {
            "model_id": 17,
            "model_name": "MTPL_FREQ",
            "model_label": "Motor frequency",
            "target_name": "ClaimNb",
            "model_type": "superglm_poisson",
            "model_status": "ACTIVE",
        }
    )

    record = get_pricing_model(con, "MTPL_FREQ")

    assert record == PricingModelRecord(
        model_id=17,
        model_name="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        model_status="ACTIVE",
    )


def test_validate_registered_model_fails_when_model_missing():
    con = FakeConnection(None)

    with pytest.raises(ModelRegistryError, match="not registered"):
        validate_registered_model(con, config())


def test_validate_registered_model_fails_on_metadata_mismatch():
    con = FakeConnection(
        {
            "model_id": 17,
            "model_name": "MTPL_FREQ",
            "model_label": "Motor frequency",
            "target_name": "LossCost",
            "model_type": "superglm_poisson",
            "model_status": "ACTIVE",
        }
    )

    with pytest.raises(ModelRegistryError, match="target_name"):
        validate_registered_model(con, config())


def test_register_pricing_model_inserts_without_updating_existing_rows():
    row = {
        "model_id": 17,
        "model_name": "MTPL_FREQ",
        "model_label": "Motor frequency",
        "target_name": "ClaimNb",
        "model_type": "superglm_poisson",
        "model_status": "ACTIVE",
    }
    con = FakeConnection(row, scalar=17)

    record = register_pricing_model(con, config(), created_by="airflow")

    assert record == PricingModelRecord(**row)
    sql = con.events[0][0]
    assert "INSERT INTO pricing.PRICING_MODEL" in sql
    assert "MERGE" not in sql
    assert "UPDATE SET" not in sql
