import pytest

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.deployment import DeploymentError, deploy_rate_package
from pricing_pipeline.publishing.lifecycle import DeploymentResult


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
    def __init__(self, *, package_row=None, current_row=None, lock_result=0):
        self.package_row = package_row
        self.current_row = current_row
        self.lock_result = lock_result
        self.events = []

    def execute(self, statement, params=None):
        sql = str(statement)
        params = params or {}
        self.events.append((sql, params))
        if "sys.sp_getapplock" in sql:
            return FakeResult(scalar=self.lock_result)
        if "FROM pricing.PRICING_RATE_PACKAGE" in sql:
            return FakeResult(self.package_row)
        if "FROM pricing.PRICING_MODEL_DEPLOYMENT" in sql:
            return FakeResult(self.current_row)
        return FakeResult()


class FakeBegin:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeEngine:
    def __init__(self, *, package_row=None, current_row=None, lock_result=0):
        self.connection = FakeConnection(
            package_row=package_row,
            current_row=current_row,
            lock_result=lock_result,
        )

    def begin(self):
        return FakeBegin(self.connection)


def config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_name="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot="MTPL_FREQ_UAT",
        default_package_status="PUBLISHED",
    )


def config_with_slot(deployment_slot: str) -> ModelBuildConfig:
    return ModelBuildConfig(
        model_name="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot=deployment_slot,
        default_package_status="PUBLISHED",
    )


def published_package(**overrides):
    row = {
        "rate_package_id": 101,
        "model_id": 17,
        "package_version": 3,
        "package_status": "PUBLISHED",
    }
    row.update(overrides)
    return row


def executed_sql(engine):
    return [sql for sql, _params in engine.connection.events]


def test_deploy_rate_package_by_id_closes_current_row_inserts_deployment_and_updates_pointer():
    engine = FakeEngine(
        package_row=published_package(),
        current_row={"rate_package_id": 99},
    )

    result = deploy_rate_package(
        engine,
        config(),
        rate_package_id=101,
        deployment_slot="MTPL_FREQ_PROD",
        deployment_reason=" approved for launch ",
        deployed_by=" airflow ",
        model_id=17,
    )

    assert result == DeploymentResult(
        model_id=17,
        deployment_slot="MTPL_FREQ_PROD",
        previous_rate_package_id=99,
        rate_package_id=101,
        package_version=3,
        deployed_by="airflow",
        deployment_reason="approved for launch",
    )

    sql = executed_sql(engine)
    lock_index = next(i for i, statement in enumerate(sql) if "sys.sp_getapplock" in statement)
    package_select_index = next(
        i for i, statement in enumerate(sql) if "FROM pricing.PRICING_RATE_PACKAGE" in statement
    )
    current_select_index = next(
        i for i, statement in enumerate(sql) if "FROM pricing.PRICING_MODEL_DEPLOYMENT" in statement
    )
    update_index = next(
        i
        for i, statement in enumerate(sql)
        if "UPDATE pricing.PRICING_MODEL_DEPLOYMENT" in statement
    )
    insert_index = next(
        i
        for i, statement in enumerate(sql)
        if "INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT" in statement
    )
    merge_index = next(
        i for i, statement in enumerate(sql) if "MERGE pricing.PRICING_PACKAGE_POINTER" in statement
    )

    assert lock_index < package_select_index < current_select_index < update_index
    assert update_index < insert_index < merge_index
    assert "deployment_note" in sql[insert_index]
    assert "MERGE pricing.PRICING_PACKAGE_POINTER WITH (HOLDLOCK) AS tgt" in sql[merge_index]

    lock_params = engine.connection.events[lock_index][1]
    assert lock_params == {
        "lock_resource": "pricing_model_deployment:17:MTPL_FREQ_PROD",
        "lock_timeout_ms": 10000,
    }

    insert_params = engine.connection.events[insert_index][1]
    assert insert_params["deployment_note"] == "approved for launch"
    assert insert_params["deployed_by"] == "airflow"
    assert insert_params["deployment_slot"] == "MTPL_FREQ_PROD"

    merge_params = engine.connection.events[merge_index][1]
    assert merge_params["pointer_name"] == "MTPL_FREQ_PROD"
    assert merge_params["updated_by"] == "airflow"
    assert merge_params["rate_package_id"] == 101


def test_deploy_rate_package_resolves_by_model_and_package_version_using_default_slot():
    engine = FakeEngine(package_row=published_package(rate_package_id=202, package_version=4))

    result = deploy_rate_package(
        engine,
        config(),
        package_version=4,
        deployment_reason="UAT signoff",
        deployed_by="airflow",
        model_id=17,
    )

    package_select, package_params = next(
        (statement, params)
        for statement, params in engine.connection.events
        if "FROM pricing.PRICING_RATE_PACKAGE" in statement
    )
    assert "package_version = :package_version" in package_select
    assert package_params == {"model_id": 17, "package_version": 4}
    assert result.previous_rate_package_id is None
    assert result.rate_package_id == 202
    assert result.package_version == 4
    assert result.deployment_slot == "MTPL_FREQ_UAT"


def test_deploy_rate_package_canonicalizes_deployment_slot_before_lock_and_writes():
    engine = FakeEngine(package_row=published_package())

    result = deploy_rate_package(
        engine,
        config(),
        rate_package_id=101,
        deployment_slot="  mtpl_FREQ_prod  ",
        deployment_reason="approved",
        deployed_by="airflow",
        model_id=17,
    )

    sql = executed_sql(engine)
    lock_index = next(i for i, statement in enumerate(sql) if "sys.sp_getapplock" in statement)
    insert_index = next(
        i
        for i, statement in enumerate(sql)
        if "INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT" in statement
    )
    merge_index = next(
        i for i, statement in enumerate(sql) if "MERGE pricing.PRICING_PACKAGE_POINTER" in statement
    )

    assert result.deployment_slot == "MTPL_FREQ_PROD"
    assert engine.connection.events[lock_index][1]["lock_resource"] == (
        "pricing_model_deployment:17:MTPL_FREQ_PROD"
    )
    assert engine.connection.events[insert_index][1]["deployment_slot"] == "MTPL_FREQ_PROD"
    assert engine.connection.events[merge_index][1]["pointer_name"] == "MTPL_FREQ_PROD"


def test_deploy_rate_package_rejects_blank_default_deployment_slot():
    engine = FakeEngine(package_row=published_package())

    with pytest.raises(DeploymentError, match="deployment_slot"):
        deploy_rate_package(
            engine,
            config_with_slot("   "),
            rate_package_id=101,
            deployment_reason="approved",
            deployed_by="airflow",
            model_id=17,
        )

    assert engine.connection.events == []


@pytest.mark.parametrize("deployment_slot", ["", "   "])
def test_deploy_rate_package_rejects_blank_deployment_slot_override(deployment_slot):
    engine = FakeEngine(package_row=published_package())

    with pytest.raises(DeploymentError, match="deployment_slot"):
        deploy_rate_package(
            engine,
            config(),
            rate_package_id=101,
            deployment_slot=deployment_slot,
            deployment_reason="approved",
            deployed_by="airflow",
            model_id=17,
        )

    assert engine.connection.events == []


def test_deploy_rate_package_rejects_negative_app_lock_result_before_writes():
    engine = FakeEngine(package_row=published_package(), lock_result=-1)

    with pytest.raises(DeploymentError, match="deployment lock"):
        deploy_rate_package(
            engine,
            config(),
            rate_package_id=101,
            deployment_reason="approved",
            deployed_by="airflow",
            model_id=17,
        )

    sql = executed_sql(engine)
    assert any("sys.sp_getapplock" in statement for statement in sql)
    write_sql = [
        statement
        for statement in sql
        if statement.lstrip().startswith(("UPDATE", "INSERT", "MERGE"))
    ]
    assert write_sql == []


@pytest.mark.parametrize(
    ("rate_package_id", "package_version"),
    [
        (None, None),
        (101, 3),
    ],
)
def test_deploy_rate_package_requires_exactly_one_package_selector(
    rate_package_id,
    package_version,
):
    engine = FakeEngine(package_row=published_package())

    with pytest.raises(DeploymentError, match="exactly one"):
        deploy_rate_package(
            engine,
            config(),
            rate_package_id=rate_package_id,
            package_version=package_version,
            deployment_reason="approved",
            deployed_by="airflow",
            model_id=17,
        )


@pytest.mark.parametrize(
    ("deployment_reason", "deployed_by", "message"),
    [
        (None, "airflow", "deployment_reason"),
        ("   ", "airflow", "deployment_reason"),
        ("approved", None, "deployed_by"),
        ("approved", "   ", "deployed_by"),
    ],
)
def test_deploy_rate_package_requires_reason_and_deployer(
    deployment_reason,
    deployed_by,
    message,
):
    engine = FakeEngine(package_row=published_package())

    with pytest.raises(DeploymentError, match=message):
        deploy_rate_package(
            engine,
            config(),
            rate_package_id=101,
            deployment_reason=deployment_reason,
            deployed_by=deployed_by,
            model_id=17,
        )


def test_deploy_rate_package_rejects_non_published_package():
    engine = FakeEngine(package_row=published_package(package_status="DRAFT"))

    with pytest.raises(DeploymentError, match="PUBLISHED"):
        deploy_rate_package(
            engine,
            config(),
            rate_package_id=101,
            deployment_reason="approved",
            deployed_by="airflow",
            model_id=17,
        )


def test_deploy_rate_package_rejects_package_model_mismatch():
    engine = FakeEngine(package_row=published_package(model_id=18))

    with pytest.raises(DeploymentError, match="model_id"):
        deploy_rate_package(
            engine,
            config(),
            rate_package_id=101,
            deployment_reason="approved",
            deployed_by="airflow",
            model_id=17,
        )


def test_deploy_rate_package_rejects_already_current_package_without_writes():
    engine = FakeEngine(
        package_row=published_package(),
        current_row={"rate_package_id": 101},
    )

    with pytest.raises(DeploymentError, match="already current"):
        deploy_rate_package(
            engine,
            config(),
            rate_package_id=101,
            deployment_reason="approved",
            deployed_by="airflow",
            model_id=17,
        )

    write_sql = [
        statement
        for statement in executed_sql(engine)
        if statement.lstrip().startswith(("UPDATE", "INSERT", "MERGE"))
    ]
    assert write_sql == []
