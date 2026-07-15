from inspect import signature

import pytest

from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.publishing.deployment import (
    DeploymentError,
    StaleChampionError,
    deploy_rate_package,
)
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


class StatefulConnection:
    def __init__(self, *, packages, current_rate_package_id):
        self.packages = {int(row["rate_package_id"]): row for row in packages}
        self.current_rate_package_id = current_rate_package_id
        self.current_deployed_by = "previous deployer"
        self.current_deployment_note = "previous deployment"
        self.events = []

    def execute(self, statement, params=None):
        sql = str(statement)
        params = params or {}
        self.events.append((sql, params))
        if "sys.sp_getapplock" in sql:
            return FakeResult(scalar=0)
        if "FROM pricing.PRICING_RATE_PACKAGE" in sql:
            if params.get("rate_package_id") is not None:
                package = self.packages.get(int(params["rate_package_id"]))
            else:
                package = next(
                    (
                        row
                        for row in self.packages.values()
                        if int(row["model_id"]) == int(params["model_id"])
                        and int(row["package_version"]) == int(params["package_version"])
                    ),
                    None,
                )
            return FakeResult(package)
        if "FROM pricing.PRICING_MODEL_DEPLOYMENT" in sql:
            current = (
                None
                if self.current_rate_package_id is None
                else {
                    "rate_package_id": self.current_rate_package_id,
                    "deployed_by": self.current_deployed_by,
                    "deployment_note": self.current_deployment_note,
                }
            )
            return FakeResult(current)
        if "INSERT INTO pricing.PRICING_MODEL_DEPLOYMENT" in sql:
            self.current_rate_package_id = int(params["rate_package_id"])
            self.current_deployed_by = params["deployed_by"]
            self.current_deployment_note = params["deployment_note"]
        return FakeResult()


class StatefulEngine:
    def __init__(self, *, packages, current_rate_package_id):
        self.connection = StatefulConnection(
            packages=packages,
            current_rate_package_id=current_rate_package_id,
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
    )


def config_with_slot(deployment_slot: str) -> ModelBuildConfig:
    return ModelBuildConfig(
        model_name="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot=deployment_slot,
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
        config_with_slot("MTPL_FREQ_PROD"),
        rate_package_id=101,
        expected_current_rate_package_id=99,
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

    package_params = engine.connection.events[package_select_index][1]
    assert package_params == {"rate_package_id": 101}

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


def test_deploy_rate_package_canonicalizes_configured_slot_before_lock_and_writes():
    engine = FakeEngine(package_row=published_package())

    result = deploy_rate_package(
        engine,
        config_with_slot("  mtpl_FREQ_prod  "),
        rate_package_id=101,
        expected_current_rate_package_id=None,
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
            expected_current_rate_package_id=None,
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
            expected_current_rate_package_id=None,
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


def test_deploy_rate_package_exposes_only_the_exact_package_id_selector():
    parameters = signature(deploy_rate_package).parameters
    assert "package_version" not in parameters
    assert "deployment_slot" not in parameters
    assert parameters["rate_package_id"].default is parameters["rate_package_id"].empty

    engine = FakeEngine(package_row=published_package())
    with pytest.raises(TypeError, match="rate_package_id"):
        deploy_rate_package(
            engine,
            config(),
            expected_current_rate_package_id=None,
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
            expected_current_rate_package_id=None,
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
            expected_current_rate_package_id=None,
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
            expected_current_rate_package_id=None,
            deployment_reason="approved",
            deployed_by="airflow",
            model_id=17,
        )


def test_deploy_rate_package_rejects_already_current_package_when_snapshot_is_stale():
    engine = FakeEngine(
        package_row=published_package(),
        current_row={"rate_package_id": 101},
    )

    with pytest.raises(StaleChampionError) as exc_info:
        deploy_rate_package(
            engine,
            config(),
            rate_package_id=101,
            expected_current_rate_package_id=99,
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
    assert "expected current rate_package_id=99" in str(exc_info.value)
    assert "found 101" in str(exc_info.value)


def test_deploy_rate_package_noop_returns_existing_deployment_evidence():
    engine = FakeEngine(
        package_row=published_package(),
        current_row={
            "rate_package_id": 101,
            "deployed_by": "original analyst",
            "deployment_note": "original approval",
        },
    )

    result = deploy_rate_package(
        engine,
        config(),
        rate_package_id=101,
        expected_current_rate_package_id=101,
        deployment_reason="retry must not replace the audit reason",
        deployed_by="retrying caller",
        model_id=17,
    )

    assert result == DeploymentResult(
        model_id=17,
        deployment_slot="MTPL_FREQ_UAT",
        previous_rate_package_id=101,
        rate_package_id=101,
        package_version=3,
        deployed_by="original analyst",
        deployment_reason="original approval",
    )
    assert not any(
        statement.lstrip().startswith(("UPDATE", "INSERT", "MERGE"))
        for statement in executed_sql(engine)
    )


def test_stale_review_cannot_replace_a_newer_champion():
    packages = [
        published_package(rate_package_id=101, package_version=1),
        published_package(rate_package_id=202, package_version=2),
        published_package(rate_package_id=303, package_version=3),
    ]
    engine = StatefulEngine(packages=packages, current_rate_package_id=101)

    first = deploy_rate_package(
        engine,
        config(),
        rate_package_id=202,
        expected_current_rate_package_id=101,
        deployment_reason="B approved",
        deployed_by="airflow",
        model_id=17,
    )
    assert first.previous_rate_package_id == 101
    assert engine.connection.current_rate_package_id == 202
    events_before_stale_attempt = len(engine.connection.events)

    with pytest.raises(StaleChampionError) as exc_info:
        deploy_rate_package(
            engine,
            config(),
            rate_package_id=303,
            expected_current_rate_package_id=101,
            deployment_reason="stale C approval",
            deployed_by="airflow",
            model_id=17,
        )

    assert "expected current rate_package_id=101" in str(exc_info.value)
    assert "found 202" in str(exc_info.value)
    assert engine.connection.current_rate_package_id == 202
    stale_sql = executed_sql(engine)[events_before_stale_attempt:]
    assert not any(
        statement.lstrip().startswith(("UPDATE", "INSERT", "MERGE")) for statement in stale_sql
    )


def test_deploy_rate_package_handles_no_current_champion():
    engine = StatefulEngine(
        packages=[published_package(rate_package_id=202, package_version=2)],
        current_rate_package_id=None,
    )

    result = deploy_rate_package(
        engine,
        config(),
        rate_package_id=202,
        expected_current_rate_package_id=None,
        deployment_reason="initial champion",
        deployed_by="airflow",
        model_id=17,
    )

    assert result.previous_rate_package_id is None
    assert engine.connection.current_rate_package_id == 202


def test_deploy_rate_package_retry_requires_refreshed_champion_snapshot():
    engine = StatefulEngine(
        packages=[published_package(rate_package_id=202, package_version=2)],
        current_rate_package_id=101,
    )

    first = deploy_rate_package(
        engine,
        config(),
        rate_package_id=202,
        expected_current_rate_package_id=101,
        deployment_reason="approved",
        deployed_by="airflow",
        model_id=17,
    )
    write_count = sum(
        statement.lstrip().startswith(("UPDATE", "INSERT", "MERGE"))
        for statement in executed_sql(engine)
    )

    with pytest.raises(StaleChampionError, match="found 202"):
        deploy_rate_package(
            engine,
            config(),
            rate_package_id=202,
            expected_current_rate_package_id=101,
            deployment_reason="approved",
            deployed_by="airflow",
            model_id=17,
        )

    refreshed_retry = deploy_rate_package(
        engine,
        config(),
        rate_package_id=202,
        expected_current_rate_package_id=202,
        deployment_reason="retry must not replace the audit reason",
        deployed_by="retrying caller",
        model_id=17,
    )

    assert refreshed_retry == DeploymentResult(
        model_id=17,
        deployment_slot="MTPL_FREQ_UAT",
        previous_rate_package_id=202,
        rate_package_id=202,
        package_version=2,
        deployed_by="airflow",
        deployment_reason="approved",
    )
    assert first.previous_rate_package_id == 101
    assert engine.connection.current_rate_package_id == 202
    assert (
        sum(
            statement.lstrip().startswith(("UPDATE", "INSERT", "MERGE"))
            for statement in executed_sql(engine)
        )
        == write_count
    )
