from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _install_fake_airflow(monkeypatch):
    airflow_module = types.ModuleType("airflow")
    airflow_sdk_module = types.ModuleType("airflow.sdk")
    task_outputs = []

    class FakeTaskOutput:
        def __init__(self, task_id, args=(), kwargs=None):
            self.task_id = task_id
            self.args = args
            self.kwargs = kwargs or {}

    def dag(**dag_kwargs):
        def decorator(func):
            def factory(*args, **kwargs):
                func(*args, **kwargs)
                return types.SimpleNamespace(**dag_kwargs)

            return factory

        return decorator

    def task(func):
        def task_factory(*args, **kwargs):
            output = FakeTaskOutput(func.__name__, args, kwargs)
            task_outputs.append(output)
            return output

        return task_factory

    airflow_sdk_module.dag = dag
    airflow_sdk_module.get_current_context = lambda: {"params": {}}
    airflow_sdk_module.task = task
    airflow_module.sdk = airflow_sdk_module
    monkeypatch.setitem(sys.modules, "airflow", airflow_module)
    monkeypatch.setitem(sys.modules, "airflow.sdk", airflow_sdk_module)
    return task_outputs


def _import_deploy_dag_module(monkeypatch):
    _install_fake_airflow(monkeypatch)
    module_name = "pricing_deploy_rate_package_test"
    sys.modules.pop(module_name, None)
    dag_path = Path(__file__).resolve().parents[1] / "dags" / "pricing_deploy_rate_package.py"
    spec = importlib.util.spec_from_file_location(module_name, dag_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_pricing_deploy_rate_package_dag_exposes_manual_params(monkeypatch):
    module = _import_deploy_dag_module(monkeypatch)

    dag_obj = module.pricing_deploy_rate_package

    assert dag_obj.dag_id == "pricing_deploy_rate_package"
    assert dag_obj.schedule is None
    assert dag_obj.catchup is False
    assert set(dag_obj.tags) >= {"pricing", "deploy"}
    assert set(dag_obj.params) == {
        "model_name",
        "rate_package_id",
        "package_version",
        "deployment_slot",
        "deployment_reason",
        "deployed_by",
    }


def test_deploy_rate_package_from_params_converts_selector_and_returns_xcom_strings(
    monkeypatch,
):
    module = _import_deploy_dag_module(monkeypatch)
    calls = []
    engine = object()
    config = types.SimpleNamespace(model_name="MTPL_FREQ")

    class FakePublisher:
        def __init__(self, engine_arg, config_arg):
            calls.append(("init", engine_arg, config_arg))

        def deploy(self, **kwargs):
            calls.append(("deploy", kwargs))
            return types.SimpleNamespace(
                deployment_slot="MTPL_FREQ_UAT",
                previous_rate_package_id=None,
                rate_package_id=202,
                package_version=4,
                deployed_by="airflow",
                deployment_reason="approved",
            )

    monkeypatch.setattr(
        module,
        "get_model_config",
        lambda model_name: calls.append(("config", model_name)) or config,
    )
    monkeypatch.setattr(module, "get_engine", lambda: calls.append(("engine",)) or engine)
    monkeypatch.setattr(module, "ModelPublisher", FakePublisher)

    result = module.deploy_rate_package_from_params(
        {
            "model_name": " MTPL_FREQ ",
            "rate_package_id": "",
            "package_version": "4",
            "deployment_slot": "MTPL_FREQ_UAT",
            "deployment_reason": " approved ",
            "deployed_by": " airflow ",
        }
    )

    assert calls == [
        ("config", "MTPL_FREQ"),
        ("engine",),
        ("init", engine, config),
        (
            "deploy",
            {
                "rate_package_id": None,
                "package_version": 4,
                "deployment_slot": "MTPL_FREQ_UAT",
                "deployment_reason": " approved ",
                "deployed_by": " airflow ",
            },
        ),
    ]
    assert result == {
        "model_name": "MTPL_FREQ",
        "deployment_slot": "MTPL_FREQ_UAT",
        "previous_rate_package_id": "",
        "rate_package_id": "202",
        "package_version": "4",
        "deployed_by": "airflow",
        "deployment_reason": "approved",
    }


def test_deploy_rate_package_from_params_accepts_whitespace_integer_string(
    monkeypatch,
):
    module = _import_deploy_dag_module(monkeypatch)
    calls = []
    engine = object()
    config = types.SimpleNamespace(model_name="MTPL_FREQ")

    class FakePublisher:
        def __init__(self, engine_arg, config_arg):
            calls.append(("init", engine_arg, config_arg))

        def deploy(self, **kwargs):
            calls.append(("deploy", kwargs))
            return types.SimpleNamespace(
                deployment_slot="MTPL_FREQ_UAT",
                previous_rate_package_id=101,
                rate_package_id=202,
                package_version=4,
                deployed_by="airflow",
                deployment_reason="approved",
            )

    monkeypatch.setattr(module, "get_model_config", lambda _model_name: config)
    monkeypatch.setattr(module, "get_engine", lambda: engine)
    monkeypatch.setattr(module, "ModelPublisher", FakePublisher)

    result = module.deploy_rate_package_from_params(
        {
            "model_name": "MTPL_FREQ",
            "rate_package_id": " 202 ",
            "package_version": None,
            "deployment_reason": "approved",
            "deployed_by": "airflow",
        }
    )

    assert calls == [
        ("init", engine, config),
        (
            "deploy",
            {
                "rate_package_id": 202,
                "package_version": None,
                "deployment_slot": None,
                "deployment_reason": "approved",
                "deployed_by": "airflow",
            },
        ),
    ]
    assert result["rate_package_id"] == "202"


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("rate_package_id", True),
        ("package_version", False),
        ("package_version", 4.9),
        ("package_version", "4.9"),
    ],
)
def test_deploy_rate_package_from_params_rejects_non_integer_selector_values(
    monkeypatch,
    field_name,
    field_value,
):
    module = _import_deploy_dag_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "get_model_config",
        lambda _model_name: pytest.fail("invalid selector should fail before config lookup"),
    )
    params = {
        "model_name": "MTPL_FREQ",
        "rate_package_id": None,
        "package_version": None,
        field_name: field_value,
    }

    with pytest.raises(ValueError, match=f"{field_name} must be an integer"):
        module.deploy_rate_package_from_params(params)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"model_name": "   ", "rate_package_id": 1}, "model_name"),
        ({"model_name": "MTPL_FREQ"}, "exactly one"),
        (
            {"model_name": "MTPL_FREQ", "rate_package_id": 1, "package_version": 4},
            "exactly one",
        ),
    ],
)
def test_deploy_rate_package_from_params_validates_model_name_and_selector(
    monkeypatch,
    params,
    message,
):
    module = _import_deploy_dag_module(monkeypatch)

    with pytest.raises(ValueError, match=message):
        module.deploy_rate_package_from_params(params)
