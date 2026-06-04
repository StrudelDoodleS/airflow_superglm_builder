from __future__ import annotations

import importlib.util
import sys
import types
from datetime import UTC, datetime
from pathlib import Path


def _install_fake_airflow(monkeypatch):
    sys.modules.pop("pricing_pipeline.orchestration.dag_factory", None)
    publish_completed_build = sys.modules.get(
        "pricing_pipeline.orchestration.publish_completed_build"
    )
    airflow_module = types.ModuleType("airflow")
    airflow_sdk_module = types.ModuleType("airflow.sdk")
    task_outputs = []

    class FakeTaskOutput:
        def __init__(self, task_id, args=(), kwargs=None):
            self.task_id = task_id
            self.args = args
            self.kwargs = kwargs or {}
            self.downstream = []

        def __rshift__(self, other):
            self.downstream.append(other)
            return other

    def dag(**dag_kwargs):
        def decorator(func):
            def factory(*args, **kwargs):
                func(*args, **kwargs)
                return types.SimpleNamespace(dag_id=dag_kwargs["dag_id"])

            return factory

        return decorator

    def task(func=None, **task_kwargs):
        def decorator(inner):
            task_id = task_kwargs.get("task_id") or inner.__name__

            def task_factory(*args, **kwargs):
                output = FakeTaskOutput(task_id, args, kwargs)
                task_outputs.append(output)
                return output

            return task_factory

        if func is None:
            return decorator
        return decorator(func)

    airflow_sdk_module.dag = dag
    airflow_sdk_module.get_current_context = lambda: {}
    airflow_sdk_module.task = task
    airflow_module.sdk = airflow_sdk_module
    monkeypatch.setitem(sys.modules, "airflow", airflow_module)
    monkeypatch.setitem(sys.modules, "airflow.sdk", airflow_sdk_module)
    if publish_completed_build is not None:
        monkeypatch.setattr(publish_completed_build, "task", task, raising=False)
        monkeypatch.setattr(
            publish_completed_build,
            "get_current_context",
            lambda: {},
            raising=False,
        )
    return task_outputs


def _import_dag_module(module_name: str, filename: str):
    dag_path = Path(__file__).resolve().parents[1] / "dags" / filename
    spec = importlib.util.spec_from_file_location(module_name, dag_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pricing_superglm_pipeline_dag_imports_without_airflow(monkeypatch):
    _install_fake_airflow(monkeypatch)

    module = _import_dag_module(
        "pricing_superglm_pipeline_test",
        "pricing_superglm_pipeline.py",
    )

    assert hasattr(module, "pricing_superglm_pipeline")
    assert (
        module._context_date_iso(
            {"dag_run": types.SimpleNamespace(run_after=datetime(2026, 4, 27, tzinfo=UTC))}
        )
        == "2026-04-27"
    )


def test_pricing_mtpl_frequency_dag_keeps_model_publish_tasks_separate(monkeypatch):
    task_outputs = _install_fake_airflow(monkeypatch)

    module = _import_dag_module(
        "pricing_mtpl_frequency_test",
        "pricing_mtpl_frequency.py",
    )

    assert hasattr(module, "pricing_mtpl_frequency")
    task_ids = [output.task_id for output in task_outputs]
    assert task_ids == [
        "apply_pricing_schema",
        "prepare_dataset",
        "train_and_export",
        "publish_export",
    ]


def test_dag_factory_accepts_explicit_runtime_module(monkeypatch):
    task_outputs = _install_fake_airflow(monkeypatch)

    from pricing_models.mtpl_frequency.spec import MODEL_CONFIG, MODEL_SPEC
    from pricing_pipeline.orchestration.dag_factory import build_pricing_model_dag

    dag = build_pricing_model_dag(
        dag_id="pricing.motor_frequency.build",
        spec=MODEL_SPEC,
        model_config=MODEL_CONFIG,
        runtime_module="work_runtime.database",
    )

    assert dag.dag_id == "pricing.motor_frequency.build"
    assert [output.task_id for output in task_outputs] == [
        "apply_pricing_schema",
        "prepare_dataset",
        "train_and_export",
        "publish_export",
    ]


def test_pricing_deploy_rate_package_dag_imports_without_airflow(monkeypatch):
    _install_fake_airflow(monkeypatch)

    module = _import_dag_module(
        "pricing_deploy_rate_package_test",
        "pricing_deploy_rate_package.py",
    )

    assert hasattr(module, "pricing_deploy_rate_package")


def test_demo_custom_publish_dag_uses_custom_tasks_and_publish_task(monkeypatch):
    task_outputs = _install_fake_airflow(monkeypatch)

    module = _import_dag_module(
        "demo_custom_publish_test",
        "demo_custom_publish.py",
    )

    assert hasattr(module, "demo_custom_publish")
    assert [output.task_id for output in task_outputs] == [
        "register_demo_model",
        "prepare_training_data",
        "create_training_manifest",
        "train_validate_export",
        "publish_completed_model_build",
    ]
