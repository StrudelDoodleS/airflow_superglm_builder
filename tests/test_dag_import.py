from __future__ import annotations

import importlib.util
import sys
import types
from datetime import UTC, datetime
from pathlib import Path


def test_pricing_superglm_pipeline_dag_imports_without_airflow(monkeypatch):
    airflow_module = types.ModuleType("airflow")
    airflow_sdk_module = types.ModuleType("airflow.sdk")

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

    def task(func):
        def task_factory(*args, **kwargs):
            return FakeTaskOutput(func.__name__, args, kwargs)

        return task_factory

    airflow_sdk_module.dag = dag
    airflow_sdk_module.get_current_context = lambda: {}
    airflow_sdk_module.task = task
    airflow_module.sdk = airflow_sdk_module
    monkeypatch.setitem(sys.modules, "airflow", airflow_module)
    monkeypatch.setitem(sys.modules, "airflow.sdk", airflow_sdk_module)

    dag_path = Path(__file__).resolve().parents[1] / "dags" / "pricing_superglm_pipeline.py"
    spec = importlib.util.spec_from_file_location("pricing_superglm_pipeline_test", dag_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "pricing_superglm_pipeline")
    assert module._context_date_iso(
        {"dag_run": types.SimpleNamespace(run_after=datetime(2026, 4, 27, tzinfo=UTC))}
    ) == "2026-04-27"
