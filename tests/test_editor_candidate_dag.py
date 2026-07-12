from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _install_fake_airflow(monkeypatch):
    airflow_module = types.ModuleType("airflow")
    airflow_sdk_module = types.ModuleType("airflow.sdk")

    def dag(**dag_kwargs):
        def decorator(function):
            def factory(*args, **kwargs):
                function(*args, **kwargs)
                return types.SimpleNamespace(**dag_kwargs)

            return factory

        return decorator

    def task(function):
        def factory(*args, **kwargs):
            return types.SimpleNamespace(task_id=function.__name__, args=args, kwargs=kwargs)

        return factory

    airflow_sdk_module.dag = dag
    airflow_sdk_module.get_current_context = lambda: {}
    airflow_sdk_module.task = task
    airflow_module.sdk = airflow_sdk_module
    monkeypatch.setitem(sys.modules, "airflow", airflow_module)
    monkeypatch.setitem(sys.modules, "airflow.sdk", airflow_sdk_module)


def _import_dag(monkeypatch):
    _install_fake_airflow(monkeypatch)
    name = "pricing_publish_editor_candidate_test"
    sys.modules.pop(name, None)
    path = Path("dags/pricing_publish_editor_candidate.py").resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_editor_candidate_dag_is_explicit_and_manual():
    source = Path("dags/pricing_publish_editor_candidate.py").read_text(encoding="utf-8")

    assert 'dag_id="pricing_publish_editor_candidate"' in source
    assert "schedule=None" in source
    assert "publish_editor_submission" in source
    assert "get_current_context" in source
    assert "dag_factory" not in source
    assert "build_pricing_model_dag" not in source


def test_editor_candidate_dag_publishes_from_dag_run_conf(monkeypatch, tmp_path):
    module = _import_dag(monkeypatch)
    engine = object()
    settings = types.SimpleNamespace(workbench_artifact_root=tmp_path)
    runtime = types.SimpleNamespace(settings=settings, get_engine=lambda: engine)
    calls = []
    monkeypatch.setattr(module, "runtime_from_env_or_module", lambda **kwargs: runtime)
    monkeypatch.setattr(
        module,
        "publish_editor_submission",
        lambda engine_arg, **kwargs: calls.append((engine_arg, kwargs))
        or types.SimpleNamespace(
            submission_id="submission-1",
            model_name="HOME_FREQ",
            parent_rate_package_id=107,
            rate_package_id=108,
            package_version=8,
            model_run_id=908,
            package_status="PUBLISHED",
            was_existing=False,
        ),
    )
    context = {
        "dag": types.SimpleNamespace(dag_id="pricing_publish_editor_candidate"),
        "dag_run": types.SimpleNamespace(
            run_id="manual__submission-1",
            triggering_user_name="analyst@example.test",
            conf={
                "submission_path": str(tmp_path / "submission.json"),
                "submission_sha256": "a" * 64,
            },
        ),
    }

    result = module.publish_editor_candidate_from_context(context)

    assert calls == [
        (
            engine,
            {
                "settings": settings,
                "submission_path": str(tmp_path / "submission.json"),
                "submission_sha256": "a" * 64,
                "dag_id": "pricing_publish_editor_candidate",
                "airflow_run_id": "manual__submission-1",
                "created_by": "analyst@example.test",
            },
        )
    ]
    assert result == {
        "submission_id": "submission-1",
        "model_name": "HOME_FREQ",
        "parent_rate_package_id": 107,
        "rate_package_id": 108,
        "package_version": 8,
        "model_run_id": 908,
        "package_status": "PUBLISHED",
        "was_existing": False,
    }
