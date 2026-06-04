from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None


logger = logging.getLogger(__name__)
_DEFAULT_BACKEND = object()


class NoOpMlflowRun:
    info = SimpleNamespace(run_id="")


class NoOpMlflowRunContext:
    def __enter__(self) -> NoOpMlflowRun:
        return NoOpMlflowRun()

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class OptionalMlflowClient:
    def __init__(self, backend: Any | None):
        self._backend = backend

    @property
    def enabled(self) -> bool:
        return self._backend is not None

    def _call(self, method_name: str, *args, **kwargs):
        if self._backend is None:
            return None
        method = getattr(self._backend, method_name, None)
        if method is None:
            return None
        try:
            return method(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - warning text is not behavior
            logger.warning("MLflow %s failed; continuing without tracking: %s", method_name, exc)
            return None

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self._call("set_tracking_uri", tracking_uri)

    def set_experiment(self, experiment_name: str) -> None:
        self._call("set_experiment", experiment_name)

    def start_run(self):
        if self._backend is None:
            return NoOpMlflowRunContext()
        return OptionalMlflowRunContext(self._backend)

    def log_param(self, key: str, value) -> None:
        self._call("log_param", key, value)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self._call("log_artifact", local_path, artifact_path=artifact_path)

    def log_metric(self, key: str, value: float, **kwargs) -> None:
        self._call("log_metric", key, value, **kwargs)


class OptionalMlflowRunContext:
    def __init__(self, backend):
        self._backend = backend
        self._active_context = None

    def __enter__(self):
        try:
            context = self._backend.start_run()
            self._active_context = context
            if hasattr(context, "__enter__"):
                return context.__enter__()
            return context
        except Exception as exc:  # pragma: no cover - warning text is not behavior
            logger.warning("MLflow start_run failed; continuing without tracking: %s", exc)
            self._active_context = None
            return NoOpMlflowRun()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._active_context is None or not hasattr(self._active_context, "__exit__"):
            return False
        try:
            return bool(self._active_context.__exit__(exc_type, exc, tb))
        except Exception as mlflow_exc:  # pragma: no cover - warning text is not behavior
            logger.warning("MLflow run close failed; continuing: %s", mlflow_exc)
            return False


def optional_mlflow_client(backend: Any | None = _DEFAULT_BACKEND, *, enabled: bool = True):
    if not enabled:
        return OptionalMlflowClient(None)
    resolved_backend = mlflow if backend is _DEFAULT_BACKEND else backend
    return OptionalMlflowClient(resolved_backend)


def configure_mlflow(tracking_uri: str, *, enabled: bool = True):
    client = optional_mlflow_client(enabled=enabled)
    client.set_tracking_uri(tracking_uri)
    return client
