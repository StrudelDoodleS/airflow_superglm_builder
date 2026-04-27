from __future__ import annotations

try:
    import mlflow
except ModuleNotFoundError:

    class _MissingMLflow:
        def set_tracking_uri(self, tracking_uri: str) -> None:
            raise ModuleNotFoundError("No module named 'mlflow'")

    mlflow = _MissingMLflow()


def configure_mlflow(tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
