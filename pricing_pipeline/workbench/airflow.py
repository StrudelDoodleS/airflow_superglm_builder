from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx


@dataclass(frozen=True)
class AirflowDagRun:
    dag_id: str
    dag_run_id: str
    state: str
    payload: dict[str, Any]


class AirflowClient:
    """Small Airflow 3 REST client used by the notebook workbench."""

    def __init__(
        self,
        api_url: str,
        *,
        token: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        cleaned_url = str(api_url).strip().rstrip("/")
        if not cleaned_url:
            raise ValueError("Airflow API URL is required")
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.api_url = cleaned_url
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    def trigger_dag(
        self,
        dag_id: str,
        *,
        run_id: str,
        conf: dict[str, Any],
    ) -> AirflowDagRun:
        response = self._client.post(
            self._dag_runs_url(dag_id),
            json={"dag_run_id": run_id, "conf": conf},
        )
        response.raise_for_status()
        return self._dag_run(dag_id, response.json(), fallback_run_id=run_id)

    def get_dag_run(self, dag_id: str, run_id: str) -> AirflowDagRun:
        encoded_run_id = quote(self._required(run_id, "run_id"), safe="")
        response = self._client.get(f"{self._dag_runs_url(dag_id)}/{encoded_run_id}")
        response.raise_for_status()
        return self._dag_run(dag_id, response.json(), fallback_run_id=run_id)

    def close(self) -> None:
        self._client.close()

    def _dag_runs_url(self, dag_id: str) -> str:
        encoded_dag_id = quote(self._required(dag_id, "dag_id"), safe="")
        return f"{self.api_url}/dags/{encoded_dag_id}/dagRuns"

    @staticmethod
    def _required(value: str, name: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError(f"{name} is required")
        return cleaned

    @staticmethod
    def _dag_run(
        dag_id: str,
        payload: dict[str, Any],
        *,
        fallback_run_id: str,
    ) -> AirflowDagRun:
        run_id = payload.get("dag_run_id") or payload.get("run_id") or fallback_run_id
        return AirflowDagRun(
            dag_id=dag_id,
            dag_run_id=str(run_id),
            state=str(payload.get("state") or "unknown").lower(),
            payload=payload,
        )
