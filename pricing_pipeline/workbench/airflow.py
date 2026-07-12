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
        username: str | None = None,
        password: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        cleaned_url = str(api_url).strip().rstrip("/")
        if not cleaned_url:
            raise ValueError("Airflow API URL is required")
        cleaned_token = None if token is None else str(token).strip() or None
        cleaned_username = None if username is None else str(username).strip() or None
        cleaned_password = None if password is None else str(password) or None
        if cleaned_token is None and (cleaned_username is None) != (cleaned_password is None):
            raise ValueError("Airflow API username and password must be supplied together")
        headers = {"Accept": "application/json"}
        if cleaned_token:
            headers["Authorization"] = f"Bearer {cleaned_token}"
        self.api_url = cleaned_url
        self._username = cleaned_username
        self._password = cleaned_password
        self._has_bearer_token = cleaned_token is not None
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
        self._ensure_authenticated()
        response = self._client.post(
            self._dag_runs_url(dag_id),
            json={"dag_run_id": run_id, "conf": conf},
        )
        if response.status_code == httpx.codes.CONFLICT:
            return self.get_dag_run(dag_id, run_id)
        response.raise_for_status()
        return self._dag_run(dag_id, response.json(), fallback_run_id=run_id)

    def get_dag_run(self, dag_id: str, run_id: str) -> AirflowDagRun:
        self._ensure_authenticated()
        encoded_run_id = quote(self._required(run_id, "run_id"), safe="")
        response = self._client.get(f"{self._dag_runs_url(dag_id)}/{encoded_run_id}")
        response.raise_for_status()
        return self._dag_run(dag_id, response.json(), fallback_run_id=run_id)

    def close(self) -> None:
        self._client.close()

    def dag_run_ui_url(self, dag_id: str, run_id: str) -> str:
        ui_root = self.api_url
        if ui_root.endswith("/api/v2"):
            ui_root = ui_root[: -len("/api/v2")]
        encoded_dag_id = quote(self._required(dag_id, "dag_id"), safe="")
        encoded_run_id = quote(self._required(run_id, "run_id"), safe="")
        return f"{ui_root}/dags/{encoded_dag_id}/runs/{encoded_run_id}"

    def _dag_runs_url(self, dag_id: str) -> str:
        encoded_dag_id = quote(self._required(dag_id, "dag_id"), safe="")
        return f"{self.api_url}/dags/{encoded_dag_id}/dagRuns"

    def _ensure_authenticated(self) -> None:
        if self._has_bearer_token or self._username is None:
            return
        api_root = self.api_url
        if api_root.endswith("/api/v2"):
            api_root = api_root[: -len("/api/v2")]
        response = self._client.post(
            f"{api_root}/auth/token",
            json={"username": self._username, "password": self._password},
        )
        response.raise_for_status()
        access_token = str(response.json().get("access_token") or "").strip()
        if not access_token:
            raise RuntimeError("Airflow authentication response did not contain an access token")
        self._client.headers["Authorization"] = f"Bearer {access_token}"
        self._has_bearer_token = True

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
