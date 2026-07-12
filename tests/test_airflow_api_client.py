from __future__ import annotations

import httpx
import pytest


def test_airflow_client_triggers_editor_dag_with_submission_path():
    from pricing_pipeline.workbench.airflow import AirflowClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith(
            "/api/v2/dags/pricing_publish_editor_candidate/dagRuns"
        )
        assert request.headers["authorization"] == "Bearer token"
        assert request.read() == (
            b'{"dag_run_id":"manual__submission-1","conf":'
            b'{"submission_path":"state/submission.json",'
            b'"submission_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}'
        )
        return httpx.Response(
            200,
            json={"dag_run_id": "manual__submission-1", "state": "queued"},
        )

    client = AirflowClient(
        "http://127.0.0.1:8080/api/v2/",
        token="token",
        transport=httpx.MockTransport(handler),
    )
    result = client.trigger_dag(
        "pricing_publish_editor_candidate",
        run_id="manual__submission-1",
        conf={"submission_path": "state/submission.json", "submission_sha256": "a" * 64},
    )

    assert result.dag_id == "pricing_publish_editor_candidate"
    assert result.dag_run_id == "manual__submission-1"
    assert result.state == "queued"


def test_airflow_client_reads_dag_run_status():
    from pricing_pipeline.workbench.airflow import AirflowClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path.endswith(
            "/api/v2/dags/pricing_publish_editor_candidate/dagRuns/manual__submission-1"
        )
        return httpx.Response(
            200,
            json={
                "dag_run_id": "manual__submission-1",
                "state": "success",
                "conf": {"published_package_version": 8},
            },
        )

    client = AirflowClient(
        "http://127.0.0.1:8080/api/v2",
        transport=httpx.MockTransport(handler),
    )

    result = client.get_dag_run(
        "pricing_publish_editor_candidate",
        "manual__submission-1",
    )

    assert result.state == "success"
    assert result.payload["conf"]["published_package_version"] == 8


def test_airflow_client_raises_for_http_errors():
    from pricing_pipeline.workbench.airflow import AirflowClient

    client = AirflowClient(
        "http://127.0.0.1:8080/api/v2",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"detail": "not authenticated"})
        ),
    )

    with pytest.raises(httpx.HTTPStatusError):
        client.trigger_dag("example", run_id="manual__1", conf={})


def test_airflow_client_treats_existing_deterministic_run_as_retry():
    from pricing_pipeline.workbench.airflow import AirflowClient

    methods = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "POST":
            return httpx.Response(409, json={"detail": "run already exists"})
        return httpx.Response(
            200,
            json={"dag_run_id": "manual__submission-1", "state": "queued"},
        )

    client = AirflowClient(
        "http://127.0.0.1:8080/api/v2",
        transport=httpx.MockTransport(handler),
    )

    result = client.trigger_dag(
        "pricing_publish_editor_candidate",
        run_id="manual__submission-1",
        conf={"submission_path": "state/submission.json"},
    )

    assert methods == ["POST", "GET"]
    assert result.dag_run_id == "manual__submission-1"
    assert result.state == "queued"


def test_airflow_client_obtains_simple_auth_token_when_static_token_is_absent():
    from pricing_pipeline.workbench.airflow import AirflowClient

    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/auth/token":
            assert request.read() == b'{"username":"admin","password":"local-secret"}'
            return httpx.Response(201, json={"access_token": "fresh-token"})
        assert request.headers["authorization"] == "Bearer fresh-token"
        return httpx.Response(
            200,
            json={"dag_run_id": "manual__submission-1", "state": "queued"},
        )

    client = AirflowClient(
        "http://127.0.0.1:8080/api/v2",
        username="admin",
        password="local-secret",
        transport=httpx.MockTransport(handler),
    )

    client.trigger_dag(
        "pricing_publish_editor_candidate",
        run_id="manual__submission-1",
        conf={},
    )

    assert paths == [
        "/auth/token",
        "/api/v2/dags/pricing_publish_editor_candidate/dagRuns",
    ]
