from __future__ import annotations

import httpx
import pytest


def test_airflow_client_trigger_body_matches_airflow_3_schema():
    from airflow.api_fastapi.core_api.datamodels.dag_run import TriggerDAGRunPostBody

    from pricing_pipeline.workbench.airflow import AirflowClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v2/dags/pricing_publish_editor_candidate/dagRuns"
        assert request.headers["authorization"] == "Bearer token"
        body = request.read()
        TriggerDAGRunPostBody.model_validate_json(body)
        assert body == (
            b'{"dag_run_id":"manual__submission-1","logical_date":null,"conf":'
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


def test_airflow_client_reuses_queued_run_with_equivalent_conf():
    from pricing_pipeline.workbench.airflow import AirflowClient

    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, request.read()))
        if request.method == "POST":
            return httpx.Response(409, json={"detail": "run already exists"})
        return httpx.Response(
            200,
            json={
                "dag_run_id": "manual__submission-1",
                "state": "queued",
                "conf": {
                    "options": {"alpha": 1, "beta": 2},
                    "submission_path": "state/submission.json",
                },
            },
        )

    client = AirflowClient(
        "http://127.0.0.1:8080/api/v2",
        transport=httpx.MockTransport(handler),
    )

    result = client.trigger_dag(
        "pricing_publish_editor_candidate",
        run_id="manual__submission-1",
        conf={
            "submission_path": "state/submission.json",
            "options": {"beta": 2, "alpha": 1},
        },
    )

    assert requests == [
        (
            "POST",
            "/api/v2/dags/pricing_publish_editor_candidate/dagRuns",
            b'{"dag_run_id":"manual__submission-1","logical_date":null,"conf":'
            b'{"submission_path":"state/submission.json","options":{"beta":2,"alpha":1}}}',
        ),
        (
            "GET",
            "/api/v2/dags/pricing_publish_editor_candidate/dagRuns/manual__submission-1",
            b"",
        ),
    ]
    assert result.dag_run_id == "manual__submission-1"
    assert result.state == "queued"


def test_airflow_client_rejects_existing_run_with_different_conf():
    from pricing_pipeline.workbench import AirflowDagRunConflictError
    from pricing_pipeline.workbench.airflow import AirflowClient

    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, request.read()))
        if request.method == "POST":
            return httpx.Response(409, json={"detail": "run already exists"})
        return httpx.Response(
            200,
            json={
                "dag_run_id": "manual__submission-1",
                "state": "queued",
                "conf": {"submission_path": "state/different-submission.json"},
            },
        )

    client = AirflowClient(
        "http://127.0.0.1:8080/api/v2",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(
        AirflowDagRunConflictError,
        match="already exists with different conf",
    ):
        client.trigger_dag(
            "pricing_publish_editor_candidate",
            run_id="manual__submission-1",
            conf={"submission_path": "state/submission.json"},
        )

    assert requests == [
        (
            "POST",
            "/api/v2/dags/pricing_publish_editor_candidate/dagRuns",
            b'{"dag_run_id":"manual__submission-1","logical_date":null,"conf":'
            b'{"submission_path":"state/submission.json"}}',
        ),
        (
            "GET",
            "/api/v2/dags/pricing_publish_editor_candidate/dagRuns/manual__submission-1",
            b"",
        ),
    ]


def test_airflow_client_clears_failed_run_with_same_conf():
    from airflow.api_fastapi.core_api.datamodels.dag_run import DAGRunClearBody

    from pricing_pipeline.workbench.airflow import AirflowClient

    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        path = request.url.path
        requests.append((request.method, path, body))
        if path.endswith("/clear"):
            DAGRunClearBody.model_validate_json(body)
            return httpx.Response(
                200,
                json={
                    "dag_run_id": "manual__submission-1",
                    "state": "queued",
                    "conf": {"submission_path": "state/submission.json"},
                    "clear_marker": True,
                },
            )
        if request.method == "POST":
            return httpx.Response(409, json={"detail": "run already exists"})
        return httpx.Response(
            200,
            json={
                "dag_run_id": "manual__submission-1",
                "state": "failed",
                "conf": {"submission_path": "state/submission.json"},
            },
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

    assert requests == [
        (
            "POST",
            "/api/v2/dags/pricing_publish_editor_candidate/dagRuns",
            b'{"dag_run_id":"manual__submission-1","logical_date":null,"conf":'
            b'{"submission_path":"state/submission.json"}}',
        ),
        (
            "GET",
            "/api/v2/dags/pricing_publish_editor_candidate/dagRuns/manual__submission-1",
            b"",
        ),
        (
            "POST",
            "/api/v2/dags/pricing_publish_editor_candidate/dagRuns/"
            "manual__submission-1/clear",
            b'{"dry_run":false}',
        ),
    ]
    assert result.state == "queued"
    assert result.payload["clear_marker"] is True


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
