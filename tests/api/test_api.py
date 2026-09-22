import json
import time

import pytest
from fastapi.testclient import TestClient

from modelport.api import create_app, operator_token


@pytest.fixture
def api_client(tmp_path):
    app = create_app(tmp_path / "api", start_worker=True)
    credential = operator_token(app.state.client.settings)
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer " + credential
        yield client, app.state.client


def wait(client, job_id, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["state"] in {"succeeded", "failed", "cancelled", "interrupted", "timed_out"}:
            assert job["state"] == "succeeded", job
            return job
        time.sleep(0.1)
    pytest.fail("real API worker did not complete within deadline")


def test_real_upload_conversion_validation_benchmark_events_and_download(api_client, workflow):
    client, service = api_client
    fixture = workflow["client"].store.path(workflow["source"].artifact_id)
    files = [
        ("files", (p.name, p.read_bytes(), "application/octet-stream")) for p in fixture.iterdir()
    ]
    uploaded = client.post("/api/v1/uploads", files=files)
    assert uploaded.status_code == 201, uploaded.text
    imported = client.post(
        "/api/v1/imports",
        json={"upload_id": uploaded.json()["upload_id"]},
        headers={"Idempotency-Key": "upload-once"},
    )
    assert imported.status_code == 202 and imported.headers["location"]
    same = client.post(
        "/api/v1/imports",
        json={"upload_id": uploaded.json()["upload_id"]},
        headers={"Idempotency-Key": "upload-once"},
    )
    assert same.json()["job_id"] == imported.json()["job_id"]
    done = wait(client, imported.json()["job_id"])
    source_id = done["result"]["artifact_id"]
    for _ in range(100):
        if client.get("/api/v1/capabilities").status_code == 200:
            break
        time.sleep(0.1)
    planned = client.post("/api/v1/plans", json={"source_id": source_id})
    assert planned.status_code == 200, planned.text
    # Emulate JavaScript's JSON.stringify: 0.0/1.0 become 0/1 in nested probe evidence.
    browser_plan = json.loads(
        json.dumps(planned.json()),
        parse_float=lambda value: int(float(value)) if float(value).is_integer() else float(value),
    )
    submitted = client.post(
        "/api/v1/conversions", json=browser_plan, headers={"Idempotency-Key": "convert-once"}
    )
    assert submitted.status_code == 202, submitted.text
    done = wait(client, submitted.json()["job_id"])
    target = done["result"]["artifact_id"]
    report = client.get("/api/v1/validations/" + done["result"]["validation_id"]).json()
    assert report["state"] == "validated" and report["source_id"] == source_id
    event_path = f"/api/v1/jobs/{done['job_id']}/events"
    all_events = client.get(event_path).text
    ids = [int(line[4:]) for line in all_events.splitlines() if line.startswith("id: ")]
    assert ids == sorted(set(ids)) and ids[0] == 1 and len(ids) >= 4
    replay = client.get(event_path, headers={"Last-Event-ID": str(ids[-2])}).text
    assert f"id: {ids[-1]}\n" in replay
    assert f"id: {ids[-2]}\n" not in replay
    measured = wait(
        client,
        client.post(
            "/api/v1/benchmarks", json={"artifact_id": target, "config": {"iterations": 4}}
        ).json()["job_id"],
    )
    benchmark = client.get("/api/v1/benchmarks/" + measured["result"]["benchmark_id"]).json()
    assert benchmark["measurements"]["sample_count"] == 4
    response = client.get(f"/api/v1/artifacts/{target}/download")
    assert response.status_code == 200 and response.content.startswith(b"PK")
    assert client.get("/api/v1/models").json()
    assert client.get("/health/ready").status_code == 200


def test_auth_origin_body_limits_schema_and_artifact_authorization(api_client):
    client, _ = api_client
    assert (
        client.get("/api/v1/models", headers={"Authorization": "Bearer invalid"}).status_code == 401
    )
    # Non-ASCII bytes must be rejected normally, not crash constant-time comparison.
    assert (
        client.get("/api/v1/models", headers={b"Authorization": b"Bearer \xff"}).status_code == 401
    )
    assert (
        client.get("/api/v1/models", headers={"Origin": "https://hostile.example"}).status_code
        == 403
    )
    assert client.post("/api/v1/imports", json={"path": "C:/secret"}).status_code == 422
    assert client.get("/api/v1/artifacts/" + "a" * 64 + "/download").status_code == 404
    assert (
        client.post(
            "/api/v1/imports", content=b"{}", headers={"Content-Length": str(5 * 1024**2)}
        ).status_code
        == 413
    )
    assert (
        client.post("/api/v1/uploads", files={"files": ("../evil.onnx", b"bad")}).status_code == 422
    )
    assert client.post("/api/v1/uploads", files={"files": ("evil.pt", b"bad")}).status_code == 422
    schema = client.get("/openapi.json").json()
    assert "HTTPBearer" in schema["components"]["securitySchemes"]
    reference = client.get("/docs")
    assert reference.status_code == 200 and "Authorization: Bearer" in reference.text
    assert "/api/v1/imports/huggingface" in reference.text
    assert "<script" not in reference.text and "cdn" not in reference.text.lower()
    assert "script-src 'self'" in reference.headers["content-security-policy"]
    assert client.get("/api-reference.css").status_code == 200
    problem = client.post("/api/v1/plans", json={"source_id": "missing", "device": "cuda"}).json()
    assert set(("code", "detail", "request_id", "retryable")) <= problem.keys()
    assert "C:/secret" not in json.dumps(problem)


def test_real_dataset_upload_inspection_and_purpose_validation(api_client):
    import io

    import numpy as np

    client, _ = api_client
    stream = io.BytesIO()
    np.savez(stream, images=np.zeros((4, 1, 8, 8), np.float32))
    response = client.post(
        "/api/v1/datasets?purpose=validation", files={"file": ("inputs.npz", stream.getvalue())}
    )
    assert response.status_code == 202, response.text
    done = wait(client, response.json()["job_id"])
    dataset = next(
        item
        for item in client.get("/api/v1/datasets").json()
        if item["dataset_id"] == done["result"]["dataset_id"]
    )
    assert dataset["sample_count"] == 4 and dataset["purpose"] == "validation"
    assert (
        client.post(
            "/api/v1/datasets?purpose=unknown", files={"file": ("inputs.npz", b"x")}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/datasets?purpose=validation", files={"file": ("../inputs.npz", b"x")}
        ).status_code
        == 422
    )


def test_idempotency_conflicts(api_client):
    client, _ = api_client
    a = client.post(
        "/api/v1/fixtures", json={"architecture": "vision-mlp"}, headers={"Idempotency-Key": "one"}
    )
    assert a.status_code == 202
    b = client.post(
        "/api/v1/fixtures", json={"architecture": "dual-input"}, headers={"Idempotency-Key": "one"}
    )
    assert b.status_code == 409 and b.json()["code"] == "IDEMPOTENCY_CONFLICT"
