from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from harness.llm import FakeLLM
from tests.integration.conftest import API_KEY, api_client

pytestmark = pytest.mark.integration


async def _wait_status(c: httpx.AsyncClient, run_id: str, *statuses: str) -> dict[str, Any]:
    for _ in range(200):
        body: dict[str, Any] = (await c.get(f"/runs/{run_id}")).json()
        if body["status"] in statuses:
            return body
        await asyncio.sleep(0.02)
    raise AssertionError(f"run never reached {statuses}: {body}")


async def test_health_ready_metrics_and_agents(make_harness: Any) -> None:
    async with api_client(make_harness({})) as c:
        assert (await c.get("/healthz")).json() == {"status": "ok"}
        ready = (await c.get("/readyz")).json()
        assert ready["status"] == "ready"
        agents = (await c.get("/agents")).json()
        assert [a["name"] for a in agents] == ["assistant", "operator"]
        metrics = await c.get("/metrics")
        assert metrics.status_code == 200 and "harness_runs_total" in metrics.text


async def test_auth_is_required(make_harness: Any) -> None:
    async with api_client(make_harness({})) as c:
        r = await c.get("/agents", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401
        assert r.json()["error"] == "http_error"
        assert r.headers["WWW-Authenticate"] == "ApiKey"
        assert (await c.get("/agents", headers={"X-User-Id": "bad id!"})).status_code == 400


async def test_run_with_tool_wait_mode(make_harness: Any) -> None:
    llm = FakeLLM(
        [
            FakeLLM.tool_call("file_read", {"path": "data/hello.txt"}),
            FakeLLM.text("The file says hello."),
        ]
    )
    async with api_client(make_harness({"assistant": llm})) as c:
        r = await c.post(
            "/runs", json={"agent": "assistant", "input": "read hello.txt", "wait": True}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "completed"
        assert body["answer"] == "The file says hello."
        assert body["stop_reason"] == "completed"
        assert r.headers["Location"] == f"/runs/{body['run_id']}"
        full = (await c.get(f"/runs/{body['run_id']}?include_messages=true")).json()
        assert any("hello from the workspace" in m["content"] for m in full["messages"])
        assert all(m["role"] != "system" for m in full["messages"])

        trace = (await c.get(f"/runs/{body['run_id']}/trace")).json()
        types = [e["type"] for s in trace["steps"] for e in s["events"]]
        assert "llm_call" in types and "observe" in types
        assert trace["totals"]["steps"] == 2
        html = await c.get(f"/runs/{body['run_id']}/trace?format=html")
        assert html.headers["content-type"].startswith("text/html")

        listed = (await c.get("/runs")).json()
        assert [x["run_id"] for x in listed] == [body["run_id"]]


async def test_async_run_and_sse_stream(make_harness: Any) -> None:
    llm = FakeLLM([FakeLLM.text("streamed answer")])
    async with api_client(make_harness({"assistant": llm})) as c:
        run = (await c.post("/runs", json={"agent": "assistant", "input": "hi"})).json()
        assert run["status"] in ("pending", "running", "completed")
        events: list[tuple[str, dict[str, Any]]] = []
        async with c.stream("GET", f"/runs/{run['run_id']}/stream") as resp:
            assert resp.status_code == 200
            event_type = ""
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event_type = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    events.append((event_type, json.loads(line.split(":", 1)[1])))
        types = [t for t, _ in events]
        assert types[0] == "run_started" and types[-1] == "run_finished"
        assert "text_delta" in types and "llm_call" in types

        # Reconnect with Last-Event-ID replays only what came after.
        async with c.stream(
            "GET", f"/runs/{run['run_id']}/stream", headers={"Last-Event-ID": "1"}
        ) as resp:
            body = await resp.aread()
        assert b"run_started" not in body and b"run_finished" in body


async def test_approval_flow_over_http(make_harness: Any) -> None:
    llm = FakeLLM(
        [
            FakeLLM.tool_call("python_exec", {"code": "print(6*7)"}, call_id="exec1"),
            FakeLLM.text("The answer is 42."),
        ]
    )
    async with api_client(make_harness({"operator": llm})) as c:
        run = (await c.post("/runs", json={"agent": "operator", "input": "compute"})).json()
        paused = await _wait_status(c, run["run_id"], "awaiting_approval")
        [pending] = paused["pending_approvals"]
        assert pending["tool"] == "python_exec" and pending["call_id"] == "exec1"

        r = await c.post(
            f"/runs/{run['run_id']}/approve",
            json={"call_id": "exec1", "approved": True, "comment": "fine"},
            headers={"X-User-Id": "reviewer-1"},
        )
        assert r.status_code == 200, r.text
        done = await _wait_status(c, run["run_id"], "completed")
        assert done["answer"] == "The answer is 42."

        again = await c.post(
            f"/runs/{run['run_id']}/approve", json={"call_id": "exec1", "approved": True}
        )
        assert again.status_code == 409


async def test_approver_keys_restrict_who_can_approve(make_harness: Any) -> None:
    llm = FakeLLM([FakeLLM.tool_call("python_exec", {"code": "1"}, call_id="x")])
    harness = make_harness({"operator": llm}, approver_keys="approver-key-999")
    async with api_client(harness) as c:
        run = (await c.post("/runs", json={"agent": "operator", "input": "go"})).json()
        await _wait_status(c, run["run_id"], "awaiting_approval")
        r = await c.post(f"/runs/{run['run_id']}/approve", json={"call_id": "x", "approved": True})
        assert r.status_code == 403


async def test_cancel_paused_run(make_harness: Any) -> None:
    llm = FakeLLM([FakeLLM.tool_call("python_exec", {"code": "1"}, call_id="x")])
    async with api_client(make_harness({"operator": llm})) as c:
        run = (await c.post("/runs", json={"agent": "operator", "input": "go"})).json()
        await _wait_status(c, run["run_id"], "awaiting_approval")
        r = await c.post(f"/runs/{run['run_id']}/cancel")
        assert r.status_code == 200 and r.json()["status"] == "cancelled"
        assert r.json()["stop_reason"] == "cancelled"


async def test_tenant_isolation(make_harness: Any) -> None:
    llm = FakeLLM([FakeLLM.text("secret answer")])
    harness = make_harness({"assistant": llm}, api_keys=f"{API_KEY},other-tenant-key-1234")
    async with api_client(harness) as c:
        run = (
            await c.post("/runs", json={"agent": "assistant", "input": "x", "wait": True})
        ).json()
        other = {"X-API-Key": "other-tenant-key-1234"}
        assert (await c.get(f"/runs/{run['run_id']}", headers=other)).status_code == 404
        assert (await c.get(f"/runs/{run['run_id']}/stream", headers=other)).status_code == 404
        assert (await c.get("/runs", headers=other)).json() == []


async def test_request_validation_and_errors(make_harness: Any) -> None:
    async with api_client(make_harness({})) as c:
        r = await c.post("/runs", json={"agent": "nope", "input": "x"})
        assert r.status_code == 404
        r = await c.post("/runs", json={"agent": "assistant", "input": "   "})
        assert r.status_code == 422 and r.json()["error"] == "invalid_input"
        r = await c.post("/runs", json={"agent": "assistant"})
        assert r.status_code == 422 and r.json()["error"] == "invalid_request"
        r = await c.post("/runs", json={"agent": "assistant", "input": "x", "extra": 1})
        assert r.status_code == 422
        assert (await c.get("/runs/run_missing")).status_code == 404
        big = await c.post(
            "/runs", content=b"x" * 1_000_001, headers={"Content-Type": "application/json"}
        )
        assert big.status_code == 413
        r = await c.get("/healthz", headers={"X-Request-ID": "abc-123"})
        assert r.headers["X-Request-ID"] == "abc-123"
        assert r.headers["X-Content-Type-Options"] == "nosniff"


async def test_api_rate_limit(make_harness: Any) -> None:
    async with api_client(make_harness({}, rate_limit_per_minute=2)) as c:
        codes = [(await c.get("/agents")).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        r = await c.get("/agents")
        assert int(r.headers["Retry-After"]) >= 1


async def test_errors_never_leak_secrets(make_harness: Any) -> None:
    from harness.llm import NonRetryableLLMError

    llm = FakeLLM([NonRetryableLLMError("auth failed for key sk-ant-api03-abcdefghijklmnopqrstu")])
    async with api_client(make_harness({"assistant": llm})) as c:
        run = (
            await c.post("/runs", json={"agent": "assistant", "input": "x", "wait": True})
        ).json()
        assert run["stop_reason"] == "llm_error"
        assert "sk-ant" not in run["error"]
