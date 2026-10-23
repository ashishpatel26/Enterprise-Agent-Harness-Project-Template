"""HTTP routes: runs, streaming, approvals, agents, health."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sse_starlette.sse import EventSourceResponse

from harness.api.auth import Principal, get_harness, principal
from harness.api.schemas import AgentView, ApproveRequest, CreateRunRequest, RunView
from harness.api.trace import build_trace, render_html
from harness.core.types import RunState, RunStatus
from harness.events import TERMINAL_EVENTS
from harness.observability.metrics import REGISTRY
from harness.runtime import Harness, RunNotFoundError

router = APIRouter()
HarnessDep = Annotated[Harness, Depends(get_harness)]
PrincipalDep = Annotated[Principal, Depends(principal)]


async def _owned_run(harness: Harness, run_id: str, who: Principal) -> RunState:
    try:
        state = await harness.get(run_id)
    except RunNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found") from None
    # Other tenants get the same 404 as a missing run: no existence oracle.
    if state.metadata.get("owner") != who.key_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    return state


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED, response_model=RunView, tags=["runs"])
async def create_run(
    body: CreateRunRequest, harness: HarnessDep, who: PrincipalDep, response: Response
) -> RunView:
    if body.agent not in harness.specs:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown agent {body.agent!r}")
    state = await harness.create_run(
        body.agent,
        body.input,
        user_id=who.user_id,
        session_id=body.session_id,
        owner=who.key_id,
    )
    task = harness.start(state.run_id)
    if body.wait:
        timeout = harness.specs[body.agent].limits.timeout_s + 30
        # shield: a client disconnect must not cancel the run itself.
        try:
            state = await asyncio.wait_for(asyncio.shield(task), timeout)
        except TimeoutError:
            state = await harness.get(state.run_id)
        response.status_code = status.HTTP_200_OK
    response.headers["Location"] = f"/runs/{state.run_id}"
    return RunView.from_state(state, harness.redactor)


@router.get("/runs", response_model=list[RunView], tags=["runs"])
async def list_runs(
    harness: HarnessDep,
    who: PrincipalDep,
    session_id: Annotated[str | None, Query(pattern=r"^[A-Za-z0-9_\-:.]{1,128}$")] = None,
    run_status: Annotated[RunStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[RunView]:
    runs = await harness.state.runs.list_runs(status=run_status, session_id=session_id, limit=1000)
    mine = [r for r in runs if r.metadata.get("owner") == who.key_id][:limit]
    return [RunView.from_state(r, harness.redactor) for r in mine]


@router.get("/runs/{run_id}", response_model=RunView, tags=["runs"])
async def get_run(
    run_id: str,
    harness: HarnessDep,
    who: PrincipalDep,
    include_messages: bool = False,
) -> RunView:
    state = await _owned_run(harness, run_id, who)
    return RunView.from_state(state, harness.redactor, include_messages=include_messages)


@router.get("/runs/{run_id}/stream", tags=["runs"])
async def stream_run(
    run_id: str,
    request: Request,
    harness: HarnessDep,
    who: PrincipalDep,
    last_event_id: Annotated[str | None, Header()] = None,
) -> EventSourceResponse:
    """Server-Sent Events. Reconnect with `Last-Event-ID` to resume without gaps."""
    await _owned_run(harness, run_id, who)

    async def events() -> AsyncIterator[dict[str, Any]]:
        async for item in harness.events.subscribe(run_id, last_event_id):
            if await request.is_disconnected():
                return
            if item is None:
                # Idle: close if the run has ended and no more events will arrive.
                state = await harness.get(run_id)
                if state.status.terminal:
                    yield {"event": "end", "data": json.dumps({"status": state.status.value})}
                    return
                continue
            event_id, event = item
            data = harness.redactor.redact_obj(event.model_dump(mode="json"))
            yield {"id": event_id, "event": event.type, "data": json.dumps(data)}
            if event.type in TERMINAL_EVENTS:
                return

    return EventSourceResponse(events(), ping=15)


@router.post("/runs/{run_id}/approve", response_model=RunView, tags=["runs"])
async def approve(
    run_id: str, body: ApproveRequest, harness: HarnessDep, who: PrincipalDep
) -> RunView:
    if not who.can_approve:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "this API key cannot approve tool calls")
    await _owned_run(harness, run_id, who)
    state = await harness.decide(
        run_id, body.call_id, approved=body.approved, reviewer=who.user_id, comment=body.comment
    )
    return RunView.from_state(state, harness.redactor)


@router.post("/runs/{run_id}/cancel", response_model=RunView, tags=["runs"])
async def cancel(run_id: str, harness: HarnessDep, who: PrincipalDep) -> RunView:
    await _owned_run(harness, run_id, who)
    return RunView.from_state(await harness.cancel(run_id), harness.redactor)


@router.get("/runs/{run_id}/trace", tags=["observability"], response_model=None)
async def trace(
    run_id: str,
    harness: HarnessDep,
    who: PrincipalDep,
    fmt: Annotated[str, Query(alias="format", pattern="^(json|html)$")] = "json",
) -> dict[str, Any] | HTMLResponse:
    state = await _owned_run(harness, run_id, who)
    events = [e for _, e in await harness.events.history(run_id)]
    data: dict[str, Any] = harness.redactor.redact_obj(build_trace(state, events))
    return HTMLResponse(render_html(data)) if fmt == "html" else data


@router.get("/agents", response_model=list[AgentView], tags=["agents"])
async def list_agents(harness: HarnessDep, who: PrincipalDep) -> list[AgentView]:
    return [
        AgentView(
            name=s.name,
            description=s.description,
            version=s.version,
            model=f"{s.model.provider}/{s.model.name}",
            tools=s.tools,
            sub_agents=s.sub_agents,
        )
        for s in sorted(harness.specs.values(), key=lambda s: s.name)
    ]


@router.get("/healthz", tags=["health"])
async def healthz() -> dict[str, str]:
    """Liveness: the process is up."""
    return {"status": "ok"}


@router.get("/readyz", tags=["health"])
async def readyz(harness: HarnessDep, response: Response) -> dict[str, Any]:
    """Readiness: dependencies respond within a short timeout."""
    checks: dict[str, str] = {}
    try:
        async with asyncio.timeout(3):
            await harness.state.runs.load("__readyz__")
        checks["state_store"] = "ok"
    except Exception as exc:
        checks["state_store"] = f"error: {type(exc).__name__}"
    checks["agents"] = "ok" if harness.specs else "error: no agents loaded"
    checks["sandbox"] = "ok" if harness.sandbox is not None else "disabled"
    ready = all(v in ("ok", "disabled") for v in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ready else "not_ready", "checks": checks}


@router.get("/metrics", tags=["observability"], include_in_schema=False)
async def metrics() -> Response:
    """Prometheus scrape endpoint. Restrict to your internal network at the ingress."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
