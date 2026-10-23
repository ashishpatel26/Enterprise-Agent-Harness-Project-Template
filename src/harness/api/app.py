"""FastAPI application factory.

uvicorn harness.api.app:create_app --factory
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from harness import __version__
from harness.api.routes import router
from harness.config import get_settings
from harness.observability.logging import configure_logging, request_id_var
from harness.observability.tracing import configure_tracing
from harness.runtime import Harness, RunConflictError, RunNotFoundError
from harness.safety import ApprovalError, InputValidationError

log = logging.getLogger(__name__)

MAX_BODY_BYTES = 1_000_000


async def _recover_periodically(harness: Harness, interval_s: float = 30.0) -> None:
    """Resume runs orphaned by a crashed replica once their lease has expired."""
    while True:
        await asyncio.sleep(interval_s)
        try:
            await harness.recover()
        except Exception:
            log.exception("run recovery sweep failed")


def _error(code: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(
        {"error": error, "detail": detail, "request_id": request_id_var.get()}, status_code=code
    )


def create_app(harness: Harness | None = None) -> FastAPI:
    settings = harness.settings if harness else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        h = harness or Harness.build(settings)
        if harness is None:
            configure_logging(settings.log_level, h.redactor)
            if settings.otel_enabled:
                configure_tracing(settings.service_name, settings.otel_endpoint)
        app.state.harness = h
        await h.startup()
        log.info("harness ready", extra={"agents": sorted(h.specs), "worker": h.worker_id})
        recovery = asyncio.create_task(_recover_periodically(h))
        try:
            yield
        finally:
            recovery.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recovery
            await h.aclose()

    app = FastAPI(
        title="agent-harness",
        version=__version__,
        description="Production harness for AI agents: runs, streaming, approvals, traces.",
        lifespan=lifespan,
    )
    if harness is not None:
        app.state.harness = harness
    app.include_router(router)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "X-User-Id", "Content-Type", "Last-Event-ID"],
        )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        rid = rid[:64] if rid.replace("-", "").isalnum() else uuid.uuid4().hex
        token = request_id_var.set(rid)
        try:
            length = request.headers.get("content-length")
            if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
                return _error(status.HTTP_413_CONTENT_TOO_LARGE, "too_large", "body too large")
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        resp = _error(exc.status_code, "http_error", str(exc.detail))
        for k, v in (exc.headers or {}).items():
            resp.headers[k] = v
        return resp

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return _error(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_request", detail)

    @app.exception_handler(InputValidationError)
    async def input_error(request: Request, exc: InputValidationError) -> JSONResponse:
        return _error(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_input", str(exc))

    @app.exception_handler(RunNotFoundError)
    async def not_found(request: Request, exc: RunNotFoundError) -> JSONResponse:
        return _error(status.HTTP_404_NOT_FOUND, "not_found", "run not found")

    @app.exception_handler(RunConflictError)
    async def conflict(request: Request, exc: RunConflictError) -> JSONResponse:
        return _error(status.HTTP_409_CONFLICT, "conflict", str(exc))

    @app.exception_handler(ApprovalError)
    async def approval_error(request: Request, exc: ApprovalError) -> JSONResponse:
        return _error(status.HTTP_409_CONFLICT, "approval_error", str(exc))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", "internal error")

    return app
