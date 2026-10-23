# syntax=docker/dockerfile:1.7
# ---- build stage: resolve and install dependencies into a venv -------------------
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra kubernetes --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra kubernetes

# ---- runtime stage: no build tools, non-root user -------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH" \
    HARNESS_AGENTS_DIR=/app/agents HARNESS_POLICY_PATH=/app/policies/default.yaml \
    HARNESS_WORKSPACE_ROOT=/app/examples HARNESS_AUDIT_LOG_PATH=/app/var/audit.log
RUN groupadd --gid 10001 harness \
 && useradd --uid 10001 --gid harness --no-create-home --shell /usr/sbin/nologin harness
WORKDIR /app
COPY --from=build --chown=harness:harness /app/.venv /app/.venv
COPY --from=build --chown=harness:harness /app/src /app/src
COPY --chown=harness:harness agents ./agents
COPY --chown=harness:harness policies ./policies
COPY --chown=harness:harness examples ./examples
RUN mkdir -p /app/var && chown harness:harness /app/var
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)" || exit 1
CMD ["uvicorn", "harness.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--timeout-graceful-shutdown", "30"]
