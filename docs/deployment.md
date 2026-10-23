# Deployment

## Configuration

All settings come from environment variables (or `.env`) and are validated at startup.
See `.env.example` for the full list. The important ones:

| Variable | Default | Notes |
|---|---|---|
| `HARNESS_ENV` | `dev` | `prod` enforces API keys, Redis, Postgres and a real sandbox |
| `HARNESS_API_KEYS` | (none) | Comma-separated; each key is a tenant |
| `HARNESS_APPROVER_KEYS` | (none) | Only these keys may approve tool calls |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | (none) | Provider credentials |
| `HARNESS_STORAGE_BACKEND` | `memory` | `redis` for anything beyond one process |
| `HARNESS_REDIS_URL` | `redis://localhost:6379/0` | State, sessions, events, rate limits, audit |
| `HARNESS_DATABASE_URL` | (none) | Postgres with pgvector for long-term memory |
| `HARNESS_SANDBOX_BACKEND` | `docker` | `docker`, `kubernetes`, `local_unsafe` (dev), or a plugin |
| `HARNESS_SANDBOX_RUNTIME` | (none) | Docker runtime or Kubernetes RuntimeClass (e.g. gVisor) |
| `HARNESS_OTEL_ENABLED` / `HARNESS_OTEL_ENDPOINT` | `false` | OTLP/gRPC trace export |
| `HARNESS_WEB_FETCH_ALLOWED_DOMAINS` | (none) | `web_fetch` is disabled until set |

## docker-compose (single host)

```bash
cp .env.example .env     # set HARNESS_API_KEYS and ANTHROPIC_API_KEY
make up                  # API :8000, Jaeger :16686, Redis, Postgres+pgvector, OTel collector
```

The API container mounts the Docker socket to start sandbox containers. **Docker socket
access is root-equivalent on that host.** Use this setup for development or on a
dedicated host, not on shared infrastructure. Set `DOCKER_GID` to the socket's group id
so the non-root API user can use it.

## Kubernetes (Helm)

The chart in `deploy/helm/agent-harness` deploys the API and runs every `python_exec` in
its own Pod. It needs no Docker socket.

```bash
# 1. Images
docker build -t ghcr.io/your-org/agent-harness:0.1.0 .
docker build -f docker/sandbox.Dockerfile -t ghcr.io/your-org/agent-harness-sandbox:0.1.0 .

# 2. Secrets (prefer External Secrets / your secret manager)
kubectl create namespace agents
kubectl -n agents create secret generic harness-secrets \
  --from-literal=HARNESS_API_KEYS=... \
  --from-literal=ANTHROPIC_API_KEY=... \
  --from-literal=HARNESS_REDIS_URL=redis://redis:6379/0 \
  --from-literal=HARNESS_DATABASE_URL=postgresql://user:pass@postgres:5432/harness

# 3. Install
helm install harness deploy/helm/agent-harness -n agents \
  --set secrets.existingSecret=harness-secrets \
  --set image.repository=ghcr.io/your-org/agent-harness \
  --set sandbox.image=ghcr.io/your-org/agent-harness-sandbox:0.1.0 \
  --set sandbox.runtimeClassName=gvisor
```

What the chart creates:

| Resource | Purpose |
|---|---|
| Deployment | Non-root, read-only root FS, all capabilities dropped, RuntimeDefault seccomp, startup/liveness/readiness probes, topology spread, rolling updates with zero unavailability |
| HPA + PDB | Scale on CPU (2-10 replicas), keep at least one available during disruptions |
| Sandbox namespace | Pod Security `restricted`, deny-all NetworkPolicy for sandbox Pods (no egress, not even DNS), ResourceQuota on Pods/CPU/memory |
| RBAC | The API's ServiceAccount may only create/get/delete Pods and ConfigMaps in the sandbox namespace |
| NetworkPolicy (API) | Ingress on 8000, optionally only from listed namespaces |
| Ingress (optional) | Buffering off and long read timeouts for SSE |
| ServiceMonitor (optional) | Prometheus Operator scraping of `/metrics` |

Override agents or the policy without rebuilding the image:

```bash
helm upgrade harness deploy/helm/agent-harness -n agents --reuse-values \
  --set-file agents.support\\.yaml=agents/support.yaml \
  --set-file policy=policies/default.yaml
```

Kubernetes sandbox limits:

- Inputs are capped at about 900 KB per execution (ConfigMap size).
- stdout and stderr arrive merged.
- The PID limit is the kubelet's `podPidsLimit`.
- For stronger isolation than namespaces and seccomp, install gVisor or Kata and set
  `sandbox.runtimeClassName`.

Redis and Postgres (with the `vector` extension) are not part of the chart. Use managed
services or your platform's operators.

## Operations

| Concern | How |
|---|---|
| Scaling | API replicas are stateless; runs are coordinated through Redis leases. Any replica can serve a run's SSE stream (Redis Streams). |
| Crash recovery | Runs are checkpointed every step. On startup, and whenever a lease expires, another replica resumes interrupted runs (`Harness.recover()`). |
| Graceful shutdown | Uvicorn drains for 30s (`terminationGracePeriodSeconds: 45`); unfinished runs are resumed elsewhere. |
| Observability | `/metrics` (Prometheus); OTLP traces with GenAI attributes; JSON logs with `request_id` and `run_id`; `/runs/{id}/trace` timeline. |
| Key metrics | `harness_runs_total{stop_reason}`, `harness_llm_latency_seconds`, `harness_cost_usd_total`, `harness_tool_calls_total{outcome}`, `harness_runs_active`, `harness_errors_total`. |
| Audit | Redis Stream `harness:audit` (multi-replica) or the hash-chained file log; ship to your SIEM. |
| Database schema | The pgvector table and HNSW index are created idempotently at startup. |
| Upgrades | `helm upgrade`; config changes roll pods via a checksum annotation. |
