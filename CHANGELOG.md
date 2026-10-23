# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-23

### Added

- Resumable ReAct agent loop (Think → Plan → Act → Observe → Reflect → Answer) with max
  steps, token and cost budgets, wall-clock timeout, parallel tool calls and a stop reason
  on every run.
- Provider-agnostic LLM layer: Anthropic (streaming, adaptive thinking, preserved-thinking
  replay, refusal fallbacks, prompt caching) and OpenAI adapters, `FakeLLM`, retries with
  full-jitter backoff, provider fallback, cost accounting, provider plugins.
- Versioned prompt templates, memory injection and append-only context compaction.
- Tool registry with schema-from-signature, JSON-schema validation, timeouts, output caps,
  permission levels and plugins; built-in `file_read`, `python_exec`, `sql_query`,
  `web_fetch`, `delegate`.
- Docker and Kubernetes sandboxes with escape tests; dev-only unsafe local sandbox.
- Memory: Redis sessions, pgvector long-term memory, episodic records.
- Safety: YAML policy engine, human approvals with four-eyes option, prompt-injection
  screening, secret redaction, rate limits, hash-chained audit log.
- State: checkpoints, leases, crash recovery, at-most-once side effects.
- Multi-agent orchestration with a supervisor and isolated sub-agent runs.
- Observability: OpenTelemetry spans, Prometheus metrics, JSON logs, run trace view.
- Eval harness with offline/live modes, LLM-as-judge rubrics and a CI regression gate.
- FastAPI with SSE streaming, approvals, API-key tenancy; CLI; offline demo.
- Dockerfile, docker-compose stack, Helm chart, GitHub Actions CI and eval workflows.
