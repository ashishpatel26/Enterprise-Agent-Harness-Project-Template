.DEFAULT_GOAL := help
SHELL := /bin/bash
UV ?= uv
CORE_COVERAGE := src/harness/core/*,src/harness/tools/*,src/harness/safety/*,src/harness/sandbox/*

.PHONY: help setup dev test cov lint format typecheck check eval eval-baseline demo sandbox-image up down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and git hooks
	$(UV) sync
	$(UV) run pre-commit install
	@test -f .env || cp .env.example .env

dev: ## Run the API with auto-reload
	$(UV) run uvicorn harness.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

test: ## Run the test suite with coverage
	$(UV) run pytest --cov --cov-report=term --cov-report=xml

cov: test ## Enforce >=80% coverage on core, tools, safety, sandbox
	$(UV) run coverage report --include="$(CORE_COVERAGE)" --fail-under=80

lint: ## Ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Auto-format and auto-fix
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck: ## mypy --strict
	$(UV) run mypy

check: lint typecheck cov ## Everything CI runs

eval: ## Run eval suites and compare against the stored baseline
	$(UV) run harness eval evals/tasks --baseline evals/baseline.json

eval-baseline: ## Re-record the eval baseline (review the diff before committing)
	$(UV) run harness eval evals/tasks --write-baseline evals/baseline.json

demo: ## Run the data-analyst agent on examples/data/sales.csv
	$(UV) run harness run data_analyst "Analyze data/sales.csv and give me insights"

sandbox-image: ## Build the locked-down sandbox image
	docker build -f docker/sandbox.Dockerfile -t agent-harness-sandbox:latest .

up: sandbox-image ## Start the full stack (api, postgres, redis, otel-collector, jaeger)
	docker compose up -d --build

down: ## Stop the stack
	docker compose down
