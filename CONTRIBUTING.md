# Contributing

Thanks for helping improve agent-harness. Bug reports, docs fixes, new tools, providers and
sandbox backends are all welcome.

## Setup

```bash
make setup           # uv sync, pre-commit hooks, .env
make check           # lint + typecheck + tests + coverage gate (what CI runs)
```

The test suite uses `FakeLLM`, so no API keys are needed. Docker sandbox escape tests run
when a Docker daemon is available (`make sandbox-image` first) and always in CI.

## Ground rules

- **One change per PR.** Keep diffs focused; open an issue first for larger designs.
- **Tests with every change.** New behaviour needs a test; bug fixes need a test that
  fails without the fix. Every failure path should have a test.
- **Types and lint.** `mypy --strict` and `ruff` must pass. No `type: ignore` without a
  specific error code.
- **Timeouts everywhere.** Any new external call (HTTP, DB, subprocess, SDK) needs an
  explicit timeout.
- **Secrets.** Never log or return secrets. Use `SecretStr` in settings and make sure the
  `Redactor` covers new token formats you introduce.
- **Safety-relevant changes** (policy, sandbox, approval, redaction, auth) need a test
  that demonstrates the protection, and a note in `docs/safety.md` if behaviour changes.
- **Evals.** If you change prompts or agent behaviour, run `make eval`. Update
  `evals/baseline.json` only with a justification in the PR.

## Adding things

| Contribution | Start from |
|---|---|
| Tool | `src/harness/tools/builtin/` and the "Add a tool" section of the README |
| Provider | `src/harness/llm/anthropic_provider.py`; register in `llm/factory.py` or via the `harness.providers` entry point |
| Sandbox | `src/harness/sandbox/base.py`; add escape tests in `tests/sandbox/` |
| Agent | `agents/*.yaml`; add an eval suite in `evals/tasks/` |

## Commits and releases

Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`,
`test:`, `chore:`). Add user-visible changes to `CHANGELOG.md` under "Unreleased".

By contributing you agree that your contributions are licensed under the MIT License.
