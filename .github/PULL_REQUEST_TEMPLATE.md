## What and why

<!-- What does this change do, and what problem does it solve? Link issues. -->

## How it was tested

<!-- Tests added/changed; `make check` output; eval report if prompts or agents changed. -->

## Checklist

- [ ] `make check` passes (ruff, mypy --strict, tests, coverage gate)
- [ ] New behaviour and failure paths have tests
- [ ] Every new external call has a timeout
- [ ] No secrets in code, logs, events or responses
- [ ] Safety-relevant change? Protection is tested and `docs/safety.md` is updated
- [ ] Prompts/agents changed? `make eval` run; baseline updates are justified
- [ ] User-visible change is in `CHANGELOG.md`
