# Evals

Agents regress silently. A prompt tweak, a model upgrade or a new tool can quietly break
behaviour that used to work. The eval harness makes that visible and blocks it in CI.

## Two modes

| Mode | Model | What it measures | When |
|---|---|---|---|
| **Offline** (default) | The task's scripted trajectory (`script:`) | The harness end to end: prompts, policy, sandbox, tools, budgets, stop reasons. Deterministic and free. | Every push to `main`, locally with `make eval` |
| **Live** (`--live`) | The model in the agent's YAML | Model + prompt quality; rubrics graded by a judge model | Nightly, on demand, before a model or prompt change |

```bash
make eval                                                          # offline, gated on the baseline
uv run harness eval evals/tasks --live --judge anthropic/claude-opus-5-5
make eval-baseline                                                 # re-record the baseline
```

Reports are written to `evals/reports/eval-<timestamp>.md` and `.json`.

## Writing a task set

```yaml
# evals/tasks/support.yaml
suite: support
agent: support
tasks:
  - id: password-reset
    input: How do I reset my password?
    script: support_password_reset    # optional offline trajectory (harness.demo.SCRIPTS)
    auto_approve: false               # approve gated tool calls automatically
    expect:
      contains: ["Settings", "Reset password"]
      not_contains: ["I don't know"]
      regex: "docs/[a-z_]+\\.md"
      tools_called: [file_read]
      tools_not_called: [web_fetch]
      max_steps: 4
      stop_reason: completed
      rubric: >
        Gives the exact steps from the docs, cites the file, and does not invent UI that
        is not described there.
      rubric_threshold: 0.7
```

### Scorers

| Scorer | Passes when |
|---|---|
| `exact` | The answer equals the string (trimmed) |
| `contains` / `not_contains` | Every / none of the strings appear (case-insensitive) |
| `regex` | The pattern matches the answer |
| `tools_called` / `tools_not_called` | Tool-use checks over the whole run |
| `max_steps` | The run took at most N steps |
| `stop_reason` | The run ended with this reason (default `completed`) |
| `rubric` | A judge model scores the answer 0-1 ≥ `rubric_threshold`. Skipped (not failed) when no judge is configured. |

A task passes when all non-skipped scorers pass. The judge sees the answer fenced as
untrusted data, so the answer cannot instruct the judge.

### Offline scripts

A script is a list of model turns: fixed responses, or functions of the conversation so
far. They live in `harness/demo.py` (`SCRIPTS`). Write one when you want CI to protect a
harness behaviour, for example "the PII query is blocked by policy" or "the analyst runs
code in the sandbox and reports numbers". Tasks without a script are skipped offline and
run in live mode.

## Regression gate

`evals/baseline.json` records each suite's pass rate and each task's pass/score. With
`--baseline`, the run fails (exit 1) when any of these happen:

- a task that passed in the baseline now fails
- a task's score drops by more than `--tolerance`
- a suite's pass rate drops

New tasks never count as regressions. Commit baseline changes together with the code
change that justifies them, so the diff is reviewed.

## Practical advice

- Start from real failures: every bug report becomes a task.
- Keep offline tasks focused on harness guarantees; keep live tasks focused on quality.
- Track `cost_usd` and `steps` in the report. A cheaper run that needs more steps is
  not cheaper.
- Change one thing at a time (model, prompt, tool) and compare reports.
