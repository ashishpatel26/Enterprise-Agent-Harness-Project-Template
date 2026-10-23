# Safety

An agent turns untrusted text (user input, web pages, files, tool output) into actions.
The harness assumes the model can be manipulated and puts deterministic controls around
everything it does.

## Threat model

| Threat | Controls |
|---|---|
| Prompt injection through tool output (indirect) | Tool output travels as tool results, never as system/user text; the system prompt says tool output is data; heuristics flag and fence suspicious output (`annotate`) or withhold it (`block`); policy gates actions regardless of what the model "decides" |
| Dangerous actions | Permission levels (`read`, `write`, `dangerous`), YAML policy with deny/allow/require-approval, human approval with optional four-eyes rule |
| Model-written code | Sandbox: no network, read-only root FS, non-root, capabilities dropped, seccomp, CPU/memory/PID/time limits, capped scratch space, output caps; optional gVisor |
| Data exfiltration | No sandbox network; `web_fetch` disabled until domains are allow-listed, HTTPS only, private IPs blocked, no redirects followed; secrets redacted from tool output before it reaches the model provider |
| Secret leakage | Redaction in logs, audit records, events, API responses, traces and tool output; configured keys are redacted by exact value; secrets only via env / Secret objects |
| Abuse and runaway cost | Per-API-key and per-user tool rate limits; per-run step, token, cost and wall-clock budgets; sub-agent depth limit |
| Cross-tenant access | Runs are owned by the API key that created them; other keys get 404 |
| Tampering with history | Hash-chained, append-only audit log (`harness audit-verify`) or a Redis Stream |
| Duplicate side effects after crashes | Idempotency keys per tool call; at-most-once execution for non-idempotent tools |

## Policy engine

`policies/default.yaml` is evaluated for every tool call. The first matching rule wins,
so put `deny` rules first. When no rule matches, the default for the tool's permission
level applies.

```yaml
version: 1
defaults: {read: allow, write: allow, dangerous: require_approval}
rules:
  - name: no-pii-tables
    tools: [sql_query]
    args:
      query: {matches: "(?i)\\b(users|customers)\\b"}
    effect: deny
    reason: PII tables are off limits
  - name: analysts-run-sandboxed-code
    tools: [python_exec]
    agents: [data_analyst]
    effect: allow
```

- **Selectors.** `tools`, `agents` and `users` take globs. `permissions` takes
  `read` / `write` / `dangerous`.
- **Argument conditions.** `equals`, `in`, `matches`, `not_matches`, `max_length`.
- **Unknown tools** are always denied.
- **Validation.** Policies are checked strictly at load time; a typo is an error, not a
  silent allow.

A denied call is not executed. The model receives `Blocked by policy: rule '<name>': <reason>`
and can adapt.

## Human approval

When a call needs approval, the run pauses (`status: awaiting_approval`) and emits an
`approval_required` event with the tool, redacted arguments and reason. A reviewer then
calls:

```http
POST /runs/{run_id}/approve
{"call_id": "...", "approved": true, "comment": "ok"}
```

Once nothing is pending, the run resumes from its checkpoint. A denied call is returned
to the model as an error that includes the reviewer's comment.

| Setting | Effect |
|---|---|
| `HARNESS_APPROVER_KEYS` | Only these API keys may approve |
| `HARNESS_ALLOW_SELF_APPROVAL=false` | The user who started the run cannot approve it (four-eyes); denying is always allowed |

Every decision is audited.

## Sandboxing

| Backend | Use | Isolation |
|---|---|---|
| `docker` (default) | single host, docker-compose | one container per execution; `network_mode=none`, read-only rootfs, inputs in a root-owned read-only `/data` volume, `noexec` tmpfs scratch, uid 10001, `cap_drop=ALL`, `no-new-privileges`, memory without swap, CPU/PID/nofile/fsize limits, wall-clock kill, output caps; `HARNESS_SANDBOX_RUNTIME=runsc` for gVisor |
| `kubernetes` | Kubernetes / Helm | one Pod per execution in a dedicated namespace with Pod Security `restricted`, deny-all NetworkPolicy, ResourceQuota, no ServiceAccount token, optional RuntimeClass (gVisor/Kata) |
| `local_unsafe` | development only | **no isolation**; refuses to start without `HARNESS_ALLOW_UNSAFE_SANDBOX=true`; forbidden in prod |

The Docker backend needs the Docker socket, which is root-equivalent on that host. Run the
API on a dedicated host or use the Kubernetes backend.

Escape tests (`tests/sandbox/test_escape.py`) check that the network is unreachable, DNS
fails, the root FS and `/data` are read-only, scratch space is bounded, the process runs
as non-root with no capabilities, and memory, fork-bomb and output limits hold. CI runs
them against a real Docker daemon.

## Prompt injection

`safety/injection.py` scores tool output for override phrasing, role reassignment, fake
role markers, exfiltration instructions and invisible Unicode (tag characters,
zero-width). Above the threshold, the output is:

- **`annotate`** (default): fenced in `<untrusted_tool_output>` with a notice.
- **`block`**: withheld.

Detections are audited. Heuristics are a tripwire, not a guarantee. The structural
controls (policy, approvals, sandbox, no network) are what bound the damage of a
successful injection.

## Redaction

`Redactor` removes known token formats (Anthropic, OpenAI, AWS, GitHub, Slack, Google,
JWT, PEM keys, bearer tokens, URL credentials, `password=` style pairs) and the exact
values of every configured secret. Values under sensitive keys (`*token*`, `*secret*`,
`authorization`, ...) are fully replaced.

It is applied to log records (message, args, extras), audit records, API responses,
SSE events, traces and tool output. Tests assert that secrets never reach logs or API
error messages.

## Audit log

Security-relevant events are recorded with actor, run id and redacted data:

- every tool decision (allow / deny / approval)
- approval decisions
- injection detections
- run start and finish
- tool executions

`FileAuditLog` chains each record to the previous one with SHA-256. `harness audit-verify`
reports the first tampered line. With several replicas, use the Redis Stream backend
(selected automatically with `HARNESS_STORAGE_BACKEND=redis`) and ship it to your SIEM.

## Production checklist

- `HARNESS_ENV=prod` (enforces API keys, Redis, Postgres, no unsafe sandbox)
- Secrets from a secret manager; never in images, YAML or Helm values
- gVisor/Kata RuntimeClass for sandbox Pods
- `HARNESS_ALLOW_SELF_APPROVAL=false` and `HARNESS_APPROVER_KEYS` for high-risk tools
- Review `policies/default.yaml` for every new tool; default `dangerous` to approval
- Restrict `/metrics` at the ingress; terminate TLS in front of the API
- Monitor `harness_errors_total`, `harness_approvals_total` and injection detections in the audit log
