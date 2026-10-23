# Security policy

agent-harness runs model-written code and acts on untrusted input, so we treat security
reports as the highest priority.

## Reporting a vulnerability

Please **do not open a public issue.** Report privately through GitHub's
**Security → Report a vulnerability** (private vulnerability reporting) on this
repository. Include:

- the affected component (sandbox, policy, approvals, API auth, redaction, ...)
- steps to reproduce or a proof of concept
- the impact you expect (escape, data access, privilege escalation, secret exposure)

We aim to acknowledge reports within 3 business days and to ship a fix or mitigation for
confirmed high-severity issues within 30 days. We will credit you in the release notes
unless you prefer otherwise.

## Scope

In scope: sandbox escapes, policy or approval bypasses, cross-tenant access through the
API, secret leakage in logs/events/responses, SSRF through built-in tools, audit-log
tampering that `audit-verify` misses, injection paths that bypass structural controls.

Out of scope: vulnerabilities that need the `local_unsafe` sandbox (it has no isolation by
design), denial of service through intentionally unlimited configuration, and issues in
third-party dependencies without a demonstrated impact on this project (report those
upstream).

## Supported versions

Security fixes go to the latest release. Projects created from this template should pull
in fixes from upstream regularly.

## Hardening guidance

See [docs/safety.md](docs/safety.md#production-checklist) for the production checklist.
