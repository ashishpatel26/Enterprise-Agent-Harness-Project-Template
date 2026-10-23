"""Alternative agent engines that run under harness governance (`engine:` in agent YAML).

- `native` (default): the harness's own resumable loop (`harness.core.loop`).
- `agno`: Agno drives the reasoning loop; tools, policy, approvals, budgets and events
  stay in the harness (`harness.engines.agno_engine`).
"""
