"""Declarative tool-use policy (YAML).

    version: 1
    defaults:              # verdict per permission level when no rule matches
      read: allow
      write: allow
      dangerous: require_approval
    rules:                 # evaluated in order; first match wins
      - name: analysts-run-sandboxed-code
        tools: [python_exec]
        agents: [data_analyst]
        effect: allow
      - name: no-pii-tables
        tools: [sql_query]
        args:
          query: {matches: "(?i)\\b(users|customers)\\b"}
        effect: deny
        reason: PII tables are off limits

Selectors (`tools`, `agents`, `users`) accept shell-style globs. `permissions` matches
the tool's permission level. `args` conditions: equals, in, matches, not_matches,
max_length. A rule matches only when every selector and condition matches.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from harness.core.types import GuardDecision, GuardVerdict

Effect = Literal["allow", "deny", "require_approval"]


class ArgCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    equals: Any = None
    in_: list[Any] | None = Field(default=None, alias="in")
    matches: str | None = None
    not_matches: str | None = None
    max_length: int | None = None

    @field_validator("matches", "not_matches")
    @classmethod
    def _valid_regex(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                re.compile(v)
            except re.error as exc:
                raise ValueError(f"invalid regex {v!r}: {exc}") from exc
        return v

    def test(self, present: bool, value: Any) -> bool:
        if not present:
            return False
        text = value if isinstance(value, str) else str(value)
        if self.equals is not None and value != self.equals:
            return False
        if self.in_ is not None and value not in self.in_:
            return False
        if self.matches is not None and not re.search(self.matches, text):
            return False
        if self.not_matches is not None and re.search(self.not_matches, text):
            return False
        return not (self.max_length is not None and len(text) > self.max_length)


class PolicyRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    effect: Effect
    reason: str = ""
    tools: list[str] = Field(default_factory=lambda: ["*"])
    agents: list[str] = Field(default_factory=lambda: ["*"])
    users: list[str] = Field(default_factory=lambda: ["*"])
    permissions: list[Literal["read", "write", "dangerous"]] | None = None
    args: dict[str, ArgCondition] = Field(default_factory=dict)

    def matches(
        self, *, tool: str, permission: str, agent: str, user: str, arguments: dict[str, Any]
    ) -> bool:
        def glob(value: str, patterns: list[str]) -> bool:
            return any(fnmatch.fnmatchcase(value, p) for p in patterns)

        if not (glob(tool, self.tools) and glob(agent, self.agents) and glob(user, self.users)):
            return False
        if self.permissions is not None and permission not in self.permissions:
            return False
        return all(
            cond.test(name in arguments, arguments.get(name)) for name, cond in self.args.items()
        )


class PolicyDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    read: Effect = "allow"
    write: Effect = "allow"
    dangerous: Effect = "require_approval"
    # Tools the registry does not know are always denied.


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    defaults: PolicyDefaults = Field(default_factory=PolicyDefaults)
    rules: list[PolicyRule] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> Policy:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def evaluate(
        self,
        *,
        tool: str,
        permission: str | None,
        agent: str,
        user: str,
        arguments: dict[str, Any],
    ) -> GuardDecision:
        if permission is None:
            return GuardDecision(verdict="deny", reason=f"unknown tool {tool!r}")
        for rule in self.rules:
            if rule.matches(
                tool=tool, permission=permission, agent=agent, user=user, arguments=arguments
            ):
                reason = f"rule {rule.name!r}" + (f": {rule.reason}" if rule.reason else "")
                return GuardDecision(verdict=rule.effect, reason=reason)
        verdict: GuardVerdict = getattr(self.defaults, permission)
        return GuardDecision(verdict=verdict, reason=f"default for {permission} tools")
