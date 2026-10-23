"""Eval task sets (YAML).

suite: data_analyst
agent: data_analyst
tasks:
  - id: sales-insights
    input: Analyze data/sales.csv and give me insights
    script: data_analyst_demo        # offline trajectory (see harness.demo.SCRIPTS)
    auto_approve: false
    expect:
      contains: [West, Gizmo]
      not_contains: [error]
      regex: "\\$[0-9,]+"
      exact: null
      tools_called: [python_exec]
      tools_not_called: [web_fetch]
      max_steps: 4
      stop_reason: completed
      rubric: The answer cites numbers for each insight and mentions caveats.
      rubric_threshold: 0.7
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Expectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exact: str | None = None
    contains: list[str] = Field(default_factory=list)
    not_contains: list[str] = Field(default_factory=list)
    regex: str | None = None
    tools_called: list[str] = Field(default_factory=list)
    tools_not_called: list[str] = Field(default_factory=list)
    max_steps: int | None = None
    stop_reason: str | None = "completed"
    rubric: str | None = None
    rubric_threshold: float = Field(default=0.7, ge=0, le=1)


class EvalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]{0,63}$")
    input: str
    agent: str | None = None
    script: str | None = None
    auto_approve: bool = False
    tags: list[str] = Field(default_factory=list)
    expect: Expectations = Field(default_factory=Expectations)


class EvalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite: str
    agent: str
    description: str = ""
    tasks: list[EvalTask]

    @classmethod
    def from_yaml(cls, path: Path) -> EvalSuite:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_suites(path: Path) -> list[EvalSuite]:
    files = [path] if path.is_file() else sorted(path.glob("*.y*ml"))
    suites = [EvalSuite.from_yaml(f) for f in files]
    names = [s.suite for s in suites]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate suite names in {path}")
    return suites
