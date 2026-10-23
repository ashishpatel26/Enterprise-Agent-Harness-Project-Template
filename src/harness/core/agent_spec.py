"""Declarative agent definitions (`agents/*.yaml`).

Users create agents without touching Python: model, prompt, tools, limits and policy
all come from YAML and are validated here before anything runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from harness.core.types import RunLimits
from harness.llm.factory import ModelConfig


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template: str = "react@v1"
    instructions: str = ""


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    top_k: int = Field(default=5, ge=0, le=50)
    include_session_history: bool = True


class AgnoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # None: use the harness model (retries, fallback, cost, budgets) bridged into Agno.
    # "<provider>:<model>" or "<module>:<Class>:<model>": use an Agno-native model.
    model: str | None = None
    approval_timeout_s: float = Field(default=600.0, gt=0, le=86_400)


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_\-]{0,63}$")
    description: str = ""
    version: str = "1"
    model: ModelConfig = Field(default_factory=ModelConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    tools: list[str] = Field(default_factory=list)
    sub_agents: list[str] = Field(default_factory=list)
    limits: RunLimits = Field(default_factory=RunLimits)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    policy: str | None = None
    # Reasoning loop: "native" (resumable harness loop) or "agno" (Agno under governance).
    engine: Literal["native", "agno"] = "native"
    agno: AgnoConfig = Field(default_factory=AgnoConfig)
    # MCP servers (from mcp_servers.yaml) whose tools this agent may use.
    mcp: list[str] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> AgentSpec:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        return cls.model_validate(data)


def load_agents(directory: Path) -> dict[str, AgentSpec]:
    """Load every `*.yaml`/`*.yml` in `directory`, keyed by agent name."""
    specs: dict[str, AgentSpec] = {}
    for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
        try:
            spec = AgentSpec.from_yaml(path)
        except Exception as exc:
            raise ValueError(f"invalid agent definition {path}: {exc}") from exc
        if spec.name in specs:
            raise ValueError(f"duplicate agent name {spec.name!r} in {path}")
        specs[spec.name] = spec
    return specs
