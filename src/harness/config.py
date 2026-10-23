"""Application settings, loaded from environment variables and `.env`.

Every setting can be overridden with an environment variable prefixed `HARNESS_`
(e.g. `HARNESS_REDIS_URL`). Provider keys use their conventional names
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["dev", "test", "prod"]
StorageBackend = Literal["memory", "redis"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HARNESS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Environment = "dev"
    log_level: str = "INFO"
    service_name: str = "agent-harness"

    # --- API ---------------------------------------------------------------
    api_keys: Annotated[list[SecretStr], NoDecode] = Field(default_factory=list)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- LLM providers -----------------------------------------------------
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "HARNESS_ANTHROPIC_API_KEY"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "HARNESS_OPENAI_API_KEY")
    )
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    # --- Storage -----------------------------------------------------------
    storage_backend: StorageBackend = "memory"
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    database_url: SecretStr | None = None
    embedding_dim: int = Field(default=256, gt=0)

    # --- Sandbox -----------------------------------------------------------
    # "docker", "local_unsafe", or the name of a `harness.sandboxes` plugin.
    sandbox_backend: str = "docker"
    sandbox_image: str = "agent-harness-sandbox:latest"
    # Optional hardened OCI runtime for sandbox containers, e.g. "runsc" (gVisor).
    sandbox_runtime: str | None = None
    allow_unsafe_sandbox: bool = False
    workspace_root: Path = Path("examples")

    # --- Agents, policies, safety -------------------------------------------
    agents_dir: Path = Path("agents")
    policy_path: Path = Path("policies/default.yaml")
    audit_log_path: Path = Path("var/audit.log")
    rate_limit_per_minute: int = Field(default=60, ge=1)
    # web_fetch is disabled until domains are allow-listed (comma-separated).
    web_fetch_allowed_domains: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- Observability -----------------------------------------------------
    otel_endpoint: str | None = None
    otel_enabled: bool = False

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_keys(cls, v: object) -> object:
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v

    @field_validator("cors_origins", "web_fetch_allowed_domains", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @model_validator(mode="after")
    def _enforce_prod_invariants(self) -> Settings:
        if self.env != "prod":
            return self
        problems: list[str] = []
        if not self.api_keys:
            problems.append("HARNESS_API_KEYS must be set in prod")
        if self.storage_backend == "memory":
            problems.append("HARNESS_STORAGE_BACKEND=memory is not allowed in prod")
        if self.sandbox_backend == "local_unsafe" or self.allow_unsafe_sandbox:
            problems.append("the local_unsafe sandbox is not allowed in prod")
        if problems:
            raise ValueError("; ".join(problems))
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Tests call `get_settings.cache_clear()`.

    `HARNESS_DOTENV` picks the dotenv file; set it to an empty string to ignore dotenv.
    """
    env_file = os.environ.get("HARNESS_DOTENV", ".env") or None
    return Settings(_env_file=env_file)
