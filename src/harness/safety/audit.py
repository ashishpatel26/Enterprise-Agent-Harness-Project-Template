"""Append-only, tamper-evident audit log.

Each record carries the SHA-256 of the previous record, so editing or deleting any line
breaks the chain and `verify()` reports where. Records are redacted before writing.

`FileAuditLog` suits a single writer process (ship the file to your SIEM). With several
API replicas use `RedisAuditLog` (one Redis Stream, append-only, ordered by Redis).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import redis.asyncio as aioredis

from harness.core.types import RunEvent
from harness.safety.redaction import Redactor

GENESIS = "0" * 64
# Run events worth a durable audit trail (text deltas and progress noise are skipped).
AUDITED_EVENTS = {
    "run_started",
    "approval_required",
    "approval_decided",
    "act",
    "observe",
    "compacted",
    "run_finished",
    "error",
}


def _canonical(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


def _digest(prev: str, record: dict[str, Any]) -> str:
    return hashlib.sha256((prev + _canonical(record)).encode()).hexdigest()


class AuditLog(Protocol):
    async def record(
        self, action: str, *, actor: str, run_id: str | None = None, **data: Any
    ) -> None: ...


class FileAuditLog:
    def __init__(self, path: Path, redactor: Redactor | None = None) -> None:
        self.path = path
        self.redactor = redactor or Redactor()
        self._lock = asyncio.Lock()
        self._last: str | None = None

    def _last_hash(self) -> str:
        if self._last is None:
            self._last = GENESIS
            if self.path.exists():
                with self.path.open("rb") as f:
                    lines = f.read().splitlines()
                if lines:
                    self._last = json.loads(lines[-1])["hash"]
        return self._last

    def _append(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
        try:
            os.write(fd, line.encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    async def record(
        self, action: str, *, actor: str, run_id: str | None = None, **data: Any
    ) -> None:
        async with self._lock:
            body = {
                "ts": datetime.now(UTC).isoformat(),
                "action": action,
                "actor": actor,
                "run_id": run_id,
                "data": self.redactor.redact_obj(data),
            }
            prev = await asyncio.to_thread(self._last_hash)
            entry = {**body, "prev": prev, "hash": _digest(prev, body)}
            await asyncio.to_thread(self._append, _canonical(entry) + "\n")
            self._last = entry["hash"]

    def verify(self) -> tuple[bool, int | None]:
        """Return (ok, first_bad_line_number)."""
        prev = GENESIS
        if not self.path.exists():
            return True, None
        with self.path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, start=1):
                entry = json.loads(line)
                body = {k: entry[k] for k in ("ts", "action", "actor", "run_id", "data")}
                if entry["prev"] != prev or entry["hash"] != _digest(prev, body):
                    return False, n
                prev = entry["hash"]
        return True, None


class RedisAuditLog:
    def __init__(
        self,
        redis: aioredis.Redis,
        *,
        stream: str = "harness:audit",
        redactor: Redactor | None = None,
    ) -> None:
        self.r = redis
        self.stream = stream
        self.redactor = redactor or Redactor()

    async def record(
        self, action: str, *, actor: str, run_id: str | None = None, **data: Any
    ) -> None:
        await self.r.xadd(
            self.stream,
            {
                "ts": datetime.now(UTC).isoformat(),
                "action": action,
                "actor": actor,
                "run_id": run_id or "",
                "data": _canonical(self.redactor.redact_obj(data)),
            },
        )


class AuditSink:
    """EventSink adapter: writes the security-relevant run events to the audit log."""

    def __init__(self, log: AuditLog) -> None:
        self.log = log

    async def __call__(self, event: RunEvent) -> None:
        if event.type in AUDITED_EVENTS:
            await self.log.record(
                f"run.{event.type}",
                actor="harness",
                run_id=event.run_id,
                step=event.step,
                event=event.data,
            )
