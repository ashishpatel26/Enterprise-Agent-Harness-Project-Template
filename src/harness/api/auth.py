"""API-key authentication, caller identity and per-key rate limiting.

Tenancy model: an API key is a tenant. Runs are owned by the key that created them and
are invisible (404) to other keys. `X-User-Id` names the end user inside a tenant; it is
trusted because only an authenticated backend holding the key can send it.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status

from harness.runtime import Harness

_USER = re.compile(r"^[A-Za-z0-9_.@\-]{1,128}$")


@dataclass(frozen=True, slots=True)
class Principal:
    key_id: str
    user_id: str
    can_approve: bool


def get_harness(request: Request) -> Harness:
    harness: Harness = request.app.state.harness
    return harness


def _key_id(key: str) -> str:
    return "key_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def _matches(candidate: str, keys: list[str]) -> bool:
    # Compare against every key so timing does not reveal which one matched.
    found = False
    for k in keys:
        found |= hmac.compare_digest(candidate.encode(), k.encode())
    return found


async def principal(
    request: Request,
    harness: Harness = Depends(get_harness),  # noqa: B008
    x_api_key: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> Principal:
    s = harness.settings
    keys = [k.get_secret_value() for k in s.api_keys]
    if keys:
        if not x_api_key or not _matches(x_api_key, keys):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "missing or invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        key_id = _key_id(x_api_key)
    else:
        # Only reachable outside prod (settings forbid an empty key list in prod).
        key_id = "dev-anonymous"

    if x_user_id is not None and not _USER.match(x_user_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid X-User-Id")
    approvers = [k.get_secret_value() for k in s.approver_keys]
    can_approve = not approvers or (x_api_key is not None and _matches(x_api_key, approvers))

    rl = await harness.rate_limiter.hit(f"api:{key_id}", s.rate_limit_per_minute, 60.0)
    if not rl.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate limit exceeded",
            headers={"Retry-After": str(max(1, round(rl.retry_after_s)))},
        )
    request.state.key_id = key_id
    return Principal(key_id=key_id, user_id=x_user_id or key_id, can_approve=can_approve)
