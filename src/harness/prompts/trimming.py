"""Context-window management by compaction.

Trimming individual old turns on every request would rewrite the conversation prefix each
time, defeating prompt caching and invalidating Claude's preserved-thinking blocks. Instead,
when the history outgrows its budget we compact once: keep the system prompt and the
original task, replace the middle with a short summary note, keep the most recent turns,
and strip provider-raw blocks (thinking) from what remains. The run then continues
append-only from the new base.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from harness.llm.types import Message

TAIL_SHARE = 0.6
_SUMMARY_CHARS = 160
_TRUNCATION_MARK = "\n...[truncated by context compaction]"


def estimate_tokens(messages: Sequence[Message]) -> int:
    # ponytail: ~4 chars/token heuristic; swap in the provider's count_tokens endpoint
    # if budgets need to be exact.
    chars = 0
    for m in messages:
        chars += len(m.content) + 8
        for c in m.tool_calls:
            chars += len(c.name) + len(json.dumps(c.arguments))
    return chars // 4 + 1


def _summarize(dropped: Sequence[Message]) -> str:
    lines: list[str] = []
    for m in dropped:
        if m.role == "tool":
            status = "error" if m.is_error else "ok"
            snippet = " ".join(m.content.split())[:_SUMMARY_CHARS]
            lines.append(f"- tool {m.name} ({status}): {snippet}")
        elif m.role == "assistant" and m.content.strip():
            lines.append(f"- assistant: {' '.join(m.content.split())[:_SUMMARY_CHARS]}")
    return "\n".join(lines[-40:]) or "- (no tool activity)"


def _clip(m: Message, max_tokens: int) -> Message:
    limit = max(200, max_tokens * 4)
    if len(m.content) <= limit:
        return m
    return m.model_copy(update={"content": m.content[:limit] + _TRUNCATION_MARK})


def compact(messages: Sequence[Message], max_tokens: int) -> list[Message] | None:
    """Return a compacted history, or None when the messages already fit."""
    if estimate_tokens(messages) <= max_tokens:
        return None

    head_end = 0
    while head_end < len(messages) and messages[head_end].role == "system":
        head_end += 1
    if head_end < len(messages) and messages[head_end].role == "user":
        head_end += 1
    head = [_clip(m, max_tokens // 4) for m in messages[:head_end]]
    rest = list(messages[head_end:])

    tail_budget = max(1, int(max_tokens * TAIL_SHARE) - estimate_tokens(head))
    start, used = len(rest), 0
    for i in range(len(rest) - 1, -1, -1):
        cost = estimate_tokens([rest[i]])
        if used + cost > tail_budget:
            break
        used += cost
        start = i
    # Never keep a tool result whose assistant tool call was dropped.
    while start < len(rest) and rest[start].role == "tool":
        start += 1

    dropped, tail = rest[:start], rest[start:]
    note = Message.user(
        f"[Context compacted: {len(dropped)} earlier messages were removed to fit the "
        f"context window. Summary of earlier activity:\n{_summarize(dropped)}\n"
        "Continue the task from here.]"
    )
    return [*head, note, *(_clip(m.without_raw(), tail_budget // 2) for m in tail)]
