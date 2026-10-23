"""`sql_query`: read-only SQL over SQLite databases in the workspace.

Read-only is enforced three ways, any one of which suffices: the database is opened with
`mode=ro`, `PRAGMA query_only` is on, and an authorizer denies every non-read action.
A statement-level check rejects anything but a single SELECT/WITH, and a progress handler
interrupts queries that run past the deadline.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

from harness.tools.builtin.files import resolve_in_workspace
from harness.tools.registry import Permission, Tool, ToolError, make_tool

DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


def _authorizer(action: int, *_: object) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def check_read_only(query: str) -> str:
    q = query.strip().rstrip(";").strip()
    if not q:
        raise ToolError("query is empty")
    if ";" in q:
        raise ToolError("only a single statement is allowed")
    if q.split(None, 1)[0].lower() not in ("select", "with"):
        raise ToolError("only SELECT queries are allowed")
    return q


def _render(columns: list[str], rows: list[tuple[object, ...]], truncated: bool) -> str:
    def cell(v: object) -> str:
        return "NULL" if v is None else str(v).replace("|", "\\|").replace("\n", " ")

    lines = [" | ".join(columns), " | ".join("---" for _ in columns)]
    lines += [" | ".join(cell(v) for v in row) for row in rows]
    note = f"\n[{len(rows)} rows shown; more rows exist, refine the query]" if truncated else ""
    return "\n".join(lines) + f"\n({len(rows)} rows){note}"


def sql_query_tool(root: Path, timeout_s: float = 10.0) -> Tool:
    async def sql_query(database: str, query: str, max_rows: int = 200) -> str:
        """Run one read-only SQL SELECT against a SQLite database in the workspace.

        `database` is a workspace-relative path to a .db/.sqlite file. Returns at most
        `max_rows` rows (1-1000) as a table. Writes, PRAGMAs and ATTACH are rejected.
        """
        if not 1 <= max_rows <= 1000:
            raise ToolError("max_rows must be between 1 and 1000")
        sql = check_read_only(query)
        path = resolve_in_workspace(root, database)
        if path.suffix.lower() not in DB_SUFFIXES or not path.is_file():
            raise ToolError(f"database not found: {database}")

        def run() -> str:
            deadline = time.monotonic() + timeout_s
            conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=2)
            try:
                conn.execute("PRAGMA query_only = ON")
                conn.set_authorizer(_authorizer)
                conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
                cur = conn.execute(sql)
                rows = cur.fetchmany(max_rows + 1)
                columns = [d[0] for d in cur.description or []]
            except sqlite3.DatabaseError as exc:
                if "interrupted" in str(exc):
                    raise ToolError(f"query exceeded {timeout_s}s") from exc
                raise ToolError(f"SQL error: {exc}") from exc
            finally:
                conn.close()
            return _render(columns, rows[:max_rows], len(rows) > max_rows)

        return await asyncio.to_thread(run)

    return make_tool(sql_query, permission=Permission.READ, timeout_s=timeout_s + 5)
