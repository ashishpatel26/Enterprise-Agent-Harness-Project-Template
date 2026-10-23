from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest

from harness.config import get_settings
from harness.core.types import ToolContext
from harness.llm import ToolCall
from harness.sandbox.base import ExecResult, Sandbox, SandboxError, SandboxLimits
from harness.tools import ToolRegistry
from harness.tools.builtin import (
    file_read_tool,
    python_exec_tool,
    register_builtin_tools,
    sql_query_tool,
    web_fetch_tool,
)
from harness.tools.builtin.sql_query import check_read_only
from harness.tools.builtin.web_fetch import host_allowed, html_to_text
from harness.tools.registry import ToolError

CTX = ToolContext(run_id="r", session_id="s", user_id="u", agent="a", step=1)


async def run(tool_reg: ToolRegistry, name: str, **args: object) -> tuple[str, bool]:
    out = await tool_reg.execute(ToolCall(id="c", name=name, arguments=dict(args)), CTX)
    return out.content, out.is_error


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "notes.txt").write_text("\n".join(f"line {i}" for i in range(1, 11)))
    (tmp_path / "data" / "blob.bin").write_bytes(b"\x00\x01binary")
    db = sqlite3.connect(tmp_path / "data" / "shop.db")
    db.execute("CREATE TABLE sales (region TEXT, revenue REAL)")
    db.executemany("INSERT INTO sales VALUES (?, ?)", [("N", 10.0), ("S", 5.5), ("N", 2.5)])
    db.commit()
    db.close()
    (tmp_path.parent / "outside.txt").write_text("secret")
    return tmp_path


# ---- file_read ---------------------------------------------------------------


async def test_file_read_with_paging(workspace: Path) -> None:
    reg = ToolRegistry()
    reg.register(file_read_tool(workspace))
    text, err = await run(reg, "file_read", path="data/notes.txt", offset=2, limit=3)
    assert not err
    assert "     3  line 3" in text and "     5  line 5" in text
    assert "[5 more lines; use offset=5]" in text


@pytest.mark.parametrize(
    ("path", "fragment"),
    [
        ("../outside.txt", "escapes the workspace"),
        ("/etc/passwd", "relative"),
        ("data/missing.txt", "not found"),
        ("data/blob.bin", "binary"),
    ],
)
async def test_file_read_rejections(workspace: Path, path: str, fragment: str) -> None:
    reg = ToolRegistry()
    reg.register(file_read_tool(workspace))
    text, err = await run(reg, "file_read", path=path)
    assert err and fragment in text


async def test_file_read_bad_paging(workspace: Path) -> None:
    reg = ToolRegistry()
    reg.register(file_read_tool(workspace))
    text, err = await run(reg, "file_read", path="data/notes.txt", limit=0)
    assert err and "limit" in text


# ---- sql_query ----------------------------------------------------------------


async def test_sql_select(workspace: Path) -> None:
    reg = ToolRegistry()
    reg.register(sql_query_tool(workspace))
    text, err = await run(
        reg,
        "sql_query",
        database="data/shop.db",
        query="SELECT region, SUM(revenue) AS total FROM sales GROUP BY region ORDER BY region;",
    )
    assert not err
    assert "region | total" in text and "N | 12.5" in text and "(2 rows)" in text


async def test_sql_row_limit(workspace: Path) -> None:
    reg = ToolRegistry()
    reg.register(sql_query_tool(workspace))
    text, _ = await run(
        reg, "sql_query", database="data/shop.db", query="SELECT * FROM sales", max_rows=1
    )
    assert "more rows exist" in text


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM sales",
        "SELECT 1; DROP TABLE sales",
        "PRAGMA writable_schema=1",
        "ATTACH DATABASE 'x.db' AS x",
        "",
    ],
)
async def test_sql_writes_are_rejected(workspace: Path, query: str) -> None:
    reg = ToolRegistry()
    reg.register(sql_query_tool(workspace))
    _, err = await run(reg, "sql_query", database="data/shop.db", query=query)
    assert err
    db = sqlite3.connect(workspace / "data" / "shop.db")
    assert db.execute("SELECT COUNT(*) FROM sales").fetchone() == (3,)
    db.close()


async def test_sql_authorizer_blocks_write_disguised_as_with(workspace: Path) -> None:
    reg = ToolRegistry()
    reg.register(sql_query_tool(workspace))
    text, err = await run(
        reg,
        "sql_query",
        database="data/shop.db",
        query="WITH x AS (SELECT 1) INSERT INTO sales SELECT 'Z', 1 FROM x",
    )
    assert err and "SQL error" in text


async def test_sql_timeout(workspace: Path) -> None:
    reg = ToolRegistry()
    reg.register(sql_query_tool(workspace, timeout_s=0.2))
    text, err = await run(
        reg,
        "sql_query",
        database="data/shop.db",
        query="WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT MAX(x) FROM c",
    )
    assert err and "exceeded" in text


async def test_sql_missing_db(workspace: Path) -> None:
    reg = ToolRegistry()
    reg.register(sql_query_tool(workspace))
    text, err = await run(reg, "sql_query", database="data/notes.txt", query="SELECT 1")
    assert err and "not found" in text


def test_check_read_only_normalizes() -> None:
    assert check_read_only("  select 1 ;  ") == "select 1"
    with pytest.raises(ToolError):
        check_read_only("update t set a=1")


# ---- python_exec (with a fake sandbox) ------------------------------------------


class FakeSandbox(Sandbox):
    def __init__(self, result: ExecResult | Exception) -> None:
        self.result = result
        self.seen_files: Mapping[str, bytes] = {}

    async def run_python(
        self,
        code: str,
        *,
        files: Mapping[str, bytes] | None = None,
        limits: SandboxLimits | None = None,
    ) -> ExecResult:
        self.seen_files = files or {}
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def test_python_exec_success_and_file_attachment(workspace: Path) -> None:
    sbx = FakeSandbox(ExecResult(stdout="42\n", stderr="", exit_code=0))
    reg = ToolRegistry()
    reg.register(python_exec_tool(sbx, workspace))
    text, err = await run(reg, "python_exec", code="print(42)", files=["data/notes.txt"])
    assert not err
    assert "exit_code: 0" in text and "42" in text
    assert set(sbx.seen_files) == {"notes.txt"}
    tool = reg.get("python_exec")
    assert tool is not None and tool.permission == "dangerous"


@pytest.mark.parametrize(
    ("result", "fragment"),
    [
        (ExecResult(stdout="", stderr="Traceback: boom", exit_code=1), "boom"),
        (ExecResult(stdout="partial", stderr="", exit_code=None, timed_out=True), "timed out"),
        (ExecResult(stdout="", stderr="", exit_code=137, oom_killed=True), "memory limit"),
        (SandboxError("daemon down"), "sandbox unavailable"),
        (ValueError("bad name"), "bad name"),
    ],
)
async def test_python_exec_failures(
    workspace: Path, result: ExecResult | Exception, fragment: str
) -> None:
    reg = ToolRegistry()
    reg.register(python_exec_tool(FakeSandbox(result), workspace))
    text, err = await run(reg, "python_exec", code="x")
    assert err and fragment in text


async def test_python_exec_rejects_bad_attachments(workspace: Path) -> None:
    reg = ToolRegistry()
    reg.register(
        python_exec_tool(FakeSandbox(ExecResult(stdout="", stderr="", exit_code=0)), workspace)
    )
    text, err = await run(reg, "python_exec", code="x", files=["../outside.txt"])
    assert err and "escapes" in text
    text, err = await run(reg, "python_exec", code="x", files=["data/nope.csv"])
    assert err and "not found" in text
    text, err = await run(reg, "python_exec", code="x", files=["data/notes.txt"] * 11)
    assert err and "at most" in text
    text, err = await run(reg, "python_exec", code="x", files=["data/notes.txt", "data/notes.txt"])
    assert err and "share the name" in text


# ---- web_fetch ------------------------------------------------------------------


def _mock(
    handler_status: int = 200,
    body: str = "<html><script>x</script><p>Hello&amp;bye</p></html>",
    ctype: str = "text/html",
    headers: dict[str, str] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            handler_status, text=body, headers={"content-type": ctype, **(headers or {})}
        )

    return httpx.MockTransport(handler)


async def test_web_fetch_disabled_by_default() -> None:
    reg = ToolRegistry()
    reg.register(web_fetch_tool([]))
    text, err = await run(reg, "web_fetch", url="https://example.com")
    assert err and "disabled" in text


@pytest.mark.parametrize(
    ("url", "fragment"),
    [
        ("http://example.com", "https"),
        ("https://evil.com/x", "not allow-listed"),
        ("https://user:pw@example.com", "credentials"),
        ("file:///etc/passwd", "https"),
    ],
)
async def test_web_fetch_url_rules(url: str, fragment: str) -> None:
    reg = ToolRegistry()
    reg.register(web_fetch_tool(["example.com"], transport=_mock(), check_dns=False))
    text, err = await run(reg, "web_fetch", url=url)
    assert err and fragment in text


async def test_web_fetch_strips_html() -> None:
    reg = ToolRegistry()
    reg.register(web_fetch_tool(["example.com"], transport=_mock(), check_dns=False))
    text, err = await run(reg, "web_fetch", url="https://docs.example.com/page")
    assert not err and text == "Hello&bye"


async def test_web_fetch_does_not_follow_redirects_or_bad_types() -> None:
    reg = ToolRegistry()
    reg.register(
        web_fetch_tool(
            ["example.com"],
            transport=_mock(302, headers={"location": "http://169.254.169.254/"}),
            check_dns=False,
        )
    )
    text, err = await run(reg, "web_fetch", url="https://example.com")
    assert not err and "redirect to http://169.254.169.254/" in text

    reg2 = ToolRegistry()
    reg2.register(
        web_fetch_tool(["example.com"], transport=_mock(ctype="image/png"), check_dns=False)
    )
    text, err = await run(reg2, "web_fetch", url="https://example.com")
    assert err and "content type" in text

    reg3 = ToolRegistry()
    reg3.register(web_fetch_tool(["example.com"], transport=_mock(500), check_dns=False))
    text, err = await run(reg3, "web_fetch", url="https://example.com")
    assert err and "HTTP 500" in text


async def test_web_fetch_blocks_private_addresses() -> None:
    reg = ToolRegistry()
    reg.register(web_fetch_tool(["localhost"], transport=_mock()))
    text, err = await run(reg, "web_fetch", url="https://localhost/admin")
    assert err and "non-public" in text


def test_helpers() -> None:
    assert host_allowed("api.example.com", ["example.com"])
    assert not host_allowed("example.com.evil.net", ["example.com"])
    assert html_to_text("<style>a{}</style><b>x</b>  y") == "x y"


def test_register_builtin_tools() -> None:
    reg = register_builtin_tools(ToolRegistry(), get_settings(), None)
    assert reg.names() == ["file_read", "sql_query", "web_fetch"]
    reg = register_builtin_tools(
        ToolRegistry(), get_settings(), FakeSandbox(ExecResult(stdout="", stderr="", exit_code=0))
    )
    assert "python_exec" in reg.names()
