from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.safety import InputValidationError, Policy, validate_user_input

REPO = Path(__file__).resolve().parents[3]


def _policy() -> Policy:
    return Policy.from_yaml(REPO / "policies" / "default.yaml")


def _eval(
    p: Policy, tool: str, perm: str | None, agent: str = "a", user: str = "u", **args: object
) -> tuple[str, str]:
    d = p.evaluate(tool=tool, permission=perm, agent=agent, user=user, arguments=dict(args))
    return d.verdict, d.reason


def test_defaults_by_permission() -> None:
    p = _policy()
    assert _eval(p, "file_read", "read")[0] == "allow"
    assert _eval(p, "x", "write")[0] == "allow"
    assert _eval(p, "python_exec", "dangerous", agent="other") == (
        "require_approval",
        "default for dangerous tools",
    )


def test_unknown_tool_is_denied() -> None:
    assert _eval(_policy(), "ghost", None)[0] == "deny"


def test_agent_rule_allows_sandboxed_code() -> None:
    verdict, reason = _eval(_policy(), "python_exec", "dangerous", agent="data_analyst", code="1")
    assert verdict == "allow" and "analysts-run-sandboxed-code" in reason


def test_deny_rules_win_over_later_allow() -> None:
    p = _policy()
    verdict, reason = _eval(p, "python_exec", "dangerous", agent="data_analyst", code="x" * 20_001)
    assert verdict == "deny" and "cap-code-size" in reason
    verdict, _ = _eval(
        p, "python_exec", "dangerous", agent="data_analyst", user="trial-42", code="1"
    )
    assert verdict == "deny"


def test_argument_regex_rule() -> None:
    p = _policy()
    assert _eval(p, "sql_query", "read", query="select * from Customers")[0] == "deny"
    assert _eval(p, "sql_query", "read", query="select * from sales")[0] == "allow"


def test_arg_conditions() -> None:
    p = Policy.model_validate(
        {
            "rules": [
                {
                    "name": "r",
                    "tools": ["t"],
                    "effect": "deny",
                    "args": {
                        "mode": {"in": ["a", "b"]},
                        "n": {"equals": 3},
                        "s": {"not_matches": "ok", "max_length": 5},
                    },
                }
            ]
        }
    )
    assert _eval(p, "t", "read", mode="a", n=3, s="bad")[0] == "deny"
    assert _eval(p, "t", "read", mode="c", n=3, s="bad")[0] == "allow"
    assert _eval(p, "t", "read", mode="a", n=4, s="bad")[0] == "allow"
    assert _eval(p, "t", "read", mode="a", n=3, s="okay")[0] == "allow"
    assert _eval(p, "t", "read", mode="a", n=3, s="toolong")[0] == "allow"
    assert _eval(p, "t", "read", mode="a", n=3)[0] == "allow"  # missing arg never matches


def test_policy_schema_is_strict() -> None:
    with pytest.raises(ValidationError):
        Policy.model_validate({"rules": [{"name": "x", "effect": "maybe"}]})
    with pytest.raises(ValidationError):
        Policy.model_validate({"rules": [{"name": "x", "effect": "deny", "typo": 1}]})
    with pytest.raises(ValidationError):
        Policy.model_validate(
            {"rules": [{"name": "x", "effect": "deny", "args": {"a": {"matches": "("}}}]}
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  hello  ", "hello"),
        ("a\x00b\x07c", "abc"),
        ("keep\nnewlines\tand tabs", "keep\nnewlines\tand tabs"),
        ("hidden\U000e0041\U000e0042text", "hiddentext"),
    ],
)
def test_input_is_cleaned(raw: str, expected: str) -> None:
    assert validate_user_input(raw) == expected


def test_input_rejections() -> None:
    with pytest.raises(InputValidationError, match="empty"):
        validate_user_input(" \x00 ")
    with pytest.raises(InputValidationError, match="exceeds"):
        validate_user_input("x" * 11, max_chars=10)
    with pytest.raises(InputValidationError, match="string"):
        validate_user_input(123)  # type: ignore[arg-type]
