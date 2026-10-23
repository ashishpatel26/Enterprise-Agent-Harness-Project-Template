"""Scripted, offline model trajectories.

They let `make demo`, the eval suite and CI exercise the full harness (prompting, policy,
sandbox, tools, memory, events) without an API key. The script stands in for the model
only; every other layer runs for real. Live runs use the model configured in YAML.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from harness.llm.fake import FakeLLM
from harness.llm.types import LLMResponse, Message

# Stdlib-only analysis so it runs in any sandbox backend (the Docker image also has
# pandas; a real model would typically use it).
ANALYSIS_CODE = """
import csv, json, statistics
from collections import defaultdict

with open("/data/sales.csv", newline="") as f:
    rows = list(csv.DictReader(f))

revenue = [float(r["revenue"]) for r in rows]
region, product, channel, month = (defaultdict(float) for _ in range(4))
discounted = [float(r["revenue"]) for r in rows if float(r["discount"]) > 0]
full_price = [float(r["revenue"]) for r in rows if float(r["discount"]) == 0]
for r in rows:
    v = float(r["revenue"])
    region[r["region"]] += v
    product[r["product"]] += v
    channel[r["channel"]] += v
    month[r["date"][:7]] += v

top = lambda d: sorted(d.items(), key=lambda kv: kv[1], reverse=True)
total = sum(revenue)
print(json.dumps({
    "rows": len(rows),
    "date_range": [min(r["date"] for r in rows), max(r["date"] for r in rows)],
    "total_revenue": round(total, 2),
    "avg_order": round(statistics.mean(revenue), 2),
    "regions": [[k, round(v, 2), round(100 * v / total, 1)] for k, v in top(region)],
    "products": [[k, round(v, 2), round(100 * v / total, 1)] for k, v in top(product)],
    "channels": [[k, round(v, 2), round(100 * v / total, 1)] for k, v in top(channel)],
    "best_month": top(month)[0],
    "worst_month": top(month)[-1],
    "avg_discounted_order": round(statistics.mean(discounted), 2),
    "avg_full_price_order": round(statistics.mean(full_price), 2),
}))
"""


def _summarize(messages: Sequence[Message]) -> LLMResponse:
    result = next((m for m in reversed(messages) if m.role == "tool"), None)
    if result is None or result.is_error:
        detail = result.content if result else "no tool result"
        return FakeLLM.text(f"I could not complete the analysis: the sandbox reported:\n{detail}")
    payload = result.content.split("--- stdout ---", 1)[-1].split("--- stderr ---", 1)[0]
    d = json.loads(payload.strip())
    r0, r_last = d["regions"][0], d["regions"][-1]
    p0 = d["products"][0]
    c0 = d["channels"][0]
    lines = [
        f"Analysis of {d['rows']} orders from {d['date_range'][0]} to {d['date_range'][1]} "
        f"(total revenue ${d['total_revenue']:,.2f}, average order ${d['avg_order']:,.2f}).",
        "",
        "Key insights:",
        f"1. {r0[0]} is the strongest region with ${r0[1]:,.2f} ({r0[2]}% of revenue); "
        f"{r_last[0]} is the weakest at {r_last[2]}%.",
        f"2. {p0[0]} is the top product by revenue (${p0[1]:,.2f}, {p0[2]}%).",
        f"3. {c0[0]} is the leading channel with {c0[2]}% of revenue.",
        f"4. Seasonality: the best month was {d['best_month'][0]} "
        f"(${d['best_month'][1]:,.2f}) and the weakest {d['worst_month'][0]} "
        f"(${d['worst_month'][1]:,.2f}).",
        f"5. Discounted orders average ${d['avg_discounted_order']:,.2f} vs "
        f"${d['avg_full_price_order']:,.2f} at full price.",
        "",
        "Caveats: one year of data; revenue is net of discount; no cost data, so these are "
        "revenue (not profit) insights.",
    ]
    return FakeLLM.text("\n".join(lines))


def data_analyst_script() -> FakeLLM:
    return FakeLLM(
        [
            FakeLLM.tool_call(
                "python_exec",
                {"code": ANALYSIS_CODE, "files": ["data/sales.csv"]},
                thought="I'll load the sales data and compute revenue breakdowns.",
            ),
            _summarize,
        ]
    )


def refusal_script() -> FakeLLM:
    """A model that asks for PII; policy must block the query."""
    return FakeLLM(
        [
            FakeLLM.tool_call(
                "sql_query", {"database": "data/shop.db", "query": "SELECT * FROM customers"}
            ),
            lambda msgs: FakeLLM.text(
                "I can't access that table: " + next(m for m in msgs if m.role == "tool").content
            ),
        ]
    )


SCRIPTS: dict[str, Callable[[], FakeLLM]] = {
    "data_analyst_demo": data_analyst_script,
    "pii_query_blocked": refusal_script,
}
