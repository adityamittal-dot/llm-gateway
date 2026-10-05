"""Research tenants (RESEARCH.md §8): each turns a public dataset into chat-completion requests.

Every tenant has its own system prompt, so a dropped system prompt is observable, and a
`shifted` variant that changes the tenant's own traffic (the confound the attribution test
must not blame on the provider).
"""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[3] / "data" / "datasets"


@dataclass
class Item:
    tenant: str
    item_id: str
    body: dict
    answer: str | None = None  # ground truth where the dataset has one (GSM8K)


def _body(system: str, user: str, max_tokens: int, **extra) -> dict:
    return {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        **extra,
    }


def math_items(n: int, shifted: bool = False) -> list[Item]:
    df = pd.read_parquet(DATA / "gsm8k_test.parquet").head(n)
    system = "Solve the problem step by step. End with a final line 'Answer: <number>'."
    if shifted:
        system = "Give only the final numeric answer, no working. Format: 'Answer: <number>'."
    return [
        Item(
            "math",
            f"gsm8k-{i}",
            _body(system, row.question, 320),
            answer=row.answer.split("####")[-1].strip(),
        )
        for i, row in df.iterrows()
    ]


def chat_items(n: int, shifted: bool = False) -> list[Item]:
    rows = [json.loads(line) for line in (DATA / "dolly.jsonl").read_text().splitlines()]
    rows = [
        r for r in rows if r["category"] in ("open_qa", "general_qa", "brainstorming", "creative_writing")
    ]
    random.Random(0).shuffle(rows)
    system = "You are a helpful assistant. Answer concisely in plain prose."
    if shifted:
        system = (
            "You are a helpful assistant. Answer in great detail as a numbered list of at least 6 points."
        )
    return [Item("chat", f"dolly-{i}", _body(system, r["instruction"], 256)) for i, r in enumerate(rows[:n])]


def code_items(n: int, shifted: bool = False) -> list[Item]:
    df = pd.read_parquet(DATA / "mbpp_test.parquet").head(n)
    system = "You write Python. Reply with a single code block and nothing else."
    if shifted:
        system = "You write Python. Explain your approach first, then give the code with comments."
    return [
        Item(
            "code", f"mbpp-{row.task_id}", _body(system, f"{row.text}\nIt must pass: {row.test_list[0]}", 320)
        )
        for _, row in df.iterrows()
    ]


_BFCL_TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def _json_schema(node):
    """BFCL uses Python type names ('dict', 'float', 'tuple'); convert to JSON Schema."""
    if isinstance(node, dict):
        out = {k: _json_schema(v) for k, v in node.items()}
        if isinstance(out.get("type"), str):
            out["type"] = _BFCL_TYPES.get(out["type"], out["type"])
        return out
    if isinstance(node, list):
        return [_json_schema(v) for v in node]
    return node


def tools_items(n: int, shifted: bool = False) -> list[Item]:
    rows = [json.loads(line) for line in (DATA / "bfcl_simple.json").read_text().splitlines()][:n]
    system = "You can call tools. When a tool fits the request, call it with correct arguments."
    if shifted:
        system = "Answer from your own knowledge. Only call a tool if you truly cannot answer."
    items = []
    for row in rows:
        tools = [
            {
                "type": "function",
                "function": {
                    **fn,
                    "name": re.sub(r"[^A-Za-z0-9_-]", "_", fn["name"]),
                    "parameters": _json_schema(fn["parameters"]),
                },
            }
            for fn in row["function"]
        ]
        items.append(
            Item("tools", row["id"], _body(system, row["question"][0][0]["content"], 160, tools=tools))
        )
    return items


TENANTS = {"math": math_items, "chat": chat_items, "code": code_items, "tools": tools_items}


def load(tenants: list[str], n: int, shifted: tuple[str, ...] = ()) -> list[Item]:
    return [item for t in tenants for item in TENANTS[t](n, shifted=t in shifted)]


def gsm8k_correct(text: str, answer: str) -> bool:
    """Check the last 'Answer: <number>' (or last number) against the GSM8K label."""
    match = re.findall(r"answer\s*[:=]\s*\$?(-?[\d,]*\.?\d+)", text, re.IGNORECASE) or re.findall(
        r"-?[\d,]*\.?\d+", text
    )
    if not match:
        return False
    try:
        return abs(float(match[-1].replace(",", "")) - float(answer.replace(",", ""))) < 1e-6
    except ValueError:
        return False
