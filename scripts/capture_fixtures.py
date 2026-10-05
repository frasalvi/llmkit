"""Review tier: plumbing.

Re-captures tests/fixtures from live calls: one plain response, one tool-call
response and one plain stream per transport. Run: uv run python scripts/capture_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmkit import LLM, Message, Tool
from llmkit.transports.base import Call

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
CASES = [
    ("gpt-5.6-luna", None),
    ("claude-sonnet-5", "foundry"),
    ("gemini-3.8-flash", None),
    ("deepseek-v4-pro", None),
]
WEATHER = Tool(
    "get_weather",
    "Current weather for a city.",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


def dump(value: Any) -> Any:
    """Return JSON-ready data for an SDK object."""
    return value.model_dump(mode="json", exclude_none=True)


def write(llm: LLM, kind: str, payload: dict[str, Any]) -> None:
    """Write one fixture file."""
    folder = ROOT / llm.route.transport
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{llm.model}__{llm.provider}__{kind}.json".replace("/", "_")
    record = {"model": llm.model, "provider": llm.provider, "kind": kind, **payload}
    (folder / name).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")


def capture(model: str, provider: str | None) -> None:
    """Capture the three fixture kinds for one model."""
    llm = LLM(model, provider=provider)
    transport, clients = llm._transport, llm._clients
    plain = Call(
        route=llm.route,
        messages=[Message.user("Say hello in five words.")],
        max_tokens=2000,
    )
    write(
        llm,
        "plain",
        {"response": dump(transport.send(clients.sync, transport.build(plain)))},
    )
    tool = Call(
        route=llm.route,
        messages=[Message.user("Weather in Paris? Use get_weather.")],
        tools=[WEATHER],
        tool_choice="auto",
        max_tokens=2000,
    )
    write(
        llm,
        "tool",
        {"response": dump(transport.send(clients.sync, transport.build(tool)))},
    )
    chunks = [
        dump(c) for c in transport.open_stream(clients.sync, transport.build(plain))
    ]
    write(llm, "stream", {"chunks": chunks})


def main() -> None:
    """Capture every case."""
    for model, provider in CASES:
        capture(model, provider)
        print(f"captured {model} on {provider or 'default'}")


if __name__ == "__main__":
    main()
