"""Parses every captured live response with the matching transport."""

import json
from pathlib import Path

import pytest

from llmkit.registry import resolve
from llmkit.transports import TRANSPORTS
from llmkit.transports.base import Call, ns
from llmkit.types import Message, Tool

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*/*.json"))
WEATHER = Tool(
    "get_weather",
    "Current weather for a city.",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


@pytest.mark.skipif(not FIXTURES, reason="no captured fixtures")
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_captured_fixture_parses(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    route = resolve(data["model"], data["provider"])
    transport = TRANSPORTS[route.transport]
    tools = [WEATHER] if data["kind"] == "tool" else []
    call = Call(route=route, messages=[Message.user("x")], tools=tools)
    if data["kind"] == "stream":
        translator = transport.translator(call)
        for chunk in data["chunks"]:
            translator.feed(ns(chunk))
        reply = translator.finish()
    else:
        reply = transport.parse(call, ns(data["response"]))
    assert reply.input_tokens + reply.cached_input_tokens > 0
    assert reply.output_tokens > 0
    if data["kind"] == "tool":
        assert reply.tool_calls and reply.tool_calls[0].name == "get_weather"
    else:
        assert reply.text
