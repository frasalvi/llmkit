from llmkit.registry import resolve
from llmkit.transports.anthropic import AnthropicTransport
from llmkit.transports.base import Call, ns
from llmkit.types import (
    Image,
    Message,
    ProviderState,
    Text,
    TextDelta,
    ThinkingDelta,
    Tool,
    ToolCall,
    ToolCallDelta,
)

T = AnthropicTransport()
WEATHER = Tool(
    "get_weather",
    "Weather",
    {"type": "object", "properties": {"city": {"type": "string"}}},
)


def call(model="claude-opus-5", **kw):
    kw.setdefault("messages", [Message.user("hi")])
    return Call(route=resolve(model), **kw)


def test_build_effort_and_thinking():
    body = T.build(call(effort="high", system="sys", max_tokens=500))
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert body["system"] == "sys" and body["max_tokens"] == 500
    assert T.build(call(effort="off"))["thinking"] == {"type": "disabled"}
    assert T.build(call("claude-sonnet-5-5", effort="off"))["thinking"] == {
        "type": "between_tools"
    }
    assert "thinking" not in T.build(call())


def test_build_cache_tools_schema_and_images():
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    body = T.build(
        call(
            messages=[Message.user([Text("see"), Image(b"\x00", "image/png")])],
            cache_prefix=True,
            tools=[WEATHER],
            tool_choice="required",
            schema=schema,
            effort="low",
        )
    )
    assert body["extra_body"] == {"cache_control": {"type": "ephemeral"}}
    assert body["tools"][0]["input_schema"]["additionalProperties"] is False
    assert body["tool_choice"] == {"type": "any"}
    assert body["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": schema},
    }
    assert body["messages"][0]["content"][1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "AA==",
    }


def test_parallel_tool_results_merge():
    a = ToolCall("t1", "get_weather", {"city": "A"})
    b = ToolCall("t2", "get_weather", {"city": "B"})
    msgs = [
        Message.user("q"),
        Message("assistant", "", tool_calls=[a, b]),
        Message.tool_result(a, "sun"),
        Message.tool_result(b, "rain", is_error=True),
    ]
    rendered = T.build(call(messages=msgs))["messages"]
    assert [m["role"] for m in rendered] == ["user", "assistant", "user"]
    assert rendered[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "sun", "is_error": False},
        {"type": "tool_result", "tool_use_id": "t2", "content": "rain", "is_error": True},
    ]
    assert rendered[1]["content"][0] == {
        "type": "tool_use",
        "id": "t1",
        "name": "get_weather",
        "input": {"city": "A"},
    }


def test_state_replayed_verbatim():
    blocks = [
        {"type": "thinking", "thinking": "x", "signature": "sig"},
        {"type": "text", "text": "hi"},
    ]
    msgs = [
        Message.user("q"),
        Message("assistant", "hi", provider_state=ProviderState("anthropic", blocks)),
        Message.user("more"),
    ]
    assert T.build(call(messages=msgs))["messages"][1]["content"] == blocks


MESSAGE = {
    "model": "claude-opus-5",
    "stop_reason": "tool_use",
    "content": [
        {"type": "thinking", "thinking": "plan", "signature": "s"},
        {"type": "text", "text": "Checking."},
        {
            "type": "tool_use",
            "id": "t1",
            "name": "get_weather",
            "input": {"city": "Rome"},
        },
    ],
    "usage": {
        "input_tokens": 50,
        "output_tokens": 30,
        "cache_read_input_tokens": 400,
        "cache_creation_input_tokens": 100,
    },
}


def test_parse():
    reply = T.parse(call(), ns(MESSAGE))
    assert (reply.text, reply.thinking, reply.stop_reason) == (
        "Checking.",
        "plan",
        "tool_use",
    )
    assert reply.tool_calls[0].arguments == {"city": "Rome"}
    assert (reply.input_tokens, reply.cached_input_tokens, reply.cache_write_tokens) == (
        50,
        400,
        100,
    )
    assert reply.provider_state.data[2]["input"] == {"city": "Rome"}
    assert (
        T.parse(call(), ns(dict(MESSAGE, stop_reason="refusal"))).stop_reason == "refusal"
    )
    assert T.parse(call(), ns(dict(MESSAGE, stop_reason="max_tokens"))).stop_reason == (
        "max_tokens"
    )


def test_translator_rebuilds_blocks():
    tr = T.translator(call())
    chunks = [
        {
            "type": "message_start",
            "message": {
                "model": "claude-opus-5",
                "usage": {"input_tokens": 50, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "pl"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Hi"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "t1",
                "name": "get_weather",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"city": "Rome"}'},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 30},
        },
        {"type": "message_stop"},
    ]
    events = []
    for chunk in chunks:
        events.extend(tr.feed(ns(chunk)))
    assert events == [
        ThinkingDelta("pl"),
        TextDelta("Hi"),
        ToolCallDelta(2, "t1", "get_weather", ""),
        ToolCallDelta(2, "", "", '{"city": "Rome"}'),
    ]
    reply = tr.finish()
    assert (reply.text, reply.thinking, reply.output_tokens) == ("Hi", "pl", 30)
    assert reply.tool_calls[0].arguments == {"city": "Rome"}
    assert reply.provider_state.data[0] == {
        "type": "thinking",
        "thinking": "pl",
        "signature": "sig",
    }


def test_null_tool_arguments_survive_parse_and_replay():
    raw = dict(
        MESSAGE,
        content=[
            {
                "type": "tool_use",
                "id": "t1",
                "name": "f",
                "input": {"limit": None, "q": "x"},
            }
        ],
    )
    reply = T.parse(call(), ns(raw))
    assert reply.tool_calls[0].arguments == {"limit": None, "q": "x"}
    assert reply.provider_state.data[0]["input"] == {"limit": None, "q": "x"}
