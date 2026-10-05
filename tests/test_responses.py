import pytest

from llmkit.errors import FatalRequest, TransientError
from llmkit.registry import resolve
from llmkit.transports.base import Call, ns
from llmkit.transports.responses import ResponsesTransport
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

T = ResponsesTransport()
ROUTE = resolve("gpt-5.6-sol")
WEATHER = Tool(
    "get_weather",
    "Weather for a city",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


def call(**kw):
    kw.setdefault("messages", [Message.user("hi")])
    return Call(route=ROUTE, **kw)


def test_build_minimal_sends_no_reasoning_by_default():
    body = T.build(call(system="be brief", max_tokens=100))
    assert body["model"] == "gpt-5.6-sol"
    assert body["instructions"] == "be brief"
    assert body["max_output_tokens"] == 100
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]
    assert "reasoning" not in body and body["store"] is False


def test_build_effort_mapping():
    assert T.build(call(effort="off"))["reasoning"] == {"effort": "none"}
    assert T.build(call(effort="max"))["reasoning"] == {
        "effort": "xhigh",
        "summary": "auto",
    }


def test_build_tools_schema_and_images():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    body = T.build(
        call(
            messages=[Message.user([Text("look"), Image(b"\x00", "image/png")])],
            tools=[WEATHER],
            tool_choice="get_weather",
            schema=schema,
            schema_name="Out",
        )
    )
    assert body["tools"][0]["name"] == "get_weather"
    assert body["tools"][0]["parameters"]["additionalProperties"] is False
    assert body["tool_choice"] == {"type": "function", "name": "get_weather"}
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "Out",
        "schema": schema,
        "strict": True,
    }
    image = body["input"][0]["content"][1]
    assert image == {"type": "input_image", "image_url": "data:image/png;base64,AA=="}


def test_build_replays_state_and_renders_tool_results():
    state = ProviderState("responses", [{"type": "reasoning", "id": "rs_1"}])
    tc = ToolCall("call_1", "get_weather", {"city": "Paris"}, '{"city": "Paris"}')
    msgs = [
        Message.user("q"),
        Message("assistant", "", tool_calls=[tc], provider_state=state),
        Message.tool_result(tc, "sunny"),
    ]
    items = T.build(call(messages=msgs))["input"]
    assert items[1] == {"type": "reasoning", "id": "rs_1"}
    assert items[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "sunny",
    }
    plain = Message("assistant", "earlier", tool_calls=[tc])
    items = T.build(call(messages=[Message.user("q"), plain]))["input"]
    assert items[1] == {"role": "assistant", "content": "earlier"}
    assert items[2]["type"] == "function_call" and items[2]["call_id"] == "call_1"


RESPONSE = {
    "model": "gpt-5.6-sol-2026",
    "status": "completed",
    "output": [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "thought"}],
        },
        {"type": "message", "content": [{"type": "output_text", "text": "Hello"}]},
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "get_weather",
            "arguments": '{"city": "Rome"}',
        },
    ],
    "usage": {
        "input_tokens": 100,
        "output_tokens": 20,
        "input_tokens_details": {"cached_tokens": 60},
    },
}


def test_parse_subtracts_cached_tokens():
    reply = T.parse(call(), ns(RESPONSE))
    assert reply.text == "Hello" and reply.thinking == "thought"
    assert reply.tool_calls == [
        ToolCall("c1", "get_weather", {"city": "Rome"}, '{"city": "Rome"}')
    ]
    assert reply.stop_reason == "tool_use"
    assert (reply.input_tokens, reply.cached_input_tokens, reply.output_tokens) == (
        40,
        60,
        20,
    )
    assert reply.served_model == "gpt-5.6-sol-2026"
    assert reply.provider_state.transport == "responses"
    assert reply.provider_state.data[0]["id"] == "rs_1"


def test_parse_incomplete_and_refusal():
    raw = dict(
        RESPONSE,
        status="incomplete",
        output=RESPONSE["output"][:2],
        incomplete_details={"reason": "max_output_tokens"},
    )
    assert T.parse(call(), ns(raw)).stop_reason == "max_tokens"
    refusal = dict(
        RESPONSE,
        output=[
            {
                "type": "message",
                "content": [{"type": "refusal", "refusal": "I can't help"}],
            }
        ],
    )
    reply = T.parse(call(), ns(refusal))
    assert (reply.stop_reason, reply.text) == ("refusal", "I can't help")


def test_translator_streams_and_finishes_with_parse():
    tr = T.translator(call())
    events = []
    for chunk in [
        {"type": "response.reasoning_summary_text.delta", "delta": "th"},
        {"type": "response.output_text.delta", "delta": "Hel"},
        {
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {"type": "function_call", "call_id": "c1", "name": "get_weather"},
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 2,
            "delta": "{}",
        },
        {"type": "response.completed", "response": RESPONSE},
    ]:
        events.extend(tr.feed(ns(chunk)))
    assert events == [
        ThinkingDelta("th"),
        TextDelta("Hel"),
        ToolCallDelta(2, "c1", "get_weather", ""),
        ToolCallDelta(2, "", "", "{}"),
    ]
    assert tr.finish().text == "Hello"


def test_translator_failed_stream_raises():
    tr = T.translator(call())
    with pytest.raises(FatalRequest):
        tr.feed(ns({"type": "response.failed", "response": {"error": {"message": "x"}}}))


def test_translator_transient_failures_are_retryable():
    failed = {"type": "response.failed", "response": {"error": {"code": "server_error"}}}
    with pytest.raises(TransientError):
        T.translator(call()).feed(ns(failed))
    with pytest.raises(TransientError):
        T.translator(call()).feed(ns({"type": "error", "code": "rate_limit_exceeded"}))
    invalid = {
        "type": "response.failed",
        "response": {"error": {"code": "invalid_prompt"}},
    }
    with pytest.raises(FatalRequest):
        T.translator(call()).feed(ns(invalid))


def test_state_from_another_transport_is_not_replayed():
    state = ProviderState("anthropic", [{"type": "thinking", "thinking": "x"}])
    msgs = [Message.user("q"), Message("assistant", "earlier", provider_state=state)]
    items = T.build(call(messages=msgs))["input"]
    assert items[1] == {"role": "assistant", "content": "earlier"}


def test_failed_stream_message_excludes_the_echoed_response():
    failed = {
        "type": "response.failed",
        "response": {
            "instructions": "SECRET-SYSTEM-PROMPT",
            "output": [{"partial": "SECRET-OUTPUT"}],
            "error": {"code": "server_error", "message": "boom"},
        },
    }
    with pytest.raises(TransientError) as info:
        T.translator(call()).feed(ns(failed))
    text = str(info.value)
    assert "SECRET" not in text and "server_error" in text and "boom" in text
