import pytest

from llmkit.errors import TransientError
from llmkit.registry import resolve
from llmkit.transports import TRANSPORTS
from llmkit.transports.base import Call, ns
from llmkit.transports.chat import ChatTransport
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

T = ChatTransport()
WEATHER = Tool(
    "get_weather",
    "Weather",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


def call(model="glm-5.3", provider=None, **kw):
    kw.setdefault("messages", [Message.user("hi")])
    return Call(route=resolve(model, provider), **kw)


def test_transport_table():
    assert set(TRANSPORTS) == {"responses", "anthropic", "gemini", "chat"}


def test_build_basic_and_effort():
    body = T.build(call(system="sys", effort="off", max_tokens=50))
    assert body["model"] == "FW-GLM-5.3"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert body["reasoning_effort"] == "none" and body["max_tokens"] == 50
    routed = T.build(call("z-ai/glm-5.2", "openrouter", effort="high"))
    assert routed["extra_body"] == {"reasoning": {"effort": "high"}}
    off = T.build(call("z-ai/glm-5.2", "openrouter", effort="off"))
    assert off["extra_body"] == {"reasoning": {"enabled": False}}


def test_build_images_tools_schema():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    body = T.build(
        call(
            messages=[Message.user([Text("see"), Image(b"\x00", "image/png")])],
            tools=[WEATHER],
            tool_choice="get_weather",
            schema=schema,
            schema_name="Out",
        )
    )
    assert body["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AA=="},
    }
    assert body["tools"][0]["function"]["name"] == "get_weather"
    assert body["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Out", "schema": schema, "strict": True},
    }


def test_build_tool_turns_and_reasoning_replay():
    tc = ToolCall("c1", "get_weather", {"city": "A"}, '{"city":"A"}')
    state = ProviderState("chat", {"reasoning_content": "because"})
    msgs = [
        Message.user("q"),
        Message("assistant", "", tool_calls=[tc], provider_state=state),
        Message.tool_result(tc, "sun"),
    ]
    rendered = T.build(call(messages=msgs))["messages"]
    assert rendered[1] == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "because",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"A"}'},
            }
        ],
    }
    assert rendered[2] == {"role": "tool", "tool_call_id": "c1", "content": "sun"}


RESPONSE = {
    "model": "glm-5.3",
    "choices": [
        {
            "finish_reason": "tool_calls",
            "message": {
                "content": "ok",
                "reasoning_content": "why",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Rome"}',
                        },
                    }
                ],
            },
        }
    ],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 9,
        "prompt_tokens_details": {"cached_tokens": 40},
    },
}


def test_parse():
    reply = T.parse(call(), ns(RESPONSE))
    assert (reply.text, reply.thinking, reply.stop_reason) == ("ok", "why", "tool_use")
    assert reply.tool_calls[0].arguments == {"city": "Rome"}
    assert (reply.input_tokens, reply.cached_input_tokens, reply.output_tokens) == (
        60,
        40,
        9,
    )
    assert reply.provider_state.data == {"reasoning_content": "why"}
    length = dict(
        RESPONSE, choices=[{"finish_reason": "length", "message": {"content": "x"}}]
    )
    assert T.parse(call(), ns(length)).stop_reason == "max_tokens"


def test_translator_accumulates_tool_call_fragments():
    tr = T.translator(call())
    chunks = [
        {"model": "glm-5.3", "choices": [{"delta": {"reasoning_content": "hm"}}]},
        {"choices": [{"delta": {"content": "Hi"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "get_weather", "arguments": '{"ci'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'ty": "Rome"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": RESPONSE["usage"]},
    ]
    events = []
    for chunk in chunks:
        events.extend(tr.feed(ns(chunk)))
    assert events == [
        ThinkingDelta("hm"),
        TextDelta("Hi"),
        ToolCallDelta(0, "c1", "get_weather", '{"ci'),
        ToolCallDelta(0, "", "", 'ty": "Rome"}'),
    ]
    reply = tr.finish()
    assert reply.tool_calls[0].arguments == {"city": "Rome"}
    assert reply.tool_calls[0].raw == '{"city": "Rome"}'
    assert (reply.text, reply.input_tokens) == ("Hi", 60)


def test_translator_truncated_stream_raises():
    tr = T.translator(call())
    tr.feed(ns({"choices": [{"delta": {"content": "Hi"}}]}))
    with pytest.raises(TransientError):
        tr.finish()


def test_missing_usage_is_flagged_not_reported_as_free():
    assert T.parse(call(), ns(RESPONSE)).usage_reported is True
    no_usage = {k: v for k, v in RESPONSE.items() if k != "usage"}
    assert T.parse(call(), ns(no_usage)).usage_reported is False
    tr = T.translator(call())
    tr.feed(ns({"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]}))
    assert tr.finish().usage_reported is False
