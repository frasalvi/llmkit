import pytest

from llmkit.errors import ContentFiltered
from llmkit.registry import resolve
from llmkit.transports.base import Call, ns
from llmkit.transports.gemini import GeminiTransport
from llmkit.types import (
    Message,
    ProviderState,
    TextDelta,
    ThinkingDelta,
    Tool,
    ToolCall,
    ToolCallDelta,
)

T = GeminiTransport()
WEATHER = Tool(
    "get_weather",
    "Weather",
    {"type": "object", "properties": {"city": {"type": "string"}}},
)


def call(**kw):
    kw.setdefault("messages", [Message.user("hi")])
    return Call(route=resolve("gemini-3.8-flash"), **kw)


def test_build_config():
    body = T.build(call(system="sys", effort="high", max_tokens=99, temperature=0.2))
    assert body["model"] == "gemini-3.8-flash"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    config = body["config"]
    assert config["system_instruction"] == "sys"
    assert config["thinking_config"] == {
        "thinking_level": "HIGH",
        "include_thoughts": True,
    }
    assert config["max_output_tokens"] == 99 and config["temperature"] == 0.2
    assert T.build(call(effort="off"))["config"]["thinking_config"] == {
        "thinking_budget": 0
    }
    assert "thinking_config" not in T.build(call())["config"]


def test_build_tools_and_schema():
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    config = T.build(call(tools=[WEATHER], tool_choice="get_weather", schema=schema))[
        "config"
    ]
    decl = config["tools"][0]["function_declarations"][0]
    assert decl["name"] == "get_weather" and "parameters_json_schema" in decl
    assert config["tool_config"] == {
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": ["get_weather"],
        }
    }
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == schema


def test_tool_round_trip_rendering():
    real = ToolCall("fc-1", "get_weather", {"city": "A"})
    made_up = ToolCall("llmkit-1", "get_weather", {"city": "B"})
    msgs = [
        Message.user("q"),
        Message("assistant", "", tool_calls=[real, made_up]),
        Message.tool_result(real, "sun"),
        Message.tool_result(made_up, "x", is_error=True),
    ]
    contents = T.build(call(messages=msgs))["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"][0] == {
        "function_call": {"name": "get_weather", "args": {"city": "A"}, "id": "fc-1"}
    }
    assert "id" not in contents[1]["parts"][1]["function_call"]
    assert contents[2]["parts"] == [
        {
            "function_response": {
                "name": "get_weather",
                "response": {"result": "sun"},
                "id": "fc-1",
            }
        },
        {"function_response": {"name": "get_weather", "response": {"error": "x"}}},
    ]


def test_state_replayed():
    state = ProviderState(
        "gemini", {"role": "model", "parts": [{"text": "hi", "thought_signature": b"s"}]}
    )
    msgs = [
        Message.user("q"),
        Message("assistant", "hi", provider_state=state),
        Message.user("again"),
    ]
    assert T.build(call(messages=msgs))["contents"][1] == state.data


RESPONSE = {
    "model_version": "gemini-3.8-flash-001",
    "candidates": [
        {
            "finish_reason": "STOP",
            "content": {
                "role": "model",
                "parts": [
                    {"text": "plan", "thought": True},
                    {"text": "Hello"},
                    {"function_call": {"name": "get_weather", "args": {"city": "Rome"}}},
                ],
            },
        }
    ],
    "usage_metadata": {
        "prompt_token_count": 100,
        "cached_content_token_count": 30,
        "candidates_token_count": 10,
        "thoughts_token_count": 5,
    },
}


def test_parse_subtracts_cached_tokens():
    reply = T.parse(call(), ns(RESPONSE))
    assert (reply.text, reply.thinking, reply.stop_reason) == (
        "Hello",
        "plan",
        "tool_use",
    )
    assert reply.tool_calls == [
        ToolCall("llmkit-2", "get_weather", {"city": "Rome"}, '{"city": "Rome"}')
    ]
    assert (reply.input_tokens, reply.cached_input_tokens, reply.output_tokens) == (
        70,
        30,
        15,
    )
    assert reply.provider_state.data["role"] == "model"


def test_parse_filters_and_blocks():
    filtered = dict(
        RESPONSE, candidates=[dict(RESPONSE["candidates"][0], finish_reason="SAFETY")]
    )
    assert T.parse(call(), ns(filtered)).stop_reason == "content_filter"
    truncated = dict(
        RESPONSE,
        candidates=[
            {
                "finish_reason": "MAX_TOKENS",
                "content": {"role": "model", "parts": [{"text": "Hel"}]},
            }
        ],
    )
    assert T.parse(call(), ns(truncated)).stop_reason == "max_tokens"
    with pytest.raises(ContentFiltered):
        T.parse(
            call(), ns({"candidates": [], "prompt_feedback": {"block_reason": "SAFETY"}})
        )


def test_translator():
    tr = T.translator(call())
    chunks = [
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "pl", "thought": True}]}}
            ]
        },
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "He"}]}}]},
        {
            "candidates": [
                {
                    "finish_reason": "STOP",
                    "content": {"role": "model", "parts": [{"text": "llo"}]},
                }
            ],
            "usage_metadata": RESPONSE["usage_metadata"],
            "model_version": "gemini-3.8-flash-001",
        },
    ]
    events = []
    for chunk in chunks:
        events.extend(tr.feed(ns(chunk)))
    assert events == [ThinkingDelta("pl"), TextDelta("He"), TextDelta("llo")]
    reply = tr.finish()
    assert (reply.text, reply.thinking, reply.output_tokens) == ("Hello", "pl", 15)
    assert reply.served_model == "gemini-3.8-flash-001"


def test_translator_reports_function_calls():
    tr = T.translator(call())
    events = tr.feed(
        ns(
            {
                "candidates": [
                    {
                        "finish_reason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "function_call": {
                                        "name": "get_weather",
                                        "args": {"city": "Rome"},
                                    }
                                }
                            ],
                        },
                    }
                ]
            }
        )
    )
    assert events == [ToolCallDelta(0, "", "get_weather", '{"city": "Rome"}')]
    assert tr.finish().tool_calls[0].id == "llmkit-0"


def test_null_tool_arguments_survive_parse_and_replay():
    raw = {
        "candidates": [
            {
                "finish_reason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [{"function_call": {"name": "f", "args": {"limit": None}}}],
                },
            }
        ]
    }
    reply = T.parse(call(), ns(raw))
    assert reply.tool_calls[0].arguments == {"limit": None}
    assert reply.provider_state.data["parts"][0]["function_call"]["args"] == {
        "limit": None
    }
