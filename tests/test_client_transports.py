"""Round trips through LLM.complete on the native transports (fakes, no network)."""

from types import SimpleNamespace

import pytest
from helpers import TEST_ENV, StaticClients

from llmkit import LLM, Message, Tool
from llmkit.transports.base import ns

pytestmark = pytest.mark.filterwarnings("ignore::llmkit.pricing.UnpricedModelWarning")

WEATHER = Tool("get_weather", "Weather", {"type": "object"})
ENV = {**TEST_ENV, "GOOGLE_CLOUD_LOCATION": "global"}


class Endpoint:
    """Records request kwargs and replays queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return ns(self.responses.pop(0))


def make(model, namespace, method, responses):
    endpoint = Endpoint(responses)
    sync = SimpleNamespace(**{namespace: SimpleNamespace(**{method: endpoint})})
    llm = LLM(
        model,
        env=ENV,
        clients=StaticClients(sync, sync),
        price_table={},
        sleep=lambda s: None,
    )
    return llm, endpoint


ANTHROPIC_TOOL_TURN = {
    "model": "claude-opus-5",
    "stop_reason": "tool_use",
    "content": [
        {"type": "thinking", "thinking": "plan", "signature": "sig"},
        {"type": "text", "text": "Checking."},
        {
            "type": "tool_use",
            "id": "t1",
            "name": "get_weather",
            "input": {"city": "Rome"},
        },
        {
            "type": "tool_use",
            "id": "t2",
            "name": "get_weather",
            "input": {"city": "Oslo"},
        },
    ],
    "usage": {"input_tokens": 5, "output_tokens": 5},
}
ANTHROPIC_FINAL = {
    "model": "claude-opus-5",
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "Both sunny."}],
    "usage": {"input_tokens": 8, "output_tokens": 3},
}


def test_anthropic_replays_provider_state_and_merges_parallel_tool_results():
    llm, endpoint = make(
        "claude-opus-5", "messages", "create", [ANTHROPIC_TOOL_TURN, ANTHROPIC_FINAL]
    )
    first = llm.complete("weather?", tools=[WEATHER])
    assert [tc.id for tc in first.tool_calls] == ["t1", "t2"]
    history = [
        Message.user("weather?"),
        first.message,
        Message.tool_result(first.tool_calls[0], "sunny"),
        Message.tool_result(first.tool_calls[1], "sunny too"),
    ]
    assert llm.complete(history, tools=[WEATHER]).text == "Both sunny."
    sent = endpoint.calls[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[1]["content"][0] == {
        "type": "thinking",
        "thinking": "plan",
        "signature": "sig",
    }
    results = sent[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert all(r["type"] == "tool_result" for r in results)


RESPONSES_TURN = {
    "model": "gpt-5.6-sol",
    "status": "completed",
    "output": [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "get_weather",
            "arguments": "{}",
        },
    ],
    "usage": {"input_tokens": 5, "output_tokens": 5},
}
RESPONSES_FINAL = {
    "model": "gpt-5.6-sol",
    "status": "completed",
    "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "Sunny."}]}
    ],
    "usage": {"input_tokens": 8, "output_tokens": 3},
}


def test_responses_replays_provider_state_and_tool_results():
    llm, endpoint = make(
        "gpt-5.6-sol", "responses", "create", [RESPONSES_TURN, RESPONSES_FINAL]
    )
    first = llm.complete("weather?", tools=[WEATHER])
    history = [
        Message.user("weather?"),
        first.message,
        Message.tool_result(first.tool_calls[0], "sunny"),
    ]
    assert llm.complete(history, tools=[WEATHER]).text == "Sunny."
    items = endpoint.calls[1]["input"]
    assert {"type": "reasoning", "id": "rs_1", "summary": []} in items
    assert {"type": "function_call_output", "call_id": "c1", "output": "sunny"} in items


GEMINI_TURN = {
    "model_version": "gemini-3.8-flash-001",
    "candidates": [
        {
            "finish_reason": "STOP",
            "content": {
                "role": "model",
                "parts": [
                    {"text": "Checking.", "thought_signature": b"sig"},
                    {"function_call": {"name": "get_weather", "args": {"city": "Rome"}}},
                ],
            },
        }
    ],
    "usage_metadata": {"prompt_token_count": 5, "candidates_token_count": 5},
}
GEMINI_FINAL = {
    "model_version": "gemini-3.8-flash-001",
    "candidates": [
        {
            "finish_reason": "STOP",
            "content": {"role": "model", "parts": [{"text": "Sunny."}]},
        }
    ],
    "usage_metadata": {"prompt_token_count": 8, "candidates_token_count": 3},
}


def test_gemini_replays_provider_state_and_tool_results():
    llm, endpoint = make(
        "gemini-3.8-flash", "models", "generate_content", [GEMINI_TURN, GEMINI_FINAL]
    )
    first = llm.complete("weather?", tools=[WEATHER])
    history = [
        Message.user("weather?"),
        first.message,
        Message.tool_result(first.tool_calls[0], "sunny"),
    ]
    assert llm.complete(history, tools=[WEATHER]).text == "Sunny."
    contents = endpoint.calls[1]["contents"]
    assert contents[1] == first.message.provider_state.data
    assert contents[1]["parts"][0]["thought_signature"] == b"sig"
    assert contents[2]["role"] == "user"
