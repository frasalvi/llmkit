"""Live checks against every route. Run: uv run pytest -m live -v

Costs a few dollars. A 404 means the model is not deployed on this account and the
case is skipped, not failed.
"""

import pytest
from pydantic import BaseModel

from llmkit import (
    LLM,
    Done,
    FatalRequest,
    Message,
    MissingCredential,
    TextDelta,
    Tool,
)

pytestmark = pytest.mark.live

OPENROUTER_MODEL = "deepseek/deepseek-v4.1-flash"

LIVE_MODELS = [
    ("gpt-5.6-luna", None),
    ("gpt-5.6-sol", None),
    ("claude-sonnet-5", "foundry"),
    ("claude-opus-5", "vertex"),
    ("claude-sonnet-5-5", None),
    ("claude-opus-5", None),
    ("claude-opus-5-5", None),
    ("claude-fable-5-1", None),
    ("gemini-3.8-flash", None),
    ("deepseek-v4-pro", None),
    ("glm-5.3", None),
    ("kimi-k3", None),
    (OPENROUTER_MODEL, "openrouter"),
]

WEATHER = Tool(
    "get_weather",
    "Current weather for a city.",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


class Verdict(BaseModel):
    answer: str
    confidence: float


def build(model, provider):
    try:
        return LLM(model, provider=provider, max_retries=2, timeout=180)
    except MissingCredential as exc:
        pytest.skip(str(exc))


def guarded(fn):
    try:
        return fn()
    except FatalRequest as exc:
        if exc.status == 404:
            pytest.skip(f"not deployed: {exc}")
        if "Organization Policy" in str(exc):
            pytest.skip(f"blocked by the cloud project's policy: {exc}")
        raise


@pytest.fixture(params=LIVE_MODELS, ids=lambda p: f"{p[0]}@{p[1] or 'default'}")
def llm(request):
    return build(*request.param)


def test_plain(llm):
    result = guarded(
        lambda: llm.complete("Reply with the single word: pong", max_tokens=2000)
    )
    assert "pong" in result.text.lower()
    assert result.usage.input_tokens > 0 and result.usage.output_tokens > 0
    if result.usage.cost is None:
        pytest.xfail(f"{llm.model} unpriced on {llm.provider}")


def test_every_served_rung(llm):
    for rung in llm.efforts:
        result = guarded(
            lambda rung=rung: llm.complete(
                "What is 17 * 23? Answer with the number only.",
                effort=rung,
                max_tokens=4000,
            )
        )
        assert "391" in result.text, rung


def test_tool_round_trip(llm):
    effort = "low" if "low" in llm.efforts else None
    first = guarded(
        lambda: llm.complete(
            "What is the weather in Paris? Use the get_weather tool.",
            tools=[WEATHER],
            tool_choice="auto",
            effort=effort,
            max_tokens=4000,
        )
    )
    assert first.tool_calls, first.text
    call = first.tool_calls[0]
    assert call.name == "get_weather" and "paris" in str(call.arguments).lower()
    history = [
        Message.user("What is the weather in Paris? Use the get_weather tool."),
        first.message,
        Message.tool_result(call, "Sunny, 24 C"),
    ]
    second = guarded(
        lambda: llm.complete(history, tools=[WEATHER], effort=effort, max_tokens=4000)
    )
    assert "sunny" in second.text.lower() or "24" in second.text


def test_structured_output(llm):
    if not llm.route.spec.structured_output:
        pytest.skip("no structured output on this model")
    result = guarded(
        lambda: llm.complete(
            "Is the sky blue on a clear day? Give a short answer and a confidence.",
            schema=Verdict,
            max_tokens=4000,
        )
    )
    assert isinstance(result.parsed, Verdict)


def test_stream(llm):
    events = guarded(lambda: list(llm.stream("Count from 1 to 5.", max_tokens=2000)))
    assert isinstance(events[-1], Done)
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == events[-1].result.text and "5" in text


async def test_async(llm):
    try:
        result = await llm.acomplete("Reply with the single word: pong", max_tokens=2000)
        events = [e async for e in llm.astream("Say hi.", max_tokens=2000)]
    except FatalRequest as exc:
        if exc.status == 404:
            pytest.skip(f"not deployed: {exc}")
        raise
    assert "pong" in result.text.lower()
    assert isinstance(events[-1], Done)


def test_prompt_cache(llm):
    if llm.route.transport == "chat":
        pytest.skip("caching is host-dependent on chat completions")
    # Gemini caches implicitly, and only above a size threshold.
    repeat = 6000 if llm.route.transport == "gemini" else 600
    system = "You are a terse assistant. " + (
        "Background fact: the sky is blue. " * repeat
    )
    for _ in range(2):
        result = guarded(
            lambda: llm.complete(
                "Say ok.", system=system, cache_prefix=True, max_tokens=2000
            )
        )
    assert result.usage.cached_input_tokens > 0
