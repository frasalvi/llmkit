import warnings

import pytest
from helpers import PRICES, TEST_ENV, chat_response, fake_chat_clients, status_error
from pydantic import BaseModel

from llmkit import (
    LLM,
    FatalRequest,
    JsonlLog,
    Message,
    MissingCredential,
    RetriesExhausted,
    SchemaError,
    Tool,
    UnknownModel,
    UnpricedModelWarning,
    UnsupportedEffort,
    UnsupportedFeature,
)
from llmkit.records import read


class Verdict(BaseModel):
    answer: str


def make(outcomes=(), async_outcomes=(), **kw):
    kw.setdefault("price_table", PRICES)
    return LLM(
        "glm-5.3",
        env=TEST_ENV,
        clients=fake_chat_clients(outcomes, async_outcomes),
        sleep=lambda s: None,
        rng=lambda: 0.0,
        **kw,
    )


def test_complete_returns_priced_result():
    llm = make([chat_response("hello", prompt=1000, completion=100, cached=400)])
    result = llm.complete("hi", effort="low", max_tokens=200)
    assert result.text == "hello" and result.stop_reason == "end"
    usage = result.usage
    assert (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (
        600,
        400,
        100,
    )
    assert usage.cost == pytest.approx(600e-6 + 400e-7 + 200e-6)
    assert usage.provider == "foundry" and usage.effort == "low"
    sent = llm._clients.sync.chat.completions.calls[0]
    assert sent["reasoning_effort"] == "low" and sent["max_tokens"] == 200
    assert result.message.role == "assistant" and result.message.content == "hello"


@pytest.mark.filterwarnings("ignore::llmkit.pricing.UnpricedModelWarning")
def test_errors_before_sending():
    with pytest.raises(UnknownModel):
        LLM("opus", env=TEST_ENV)
    with pytest.raises(MissingCredential, match="AZURE_ENDPOINT"):
        LLM("glm-5.3", env={"AZURE_API_KEY": "k"}, price_table=PRICES)
    llm = make()
    with pytest.raises(UnsupportedEffort):
        llm.complete("hi", effort="max")
    claude = LLM(
        "claude-fable-5-1", env=TEST_ENV, clients=fake_chat_clients(), price_table={}
    )
    with pytest.raises(UnsupportedFeature, match="temperature"):
        claude.complete("hi", temperature=0.5)
    with pytest.raises(UnsupportedFeature, match="tool choice"):
        claude.complete(
            "hi", tools=[Tool("t", "d", {"type": "object"})], tool_choice="required"
        )
    assert llm._clients.sync.chat.completions.calls == []


def test_offline_price_table_warns_and_calls_proceed(monkeypatch):
    import llmkit.client as client_module

    loads = []

    def offline():
        loads.append(1)
        return None

    monkeypatch.setattr(client_module, "load_table", offline)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        llm = LLM(
            "glm-5.3",
            env=TEST_ENV,
            clients=fake_chat_clients([chat_response(), chat_response()]),
        )
        assert llm.complete("hi").usage.cost is None
        assert llm.complete("hi").usage.cost is None
    unpriced = [w for w in caught if w.category is UnpricedModelWarning]
    assert len(unpriced) == 1
    assert loads == [1]


def test_unreported_usage_leaves_cost_none():
    response = chat_response("hi")
    del response.usage
    llm = make([response], on_call=(seen := []).append)
    result = llm.complete("hi")
    assert result.usage.cost is None
    assert result.usage.input_tokens == 0 and result.usage.output_tokens == 0
    assert seen[0].usage["cost"] is None


def test_retries_then_success_and_records(tmp_path):
    path = tmp_path / "calls.jsonl"
    llm = make(
        [status_error(503), chat_response("ok")],
        on_call=JsonlLog(path),
        tags={"run": "r1"},
    )
    assert llm.complete("hi", tags={"row": 7}).text == "ok"
    rows = read(path)
    assert len(rows) == 1
    assert rows[0]["attempts"] == 2 and rows[0]["tags"] == {"run": "r1", "row": 7}


def test_failures_are_recorded_and_raised():
    seen = []
    llm = make([status_error(400)], on_call=seen.append)
    with pytest.raises(FatalRequest):
        llm.complete("hi")
    assert seen[0].error_type == "FatalRequest" and seen[0].usage is None
    llm = make([status_error(503)] * 4, on_call=seen.append)
    with pytest.raises(RetriesExhausted):
        llm.complete("hi")
    assert seen[-1].attempts == 4
    assert len(seen) == 2


def test_unclassified_failures_are_recorded_once():
    seen = []
    llm = make([KeyError("bug")], on_call=seen.append)
    with pytest.raises(KeyError):
        llm.complete("hi")
    assert len(seen) == 1 and seen[0].error_type == "KeyError"


def test_hook_errors_propagate():
    def broken(record):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        make([chat_response()], on_call=broken).complete("hi")


def test_schema_parsing_and_schema_errors():
    seen = []
    llm = make(
        [chat_response('{"answer": "yes"}'), chat_response("nope")], on_call=seen.append
    )
    assert llm.complete("q", schema=Verdict).parsed == Verdict(answer="yes")
    sent = llm._clients.sync.chat.completions.calls[0]["response_format"]["json_schema"]
    assert sent["name"] == "Verdict" and sent["schema"]["additionalProperties"] is False
    with pytest.raises(SchemaError):
        llm.complete("q", schema=Verdict)
    assert len(seen) == 2 and seen[1].error_type == "SchemaError"


def test_multi_turn_prompt():
    llm = make([chat_response("two")])
    history = [Message.user("one"), Message.assistant("ok"), Message.user("again")]
    llm.complete(history)
    sent = llm._clients.sync.chat.completions.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]


async def test_acomplete():
    llm = make(async_outcomes=[status_error(429), chat_response("async ok")])

    async def no_sleep(seconds):
        return None

    llm._asleep = no_sleep
    result = await llm.acomplete("hi")
    assert result.text == "async ok"


async def test_acomplete_failure_is_recorded():
    seen = []
    llm = make(async_outcomes=[status_error(400)], on_call=seen.append)
    with pytest.raises(FatalRequest):
        await llm.acomplete("hi")
    assert len(seen) == 1 and seen[0].error_type == "FatalRequest"
