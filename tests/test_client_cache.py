import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from helpers import (
    PRICES,
    TEST_ENV,
    FakeCompletions,
    chat_response,
    fake_chat_clients,
    status_error,
)
from pydantic import BaseModel

from llmkit import (
    LLM,
    ContentFiltered,
    FatalRequest,
    Message,
    SchemaError,
    UnsupportedFeature,
)
from llmkit.cache import Cache
from llmkit.transports.base import ns

FILTER_BODY = {
    "code": "content_filter",
    "innererror": {"content_filter_result": {"hate": {"filtered": True}}},
}


class Verdict(BaseModel):
    answer: str


def make(cache, outcomes=(), async_outcomes=(), clients=None, **kw):
    kw.setdefault("price_table", PRICES)
    return LLM(
        "glm-5.3",
        env=TEST_ENV,
        cache=cache,
        clients=clients or fake_chat_clients(outcomes, async_outcomes),
        sleep=lambda s: None,
        rng=lambda: 0.0,
        **kw,
    )


def sent(llm):
    return llm._clients.sync.chat.completions.calls


def reasoning_response(text):
    return ns(
        {
            "model": "glm-5.3",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": text, "reasoning_content": "because"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )


@pytest.fixture
def cache(tmp_path):
    return Cache(tmp_path / "cache" / "calls.sqlite")


def test_second_identical_call_replays(cache):
    seen = []
    llm = make(
        cache, [chat_response("hello", prompt=1000, completion=100)], on_call=seen.append
    )
    first = llm.complete("hi")
    second = llm.complete("hi")
    assert len(sent(llm)) == 1
    assert (first.cache, second.cache) == ("miss", "hit")
    assert second.text == "hello" and second.usage.cost == 0.0
    assert second.usage.replayed_cost == pytest.approx(first.usage.cost)
    assert first.usage.replayed_cost is None
    assert [r.cache for r in seen] == ["miss", "hit"]
    assert (seen[0].cache_attempts, seen[1].attempts, seen[1].cache_attempts) == (1, 0, 1)


def test_samples_are_independent(cache):
    llm = make(cache, [chat_response("a"), chat_response("b")])
    assert llm.complete("hi", sample=0).text == "a"
    assert llm.complete("hi", sample=1).text == "b"
    assert llm.complete("hi", sample=1).text == "b"
    assert len(sent(llm)) == 2


def test_sample_needs_a_cache():
    llm = LLM("glm-5.3", env=TEST_ENV, clients=fake_chat_clients(), price_table=PRICES)
    with pytest.raises(ValueError, match="needs a cache"):
        llm.complete("hi", sample=1)
    with pytest.raises(ValueError, match="0 or more"):
        llm.complete("hi", sample=-1)


def test_truncation_is_retried_up_to_max_attempts(cache):
    llm = make(cache, [chat_response("cut", finish="length")] * 3)
    statuses = [llm.complete("hi").cache for _ in range(4)]
    assert statuses == ["miss", "retry", "retry", "hit"] and len(sent(llm)) == 3


def test_filtered_reply_is_kept_by_default(cache):
    llm = make(cache, [chat_response("", finish="content_filter")])
    assert llm.complete("hi").stop_reason == "content_filter"
    assert llm.complete("hi").cache == "hit" and len(sent(llm)) == 1


def test_blocked_prompt_is_stored_and_raised_again(cache):
    seen = []
    llm = make(cache, [status_error(400, body=FILTER_BODY)], on_call=seen.append)
    for _ in range(2):
        with pytest.raises(ContentFiltered):
            llm.complete("hi")
    assert len(sent(llm)) == 1 and [r.cache for r in seen] == ["miss", "hit"]


def test_errors_are_not_stored(cache):
    llm = make(cache, [status_error(400), chat_response("ok")])
    with pytest.raises(FatalRequest):
        llm.complete("hi")
    assert llm.complete("hi").text == "ok" and len(sent(llm)) == 2


def test_schema_mismatch_replays_as_schema_error(cache):
    llm = make(cache, [chat_response("not json")])
    for _ in range(2):
        with pytest.raises(SchemaError):
            llm.complete("hi", schema=Verdict)
    assert len(sent(llm)) == 1


def test_multi_turn_conversation_replays_turn_by_turn(tmp_path):
    path = tmp_path / "cache" / "calls.sqlite"

    def converse(llm):
        first = llm.complete("one")
        second = llm.complete([Message.user("one"), first.message, Message.user("two")])
        return first, second

    live = make(Cache(path), [reasoning_response("a"), reasoning_response("b")])
    converse(live)
    assert sent(live)[1]["messages"][1]["reasoning_content"] == "because"
    replay = make(Cache(path))
    first, second = converse(replay)
    assert (first.cache, second.cache) == ("hit", "hit") and second.text == "b"
    assert sent(replay) == []


class SlowCompletions(FakeCompletions):
    def create(self, **kwargs):
        time.sleep(0.05)
        return super().create(**kwargs)


def test_concurrent_identical_calls_send_once(cache):
    clients = fake_chat_clients()
    clients.sync.chat.completions = SlowCompletions([chat_response("once")])
    llm = make(cache, clients=clients)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: llm.complete("hi"), range(2)))
    assert [r.text for r in results] == ["once", "once"]
    assert sorted(r.cache for r in results) == ["hit", "miss"]


def test_a_call_that_loses_the_race_returns_the_stored_outcome(tmp_path):
    path = tmp_path / "cache" / "calls.sqlite"
    rival = make(Cache(path), [chat_response("rival")])

    class Racing(FakeCompletions):
        def create(self, **kwargs):
            rival.complete("hi")  # another process stores first
            return super().create(**kwargs)

    clients = fake_chat_clients()
    clients.sync.chat.completions = Racing([chat_response("mine")])
    seen = []
    llm = make(Cache(path), clients=clients, on_call=seen.append)
    result = llm.complete("hi")
    assert result.text == "rival" and result.cache == "hit"
    assert [r.cache for r in seen] == ["miss", "hit"] and seen[0].text == "mine"


def test_stream_refuses_a_cache(cache):
    with pytest.raises(UnsupportedFeature, match="cache"):
        next(iter(make(cache).stream("hi")))


async def test_acomplete_uses_the_cache(cache):
    llm = make(cache, async_outcomes=[chat_response("async")])
    assert (await llm.acomplete("hi")).cache == "miss"
    again = await llm.acomplete("hi")
    assert again.cache == "hit" and again.text == "async"
    assert len(llm._clients.async_.chat.completions.calls) == 1
