import pytest
from helpers import (
    PRICES,
    TEST_ENV,
    chat_chunks,
    failing_after,
    fake_chat_clients,
    status_error,
)

from llmkit import LLM, Done, RetriesExhausted, TextDelta, TransientError


def make(outcomes=(), async_outcomes=(), **kw):
    return LLM(
        "glm-5.3",
        env=TEST_ENV,
        clients=fake_chat_clients(outcomes, async_outcomes),
        price_table=PRICES,
        sleep=lambda s: None,
        rng=lambda: 0.0,
        **kw,
    )


def test_stream_yields_deltas_then_done():
    events = list(make([chat_chunks(["Hel", "lo"])]).stream("hi"))
    assert events[:2] == [TextDelta("Hel"), TextDelta("lo")]
    assert isinstance(events[-1], Done) and len(events) == 3
    result = events[-1].result
    assert result.text == "Hello" and result.usage.input_tokens == 10
    assert result.usage.cost is not None


def test_stream_matches_complete_record():
    seen = []
    list(make([chat_chunks(["Hel", "lo"])], on_call=seen.append).stream("hi"))
    assert len(seen) == 1
    assert seen[0].text == "Hello" and seen[0].error is None and seen[0].attempts == 1
    assert seen[0].usage["cost"] is not None


def test_stream_retries_before_first_event():
    llm = make([status_error(503), chat_chunks(["ok"])])
    assert list(llm.stream("hi"))[-1].result.text == "ok"


def test_stream_failure_after_output_is_not_retried():
    seen = []
    broken = failing_after(chat_chunks(["partial"])[:1], status_error(503))
    llm = make([broken, chat_chunks(["never"])], on_call=seen.append)
    events = []
    with pytest.raises(TransientError):
        for event in llm.stream("hi"):
            events.append(event)
    assert events == [TextDelta("partial")]
    assert len(llm._clients.sync.chat.completions.calls) == 1
    assert len(seen) == 1 and seen[0].error_type == "TransientError"


def test_stream_exhausts_retries():
    seen = []
    llm = make([status_error(503)] * 2, max_retries=1, on_call=seen.append)
    with pytest.raises(RetriesExhausted):
        list(llm.stream("hi"))
    assert len(seen) == 1 and seen[0].attempts == 2


def test_stream_unclassified_failure_is_recorded_once():
    seen = []
    llm = make([KeyError("bug")], on_call=seen.append)
    with pytest.raises(KeyError):
        list(llm.stream("hi"))
    assert len(seen) == 1 and seen[0].error_type == "KeyError"


def test_abandoned_stream_records_nothing():
    seen = []
    llm = make([chat_chunks(["a", "b"])], on_call=seen.append)
    stream = llm.stream("hi")
    assert next(stream) == TextDelta("a")
    stream.close()
    assert seen == []


async def test_astream():
    llm = make(async_outcomes=[chat_chunks(["a", "b"])])
    events = [event async for event in llm.astream("hi")]
    assert [e.text for e in events[:-1]] == ["a", "b"]
    assert events[-1].result.text == "ab"


async def _async_failing_after(chunks, exc):
    for chunk in chunks:
        yield chunk
    raise exc


async def test_astream_failure_after_output_is_not_retried():
    seen = []
    broken = _async_failing_after(chat_chunks(["partial"])[:1], status_error(503))
    llm = make(async_outcomes=[broken, chat_chunks(["never"])], on_call=seen.append)
    events = []
    with pytest.raises(TransientError):
        async for event in llm.astream("hi"):
            events.append(event)
    assert events == [TextDelta("partial")]
    assert len(llm._clients.async_.chat.completions.calls) == 1
    assert len(seen) == 1 and seen[0].error_type == "TransientError"
