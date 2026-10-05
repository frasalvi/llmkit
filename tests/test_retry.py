import httpx
import openai
import pytest
from helpers import status_error

from llmkit.errors import (
    ContentFiltered,
    FatalRequest,
    RequestTimeout,
    RetriesExhausted,
    TransientError,
)
from llmkit.retry import acall_with_retries, call_with_retries, classify, retry_delay


def test_classify_status_codes():
    assert isinstance(classify(status_error(429), "foundry"), TransientError)
    assert isinstance(classify(status_error(503), "foundry"), TransientError)
    assert isinstance(classify(status_error(408), "foundry"), TransientError)
    fatal = classify(status_error(404), "foundry")
    assert isinstance(fatal, FatalRequest) and fatal.status == 404


def test_classify_retry_after_headers():
    assert (
        classify(
            status_error(429, headers={"retry-after-ms": "250"}), "foundry"
        ).retry_after
        == 0.25
    )
    assert (
        classify(status_error(429, headers={"retry-after": "3"}), "foundry").retry_after
        == 3.0
    )


def test_classify_azure_content_filter():
    body = {
        "code": "content_filter",
        "innererror": {
            "content_filter_result": {
                "hate": {"filtered": True},
                "violence": {"filtered": False},
            }
        },
    }
    err = classify(status_error(400, body=body), "foundry")
    assert isinstance(err, ContentFiltered) and err.categories == ["hate"]


def test_classify_timeouts_and_connections():
    request = httpx.Request("POST", "https://x")
    assert isinstance(
        classify(openai.APITimeoutError(request), "foundry"), RequestTimeout
    )
    assert isinstance(classify(httpx.ConnectError("down"), "vertex"), TransientError)


def test_classify_reraises_unknown_exceptions():
    with pytest.raises(KeyError):
        classify(KeyError("bug"), "foundry")


def test_retry_delay_honours_retry_after_and_limits():
    err = TransientError("x", retry_after=2.0)
    assert retry_delay(err, 1, 3, lambda: 0.5) == 2.0
    assert retry_delay(err, 4, 3, lambda: 0.5) is None
    assert retry_delay(FatalRequest("x"), 1, 3, lambda: 0.5) is None
    assert retry_delay(TransientError("x"), 3, 3, lambda: 1.0) == 4.0


def test_call_with_retries_succeeds_after_transient_failures():
    outcomes = [status_error(503), status_error(429), "ok"]
    slept = []

    def fn():
        value = outcomes.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    assert call_with_retries(
        fn, provider="foundry", max_retries=3, sleep=slept.append, rng=lambda: 0.0
    ) == ("ok", 3)
    assert slept == [0.0, 0.0]


def test_exhausted_retries_and_zero_retries():
    def always_503():
        raise status_error(503)

    with pytest.raises(RetriesExhausted) as info:
        call_with_retries(
            always_503,
            provider="foundry",
            max_retries=2,
            sleep=lambda s: None,
            rng=lambda: 0.0,
        )
    assert info.value.attempts == 3 and info.value.status == 503
    with pytest.raises(TransientError) as single:
        call_with_retries(
            always_503,
            provider="foundry",
            max_retries=0,
            sleep=lambda s: None,
            rng=lambda: 0.0,
        )
    assert single.value.attempts == 1


def test_fatal_errors_are_not_retried():
    calls = []

    def fn():
        calls.append(1)
        raise status_error(400)

    with pytest.raises(FatalRequest):
        call_with_retries(
            fn, provider="foundry", max_retries=3, sleep=lambda s: None, rng=lambda: 0.0
        )
    assert len(calls) == 1


async def test_async_retries():
    outcomes = [status_error(500), "ok"]

    async def fn():
        value = outcomes.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    async def no_sleep(seconds):
        return None

    assert await acall_with_retries(
        fn, provider="foundry", max_retries=1, sleep=no_sleep, rng=lambda: 0.0
    ) == ("ok", 2)
