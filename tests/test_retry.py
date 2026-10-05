import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors
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


def test_classify_genai_errors_with_retry_after():
    response = httpx.Response(429, headers={"retry-after": "7"})
    err = classify(
        genai_errors.ClientError(429, {"error": {"message": "slow"}}, response), "vertex"
    )
    assert isinstance(err, TransientError)
    assert (err.status, err.retry_after, err.provider) == (429, 7.0, "vertex")
    assert isinstance(
        classify(genai_errors.ServerError(503, {}, None), "vertex"), TransientError
    )
    fatal = classify(
        genai_errors.ClientError(400, {"error": {"message": "bad"}}, None), "vertex"
    )
    assert isinstance(fatal, FatalRequest) and fatal.retry_after is None


def test_classify_anthropic_errors():
    request = httpx.Request("POST", "https://x")
    response = httpx.Response(529, request=request, headers={"retry-after-ms": "500"})
    err = classify(
        anthropic.APIStatusError("overloaded", response=response, body=None), "foundry"
    )
    assert isinstance(err, TransientError) and err.retry_after == 0.5
    assert isinstance(
        classify(anthropic.APITimeoutError(request), "foundry"), RequestTimeout
    )
    assert isinstance(
        classify(anthropic.APIConnectionError(request=request), "foundry"), TransientError
    )
    assert isinstance(classify(httpx.ReadError("reset"), "foundry"), TransientError)


def test_retry_after_is_capped_and_clamped():
    assert (
        classify(status_error(429, headers={"retry-after": "-1"}), "foundry").retry_after
        == 0.0
    )
    assert (
        classify(
            status_error(429, headers={"retry-after": "soon"}), "foundry"
        ).retry_after
        is None
    )
    assert (
        classify(status_error(429, headers={"retry-after": "nan"}), "foundry").retry_after
        is None
    )
    assert retry_delay(TransientError("x", retry_after=600.0), 1, 3, lambda: 0.5) == 60.0


def _sse_status_error(body):
    request = httpx.Request("POST", "https://example.test/v1/messages")
    response = httpx.Response(200, request=request)
    return anthropic.APIStatusError("stream error", response=response, body=body)


def test_classify_anthropic_sse_error_events():
    overloaded = _sse_status_error(
        {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    )
    assert isinstance(classify(overloaded, "foundry"), TransientError)
    invalid = _sse_status_error(
        {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
    )
    assert isinstance(classify(invalid, "foundry"), FatalRequest)


def test_classify_chat_stream_error_chunk():
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    busy = openai.APIError("busy", request, body={"code": "rate_limit_exceeded"})
    assert isinstance(classify(busy, "openrouter"), TransientError)
    bad = openai.APIError("bad", request, body={"code": "invalid_prompt"})
    assert isinstance(classify(bad, "openrouter"), FatalRequest)
    assert isinstance(
        classify(openai.APIError("x", request, body=None), "openrouter"), FatalRequest
    )


def test_classify_builtin_timeout_and_google_auth_transport():
    from google.auth import exceptions as auth_exceptions

    assert isinstance(classify(TimeoutError("slow"), "vertex"), RequestTimeout)
    assert isinstance(
        classify(auth_exceptions.TransportError("net"), "vertex"), TransientError
    )


def test_classify_aiohttp_errors_when_installed(monkeypatch):
    import llmkit.retry as retry_module

    class FakeClientError(Exception):
        pass

    monkeypatch.setattr(retry_module, "_AIOHTTP_ERRORS", (FakeClientError,))
    assert isinstance(classify(FakeClientError("reset"), "vertex"), TransientError)
