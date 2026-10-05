"""Review tier: plumbing.

The one place that decides whether a failed attempt is retried. SDK retries are
disabled, so every provider follows the same policy: retry 408, 429, 5xx, dropped
connections and timeouts; wait what the provider asks, else back off exponentially
with full jitter. With retries attempted and all failed, the last error is wrapped in
``RetriesExhausted``; with no retries allowed, it is raised as is.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, NoReturn, TypeVar

import anthropic
import httpx
import openai
from google.genai import errors as genai_errors

from .errors import (
    ContentFiltered,
    FatalRequest,
    RequestError,
    RequestTimeout,
    RetriesExhausted,
    TransientError,
)

T = TypeVar("T")

MAX_BACKOFF = 60.0
BASE_BACKOFF = 1.0
_TRANSIENT_STATUSES = frozenset({408, 429})


def _retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Read ``retry-after-ms`` or ``retry-after`` (seconds) from response headers."""
    if not headers:
        return None
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            seconds = float(raw) / scale
        except ValueError:
            continue
        if math.isfinite(seconds):
            return max(0.0, seconds)
    return None


def _filtered_categories(body: Any) -> list[str] | None:
    """Return Azure content-filter categories when *body* is a filter rejection."""
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    if not isinstance(error, dict):
        return None
    inner = error.get("innererror") or {}
    if error.get("code") != "content_filter" and inner.get("code") != (
        "ResponsibleAIPolicyViolation"
    ):
        return None
    result = inner.get("content_filter_result") or {}
    return [k for k, v in result.items() if isinstance(v, dict) and v.get("filtered")]


def _from_status(
    message: str,
    status: int,
    *,
    provider: str,
    body: Any = None,
    request_id: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> RequestError:
    """Map an HTTP status to llmkit's hierarchy."""
    fields: dict[str, Any] = {
        "provider": provider,
        "status": status,
        "request_id": request_id,
        "body": body,
    }
    categories = _filtered_categories(body)
    if categories is not None:
        return ContentFiltered(message, categories=categories, **fields)
    if status in _TRANSIENT_STATUSES or status >= 500:
        return TransientError(message, retry_after=_retry_after(headers), **fields)
    return FatalRequest(message, **fields)


def classify(exc: BaseException, provider: str) -> RequestError:
    """Translate an SDK or transport exception into llmkit's hierarchy.

    Args:
        exc: What the SDK raised.
        provider: The provider that was called.

    Returns:
        The equivalent llmkit error; an llmkit error is returned unchanged.

    Raises:
        BaseException: *exc* itself when it is not a recognised request failure,
            so programming errors are never mistaken for provider failures.
    """
    if isinstance(exc, RequestError):
        return exc
    if isinstance(
        exc, openai.APITimeoutError | anthropic.APITimeoutError | httpx.TimeoutException
    ):
        return RequestTimeout(str(exc) or "request timed out", provider=provider)
    if isinstance(
        exc,
        openai.APIConnectionError | anthropic.APIConnectionError | httpx.TransportError,
    ):
        return TransientError(f"connection error: {exc}", provider=provider)
    if isinstance(exc, openai.APIStatusError | anthropic.APIStatusError):
        return _from_status(
            str(exc),
            exc.status_code,
            provider=provider,
            body=exc.body,
            request_id=exc.request_id,
            headers=exc.response.headers,
        )
    if isinstance(exc, genai_errors.APIError):
        return _from_status(
            str(exc),
            int(exc.code or 0),
            provider=provider,
            body=exc.details,
            headers=getattr(exc.response, "headers", None),
        )
    raise exc


def is_retryable(err: RequestError) -> bool:
    """Return whether another attempt might succeed."""
    return isinstance(err, TransientError | RequestTimeout)


def retry_delay(
    err: RequestError, attempt: int, max_retries: int, rng: Callable[[], float]
) -> float | None:
    """Return how long to wait before the next attempt, or ``None`` to stop.

    Args:
        err: The failure of attempt number *attempt* (1-based).
        attempt: The attempt that just failed.
        max_retries: Retries allowed after the first attempt.
        rng: Uniform [0, 1) source, for jitter.

    Returns:
        Seconds to wait, or ``None`` when the error is final.
    """
    if not is_retryable(err) or attempt > max_retries:
        return None
    if err.retry_after is not None:
        return min(err.retry_after, MAX_BACKOFF)
    return rng() * min(MAX_BACKOFF, BASE_BACKOFF * 2 ** (attempt - 1))


def final_error(err: RequestError, attempt: int) -> RequestError:
    """Return the error to raise once retrying stops.

    Args:
        err: The last failure.
        attempt: How many attempts were made.

    Returns:
        ``RetriesExhausted`` when retries were attempted on a retryable error,
        else *err*; either way with ``attempts`` set.
    """
    out: RequestError = (
        RetriesExhausted(err, attempt) if is_retryable(err) and attempt > 1 else err
    )
    out.attempts = attempt
    return out


def raise_from(err: RequestError, cause: BaseException) -> NoReturn:
    """Raise *err*, chained to *cause* unless they are the same object."""
    if err is cause:
        raise err
    raise err from cause


def _next_delay(
    exc: Exception,
    provider: str,
    attempt: int,
    max_retries: int,
    rng: Callable[[], float],
) -> float:
    """Return the wait before the next attempt, or raise the final error.

    Args:
        exc: What the failed attempt raised.
        provider: For error labelling.
        attempt: The attempt that just failed (1-based).
        max_retries: Retries allowed after the first attempt.
        rng: Jitter source.

    Returns:
        Seconds to wait before retrying.

    Raises:
        RequestError: The final failure, when no retry is due.
    """
    err = classify(exc, provider)
    delay = retry_delay(err, attempt, max_retries, rng)
    if delay is None:
        raise_from(final_error(err, attempt), exc)
    return delay


def call_with_retries(
    fn: Callable[[], T],
    *,
    provider: str,
    max_retries: int,
    sleep: Callable[[float], None],
    rng: Callable[[], float],
) -> tuple[T, int]:
    """Call *fn* until it succeeds or the failure is final.

    Args:
        fn: One attempt.
        provider: For error labelling.
        max_retries: Retries allowed after the first attempt.
        sleep: Blocking sleep.
        rng: Jitter source.

    Returns:
        The value and the number of attempts made.

    Raises:
        RequestError: The final failure.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn(), attempt
        except Exception as exc:
            sleep(_next_delay(exc, provider, attempt, max_retries, rng))


async def acall_with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    provider: str,
    max_retries: int,
    sleep: Callable[[float], Awaitable[None]],
    rng: Callable[[], float],
) -> tuple[T, int]:
    """Async counterpart of :func:`call_with_retries`.

    Args:
        fn: One attempt, returning an awaitable.
        provider: For error labelling.
        max_retries: Retries allowed after the first attempt.
        sleep: Async sleep.
        rng: Jitter source.

    Returns:
        The value and the number of attempts made.

    Raises:
        RequestError: The final failure.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn(), attempt
        except Exception as exc:
            await sleep(_next_delay(exc, provider, attempt, max_retries, rng))
