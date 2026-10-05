"""Test helpers shared across modules."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import openai

from llmkit.clients import StaticClients
from llmkit.transports.base import ns

TEST_ENV = {
    "AZURE_API_KEY": "k",
    "AZURE_ENDPOINT": "https://res.services.ai.azure.com",
    "GOOGLE_CLOUD_PROJECT": "p",
    "OPENROUTER_API_KEY": "o",
}

PRICES = {
    "azure_ai/FW-GLM-5.3": {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "cache_read_input_token_cost": 1e-7,
    },
}


def status_error(
    status: int, *, headers: dict[str, str] | None = None, body: Any = None
) -> openai.APIStatusError:
    """Build an OpenAI SDK status error as the SDK would raise it."""
    request = httpx.Request("POST", "https://example.test/v1/x")
    response = httpx.Response(status, request=request, headers=headers or {})
    return openai.APIStatusError(f"HTTP {status}", response=response, body=body)


class FakeCompletions:
    """Stands in for ``client.chat.completions``; replays queued outcomes."""

    def __init__(self, outcomes: Iterable[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def _next(self, kwargs: dict[str, Any]) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def create(self, **kwargs: Any) -> Any:
        outcome = self._next(kwargs)
        return iter(outcome) if isinstance(outcome, list) else outcome


class AsyncIter:
    """Wraps an iterable as an async iterator."""

    def __init__(self, items: Iterable[Any]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> AsyncIter:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None


class AsyncFakeCompletions(FakeCompletions):
    """Async counterpart of :class:`FakeCompletions`."""

    async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
        outcome = self._next(kwargs)
        return AsyncIter(outcome) if isinstance(outcome, list) else outcome


def fake_chat_clients(
    sync_outcomes: Iterable[Any] = (), async_outcomes: Iterable[Any] = ()
) -> StaticClients:
    """Build fake chat clients answering from queued outcomes."""
    sync = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions(sync_outcomes))
    )
    async_ = SimpleNamespace(
        chat=SimpleNamespace(completions=AsyncFakeCompletions(async_outcomes))
    )
    return StaticClients(sync, async_)


def chat_response(
    text: str = "hi",
    *,
    finish: str = "stop",
    prompt: int = 10,
    completion: int = 5,
    cached: int = 0,
) -> Any:
    """Build a chat-completions response object."""
    return ns(
        {
            "model": "glm-5.3",
            "choices": [{"finish_reason": finish, "message": {"content": text}}],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }
    )


def chat_chunks(texts: list[str]) -> list[Any]:
    """Build a chat-completions stream that says *texts* then reports usage."""
    chunks = [
        ns({"model": "glm-5.3", "choices": [{"delta": {"content": t}}]}) for t in texts
    ]
    chunks.append(ns({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    chunks.append(
        ns(
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": len(texts)},
            }
        )
    )
    return chunks


def failing_after(chunks: list[Any], exc: BaseException) -> Iterator[Any]:
    """Yield *chunks*, then raise *exc*."""
    yield from chunks
    raise exc
