"""Test helpers shared across modules."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC
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


def make_resource(
    name: str,
    *,
    order: int | None,
    cap: float | None = 450.0,
    deployments: tuple[Any, ...] = (),
    subscription_id: str | None = None,
) -> Any:
    """Build an invented Azure ``Resource`` for tests.

    Args:
        name: The llmkit name; the subdomain is ``f"{name}-sub"``.
        order: Selection order, or None for off.
        cap: Monthly spend cap in USD.
        deployments: Deployments attached to the resource.
        subscription_id: Defaults to ``f"sub-{name}"``.

    Returns:
        A frozen ``Resource``.
    """
    from llmkit.azure.inventory import Resource

    return Resource(
        name=name,
        account=f"acct-{name}",
        subdomain=f"{name}-sub",
        subscription_id=subscription_id or f"sub-{name}",
        subscription_name=f"Subscription {name}",
        resource_group=f"rg-{name}",
        location="eastus",
        order=order,
        cap=cap,
        anthropic_org=None,
        deployments=deployments,
    )


class FakeAzure:
    """An ``AzureSource`` over fixed resources that records every call."""

    def __init__(
        self,
        resources: list[Any],
        spend: dict[str, float] | None = None,
        cached: list[Any] | None = None,
    ) -> None:
        """Create the fake.

        Args:
            resources: What ``resources()`` returns.
            spend: Month-to-date USD by llmkit name; 0 when absent.
            cached: What ``cached_resources()`` returns.
        """
        self._resources = resources
        self._spend = spend or {}
        self._cached = cached
        self.calls: list[str] = []

    def resources(self, refresh: bool = False) -> list[Any]:
        """Record and return the resources."""
        self.calls.append("resources")
        return self._resources

    def cached_resources(self) -> list[Any] | None:
        """Record and return the cached resources."""
        self.calls.append("cached")
        return self._cached

    def spend(self, r: Any, refresh: bool = False) -> Any:
        """Record and return the resource's spend."""
        from datetime import datetime

        from llmkit.azure.inventory import Spend

        self.calls.append(f"spend:{r.name}")
        return Spend(
            self._spend.get(r.name, 0.0),
            "USD",
            datetime(2026, 11, 24, 9, 0, tzinfo=UTC),
        )

    def key(self, r: Any) -> str:
        """Record and return a fake key."""
        self.calls.append(f"key:{r.name}")
        return f"key-{r.name}"

    def invalidate(self) -> None:
        """Record the call."""
        self.calls.append("invalidate")
