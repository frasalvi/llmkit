"""Review tier: plumbing.

List prices from LiteLLM's public table, cached on disk.

Costs are list prices, so they are a lower bound on an invoice. A model with no price
anywhere gets a loud warning once per process and ``cost=None`` on every call, never
zero: an unknown cost must not read as a free one.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .registry import Route
from .types import Usage

log = logging.getLogger("llmkit.pricing")

PRICE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
DEFAULT_CACHE = Path.home() / ".cache" / "llmkit" / "prices.json"
MAX_AGE_DAYS = 30

_PREFIXES: dict[tuple[str, str], tuple[str, ...]] = {
    ("foundry", "responses"): ("azure/",),
    ("foundry", "anthropic"): ("azure_ai/",),
    ("foundry", "chat"): ("azure_ai/",),
    ("vertex", "anthropic"): ("vertex_ai/",),
    ("vertex", "gemini"): ("vertex_ai/",),
    ("vertex", "chat"): ("vertex_ai/",),
    ("openrouter", "chat"): ("openrouter/",),
}

_WARNED: set[tuple[str, str]] = set()


class UnpricedModelWarning(UserWarning):
    """No list price could be resolved for a model."""


@dataclass(frozen=True)
class Rates:
    """USD per token for each kind of token.

    Attributes:
        input: Full-rate input.
        output: Output, thinking included.
        cached_input: Cache reads, or ``None`` when unknown.
        cache_write: Cache writes, or ``None`` when unknown.
    """

    input: float
    output: float
    cached_input: float | None
    cache_write: float | None


def fetch_table(timeout: float = 15.0) -> dict[str, Any]:
    """Download the LiteLLM price table.

    Args:
        timeout: Seconds before giving up.

    Returns:
        The table.

    Raises:
        httpx.HTTPError: On a network or HTTP failure.
        ValueError: If the payload is not a JSON object.
    """
    response = httpx.get(PRICE_URL, timeout=timeout)
    response.raise_for_status()
    table = response.json()
    if not isinstance(table, dict):
        raise ValueError("price table is not a JSON object")
    return table


def load_table(
    cache_path: Path = DEFAULT_CACHE,
    *,
    fetch: Callable[[], dict[str, Any]] = fetch_table,
    max_age_days: int = MAX_AGE_DAYS,
    now: Callable[[], float] = time.time,
) -> dict[str, Any] | None:
    """Return the price table, refreshing the disk cache when it is stale.

    Args:
        cache_path: Where the stamped table is cached.
        fetch: Downloads a fresh table.
        max_age_days: Refresh a cache older than this.
        now: Clock, for tests.

    Returns:
        The table; a stale cache when the fetch fails; ``None`` when there is
        neither.
    """
    cached: dict[str, Any] | None = None
    fresh = False
    try:
        stamped = json.loads(cache_path.read_text(encoding="utf-8"))
        cached = stamped["table"]
        fresh = now() - float(stamped["fetched_at"]) < max_age_days * 86400
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if cached is not None and fresh:
        return cached
    try:
        table = fetch()
    except Exception as exc:
        log.warning("could not fetch the LiteLLM price table: %s", exc)
        return cached
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"fetched_at": now(), "table": table}), encoding="utf-8"
        )
    except OSError as exc:
        log.warning("could not write the price cache %s: %s", cache_path, exc)
    return table


def price_keys(route: Route) -> tuple[str, ...]:
    """Candidate table keys for a route, most specific first.

    Args:
        route: The resolved model.

    Returns:
        Provider-prefixed deployment and model names, then the bare names.
    """
    names = list(dict.fromkeys([route.deployment, route.model]))
    prefixes = _PREFIXES.get((route.provider, route.transport), ())
    keys = [p + n for p in prefixes for n in names] + names
    return tuple(dict.fromkeys(keys))


def _rate(value: Any) -> float | None:
    """Parse an optional per-token rate."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def resolve_rates(route: Route, table: dict[str, Any]) -> Rates | None:
    """Find a route's rates: a registry override, else the first matching table key.

    Args:
        route: The resolved model.
        table: The LiteLLM table, possibly empty.

    Returns:
        The rates, or ``None`` when nothing resolves.
    """
    if route.spec.price is not None:
        usd_in, usd_out = route.spec.price
        return Rates(usd_in / 1e6, usd_out / 1e6, None, None)
    for key in price_keys(route):
        entry = table.get(key)
        if not isinstance(entry, dict):
            continue
        per_input = _rate(entry.get("input_cost_per_token"))
        per_output = _rate(entry.get("output_cost_per_token"))
        if per_input is None or per_output is None:
            continue
        return Rates(
            per_input,
            per_output,
            _rate(entry.get("cache_read_input_token_cost")),
            _rate(entry.get("cache_creation_input_token_cost")),
        )
    return None


def compute_cost(usage: Usage, rates: Rates | None) -> float | None:
    """Price a call at list rates.

    Args:
        usage: Token counts.
        rates: The model's rates.

    Returns:
        USD, or ``None`` when the rates, or a rate needed for these tokens, are
        unknown.
    """
    if rates is None:
        return None
    total = usage.input_tokens * rates.input + usage.output_tokens * rates.output
    if usage.cached_input_tokens:
        if rates.cached_input is None:
            return None
        total += usage.cached_input_tokens * rates.cached_input
    if usage.cache_write_tokens:
        if rates.cache_write is None:
            return None
        total += usage.cache_write_tokens * rates.cache_write
    return round(total, 10)


def warn_unpriced(route: Route) -> None:
    """Warn, once per model and provider per process, that costs will be unknown.

    Args:
        route: The unpriced model.
    """
    key = (route.model, route.provider)
    if key in _WARNED:
        return
    _WARNED.add(key)
    message = (
        f"no list price for {route.model} on {route.provider}; cost will be None. "
        f"Add a price override to the llmkit registry if LiteLLM does not list it."
    )
    log.warning(message)
    warnings.warn(message, UnpricedModelWarning, stacklevel=3)
