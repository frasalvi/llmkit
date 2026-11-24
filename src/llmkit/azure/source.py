"""Review tier: plumbing.

The Azure facade: inventory and spend with a small on-disk cache, account keys held
in memory only.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
import time
import warnings
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from llmkit.azure.arm import Arm
from llmkit.azure.inventory import (
    Resource,
    Spend,
    account_key,
    discover,
    month_spend,
    resource_from_dict,
    resource_to_dict,
)
from llmkit.errors import ArmError, MissingCredential

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path.home() / ".cache" / "llmkit" / "azure.json"
INVENTORY_TTL = timedelta(hours=24)
SPEND_TTL = timedelta(hours=1)
CACHE_VERSION = 1
_REPLACE_ATTEMPTS = 5


class AzureInventoryWarning(UserWarning):
    """The inventory could not be refreshed, so a stale cached copy is in use."""


class AzureSource(Protocol):
    """What the rest of llmkit needs from Azure."""

    def resources(self, refresh: bool = False) -> list[Resource]:
        """Return the tagged resources."""
        ...

    def cached_resources(self) -> list[Resource] | None:
        """Return cached resources without touching the network."""
        ...

    def spend(self, r: Resource, refresh: bool = False) -> Spend:
        """Return month-to-date spend for the resource's subscription."""
        ...

    def key(self, r: Resource) -> str:
        """Return the resource's API key."""
        ...

    def invalidate(self) -> None:
        """Drop the cached inventory."""
        ...


def _utc_now() -> datetime:
    """Return the current time in UTC.

    Returns:
        A timezone-aware datetime.
    """
    return datetime.now(UTC)


def _parse_time(value: Any) -> datetime:
    """Parse a timezone-aware ISO timestamp.

    Args:
        value: The stored value.

    Returns:
        The parsed datetime.

    Raises:
        ValueError: If it is not a string, not ISO, or has no timezone.
    """
    if not isinstance(value, str):
        raise ValueError("timestamp is not a string")
    t = datetime.fromisoformat(value)
    if t.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return t


class Azure:
    """Azure inventory, spend and keys, with an on-disk cache of the first two."""

    def __init__(
        self,
        arm: Arm | None = None,
        *,
        cache_path: Path = DEFAULT_CACHE,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Create the facade; no ``az`` call is made until something is needed.

        Args:
            arm: ARM client; built lazily when omitted.
            cache_path: Where the inventory and spend cache lives.
            now: Clock, for tests.
        """
        self._arm = arm
        self._path = Path(cache_path)
        self._now = now
        self._lock = threading.Lock()
        self._keys: dict[tuple[str, str], str] = {}
        self._mem: dict[str, Any] = {}
        self._flights: dict[object, threading.Lock] = {}

    @property
    def arm(self) -> Arm:
        """The ARM client, built on first use."""
        return self._get_arm()

    def _get_arm(self) -> Arm:
        """Return the ARM client, building it on first use.

        Returns:
            The client.
        """
        with self._lock:
            if self._arm is None:
                self._arm = Arm()
            return self._arm

    def _load(self) -> dict[str, Any]:
        """Read the cache; the caller holds the lock.

        Returns:
            The cache sections (``inventory`` and ``spend``), empty where the file is
            missing, corrupt, truncated or of another version.
        """
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return dict(self._mem)
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
            return dict(self._mem)
        out: dict[str, Any] = {}
        inv: Any = data.get("inventory")
        with contextlib.suppress(KeyError, TypeError, ValueError):
            out["inventory"] = {
                "fetched": _parse_time(inv["fetched"]),
                "resources": [resource_from_dict(x) for x in inv["resources"]],
            }
        spend: dict[str, Spend] = {}
        raw = data.get("spend")
        for sub, s in raw.items() if isinstance(raw, dict) else ():
            try:
                spend[sub] = Spend(
                    float(s["amount"]), str(s["currency"]), _parse_time(s["fetched"])
                )
            except (KeyError, TypeError, ValueError):
                continue
        out["spend"] = spend
        return out

    def _store(self, state: dict[str, Any]) -> None:
        """Write the cache atomically; the caller holds the lock.

        A failed write is dropped: the cache is an optimization.

        Args:
            state: The cache sections to persist.
        """
        self._mem = dict(state)
        payload: dict[str, Any] = {"version": CACHE_VERSION}
        if "inventory" in state:
            inv = state["inventory"]
            payload["inventory"] = {
                "fetched": inv["fetched"].isoformat(),
                "resources": [resource_to_dict(r) for r in inv["resources"]],
            }
        payload["spend"] = {
            sub: {
                "amount": s.amount,
                "currency": s.currency,
                "fetched": s.fetched.isoformat(),
            }
            for sub, s in state.get("spend", {}).items()
        }
        tmp = ""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            for attempt in range(_REPLACE_ATTEMPTS):
                try:
                    os.replace(tmp, self._path)
                    tmp = ""
                    return
                except PermissionError:
                    if attempt == _REPLACE_ATTEMPTS - 1:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        except OSError as e:
            log.debug("could not write Azure cache %s: %s", self._path, e)
        finally:
            if tmp:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)

    def cached_resources(self) -> list[Resource] | None:
        """Return the cached inventory, however old, without any network call.

        Returns:
            The resources, or None when there is no usable cache.
        """
        with self._lock:
            inv = self._load().get("inventory")
        return list(inv["resources"]) if inv else None

    def resources(self, refresh: bool = False) -> list[Resource]:
        """Return the tagged resources, refreshing when older than 24 hours.

        Args:
            refresh: Ignore the cache and fetch.

        Returns:
            The inventory. If a refresh fails because Azure or the login is
            unavailable and a cached copy exists, that copy is returned with an
            :class:`AzureInventoryWarning`.

        Raises:
            ArmError: If the fetch fails and there is no cache.
            MissingCredential: If not logged in and there is no cache.
            AzureConfigError: If the tags are invalid (never masked by the cache).
        """
        now = self._now()
        inv = self._cached_inventory()
        if inv and not refresh and now - inv["fetched"] < INVENTORY_TTL:
            return list(inv["resources"])
        # one fetch at a time; threads that waited re-check the cache first
        with self._flight("inventory"):
            inv = self._cached_inventory()
            if inv and not refresh and now - inv["fetched"] < INVENTORY_TTL:
                return list(inv["resources"])
            try:
                found = discover(self._get_arm())
            except (ArmError, MissingCredential) as e:
                if inv is None:
                    raise
                age = now - inv["fetched"]
                warnings.warn(
                    f"Azure inventory refresh failed ({e}); using the cached copy "
                    f"from {age.total_seconds() / 3600:.0f} hours ago",
                    AzureInventoryWarning,
                    stacklevel=2,
                )
                return list(inv["resources"])
            with self._lock:
                state = self._load()
                state["inventory"] = {"fetched": now, "resources": found}
                self._store(state)
            return found

    def _cached_inventory(self) -> dict[str, Any] | None:
        """Read the cached inventory section.

        Returns:
            The section (``fetched`` and ``resources``), or None.
        """
        with self._lock:
            inv: dict[str, Any] | None = self._load().get("inventory")
        return inv

    def _flight(self, key: object) -> threading.Lock:
        """Return the lock that serializes fetches of one kind.

        Args:
            key: What is being fetched, e.g. ``("spend", subscription_id)``.

        Returns:
            A lock shared by every caller with the same key.
        """
        with self._lock:
            return self._flights.setdefault(key, threading.Lock())

    def spend(self, r: Resource, refresh: bool = False) -> Spend:
        """Return the subscription's month-to-date spend, cached for one hour.

        Args:
            r: A resource in the subscription.
            refresh: Ignore the cache and fetch.

        Returns:
            The spend.

        Raises:
            SpendUnavailable: If a fetch is needed and fails; an old cached figure is
                never used as a fallback.
        """
        now = self._now()
        cached = self._cached_spend(r.subscription_id)
        if cached and not refresh and now - cached.fetched < SPEND_TTL:
            return cached
        with self._flight(("spend", r.subscription_id)):
            cached = self._cached_spend(r.subscription_id)
            if cached and not refresh and now - cached.fetched < SPEND_TTL:
                return cached
            fresh = month_spend(
                self._get_arm(), r.subscription_id, r.subscription_name, now
            )
            with self._lock:
                state = self._load()
                state.setdefault("spend", {})[r.subscription_id] = fresh
                self._store(state)
            return fresh

    def _cached_spend(self, subscription_id: str) -> Spend | None:
        """Read one subscription's cached spend.

        Args:
            subscription_id: The subscription id.

        Returns:
            The cached figure, or None.
        """
        with self._lock:
            found: Spend | None = self._load().get("spend", {}).get(subscription_id)
        return found

    def key(self, r: Resource) -> str:
        """Return the resource's API key, fetched once per process.

        Args:
            r: The resource.

        Returns:
            The key. It is held in memory only and never written to disk.
        """
        ident = (r.subscription_id, r.account)
        with self._lock:
            cached = self._keys.get(ident)
        if cached is not None:
            return cached
        with self._flight(("key", ident)):
            with self._lock:
                cached = self._keys.get(ident)
            if cached is not None:
                return cached
            key = account_key(self._get_arm(), r)
            with self._lock:
                self._keys[ident] = key
            return key

    def invalidate(self) -> None:
        """Drop the cached inventory; spend and keys are kept."""
        with self._lock:
            state = self._load()
            state.pop("inventory", None)
            self._store(state)


_default: Azure | None = None
_default_lock = threading.Lock()


def default_azure() -> Azure:
    """Return the process-wide :class:`Azure`, so keys are shared across clients.

    Returns:
        The singleton.
    """
    global _default
    with _default_lock:
        if _default is None:
            _default = Azure()
        return _default
