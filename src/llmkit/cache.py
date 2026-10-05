"""Review tier: plumbing.

A persistent cache of call outcomes, keyed on everything that shapes the reply.

One SQLite file holds one row per key: the stored outcome (a reply, or a prompt the
content filter blocked), how many outcomes have been stored under the key, and the
outcome's original cost. Whether a call replays, retries or sends is decided here;
sending and recording stay in :mod:`llmkit.client`. Only a directory the cache creates
gets a ``.gitignore``, so a cache file placed in an existing folder never hides that
folder from git.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sqlite3
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import ContentFiltered
from .schemas import to_json_schema
from .transports.base import Call, Reply
from .types import (
    STOP_CONTENT_FILTER,
    STOP_END,
    STOP_MAX_TOKENS,
    STOP_REFUSAL,
    STOP_TOOL_USE,
    Message,
    ProviderState,
    Text,
    ToolCall,
)

FORMAT = 1
HIT = "hit"
MISS = "miss"
RETRY = "retry"
STOP_REASONS = frozenset(
    {STOP_END, STOP_MAX_TOKENS, STOP_TOOL_USE, STOP_REFUSAL, STOP_CONTENT_FILTER}
)
_COLUMNS = "key, model, outcome, stop_reason, cost, count"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    outcome TEXT NOT NULL,
    stop_reason TEXT NOT NULL,
    cost REAL,
    count INTEGER NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _default(value: Any) -> Any:
    """Encode bytes for JSON.

    Args:
        value: A value ``json`` cannot encode.

    Returns:
        A tagged base64 object.

    Raises:
        TypeError: If *value* is not bytes.
    """
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"cannot store {type(value).__name__}")


def _hook(obj: dict[str, Any]) -> Any:
    """Decode a tagged base64 object back to bytes.

    Args:
        obj: A decoded JSON object.

    Returns:
        The bytes, or *obj* unchanged.
    """
    if set(obj) == {"$bytes"}:
        return base64.b64decode(obj["$bytes"])
    return obj


def dumps(value: Any) -> str:
    """Return canonical JSON for *value*, bytes included.

    Args:
        value: Plain data.

    Returns:
        The JSON text, with sorted keys and no insignificant whitespace.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_default,
    )


def loads(text: str) -> Any:
    """Inverse of :func:`dumps`.

    Args:
        text: JSON written by :func:`dumps`.

    Returns:
        The data, bytes restored.
    """
    return json.loads(text, object_hook=_hook)


def _message(message: Message) -> dict[str, Any]:
    """Render a message for the key, hashing image bytes.

    Args:
        message: The message.

    Returns:
        Plain data.
    """
    content: Any = message.content
    if not isinstance(content, str):
        content = [
            {"text": p.text}
            if isinstance(p, Text)
            else {
                "media_type": p.media_type,
                "sha256": hashlib.sha256(p.data).hexdigest(),
            }
            for p in content
        ]
    state = message.provider_state
    return {
        "role": message.role,
        "content": content,
        "tool_calls": [asdict(tc) for tc in message.tool_calls],
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
        "is_error": message.is_error,
        "provider_state": None
        if state is None
        else {"transport": state.transport, "data": state.data},
    }


def request_key(call: Call, sample: int) -> str:
    """Return the cache key of one request.

    Args:
        call: The provider-neutral request.
        sample: The sample index.

    Returns:
        A hex SHA-256 over everything that shapes the reply.
    """
    route = call.route
    payload = {
        "format": FORMAT,
        "provider": route.provider,
        "deployment": route.deployment,
        "transport": route.transport,
        "system": call.system,
        "messages": [_message(m) for m in call.messages],
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "parameters": to_json_schema(t.parameters),
            }
            for t in call.tools
        ],
        "tool_choice": call.tool_choice,
        "schema": call.schema,
        "schema_name": call.schema_name if call.schema is not None else None,
        "effort": call.effort,
        "max_tokens": call.max_tokens,
        "temperature": call.temperature,
        "top_p": call.top_p,
        "sample": sample,
    }
    return hashlib.sha256(dumps(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Entry:
    """One stored outcome.

    Attributes:
        key: The request key.
        model: The requested model name.
        reply: The stored reply, or ``None`` for a blocked prompt.
        filtered: The stored filter error, or ``None`` for a reply.
        stop_reason: The reply's stop reason, or ``content_filter`` for a blocked prompt.
        cost: What the outcome originally cost, or ``None`` when unknown.
        count: How many outcomes have been stored under the key.
    """

    key: str
    model: str
    reply: Reply | None
    filtered: ContentFiltered | None
    stop_reason: str
    cost: float | None
    count: int


@dataclass(frozen=True)
class Lookup:
    """What the cache decided for one request.

    Attributes:
        key: The request key.
        status: ``hit`` (replay ``entry``), ``miss`` or ``retry`` (send).
        entry: The stored outcome, ``None`` on a miss.
    """

    key: str
    status: str
    entry: Entry | None


def _entry(row: tuple[Any, ...]) -> Entry:
    """Build an entry from a ``calls`` row.

    Args:
        row: The row's ``_COLUMNS``.

    Returns:
        The entry.
    """
    key, model, outcome, stop_reason, cost, count = row
    data = loads(outcome)
    reply = filtered = None
    if "reply" in data:
        fields = data["reply"]
        state = fields["provider_state"]
        reply = Reply(
            **{
                **fields,
                "tool_calls": [ToolCall(**tc) for tc in fields["tool_calls"]],
                "provider_state": ProviderState(**state) if state else None,
            }
        )
    else:
        stored = data["filtered"]
        filtered = ContentFiltered(
            stored["message"],
            categories=stored["categories"],
            provider=stored["provider"],
            status=stored["status"],
        )
    return Entry(key, model, reply, filtered, stop_reason, cost, count)


def _now() -> str:
    """Return the current UTC time as ISO-8601."""
    return datetime.now(UTC).isoformat(timespec="seconds")


class Cache:
    """Stored call outcomes in one SQLite file, shared safely across processes."""

    def __init__(
        self,
        path: str | Path,
        *,
        retry_on: Iterable[str] = (STOP_MAX_TOKENS,),
        max_attempts: int | None = 3,
    ) -> None:
        """Open or create the cache file.

        Args:
            path: The SQLite file. When its directory does not exist it is created
                with a ``.gitignore`` that ignores everything in it.
            retry_on: Stop reasons asked again on a later call, until ``max_attempts``
                outcomes are stored under the key.
            max_attempts: Most outcomes stored per key, or ``None`` for no cap.

        Raises:
            ValueError: If ``retry_on`` names an unknown stop reason,
                ``max_attempts`` is below 1, or the file holds another cache format.
        """
        retry = frozenset(retry_on)
        unknown = retry - STOP_REASONS
        if unknown:
            raise ValueError(f"retry_on has unknown stop reasons: {sorted(unknown)}")
        if max_attempts is not None and max_attempts < 1:
            raise ValueError("max_attempts must be at least 1, or None")
        self.path = Path(path)
        self.retry_on = retry
        self.max_attempts = max_attempts
        # Only a directory created here is ignored wholesale.
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True)
            (self.path.parent / ".gitignore").write_text("*\n", encoding="utf-8")
        self._db = sqlite3.connect(
            self.path, timeout=30.0, isolation_level=None, check_same_thread=False
        )
        self._db_lock = threading.Lock()
        with self._db_lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            self._db.execute(
                "INSERT OR IGNORE INTO meta VALUES ('format', ?)", (str(FORMAT),)
            )
            (found,) = self._db.execute(
                "SELECT value FROM meta WHERE name = 'format'"
            ).fetchone()
        if found != str(FORMAT):
            raise ValueError(
                f"{self.path} holds cache format {found}; this llmkit writes {FORMAT}. "
                "Use a new path."
            )
        self._guard = threading.Lock()
        self._locks: dict[str, list[Any]] = {}
        self._alocks: dict[str, list[Any]] = {}

    def _row(self, key: str) -> tuple[Any, ...] | None:
        """Read one row; the caller holds ``_db_lock``.

        Args:
            key: The request key.

        Returns:
            The row, or ``None``.
        """
        row: tuple[Any, ...] | None = self._db.execute(
            f"SELECT {_COLUMNS} FROM calls WHERE key = ?", (key,)
        ).fetchone()
        return row

    def lookup(self, key: str) -> Lookup:
        """Decide whether a request replays, retries or is sent for the first time.

        Args:
            key: The request key.

        Returns:
            The decision and any stored outcome.
        """
        with self._db_lock:
            row = self._row(key)
        if row is None:
            return Lookup(key, MISS, None)
        entry = _entry(row)
        capped = self.max_attempts is not None and entry.count >= self.max_attempts
        if entry.stop_reason in self.retry_on and not capped:
            return Lookup(key, RETRY, entry)
        return Lookup(key, HIT, entry)

    def store_reply(
        self, lookup: Lookup, model: str, reply: Reply, cost: float | None
    ) -> tuple[Entry, bool]:
        """Store a reply under the key *lookup* was made for.

        Args:
            lookup: The decision the request was sent under.
            model: The requested model name.
            reply: The reply.
            cost: What it cost, or ``None`` when unknown.

        Returns:
            The entry now stored, and whether it is this reply.
        """
        return self._store(
            lookup, model, {"reply": asdict(reply)}, reply.stop_reason, cost
        )

    def store_filtered(
        self, lookup: Lookup, model: str, err: ContentFiltered
    ) -> tuple[Entry, bool]:
        """Store a prompt the content filter blocked.

        Args:
            lookup: The decision the request was sent under.
            model: The requested model name.
            err: The filter error.

        Returns:
            The entry now stored, and whether it is this error.
        """
        outcome = {
            "filtered": {
                "message": str(err),
                "categories": err.categories,
                "provider": err.provider,
                "status": err.status,
            }
        }
        return self._store(lookup, model, outcome, STOP_CONTENT_FILTER, None)

    def _store(
        self,
        lookup: Lookup,
        model: str,
        outcome: dict[str, Any],
        stop_reason: str,
        cost: float | None,
    ) -> tuple[Entry, bool]:
        """Insert or replace an outcome unless another writer got there first.

        Args:
            lookup: The decision the request was sent under; a replacement only lands
                if the stored count is still the one *lookup* saw.
            model: The requested model name.
            outcome: The outcome as plain data.
            stop_reason: Its stop reason.
            cost: Its cost.

        Returns:
            The entry now stored, and whether it is *outcome*.
        """
        text, now = dumps(outcome), _now()
        with self._db_lock:
            if lookup.entry is None:
                cursor = self._db.execute(
                    "INSERT OR IGNORE INTO calls VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                    (lookup.key, model, text, stop_reason, cost, now, now),
                )
            else:
                cursor = self._db.execute(
                    "UPDATE calls SET outcome = ?, stop_reason = ?, cost = ?, "
                    "count = count + 1, updated = ? WHERE key = ? AND count = ?",
                    (text, stop_reason, cost, now, lookup.key, lookup.entry.count),
                )
            won = cursor.rowcount == 1
            row = self._row(lookup.key)
        assert row is not None
        return _entry(row), won

    def _slot(
        self, table: dict[str, list[Any]], key: str, factory: Callable[[], Any]
    ) -> list[Any]:
        """Take a reference on the lock for *key*, creating it on first use.

        Args:
            table: The lock table.
            key: The request key.
            factory: Builds a new lock.

        Returns:
            ``[lock, references]``.
        """
        with self._guard:
            slot = table.setdefault(key, [factory(), 0])
            slot[1] += 1
            return slot

    def _release(self, table: dict[str, list[Any]], key: str, slot: list[Any]) -> None:
        """Drop a reference, forgetting the lock when nobody holds or waits on it.

        Args:
            table: The lock table.
            key: The request key.
            slot: The slot from :meth:`_slot`.
        """
        with self._guard:
            slot[1] -= 1
            if slot[1] == 0:
                del table[key]

    @contextmanager
    def locked(self, key: str) -> Iterator[None]:
        """Serialise threads in this process that ask for the same key.

        Args:
            key: The request key.

        Yields:
            Nothing; the lock is held inside the block.
        """
        slot = self._slot(self._locks, key, threading.Lock)
        try:
            with slot[0]:
                yield
        finally:
            self._release(self._locks, key, slot)

    @asynccontextmanager
    async def alocked(self, key: str) -> AsyncIterator[None]:
        """Serialise tasks in this event loop that ask for the same key.

        Args:
            key: The request key.

        Yields:
            Nothing; the lock is held inside the block.
        """
        slot = self._slot(self._alocks, key, asyncio.Lock)
        try:
            async with slot[0]:
                yield
        finally:
            self._release(self._alocks, key, slot)

    def stats(self) -> dict[str, Any]:
        """Summarise what the cache holds.

        Returns:
            ``entries``, ``by_model``, ``by_stop_reason`` and ``stored_cost`` (the
            summed original cost of every stored outcome).
        """
        with self._db_lock:
            entries, cost = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(cost), 0) FROM calls"
            ).fetchone()
            by_model = dict(
                self._db.execute("SELECT model, COUNT(*) FROM calls GROUP BY model")
            )
            by_stop = dict(
                self._db.execute(
                    "SELECT stop_reason, COUNT(*) FROM calls GROUP BY stop_reason"
                )
            )
        return {
            "entries": entries,
            "by_model": by_model,
            "by_stop_reason": by_stop,
            "stored_cost": cost,
        }

    def close(self) -> None:
        """Close the database connection."""
        with self._db_lock:
            self._db.close()


__all__ = ["HIT", "MISS", "RETRY", "Cache", "Entry", "Lookup", "request_key"]
