"""Review tier: plumbing.

``query``: run a function over many items, with per-item accounting, a spend cap and a
results file rebuilt from scratch on every run.

The runner keeps no done-state. A rerun calls the function on every item again, and
calls already stored in a :class:`~llmkit.Cache` replay for free, so the results file is
always the cache seen through the current code. It is written only when the run
finishes, in input order and through a temporary file, so an interrupted run leaves the
previous file untouched.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import os
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._version import __version__
from .ledger import LEDGER, Budget, Ledger


@dataclass(frozen=True)
class Failure:
    """One item that did not produce a result.

    Attributes:
        key: The item's key.
        error_type: The exception's class name.
        message: The exception's message.
    """

    key: str
    error_type: str
    message: str


@dataclass
class Report:
    """What one query run did.

    Attributes:
        ok: Items that produced a result.
        failed: Items that raised.
        calls: Calls made, replays included.
        new_calls: Requests actually sent.
        new_cost: USD spent by this run, or ``None`` when any sent call was unpriced.
        replayed_cost: Original USD cost of the outcomes replayed from the cache.
        failures: The failed items.
        drift: For each model name served by more than one version in this run, the
            versions and how many outcomes each served.
        budget_reached: Whether ``max_cost`` stopped a request.
        written: Whether the results and metadata files were written.
    """

    ok: int
    failed: int
    calls: int
    new_calls: int
    new_cost: float | None
    replayed_cost: float
    failures: list[Failure]
    drift: dict[str, dict[str, int]]
    budget_reached: bool
    written: bool

    def __str__(self) -> str:
        """Return the one-line summary."""
        new = "unpriced" if self.new_cost is None else f"${self.new_cost:.2f}"
        parts = [
            f"{self.ok} ok",
            f"{self.failed} failed",
            f"{new} new",
            f"${self.replayed_cost:.2f} replayed",
        ]
        if self.budget_reached:
            parts.append("budget reached")
        if self.drift:
            parts.append("model drift: " + ", ".join(sorted(self.drift)))
        return " · ".join(parts)


@dataclass
class _Outcome:
    """What one item produced.

    Attributes:
        key: The item's key.
        result: The function's return value, when it succeeded.
        error: The exception, when it failed.
        ledger: The item's calls.
    """

    key: str
    result: Any
    error: BaseException | None
    ledger: Ledger

    def line(self) -> dict[str, Any]:
        """Return the item's results-file line."""
        out: dict[str, Any] = {"key": self.key}
        if self.error is None:
            out["result"] = self.result
        else:
            out["error"] = {"type": type(self.error).__name__, "message": str(self.error)}
        out.update(self.ledger.line_fields())
        return out


def _plan(
    items: Iterable[Any], key: Callable[[Any], str], limit: int | None
) -> list[tuple[str, Any]]:
    """Pair every item with its key, refusing bad keys before any call.

    Args:
        items: The items.
        key: Maps an item to its key.
        limit: Keep only the first *limit* items, or ``None`` for all.

    Returns:
        ``(key, item)`` pairs in input order.

    Raises:
        ValueError: If *limit* is below 1 or two items share a key.
        TypeError: If a key is not a string.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    listed = list(items)
    if limit is not None:
        listed = listed[:limit]
    keyed: list[tuple[str, Any]] = []
    for item in listed:
        name = key(item)
        if not isinstance(name, str):
            raise TypeError(f"key must return a str, got {type(name).__name__}")
        keyed.append((name, item))
    counts = Counter(name for name, _ in keyed)
    duplicates = sorted(name for name, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate keys: {duplicates[:5]}")
    return keyed


def _checked(value: Any) -> Any:
    """Return *value* if it can be written as JSON.

    Args:
        value: The function's return value.

    Returns:
        *value*.

    Raises:
        TypeError: If it is not JSON-serializable.
    """
    json.dumps(value)
    return value


def _write_atomic(path: Path, text: str) -> None:
    """Replace *path* with *text* through a temporary file.

    Args:
        path: The destination.
        text: The content.
    """
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _git_state() -> tuple[str | None, bool | None]:
    """Return the working directory's commit and whether it has uncommitted changes.

    Returns:
        ``(commit, dirty)``, or ``(None, None)`` outside a git repository.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def _now() -> str:
    """Return the current UTC time as ISO-8601."""
    return datetime.now(UTC).isoformat(timespec="seconds")


class _Run:
    """One query run's shared state."""

    def __init__(
        self, keyed: list[tuple[str, Any]], max_cost: float | None, progress: bool
    ) -> None:
        """Prepare a run.

        Args:
            keyed: ``(key, item)`` pairs in input order.
            max_cost: The cap on new spend, or ``None``.
            progress: Whether to write a counter to stderr.
        """
        self.keyed = keyed
        self.max_cost = max_cost
        self.budget = Budget(max_cost)
        self.outcomes: list[_Outcome | None] = [None] * len(keyed)
        self.stop = threading.Event()
        self.started = _now()
        self._progress = progress
        self._done = 0
        self._lock = threading.Lock()

    def _settle(self, index: int, outcome: _Outcome) -> None:
        """Keep an item's outcome and advance the counter.

        Args:
            index: The item's position.
            outcome: What it produced.
        """
        self.outcomes[index] = outcome
        with self._lock:
            self._done += 1
            if self._progress:
                sys.stderr.write(f"\rquery: {self._done}/{len(self.keyed)}")
                sys.stderr.flush()

    def run_sync(self, index: int, fn: Callable[[Any], Any]) -> None:
        """Run one item on the current thread, under its own ledger.

        Args:
            index: The item's position.
            fn: The function.
        """
        if self.stop.is_set():
            return
        key, item = self.keyed[index]
        ledger = Ledger(self.budget)
        token = LEDGER.set(ledger)
        try:
            outcome = _Outcome(key, _checked(fn(item)), None, ledger)
        except Exception as err:
            outcome = _Outcome(key, None, err, ledger)
        except BaseException:
            # Stop this worker's next item before the main thread sees the interrupt.
            self.stop.set()
            raise
        finally:
            LEDGER.reset(token)
        self._settle(index, outcome)

    async def run_async(
        self, index: int, fn: Callable[[Any], Any], gate: asyncio.Semaphore
    ) -> None:
        """Run one item as a task, under its own ledger.

        Args:
            index: The item's position.
            fn: The async function.
            gate: Bounds how many items run at once.
        """
        async with gate:
            if self.stop.is_set():
                return
            key, item = self.keyed[index]
            ledger = Ledger(self.budget)
            # Each task runs in its own context copy, so this does not leak.
            LEDGER.set(ledger)
            try:
                outcome = _Outcome(key, _checked(await fn(item)), None, ledger)
            except Exception as err:
                outcome = _Outcome(key, None, err, ledger)
            self._settle(index, outcome)

    def finish(self, out: str | Path, limit: int | None) -> Report:
        """Build the report and, unless *limit* was set, write the files.

        Args:
            out: The results file.
            limit: The run's limit.

        Returns:
            The report.
        """
        if self._progress and self.keyed:
            sys.stderr.write("\n")
        outcomes = [o for o in self.outcomes if o is not None]
        report = self._report(outcomes, written=limit is None)
        if limit is None:
            path = Path(out)
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(
                path,
                "".join(
                    json.dumps(o.line(), ensure_ascii=False) + "\n" for o in outcomes
                ),
            )
            meta = self._meta(report)
            _write_atomic(
                path.with_suffix(".meta.json"), json.dumps(meta, indent=2) + "\n"
            )
        return report

    def _report(self, outcomes: list[_Outcome], *, written: bool) -> Report:
        """Summarise the outcomes.

        Args:
            outcomes: Every item's outcome.
            written: Whether the files will be written.

        Returns:
            The report.
        """
        totals = [o.ledger.totals() for o in outcomes]
        costs = [cost for _, cost, _ in totals]
        served: dict[str, Counter[str]] = {}
        for outcome in outcomes:
            for record in outcome.ledger.records:
                if record.usage is not None:
                    served.setdefault(record.model, Counter())[record.usage["model"]] += 1
        return Report(
            ok=sum(o.error is None for o in outcomes),
            failed=sum(o.error is not None for o in outcomes),
            calls=sum(len(o.ledger.records) for o in outcomes),
            new_calls=sum(sent for sent, _, _ in totals),
            new_cost=None
            if any(c is None for c in costs)
            else sum(c or 0.0 for c in costs),
            replayed_cost=sum(replayed for _, _, replayed in totals),
            failures=[
                Failure(o.key, type(o.error).__name__, str(o.error))
                for o in outcomes
                if o.error is not None
            ],
            drift={m: dict(c) for m, c in served.items() if len(c) > 1},
            budget_reached=self.budget.reached,
            written=written,
        )

    def _meta(self, report: Report) -> dict[str, Any]:
        """Return the metadata written next to the results file.

        Args:
            report: The run's report.

        Returns:
            Plain data.
        """
        commit, dirty = _git_state()
        return {
            "llmkit_version": __version__,
            "git_commit": commit,
            "git_dirty": dirty,
            "started": self.started,
            "ended": _now(),
            "items": len(self.keyed),
            "ok": report.ok,
            "failed": report.failed,
            "new_cost": report.new_cost,
            "replayed_cost": report.replayed_cost,
            "max_cost": self.max_cost,
            "budget_reached": report.budget_reached,
            "drift": report.drift,
        }


def query(
    items: Iterable[Any],
    fn: Callable[[Any], Any],
    *,
    key: Callable[[Any], str],
    out: str | Path,
    concurrency: int = 8,
    max_cost: float | None = None,
    limit: int | None = None,
    progress: bool = True,
) -> Report:
    """Run *fn* over every item and rebuild the results file.

    A plain *fn* runs on a pool of threads; an ``async def`` runs on one event loop.
    Every llmkit call *fn* makes is accounted to its item. An exception from *fn*
    fails only that item.

    Args:
        items: The items; read into a list.
        fn: Maps one item to a JSON-serializable result.
        key: Maps an item to its unique string key.
        out: The results file, rewritten in input order when the run finishes. The
            metadata goes beside it, ``r.jsonl`` → ``r.meta.json``.
        concurrency: Items in flight at once.
        max_cost: Cap on this run's new spend in USD; replays are always served.
        limit: Run only the first *limit* items and write nothing.
        progress: Write a counter to stderr.

    Returns:
        The report.

    Raises:
        ValueError: If two items share a key or *limit* is below 1.
        TypeError: If a key is not a string.
        KeyboardInterrupt: On Ctrl-C, after in-flight items finish; nothing is written.
    """
    if inspect.iscoroutinefunction(fn):
        return asyncio.run(
            aquery(
                items, fn, key=key, out=out, concurrency=concurrency,
                max_cost=max_cost, limit=limit, progress=progress,
            )
        )  # fmt: skip
    run = _Run(_plan(items, key, limit), max_cost, progress)
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = [
            pool.submit(contextvars.copy_context().run, run.run_sync, index, fn)
            for index in range(len(run.keyed))
        ]
        for future in futures:
            future.result()
    except BaseException:
        run.stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown()
    return run.finish(out, limit)


async def aquery(
    items: Iterable[Any],
    fn: Callable[[Any], Any],
    *,
    key: Callable[[Any], str],
    out: str | Path,
    concurrency: int = 8,
    max_cost: float | None = None,
    limit: int | None = None,
    progress: bool = True,
) -> Report:
    """Async counterpart of :func:`query`, for callers already inside an event loop.

    An ``async def`` *fn* runs as tasks on the running loop; a plain *fn* runs
    :func:`query` on a worker thread. On cancellation, in-flight requests are
    cancelled and nothing is written.

    Args:
        items: The items; read into a list.
        fn: Maps one item to a JSON-serializable result.
        key: Maps an item to its unique string key.
        out: The results file.
        concurrency: Items in flight at once.
        max_cost: Cap on this run's new spend in USD.
        limit: Run only the first *limit* items and write nothing.
        progress: Write a counter to stderr.

    Returns:
        The report.
    """
    if not inspect.iscoroutinefunction(fn):
        return await asyncio.to_thread(
            query, items, fn, key=key, out=out, concurrency=concurrency,
            max_cost=max_cost, limit=limit, progress=progress,
        )  # fmt: skip
    run = _Run(_plan(items, key, limit), max_cost, progress)
    gate = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(run.run_async(index, fn, gate))
        for index in range(len(run.keyed))
    ]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        run.stop.set()
        for task in tasks:
            task.cancel()
        raise
    return run.finish(out, limit)


__all__ = ["Failure", "Report", "aquery", "query"]
