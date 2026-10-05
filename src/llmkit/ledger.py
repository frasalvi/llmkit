"""Review tier: plumbing.

Per-item accounting for :func:`llmkit.query`.

While a query runs an item, a context variable holds that item's :class:`Ledger`. Every
llmkit call made under it adds its call record, and asks the run's :class:`Budget` for
permission before sending. Calls made outside a query see no ledger and are unaffected.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from .errors import BudgetExceeded
from .records import CallRecord

LEDGER: ContextVar[Ledger | None] = ContextVar("llmkit_ledger", default=None)


def current_ledger() -> Ledger | None:
    """Return the ledger of the item being run, or ``None`` outside a query."""
    return LEDGER.get()


class Budget:
    """New spend of one query run, against an optional cap."""

    def __init__(self, max_cost: float | None) -> None:
        """Start an empty budget.

        Args:
            max_cost: The cap on new spend in USD, or ``None`` for no cap.
        """
        self.max_cost = max_cost
        self.spent = 0.0
        self.reached = False
        self._lock = threading.Lock()

    def admit(self, *, priced: bool, model: str) -> None:
        """Allow one request, or refuse it before anything is sent.

        Args:
            priced: Whether the model has a price.
            model: The model name, for the message.

        Raises:
            BudgetExceeded: If the cap is reached, or the model has no price under a cap.
        """
        if self.max_cost is None:
            return
        if not priced:
            raise BudgetExceeded(f"{model} has no price, so max_cost cannot be enforced")
        with self._lock:
            if self.spent >= self.max_cost:
                self.reached = True
                raise BudgetExceeded(f"max_cost ${self.max_cost:.2f} reached")

    def charge(self, cost: float | None) -> None:
        """Add the cost of one sent request.

        Args:
            cost: USD, or ``None`` when unknown.
        """
        with self._lock:
            self.spent += cost or 0.0


def _original_cost(record: CallRecord) -> float | None:
    """Return what a record's outcome originally cost, whether sent or replayed.

    Args:
        record: A record that carries usage.

    Returns:
        USD, or ``None`` when unknown.
    """
    assert record.usage is not None
    cost: float | None = record.usage[
        "replayed_cost" if record.cache == "hit" else "cost"
    ]
    return cost


@dataclass
class Ledger:
    """The calls one item made.

    Attributes:
        budget: The run's shared budget.
        records: Every call record, in order.
    """

    budget: Budget
    records: list[CallRecord] = field(default_factory=list)

    def admit(self, *, priced: bool, model: str) -> None:
        """Ask the run's budget to allow one request. See :meth:`Budget.admit`.

        Args:
            priced: Whether the model has a price.
            model: The model name.
        """
        self.budget.admit(priced=priced, model=model)

    def add(self, record: CallRecord) -> None:
        """Keep a record and charge the budget for a sent request.

        Args:
            record: The call record.
        """
        self.records.append(record)
        if record.cache != "hit" and record.usage is not None:
            self.budget.charge(record.usage["cost"])

    def line_fields(self) -> dict[str, Any]:
        """Return the item's accounting for the results file.

        Identical on a rerun that replays every call, so committed results files only
        change when results do.

        Returns:
            ``calls``, ``cost`` (original cost of the outcomes used, ``None`` when any
            is unknown), ``stop_reasons`` and ``served_models``.
        """
        costs = [_original_cost(r) for r in self.records if r.usage is not None]
        return {
            "calls": len(self.records),
            "cost": None
            if any(c is None for c in costs)
            else sum(c or 0.0 for c in costs),
            "stop_reasons": [r.stop_reason for r in self.records if r.stop_reason],
            "served_models": sorted(
                {r.usage["model"] for r in self.records if r.usage is not None}
            ),
        }

    def totals(self) -> tuple[int, float | None, float]:
        """Return this item's new and replayed spend.

        Returns:
            Requests sent, their cost (``None`` when any is unknown), and the original
            cost of the outcomes replayed.
        """
        sent = [r for r in self.records if r.cache != "hit"]
        costs = [r.usage["cost"] for r in sent if r.usage is not None]
        new_cost = None if any(c is None for c in costs) else sum(c or 0.0 for c in costs)
        replayed = sum(
            r.usage["replayed_cost"] or 0.0
            for r in self.records
            if r.cache == "hit" and r.usage is not None
        )
        return len(sent), new_cost, replayed
