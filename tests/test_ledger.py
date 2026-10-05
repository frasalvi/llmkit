import contextvars

import pytest
from helpers import PRICES, TEST_ENV, chat_response, fake_chat_clients, status_error

from llmkit import LLM, BudgetExceeded, FatalRequest
from llmkit.ledger import LEDGER, Budget, Ledger, current_ledger


def make(outcomes=(), async_outcomes=(), **kw):
    kw.setdefault("price_table", PRICES)
    return LLM(
        "glm-5.3",
        env=TEST_ENV,
        clients=fake_chat_clients(outcomes, async_outcomes),
        sleep=lambda s: None,
        rng=lambda: 0.0,
        **kw,
    )


def under(ledger, fn):
    """Run *fn* with *ledger* active, in a copied context."""

    def body():
        LEDGER.set(ledger)
        return fn()

    return contextvars.copy_context().run(body)


def test_no_ledger_outside_a_query():
    assert current_ledger() is None


def test_calls_are_added_to_the_active_ledger_and_hooks():
    seen = []
    ledger = Ledger(Budget(None))
    llm = make([chat_response("a", prompt=1000, completion=100)], on_call=seen.append)
    under(ledger, lambda: llm.complete("hi"))
    assert len(ledger.records) == 1 and seen == ledger.records
    fields = ledger.line_fields()
    assert fields["calls"] == 1
    assert fields["cost"] == pytest.approx(1000e-6 + 200e-6)
    assert fields["stop_reasons"] == ["end"] and fields["served_models"] == ["glm-5.3"]
    sent, new_cost, replayed = ledger.totals()
    assert sent == 1 and new_cost == pytest.approx(fields["cost"]) and replayed == 0.0
    assert ledger.budget.spent == pytest.approx(new_cost)


def test_ledger_receives_records_without_hooks():
    ledger = Ledger(Budget(None))
    under(ledger, lambda: make([chat_response("a")]).complete("hi"))
    assert len(ledger.records) == 1


def test_budget_refuses_sends_once_spent():
    budget = Budget(1e-9)
    llm = make([chat_response("a"), chat_response("b")])
    under(Ledger(budget), lambda: llm.complete("one"))
    with pytest.raises(BudgetExceeded, match="max_cost"):
        under(Ledger(budget), lambda: llm.complete("two"))
    assert len(llm._clients.sync.chat.completions.calls) == 1 and budget.reached


@pytest.mark.filterwarnings("ignore::llmkit.pricing.UnpricedModelWarning")
def test_unpriced_model_under_a_cap_is_refused():
    llm = make([chat_response("a")], price_table={})
    with pytest.raises(BudgetExceeded, match="no price"):
        under(Ledger(Budget(5.0)), lambda: llm.complete("hi"))
    assert llm._clients.sync.chat.completions.calls == []
    ledger = Ledger(Budget(None))
    under(ledger, lambda: llm.complete("hi"))
    assert ledger.line_fields()["cost"] is None and ledger.totals()[1] is None


def test_failed_calls_are_recorded_but_cost_nothing():
    ledger = Ledger(Budget(None))
    llm = make([status_error(400)])
    with pytest.raises(FatalRequest):
        under(ledger, lambda: llm.complete("hi"))
    assert ledger.line_fields()["calls"] == 1
    assert ledger.totals() == (1, 0, 0.0)


async def test_async_calls_use_the_ledger():
    ledger = Ledger(Budget(None))
    llm = make(async_outcomes=[chat_response("a")])
    token = LEDGER.set(ledger)
    try:
        await llm.acomplete("hi")
    finally:
        LEDGER.reset(token)
    assert len(ledger.records) == 1
