from llmkit import BudgetExceeded, LLMKitError, Message, RequestError, Usage
from llmkit.records import build_record
from llmkit.registry import resolve
from llmkit.transports.base import Call
from llmkit.types import Result


def call():
    return Call(route=resolve("glm-5.3"), messages=[Message.user("hi")])


def test_usage_pools_replayed_cost():
    pooled = Usage(cost=0.0, replayed_cost=0.5) + Usage(cost=0.25)
    assert pooled.cost == 0.25 and pooled.replayed_cost == 0.5
    assert (Usage(cost=1.0) + Usage(cost=2.0)).replayed_cost is None


def test_result_cache_defaults_to_none():
    result = Result("t", "", [], "end", Usage(), Message.assistant("t"))
    assert result.cache is None


def test_record_carries_cache_fields():
    plain = build_record(
        call(), result=None, error=None, attempts=1, latency_ms=0, tags={}
    )
    assert (plain.cache, plain.sample, plain.cache_attempts) == (None, 0, None)
    hit = build_record(
        call(),
        result=None,
        error=None,
        attempts=0,
        latency_ms=0,
        tags={},
        cache="hit",
        sample=2,
        cache_attempts=1,
    )
    assert (hit.cache, hit.sample, hit.cache_attempts) == ("hit", 2, 1)
    assert hit.to_dict()["cache"] == "hit"


def test_budget_exceeded_is_not_a_request_error():
    err = BudgetExceeded("max_cost $1.00 reached")
    assert isinstance(err, LLMKitError) and not isinstance(err, RequestError)
