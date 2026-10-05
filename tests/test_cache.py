import dataclasses
import sqlite3

import pytest

from llmkit import ContentFiltered, Message, Tool
from llmkit.cache import HIT, MISS, RETRY, Cache, dumps, loads, request_key
from llmkit.registry import resolve
from llmkit.transports.base import Call, Reply
from llmkit.types import Image, ProviderState, Text, ToolCall

BASE = Call(route=resolve("glm-5.3"), messages=[Message.user("hi")], system="s")
SIGNED = ProviderState(
    "gemini",
    {"role": "model", "parts": [{"text": "x", "thought_signature": b"\x00\xffsig"}]},
)


def reply(stop="end", text="hello", state=None):
    return Reply(
        text=text,
        stop_reason=stop,
        tool_calls=[ToolCall("c1", "t", {"a": 1}, '{"a": 1}')],
        served_model="glm-5.3",
        provider_state=state,
    )


def open_cache(tmp_path, **kw):
    return Cache(tmp_path / "cache" / "calls.sqlite", **kw)


def test_key_changes_with_every_keyed_field():
    variants = [
        dataclasses.replace(BASE, system="other"),
        dataclasses.replace(BASE, messages=[Message.user("bye")]),
        dataclasses.replace(
            BASE, messages=[Message.user([Text("hi"), Image(b"\x01", "image/png")])]
        ),
        dataclasses.replace(
            BASE, messages=[Message.user([Text("hi"), Image(b"\x02", "image/png")])]
        ),
        dataclasses.replace(BASE, effort="low"),
        dataclasses.replace(BASE, max_tokens=10),
        dataclasses.replace(BASE, temperature=0.5),
        dataclasses.replace(BASE, top_p=0.9),
        dataclasses.replace(BASE, tools=[Tool("t", "d", {"type": "object"})]),
        dataclasses.replace(BASE, tool_choice="auto"),
        dataclasses.replace(BASE, schema={"type": "object"}),
        dataclasses.replace(BASE, route=resolve("deepseek-v4-pro")),
    ]
    keys = {request_key(v, 0) for v in variants} | {request_key(BASE, 1)}
    assert request_key(BASE, 0) not in keys
    assert len(keys) == len(variants) + 1


def test_key_ignores_cache_prefix():
    prefixed = dataclasses.replace(BASE, cache_prefix=True)
    assert request_key(prefixed, 0) == request_key(BASE, 0)


def test_bytes_in_provider_state_round_trip(tmp_path):
    cache = open_cache(tmp_path)
    first = cache.lookup(request_key(BASE, 0))
    cache.store_reply(first, "glm-5.3", reply(state=SIGNED), 0.01)
    stored = cache.lookup(first.key).entry.reply
    assert stored == reply(state=SIGNED)

    def turn(state):
        return dataclasses.replace(
            BASE,
            messages=[
                Message.user("hi"),
                Message("assistant", "hello", provider_state=state),
                Message.user("more"),
            ],
        )

    assert request_key(turn(stored.provider_state), 0) == request_key(turn(SIGNED), 0)
    assert loads(dumps({"b": b"\x00"})) == {"b": b"\x00"}


def test_miss_then_hit(tmp_path):
    cache = open_cache(tmp_path)
    lookup = cache.lookup("k")
    assert lookup.status == MISS and lookup.entry is None
    entry, won = cache.store_reply(lookup, "glm-5.3", reply(), 0.02)
    assert won and entry.count == 1 and entry.cost == 0.02
    again = cache.lookup("k")
    assert again.status == HIT and again.entry.reply == reply()


def test_retry_on_until_max_attempts(tmp_path):
    cache = open_cache(tmp_path)
    lookup = cache.lookup("k")
    for expected in (1, 2, 3):
        entry, won = cache.store_reply(lookup, "glm-5.3", reply("max_tokens"), 0.01)
        assert won and entry.count == expected
        lookup = cache.lookup("k")
        assert lookup.status == (RETRY if expected < 3 else HIT)


def test_unlimited_attempts(tmp_path):
    cache = open_cache(tmp_path, max_attempts=None)
    lookup = cache.lookup("k")
    for _ in range(5):
        cache.store_reply(lookup, "m", reply("max_tokens"), None)
        lookup = cache.lookup("k")
    assert lookup.status == RETRY and lookup.entry.count == 5


def test_refusals_are_kept_unless_listed(tmp_path):
    kept = open_cache(tmp_path)
    kept.store_reply(kept.lookup("k"), "m", reply("refusal"), 0.0)
    assert kept.lookup("k").status == HIT
    retried = Cache(tmp_path / "other" / "calls.sqlite", retry_on={"refusal"})
    retried.store_reply(retried.lookup("k"), "m", reply("refusal"), 0.0)
    assert retried.lookup("k").status == RETRY


def test_filtered_prompt_is_stored(tmp_path):
    cache = open_cache(tmp_path)
    err = ContentFiltered("blocked", categories=["hate"], provider="foundry", status=400)
    entry, won = cache.store_filtered(cache.lookup("k"), "glm-5.3", err)
    assert won and entry.stop_reason == "content_filter" and entry.reply is None
    stored = cache.lookup("k")
    assert stored.status == HIT
    assert stored.entry.filtered.categories == ["hate"]
    assert str(stored.entry.filtered) == "blocked"
    retried = Cache(tmp_path / "cf" / "calls.sqlite", retry_on={"content_filter"})
    retried.store_filtered(retried.lookup("k"), "m", err)
    assert retried.lookup("k").status == RETRY


def test_first_stored_outcome_wins_across_processes(tmp_path):
    path = tmp_path / "shared" / "calls.sqlite"
    a, b = Cache(path), Cache(path)
    seen_a, seen_b = a.lookup("k"), b.lookup("k")
    _, won_a = a.store_reply(seen_a, "m", reply(text="from a"), 0.01)
    entry, won_b = b.store_reply(seen_b, "m", reply(text="from b"), 0.01)
    assert won_a and not won_b and entry.reply.text == "from a"

    retry_path = tmp_path / "retry" / "calls.sqlite"
    c = Cache(retry_path)
    c.store_reply(c.lookup("k"), "m", reply("max_tokens"), 0.01)
    d = Cache(retry_path)
    seen_c, seen_d = c.lookup("k"), d.lookup("k")
    _, won_c = c.store_reply(seen_c, "m", reply(text="c2"), 0.01)
    entry, won_d = d.store_reply(seen_d, "m", reply(text="d2"), 0.01)
    assert won_c and not won_d and entry.reply.text == "c2" and entry.count == 2


def test_gitignore_only_in_a_directory_the_cache_created(tmp_path):
    open_cache(tmp_path)
    assert (tmp_path / "cache" / ".gitignore").read_text() == "*\n"
    Cache(tmp_path / "calls.sqlite")
    assert not (tmp_path / ".gitignore").exists()


def test_other_format_is_refused(tmp_path):
    path = tmp_path / "cache" / "calls.sqlite"
    Cache(path).close()
    db = sqlite3.connect(path)
    db.execute("UPDATE meta SET value = '99' WHERE name = 'format'")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="format 99"):
        Cache(path)


def test_settings_are_validated(tmp_path):
    with pytest.raises(ValueError, match="bogus"):
        open_cache(tmp_path, retry_on={"bogus"})
    with pytest.raises(ValueError, match="max_attempts"):
        open_cache(tmp_path, max_attempts=0)


def test_stats(tmp_path):
    cache = open_cache(tmp_path)
    cache.store_reply(cache.lookup("a"), "glm-5.3", reply(), 0.25)
    cache.store_reply(cache.lookup("b"), "kimi-k3", reply("refusal"), 0.5)
    assert cache.stats() == {
        "entries": 2,
        "by_model": {"glm-5.3": 1, "kimi-k3": 1},
        "by_stop_reason": {"end": 1, "refusal": 1},
        "stored_cost": 0.75,
    }
