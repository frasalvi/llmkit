import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import PRICES, TEST_ENV, FakeCompletions

from llmkit import LLM
from llmkit.batch import aquery, query
from llmkit.cache import Cache
from llmkit.clients import StaticClients
from llmkit.transports.base import ns

ITEMS = [{"id": c} for c in "abcde"]


def KEY(item):
    return item["id"]


class Echo(FakeCompletions):
    """Answers with the last user message, so completion order never matters."""

    def __init__(self, served=None):
        super().__init__([])
        self.served = served

    def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        model = self.served.pop(0) if self.served else "glm-5.3"
        return ns(
            {
                "model": model,
                "choices": [
                    {"finish_reason": "stop", "message": {"content": f"echo:{prompt}"}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )


class AsyncEcho(Echo):
    async def create(self, **kwargs):  # type: ignore[override]
        return Echo.create(self, **kwargs)


def echo_llm(cache=None, served=None, **kw):
    kw.setdefault("price_table", PRICES)
    clients = StaticClients(
        SimpleNamespace(chat=SimpleNamespace(completions=Echo(served))),
        SimpleNamespace(chat=SimpleNamespace(completions=AsyncEcho())),
    )
    return LLM(
        "glm-5.3",
        env=TEST_ENV,
        clients=clients,
        cache=cache,
        sleep=lambda s: None,
        rng=lambda: 0.0,
        **kw,
    )


def lines(path):
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]


def test_results_in_input_order_with_metadata(tmp_path):
    llm = echo_llm()
    out = tmp_path / "runs" / "r.jsonl"
    report = query(
        ITEMS, lambda it: llm.complete(it["id"]).text,
        key=KEY, out=out, concurrency=3, progress=False,
    )  # fmt: skip
    rows = lines(out)
    assert [r["key"] for r in rows] == list("abcde")
    assert rows[0]["result"] == "echo:a" and rows[0]["calls"] == 1
    assert rows[0]["cost"] > 0 and rows[0]["stop_reasons"] == ["end"]
    assert report.ok == 5 and report.new_calls == 5 and report.written
    meta = json.loads((tmp_path / "runs" / "r.meta.json").read_text())
    assert {
        "llmkit_version", "git_commit", "git_dirty", "started", "ended", "items",
        "ok", "failed", "new_cost", "replayed_cost", "max_cost", "budget_reached",
        "drift",
    } <= set(meta)  # fmt: skip
    assert meta["items"] == 5 and meta["ok"] == 5


def test_rerun_replays_everything_and_writes_the_same_file(tmp_path):
    llm = echo_llm(Cache(tmp_path / "cache" / "calls.sqlite"))
    out = tmp_path / "r.jsonl"

    def run():
        return query(
            ITEMS, lambda it: llm.complete(it["id"]).text,
            key=KEY, out=out, progress=False,
        )  # fmt: skip

    first = run()
    before = out.read_text(encoding="utf-8")
    second = run()
    assert first.new_calls == 5 and second.new_calls == 0
    assert second.new_cost == 0 and second.replayed_cost == pytest.approx(first.new_cost)
    assert out.read_text(encoding="utf-8") == before
    assert "$0.00 new" in str(second)


def test_an_exception_fails_only_its_item(tmp_path):
    def fn(item):
        if item["id"] == "b":
            raise ValueError("bad item")
        return item["id"]

    report = query(ITEMS, fn, key=KEY, out=tmp_path / "r.jsonl", progress=False)
    rows = {r["key"]: r for r in lines(tmp_path / "r.jsonl")}
    assert rows["b"]["error"] == {"type": "ValueError", "message": "bad item"}
    assert rows["a"]["result"] == "a" and report.ok == 4 and report.failed == 1
    assert report.failures[0].key == "b"


def test_a_result_that_is_not_json_fails_its_item(tmp_path):
    report = query(
        ITEMS[:1], lambda it: {1, 2}, key=KEY, out=tmp_path / "r.jsonl", progress=False
    )
    assert report.failures[0].error_type == "TypeError"


def test_bad_keys_and_limits_are_refused_before_any_call(tmp_path):
    llm = echo_llm()

    def fn(it):
        return llm.complete(it["id"]).text

    with pytest.raises(ValueError, match="duplicate"):
        query(ITEMS + ITEMS[:1], fn, key=KEY, out=tmp_path / "r.jsonl", progress=False)
    with pytest.raises(TypeError, match="str"):
        query(ITEMS, fn, key=lambda it: 1, out=tmp_path / "r.jsonl", progress=False)
    with pytest.raises(ValueError, match="limit"):
        query(ITEMS, fn, key=KEY, out=tmp_path / "r.jsonl", limit=0, progress=False)
    assert llm._clients.sync.chat.completions.calls == []


def test_limit_runs_a_prefix_and_writes_nothing(tmp_path):
    llm = echo_llm()
    out = tmp_path / "r.jsonl"
    report = query(
        ITEMS, lambda it: llm.complete(it["id"]).text,
        key=KEY, out=out, limit=2, progress=False,
    )  # fmt: skip
    assert report.ok == 2 and not report.written
    assert not out.exists() and not out.with_suffix(".meta.json").exists()


def test_budget_stops_new_calls_but_serves_replays(tmp_path):
    llm = echo_llm(Cache(tmp_path / "cache" / "calls.sqlite"))

    def fn(it):
        return llm.complete(it["id"]).text

    query(ITEMS[:2], fn, key=KEY, out=tmp_path / "first.jsonl", progress=False)
    report = query(
        ITEMS, fn, key=KEY, out=tmp_path / "second.jsonl",
        concurrency=1, max_cost=1e-9, progress=False,
    )  # fmt: skip
    rows = {r["key"]: r for r in lines(tmp_path / "second.jsonl")}
    assert all("result" in rows[k] for k in "abc")
    assert rows["d"]["error"]["type"] == rows["e"]["error"]["type"] == "BudgetExceeded"
    assert (
        report.budget_reached and report.failed == 2 and "budget reached" in str(report)
    )


@pytest.mark.filterwarnings("ignore::llmkit.pricing.UnpricedModelWarning")
def test_unpriced_calls(tmp_path):
    llm = echo_llm(price_table={})

    def fn(it):
        return llm.complete(it["id"]).text

    capped = query(
        ITEMS[:1], fn, key=KEY, out=tmp_path / "c.jsonl", max_cost=1.0, progress=False
    )
    assert capped.failures[0].error_type == "BudgetExceeded"
    free = query(ITEMS[:1], fn, key=KEY, out=tmp_path / "f.jsonl", progress=False)
    assert free.ok == 1 and free.new_cost is None and "unpriced new" in str(free)
    assert lines(tmp_path / "f.jsonl")[0]["cost"] is None


def test_drift_is_reported(tmp_path):
    llm = echo_llm(served=["glm-5.3-a", "glm-5.3-b"])
    out = tmp_path / "d.jsonl"
    report = query(
        ITEMS[:2], lambda it: llm.complete(it["id"]).text,
        key=KEY, out=out, concurrency=1, progress=False,
    )  # fmt: skip
    assert report.drift == {"glm-5.3": {"glm-5.3-a": 1, "glm-5.3-b": 1}}
    assert json.loads(out.with_suffix(".meta.json").read_text())["drift"] == report.drift


def test_each_item_accounts_for_its_own_calls(tmp_path):
    llm = echo_llm()

    def fn(it):
        llm.complete(it["id"])
        return llm.complete(it["id"] + "!").text

    query(ITEMS, fn, key=KEY, out=tmp_path / "m.jsonl", concurrency=4, progress=False)
    assert all(r["calls"] == 2 for r in lines(tmp_path / "m.jsonl"))


def test_async_fn_gives_the_same_results(tmp_path):
    sync_llm, async_llm = echo_llm(), echo_llm()
    query(
        ITEMS, lambda it: sync_llm.complete(it["id"]).text,
        key=KEY, out=tmp_path / "s.jsonl", progress=False,
    )  # fmt: skip

    async def afn(it):
        return (await async_llm.acomplete(it["id"])).text

    query(ITEMS, afn, key=KEY, out=tmp_path / "a.jsonl", progress=False)

    def strip(rows):
        return [(r["key"], r["result"], r["calls"]) for r in rows]

    assert strip(lines(tmp_path / "s.jsonl")) == strip(lines(tmp_path / "a.jsonl"))


async def test_aquery_inside_a_running_loop(tmp_path):
    llm = echo_llm()

    async def afn(it):
        return (await llm.acomplete(it["id"])).text

    report = await aquery(ITEMS, afn, key=KEY, out=tmp_path / "a.jsonl", progress=False)
    assert report.ok == 5


async def test_aquery_accepts_a_plain_function(tmp_path):
    llm = echo_llm()
    report = await aquery(
        ITEMS, lambda it: llm.complete(it["id"]).text,
        key=KEY, out=tmp_path / "p.jsonl", progress=False,
    )  # fmt: skip
    assert report.ok == 5


def test_interrupt_keeps_the_previous_results(tmp_path):
    out = tmp_path / "r.jsonl"
    out.write_text("previous\n", encoding="utf-8")
    llm = echo_llm(Cache(tmp_path / "cache" / "calls.sqlite"))

    def interrupted(it):
        if it["id"] == "c":
            raise KeyboardInterrupt
        return llm.complete(it["id"]).text

    with pytest.raises(KeyboardInterrupt):
        query(ITEMS, interrupted, key=KEY, out=out, concurrency=1, progress=False)
    assert out.read_text(encoding="utf-8") == "previous\n"
    report = query(
        ITEMS, lambda it: llm.complete(it["id"]).text,
        key=KEY, out=out, concurrency=1, progress=False,
    )  # fmt: skip
    assert report.ok == 5 and report.new_calls == 3


def test_empty_items_write_an_empty_file(tmp_path):
    out = tmp_path / "new" / "dir" / "r.jsonl"
    report = query([], lambda it: it, key=KEY, out=out, progress=False)
    assert out.read_text() == "" and report.ok == 0 and str(report).startswith("0 ok")


def test_progress_counter(tmp_path, capsys):
    query(ITEMS[:2], lambda it: it["id"], key=KEY, out=tmp_path / "r.jsonl")
    assert "query: 2/2" in capsys.readouterr().err


def test_git_state_is_read_before_the_results_are_written(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run(
        [*git, "-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "-m", "i"],
        check=True,
    )
    monkeypatch.chdir(repo)
    query(ITEMS[:1], lambda it: it["id"], key=KEY, out="runs/r.jsonl", progress=False)
    meta = json.loads((repo / "runs" / "r.meta.json").read_text())
    assert meta["git_commit"] and meta["git_dirty"] is False


def test_concurrency_below_one_is_refused(tmp_path):
    async def afn(it):
        return it["id"]

    for fn in (lambda it: it["id"], afn):
        with pytest.raises(ValueError, match="concurrency"):
            query(
                ITEMS,
                fn,
                key=KEY,
                out=tmp_path / "r.jsonl",
                concurrency=0,
                progress=False,
            )
