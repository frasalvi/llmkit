import json
import warnings

import pytest

from llmkit.pricing import (
    UnpricedModelWarning,
    compute_cost,
    load_table,
    price_keys,
    resolve_rates,
    warn_unpriced,
)
from llmkit.registry import resolve
from llmkit.types import Usage

TABLE = {
    "azure_ai/FW-GLM-5.3": {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "cache_read_input_token_cost": 1e-7,
    },
    "claude-opus-5": {
        "input_cost_per_token": 5e-6,
        "output_cost_per_token": 25e-6,
        "cache_read_input_token_cost": 5e-7,
        "cache_creation_input_token_cost": 6.25e-6,
    },
}


def test_price_keys_most_specific_first():
    keys = price_keys(resolve("glm-5.3"))
    assert keys[0] == "azure_ai/FW-GLM-5.3"
    assert keys[-2:] == ("FW-GLM-5.3", "glm-5.3")
    assert price_keys(resolve("claude-opus-5", "vertex"))[0] == "vertex_ai/claude-opus-5"


def test_resolve_rates_falls_back_to_bare_name():
    rates = resolve_rates(resolve("claude-opus-5"), TABLE)
    assert rates is not None and rates.cache_write == 6.25e-6
    assert resolve_rates(resolve("gpt-unknown"), TABLE) is None


def test_compute_cost_prices_each_token_kind():
    rates = resolve_rates(resolve("claude-opus-5"), TABLE)
    usage = Usage(
        input_tokens=1000,
        output_tokens=100,
        cached_input_tokens=2000,
        cache_write_tokens=400,
    )
    expected = 1000 * 5e-6 + 100 * 25e-6 + 2000 * 5e-7 + 400 * 6.25e-6
    assert compute_cost(usage, rates) == pytest.approx(expected)
    assert compute_cost(usage, None) is None


def test_missing_cached_rate_makes_cost_unknown():
    rates = resolve_rates(resolve("glm-5.3"), TABLE)
    assert compute_cost(Usage(input_tokens=1, cache_write_tokens=5), rates) is None


def test_load_table_uses_fresh_cache_without_fetching(tmp_path):
    cache = tmp_path / "prices.json"
    cache.write_text(json.dumps({"fetched_at": 1000.0, "table": TABLE}))

    def fail():
        raise AssertionError("should not fetch")

    assert load_table(cache, fetch=fail, now=lambda: 1000.0 + 60) == TABLE


def test_load_table_refreshes_stale_cache_and_reuses_it_offline(tmp_path):
    cache = tmp_path / "prices.json"
    cache.write_text(json.dumps({"fetched_at": 0.0, "table": {"old": {}}}))
    fresh = load_table(cache, fetch=lambda: TABLE, now=lambda: 40 * 86400.0)
    assert fresh == TABLE
    assert json.loads(cache.read_text())["table"] == TABLE

    def offline():
        raise OSError("no network")

    assert load_table(cache, fetch=offline, now=lambda: 90 * 86400.0) == TABLE
    assert load_table(tmp_path / "none.json", fetch=offline) is None


def test_warn_unpriced_once_per_model():
    route = resolve("gpt-unknown")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_unpriced(route)
        warn_unpriced(route)
    assert [w.category for w in caught] == [UnpricedModelWarning]
