import json
import os
import threading
import warnings
from datetime import UTC, datetime, timedelta

import pytest
from helpers import make_resource

from llmkit.azure import source
from llmkit.azure.inventory import Spend, resource_to_dict
from llmkit.azure.source import Azure, AzureInventoryWarning, default_azure
from llmkit.errors import ArmError, AzureConfigError, MissingCredential, SpendUnavailable

T0 = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


class Calls:
    def __init__(self, monkeypatch):
        self.discover = 0
        self.spend = 0
        self.key = 0
        self.discover_error: Exception | None = None
        self.spend_error: Exception | None = None
        self.resources = [make_resource("alpha", order=1)]
        monkeypatch.setattr(source, "discover", self._discover)
        monkeypatch.setattr(source, "month_spend", self._spend)
        monkeypatch.setattr(source, "account_key", self._key)

    def _discover(self, arm):
        self.discover += 1
        if self.discover_error:
            raise self.discover_error
        return self.resources

    def _spend(self, arm, sub_id, sub_name, now):
        self.spend += 1
        if self.spend_error:
            raise self.spend_error
        return Spend(100.0 * self.spend, "USD", now)

    def _key(self, arm, r):
        self.key += 1
        return "secret-key"


@pytest.fixture
def env(monkeypatch, tmp_path):
    calls = Calls(monkeypatch)
    clock = Clock()
    path = tmp_path / "sub" / "azure.json"
    az = Azure(arm=object(), cache_path=path, now=clock)  # type: ignore[arg-type]
    return az, calls, clock, path


def test_resources_cached_for_24h(env):
    az, calls, clock, _ = env
    assert az.resources() == calls.resources
    clock.advance(hours=23)
    az.resources()
    assert calls.discover == 1
    clock.advance(hours=2)
    az.resources()
    assert calls.discover == 2


def test_cache_persists_across_instances(env):
    az, calls, clock, path = env
    az.resources()
    az2 = Azure(arm=object(), cache_path=path, now=clock)  # type: ignore[arg-type]
    assert az2.resources() == calls.resources
    assert calls.discover == 1


def test_refresh_bypasses_cache(env):
    az, calls, _, _ = env
    az.resources()
    az.resources(refresh=True)
    assert calls.discover == 2


def test_stale_inventory_used_when_refresh_fails(env):
    az, calls, clock, _ = env
    az.resources()
    clock.advance(hours=30)
    calls.discover_error = ArmError("boom", status=500)
    with pytest.warns(AzureInventoryWarning, match="boom"):
        assert az.resources() == calls.resources


def test_stale_inventory_used_when_not_logged_in(env):
    az, calls, clock, _ = env
    az.resources()
    clock.advance(hours=30)
    calls.discover_error = MissingCredential("azure", ["az login"])
    with pytest.warns(AzureInventoryWarning):
        assert az.resources() == calls.resources


def test_config_error_never_falls_back(env):
    az, calls, clock, _ = env
    az.resources()
    clock.advance(hours=30)
    calls.discover_error = AzureConfigError("bad tag")
    with pytest.raises(AzureConfigError):
        az.resources()


def test_no_inventory_and_failure_raises(env):
    az, calls, _, _ = env
    calls.discover_error = ArmError("boom")
    with pytest.raises(ArmError):
        az.resources()


def test_failed_refresh_keeps_stale_timestamp(env):
    az, calls, clock, _ = env
    az.resources()
    clock.advance(hours=30)
    calls.discover_error = ArmError("boom")
    with pytest.warns(AzureInventoryWarning):
        az.resources()
    calls.discover_error = None
    az.resources()
    assert calls.discover == 3


def test_spend_cached_for_1h_then_refetched(env):
    az, calls, clock, _ = env
    r = calls.resources[0]
    assert az.spend(r).amount == 100.0
    clock.advance(minutes=59)
    assert az.spend(r).amount == 100.0
    clock.advance(minutes=2)
    assert az.spend(r).amount == 200.0
    assert az.spend(r, refresh=True).amount == 300.0


def test_spend_failure_raises_even_with_old_cache(env):
    az, calls, clock, _ = env
    r = calls.resources[0]
    az.spend(r)
    clock.advance(hours=2)
    calls.spend_error = SpendUnavailable("Subscription alpha", "429")
    with pytest.raises(SpendUnavailable):
        az.spend(r)


def test_spend_persisted_across_instances(env):
    az, calls, clock, path = env
    r = calls.resources[0]
    az.spend(r)
    az2 = Azure(arm=object(), cache_path=path, now=clock)  # type: ignore[arg-type]
    assert az2.spend(r).amount == 100.0
    assert calls.spend == 1


def test_key_fetched_once_and_never_written(env):
    az, calls, _, path = env
    r = calls.resources[0]
    az.resources()
    az.spend(r)
    assert az.key(r) == "secret-key" and az.key(r) == "secret-key"
    assert calls.key == 1
    assert "secret-key" not in path.read_text()


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        '{"version": 2, "inventory": {}}',
        "[]",
        '{"version": 1, "inventory": 5}',
    ],
)
def test_bad_cache_treated_as_absent(env, content):
    az, calls, _, path = env
    path.parent.mkdir(parents=True)
    path.write_text(content)
    assert az.cached_resources() is None
    az.resources()
    assert calls.discover == 1


def test_naive_timestamp_entry_is_absent(env):
    az, calls, _, path = env
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "inventory": {
                    "fetched": "2026-11-24T12:00:00",
                    "resources": [resource_to_dict(calls.resources[0])],
                },
            }
        )
    )
    assert az.cached_resources() is None


def test_cache_write_is_atomic(env, monkeypatch):
    az, calls, _, path = env
    az.resources()
    before = path.read_text()
    used = []
    real = os.replace
    monkeypatch.setattr(
        source.os, "replace", lambda a, b: used.append((a, b)) or real(a, b)
    )
    az.resources(refresh=True)
    assert used and used[0][1] == path

    def broken(a, b):
        raise OSError("disk")

    monkeypatch.setattr(source.os, "replace", broken)
    monkeypatch.setattr(source.time, "sleep", lambda s: None)
    az.resources(refresh=True)
    assert path.read_text() == before or json.loads(path.read_text())
    assert [p.name for p in path.parent.iterdir()] == ["azure.json"]


def test_permission_error_retried(env, monkeypatch):
    az, calls, _, path = env
    real = os.replace
    attempts = []

    def flaky(a, b):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("in use")
        real(a, b)

    monkeypatch.setattr(source.os, "replace", flaky)
    monkeypatch.setattr(source.time, "sleep", lambda s: None)
    az.resources()
    assert len(attempts) == 3 and path.exists()


def test_invalidate_drops_inventory_keeps_spend(env):
    az, calls, _, path = env
    r = calls.resources[0]
    az.resources()
    az.spend(r)
    az.invalidate()
    assert az.cached_resources() is None
    az.spend(r)
    assert calls.spend == 1
    data = json.loads(path.read_text())
    assert "inventory" not in data and r.subscription_id in data["spend"]


def test_cached_resources_never_calls_network(env):
    az, calls, clock, _ = env
    assert az.cached_resources() is None
    az.resources()
    clock.advance(hours=48)
    assert az.cached_resources() == calls.resources
    assert calls.discover == 1


def test_lazy_arm_not_built_on_init(monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr(source, "Arm", lambda: built.append(1) or object())
    Azure(cache_path=tmp_path / "a.json")
    assert built == []


def test_default_azure_is_singleton_across_threads(monkeypatch):
    monkeypatch.setattr(source, "_default", None)
    out = []
    barrier = threading.Barrier(8)

    def go():
        barrier.wait()
        out.append(default_azure())

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(out) == 8 and len({id(a) for a in out}) == 1


def test_no_warning_on_fresh_path(env):
    az, _, _, _ = env
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        az.resources()
