import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from helpers import make_resource

from llmkit.azure import arm as arm_mod
from llmkit.azure.arm import Arm, az_token
from llmkit.azure.inventory import (
    account_key,
    discover,
    find_account,
    month_spend,
    set_tags,
)
from llmkit.errors import ArmError, AzureConfigError, MissingCredential, SpendUnavailable

FAR = datetime.now(UTC) + timedelta(days=1)
NOW = datetime(2026, 11, 24, tzinfo=UTC)


def make_arm(handler, token=lambda: ("tok", FAR), sleep=lambda s: None):
    return Arm(token=token, transport=httpx.MockTransport(handler), sleep=sleep)


def test_api_versions_are_the_planned_ones():
    assert arm_mod.GRAPH_API == "2022-10-01"
    assert arm_mod.COGNITIVE_API == "2026-09-15-preview"


def test_429_is_retried_with_retry_after_then_succeeds():
    calls, sleeps = [], []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"Retry-After": "7"}, json={})
        return httpx.Response(200, json={"ok": 1})

    arm = make_arm(handler, sleep=sleeps.append)
    assert arm.request("GET", "/x", api_version="v") == {"ok": 1}
    assert sleeps == [7.0, 7.0]


def test_retry_after_capped_and_defaulted():
    sleeps = []

    def handler(req):
        if len(sleeps) == 0:
            return httpx.Response(429, headers={"Retry-After": "300"}, json={})
        if len(sleeps) == 1:
            return httpx.Response(429, json={})
        return httpx.Response(200, json={})

    make_arm(handler, sleep=sleeps.append).request("GET", "/x", api_version="v")
    assert sleeps == [30.0, 5.0]


def test_429_gives_up_after_three_retries():
    calls, sleeps = [], []

    def handler(req):
        calls.append(1)
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    with pytest.raises(ArmError, match="slow down") as exc:
        make_arm(handler, sleep=sleeps.append).request("GET", "/x", api_version="v")
    assert exc.value.status == 429 and len(calls) == 4 and len(sleeps) == 3


@pytest.mark.parametrize("status", [400, 403, 500, 503])
def test_other_statuses_are_not_retried(status):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(status, json={})

    with pytest.raises(ArmError):
        make_arm(handler).request("GET", "/x", api_version="v")
    assert len(calls) == 1


def test_token_fetched_once_across_threads():
    import threading
    import time

    calls = []

    def slow_token():
        calls.append(1)
        time.sleep(0.05)
        return "tok", FAR

    arm = make_arm(lambda req: httpx.Response(200, json={}), token=slow_token)
    threads = [
        threading.Thread(
            target=arm.request, args=("GET", "/x"), kwargs={"api_version": "v"}
        )
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1


def test_az_token_missing_cli(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(MissingCredential, match="az login"):
        az_token()


def test_az_token_not_logged_in(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "az")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 1, stdout="", stderr="Please run 'az login'"
        ),
    )
    with pytest.raises(MissingCredential, match="az login"):
        az_token()


def test_az_token_bad_json(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "az")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="not json", stderr=""),
    )
    with pytest.raises(MissingCredential, match="az login"):
        az_token()


def test_az_token_parses_and_uses_resolved_path(monkeypatch):
    seen = {}
    monkeypatch.setattr(shutil, "which", lambda _: "C:/x/az.cmd")

    def run(cmd, **k):
        seen["cmd"] = cmd
        out = json.dumps({"accessToken": "t1", "expires_on": 1900000000})
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    token, exp = az_token()
    assert token == "t1"
    assert exp == datetime.fromtimestamp(1900000000, tz=UTC)
    assert seen["cmd"][0] == "C:/x/az.cmd"


def test_request_sends_bearer_and_api_version():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["Authorization"]
        seen["v"] = req.url.params["api-version"]
        seen["path"] = req.url.path
        return httpx.Response(200, json={"ok": 1})

    out = make_arm(handler).request("GET", "/subscriptions/s", api_version="2020-01-01")
    assert out == {"ok": 1}
    assert seen == {"auth": "Bearer tok", "v": "2020-01-01", "path": "/subscriptions/s"}


def test_request_empty_body_is_empty_dict():
    arm = make_arm(lambda req: httpx.Response(200))
    assert arm.request("POST", "/x", api_version="v") == {}


def test_request_follows_next_link():
    def handler(req):
        if req.url.params.get("page") == "2":
            return httpx.Response(200, json={"value": [3]})
        return httpx.Response(
            200,
            json={
                "value": [1, 2],
                "nextLink": "https://management.azure.com/x?page=2&api-version=v",
            },
        )

    out = make_arm(handler).request("GET", "/x", api_version="v")
    assert out["value"] == [1, 2, 3]


def test_request_error_raises_arm_error():
    arm = make_arm(lambda req: httpx.Response(403, json={"error": {"message": "nope"}}))
    with pytest.raises(ArmError) as exc:
        arm.request("GET", "/x", api_version="v")
    assert exc.value.status == 403 and "nope" in str(exc.value)


def test_token_refetched_near_expiry():
    calls = []

    def token():
        calls.append(1)
        return "tok", datetime.now(UTC) + timedelta(minutes=2)

    arm = make_arm(lambda req: httpx.Response(200, json={}), token=token)
    arm.request("GET", "/x", api_version="v")
    arm.request("GET", "/x", api_version="v")
    assert len(calls) == 2


def test_token_cached_when_fresh():
    calls = []

    def token():
        calls.append(1)
        return "tok", FAR

    arm = make_arm(lambda req: httpx.Response(200, json={}), token=token)
    arm.request("GET", "/x", api_version="v")
    arm.request("GET", "/x", api_version="v")
    assert len(calls) == 1


def test_discover_builds_resources():
    tags = {"llmkit-name": "alpha", "llmkit-order": "1", "llmkit-cap": "450"}

    def handler(req):
        if req.method == "POST" and req.url.path.endswith("/resources"):
            assert "llmkit-name" in json.loads(req.content)["query"]
            assert req.url.params["api-version"] == "2022-10-01"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "name": "acct-a",
                            "subscriptionId": "sub-a",
                            "subscriptionName": "Sub A",
                            "resourceGroup": "rg-a",
                            "location": "eastus",
                            "subdomain": "acct-a",
                            "tags": tags,
                        }
                    ]
                },
            )
        assert req.url.params["api-version"] == "2026-09-15-preview"
        assert req.url.path == (
            "/subscriptions/sub-a/resourceGroups/rg-a/providers/"
            "Microsoft.CognitiveServices/accounts/acct-a/deployments"
        )
        return httpx.Response(
            200,
            json={
                "value": [
                    {
                        "name": "gpt-5.6-luna",
                        "properties": {
                            "model": {
                                "name": "gpt-5.6",
                                "version": "2",
                                "format": "OpenAI",
                            },
                            "provisioningState": "Succeeded",
                        },
                    }
                ]
            },
        )

    [r] = discover(make_arm(handler))
    assert (r.name, r.order, r.cap, r.subscription_name) == (
        "alpha",
        1,
        450.0,
        "Sub A",
    )
    assert r.account == "acct-a" and r.endpoint == "https://acct-a.services.ai.azure.com"
    d = r.deployments[0]
    assert d.name == "gpt-5.6-luna" and d.model == "gpt-5.6" and d.ready


def test_discover_runs_check_inventory():
    def row(acct, sub):
        return {
            "name": acct,
            "subscriptionId": sub,
            "subscriptionName": sub,
            "resourceGroup": "rg",
            "location": "eastus",
            "subdomain": acct,
            "tags": {"llmkit-name": "same", "llmkit-order": "off"},
        }

    def handler(req):
        if req.method == "POST":
            return httpx.Response(
                200, json={"data": [row("acct-a", "s1"), row("acct-b", "s2")]}
            )
        return httpx.Response(200, json={"value": []})

    with pytest.raises(AzureConfigError, match="same"):
        discover(make_arm(handler))


COLS = [{"name": "Cost"}, {"name": "Currency"}]


def spend_arm(rows, cols=COLS, status=200, body=None):
    def handler(req):
        assert req.method == "POST"
        assert req.url.path == (
            "/subscriptions/sub-a/providers/Microsoft.CostManagement/query"
        )
        assert req.url.params["api-version"] == "2023-11-01"
        payload = json.loads(req.content)
        assert payload["type"] == "ActualCost"
        assert payload["timeframe"] == "MonthToDate"
        if status != 200:
            return httpx.Response(status, json=body)
        return httpx.Response(200, json={"properties": {"columns": cols, "rows": rows}})

    return make_arm(handler)


def test_month_spend_sums_rows():
    s = month_spend(spend_arm([[313.87, "USD"]]), "sub-a", "Sub A", NOW)
    assert (s.amount, s.currency, s.fetched) == (313.87, "USD", NOW)


def test_month_spend_finds_columns_by_name():
    cols = [{"name": "Currency"}, {"name": "UsageDate"}, {"name": "PreTaxCost"}]
    s = month_spend(spend_arm([["USD", 20261101, 5.5]], cols), "sub-a", "Sub A", NOW)
    assert s.amount == 5.5


def test_month_spend_empty_rows_is_zero():
    s = month_spend(spend_arm([]), "sub-a", "Sub A", NOW)
    assert (s.amount, s.currency) == (0.0, "USD")


def test_month_spend_429_raises():
    arm = spend_arm(
        [], status=429, body={"error": {"message": "Too many requests, retry later"}}
    )
    with pytest.raises(SpendUnavailable) as exc:
        month_spend(arm, "sub-a", "Sub A", NOW)
    assert exc.value.subscription == "Sub A"
    assert "Too many requests" in exc.value.reason


def test_month_spend_non_usd_raises():
    with pytest.raises(SpendUnavailable, match="EUR"):
        month_spend(spend_arm([[10.0, "EUR"]]), "sub-a", "Sub A", NOW)


def test_month_spend_transport_error_raises():
    def handler(req):
        raise httpx.ConnectError("down")

    with pytest.raises(SpendUnavailable):
        month_spend(make_arm(handler), "sub-a", "Sub A", NOW)


def test_account_key_posts_list_keys():
    r = make_resource("alpha", order=1)

    def handler(req):
        assert req.method == "POST"
        assert req.url.params["api-version"] == "2026-09-15-preview"
        assert req.url.path == (
            "/subscriptions/sub-alpha/resourceGroups/rg-alpha/providers/"
            "Microsoft.CognitiveServices/accounts/acct-alpha/listKeys"
        )
        return httpx.Response(200, json={"key1": "k-one", "key2": "k-two"})

    assert account_key(make_arm(handler), r) == "k-one"


def test_find_account():
    def handler(req):
        assert "acct-a" in json.loads(req.content)["query"]
        return httpx.Response(
            200,
            json={
                "data": [{"name": "acct-a", "subscriptionId": "s", "resourceGroup": "g"}]
            },
        )

    assert find_account(make_arm(handler), "acct-a") == ("s", "g")


def test_find_account_missing():
    arm = make_arm(lambda req: httpx.Response(200, json={"data": []}))
    with pytest.raises(AzureConfigError, match="acct-z"):
        find_account(arm, "acct-z")


def test_set_tags_merges():
    seen = {}

    def handler(req):
        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={})

    set_tags(make_arm(handler), "s", "g", "acct-a", {"llmkit-name": "alpha"})
    assert seen["method"] == "PATCH"
    assert seen["path"].endswith(
        "/accounts/acct-a/providers/Microsoft.Resources/tags/default"
    )
    assert seen["body"] == {
        "operation": "Merge",
        "properties": {"tags": {"llmkit-name": "alpha"}},
    }


def test_module_constants():
    assert arm_mod.ARM_BASE == "https://management.azure.com"


def test_get_tags_reads_current_tags():
    from llmkit.azure.inventory import get_tags

    def handler(req):
        assert req.method == "GET"
        assert req.url.path.endswith(
            "/accounts/acct-a/providers/Microsoft.Resources/tags/default"
        )
        return httpx.Response(200, json={"properties": {"tags": {"a": "1"}}})

    assert get_tags(make_arm(handler), "s", "g", "acct-a") == {"a": "1"}
    empty = make_arm(lambda req: httpx.Response(200, json={"properties": {}}))
    assert get_tags(empty, "s", "g", "acct-a") == {}
