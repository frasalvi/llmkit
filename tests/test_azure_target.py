import warnings

import pytest
from helpers import FakeAzure, make_resource

from llmkit.azure.inventory import Deployment
from llmkit.azure.target import (
    AzureEnvKeyWarning,
    AzureRoutingWarning,
    FoundryTarget,
    foundry_target,
)
from llmkit.errors import MissingCredential
from llmkit.registry import resolve

ROUTE = resolve("glm-5.3")
DEP = Deployment(ROUTE.deployment, ROUTE.deployment, "1", "Fireworks", "Succeeded")
ENV_PAIR = {
    "AZURE_API_KEY": "env-key",
    "AZURE_ENDPOINT": "https://echo-sub.services.ai.azure.com",
}


def target(azure, **kw):
    args = {
        "env": {},
        "env_source": None,
        "azure_key": None,
        "azure_endpoint": None,
        "resource": None,
        **kw,
    }
    return foundry_target(ROUTE, azure=azure, **args)


def fleet():
    return [
        make_resource("alpha", order=1, deployments=(DEP,)),
        make_resource("beta", order=2, deployments=(DEP,)),
    ]


def test_explicit_args_win():
    az = FakeAzure(fleet())
    t = target(
        az,
        azure_key="k",
        azure_endpoint="https://x.services.ai.azure.com",
        resource="myres",
        env=ENV_PAIR,
        env_source="/p/.env",
    )
    assert isinstance(t, FoundryTarget)
    assert t.creds == {
        "AZURE_API_KEY": "k",
        "AZURE_ENDPOINT": "https://x.services.ai.azure.com",
    }
    assert t.resource == "myres" and az.calls == []


def test_explicit_args_default_resource_is_subdomain():
    t = target(
        FakeAzure([]), azure_key="k", azure_endpoint="https://x.services.ai.azure.com"
    )
    assert t.resource == "x"


@pytest.mark.parametrize(
    "kw, missing",
    [
        ({"azure_key": "k"}, "azure_endpoint"),
        ({"azure_endpoint": "https://x.services.ai.azure.com"}, "azure_key"),
    ],
)
def test_explicit_args_need_both(kw, missing):
    with pytest.raises(MissingCredential) as exc:
        target(FakeAzure([]), **kw)
    assert missing in str(exc.value)


def test_env_pair_used_and_warns_every_time():
    az = FakeAzure(fleet())
    for _ in range(2):
        with pytest.warns(AzureEnvKeyWarning) as rec:
            t = target(az, env=ENV_PAIR, env_source="/p/.env")
        msg = str(rec[0].message)
        assert "/p/.env" in msg and "echo-sub" in msg and "delete" in msg
        assert t.creds == ENV_PAIR and t.resource == "echo-sub"
    assert "resources" not in az.calls
    assert not any(c.startswith(("spend", "key")) for c in az.calls)


def test_env_pair_half_set_raises():
    with pytest.raises(MissingCredential, match="AZURE_ENDPOINT"):
        target(FakeAzure([]), env={"AZURE_API_KEY": "k"}, env_source="/p/.env")
    with pytest.raises(MissingCredential, match="AZURE_API_KEY"):
        target(FakeAzure([]), env={"AZURE_ENDPOINT": "https://e.services.ai.azure.com"})


def test_env_pair_on_off_resource_says_so():
    echo = make_resource("echo", order=None, cap=None)
    az = FakeAzure([], cached=[echo])
    with pytest.warns(AzureEnvKeyWarning, match="tagged off"):
        target(az, env=ENV_PAIR, env_source="/p/.env")
    assert "resources" not in az.calls


def test_env_pair_no_off_note_for_ordered_resource():
    echo = make_resource("echo", order=1)
    az = FakeAzure([], cached=[echo])
    with pytest.warns(AzureEnvKeyWarning) as rec:
        target(az, env=ENV_PAIR, env_source="/p/.env")
    assert "tagged off" not in str(rec[0].message)


def test_injected_env_does_not_warn():
    az = FakeAzure(fleet())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = target(az, env=ENV_PAIR, env_source=None)
    assert t.creds == ENV_PAIR and az.calls == []


def test_pin_beats_env_pair_but_still_warns():
    az = FakeAzure(fleet())
    with pytest.warns(AzureEnvKeyWarning):
        t = target(az, env=ENV_PAIR, env_source="/p/.env", resource="beta")
    assert t.resource == "beta"
    assert t.creds["AZURE_API_KEY"] == "key-beta"


def test_order_path_fetches_key():
    az = FakeAzure(fleet())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = target(az)
    alpha = fleet()[0]
    assert t.resource == "alpha"
    assert t.creds == {"AZURE_API_KEY": "key-alpha", "AZURE_ENDPOINT": alpha.endpoint}
    assert "key:alpha" in az.calls and "key:beta" not in az.calls


def test_pin_on_order_path():
    assert target(FakeAzure(fleet()), resource="beta").resource == "beta"


def test_routing_warning_deduplicated():
    az = FakeAzure(fleet(), spend={"alpha": 999.0})
    with pytest.warns(AzureRoutingWarning, match="using beta") as rec:
        target(az)
        target(az)
    assert len(rec) == 1
