from datetime import UTC, datetime

import pytest
from helpers import make_resource

from llmkit.azure.inventory import Deployment, Spend
from llmkit.errors import AzureConfigError, SpendUnavailable, UnknownModel
from llmkit.selection import Choice, choose

FETCHED = datetime(2026, 11, 24, 9, 5, tzinfo=UTC)


def dep(name="x", state="Succeeded", model=None):
    return Deployment(name, model or name, "1", "OpenAI", state)


def fleet(deployment=None, **per_name):
    """alpha(1) beta(2) gamma(3) delta(4) echo(off), all carrying `deployment`."""
    out = []
    for name, order in [("alpha", 1), ("beta", 2), ("gamma", 3), ("delta", 4)]:
        d = per_name.get(name, deployment)
        out.append(make_resource(name, order=order, deployments=(d,) if d else ()))
    d = per_name.get("echo", deployment)
    out.append(make_resource("echo", order=None, deployments=(d,) if d else ()))
    return out


class Spends:
    def __init__(self, **amounts):
        self.amounts = amounts
        self.calls = []

    def __call__(self, r):
        self.calls.append(r.name)
        return Spend(self.amounts.get(r.name, 0.0), "USD", FETCHED)


def test_first_in_order_no_warning():
    s = Spends()
    c = choose(fleet(dep()), "x", s)
    assert isinstance(c, Choice)
    assert c.resource.name == "alpha" and c.warning is None
    assert s.calls == ["alpha"]


def test_skips_over_cap_with_reason():
    c = choose(fleet(dep()), "x", Spends(alpha=450.0))
    assert c.resource.name == "beta"
    assert "using beta" in c.warning
    assert "alpha: $450/$450" in c.warning
    assert "09:05 UTC" in c.warning


def test_just_under_cap_is_used():
    assert choose(fleet(dep()), "x", Spends(alpha=449.99)).resource.name == "alpha"


def test_skips_missing_deployment_with_reason():
    s = Spends()
    c = choose(fleet(dep(), alpha=None), "x", s)
    assert c.resource.name == "beta"
    assert "alpha: no x deployment" in c.warning
    assert s.calls == ["beta"]


def test_not_ready_deployment_is_skipped():
    s = Spends()
    c = choose(fleet(dep(), alpha=dep(state="Disabled")), "x", s)
    assert c.resource.name == "beta"
    assert "Disabled" in c.warning and "alpha" in c.warning
    assert s.calls == ["beta"]


def test_all_over_cap_uses_last_candidate():
    s = Spends(alpha=500, beta=500, gamma=500, delta=500, echo=0)
    c = choose(fleet(dep()), "x", s)
    assert c.resource.name == "delta"
    assert c.warning.startswith("all subscriptions over cap")
    assert "echo" not in s.calls


def test_all_over_cap_last_candidate_is_last_with_deployment():
    s = Spends(alpha=500, beta=500)
    c = choose(fleet(dep(), gamma=None, delta=None), "x", s)
    assert c.resource.name == "beta"
    assert c.warning.startswith("all subscriptions over cap")


def test_off_never_auto_picked():
    rs = fleet(None, echo=dep())
    with pytest.raises(UnknownModel) as exc:
        choose(rs, "x", Spends())
    assert "echo" in str(exc.value) and "llmkit azure deploy x" in str(exc.value)


def test_nowhere_raises_unknown_model():
    with pytest.raises(UnknownModel, match="llmkit azure deploy x"):
        choose(fleet(None), "x", Spends())


def test_empty_inventory_raises_unknown_model():
    with pytest.raises(UnknownModel):
        choose([], "x", Spends())


def test_matches_deployment_name_not_model():
    rs = fleet(Deployment("FW-GLM-5.3", "FW-GLM-5.3", "1", "Fireworks", "Succeeded"))
    assert choose(rs, "FW-GLM-5.3", Spends()).resource.name == "alpha"
    rs = fleet(Deployment("my-name", "x", "1", "OpenAI", "Succeeded"))
    with pytest.raises(UnknownModel):
        choose(rs, "x", Spends())


def test_pin_picks_named_even_off():
    c = choose(fleet(dep()), "x", Spends(), pin="echo")
    assert c.resource.name == "echo" and c.warning is None


def test_pin_over_cap_warns():
    c = choose(fleet(dep()), "x", Spends(alpha=500), pin="alpha")
    assert c.resource.name == "alpha" and "over cap" in c.warning


def test_pin_unknown_name_raises_config_error():
    with pytest.raises(AzureConfigError) as exc:
        choose(fleet(dep()), "x", Spends(), pin="zulu")
    assert "zulu" in str(exc.value)
    for n in ("alpha", "beta", "gamma", "delta", "echo"):
        assert n in str(exc.value)


def test_pin_without_deployment_raises_unknown_model():
    with pytest.raises(UnknownModel, match="alpha"):
        choose(fleet(dep(), alpha=None), "x", Spends(), pin="alpha")


def test_pin_not_ready_raises_unknown_model():
    with pytest.raises(UnknownModel, match="Disabled"):
        choose(fleet(dep(), alpha=dep(state="Disabled")), "x", Spends(), pin="alpha")


def test_spend_unavailable_propagates():
    def boom(r):
        raise SpendUnavailable(r.subscription_name, "429")

    with pytest.raises(SpendUnavailable):
        choose(fleet(dep()), "x", boom)
    with pytest.raises(SpendUnavailable):
        choose(fleet(dep()), "x", boom, pin="alpha")
