import pytest
from helpers import make_resource

from llmkit.azure.inventory import (
    Deployment,
    check_inventory,
    parse_tags,
    resource_from_dict,
    resource_to_dict,
)
from llmkit.errors import ArmError, AzureConfigError, LLMKitError, SpendUnavailable


def test_parse_tags_full():
    out = parse_tags(
        {
            "llmkit-name": "alpha",
            "llmkit-order": "1",
            "llmkit-cap": "450",
            "llmkit-anthropic-org": "Example University|Education|US",
        },
        "acct-a",
    )
    assert out == {
        "name": "alpha",
        "order": 1,
        "cap": 450.0,
        "anthropic_org": ("Example University", "Education", "US"),
    }


def test_parse_tags_off_without_cap():
    out = parse_tags({"llmkit-name": "echo", "llmkit-order": "off"}, "acct-e")
    assert out["order"] is None and out["cap"] is None


@pytest.mark.parametrize(
    "tags, match",
    [
        ({"llmkit-name": "Alpha!", "llmkit-order": "off"}, "llmkit-name"),
        ({"llmkit-order": "off"}, "llmkit-name"),
        ({"llmkit-name": "a"}, "llmkit-order"),
        ({"llmkit-name": "a", "llmkit-order": "0", "llmkit-cap": "450"}, "llmkit-order"),
        (
            {"llmkit-name": "a", "llmkit-order": "first", "llmkit-cap": "450"},
            "llmkit-order",
        ),
        ({"llmkit-name": "a", "llmkit-order": "1"}, "llmkit-cap"),
        ({"llmkit-name": "a", "llmkit-order": "1", "llmkit-cap": "-5"}, "llmkit-cap"),
        ({"llmkit-name": "a", "llmkit-order": "1", "llmkit-cap": "abc"}, "llmkit-cap"),
        (
            {"llmkit-name": "a", "llmkit-order": "off", "llmkit-anthropic-org": "X|Y"},
            "llmkit-anthropic-org",
        ),
        (
            {
                "llmkit-name": "a",
                "llmkit-order": "off",
                "llmkit-anthropic-org": "X|Y|USA",
            },
            "llmkit-anthropic-org",
        ),
        (
            {"llmkit-name": "a", "llmkit-order": "off", "llmkit-anthropic-org": "X||US"},
            "llmkit-anthropic-org",
        ),
    ],
)
def test_parse_tags_rejects(tags, match):
    with pytest.raises(AzureConfigError, match=match) as exc:
        parse_tags(tags, "acct-x")
    assert "acct-x" in str(exc.value)


def test_check_inventory_duplicate_name():
    a = make_resource("alpha", order=1)
    b = make_resource("beta", order=2)
    b = type(b)(**{**b.__dict__, "name": "alpha"})
    with pytest.raises(AzureConfigError, match="alpha") as exc:
        check_inventory([a, b])
    assert "acct-alpha" in str(exc.value) and "acct-beta" in str(exc.value)


def test_check_inventory_duplicate_order():
    a = make_resource("alpha", order=1)
    b = make_resource("beta", order=1)
    with pytest.raises(AzureConfigError, match="order") as exc:
        check_inventory([a, b])
    assert "acct-alpha" in str(exc.value) and "acct-beta" in str(exc.value)


def test_check_inventory_two_in_one_subscription():
    a = make_resource("alpha", order=1, subscription_id="sub-x")
    b = make_resource("beta", order=2, subscription_id="sub-x")
    with pytest.raises(AzureConfigError, match="subscription") as exc:
        check_inventory([a, b])
    assert "acct-alpha" in str(exc.value) and "acct-beta" in str(exc.value)


def test_check_inventory_accepts_valid_and_multiple_off():
    check_inventory(
        [
            make_resource("alpha", order=1),
            make_resource("beta", order=2),
            make_resource("echo", order=None, cap=None),
            make_resource("foxtrot", order=None, cap=None),
        ]
    )


def test_resource_dict_round_trip():
    r = make_resource(
        "alpha",
        order=1,
        deployments=(
            Deployment("FW-GLM-5.3", "FW-GLM-5.3", "1", "Fireworks", "Succeeded"),
        ),
    )
    assert resource_from_dict(resource_to_dict(r)) == r
    assert r.endpoint == "https://alpha-sub.services.ai.azure.com"
    d = r.deployment("FW-GLM-5.3")
    assert d is not None and d.ready
    assert r.deployment("nope") is None


def test_round_trip_with_org_and_no_order():
    r = make_resource("echo", order=None, cap=None)
    r = type(r)(**{**r.__dict__, "anthropic_org": ("U", "Education", "US")})
    assert resource_from_dict(resource_to_dict(r)) == r


def test_not_ready_deployment():
    assert not Deployment("d", "m", "1", "f", "Creating").ready


def test_new_errors():
    e = SpendUnavailable("sub-a", "no access")
    assert (e.subscription, e.reason) == ("sub-a", "no access")
    assert ArmError("x", status=403).status == 403 and ArmError("x").status is None
    for cls in (AzureConfigError, SpendUnavailable, ArmError):
        assert issubclass(cls, LLMKitError)
