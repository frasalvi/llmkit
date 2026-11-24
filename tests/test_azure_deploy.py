import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from helpers import make_resource

from llmkit.azure.arm import Arm
from llmkit.azure.deploy import DeployOutcome, deploy, deployment_name
from llmkit.azure.inventory import Deployment

FAR = datetime.now(UTC) + timedelta(days=1)
DEP = "FW-GLM-5.3"


class Fake:
    """Routes ARM calls for deploy tests and records every request."""

    def __init__(self, models=None, usages=None, states=("Succeeded",), fail_put=None):
        self.models = models if models is not None else [model("1")]
        self.usages = usages or []
        self.states = list(states)
        self.fail_put = fail_put or {}
        self.requests: list[tuple[str, str]] = []
        self.puts: list[dict] = []
        self.versions: list[str | None] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        self.requests.append((req.method, path))
        self.versions.append(req.url.params.get("api-version"))
        if path.endswith("/models"):
            if req.url.params.get("page") == "2":
                return httpx.Response(200, json={"value": self.models[1:]})
            body = {"value": self.models[:1]}
            if len(self.models) > 1:
                link = f"https://management.azure.com{path}?api-version=2026-09-15-preview&page=2"
                body["nextLink"] = link
            return httpx.Response(200, json=body)
        if path.endswith("/usages"):
            return httpx.Response(200, json={"value": self.usages})
        if req.method == "PUT":
            acct = path.split("/accounts/")[1].split("/")[0]
            if acct in self.fail_put:
                return httpx.Response(
                    400, json={"error": {"message": self.fail_put[acct]}}
                )
            self.puts.append({"path": path, "body": json.loads(req.content)})
            return httpx.Response(201, json={})
        if req.method == "GET" and "/deployments/" in path:
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return httpx.Response(200, json={"properties": {"provisioningState": state}})
        return httpx.Response(404, json={"error": {"message": f"unexpected {path}"}})

    def arm(self) -> Arm:
        return Arm(
            token=lambda: ("tok", FAR), transport=httpx.MockTransport(self.handler)
        )


def model(version, skus=("GlobalStandard",), fmt="Fireworks", name=DEP):
    return {
        "name": name,
        "version": version,
        "format": fmt,
        "skus": [{"name": s} for s in skus],
    }


def usage(deployment, limit, current):
    return {
        "name": {"value": f"OpenAI.GlobalStandard.{deployment}"},
        "limit": limit,
        "currentValue": current,
    }


def fleet(**kw):
    return [
        make_resource("alpha", order=1, **kw),
        make_resource("echo", order=None, cap=None),
    ]


def run(fake, resources=None, model_name="glm-5.3", **kw):
    kw.setdefault("sleep", lambda s: None)
    return deploy(fake.arm(), resources or fleet(), model_name, **kw)


def test_deployment_name_maps_registry():
    assert deployment_name("glm-5.3") == "FW-GLM-5.3"
    assert deployment_name("claude-opus-5-5") == "claude-opus-5-5"
    assert deployment_name("never-heard-of-it") == "never-heard-of-it"


def test_present_any_state_skipped():
    d = Deployment(DEP, DEP, "1", "Fireworks", "Disabled")
    fake = Fake()
    [out] = run(fake, fleet(deployments=(d,)))
    assert (out.resource, out.status) == ("alpha", "present")
    assert "Disabled" in out.detail
    assert not any(m == "PUT" for m, _ in fake.requests)


def test_off_resources_untouched():
    fake = Fake()
    outs = run(fake)
    assert [o.resource for o in outs] == ["alpha"]
    assert not any("acct-echo" in p for _, p in fake.requests)


def test_outcomes_in_order():
    rs = [
        make_resource("beta", order=2),
        make_resource("alpha", order=1),
    ]
    assert [o.resource for o in run(Fake(), rs)] == ["alpha", "beta"]


def test_dry_run_sends_nothing():
    fake = Fake(usages=[usage(DEP, 3000, 1000)])
    [out] = run(fake, dry_run=True)
    assert out.status == "would create"
    assert "1" in out.detail and "2000" in out.detail
    assert {m for m, _ in fake.requests} == {"GET"}


def test_every_call_uses_the_preview_api_version():
    fake = Fake(models=[model("1"), model("2")], usages=[usage(DEP, 8000, 0)])
    run(fake)
    methods = {m for m, _ in fake.requests}
    assert methods == {"GET", "PUT"}
    assert set(fake.versions) == {"2026-09-15-preview"}


def test_created_request_shape():
    fake = Fake()
    [out] = run(fake)
    assert out == DeployOutcome("alpha", "created", out.detail)
    [put] = fake.puts
    assert put["path"].endswith("/accounts/acct-alpha/deployments/FW-GLM-5.3")
    assert put["body"] == {
        "sku": {"name": "GlobalStandard", "capacity": 5000},
        "properties": {"model": {"format": "Fireworks", "name": DEP, "version": "1"}},
    }


def test_picks_newest_globalstandard_version():
    models = [model("1"), model("2"), model("3", skus=("DataZoneStandard",))]
    fake = Fake(models=models)
    run(fake)
    assert fake.puts[0]["body"]["properties"]["model"]["version"] == "2"
    fake = Fake(models=models)
    run(fake, version="1")
    assert fake.puts[0]["body"]["properties"]["model"]["version"] == "1"


def test_model_on_second_catalog_page_is_found():
    other = model("1", name="something-else")
    fake = Fake(models=[other, model("7")])
    [out] = run(fake)
    assert out.status == "created"
    assert fake.puts[0]["body"]["properties"]["model"]["version"] == "7"


@pytest.mark.parametrize(
    "usages, expected",
    [
        ([usage(DEP, 8000, 1000)], 5000),
        ([usage(DEP, 3000, 1000)], 2000),
        ([], 5000),
        ([usage(DEP.lower(), 3000, 1000)], 2000),
    ],
)
def test_capacity_from_quota_capped_at_5000(usages, expected):
    fake = Fake(usages=usages)
    run(fake)
    assert fake.puts[0]["body"]["sku"]["capacity"] == expected


def test_explicit_capacity_is_used():
    fake = Fake(usages=[usage(DEP, 3000, 1000)])
    run(fake, capacity=100)
    assert fake.puts[0]["body"]["sku"]["capacity"] == 100


def test_no_quota_left_fails():
    fake = Fake(usages=[usage(DEP, 1000, 1000)])
    [out] = run(fake)
    assert out.status == "failed" and "quota" in out.detail
    assert fake.puts == []


def test_anthropic_sends_model_provider_data():
    name = "claude-opus-5-5"
    fake = Fake(models=[model("1", fmt="Anthropic", name=name)])
    r = make_resource("alpha", order=1)
    r = type(r)(
        **{**r.__dict__, "anthropic_org": ("Example University", "Education", "US")}
    )
    [out] = run(fake, [r], model_name=name)
    assert out.status == "created"
    props = fake.puts[0]["body"]["properties"]
    assert props["modelProviderData"] == {
        "organizationName": "Example University",
        "industry": "Education",
        "countryCode": "US",
    }


def test_anthropic_without_org_tag_fails_that_resource():
    name = "claude-opus-5-5"
    fake = Fake(models=[model("1", fmt="Anthropic", name=name)])
    org = make_resource("beta", order=2)
    org = type(org)(**{**org.__dict__, "anthropic_org": ("U", "Education", "US")})
    rs = [make_resource("alpha", order=1), org]
    outs = run(fake, rs, model_name=name)
    assert [(o.resource, o.status) for o in outs] == [
        ("alpha", "failed"),
        ("beta", "created"),
    ]
    assert "llmkit-anthropic-org" in outs[0].detail
    assert len(fake.puts) == 1


def test_non_anthropic_has_no_provider_data():
    fake = Fake()
    run(fake)
    assert "modelProviderData" not in fake.puts[0]["body"]["properties"]


def test_waits_until_succeeded():
    sleeps = []
    fake = Fake(states=("Creating", "Creating", "Succeeded"))
    [out] = run(fake, sleep=sleeps.append)
    assert out.status == "created"
    assert sleeps == [10, 10]


def test_timeout_is_failed_still_provisioning():
    sleeps = []
    fake = Fake(states=("Creating",))
    [out] = run(fake, sleep=sleeps.append)
    assert out.status == "failed" and "still provisioning" in out.detail
    assert sum(sleeps) >= 600


def test_provisioning_failure_reported():
    [out] = run(Fake(states=("Failed",)))
    assert out.status == "failed" and "Failed" in out.detail


def test_model_not_in_catalog_fails():
    fake = Fake(models=[model("1", name="other")])
    [out] = run(fake)
    assert out.status == "failed" and DEP in out.detail
    assert fake.puts == []


def test_catalog_without_globalstandard_fails():
    [out] = run(Fake(models=[model("1", skus=("DataZoneStandard",))]))
    assert out.status == "failed" and "GlobalStandard" in out.detail


def test_arm_error_becomes_failed_outcome_and_continues():
    rs = [make_resource("alpha", order=1), make_resource("beta", order=2)]
    fake = Fake(fail_put={"acct-alpha": "quota exhausted in region"})
    outs = run(fake, rs)
    assert [(o.resource, o.status) for o in outs] == [
        ("alpha", "failed"),
        ("beta", "created"),
    ]
    assert "quota exhausted in region" in outs[0].detail
