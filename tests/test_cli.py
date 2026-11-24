import io
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from helpers import make_resource

from llmkit import cli
from llmkit.azure.arm import Arm
from llmkit.azure.deploy import DeployOutcome
from llmkit.azure.inventory import Deployment, Spend
from llmkit.azure.source import Azure
from llmkit.cli import main, status_text
from llmkit.errors import SpendUnavailable

FAR = datetime.now(UTC) + timedelta(days=1)
FETCHED = datetime(2026, 11, 24, 9, 5, tzinfo=UTC)


def dep(name, model=None, state="Succeeded"):
    return Deployment(name, model or name, "1", "Fireworks", state)


def sample():
    alpha = make_resource("alpha", order=1, deployments=(dep("FW-GLM-5.3"), dep("kimi")))
    beta = make_resource(
        "beta", order=2, deployments=(dep("FW-GLM-5.3", state="Disabled"),)
    )
    echo = make_resource("echo", order=None, cap=None)
    kimi = Deployment("kimi", "FW-Kimi-K3", "1", "Fireworks", "Succeeded")
    alpha = type(alpha)(**{**alpha.__dict__, "deployments": (dep("FW-GLM-5.3"), kimi)})
    return [echo, beta, alpha]


def test_status_text_rows_and_matrix():
    rs = sample()
    spends = {
        "sub-alpha": Spend(313.88, "USD", FETCHED),
        "sub-beta": Spend(10.0, "USD", FETCHED),
        "sub-echo": Spend(0.0, "USD", FETCHED),
    }
    text = status_text(rs, spends)
    lines = text.splitlines()
    assert text.index("alpha") < text.index("beta") < text.index("echo")
    assert "$313.88 / $450" in text
    assert "09:05 UTC" in text
    assert "lag" in lines[0].lower()
    assert "✓" in text and "!" in text
    assert "Check:" in text
    check = text[text.index("Check:") :]
    assert "kimi" in check and "FW-Kimi-K3" in check
    assert "FW-GLM-5.3" in text


def test_status_text_has_no_check_section_when_names_match():
    rs = [make_resource("alpha", order=1, deployments=(dep("FW-GLM-5.3"),))]
    text = status_text(rs, {"sub-alpha": Spend(1.0, "USD", FETCHED)})
    assert "Check:" not in text


def test_status_shows_spend_error_inline():
    rs = sample()
    spends = {
        "sub-alpha": Spend(313.88, "USD", FETCHED),
        "sub-beta": SpendUnavailable("Subscription beta", "429 throttled"),
        "sub-echo": Spend(0.0, "USD", FETCHED),
    }
    text = status_text(rs, spends)
    assert "spend unavailable" in text and "$313.88" in text


class Backend:
    """A fake ARM for CLI tests: graph, deployments, cost, tags."""

    def __init__(self, tags_of_beta=None):
        self.requests: list[tuple[str, str]] = []
        self.patches: list[dict] = []
        self.account_tags = tags_of_beta or {}

    def arm(self):
        return Arm(token=lambda: ("tok", FAR), transport=httpx.MockTransport(self.handle))

    def handle(self, req):
        path = req.url.path
        self.requests.append((req.method, path))
        if path.endswith("/resources"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        self.row("acct-alpha", "sub-alpha", "alpha", "1", "450"),
                        self.row("acct-beta", "sub-beta", "beta", "2", "450"),
                        self.row("acct-echo", "sub-echo", "echo", "off", None),
                    ]
                },
            )
        if path.endswith("/deployments"):
            if "acct-alpha" in path:
                return httpx.Response(
                    200,
                    json={
                        "value": [
                            {
                                "name": "FW-GLM-5.3",
                                "properties": {
                                    "model": {
                                        "name": "FW-GLM-5.3",
                                        "version": "1",
                                        "format": "F",
                                    },
                                    "provisioningState": "Succeeded",
                                },
                            }
                        ]
                    },
                )
            return httpx.Response(200, json={"value": []})
        if path.endswith("/query"):
            if "sub-beta" in path:
                return httpx.Response(429, json={"error": {"message": "throttled"}})
            return httpx.Response(
                200,
                json={
                    "properties": {
                        "columns": [{"name": "Cost"}, {"name": "Currency"}],
                        "rows": [[313.87, "USD"]] if "sub-alpha" in path else [],
                    }
                },
            )
        if path.endswith("/tags/default"):
            if req.method == "PATCH":
                self.patches.append(json.loads(req.content))
                return httpx.Response(200, json={})
            return httpx.Response(200, json={"properties": {"tags": self.account_tags}})
        if path.endswith("/listKeys"):
            raise AssertionError("the CLI must never fetch keys")
        return httpx.Response(404, json={"error": {"message": f"unexpected {path}"}})

    @staticmethod
    def row(account, sub, name, order, cap):
        tags = {"llmkit-name": name, "llmkit-order": order}
        if cap:
            tags["llmkit-cap"] = cap
        return {
            "name": account,
            "subscriptionId": sub,
            "subscriptionName": f"Subscription {name}",
            "resourceGroup": f"rg-{name}",
            "location": "eastus",
            "subdomain": account,
            "tags": tags,
        }

    def azure(self, tmp_path):
        return Azure(arm=self.arm(), cache_path=tmp_path / "azure.json")


def test_status_end_to_end(tmp_path, capsys):
    be = Backend()
    code = main(["azure", "status"], azure=be.azure(tmp_path))
    out = capsys.readouterr().out
    assert code == 0
    assert "alpha" in out and "$313.87 / $450" in out
    assert "spend unavailable" in out and "throttled" in out
    assert not any(p.endswith("listKeys") for _, p in be.requests)


def test_status_refresh_bypasses_cache(tmp_path, capsys):
    be = Backend()
    az = be.azure(tmp_path)
    main(["azure", "status"], azure=az)
    n = len(be.requests)
    main(["azure", "status"], azure=az)
    cached = len(be.requests)
    main(["azure", "status", "--refresh"], azure=az)
    assert cached < n * 2 and len(be.requests) > cached


def test_status_survives_cp1252_stream(tmp_path, monkeypatch):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="")
    monkeypatch.setattr("sys.stdout", stream)
    code = main(["azure", "status"], azure=Backend().azure(tmp_path))
    stream.flush()
    assert code == 0
    assert "✓" in raw.getvalue().decode("utf-8")


def find_account_azure(tmp_path, be):
    return be.azure(tmp_path)


class TagBackend(Backend):
    def handle(self, req):
        if req.url.path.endswith("/resources") and b"name =~" in req.content:
            self.requests.append((req.method, req.url.path))
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "name": "acct-beta",
                            "subscriptionId": "sub-beta",
                            "resourceGroup": "rg-beta",
                        }
                    ]
                },
            )
        return super().handle(req)


def test_tag_validates_before_patch(tmp_path, capsys):
    be = TagBackend(tags_of_beta={"llmkit-name": "beta", "llmkit-cap": "450"})
    code = main(["azure", "tag", "acct-beta", "--order", "0"], azure=be.azure(tmp_path))
    assert code == 2
    assert "llmkit-order" in capsys.readouterr().err
    assert be.patches == []


def test_tag_needs_name_when_untagged(tmp_path, capsys):
    be = TagBackend(tags_of_beta={})
    code = main(["azure", "tag", "acct-beta", "--order", "off"], azure=be.azure(tmp_path))
    assert code == 2 and "--name" in capsys.readouterr().err
    assert be.patches == []


def test_tag_numeric_order_needs_cap(tmp_path, capsys):
    be = TagBackend(tags_of_beta={})
    code = main(
        ["azure", "tag", "acct-beta", "--name", "beta", "--order", "2"],
        azure=be.azure(tmp_path),
    )
    assert code == 2 and "llmkit-cap" in capsys.readouterr().err
    assert be.patches == []


def test_tag_merges_and_invalidates(tmp_path, capsys):
    be = TagBackend(
        tags_of_beta={"llmkit-name": "beta", "llmkit-order": "off", "team": "x"}
    )
    az = be.azure(tmp_path)
    az.resources()
    assert az.cached_resources() is not None
    code = main(["azure", "tag", "acct-beta", "--order", "2", "--cap", "300"], azure=az)
    assert code == 0
    assert be.patches == [
        {
            "operation": "Merge",
            "properties": {"tags": {"llmkit-order": "2", "llmkit-cap": "300"}},
        }
    ]
    assert az.cached_resources() is None


def test_tag_unknown_account_exits_nonzero(tmp_path, capsys):
    be = Backend()

    def handle(req):
        if req.url.path.endswith("/resources") and b"name =~" in req.content:
            return httpx.Response(200, json={"data": []})
        return Backend.handle(be, req)

    az = Azure(
        arm=Arm(token=lambda: ("t", FAR), transport=httpx.MockTransport(handle)),
        cache_path=tmp_path / "a.json",
    )
    assert main(["azure", "tag", "nope", "--name", "x", "--order", "off"], azure=az) == 2
    assert "nope" in capsys.readouterr().err


class RecordingAzure:
    def __init__(self):
        self.invalidated = 0
        self.arm = object()
        self.refreshed = []

    def resources(self, refresh=False):
        self.refreshed.append(refresh)
        return [make_resource("alpha", order=1)]

    def invalidate(self):
        self.invalidated += 1


@pytest.mark.parametrize(
    "statuses, code", [(["created", "present"], 0), (["created", "failed"], 1)]
)
def test_deploy_prints_outcomes_and_invalidates(monkeypatch, capsys, statuses, code):
    seen = {}

    def fake_deploy(arm, resources, model, **kw):
        seen.update(model=model, **kw)
        return [DeployOutcome(f"r{i}", s, f"detail{i}") for i, s in enumerate(statuses)]

    monkeypatch.setattr(cli, "deploy", fake_deploy)
    az = RecordingAzure()
    rc = main(
        ["azure", "deploy", "glm-5.3", "--version", "2", "--capacity", "100"],
        azure=az,
    )
    out = capsys.readouterr().out
    assert rc == code
    assert "r0" in out and "created" in out and "detail0" in out
    assert az.invalidated == 1
    assert seen["model"] == "glm-5.3" and seen["version"] == "2"
    assert seen["capacity"] == 100 and seen["dry_run"] is False
    assert az.refreshed == [True]


def test_deploy_dry_run_does_not_invalidate(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "deploy", lambda *a, **k: [DeployOutcome("alpha", "would create", "x")]
    )
    az = RecordingAzure()
    assert main(["azure", "deploy", "glm-5.3", "--dry-run"], azure=az) == 0
    assert az.invalidated == 0


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["azure", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "status" in out and "deploy" in out and "tag" in out


def test_bad_arguments_return_2(capsys):
    assert main(["azure", "bogus"]) == 2


def test_missing_credential_exits_1(tmp_path, capsys):
    from llmkit.errors import MissingCredential

    class Broken:
        def resources(self, refresh=False):
            raise MissingCredential("azure", ["an az login session (run `az login`)"])

    assert main(["azure", "status"], azure=Broken()) == 1
    assert "az login" in capsys.readouterr().err
