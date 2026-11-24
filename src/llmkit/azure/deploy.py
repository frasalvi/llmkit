"""Review tier: plumbing.

Deploys a model to every in-order Foundry resource: picks the newest GlobalStandard
version from the account's catalog, sizes capacity from the subscription's quota, and
waits for provisioning.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from llmkit.azure.arm import COGNITIVE_API, Arm
from llmkit.azure.inventory import Resource, _account_path
from llmkit.errors import ArmError
from llmkit.registry import MODELS

SKU = "GlobalStandard"
CAPACITY_CAP = 5000
POLL_SECONDS = 10
POLL_LIMIT = 60


@dataclass(frozen=True)
class DeployOutcome:
    """What happened on one resource: created, present, would create or failed."""

    resource: str
    status: str
    detail: str


def deployment_name(model: str) -> str:
    """Return the Foundry deployment name for a model.

    Args:
        model: A registry model name or a raw deployment name.

    Returns:
        The registry's Foundry route when there is one, else ``model`` unchanged.
    """
    spec = MODELS.get(model)
    if spec is not None and "foundry" in spec.routes:
        return spec.routes["foundry"]
    return model


def _pick_model(arm: Arm, r: Resource, name: str, version: str | None) -> dict[str, str]:
    """Find the catalog entry to deploy.

    Args:
        arm: The ARM client.
        r: The resource whose catalog is read (all pages).
        name: The deployment name.
        version: A required version, or None for the newest.

    Returns:
        The entry's ``format``, ``name`` and ``version``.

    Raises:
        ArmError: If the catalog has no GlobalStandard entry for the name.
    """
    body = arm.request(
        "GET",
        _account_path(r.subscription_id, r.resource_group, r.account) + "/models",
        api_version=COGNITIVE_API,
    )
    named = [m for m in body.get("value", []) if m.get("name") == name]
    if not named:
        raise ArmError(f"{name} is not in the model catalog of {r.account}")
    usable = [
        m
        for m in named
        if SKU in {s.get("name") for s in m.get("skus", [])}
        and (version is None or m.get("version") == version)
    ]
    if not usable:
        wanted = f" version {version}" if version else ""
        raise ArmError(f"{name}{wanted} has no {SKU} sku in the catalog of {r.account}")
    best = max(usable, key=lambda m: str(m.get("version", "")))
    return {
        "format": str(best.get("format", "")),
        "name": name,
        "version": str(best.get("version", "")),
    }


def _capacity(arm: Arm, r: Resource, name: str) -> int:
    """Size the deployment from the subscription's remaining quota.

    Args:
        arm: The ARM client.
        r: The resource.
        name: The deployment name.

    Returns:
        ``min(5000, limit - currentValue)``, or 5000 when there is no quota row.

    Raises:
        ArmError: If the quota is exhausted.
    """
    body = arm.request(
        "GET",
        f"/subscriptions/{r.subscription_id}/providers/Microsoft.CognitiveServices"
        f"/locations/{r.location}/usages",
        api_version=COGNITIVE_API,
    )
    suffix = f".{SKU}.{name}".lower()
    for row in body.get("value", []):
        if str(row.get("name", {}).get("value", "")).lower().endswith(suffix):
            left = int(row.get("limit", 0)) - int(row.get("currentValue", 0))
            if left <= 0:
                raise ArmError(f"no {SKU} quota left for {name} in {r.location}")
            return min(CAPACITY_CAP, left)
    return CAPACITY_CAP


def _wait(arm: Arm, path: str, sleep: Callable[[float], None]) -> str | None:
    """Poll a deployment until it finishes provisioning.

    Args:
        arm: The ARM client.
        path: The deployment's path.
        sleep: Blocking sleep.

    Returns:
        None on success, otherwise a description of the failure.
    """
    state = ""
    for _ in range(POLL_LIMIT):
        body = arm.request("GET", path, api_version=COGNITIVE_API)
        state = str(body.get("properties", {}).get("provisioningState", ""))
        if state == "Succeeded":
            return None
        if state in ("Failed", "Canceled"):
            return f"provisioning {state}"
        sleep(POLL_SECONDS)
    return f"still provisioning after {POLL_LIMIT * POLL_SECONDS // 60} minutes ({state})"


def _deploy_one(
    arm: Arm,
    r: Resource,
    name: str,
    version: str | None,
    capacity: int | None,
    dry_run: bool,
    sleep: Callable[[float], None],
) -> DeployOutcome:
    """Deploy to one resource.

    Args:
        arm: The ARM client.
        r: The resource.
        name: The deployment name.
        version: A required version, or None for the newest.
        capacity: Capacity override, or None to size from quota.
        dry_run: Look everything up but create nothing.
        sleep: Blocking sleep.

    Returns:
        The outcome.

    Raises:
        ArmError: On any Azure failure; the caller turns it into ``failed``.
    """
    present = r.deployment(name)
    if present is not None:
        return DeployOutcome(
            r.name, "present", f"{name} already exists ({present.state})"
        )
    chosen = _pick_model(arm, r, name, version)
    cap = capacity if capacity is not None else _capacity(arm, r, name)
    properties: dict[str, object] = {"model": chosen}
    if chosen["format"] == "Anthropic":
        if r.anthropic_org is None:
            raise ArmError(
                f"{name} is an Anthropic model; tag {r.account} with "
                "llmkit-anthropic-org (name|industry|XX) first"
            )
        org, industry, country = r.anthropic_org
        properties["modelProviderData"] = {
            "organizationName": org,
            "industry": industry,
            "countryCode": country,
        }
    summary = f"{name} v{chosen['version']} capacity {cap}"
    if dry_run:
        return DeployOutcome(r.name, "would create", summary)
    path = _account_path(r.subscription_id, r.resource_group, r.account) + (
        f"/deployments/{name}"
    )
    arm.request(
        "PUT",
        path,
        api_version=COGNITIVE_API,
        json={"sku": {"name": SKU, "capacity": cap}, "properties": properties},
    )
    problem = _wait(arm, path, sleep)
    if problem is not None:
        return DeployOutcome(r.name, "failed", f"{summary}: {problem}")
    return DeployOutcome(r.name, "created", summary)


def deploy(
    arm: Arm,
    resources: Sequence[Resource],
    model: str,
    *,
    version: str | None = None,
    capacity: int | None = None,
    dry_run: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> list[DeployOutcome]:
    """Deploy a model to every resource with a numeric order, in order.

    Args:
        arm: The ARM client.
        resources: The inventory; resources tagged off are never touched.
        model: A registry model name or a raw deployment name.
        version: Deploy this catalog version instead of the newest.
        capacity: Deploy with this capacity instead of sizing from quota.
        dry_run: Report what would be created without creating anything.
        sleep: Blocking sleep, for tests.

    Returns:
        One outcome per ordered resource; a failure on one does not stop the rest.
    """
    name = deployment_name(model)
    ordered = sorted(
        (r for r in resources if r.order is not None), key=lambda r: r.order or 0
    )
    outcomes: list[DeployOutcome] = []
    for r in ordered:
        try:
            outcomes.append(_deploy_one(arm, r, name, version, capacity, dry_run, sleep))
        except ArmError as e:
            outcomes.append(DeployOutcome(r.name, "failed", str(e)))
    return outcomes
