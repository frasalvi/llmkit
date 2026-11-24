"""Review tier: plumbing.

Types for discovered Azure Foundry resources, parsing of the ``llmkit-*`` tags that
describe them, and validation of the whole inventory.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from llmkit.errors import AzureConfigError

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_COUNTRY_RE = re.compile(r"^[A-Z]{2}$")


@dataclass(frozen=True)
class Deployment:
    """One model deployment on a resource."""

    name: str
    model: str
    version: str
    format: str
    state: str

    @property
    def ready(self) -> bool:
        """Whether the deployment finished provisioning."""
        return self.state == "Succeeded"


@dataclass(frozen=True)
class Resource:
    """A tagged Azure Foundry resource and its deployments."""

    name: str
    account: str
    subdomain: str
    subscription_id: str
    subscription_name: str
    resource_group: str
    location: str
    order: int | None
    cap: float | None
    anthropic_org: tuple[str, str, str] | None
    deployments: tuple[Deployment, ...] = ()

    @property
    def endpoint(self) -> str:
        """The resource's Foundry endpoint URL."""
        return f"https://{self.subdomain}.services.ai.azure.com"

    def deployment(self, name: str) -> Deployment | None:
        """Find a deployment by name.

        Args:
            name: The deployment name.

        Returns:
            The deployment, or None when the resource has no such deployment.
        """
        for d in self.deployments:
            if d.name == name:
                return d
        return None


@dataclass(frozen=True)
class Spend:
    """Month-to-date spend for a subscription."""

    amount: float
    currency: str
    fetched: datetime


def parse_tags(tags: Mapping[str, str], account: str) -> dict[str, Any]:
    """Parse and validate the ``llmkit-*`` tags of one account.

    Args:
        tags: The account's Azure tags.
        account: The account name, used in error messages.

    Returns:
        A dict with keys ``name``, ``order`` (int or None for off), ``cap``
        (float or None) and ``anthropic_org`` (3-tuple or None).

    Raises:
        AzureConfigError: If a tag is missing or malformed.
    """
    # name
    name = tags.get("llmkit-name", "")
    if not _NAME_RE.match(name):
        raise AzureConfigError(
            f"{account}: tag llmkit-name must match [a-z0-9][a-z0-9-]*, got {name!r}"
        )

    # order
    raw_order = tags.get("llmkit-order")
    order: int | None
    if raw_order == "off":
        order = None
    else:
        try:
            order = int(raw_order) if raw_order is not None else 0
        except ValueError:
            order = 0
        if order < 1:
            raise AzureConfigError(
                f"{account}: tag llmkit-order must be 'off' or an integer >= 1, "
                f"got {raw_order!r}"
            )

    # cap
    raw_cap = tags.get("llmkit-cap")
    cap: float | None = None
    if raw_cap is not None:
        try:
            cap = float(raw_cap)
        except ValueError:
            cap = -1.0
        if not cap > 0:
            raise AzureConfigError(
                f"{account}: tag llmkit-cap must be a number > 0, got {raw_cap!r}"
            )
    elif order is not None:
        raise AzureConfigError(
            f"{account}: tag llmkit-cap is required when llmkit-order is numeric"
        )

    # anthropic org
    org: tuple[str, str, str] | None = None
    raw_org = tags.get("llmkit-anthropic-org")
    if raw_org is not None:
        parts = raw_org.split("|")
        if (
            len(parts) != 3
            or not all(p.strip() for p in parts)
            or not _COUNTRY_RE.match(parts[2])
        ):
            raise AzureConfigError(
                f"{account}: tag llmkit-anthropic-org must be "
                f"'name|industry|XX' with a two-letter country code, got {raw_org!r}"
            )
        org = (parts[0], parts[1], parts[2])

    return {"name": name, "order": order, "cap": cap, "anthropic_org": org}


def check_inventory(resources: Sequence[Resource]) -> None:
    """Reject inventories with ambiguous names, orders or subscriptions.

    Args:
        resources: Every tagged resource found.

    Raises:
        AzureConfigError: On a duplicate name, a duplicate numeric order, or two
            resources in one subscription.
    """
    seen_names: dict[str, Resource] = {}
    seen_orders: dict[int, Resource] = {}
    seen_subs: dict[str, Resource] = {}
    for r in resources:
        prior = seen_names.get(r.name)
        if prior is not None:
            raise AzureConfigError(
                f"llmkit-name {r.name!r} is used by both {prior.account} and {r.account}"
            )
        seen_names[r.name] = r
        if r.order is not None:
            prior = seen_orders.get(r.order)
            if prior is not None:
                raise AzureConfigError(
                    f"llmkit-order {r.order} is used by both {prior.account} and "
                    f"{r.account}"
                )
            seen_orders[r.order] = r
        prior = seen_subs.get(r.subscription_id)
        if prior is not None:
            raise AzureConfigError(
                f"subscription {r.subscription_id} holds two tagged resources: "
                f"{prior.account} and {r.account}"
            )
        seen_subs[r.subscription_id] = r


def resource_to_dict(r: Resource) -> dict[str, Any]:
    """Convert a resource to a JSON-serializable dict.

    Args:
        r: The resource.

    Returns:
        A dict that :func:`resource_from_dict` turns back into ``r``.
    """
    return {
        "name": r.name,
        "account": r.account,
        "subdomain": r.subdomain,
        "subscription_id": r.subscription_id,
        "subscription_name": r.subscription_name,
        "resource_group": r.resource_group,
        "location": r.location,
        "order": r.order,
        "cap": r.cap,
        "anthropic_org": list(r.anthropic_org) if r.anthropic_org else None,
        "deployments": [
            {
                "name": d.name,
                "model": d.model,
                "version": d.version,
                "format": d.format,
                "state": d.state,
            }
            for d in r.deployments
        ],
    }


def resource_from_dict(d: Mapping[str, Any]) -> Resource:
    """Rebuild a resource from :func:`resource_to_dict` output.

    Args:
        d: The dict.

    Returns:
        The resource.
    """
    org = d.get("anthropic_org")
    return Resource(
        name=d["name"],
        account=d["account"],
        subdomain=d["subdomain"],
        subscription_id=d["subscription_id"],
        subscription_name=d["subscription_name"],
        resource_group=d["resource_group"],
        location=d["location"],
        order=d["order"],
        cap=d["cap"],
        anthropic_org=(org[0], org[1], org[2]) if org else None,
        deployments=tuple(Deployment(**x) for x in d.get("deployments", ())),
    )
