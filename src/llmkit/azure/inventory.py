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

import httpx

from llmkit.azure.arm import COGNITIVE_API, COST_API, GRAPH_API, TAGS_API, Arm
from llmkit.errors import ArmError, AzureConfigError, SpendUnavailable

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


_GRAPH_QUERY = """resources
| where type =~ 'microsoft.cognitiveservices/accounts' and isnotempty(tags['llmkit-name'])
| join kind=leftouter (resourcecontainers
    | where type =~ 'microsoft.resources/subscriptions'
    | project subscriptionId, subscriptionName=name) on subscriptionId
| project name, subscriptionId, subscriptionName, resourceGroup, location,
          subdomain=tostring(properties.customSubDomainName), tags"""

_COST_BODY = {
    "type": "ActualCost",
    "timeframe": "MonthToDate",
    "dataset": {
        "granularity": "None",
        "aggregation": {"totalCost": {"name": "Cost", "function": "Sum"}},
    },
}


def _account_path(subscription_id: str, resource_group: str, account: str) -> str:
    """Build the ARM path of a Cognitive Services account.

    Args:
        subscription_id: The subscription id.
        resource_group: The resource group name.
        account: The account name.

    Returns:
        The path relative to the ARM base URL.
    """
    return (
        f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}"
        f"/providers/Microsoft.CognitiveServices/accounts/{account}"
    )


def discover(arm: Arm) -> list[Resource]:
    """Find every llmkit-tagged Foundry account and its deployments.

    Args:
        arm: The ARM client.

    Returns:
        The validated inventory.

    Raises:
        AzureConfigError: If tags are malformed or the inventory is ambiguous.
        ArmError: If a request fails.
    """
    body = arm.request(
        "POST",
        "/providers/Microsoft.ResourceGraph/resources",
        api_version=GRAPH_API,
        json={"query": _GRAPH_QUERY},
    )
    out: list[Resource] = []
    for row in body.get("data", []):
        account = row["name"]
        parsed = parse_tags(row.get("tags") or {}, account)
        sub_id = row["subscriptionId"]
        rg = row["resourceGroup"]
        dep_body = arm.request(
            "GET",
            _account_path(sub_id, rg, account) + "/deployments",
            api_version=COGNITIVE_API,
        )
        deployments = []
        for d in dep_body.get("value", []):
            props = d.get("properties", {})
            model = props.get("model", {})
            deployments.append(
                Deployment(
                    name=d["name"],
                    model=model.get("name", ""),
                    version=model.get("version", ""),
                    format=model.get("format", ""),
                    state=props.get("provisioningState", ""),
                )
            )
        out.append(
            Resource(
                name=parsed["name"],
                account=account,
                subdomain=row.get("subdomain") or account,
                subscription_id=sub_id,
                subscription_name=row.get("subscriptionName") or sub_id,
                resource_group=rg,
                location=row.get("location", ""),
                order=parsed["order"],
                cap=parsed["cap"],
                anthropic_org=parsed["anthropic_org"],
                deployments=tuple(deployments),
            )
        )
    check_inventory(out)
    return out


def check_tag_change(
    arm: Arm,
    account: str,
    subscription_id: str,
    resource_group: str,
    tags: Mapping[str, str],
) -> None:
    """Check that giving ``account`` these tags keeps the whole inventory valid.

    Args:
        arm: The ARM client.
        account: The account being tagged.
        subscription_id: Its subscription id.
        resource_group: Its resource group.
        tags: The account's tags after the change (current tags with the new ones).

    Raises:
        AzureConfigError: If the change would duplicate a name or an order, or put
            two tagged resources in one subscription.
    """
    body = arm.request(
        "POST",
        "/providers/Microsoft.ResourceGraph/resources",
        api_version=GRAPH_API,
        json={"query": _GRAPH_QUERY},
    )
    stubs: list[Resource] = []
    seen = False
    for row in body.get("data", []):
        is_target = (
            row["name"].lower() == account.lower()
            and row["subscriptionId"] == subscription_id
        )
        seen = seen or is_target
        stubs.append(
            _stub(
                row["name"],
                row["subscriptionId"],
                row["resourceGroup"],
                tags if is_target else row.get("tags") or {},
            )
        )
    if not seen:
        stubs.append(_stub(account, subscription_id, resource_group, tags))
    check_inventory(stubs)


def _stub(
    account: str, subscription_id: str, resource_group: str, tags: Mapping[str, str]
) -> Resource:
    """Build a deployment-less resource from tags, for cross-resource checks.

    Args:
        account: The account name.
        subscription_id: The subscription id.
        resource_group: The resource group.
        tags: The account's tags.

    Returns:
        A resource carrying only what ``check_inventory`` reads.

    Raises:
        AzureConfigError: If the tags are invalid.
    """
    parsed = parse_tags(tags, account)
    return Resource(
        name=parsed["name"],
        account=account,
        subdomain=account,
        subscription_id=subscription_id,
        subscription_name=subscription_id,
        resource_group=resource_group,
        location="",
        order=parsed["order"],
        cap=parsed["cap"],
        anthropic_org=parsed["anthropic_org"],
    )


def month_spend(
    arm: Arm, subscription_id: str, subscription_name: str, now: datetime
) -> Spend:
    """Read a subscription's month-to-date actual cost.

    Args:
        arm: The ARM client.
        subscription_id: The subscription id.
        subscription_name: Display name, used in errors.
        now: Timestamp recorded on the result.

    Returns:
        The spend in USD; no rows means 0.0.

    Raises:
        SpendUnavailable: If the query fails or the currency is not USD.
    """
    try:
        body = arm.request(
            "POST",
            f"/subscriptions/{subscription_id}/providers/Microsoft.CostManagement/query",
            api_version=COST_API,
            json=_COST_BODY,
        )
    except (ArmError, httpx.HTTPError) as e:
        raise SpendUnavailable(subscription_name, str(e)) from e
    props = body.get("properties", {})
    rows = props.get("rows") or []
    if not rows:
        return Spend(0.0, "USD", now)
    names = [str(c.get("name", "")).lower() for c in props.get("columns", [])]
    try:
        i_cost = next(i for i, n in enumerate(names) if n in ("cost", "pretaxcost"))
        i_cur = names.index("currency")
        amount = sum(float(r[i_cost]) for r in rows)
        currencies = {str(r[i_cur]) for r in rows}
    except (StopIteration, ValueError, IndexError, TypeError) as e:
        raise SpendUnavailable(
            subscription_name, f"unexpected cost response shape: {e!r}"
        ) from e
    if currencies != {"USD"}:
        raise SpendUnavailable(
            subscription_name,
            f"cost is reported in {', '.join(sorted(currencies))}, not USD",
        )
    return Spend(amount, "USD", now)


def account_key(arm: Arm, r: Resource) -> str:
    """Fetch an account's primary key; it is never written to disk.

    Args:
        arm: The ARM client.
        r: The resource.

    Returns:
        ``key1``.

    Raises:
        ArmError: If the request fails or the response has no key.
    """
    body = arm.request(
        "POST",
        _account_path(r.subscription_id, r.resource_group, r.account) + "/listKeys",
        api_version=COGNITIVE_API,
    )
    key = body.get("key1")
    if not key:
        raise ArmError(f"listKeys for {r.account} returned no key1")
    return str(key)


def find_account(arm: Arm, account: str) -> tuple[str, str]:
    """Locate a Cognitive Services account by Azure resource name.

    Args:
        arm: The ARM client.
        account: The Azure resource name, tagged or not.

    Returns:
        ``(subscription_id, resource_group)``.

    Raises:
        AzureConfigError: If no account, or more than one, has that name.
    """
    safe = account.replace("'", "''")
    query = (
        "resources | where type =~ 'microsoft.cognitiveservices/accounts' "
        f"and name =~ '{safe}' | project name, subscriptionId, resourceGroup"
    )
    rows = arm.request(
        "POST",
        "/providers/Microsoft.ResourceGraph/resources",
        api_version=GRAPH_API,
        json={"query": query},
    ).get("data", [])
    if not rows:
        raise AzureConfigError(f"no Azure account named {account!r} is visible")
    if len(rows) > 1:
        raise AzureConfigError(f"more than one Azure account is named {account!r}")
    return rows[0]["subscriptionId"], rows[0]["resourceGroup"]


def get_tags(
    arm: Arm, subscription_id: str, resource_group: str, account: str
) -> dict[str, str]:
    """Read an account's current tags.

    Args:
        arm: The ARM client.
        subscription_id: The subscription id.
        resource_group: The resource group name.
        account: The account name.

    Returns:
        The tags, empty when the account has none.

    Raises:
        ArmError: If the request fails.
    """
    body = arm.request(
        "GET",
        _account_path(subscription_id, resource_group, account)
        + "/providers/Microsoft.Resources/tags/default",
        api_version=TAGS_API,
    )
    tags = body.get("properties", {}).get("tags") or {}
    return {str(k): str(v) for k, v in tags.items()}


def set_tags(
    arm: Arm,
    subscription_id: str,
    resource_group: str,
    account: str,
    tags: Mapping[str, str],
) -> None:
    """Merge tags onto an account, leaving its other tags alone.

    Args:
        arm: The ARM client.
        subscription_id: The subscription id.
        resource_group: The resource group name.
        account: The account name.
        tags: Tags to merge.

    Raises:
        ArmError: If the request fails.
    """
    arm.request(
        "PATCH",
        _account_path(subscription_id, resource_group, account)
        + "/providers/Microsoft.Resources/tags/default",
        api_version=TAGS_API,
        json={"operation": "Merge", "properties": {"tags": dict(tags)}},
    )
