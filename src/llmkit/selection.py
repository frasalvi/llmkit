"""Review tier: plumbing.

Pure choice of which Foundry resource serves a deployment: first in order that has
the deployment ready and is under its monthly cap, or the one the caller pinned.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from llmkit.azure.inventory import Resource, Spend
from llmkit.errors import AzureConfigError, UnknownModel


@dataclass(frozen=True)
class Choice:
    """The selected resource and, when it is not the plain first choice, why."""

    resource: Resource
    warning: str | None


def _over_cap(r: Resource, s: Spend) -> bool:
    """Whether spend has reached the resource's cap.

    Args:
        r: The resource.
        s: Its subscription's spend.

    Returns:
        True when the amount is at or above the cap.
    """
    return r.cap is not None and s.amount >= r.cap


def _cap_reason(r: Resource, s: Spend) -> str:
    """Describe an over-cap subscription.

    Args:
        r: The resource.
        s: Its subscription's spend.

    Returns:
        ``$amount/$cap (fetched HH:MM UTC)``.
    """
    return f"${s.amount:.0f}/${r.cap or 0:.0f} (fetched {s.fetched:%H:%M} UTC)"


def _deploy_hint(deployment: str) -> str:
    """Tell the user how to create a missing deployment.

    Args:
        deployment: The deployment name.

    Returns:
        A short hint.
    """
    return f"run `llmkit azure deploy {deployment}` to create it"


def choose(
    resources: Sequence[Resource],
    deployment: str,
    spend: Callable[[Resource], Spend],
    *,
    pin: str | None = None,
) -> Choice:
    """Pick the resource that serves ``deployment``.

    Args:
        resources: The inventory.
        deployment: The deployment name to serve (matched by name, not model).
        spend: Returns a resource's month-to-date spend; called lazily, only for
            resources with a ready deployment that are being examined.
        pin: A llmkit name to use regardless of order (even ``off``).

    Returns:
        The choice; ``warning`` is set when earlier candidates were skipped, all
        were over cap, or a pinned resource is over cap.

    Raises:
        AzureConfigError: If ``pin`` names no resource.
        UnknownModel: If no eligible resource has a ready deployment.
        SpendUnavailable: If ``spend`` raises it.
    """
    if pin is not None:
        return _pinned(resources, deployment, spend, pin)

    # walk the ordered candidates, recording why each is skipped
    ordered = sorted(
        (r for r in resources if r.order is not None), key=lambda r: r.order or 0
    )
    skipped: list[str] = []
    over: list[Resource] = []
    for r in ordered:
        d = r.deployment(deployment)
        if d is None:
            skipped.append(f"{r.name}: no {deployment} deployment")
            continue
        if not d.ready:
            skipped.append(f"{r.name}: {deployment} is {d.state}")
            continue
        s = spend(r)
        if not _over_cap(r, s):
            if not skipped:
                return Choice(r, None)
            return Choice(r, f"using {r.name}; " + "; ".join(skipped))
        over.append(r)
        skipped.append(f"{r.name}: {_cap_reason(r, s)}")

    # every examined candidate was over cap: fall back to the last one
    if over:
        last = over[-1]
        return Choice(
            last,
            f"all subscriptions over cap; using {last.name}; " + "; ".join(skipped),
        )

    off = [r.name for r in resources if r.order is None and r.deployment(deployment)]
    msg = f"no ready deployment {deployment!r} on any ordered resource; "
    if off:
        msg += f"{', '.join(off)} (off) has it: pin one to use it, or "
    msg += _deploy_hint(deployment)
    raise UnknownModel(msg)


def _pinned(
    resources: Sequence[Resource],
    deployment: str,
    spend: Callable[[Resource], Spend],
    pin: str,
) -> Choice:
    """Resolve a pinned resource.

    Args:
        resources: The inventory.
        deployment: The deployment name.
        spend: Spend lookup.
        pin: The llmkit name.

    Returns:
        The pinned resource, warning only when it is over cap.

    Raises:
        AzureConfigError: If no resource has that name.
        UnknownModel: If it lacks a ready deployment.
    """
    r = next((x for x in resources if x.name == pin), None)
    if r is None:
        known = ", ".join(x.name for x in resources) or "none"
        raise AzureConfigError(f"no resource named {pin!r}; known names: {known}")
    d = r.deployment(deployment)
    if d is None:
        raise UnknownModel(
            f"{pin} has no {deployment} deployment; {_deploy_hint(deployment)}"
        )
    if not d.ready:
        raise UnknownModel(f"{pin}: {deployment} is {d.state}, not ready")
    if r.cap is None:
        return Choice(r, None)
    s = spend(r)
    if _over_cap(r, s):
        return Choice(r, f"using {pin} (pinned); over cap: {_cap_reason(r, s)}")
    return Choice(r, None)
