"""Review tier: plumbing.

Turns a Foundry route into the key, endpoint and resource name a client is built
from. Precedence: explicit arguments, then a pinned resource, then an AZURE_* pair in
the environment, then tag order and monthly cap.
"""

from __future__ import annotations

import sys
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

from llmkit.azure.source import AzureSource
from llmkit.clients import foundry_resource
from llmkit.errors import MissingCredential
from llmkit.registry import Route
from llmkit.selection import choose

_WARNED: set[tuple[str, str]] = set()
_WARNED_LOCK = threading.Lock()


class AzureEnvKeyWarning(UserWarning):
    """A key and endpoint from the environment bypass tag routing and spend caps."""


class AzureRoutingWarning(UserWarning):
    """Tag routing skipped a resource, or every resource is over its cap."""


@dataclass(frozen=True)
class FoundryTarget:
    """What a Foundry client is built from."""

    creds: dict[str, str]
    resource: str


def _warn_env_key(
    env_source: str, endpoint: str, azure: AzureSource, pinned: str | None
) -> None:
    """Warn that an AZURE_* pair from the environment is present.

    Only the cached inventory is consulted, never the network.

    Args:
        env_source: Where the pair was read from.
        endpoint: The endpoint in the environment.
        azure: Source of the cached inventory.
        pinned: The pinned resource name that overrides the pair, if any.
    """
    try:
        name = foundry_resource(endpoint)
    except MissingCredential:
        name = endpoint or "(no endpoint)"
    note = ""
    for r in azure.cached_resources() or []:
        if r.subdomain == name and r.order is None:
            note = " (tagged off, so tag routing would never pick it)"
            break
    if pinned is not None:
        action = f"; they are ignored because resource={pinned!r} was requested"
    else:
        action = "; this bypasses tag routing and spend caps"
    message = (
        f"AZURE_API_KEY/AZURE_ENDPOINT from {env_source} point at resource "
        f"{name}{note}{action}. To route by Azure tags, delete AZURE_API_KEY and "
        f"AZURE_ENDPOINT from {env_source}."
    )
    # attribute the warning to the caller of LLM(...), with a fresh registry so the
    # default filter's once-per-location suppression never applies
    try:
        frame = sys._getframe(3)
        filename, lineno = frame.f_code.co_filename, frame.f_lineno
        module = frame.f_globals.get("__name__", "<unknown>")
    except ValueError:
        filename, lineno, module = "<unknown>", 0, "<unknown>"
    warnings.warn_explicit(
        message,
        AzureEnvKeyWarning,
        filename,
        lineno,
        module=module,
        registry=None,
    )


def foundry_target(
    route: Route,
    *,
    env: Mapping[str, str],
    env_source: str | None,
    azure_key: str | None,
    azure_endpoint: str | None,
    resource: str | None,
    azure: AzureSource,
) -> FoundryTarget:
    """Resolve the key, endpoint and resource name for a Foundry route.

    Args:
        route: The resolved route; ``route.deployment`` is what must exist.
        env: Merged environment variables.
        env_source: Where AZURE_* in ``env`` came from (a file path or "process
            environment"), or None when ``env`` was injected; None silences the
            env-pair warning.
        azure_key: Explicit key; requires ``azure_endpoint``.
        azure_endpoint: Explicit endpoint; requires ``azure_key``.
        resource: A llmkit resource name to pin; with explicit arguments, just the
            name recorded.
        azure: Inventory, spend and key source.

    Returns:
        The credentials ``make_clients`` reads, and the resource name.

    Raises:
        MissingCredential: If only half of a key/endpoint pair is given.
        AzureConfigError: If ``resource`` names no known resource.
        UnknownModel: If no eligible resource serves the deployment.
        SpendUnavailable: If a spend figure is needed and cannot be had.
    """
    # explicit arguments
    if azure_key or azure_endpoint:
        if not azure_key or not azure_endpoint:
            missing = "azure_endpoint" if azure_key else "azure_key"
            raise MissingCredential("foundry", [missing])
        return FoundryTarget(
            {"AZURE_API_KEY": azure_key, "AZURE_ENDPOINT": azure_endpoint},
            resource or foundry_resource(azure_endpoint),
        )

    env_key = env.get("AZURE_API_KEY") or ""
    env_endpoint = env.get("AZURE_ENDPOINT") or ""

    # AZURE_* pair, unless a resource is pinned
    if (env_key or env_endpoint) and resource is None:
        missing_vars = [
            n
            for n, v in (("AZURE_API_KEY", env_key), ("AZURE_ENDPOINT", env_endpoint))
            if not v
        ]
        if missing_vars:
            raise MissingCredential("foundry", missing_vars)
        if env_source is not None:
            _warn_env_key(env_source, env_endpoint, azure, None)
        return FoundryTarget(
            {"AZURE_API_KEY": env_key, "AZURE_ENDPOINT": env_endpoint},
            foundry_resource(env_endpoint),
        )
    if (env_key or env_endpoint) and env_source is not None:
        _warn_env_key(env_source, env_endpoint, azure, resource)

    # tag order and cap, or the pinned resource
    choice = choose(azure.resources(), route.deployment, azure.spend, pin=resource)
    chosen = choice.resource
    if choice.warning:
        with _WARNED_LOCK:
            fresh = (route.deployment, chosen.name) not in _WARNED
            _WARNED.add((route.deployment, chosen.name))
        if fresh:
            warnings.warn(choice.warning, AzureRoutingWarning, stacklevel=3)
    return FoundryTarget(
        {"AZURE_API_KEY": azure.key(chosen), "AZURE_ENDPOINT": chosen.endpoint},
        chosen.name,
    )
