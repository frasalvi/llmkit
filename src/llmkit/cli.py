"""Review tier: plumbing.

The ``llmkit azure`` command line: ``status``, ``deploy`` and ``tag``. Nothing it
prints includes an API key, and it never fetches one.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence

from llmkit.azure.deploy import deploy
from llmkit.azure.inventory import (
    Resource,
    Spend,
    find_account,
    get_tags,
    parse_tags,
    set_tags,
)
from llmkit.azure.source import Azure, default_azure
from llmkit.errors import AzureConfigError, LLMKitError, SpendUnavailable

READY = "✓"
NOT_READY = "!"
ABSENT = "-"


def _ordered(resources: Sequence[Resource]) -> list[Resource]:
    """Sort resources by order, with ``off`` ones last.

    Args:
        resources: The inventory.

    Returns:
        The resources in spending order, then the off ones by name.
    """
    on = sorted((r for r in resources if r.order is not None), key=lambda r: r.order or 0)
    off = sorted((r for r in resources if r.order is None), key=lambda r: r.name)
    return on + off


def _spend_text(r: Resource, spend: Spend | Exception | None) -> str:
    """Format one resource's spend against its cap.

    Args:
        r: The resource.
        spend: The subscription's spend, the error that prevented it, or None.

    Returns:
        A short cell such as ``$313.88 / $450 (as of 09:05 UTC)``.
    """
    if isinstance(spend, SpendUnavailable):
        return f"spend unavailable ({spend.reason})"
    if isinstance(spend, Exception):
        return f"spend unavailable ({spend})"
    if spend is None:
        return "spend unknown"
    cap = f" / ${r.cap:.0f}" if r.cap is not None else ""
    return f"${spend.amount:.2f}{cap} (as of {spend.fetched:%H:%M} UTC)"


def status_text(
    resources: Sequence[Resource], spends: Mapping[str, Spend | Exception]
) -> str:
    """Render the status table and the deployment matrix.

    Args:
        resources: The inventory.
        spends: Each subscription id's spend, or the error that prevented it.

    Returns:
        The text to print.
    """
    rows = _ordered(resources)
    lines = ["Spend is month-to-date; Azure billing figures lag by hours.", ""]
    width = max([len(r.name) for r in rows] + [4])
    for r in rows:
        order = "off" if r.order is None else str(r.order)
        lines.append(
            f"{r.name:<{width}}  {order:<3}  "
            f"{_spend_text(r, spends.get(r.subscription_id)):<44}  "
            f"{len(r.deployments)} deployments"
        )

    # model x resource matrix
    names = sorted({d.name for r in rows for d in r.deployments})
    if names:
        lines += ["", "Deployments (✓ ready, ! not ready, - absent):"]
        label = max(len(n) for n in names)
        lines.append(f"{'':<{label}}  " + "  ".join(r.name for r in rows))
        for n in names:
            cells = []
            for r in rows:
                d = r.deployment(n)
                mark = ABSENT if d is None else READY if d.ready else NOT_READY
                cells.append(mark.center(len(r.name)))
            lines.append(f"{n:<{label}}  " + "  ".join(cells))

    # deployments whose name differs from their model
    odd = [
        f"  {r.name}: deployment {d.name} runs model {d.model}"
        for r in rows
        for d in r.deployments
        if d.name != d.model
    ]
    if odd:
        lines += ["", "Check:", *odd]
    return "\n".join(lines) + "\n"


def _status(azure: Azure, args: argparse.Namespace) -> int:
    """Print the status table.

    Args:
        azure: The Azure facade.
        args: Parsed arguments.

    Returns:
        The exit code.
    """
    resources = azure.resources(refresh=args.refresh)
    spends: dict[str, Spend | Exception] = {}
    for r in resources:
        if r.subscription_id in spends:
            continue
        try:
            spends[r.subscription_id] = azure.spend(r, refresh=args.refresh)
        except SpendUnavailable as e:
            spends[r.subscription_id] = e
    print(status_text(resources, spends), end="")
    return 0


def _deploy(azure: Azure, args: argparse.Namespace) -> int:
    """Deploy a model everywhere in order and print the outcomes.

    Args:
        azure: The Azure facade.
        args: Parsed arguments.

    Returns:
        0 when nothing failed, otherwise 1.
    """
    resources = azure.resources(refresh=True)
    outcomes = deploy(
        azure.arm,
        resources,
        args.model,
        version=args.version,
        capacity=args.capacity,
        dry_run=args.dry_run,
    )
    for o in outcomes:
        print(f"{o.resource}: {o.status} - {o.detail}")
    if not args.dry_run:
        azure.invalidate()
    return 1 if any(o.status == "failed" for o in outcomes) else 0


def _tag(azure: Azure, args: argparse.Namespace) -> int:
    """Validate the merged tag set, then merge the given tags onto the account.

    Args:
        azure: The Azure facade.
        args: Parsed arguments.

    Returns:
        The exit code: 2 for an invalid tag set, 0 on success.

    Raises:
        AzureConfigError: If the merged tags are invalid or the account is unknown.
        ArmError: If Azure rejects a request.
    """
    given = {
        key: value
        for key, value in (
            ("llmkit-name", args.name),
            ("llmkit-order", args.order),
            ("llmkit-cap", args.cap),
            ("llmkit-anthropic-org", args.anthropic_org),
        )
        if value is not None
    }
    if not given:
        raise AzureConfigError(
            "nothing to tag: pass --name, --order, --cap or --anthropic-org"
        )
    sub, rg = find_account(azure.arm, args.account)
    merged = {**get_tags(azure.arm, sub, rg, args.account), **given}
    if "llmkit-name" not in merged:
        raise AzureConfigError(f"{args.account} has no llmkit-name yet; pass --name")
    parse_tags(merged, args.account)
    set_tags(azure.arm, sub, rg, args.account, given)
    azure.invalidate()
    print(f"{args.account}: tagged " + ", ".join(f"{k}={v}" for k, v in given.items()))
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser for ``llmkit azure ...``.
    """
    parser = argparse.ArgumentParser(prog="llmkit")
    top = parser.add_subparsers(dest="group", required=True)
    group = top.add_parser("azure", help="manage Azure Foundry resources")
    cmds = group.add_subparsers(dest="command", required=True)

    status = cmds.add_parser("status", help="show spend and deployments per resource")
    status.add_argument("--refresh", action="store_true", help="bypass the caches")

    dep = cmds.add_parser("deploy", help="deploy a model to every in-order resource")
    dep.add_argument("model")
    dep.add_argument("--version")
    dep.add_argument("--capacity", type=int)
    dep.add_argument("--dry-run", action="store_true")

    tag = cmds.add_parser("tag", help="set llmkit tags on an Azure account")
    tag.add_argument("account", help="the Azure resource name")
    tag.add_argument("--name")
    tag.add_argument("--order", help="1, 2, ... or off")
    tag.add_argument("--cap", help="monthly USD cap of the subscription")
    tag.add_argument("--anthropic-org", help='"Org|Industry|CC"')
    return parser


def main(argv: list[str] | None = None, *, azure: Azure | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Arguments after the program name; ``sys.argv`` when omitted.
        azure: The Azure facade, for tests; the process-wide one when omitted.

    Returns:
        The exit code: 0 on success, 1 on a failure, 2 on bad usage or tags.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:
        if e.code in (0, None):
            raise
        return int(e.code) if isinstance(e.code, int) else 2
    handlers = {"status": _status, "deploy": _deploy, "tag": _tag}
    try:
        return handlers[args.command](azure or default_azure(), args)
    except AzureConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except LLMKitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
