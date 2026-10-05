"""Review tier: plumbing.

Where endpoints and keys come from. One fixed set of variable names is used in every
project. Values in the nearest ``.env`` win over the process environment, so a stray
shell ``export`` cannot silently redirect a run to another resource.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

from .errors import MissingCredential

REQUIRED: dict[str, tuple[str, ...]] = {
    "foundry": ("AZURE_API_KEY", "AZURE_ENDPOINT"),
    "vertex": ("GOOGLE_CLOUD_PROJECT",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def find_dotenv(start: Path) -> Path | None:
    """Find the nearest ``.env`` at or above *start*.

    Args:
        start: The directory to search from.

    Returns:
        The file, or ``None`` when no ancestor has one.
    """
    for folder in (start.resolve(), *start.resolve().parents):
        candidate = folder / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env(
    env_file: Path | None = None,
    *,
    start: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge the process environment with a ``.env``, the file winning.

    Args:
        env_file: A specific file to read. Searched for from *start* when omitted.
        start: Where to start searching; the working directory when omitted.
        environ: The base environment; ``os.environ`` when omitted.

    Returns:
        The merged variables.

    Raises:
        FileNotFoundError: If *env_file* is given and does not exist.
    """
    merged = dict(os.environ if environ is None else environ)
    if env_file is not None:
        if not env_file.is_file():
            raise FileNotFoundError(f"no credential file at {env_file}")
        path: Path | None = env_file
    else:
        path = find_dotenv(start or Path.cwd())
    if path is not None:
        merged.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    return merged


def credentials_for(provider: str, env: Mapping[str, str]) -> dict[str, str]:
    """Pick out the settings a provider needs.

    Args:
        provider: ``foundry``, ``vertex`` or ``openrouter``.
        env: Merged variables from :func:`load_env`.

    Returns:
        The provider's settings.

    Raises:
        MissingCredential: If any required setting is unset or empty.
    """
    missing = [name for name in REQUIRED[provider] if not env.get(name)]
    if missing:
        raise MissingCredential(provider, missing)
    out = {name: env[name] for name in REQUIRED[provider]}
    if provider == "vertex":
        out["GOOGLE_CLOUD_LOCATION"] = env.get("GOOGLE_CLOUD_LOCATION") or "global"
        if env.get("GOOGLE_APPLICATION_CREDENTIALS"):
            out["GOOGLE_APPLICATION_CREDENTIALS"] = env["GOOGLE_APPLICATION_CREDENTIALS"]
    return out
