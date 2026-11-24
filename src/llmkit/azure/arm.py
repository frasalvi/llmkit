"""Review tier: plumbing.

A minimal Azure Resource Manager client: bearer token from the ``az`` CLI, JSON
requests, ``nextLink`` paging and error mapping. Nothing is written to disk.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from llmkit.errors import ArmError, MissingCredential

ARM_BASE = "https://management.azure.com"
GRAPH_API = "2022-10-01"
COGNITIVE_API = "2026-09-15-preview"
COST_API = "2023-11-01"
TAGS_API = "2021-04-01"
TOKEN_REFRESH_MARGIN = timedelta(minutes=5)
MAX_429_RETRIES = 3
DEFAULT_RETRY_AFTER = 5.0
MAX_RETRY_AFTER = 30.0


def _retry_after(resp: httpx.Response) -> float:
    """Read a Retry-After header in seconds, capped.

    Args:
        resp: The throttled response.

    Returns:
        Seconds to wait: the header (at most 30), or 5 when absent or not a number.
    """
    try:
        wait = float(resp.headers["Retry-After"])
    except (KeyError, ValueError):
        return DEFAULT_RETRY_AFTER
    return min(max(wait, 0.0), MAX_RETRY_AFTER)


def az_token() -> tuple[str, datetime]:
    """Fetch an ARM access token from the logged-in ``az`` CLI.

    Returns:
        The token and its expiry (timezone-aware UTC).

    Raises:
        MissingCredential: If ``az`` is missing, not logged in, or answers badly.
    """
    exe = shutil.which("az")
    if exe is None:
        raise MissingCredential("azure", ["the az CLI (install it, then `az login`)"])
    cmd = [exe, "account", "get-access-token", "--resource", ARM_BASE, "--output", "json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise MissingCredential("azure", [f"a working az CLI ({e}); `az login`"]) from e
    if proc.returncode != 0:
        raise MissingCredential(
            "azure", [f"an az login session (run `az login`): {proc.stderr.strip()}"]
        )
    try:
        data = json.loads(proc.stdout)
        token = str(data["accessToken"])
        expires = datetime.fromtimestamp(int(data["expires_on"]), tz=UTC)
    except (ValueError, KeyError, TypeError) as e:
        raise MissingCredential(
            "azure", [f"a readable az token ({e!r}); run `az login`"]
        ) from e
    return token, expires


class Arm:
    """Authenticated JSON client for Azure Resource Manager."""

    def __init__(
        self,
        *,
        token: Callable[[], tuple[str, datetime]] = az_token,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a client.

        Args:
            token: Returns an access token and its expiry.
            transport: Optional httpx transport, for tests.
            timeout: Per-request timeout in seconds.
            sleep: Blocking sleep used between 429 retries, for tests.
        """
        self._sleep = sleep
        self._token_lock = threading.Lock()
        self._token_fn = token
        self._token: str | None = None
        self._expires: datetime | None = None
        self._http = httpx.Client(base_url=ARM_BASE, transport=transport, timeout=timeout)

    def _bearer(self) -> str:
        """Return a token, refetching when within 5 minutes of expiry.

        Returns:
            The access token.
        """
        with self._token_lock:
            now = datetime.now(UTC)
            if (
                self._token is None
                or self._expires is None
                or self._expires - now <= TOKEN_REFRESH_MARGIN
            ):
                self._token, self._expires = self._token_fn()
            return self._token

    def request(
        self, method: str, path: str, *, api_version: str, json: Any = None
    ) -> Any:
        """Send a request and return parsed JSON.

        GET responses carrying a ``value`` list are followed through ``nextLink``
        and merged into one list under ``value``.

        Args:
            method: HTTP method.
            path: Path relative to ``https://management.azure.com``.
            api_version: The ``api-version`` query parameter.
            json: Optional JSON request body.

        Returns:
            The parsed body, or ``{}`` for an empty body.

        Raises:
            ArmError: On a non-2xx status or a transport failure.
        """
        body = self._parse(self._send(method, path, {"api-version": api_version}, json))
        if method.upper() == "GET" and isinstance(body.get("value"), list):
            items = list(body["value"])
            link = body.get("nextLink")
            while link:
                page = self._parse(self._send("GET", link, None, None))
                items.extend(page.get("value", []))
                link = page.get("nextLink")
            body = {**body, "value": items}
            body.pop("nextLink", None)
        return body

    def _send(
        self, method: str, url: str, params: dict[str, str] | None, body: Any
    ) -> httpx.Response:
        """Send a request, retrying HTTP 429 (and only 429) a few times.

        Args:
            method: HTTP method.
            url: Path or absolute URL.
            params: Query parameters, or None when the URL carries them.
            body: Optional JSON body.

        Returns:
            The response.

        Raises:
            ArmError: On a non-2xx status or a transport failure.
        """
        for attempt in range(MAX_429_RETRIES + 1):
            try:
                return self._send_once(method, url, params, body)
            except ArmError as e:
                if e.status != 429 or attempt == MAX_429_RETRIES:
                    raise
                self._sleep(
                    e.retry_after if e.retry_after is not None else DEFAULT_RETRY_AFTER
                )
        raise AssertionError("unreachable")

    def _send_once(
        self, method: str, url: str, params: dict[str, str] | None, body: Any
    ) -> httpx.Response:
        """Send one authenticated request.

        Args:
            method: HTTP method.
            url: Path or absolute URL.
            params: Query parameters, or None when the URL carries them.
            body: Optional JSON body.

        Returns:
            The response.

        Raises:
            ArmError: On a non-2xx status or a transport failure.
        """
        try:
            resp = self._http.request(
                method,
                url,
                params=params,
                json=body,
                headers={"Authorization": f"Bearer {self._bearer()}"},
            )
        except httpx.HTTPError as e:
            raise ArmError(f"{method} {url} failed: {e}") from e
        if resp.status_code >= 300:
            msg = resp.text
            with contextlib.suppress(ValueError, KeyError, TypeError):
                msg = resp.json()["error"]["message"]
            err = ArmError(
                f"{method} {url} returned {resp.status_code}: {msg}",
                status=resp.status_code,
            )
            err.retry_after = _retry_after(resp)
            raise err
        return resp

    @staticmethod
    def _parse(resp: httpx.Response) -> Any:
        """Parse a response body, treating an empty one as ``{}``.

        Args:
            resp: The response.

        Returns:
            Parsed JSON or ``{}``.

        Raises:
            ArmError: If the body is not JSON.
        """
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError as e:
            raise ArmError(f"non-JSON response: {e}", status=resp.status_code) from e
