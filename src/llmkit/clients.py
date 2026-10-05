"""Review tier: plumbing.

Builds the SDK clients a route needs, sync and async, with SDK retries disabled so
retrying happens in one place. Every credential is passed explicitly; no SDK reads
its own ambient variables.

Vertex's OpenAI-compatible endpoint authenticates with a short-lived Google token, so
its clients are rebuilt whenever the token is refreshed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import google.auth
from anthropic import (
    Anthropic,
    AnthropicFoundry,
    AnthropicVertex,
    AsyncAnthropic,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
)
from google import genai
from google.auth import exceptions as auth_exceptions
from google.auth.transport.requests import Request
from google.genai import types as genai_types
from openai import AsyncOpenAI, OpenAI

from .errors import MissingCredential, UnknownModel
from .registry import Route

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class SdkClients(Protocol):
    """A route's SDK clients; read the attributes per call (they may be rebuilt)."""

    @property
    def sync(self) -> Any:
        """The blocking client."""
        ...

    @property
    def async_(self) -> Any:
        """The async client."""
        ...


@dataclass
class StaticClients:
    """Clients that never need rebuilding."""

    sync: Any
    async_: Any


class _Foundry(AnthropicFoundry):
    """AnthropicFoundry that reads none of the ``ANTHROPIC_FOUNDRY_*`` variables."""

    def __init__(self, *, api_key: str, base_url: str, timeout: float) -> None:
        """Build the client from explicit settings only.

        Args:
            api_key: ``AZURE_API_KEY``.
            base_url: The resource's ``/anthropic/`` URL.
            timeout: Per-attempt timeout in seconds.
        """
        Anthropic.__init__(
            self, api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0
        )
        self._azure_ad_token_provider = None


class _AsyncFoundry(AsyncAnthropicFoundry):
    """Async twin of :class:`_Foundry`."""

    def __init__(self, *, api_key: str, base_url: str, timeout: float) -> None:
        """Build the client from explicit settings only.

        Args:
            api_key: ``AZURE_API_KEY``.
            base_url: The resource's ``/anthropic/`` URL.
            timeout: Per-attempt timeout in seconds.
        """
        AsyncAnthropic.__init__(
            self, api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0
        )
        self._azure_ad_token_provider = None


def foundry_base_url(endpoint: str) -> str:
    """Return the OpenAI-compatible base URL of a Foundry resource.

    Args:
        endpoint: ``AZURE_ENDPOINT``.

    Returns:
        The ``/openai/v1/`` URL.
    """
    return endpoint.rstrip("/") + "/openai/v1/"


def foundry_resource(endpoint: str) -> str:
    """Return a Foundry resource name from its endpoint host.

    Args:
        endpoint: ``AZURE_ENDPOINT``.

    Returns:
        The host's first label.

    Raises:
        MissingCredential: If the endpoint is not a URL.
    """
    host = urlparse(endpoint).hostname or ""
    if not host:
        raise MissingCredential(
            "foundry",
            ["AZURE_ENDPOINT (a URL such as https://<resource>.services.ai.azure.com)"],
        )
    return host.split(".")[0]


def vertex_anthropic_base_url(location: str) -> str:
    """Return the base URL of Vertex's Anthropic endpoint, as the Anthropic SDK builds it.

    Passed explicitly so ``ANTHROPIC_VERTEX_BASE_URL`` is never read.

    Args:
        location: A Vertex location, ``global``, ``us`` or ``eu``.

    Returns:
        The base URL.
    """
    if location == "global":
        return "https://aiplatform.googleapis.com/v1"
    if location in ("us", "eu"):
        return f"https://aiplatform.{location}.rep.googleapis.com/v1"
    return f"https://{location}-aiplatform.googleapis.com/v1"


def vertex_openapi_base_url(project: str, location: str) -> str:
    """Return the base URL of Vertex's OpenAI-compatible endpoint.

    Args:
        project: ``GOOGLE_CLOUD_PROJECT``.
        location: A Vertex location, or ``global``.

    Returns:
        The base URL.
    """
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    return f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"


def google_credentials(creds: dict[str, str]) -> Any:
    """Load Google credentials from a key file, else Application Default Credentials.

    Args:
        creds: Vertex settings from :func:`llmkit.credentials.credentials_for`.

    Returns:
        A google-auth credentials object.

    Raises:
        MissingCredential: If no credentials can be found.
    """
    path = creds.get("GOOGLE_APPLICATION_CREDENTIALS")
    try:
        if path:
            credentials, _ = google.auth.load_credentials_from_file(
                path, scopes=[CLOUD_SCOPE]
            )
        else:
            credentials, _ = google.auth.default(scopes=[CLOUD_SCOPE])
    except auth_exceptions.DefaultCredentialsError as exc:
        raise MissingCredential(
            "vertex",
            [
                "Application Default Credentials (run: gcloud auth application-default login)"
            ],
        ) from exc
    return credentials


class VertexChatClients:
    """OpenAI clients for Vertex's OpenAI-compatible endpoint, rebuilt on token refresh."""

    def __init__(self, credentials: Any, base_url: str, timeout: float) -> None:
        """Hold the credentials; clients are built on first use.

        Args:
            credentials: google-auth credentials.
            base_url: From :func:`vertex_openapi_base_url`.
            timeout: Per-attempt timeout in seconds.
        """
        self._credentials = credentials
        self._base_url = base_url
        self._timeout = timeout
        self._sync: tuple[str, OpenAI] | None = None
        self._async: tuple[str, AsyncOpenAI] | None = None

    def _token(self) -> str:
        """Return a valid access token, refreshing it when needed."""
        if not self._credentials.valid:
            self._credentials.refresh(Request())
        return str(self._credentials.token)

    @property
    def sync(self) -> OpenAI:
        """Return a blocking client carrying a current token."""
        token = self._token()
        if self._sync is None or self._sync[0] != token:
            self._sync = (
                token,
                OpenAI(
                    base_url=self._base_url,
                    api_key=token,
                    timeout=self._timeout,
                    max_retries=0,
                ),
            )
        return self._sync[1]

    @property
    def async_(self) -> AsyncOpenAI:
        """Return an async client carrying a current token."""
        token = self._token()
        if self._async is None or self._async[0] != token:
            self._async = (
                token,
                AsyncOpenAI(
                    base_url=self._base_url,
                    api_key=token,
                    timeout=self._timeout,
                    max_retries=0,
                ),
            )
        return self._async[1]


def make_clients(route: Route, creds: dict[str, str], timeout: float) -> SdkClients:
    """Build the SDK clients a route needs.

    Args:
        route: The resolved model.
        creds: The provider's settings.
        timeout: Per-attempt timeout in seconds.

    Returns:
        The clients.

    Raises:
        MissingCredential: If Vertex credentials cannot be loaded.
        UnknownModel: If no client exists for the route's provider and transport.
    """
    provider, transport = route.provider, route.transport
    if provider == "foundry":
        key, endpoint = creds["AZURE_API_KEY"], creds["AZURE_ENDPOINT"]
        if transport == "anthropic":
            # base_url, not resource=: the SDK reads ANTHROPIC_FOUNDRY_BASE_URL otherwise.
            base = (
                f"https://{foundry_resource(endpoint)}.services.ai.azure.com/anthropic/"
            )
            return StaticClients(
                _Foundry(api_key=key, base_url=base, timeout=timeout),
                _AsyncFoundry(api_key=key, base_url=base, timeout=timeout),
            )
        if transport in ("responses", "chat"):
            base = foundry_base_url(endpoint)
            return StaticClients(
                OpenAI(base_url=base, api_key=key, timeout=timeout, max_retries=0),
                AsyncOpenAI(base_url=base, api_key=key, timeout=timeout, max_retries=0),
            )
    if provider == "openrouter":
        key = creds["OPENROUTER_API_KEY"]
        return StaticClients(
            OpenAI(
                base_url=OPENROUTER_BASE_URL, api_key=key, timeout=timeout, max_retries=0
            ),
            AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL, api_key=key, timeout=timeout, max_retries=0
            ),
        )
    if provider == "vertex":
        project = creds["GOOGLE_CLOUD_PROJECT"]
        location = route.spec.region or creds["GOOGLE_CLOUD_LOCATION"]
        credentials = google_credentials(creds)
        if transport == "anthropic":
            return StaticClients(
                AnthropicVertex(
                    project_id=project,
                    region=location,
                    base_url=vertex_anthropic_base_url(location),
                    credentials=credentials,
                    timeout=timeout,
                    max_retries=0,
                ),
                AsyncAnthropicVertex(
                    project_id=project,
                    region=location,
                    base_url=vertex_anthropic_base_url(location),
                    credentials=credentials,
                    timeout=timeout,
                    max_retries=0,
                ),
            )
        if transport == "gemini":
            client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                credentials=credentials,
                http_options=genai_types.HttpOptions(timeout=int(timeout * 1000)),
            )
            return StaticClients(client, client.aio)
        if transport == "chat":
            return VertexChatClients(
                credentials, vertex_openapi_base_url(project, location), timeout
            )
    raise UnknownModel(f"no client for {transport} on {provider}")
