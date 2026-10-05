import pytest
from anthropic import AnthropicFoundry, AsyncAnthropicFoundry
from openai import AsyncOpenAI, OpenAI

from llmkit.clients import (
    VertexChatClients,
    foundry_base_url,
    foundry_resource,
    make_clients,
    vertex_openapi_base_url,
)
from llmkit.errors import MissingCredential
from llmkit.registry import resolve

FOUNDRY = {"AZURE_API_KEY": "k", "AZURE_ENDPOINT": "https://myres.services.ai.azure.com/"}


def test_url_helpers():
    assert foundry_base_url("https://r.services.ai.azure.com/") == (
        "https://r.services.ai.azure.com/openai/v1/"
    )
    assert foundry_resource("https://myres.services.ai.azure.com") == "myres"
    with pytest.raises(MissingCredential):
        foundry_resource("not a url")
    assert vertex_openapi_base_url("p", "global") == (
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi"
    )
    assert vertex_openapi_base_url("p", "us-east5").startswith(
        "https://us-east5-aiplatform.googleapis.com/"
    )


def test_foundry_clients_disable_sdk_retries():
    clients = make_clients(resolve("gpt-5.6-sol"), FOUNDRY, 30.0)
    assert isinstance(clients.sync, OpenAI) and isinstance(clients.async_, AsyncOpenAI)
    assert clients.sync.max_retries == 0
    assert str(clients.sync.base_url) == "https://myres.services.ai.azure.com/openai/v1/"
    claude = make_clients(resolve("claude-opus-5"), FOUNDRY, 30.0)
    assert isinstance(claude.sync, AnthropicFoundry)
    assert isinstance(claude.async_, AsyncAnthropicFoundry)
    assert claude.sync.max_retries == 0


def test_openrouter_clients():
    clients = make_clients(
        resolve("z-ai/glm-5.2", "openrouter"), {"OPENROUTER_API_KEY": "o"}, 30.0
    )
    assert str(clients.sync.base_url).startswith("https://openrouter.ai/api/v1")


def test_vertex_without_adc_names_the_fix(monkeypatch):
    import google.auth
    from google.auth import exceptions

    def no_adc(*args, **kwargs):
        raise exceptions.DefaultCredentialsError("none")

    monkeypatch.setattr(google.auth, "default", no_adc)
    with pytest.raises(MissingCredential, match="application-default login"):
        make_clients(
            resolve("gemini-3.8-flash"),
            {"GOOGLE_CLOUD_PROJECT": "p", "GOOGLE_CLOUD_LOCATION": "global"},
            30.0,
        )


class FakeCredentials:
    def __init__(self):
        self.valid = False
        self.token = None
        self.refreshes = 0

    def refresh(self, request):
        self.refreshes += 1
        self.token = f"tok{self.refreshes}"
        self.valid = True


def test_vertex_chat_clients_refresh_token_lazily():
    creds = FakeCredentials()
    clients = VertexChatClients(creds, vertex_openapi_base_url("p", "global"), 30.0)
    assert creds.refreshes == 0
    first = clients.sync
    assert first.api_key == "tok1" and first.max_retries == 0
    assert clients.sync is first
    creds.valid = False
    second = clients.sync
    assert second is not first and second.api_key == "tok2"
    assert clients.async_.api_key == "tok2"


def test_anthropic_clients_ignore_ambient_base_urls(monkeypatch):
    import llmkit.clients as clients_module

    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://elsewhere.test/anthropic/")
    monkeypatch.setenv("ANTHROPIC_VERTEX_BASE_URL", "https://elsewhere.test/v1")
    foundry = make_clients(resolve("claude-opus-5"), FOUNDRY, 30.0)
    assert str(foundry.sync.base_url) == "https://myres.services.ai.azure.com/anthropic/"
    assert (
        str(foundry.async_.base_url) == "https://myres.services.ai.azure.com/anthropic/"
    )

    monkeypatch.setattr(
        clients_module, "google_credentials", lambda creds: FakeCredentials()
    )
    env = {"GOOGLE_CLOUD_PROJECT": "p", "GOOGLE_CLOUD_LOCATION": "global"}
    vertex = make_clients(resolve("claude-opus-5", "vertex"), env, 30.0)
    assert str(vertex.sync.base_url).startswith("https://aiplatform.googleapis.com/v1")
    assert str(vertex.async_.base_url).startswith("https://aiplatform.googleapis.com/v1")


def test_vertex_anthropic_base_url_mirrors_the_sdk():
    from llmkit.clients import vertex_anthropic_base_url

    assert vertex_anthropic_base_url("global") == "https://aiplatform.googleapis.com/v1"
    assert (
        vertex_anthropic_base_url("us") == "https://aiplatform.us.rep.googleapis.com/v1"
    )
    assert (
        vertex_anthropic_base_url("eu") == "https://aiplatform.eu.rep.googleapis.com/v1"
    )
    assert (
        vertex_anthropic_base_url("us-east5")
        == "https://us-east5-aiplatform.googleapis.com/v1"
    )
