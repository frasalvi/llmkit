import pytest

from llmkit.errors import UnknownModel, UnsupportedEffort
from llmkit.registry import MODELS, check_effort, efforts_for, resolve, transport_for


def test_family_rules_route_unregistered_names():
    r = resolve("gpt-9-new")
    assert (r.provider, r.transport, r.deployment) == (
        "foundry",
        "responses",
        "gpt-9-new",
    )
    assert resolve("gemini-9-pro").transport == "gemini"
    assert resolve("gemini-9-pro").provider == "vertex"
    claude = resolve("claude-new-1")
    assert (claude.provider, claude.transport) == ("foundry", "anthropic")
    assert claude.spec.sampling is False


def test_claude_provider_override():
    r = resolve("claude-opus-5", provider="vertex")
    assert (r.provider, r.transport, r.deployment) == (
        "vertex",
        "anthropic",
        "claude-opus-5",
    )


def test_aliases_are_refused():
    with pytest.raises(UnknownModel, match="provider="):
        resolve("opus")


def test_other_models_need_a_provider_or_registry_entry():
    with pytest.raises(UnknownModel):
        resolve("some-model")
    r = resolve("moonshotai/kimi-k9", provider="openrouter")
    assert (r.transport, r.deployment) == ("chat", "moonshotai/kimi-k9")


def test_registered_deployment_name_differs_from_model_name():
    r = resolve("glm-5.3")
    assert (r.provider, r.transport, r.deployment) == ("foundry", "chat", "FW-GLM-5.3")


def test_missing_route_and_unknown_provider():
    with pytest.raises(UnknownModel, match="foundry"):
        resolve("gemini-3.8-flash", provider="foundry")
    with pytest.raises(UnknownModel, match="unknown provider"):
        resolve("gpt-5.6-sol", provider="azure")
    with pytest.raises(UnknownModel):
        resolve("gpt-5.6-sol", provider="openrouter")


def test_transport_for():
    assert transport_for("gpt-x") == "responses"
    assert transport_for("claude-x") == "anthropic"
    assert transport_for("gemini-x") == "gemini"
    assert transport_for("deepseek-x") == "chat"


def test_effort_ladder_checks():
    check_effort(resolve("claude-opus-5"), None)
    check_effort(resolve("claude-opus-5"), "off")
    with pytest.raises(UnsupportedEffort, match="claude-fable-5-1"):
        check_effort(resolve("claude-fable-5-1"), "off")
    with pytest.raises(UnsupportedEffort):
        check_effort(resolve("gemini-3.8-flash"), "medium")
    with pytest.raises(UnsupportedEffort, match="not one of"):
        check_effort(resolve("gpt-5.6-sol"), "xhigh")
    assert efforts_for("gemini-3.8-flash") == ("off", "low", "high")
    assert efforts_for("gpt-anything") == ("off", "low", "medium", "high", "max")


def test_registry_entries_are_consistent():
    for name, spec in MODELS.items():
        assert spec.default in spec.routes, name
        assert set(spec.ladder) <= {"off", "low", "medium", "high", "max"}, name
