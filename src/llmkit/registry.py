"""Review tier: plumbing.

Which provider and wire format serve a model name, and what that model accepts.

Names starting with ``gpt-``, ``gemini-`` or ``claude-`` route by family without a
registry entry, so a newly deployed model works without a release. The registry holds
only models that need more: a deployment name that differs per provider, effort rungs
the model does not serve, or a capability it lacks. There are no aliases.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import UnknownModel, UnsupportedEffort
from .types import LADDER

PROVIDERS = ("foundry", "vertex", "openrouter")

STYLE_LADDERS: dict[str, tuple[str, ...]] = {
    "openai": LADDER,
    "anthropic_effort": LADDER,
    "gemini": ("off", "low", "medium", "high"),
    "chat_reasoning": ("off", "low", "medium", "high"),
    "none": (),
}


@dataclass(frozen=True)
class ModelSpec:
    """What llmkit knows about one model.

    Attributes:
        routes: Provider name to the deployment name on that provider.
        default: The provider used when the caller names none.
        effort_style: A key of :data:`STYLE_LADDERS`.
        efforts: The rungs actually served; ``None`` means the style's full ladder.
        thinking_off: Anthropic only: the thinking type sent for ``off``
            (``disabled`` or ``between_tools``).
        sampling: Whether ``temperature`` and ``top_p`` are accepted.
        forced_tool_choice: Whether ``required`` or a named tool choice is accepted.
        structured_output: Whether a JSON-schema output format is accepted.
        region: A Vertex location this model requires, overriding the default.
        price: USD per million tokens (input, output) for models LiteLLM lacks.
    """

    routes: Mapping[str, str]
    default: str
    effort_style: str
    efforts: tuple[str, ...] | None = None
    thinking_off: str = "disabled"
    sampling: bool = True
    forced_tool_choice: bool = True
    structured_output: bool = True
    region: str | None = None
    price: tuple[float, float] | None = None

    @property
    def ladder(self) -> tuple[str, ...]:
        """Return the rungs this model serves."""
        if self.efforts is not None:
            return self.efforts
        return STYLE_LADDERS[self.effort_style]


@dataclass(frozen=True)
class Route:
    """A resolved model: where it is served and how to talk to it.

    Attributes:
        model: The name the caller passed.
        provider: ``foundry``, ``vertex`` or ``openrouter``.
        transport: ``responses``, ``anthropic``, ``gemini`` or ``chat``.
        deployment: The name sent to the provider.
        spec: The model's capabilities.
    """

    model: str
    provider: str
    transport: str
    deployment: str
    spec: ModelSpec


@dataclass(frozen=True)
class _Family:
    """A name prefix that routes without a registry entry."""

    prefix: str
    transport: str
    providers: tuple[str, ...]
    effort_style: str
    sampling: bool = True


FAMILIES = (
    _Family("gpt-", "responses", ("foundry",), "openai"),
    _Family("gemini-", "gemini", ("vertex",), "gemini"),
    _Family(
        "claude-", "anthropic", ("foundry", "vertex"), "anthropic_effort", sampling=False
    ),
)


def _claude(
    name: str,
    *,
    no_off: bool = False,
    thinking_off: str = "disabled",
    forced_tool_choice: bool = True,
) -> ModelSpec:
    """Build the spec shared by current Claude models.

    Args:
        name: The model name, identical on Foundry and Vertex.
        no_off: The model cannot turn thinking off.
        thinking_off: The thinking type that turns thinking off.
        forced_tool_choice: Whether forced tool choice is accepted.

    Returns:
        The spec.
    """
    return ModelSpec(
        routes={"foundry": name, "vertex": name},
        default="foundry",
        effort_style="anthropic_effort",
        efforts=LADDER[1:] if no_off else None,
        thinking_off=thinking_off,
        sampling=False,
        forced_tool_choice=forced_tool_choice,
    )


MODELS: dict[str, ModelSpec] = {
    "claude-fable-5-1": _claude(
        "claude-fable-5-1", no_off=True, forced_tool_choice=False
    ),
    "claude-opus-5-5": _claude("claude-opus-5-5", no_off=True, forced_tool_choice=False),
    "claude-opus-5": _claude("claude-opus-5"),
    "claude-sonnet-5-5": _claude(
        "claude-sonnet-5-5", thinking_off="between_tools", forced_tool_choice=False
    ),
    "claude-sonnet-5": _claude("claude-sonnet-5"),
    "gemini-3.8-flash": ModelSpec(
        routes={"vertex": "gemini-3.8-flash"},
        default="vertex",
        effort_style="gemini",
        efforts=("off", "low", "high"),
    ),
    "deepseek-v4-pro": ModelSpec(
        routes={"foundry": "DeepSeek-V4-Pro"},
        default="foundry",
        effort_style="chat_reasoning",
    ),
    "glm-5.3": ModelSpec(
        routes={"foundry": "FW-GLM-5.3"},
        default="foundry",
        effort_style="chat_reasoning",
        efforts=("low", "medium", "high"),
    ),
    "kimi-k3": ModelSpec(
        routes={"foundry": "FW-Kimi-K3"}, default="foundry", effort_style="chat_reasoning"
    ),
}


def _family(name: str) -> _Family | None:
    """Return the family whose prefix *name* starts with, if any."""
    return next((f for f in FAMILIES if name.startswith(f.prefix)), None)


def transport_for(name: str) -> str:
    """Return the transport a model name speaks.

    Args:
        name: A model name.

    Returns:
        The family's transport, or ``chat`` for every other model.
    """
    family = _family(name)
    return family.transport if family else "chat"


def resolve(name: str, provider: str | None = None) -> Route:
    """Resolve a model name to a route.

    Args:
        name: The exact model name.
        provider: Force a provider instead of the model's default.

    Returns:
        The route.

    Raises:
        UnknownModel: If the name matches nothing, the provider is unknown, or the
            model has no route on the requested provider.
    """
    if provider is not None and provider not in PROVIDERS:
        raise UnknownModel(
            f"unknown provider {provider!r}; one of {', '.join(PROVIDERS)}"
        )
    family = _family(name)
    spec = MODELS.get(name)
    if spec is None:
        if family is not None:
            spec = ModelSpec(
                routes=dict.fromkeys(family.providers, name),
                default=family.providers[0],
                effort_style=family.effort_style,
                sampling=family.sampling,
            )
        elif provider is not None:
            spec = ModelSpec(
                routes={provider: name}, default=provider, effort_style="chat_reasoning"
            )
        else:
            raise UnknownModel(
                f"{name!r} is not a registered model or a gpt-/gemini-/claude- name; "
                "pass provider= to route another model through chat completions"
            )
    chosen = provider or spec.default
    if chosen not in spec.routes:
        raise UnknownModel(
            f"{name} has no {chosen} route; it is served by {', '.join(spec.routes)}"
        )
    transport = family.transport if family else "chat"
    return Route(name, chosen, transport, spec.routes[chosen], spec)


def check_effort(route: Route, effort: str | None) -> None:
    """Reject an effort rung the route cannot serve.

    Args:
        route: The resolved model.
        effort: The requested rung, or ``None`` for the provider default.

    Raises:
        UnsupportedEffort: If *effort* is off the ladder or not served by the model.
    """
    if effort is None:
        return
    if effort not in LADDER:
        raise UnsupportedEffort(f"{effort!r} is not one of {', '.join(LADDER)}")
    ladder = route.spec.ladder
    if effort not in ladder:
        served = ", ".join(ladder) or "no rungs"
        raise UnsupportedEffort(
            f"{route.model} has no {effort!r} rung; it serves {served}"
        )


def efforts_for(name: str, provider: str | None = None) -> tuple[str, ...]:
    """Return the rungs a model serves.

    Args:
        name: The exact model name.
        provider: The provider, as for :func:`resolve`.

    Returns:
        The rungs in ladder order.
    """
    return resolve(name, provider).spec.ladder
