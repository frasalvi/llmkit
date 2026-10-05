"""Review tier: plumbing.

The value types every module passes around. No provider SDK is imported here, so
callers, call records and tests can use this vocabulary without an SDK import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

LADDER = ("off", "low", "medium", "high", "max")

STOP_END = "end"
STOP_MAX_TOKENS = "max_tokens"
STOP_TOOL_USE = "tool_use"
STOP_REFUSAL = "refusal"
STOP_CONTENT_FILTER = "content_filter"


@dataclass(frozen=True)
class Text:
    """A text part of a message."""

    text: str


@dataclass(frozen=True)
class Image:
    """An inline image part of a message.

    Attributes:
        data: Raw image bytes.
        media_type: MIME type such as ``image/png``.
    """

    data: bytes
    media_type: str


Part = Text | Image


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation the model asked for.

    Attributes:
        id: Provider call id, echoed back with the tool result.
        name: The tool's name.
        arguments: Parsed arguments; empty when the model emitted invalid JSON.
        raw: The arguments exactly as the provider sent them.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    raw: str = ""


@dataclass(frozen=True)
class ProviderState:
    """Provider data replayed only to the transport that produced it.

    Attributes:
        transport: Name of the transport that produced *data*.
        data: Raw output (GPT reasoning items, Claude thinking blocks, Gemini parts).
    """

    transport: str
    data: Any


@dataclass
class Message:
    """One turn of a conversation.

    Attributes:
        role: ``user``, ``assistant`` or ``tool``.
        content: A string or a list of :class:`Text` and :class:`Image` parts.
        tool_calls: Tool calls on an assistant turn.
        tool_call_id: On a tool turn, the call this result answers.
        tool_name: On a tool turn, the tool's name (Gemini needs it).
        is_error: On a tool turn, whether the tool failed.
        provider_state: Opaque state from the producing transport.
    """

    role: str
    content: str | list[Part] = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    tool_name: str = ""
    is_error: bool = False
    provider_state: ProviderState | None = None

    @classmethod
    def user(cls, content: str | list[Part]) -> Message:
        """Build a user turn.

        Args:
            content: Text or parts.

        Returns:
            The message.
        """
        return cls(ROLE_USER, content)

    @classmethod
    def assistant(cls, content: str | list[Part] = "") -> Message:
        """Build an assistant turn with no provider state.

        Args:
            content: Text or parts.

        Returns:
            The message.
        """
        return cls(ROLE_ASSISTANT, content)

    @classmethod
    def tool_result(
        cls, call: ToolCall, content: str, *, is_error: bool = False
    ) -> Message:
        """Build the turn that answers a tool call.

        Args:
            call: The call being answered.
            content: The tool's output as text.
            is_error: Whether the tool failed.

        Returns:
            The message.
        """
        return cls(
            ROLE_TOOL,
            content,
            tool_call_id=call.id,
            tool_name=call.name,
            is_error=is_error,
        )


def parts(content: str | list[Part]) -> list[Part]:
    """Normalise message content to a list of parts.

    Args:
        content: A string or a list of parts.

    Returns:
        The parts; an empty string gives an empty list.
    """
    if isinstance(content, str):
        return [Text(content)] if content else []
    return list(content)


def text_of(content: str | list[Part]) -> str:
    """Concatenate the text parts of message content.

    Args:
        content: A string or a list of parts.

    Returns:
        The text, images dropped.
    """
    return "".join(p.text for p in parts(content) if isinstance(p, Text))


@dataclass(frozen=True)
class Tool:
    """A function the model may call.

    Attributes:
        name: The tool's name.
        description: What it does, for the model.
        parameters: A JSON schema or a Pydantic model describing the arguments.
    """

    name: str
    description: str
    parameters: dict[str, Any] | type[BaseModel]


@dataclass
class Usage:
    """What one call consumed.

    Attributes:
        model: The model the provider reported, or the requested one.
        provider: The provider that served the call.
        effort: The rung requested, or ``None`` for the provider default.
        input_tokens: Prompt tokens billed at the full input rate.
        output_tokens: Completion tokens, thinking included.
        cached_input_tokens: Prompt tokens served from a cache.
        cache_write_tokens: Prompt tokens written to a cache (Anthropic).
        latency_ms: Wall-clock duration including retries.
        cost: USD at list price, or ``None`` when unknown.
    """

    model: str = ""
    provider: str = ""
    effort: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: int = 0
    cost: float | None = None

    def __add__(self, other: Usage) -> Usage:
        """Pool two usages; an unpriced side makes the total unpriced.

        Args:
            other: The usage to add.

        Returns:
            The pooled usage, labelled with *other*'s model, provider and effort.
        """
        cost = None if self.cost is None or other.cost is None else self.cost + other.cost
        return Usage(
            model=other.model or self.model,
            provider=other.provider or self.provider,
            effort=other.effort if other.effort is not None else self.effort,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
            cost=cost,
        )


@dataclass
class Result:
    """A completed call.

    Attributes:
        text: The visible reply.
        thinking: Reasoning text where the provider returned it, else empty.
        tool_calls: Tool calls the model asked for.
        stop_reason: One of the ``STOP_*`` constants.
        usage: Tokens, latency and cost.
        message: The assistant turn to append for the next call.
        parsed: The validated structured output when a schema was requested.
    """

    text: str
    thinking: str
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: Usage
    message: Message
    parsed: Any = None


@dataclass(frozen=True)
class TextDelta:
    """A fragment of visible reply text."""

    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    """A fragment of reasoning text."""

    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    """A fragment of a tool call: id and name arrive once, arguments in pieces.

    Attributes:
        index: Position of the call in the reply, stable across fragments.
        id: The call id, empty on argument-only fragments.
        name: The tool name, empty on argument-only fragments.
        arguments: A fragment of the JSON arguments.
    """

    index: int
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Done:
    """The end of a stream, carrying the complete result."""

    result: Result


Event = TextDelta | ThinkingDelta | ToolCallDelta | Done
