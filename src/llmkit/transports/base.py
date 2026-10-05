"""Review tier: plumbing.

The contract every transport implements and the helpers they share.

A transport turns a provider-neutral :class:`Call` into a request body
(:meth:`Transport.build`), turns a provider response into a :class:`Reply`
(:meth:`Transport.parse`), and turns stream chunks into events through a
:class:`Translator`. Building and parsing are pure, so they are tested without a
network; the senders are one-line SDK calls.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Protocol

from ..registry import Route
from ..schemas import close_objects, to_json_schema
from ..types import (
    STOP_END,
    Image,
    Message,
    ProviderState,
    TextDelta,
    ThinkingDelta,
    Tool,
    ToolCall,
    ToolCallDelta,
)

StreamEvent = TextDelta | ThinkingDelta | ToolCallDelta


@dataclass(frozen=True)
class Call:
    """Everything a transport needs for one request, provider-neutral.

    Attributes:
        route: The resolved model.
        messages: The conversation, oldest first.
        system: System instructions, empty for none.
        effort: A ladder rung, or ``None`` to send no reasoning setting.
        max_tokens: Output cap.
        temperature: Sampling temperature, or ``None`` to send none.
        top_p: Nucleus cutoff, or ``None`` to send none.
        cache_prefix: Ask for explicit prompt caching where the provider needs it.
        tools: Tools the model may call.
        tool_choice: ``auto``, ``none``, ``required``, a tool name, or ``None``.
        schema: A closed JSON schema for structured output, or ``None``.
        schema_name: A provider-safe name for *schema*.
    """

    route: Route
    messages: list[Message]
    system: str = ""
    effort: str | None = None
    max_tokens: int = 16000
    temperature: float | None = None
    top_p: float | None = None
    cache_prefix: bool = False
    tools: list[Tool] = field(default_factory=list)
    tool_choice: str | None = None
    schema: dict[str, Any] | None = None
    schema_name: str = "output"


@dataclass
class Reply:
    """What a transport extracted from one response, before pricing.

    Attributes:
        text: Visible reply.
        thinking: Reasoning text, empty when withheld.
        tool_calls: Requested tool calls.
        stop_reason: One of the ``STOP_*`` constants.
        input_tokens: Prompt tokens at the full rate (cached ones excluded).
        output_tokens: Output tokens, thinking included.
        cached_input_tokens: Prompt tokens read from a cache.
        cache_write_tokens: Prompt tokens written to a cache.
        served_model: The model string the provider reported.
        provider_state: State to replay on the next turn.
        usage_reported: Whether the response carried token usage; when ``False`` the
            token fields are zeros that mean "unknown", not "free".
    """

    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = STOP_END
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    served_model: str = ""
    provider_state: ProviderState | None = None
    usage_reported: bool = True


class Translator(Protocol):
    """Turns one stream's chunks into events and, at the end, a reply."""

    def feed(self, chunk: Any) -> list[StreamEvent]:
        """Consume one chunk and return the events it carries."""
        ...

    def finish(self) -> Reply:
        """Return the complete reply once the stream has ended."""
        ...


class Transport(Protocol):
    """One wire format."""

    name: str

    def build(self, call: Call) -> dict[str, Any]:
        """Build the request body (keyword arguments for the SDK call)."""
        ...

    def parse(self, call: Call, raw: Any) -> Reply:
        """Extract a reply from a non-streaming response."""
        ...

    def send(self, client: Any, body: dict[str, Any]) -> Any:
        """Issue a non-streaming request."""
        ...

    async def asend(self, client: Any, body: dict[str, Any]) -> Any:
        """Issue a non-streaming request asynchronously."""
        ...

    def open_stream(self, client: Any, body: dict[str, Any]) -> Iterable[Any]:
        """Open a stream of raw chunks."""
        ...

    async def aopen_stream(self, client: Any, body: dict[str, Any]) -> AsyncIterable[Any]:
        """Open an async stream of raw chunks."""
        ...

    def translator(self, call: Call) -> Translator:
        """Return a fresh translator for one stream."""
        ...


def ns(value: Any) -> Any:
    """Convert nested dicts to attribute objects, mirroring SDK response objects.

    Args:
        value: JSON-like data.

    Returns:
        The same data with every dict replaced by a ``SimpleNamespace``.
    """
    if isinstance(value, dict):
        return SimpleNamespace(**{k: ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [ns(v) for v in value]
    return value


def to_dict(value: Any, *, keep_none: bool = False) -> Any:
    """Convert SDK objects or namespaces back to plain data.

    Args:
        value: An SDK model, a namespace, or plain data.
        keep_none: Keep ``None`` values. Leave unset for SDK models, whose unset optional
            fields the API rejects; set it for model-authored data such as tool arguments,
            where ``None`` is a real value.

    Returns:
        Dicts, lists and scalars only.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=not keep_none)
    if isinstance(value, SimpleNamespace):
        value = vars(value)
    if isinstance(value, dict):
        return {
            k: to_dict(v, keep_none=keep_none)
            for k, v in value.items()
            if keep_none or v is not None
        }
    if isinstance(value, list):
        return [to_dict(v, keep_none=keep_none) for v in value]
    return value


def b64(data: bytes) -> str:
    """Return base64 text for *data*."""
    return base64.b64encode(data).decode("ascii")


def data_url(image: Image) -> str:
    """Return a ``data:`` URL for an image part."""
    return f"data:{image.media_type};base64,{b64(image.data)}"


def loads_arguments(raw: str | None) -> dict[str, Any]:
    """Parse tool-call arguments, tolerating invalid JSON.

    Args:
        raw: The JSON text the provider sent.

    Returns:
        The decoded object, or an empty dict when it is not a JSON object.
    """
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def tool_schema(tool: Tool) -> dict[str, Any]:
    """Return a tool's closed parameter schema."""
    return close_objects(to_json_schema(tool.parameters))


def merge_adjacent(
    messages: list[dict[str, Any]], content_key: str = "content"
) -> list[dict[str, Any]]:
    """Merge consecutive messages with the same role into one.

    Parallel tool results must reach Claude and Gemini as one turn.

    Args:
        messages: Rendered messages whose *content_key* holds a list.
        content_key: The key holding the content list.

    Returns:
        The merged list.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if out and out[-1]["role"] == message["role"]:
            out[-1] = {
                **out[-1],
                content_key: [*out[-1][content_key], *message[content_key]],
            }
        else:
            out.append(message)
    return out
