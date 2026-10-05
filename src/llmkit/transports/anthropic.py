"""Review tier: plumbing.

Anthropic's Messages API, served the same way by Foundry (``AnthropicFoundry``) and
Vertex (``AnthropicVertex``).

Thinking is adaptive with an ``effort`` level; turning it off uses the type the model
accepts (``disabled``, or ``between_tools`` on Sonnet 5.5). Thinking text is requested
as a summary. Thinking blocks are signed and must come back unchanged on the next turn,
so the raw content blocks travel in ``provider_state``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, Iterable
from types import SimpleNamespace
from typing import Any

from ..errors import TransientError
from ..types import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    STOP_END,
    STOP_MAX_TOKENS,
    STOP_REFUSAL,
    STOP_TOOL_USE,
    Message,
    Part,
    ProviderState,
    Text,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolCallDelta,
    parts,
    text_of,
)
from .base import (
    Call,
    Reply,
    StreamEvent,
    b64,
    loads_arguments,
    merge_adjacent,
    ns,
    to_dict,
    tool_schema,
)

STOPS = {
    "end_turn": STOP_END,
    "stop_sequence": STOP_END,
    "pause_turn": STOP_END,
    "max_tokens": STOP_MAX_TOKENS,
    "model_context_window_exceeded": STOP_MAX_TOKENS,
    "tool_use": STOP_TOOL_USE,
    "refusal": STOP_REFUSAL,
}

_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _blocks(content: str | list[Part]) -> list[dict[str, Any]]:
    """Render content as text and image blocks.

    Args:
        content: Plain text or a list of parts.

    Returns:
        Messages API content blocks.
    """
    out: list[dict[str, Any]] = []
    for part in parts(content):
        if isinstance(part, Text):
            out.append({"type": "text", "text": part.text})
        else:
            out.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.media_type,
                        "data": b64(part.data),
                    },
                }
            )
    return out


def _messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Render the conversation, merging consecutive same-role turns.

    Args:
        messages: The conversation, oldest first.

    Returns:
        Messages API messages; parallel tool results share one user message.
    """
    rendered: list[dict[str, Any]] = []
    for m in messages:
        if m.role == ROLE_USER:
            rendered.append({"role": "user", "content": _blocks(m.content)})
        elif m.role == ROLE_ASSISTANT:
            state = m.provider_state
            if state is not None and state.transport == "anthropic":
                rendered.append({"role": "assistant", "content": list(state.data)})
                continue
            content = _blocks(m.content) + [
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                for tc in m.tool_calls
            ]
            if content:
                rendered.append({"role": "assistant", "content": content})
        elif m.role == ROLE_TOOL:
            rendered.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": text_of(m.content),
                            "is_error": m.is_error,
                        }
                    ],
                }
            )
    return merge_adjacent(rendered)


class _Translator:
    """Rebuilds content blocks from raw Messages stream events."""

    def __init__(self, transport: AnthropicTransport, call: Call) -> None:
        """Bind to one call.

        Args:
            transport: The owning transport, whose ``parse`` builds the reply.
            call: The call being streamed.
        """
        self._transport = transport
        self._call = call
        self._blocks: dict[int, dict[str, Any]] = {}
        self._partial: dict[int, str] = {}
        self._usage: dict[str, int] = {}
        self._model = ""
        self._stop: str | None = None

    def _merge_usage(self, usage: Any) -> None:
        """Keep the latest value of each usage counter.

        Args:
            usage: A usage object from ``message_start`` or ``message_delta``.
        """
        for key in _USAGE_KEYS:
            value = getattr(usage, key, None)
            if value is not None:
                self._usage[key] = int(value)

    def feed(self, chunk: Any) -> list[StreamEvent]:
        """Translate one event. See :class:`llmkit.transports.base.Translator`."""
        kind = getattr(chunk, "type", "")
        if kind == "message_start":
            self._model = getattr(chunk.message, "model", "") or ""
            self._merge_usage(getattr(chunk.message, "usage", None))
        elif kind == "content_block_start":
            block = to_dict(chunk.content_block)
            self._blocks[chunk.index] = block
            if block.get("type") == "tool_use":
                self._partial[chunk.index] = ""
                return [
                    ToolCallDelta(
                        chunk.index, block.get("id", ""), block.get("name", ""), ""
                    )
                ]
        elif kind == "content_block_delta":
            delta = chunk.delta
            delta_type = getattr(delta, "type", "")
            block = self._blocks.setdefault(chunk.index, {})
            if delta_type == "text_delta":
                block["text"] = block.get("text", "") + delta.text
                return [TextDelta(delta.text)]
            if delta_type == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.thinking
                return [ThinkingDelta(delta.thinking)]
            if delta_type == "signature_delta":
                block["signature"] = delta.signature
            elif delta_type == "input_json_delta":
                self._partial[chunk.index] = self._partial.get(chunk.index, "") + (
                    delta.partial_json
                )
                return [ToolCallDelta(chunk.index, "", "", delta.partial_json)]
        elif kind == "content_block_stop":
            if chunk.index in self._partial:
                self._blocks[chunk.index]["input"] = loads_arguments(
                    self._partial[chunk.index]
                )
        elif kind == "message_delta":
            self._stop = getattr(chunk.delta, "stop_reason", None) or self._stop
            self._merge_usage(getattr(chunk, "usage", None))
        return []

    def finish(self) -> Reply:
        """Return the reply. See :class:`llmkit.transports.base.Translator`."""
        if self._stop is None:
            raise TransientError(
                "stream ended before the stop reason", provider=self._call.route.provider
            )
        message = SimpleNamespace(
            content=[ns(self._blocks[i]) for i in sorted(self._blocks)],
            stop_reason=self._stop,
            usage=SimpleNamespace(**self._usage),
            model=self._model,
        )
        return self._transport.parse(self._call, message)


class AnthropicTransport:
    """Anthropic's Messages API."""

    name = "anthropic"

    def build(self, call: Call) -> dict[str, Any]:
        """Build the request body. See :class:`llmkit.transports.base.Transport`."""
        body: dict[str, Any] = {
            "model": call.route.deployment,
            "max_tokens": call.max_tokens,
            "messages": _messages(call.messages),
        }
        output_config: dict[str, Any] = {}
        if call.system:
            body["system"] = call.system
        if call.effort == "off":
            body["thinking"] = {"type": call.route.spec.thinking_off}
        elif call.effort is not None:
            body["thinking"] = {"type": "adaptive", "display": "summarized"}
            output_config["effort"] = call.effort
        if call.temperature is not None:
            body["temperature"] = call.temperature
        if call.top_p is not None:
            body["top_p"] = call.top_p
        if call.cache_prefix:
            body["extra_body"] = {"cache_control": {"type": "ephemeral"}}
        if call.tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": tool_schema(t),
                }
                for t in call.tools
            ]
        if call.tool_choice is not None:
            choices = {
                "auto": {"type": "auto"},
                "none": {"type": "none"},
                "required": {"type": "any"},
            }
            body["tool_choice"] = choices.get(
                call.tool_choice, {"type": "tool", "name": call.tool_choice}
            )
        if call.schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": call.schema}
        if output_config:
            body["output_config"] = output_config
        return body

    def parse(self, call: Call, raw: Any) -> Reply:
        """Extract the reply. See :class:`llmkit.transports.base.Transport`."""
        texts: list[str] = []
        thoughts: list[str] = []
        calls: list[ToolCall] = []
        content = getattr(raw, "content", None) or []
        # Walk the content blocks by kind.
        for block in content:
            kind = getattr(block, "type", "")
            if kind == "text":
                texts.append(block.text)
            elif kind == "thinking":
                thoughts.append(getattr(block, "thinking", "") or "")
            elif kind == "tool_use":
                arguments = to_dict(getattr(block, "input", None)) or {}
                if not isinstance(arguments, dict):
                    arguments = {}
                calls.append(
                    ToolCall(block.id, block.name, arguments, json.dumps(arguments))
                )
        # Anthropic reports input_tokens already excluding cache reads and writes.
        usage = getattr(raw, "usage", None)
        return Reply(
            text="".join(texts),
            thinking="\n\n".join(t for t in thoughts if t),
            tool_calls=calls,
            stop_reason=STOPS.get(str(getattr(raw, "stop_reason", "") or ""), STOP_END),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cached_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            served_model=str(getattr(raw, "model", "") or ""),
            provider_state=ProviderState("anthropic", [to_dict(b) for b in content]),
        )

    def send(self, client: Any, body: dict[str, Any]) -> Any:
        """Issue the request. See :class:`llmkit.transports.base.Transport`."""
        return client.messages.create(**body)

    async def asend(self, client: Any, body: dict[str, Any]) -> Any:
        """Issue the request. See :class:`llmkit.transports.base.Transport`."""
        return await client.messages.create(**body)

    def open_stream(self, client: Any, body: dict[str, Any]) -> Iterable[Any]:
        """Open a stream. See :class:`llmkit.transports.base.Transport`."""
        return client.messages.create(**body, stream=True)

    async def aopen_stream(self, client: Any, body: dict[str, Any]) -> AsyncIterable[Any]:
        """Open a stream. See :class:`llmkit.transports.base.Transport`."""
        return await client.messages.create(**body, stream=True)

    def translator(self, call: Call) -> _Translator:
        """Return a translator. See :class:`llmkit.transports.base.Transport`."""
        return _Translator(self, call)
