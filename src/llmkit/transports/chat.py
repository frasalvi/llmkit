"""Review tier: plumbing.

OpenAI-compatible chat completions: Foundry models without a native surface, all of
OpenRouter, and Vertex model-garden endpoints.

Reasoning text arrives as ``reasoning_content`` (DeepSeek-style hosts) or
``reasoning`` (OpenRouter). Some hosts require the reasoning of a tool-calling turn to
be sent back, so it travels in ``provider_state``. Usage arrives in streams only on a
final chunk with no choices, which is why ``include_usage`` is always requested.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, Iterable
from types import SimpleNamespace
from typing import Any

from ..errors import TransientError
from ..schemas import all_required
from ..types import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    STOP_CONTENT_FILTER,
    STOP_END,
    STOP_MAX_TOKENS,
    STOP_TOOL_USE,
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
from .base import Call, Reply, StreamEvent, data_url, loads_arguments, tool_schema

_STOPS = {
    "length": STOP_MAX_TOKENS,
    "tool_calls": STOP_TOOL_USE,
    "content_filter": STOP_CONTENT_FILTER,
}


def _user(content: str | list[Part]) -> str | list[dict[str, Any]]:
    """Render user content: a plain string unless it carries images.

    Args:
        content: Plain text or a list of parts.

    Returns:
        A string, or a list of ``text`` / ``image_url`` parts.
    """
    items = parts(content)
    if all(isinstance(p, Text) for p in items):
        return text_of(content)
    return [
        {"type": "text", "text": p.text}
        if isinstance(p, Text)
        else {"type": "image_url", "image_url": {"url": data_url(p)}}
        for p in items
    ]


def _messages(call: Call) -> list[dict[str, Any]]:
    """Render the system prompt and conversation.

    Args:
        call: The call being built.

    Returns:
        Chat messages; a chat-transport assistant turn gets its reasoning replayed.
    """
    # OpenRouter reads replayed reasoning as ``reasoning``; other hosts as
    # ``reasoning_content``.
    replay_key = (
        "reasoning" if call.route.provider == "openrouter" else "reasoning_content"
    )
    out: list[dict[str, Any]] = []
    if call.system:
        out.append({"role": "system", "content": call.system})
    for m in call.messages:
        if m.role == ROLE_USER:
            out.append({"role": "user", "content": _user(m.content)})
        elif m.role == ROLE_ASSISTANT:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": text_of(m.content) or None,
            }
            state = m.provider_state
            if state is not None and state.transport == "chat":
                reasoning = state.data.get("reasoning_content")
                if reasoning:
                    message[replay_key] = reasoning
            if m.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.raw or json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(message)
        elif m.role == ROLE_TOOL:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": text_of(m.content),
                }
            )
    return out


def _reasoning(message: Any) -> str:
    """Return reasoning text under either host's field name.

    Args:
        message: A response message or stream delta.

    Returns:
        The reasoning text, empty when absent.
    """
    return str(
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or ""
    )


class _Translator:
    """Accumulates streamed deltas, including fragmented tool calls."""

    def __init__(self, transport: ChatTransport, call: Call) -> None:
        """Bind to one call.

        Args:
            transport: The owning transport, whose ``parse`` builds the reply.
            call: The call being streamed.
        """
        self._transport = transport
        self._call = call
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._tools: dict[int, dict[str, str]] = {}
        self._finish: str | None = None
        self._usage: Any = None
        self._model = ""

    def feed(self, chunk: Any) -> list[StreamEvent]:
        """Translate one chunk. See :class:`llmkit.transports.base.Translator`."""
        self._model = getattr(chunk, "model", None) or self._model
        self._usage = getattr(chunk, "usage", None) or self._usage
        events: list[StreamEvent] = []
        # The usage-only final chunk has no choices, so the loop body is skipped.
        for choice in getattr(chunk, "choices", None) or []:
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None)
            if content:
                self._text.append(content)
                events.append(TextDelta(content))
            reasoning = _reasoning(delta)
            if reasoning:
                self._reasoning.append(reasoning)
                events.append(ThinkingDelta(reasoning))
            for tool_call in getattr(delta, "tool_calls", None) or []:
                slot = self._tools.setdefault(
                    tool_call.index, {"id": "", "name": "", "arguments": ""}
                )
                call_id = getattr(tool_call, "id", None) or ""
                function = getattr(tool_call, "function", None)
                name = getattr(function, "name", None) or ""
                arguments = getattr(function, "arguments", None) or ""
                if call_id:
                    slot["id"] = call_id
                if name:
                    slot["name"] = name
                slot["arguments"] += arguments
                events.append(ToolCallDelta(tool_call.index, call_id, name, arguments))
            self._finish = getattr(choice, "finish_reason", None) or self._finish
        return events

    def finish(self) -> Reply:
        """Return the reply. See :class:`llmkit.transports.base.Translator`."""
        if self._finish is None:
            raise TransientError(
                "stream ended before a finish reason", provider=self._call.route.provider
            )
        message = SimpleNamespace(
            content="".join(self._text) or None,
            reasoning_content="".join(self._reasoning) or None,
            tool_calls=[
                SimpleNamespace(
                    id=slot["id"],
                    function=SimpleNamespace(
                        name=slot["name"], arguments=slot["arguments"]
                    ),
                )
                for _, slot in sorted(self._tools.items())
            ],
        )
        raw = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=self._finish)],
            usage=self._usage,
            model=self._model,
        )
        return self._transport.parse(self._call, raw)


class ChatTransport:
    """OpenAI-compatible chat completions."""

    name = "chat"

    def build(self, call: Call) -> dict[str, Any]:
        """Build the request body. See :class:`llmkit.transports.base.Transport`."""
        body: dict[str, Any] = {
            "model": call.route.deployment,
            "messages": _messages(call),
            "max_tokens": call.max_tokens,
        }
        if call.effort is not None:
            if call.route.provider == "openrouter":
                reasoning: dict[str, Any] = (
                    {"enabled": False}
                    if call.effort == "off"
                    else {"effort": call.effort}
                )
                body["extra_body"] = {"reasoning": reasoning}
            else:
                body["reasoning_effort"] = "none" if call.effort == "off" else call.effort
        if call.temperature is not None:
            body["temperature"] = call.temperature
        if call.top_p is not None:
            body["top_p"] = call.top_p
        if call.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": tool_schema(t),
                    },
                }
                for t in call.tools
            ]
        if call.tool_choice is not None:
            if call.tool_choice in ("auto", "none", "required"):
                body["tool_choice"] = call.tool_choice
            else:
                body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": call.tool_choice},
                }
        if call.schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": call.schema_name,
                    "schema": call.schema,
                    "strict": all_required(call.schema),
                },
            }
        return body

    def parse(self, call: Call, raw: Any) -> Reply:
        """Extract the reply. See :class:`llmkit.transports.base.Transport`.

        Raises:
            TransientError: If the host returned no choices.
        """
        choices = getattr(raw, "choices", None) or []
        if not choices:
            raise TransientError(
                "chat completion returned no choices", provider=call.route.provider
            )
        choice = choices[0]
        message = choice.message
        calls = [
            ToolCall(
                tc.id,
                tc.function.name,
                loads_arguments(tc.function.arguments),
                tc.function.arguments or "",
            )
            for tc in getattr(message, "tool_calls", None) or []
        ]
        finish = str(getattr(choice, "finish_reason", "") or "")
        stop = _STOPS.get(finish, STOP_TOOL_USE if calls else STOP_END)
        reasoning = _reasoning(message)
        # Usage: prompt_tokens includes cached tokens, so bill only the remainder as input.
        usage = getattr(raw, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0)
        return Reply(
            text=str(getattr(message, "content", None) or ""),
            thinking=reasoning,
            tool_calls=calls,
            stop_reason=stop,
            input_tokens=prompt - cached,
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cached_input_tokens=cached,
            served_model=str(getattr(raw, "model", "") or ""),
            usage_reported=usage is not None,
            provider_state=(
                ProviderState("chat", {"reasoning_content": reasoning})
                if reasoning
                else None
            ),
        )

    def send(self, client: Any, body: dict[str, Any]) -> Any:
        """Issue the request. See :class:`llmkit.transports.base.Transport`."""
        return client.chat.completions.create(**body)

    async def asend(self, client: Any, body: dict[str, Any]) -> Any:
        """Issue the request. See :class:`llmkit.transports.base.Transport`."""
        return await client.chat.completions.create(**body)

    def open_stream(self, client: Any, body: dict[str, Any]) -> Iterable[Any]:
        """Open a stream. See :class:`llmkit.transports.base.Transport`."""
        return client.chat.completions.create(
            **body, stream=True, stream_options={"include_usage": True}
        )

    async def aopen_stream(self, client: Any, body: dict[str, Any]) -> AsyncIterable[Any]:
        """Open a stream. See :class:`llmkit.transports.base.Transport`."""
        return await client.chat.completions.create(
            **body, stream=True, stream_options={"include_usage": True}
        )

    def translator(self, call: Call) -> _Translator:
        """Return a translator. See :class:`llmkit.transports.base.Transport`."""
        return _Translator(self, call)
