"""Review tier: plumbing.

The OpenAI Responses API, which GPT deployments on Foundry speak.

Requests are stateless (``store: false``) with encrypted reasoning included, so a
reasoning item can be replayed on the next turn without server-side storage. Reported
``input_tokens`` include cached tokens; they are split here so nothing is billed twice.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, Iterable
from typing import Any

from ..errors import FatalRequest, TransientError
from ..schemas import all_required
from ..types import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    STOP_CONTENT_FILTER,
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
    data_url,
    loads_arguments,
    to_dict,
    tool_schema,
)

TRANSIENT_CODES = frozenset(
    {"server_error", "rate_limit_exceeded", "vector_store_timeout"}
)
EFFORTS = {
    "off": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "xhigh",
}


def _user_content(content: str | list[Part]) -> list[dict[str, Any]]:
    """Render user content as Responses input parts.

    Args:
        content: Plain text or a list of parts.

    Returns:
        Responses ``input_text`` / ``input_image`` parts.
    """
    out: list[dict[str, Any]] = []
    for part in parts(content):
        if isinstance(part, Text):
            out.append({"type": "input_text", "text": part.text})
        else:
            out.append({"type": "input_image", "image_url": data_url(part)})
    return out


def _input(messages: list[Message]) -> list[dict[str, Any]]:
    """Render the conversation as Responses input items.

    Args:
        messages: The conversation, oldest first.

    Returns:
        Input items; an assistant turn carrying Responses state is replayed verbatim.
    """
    items: list[dict[str, Any]] = []
    for m in messages:
        if m.role == ROLE_USER:
            items.append({"role": "user", "content": _user_content(m.content)})
        elif m.role == ROLE_ASSISTANT:
            state = m.provider_state
            if state is not None and state.transport == "responses":
                items.extend(state.data)
                continue
            text = text_of(m.content)
            if text:
                items.append({"role": "assistant", "content": text})
            for tc in m.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.raw or json.dumps(tc.arguments),
                    }
                )
        elif m.role == ROLE_TOOL:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": m.tool_call_id,
                    "output": text_of(m.content),
                }
            )
    return items


def _failure(error: Any, provider: str) -> Exception:
    """Classify a failed response's error.

    Args:
        error: The response's ``error`` object, or a stream ``error`` event.
        provider: The route's provider.

    Returns:
        A :class:`TransientError` for retryable codes, else a :class:`FatalRequest`.
    """
    code = getattr(error, "code", None)
    # Code and message only: the whole response echoes the prompt and output.
    detail = f"{code}: {getattr(error, 'message', None)}"
    error_type = TransientError if code in TRANSIENT_CODES else FatalRequest
    return error_type(f"response failed: {detail}", provider=provider)


class _Translator:
    """Maps Responses stream events to llmkit events."""

    def __init__(self, transport: ResponsesTransport, call: Call) -> None:
        """Bind to one call.

        Args:
            transport: The owning transport, whose ``parse`` builds the reply.
            call: The call being streamed.
        """
        self._transport = transport
        self._call = call
        self._final: Any = None

    def feed(self, chunk: Any) -> list[StreamEvent]:
        """Translate one event. See :class:`llmkit.transports.base.Translator`."""
        kind = getattr(chunk, "type", "")
        if kind == "response.output_text.delta":
            return [TextDelta(chunk.delta)]
        if kind == "response.reasoning_summary_text.delta":
            return [ThinkingDelta(chunk.delta)]
        if kind == "response.output_item.added":
            item = chunk.item
            if getattr(item, "type", "") == "function_call":
                return [ToolCallDelta(chunk.output_index, item.call_id, item.name, "")]
        elif kind == "response.function_call_arguments.delta":
            return [ToolCallDelta(chunk.output_index, "", "", chunk.delta)]
        elif kind in ("response.completed", "response.incomplete"):
            self._final = chunk.response
        elif kind in ("response.failed", "error"):
            response = getattr(chunk, "response", None)
            error = getattr(response, "error", None) or chunk
            raise _failure(error, self._call.route.provider)
        return []

    def finish(self) -> Reply:
        """Return the reply. See :class:`llmkit.transports.base.Translator`."""
        if self._final is None:
            raise TransientError(
                "stream ended before the final response",
                provider=self._call.route.provider,
            )
        return self._transport.parse(self._call, self._final)


class ResponsesTransport:
    """The OpenAI Responses API."""

    name = "responses"

    def build(self, call: Call) -> dict[str, Any]:
        """Build the request body. See :class:`llmkit.transports.base.Transport`."""
        body: dict[str, Any] = {
            "model": call.route.deployment,
            "input": _input(call.messages),
            "max_output_tokens": call.max_tokens,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if call.system:
            body["instructions"] = call.system
        if call.effort is not None:
            reasoning: dict[str, Any] = {"effort": EFFORTS[call.effort]}
            if call.effort != "off":
                reasoning["summary"] = "auto"
            body["reasoning"] = reasoning
        if call.temperature is not None:
            body["temperature"] = call.temperature
        if call.top_p is not None:
            body["top_p"] = call.top_p
        if call.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": tool_schema(t),
                    "strict": False,
                }
                for t in call.tools
            ]
        if call.tool_choice is not None:
            if call.tool_choice in ("auto", "none", "required"):
                body["tool_choice"] = call.tool_choice
            else:
                body["tool_choice"] = {"type": "function", "name": call.tool_choice}
        if call.schema is not None:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": call.schema_name,
                    "schema": call.schema,
                    "strict": all_required(call.schema),
                }
            }
        return body

    def parse(self, call: Call, raw: Any) -> Reply:
        """Extract the reply. See :class:`llmkit.transports.base.Transport`.

        Raises:
            TransientError: If the response failed with a retryable error code.
            FatalRequest: If the response failed with any other error code.
        """
        # A 200 can still carry a failed response.
        if getattr(raw, "status", "") == "failed":
            raise _failure(getattr(raw, "error", None), call.route.provider)
        texts: list[str] = []
        thoughts: list[str] = []
        calls: list[ToolCall] = []
        refusal = ""
        output = getattr(raw, "output", None) or []
        # Walk the output items by kind.
        for item in output:
            kind = getattr(item, "type", "")
            if kind == "message":
                for content in getattr(item, "content", None) or []:
                    content_type = getattr(content, "type", "")
                    if content_type == "output_text":
                        texts.append(content.text)
                    elif content_type == "refusal":
                        refusal = content.refusal
            elif kind == "reasoning":
                for summary in getattr(item, "summary", None) or []:
                    thoughts.append(getattr(summary, "text", "") or "")
            elif kind == "function_call":
                arguments = getattr(item, "arguments", "") or ""
                calls.append(
                    ToolCall(
                        item.call_id, item.name, loads_arguments(arguments), arguments
                    )
                )
        # Stop reason.
        stop = STOP_TOOL_USE if calls else STOP_END
        if getattr(raw, "status", "") == "incomplete":
            reason = getattr(getattr(raw, "incomplete_details", None), "reason", "")
            stop = STOP_CONTENT_FILTER if reason == "content_filter" else STOP_MAX_TOKENS
        text = "".join(texts)
        if refusal and not text:
            text, stop = refusal, STOP_REFUSAL
        # Usage: input_tokens includes cached tokens, so bill only the remainder as input.
        usage = getattr(raw, "usage", None)
        total_input = int(getattr(usage, "input_tokens", 0) or 0)
        details = getattr(usage, "input_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0)
        return Reply(
            text=text,
            thinking="\n\n".join(t for t in thoughts if t),
            tool_calls=calls,
            stop_reason=stop,
            input_tokens=total_input - cached,
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cached_input_tokens=cached,
            served_model=str(getattr(raw, "model", "") or ""),
            usage_reported=usage is not None,
            provider_state=ProviderState("responses", [to_dict(i) for i in output]),
        )

    def send(self, client: Any, body: dict[str, Any]) -> Any:
        """Issue the request. See :class:`llmkit.transports.base.Transport`."""
        return client.responses.create(**body)

    async def asend(self, client: Any, body: dict[str, Any]) -> Any:
        """Issue the request. See :class:`llmkit.transports.base.Transport`."""
        return await client.responses.create(**body)

    def open_stream(self, client: Any, body: dict[str, Any]) -> Iterable[Any]:
        """Open a stream. See :class:`llmkit.transports.base.Transport`."""
        return client.responses.create(**body, stream=True)

    async def aopen_stream(self, client: Any, body: dict[str, Any]) -> AsyncIterable[Any]:
        """Open a stream. See :class:`llmkit.transports.base.Transport`."""
        return await client.responses.create(**body, stream=True)

    def translator(self, call: Call) -> _Translator:
        """Return a translator. See :class:`llmkit.transports.base.Transport`."""
        return _Translator(self, call)
