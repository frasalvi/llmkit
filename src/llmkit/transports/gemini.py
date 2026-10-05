"""Review tier: plumbing.

Vertex ``generateContent``, which Gemini models speak.

Requests are plain dicts, which the SDK accepts in place of its typed objects. Thinking
is a discrete level; ``off`` is a zero budget. Function calls may arrive without an id;
llmkit then makes one up (prefixed ``llmkit-``) and never sends it back. The model's
content, including thought signatures, is replayed unchanged on the next turn.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterable, Iterable
from types import SimpleNamespace
from typing import Any

from ..errors import ContentFiltered
from ..types import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    STOP_CONTENT_FILTER,
    STOP_END,
    STOP_MAX_TOKENS,
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
from .base import Call, Reply, StreamEvent, merge_adjacent, to_dict, tool_schema

LEVELS = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
FILTERED = frozenset(
    {
        "SAFETY",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "RECITATION",
        "IMAGE_SAFETY",
        "MODEL_ARMOR",
    }
)
GENERATED_ID = "llmkit-"


class _NoAutomaticFunctionCallingNotice(logging.Filter):
    """Drops the SDK's per-call advice about automatic function calling."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for the notice, True otherwise."""
        return "automatic function calling" not in record.getMessage().lower()


for _name in ("google_genai.models", "google.genai.models"):
    logging.getLogger(_name).addFilter(_NoAutomaticFunctionCallingNotice())


def _enum_name(value: Any) -> str:
    """Return an SDK enum's bare name, or the string itself.

    Args:
        value: An SDK enum, a string, or ``None``.

    Returns:
        The bare name, empty for ``None``.
    """
    if value is None:
        return ""
    name = getattr(value, "name", None)
    return str(name if name else value).split(".")[-1]


def _dump_content(content: Any) -> dict[str, Any]:
    """Convert a response content to plain data for replay.

    Args:
        content: An SDK ``Content`` or namespace.

    Returns:
        The content as a dict, with function-call arguments kept exactly as sent.
    """
    out: dict[str, Any] = to_dict(content)
    dumped = out.get("parts") or []
    for raw_part, part in zip(
        getattr(content, "parts", None) or [], dumped, strict=False
    ):
        function_call = getattr(raw_part, "function_call", None)
        if function_call is not None and "function_call" in part:
            part["function_call"]["args"] = (
                to_dict(getattr(function_call, "args", None), keep_none=True) or {}
            )
    return out


def _user_parts(content: str | list[Part]) -> list[dict[str, Any]]:
    """Render user content as Gemini parts.

    Args:
        content: Plain text or a list of parts.

    Returns:
        Gemini ``text`` / ``inline_data`` parts.
    """
    out: list[dict[str, Any]] = []
    for part in parts(content):
        if isinstance(part, Text):
            out.append({"text": part.text})
        else:
            out.append({"inline_data": {"mime_type": part.media_type, "data": part.data}})
    return out


def _contents(messages: list[Message]) -> list[dict[str, Any]]:
    """Render the conversation as Gemini contents.

    Args:
        messages: The conversation, oldest first.

    Returns:
        Gemini contents; parallel tool results share one user turn.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == ROLE_USER:
            out.append({"role": "user", "parts": _user_parts(m.content)})
        elif m.role == ROLE_ASSISTANT:
            state = m.provider_state
            if state is not None and state.transport == "gemini":
                out.append(state.data)
                continue
            model_parts: list[dict[str, Any]] = []
            text = text_of(m.content)
            if text:
                model_parts.append({"text": text})
            for tc in m.tool_calls:
                function_call: dict[str, Any] = {"name": tc.name, "args": tc.arguments}
                if not tc.id.startswith(GENERATED_ID):
                    function_call["id"] = tc.id
                model_parts.append({"function_call": function_call})
            if model_parts:
                out.append({"role": "model", "parts": model_parts})
        elif m.role == ROLE_TOOL:
            key = "error" if m.is_error else "result"
            response: dict[str, Any] = {
                "name": m.tool_name,
                "response": {key: text_of(m.content)},
            }
            if m.tool_call_id and not m.tool_call_id.startswith(GENERATED_ID):
                response["id"] = m.tool_call_id
            out.append({"role": "user", "parts": [{"function_response": response}]})
    return merge_adjacent(out, content_key="parts")


class _Translator:
    """Collects streamed parts and rebuilds a full response at the end."""

    def __init__(self, transport: GeminiTransport, call: Call) -> None:
        """Bind to one call.

        Args:
            transport: The owning transport, whose ``parse`` builds the reply.
            call: The call being streamed.
        """
        self._transport = transport
        self._call = call
        self._parts: list[Any] = []
        self._usage: Any = None
        self._finish_reason: Any = None
        self._model = ""
        self._feedback: Any = None

    def feed(self, chunk: Any) -> list[StreamEvent]:
        """Translate one chunk. See :class:`llmkit.transports.base.Translator`."""
        self._model = getattr(chunk, "model_version", None) or self._model
        self._usage = getattr(chunk, "usage_metadata", None) or self._usage
        self._feedback = getattr(chunk, "prompt_feedback", None) or self._feedback
        events: list[StreamEvent] = []
        for candidate in (getattr(chunk, "candidates", None) or [])[:1]:
            self._finish_reason = getattr(candidate, "finish_reason", None) or (
                self._finish_reason
            )
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                self._parts.append(part)
                function_call = getattr(part, "function_call", None)
                if function_call is not None:
                    arguments = (
                        to_dict(getattr(function_call, "args", None), keep_none=True)
                        or {}
                    )
                    events.append(
                        ToolCallDelta(
                            len(self._parts) - 1,
                            getattr(function_call, "id", None) or "",
                            function_call.name,
                            json.dumps(arguments),
                        )
                    )
                elif getattr(part, "text", None):
                    if getattr(part, "thought", None):
                        events.append(ThinkingDelta(part.text))
                    else:
                        events.append(TextDelta(part.text))
        return events

    def finish(self) -> Reply:
        """Return the reply. See :class:`llmkit.transports.base.Translator`."""
        candidates = []
        if self._parts or self._finish_reason is not None:
            candidates = [
                SimpleNamespace(
                    content=SimpleNamespace(role="model", parts=self._parts),
                    finish_reason=self._finish_reason,
                )
            ]
        raw = SimpleNamespace(
            candidates=candidates,
            usage_metadata=self._usage,
            model_version=self._model,
            prompt_feedback=self._feedback,
        )
        return self._transport.parse(self._call, raw)


class GeminiTransport:
    """Vertex generateContent."""

    name = "gemini"

    def build(self, call: Call) -> dict[str, Any]:
        """Build the request body. See :class:`llmkit.transports.base.Transport`."""
        config: dict[str, Any] = {
            "max_output_tokens": call.max_tokens,
            "automatic_function_calling": {"disable": True},
        }
        if call.system:
            config["system_instruction"] = call.system
        if call.effort == "off":
            config["thinking_config"] = {"thinking_budget": 0}
        elif call.effort is not None:
            config["thinking_config"] = {
                "thinking_level": LEVELS[call.effort],
                "include_thoughts": True,
            }
        if call.temperature is not None:
            config["temperature"] = call.temperature
        if call.top_p is not None:
            config["top_p"] = call.top_p
        if call.tools:
            config["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters_json_schema": tool_schema(t),
                        }
                        for t in call.tools
                    ]
                }
            ]
        if call.tool_choice is not None:
            modes = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
            calling: dict[str, Any] = {"mode": modes.get(call.tool_choice, "ANY")}
            if call.tool_choice not in modes:
                calling["allowed_function_names"] = [call.tool_choice]
            config["tool_config"] = {"function_calling_config": calling}
        if call.schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = call.schema
        return {
            "model": call.route.deployment,
            "contents": _contents(call.messages),
            "config": config,
        }

    def parse(self, call: Call, raw: Any) -> Reply:
        """Extract the reply. See :class:`llmkit.transports.base.Transport`.

        Raises:
            ContentFiltered: If Vertex blocked the prompt and returned no candidate.
        """
        candidates = getattr(raw, "candidates", None) or []
        if not candidates:
            reason = _enum_name(
                getattr(getattr(raw, "prompt_feedback", None), "block_reason", None)
            )
            raise ContentFiltered(
                f"Vertex blocked the prompt: {reason or 'no candidates'}",
                categories=[reason] if reason else [],
                provider=call.route.provider,
            )
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        texts: list[str] = []
        thoughts: list[str] = []
        calls: list[ToolCall] = []
        # Walk the parts; a missing function-call id is made up from the part index.
        for index, part in enumerate(getattr(content, "parts", None) or []):
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                arguments = (
                    to_dict(getattr(function_call, "args", None), keep_none=True) or {}
                )
                call_id = getattr(function_call, "id", None) or f"{GENERATED_ID}{index}"
                calls.append(
                    ToolCall(
                        call_id, function_call.name, arguments, json.dumps(arguments)
                    )
                )
                continue
            text = getattr(part, "text", None)
            if text:
                (thoughts if getattr(part, "thought", None) else texts).append(text)
        # Stop reason.
        finish = _enum_name(getattr(candidate, "finish_reason", None))
        stop = STOP_TOOL_USE if calls else STOP_END
        if finish == "MAX_TOKENS":
            stop = STOP_MAX_TOKENS
        elif finish in FILTERED:
            stop = STOP_CONTENT_FILTER
        # Usage: prompt tokens include cached ones; thoughts are billed as output.
        usage = getattr(raw, "usage_metadata", None)
        prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
        cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
        output = int(getattr(usage, "candidates_token_count", 0) or 0) + int(
            getattr(usage, "thoughts_token_count", 0) or 0
        )
        return Reply(
            text="".join(texts),
            thinking="".join(thoughts),
            tool_calls=calls,
            stop_reason=stop,
            input_tokens=prompt - cached,
            output_tokens=output,
            cached_input_tokens=cached,
            served_model=str(getattr(raw, "model_version", "") or ""),
            usage_reported=usage is not None,
            provider_state=(
                ProviderState("gemini", _dump_content(content))
                if content is not None
                else None
            ),
        )

    def send(self, client: Any, body: dict[str, Any]) -> Any:
        """Issue the request. See :class:`llmkit.transports.base.Transport`."""
        return client.models.generate_content(**body)

    async def asend(self, client: Any, body: dict[str, Any]) -> Any:
        """Issue the request on the client's ``aio`` side."""
        return await client.models.generate_content(**body)

    def open_stream(self, client: Any, body: dict[str, Any]) -> Iterable[Any]:
        """Open a stream. See :class:`llmkit.transports.base.Transport`."""
        return client.models.generate_content_stream(**body)

    async def aopen_stream(self, client: Any, body: dict[str, Any]) -> AsyncIterable[Any]:
        """Open a stream on the client's ``aio`` side."""
        return await client.models.generate_content_stream(**body)

    def translator(self, call: Call) -> _Translator:
        """Return a translator. See :class:`llmkit.transports.base.Transport`."""
        return _Translator(self, call)
