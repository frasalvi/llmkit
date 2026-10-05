"""Review tier: plumbing.

``LLM``: one model, ready to call.

Construction does every check that can fail without a request: the name resolves,
the provider's credentials exist, and a price is found (or a loud warning says it was
not). Each call then validates its options against the model, sends through the
route's transport with llmkit's retry policy, prices the result, and hands a record to
every ``on_call`` hook.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .clients import SdkClients, make_clients
from .credentials import credentials_for, load_env
from .errors import SchemaError, UnsupportedFeature
from .pricing import compute_cost, load_table, resolve_rates, warn_unpriced
from .records import Hook, build_record
from .registry import check_effort, resolve
from .retry import acall_with_retries, call_with_retries
from .schemas import close_objects, parse_output, schema_name, to_json_schema
from .transports import TRANSPORTS
from .transports.base import Call, Reply
from .types import ROLE_ASSISTANT, STOP_TOOL_USE, Message, Result, Tool, Usage

Prompt = str | list[Message]
Schema = dict[str, Any] | type[BaseModel]


def _elapsed_ms(started: float) -> int:
    """Return milliseconds since *started* (a ``time.monotonic`` value)."""
    return int((time.monotonic() - started) * 1000)


class LLM:
    """One model on one route."""

    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        max_retries: int = 3,
        timeout: float = 600.0,
        on_call: Hook | list[Hook] | None = None,
        tags: Mapping[str, Any] | None = None,
        env_file: Path | None = None,
        clients: SdkClients | None = None,
        env: Mapping[str, str] | None = None,
        price_table: dict[str, Any] | None = None,
        sleep: Callable[[float], None] | None = None,
        asleep: Callable[[float], Awaitable[None]] | None = None,
        rng: Callable[[], float] | None = None,
    ) -> None:
        """Resolve the model and check everything that can be checked offline.

        Args:
            model: The exact model name.
            provider: Override the model's default provider.
            max_retries: Retries after the first attempt.
            timeout: Per-attempt timeout in seconds.
            on_call: Hook or hooks receiving a record after every call.
            tags: Labels added to every record.
            env_file: A specific ``.env``; the nearest one is used when omitted.
            clients: Pre-built SDK clients, for tests.
            env: Variables to use instead of loading them, for tests.
            price_table: A price table to use instead of loading it, for tests.
            sleep: Blocking sleep, for tests.
            asleep: Async sleep, for tests.
            rng: Jitter source, for tests.

        Raises:
            UnknownModel: If the name does not resolve.
            MissingCredential: If the provider's settings are absent.
        """
        self.route = resolve(model, provider)
        self.max_retries = max_retries
        self.timeout = timeout
        if on_call is None:
            self._hooks: list[Hook] = []
        elif isinstance(on_call, list):
            self._hooks = list(on_call)
        else:
            self._hooks = [on_call]
        self._tags = dict(tags or {})
        merged = dict(env) if env is not None else load_env(env_file)
        creds = credentials_for(self.route.provider, merged)
        self._clients = (
            clients if clients is not None else make_clients(self.route, creds, timeout)
        )
        table = price_table if price_table is not None else load_table()
        self.rates = resolve_rates(self.route, table or {})
        if self.rates is None:
            warn_unpriced(self.route)
        self._transport = TRANSPORTS[self.route.transport]
        self._sleep = sleep or time.sleep
        self._asleep = asleep or asyncio.sleep
        self._rng = rng or random.random

    @property
    def model(self) -> str:
        """The requested model name."""
        return self.route.model

    @property
    def provider(self) -> str:
        """The provider serving this model."""
        return self.route.provider

    @property
    def efforts(self) -> tuple[str, ...]:
        """The effort rungs this model serves."""
        return self.route.spec.ladder

    def cost(self, usage: Usage) -> float | None:
        """Price token counts at this model's list rates.

        Args:
            usage: Token counts.

        Returns:
            USD, or ``None`` when unknown.
        """
        return compute_cost(usage, self.rates)

    def _prepare(
        self,
        prompt: Prompt,
        system: str,
        effort: str | None,
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        cache_prefix: bool,
        tools: list[Tool] | None,
        tool_choice: str | None,
        schema: Schema | None,
    ) -> Call:
        """Validate options against the model and build a provider-neutral call.

        Args:
            prompt: A user message, or the whole conversation.
            system: System instructions.
            effort: A ladder rung, or ``None``.
            max_tokens: Output cap.
            temperature: Sampling temperature, or ``None``.
            top_p: Nucleus cutoff, or ``None``.
            cache_prefix: Whether to request explicit prompt caching.
            tools: Tools the model may call.
            tool_choice: ``auto``, ``none``, ``required`` or a tool name.
            schema: Structured-output schema, or ``None``.

        Returns:
            The call.

        Raises:
            UnsupportedEffort: If the rung is not served.
            UnsupportedFeature: If sampling, forced tool choice or structured output
                is requested from a model that rejects it.
            ValueError: If the prompt is empty or the tool choice names no tool.
        """
        spec = self.route.spec
        check_effort(self.route, effort)
        if (temperature is not None or top_p is not None) and not spec.sampling:
            raise UnsupportedFeature(f"{self.model} rejects temperature and top_p")
        tool_list = list(tools or [])
        if tool_choice is not None and tool_choice not in ("auto", "none"):
            if not spec.forced_tool_choice:
                raise UnsupportedFeature(
                    f"{self.model} rejects forced tool choice; use 'auto' and say which "
                    "tool to use in the prompt"
                )
            if tool_choice != "required" and tool_choice not in {
                t.name for t in tool_list
            }:
                raise ValueError(f"tool_choice {tool_choice!r} names no offered tool")
        if schema is not None and not spec.structured_output:
            raise UnsupportedFeature(f"{self.model} has no structured-output mode")
        messages = [Message.user(prompt)] if isinstance(prompt, str) else list(prompt)
        if not messages:
            raise ValueError("prompt is empty")
        return Call(
            route=self.route,
            messages=messages,
            system=system,
            effort=effort,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            cache_prefix=cache_prefix,
            tools=tool_list,
            tool_choice=tool_choice,
            schema=close_objects(to_json_schema(schema)) if schema is not None else None,
            schema_name=schema_name(schema) if schema is not None else "output",
        )

    def _emit(
        self,
        call: Call,
        result: Result | None,
        error: BaseException | None,
        attempts: int,
        latency_ms: int,
        tags: Mapping[str, Any] | None,
    ) -> None:
        """Hand a record to every hook; hook exceptions propagate.

        Args:
            call: The request.
            result: The result, or ``None`` on failure.
            error: The failure, or ``None``.
            attempts: Attempts made.
            latency_ms: Wall-clock duration.
            tags: Labels for this call, merged over the instance's.
        """
        if not self._hooks:
            return
        record = build_record(
            call,
            result=result,
            error=error,
            attempts=attempts,
            latency_ms=latency_ms,
            tags={**self._tags, **(tags or {})},
        )
        for hook in self._hooks:
            hook(record)

    def _finish(
        self,
        call: Call,
        reply: Reply,
        attempts: int,
        latency_ms: int,
        schema: Schema | None,
        tags: Mapping[str, Any] | None,
    ) -> Result:
        """Price a reply, parse structured output, record the call.

        Args:
            call: The request.
            reply: The transport's parsed reply.
            attempts: Attempts made.
            latency_ms: Wall-clock duration.
            schema: The requested output schema, or ``None``.
            tags: Labels for this call's record.

        Returns:
            The result.

        Raises:
            SchemaError: If structured output does not validate (after recording).
        """
        usage = Usage(
            model=reply.served_model or self.model,
            provider=self.provider,
            effort=call.effort,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            cached_input_tokens=reply.cached_input_tokens,
            cache_write_tokens=reply.cache_write_tokens,
            latency_ms=latency_ms,
        )
        # Zero counts from a reply without usage mean unknown, not free.
        usage.cost = compute_cost(usage, self.rates) if reply.usage_reported else None
        message = Message(
            ROLE_ASSISTANT,
            reply.text,
            tool_calls=list(reply.tool_calls),
            provider_state=reply.provider_state,
        )
        result = Result(
            text=reply.text,
            thinking=reply.thinking,
            tool_calls=list(reply.tool_calls),
            stop_reason=reply.stop_reason,
            usage=usage,
            message=message,
        )
        if schema is not None and reply.stop_reason != STOP_TOOL_USE:
            try:
                result.parsed = parse_output(schema, reply.text)
            except SchemaError as err:
                self._emit(call, result, err, attempts, latency_ms, tags)
                raise
        self._emit(call, result, None, attempts, latency_ms, tags)
        return result

    def complete(
        self,
        prompt: Prompt,
        *,
        system: str = "",
        effort: str | None = None,
        max_tokens: int = 16000,
        temperature: float | None = None,
        top_p: float | None = None,
        cache_prefix: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        schema: Schema | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> Result:
        """Make one non-streaming call.

        Args:
            prompt: A user message, or the whole conversation.
            system: System instructions.
            effort: A ladder rung, or ``None`` for the provider default.
            max_tokens: Output cap.
            temperature: Sampling temperature, or ``None`` to send none.
            top_p: Nucleus cutoff, or ``None`` to send none.
            cache_prefix: Request explicit prompt caching where it is not automatic.
            tools: Tools the model may call.
            tool_choice: ``auto``, ``none``, ``required`` or a tool name.
            schema: A Pydantic model or JSON schema for structured output.
            tags: Labels for this call's record.

        Returns:
            The result.

        Raises:
            LLMKitError: Any llmkit error; see :mod:`llmkit.errors`.
        """
        call = self._prepare(
            prompt, system, effort, max_tokens, temperature, top_p,
            cache_prefix, tools, tool_choice, schema,
        )  # fmt: skip
        body = self._transport.build(call)
        started = time.monotonic()

        def attempt() -> Reply:
            raw = self._transport.send(self._clients.sync, body)
            return self._transport.parse(call, raw)

        try:
            reply, attempts = call_with_retries(
                attempt,
                provider=self.provider,
                max_retries=self.max_retries,
                sleep=self._sleep,
                rng=self._rng,
            )
        except Exception as err:
            self._emit(
                call, None, err, getattr(err, "attempts", 1), _elapsed_ms(started), tags
            )
            raise
        return self._finish(call, reply, attempts, _elapsed_ms(started), schema, tags)

    async def acomplete(
        self,
        prompt: Prompt,
        *,
        system: str = "",
        effort: str | None = None,
        max_tokens: int = 16000,
        temperature: float | None = None,
        top_p: float | None = None,
        cache_prefix: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        schema: Schema | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> Result:
        """Async counterpart of :meth:`complete`; same arguments and result.

        Args:
            prompt: A user message, or the whole conversation.
            system: System instructions.
            effort: A ladder rung, or ``None`` for the provider default.
            max_tokens: Output cap.
            temperature: Sampling temperature, or ``None`` to send none.
            top_p: Nucleus cutoff, or ``None`` to send none.
            cache_prefix: Request explicit prompt caching where it is not automatic.
            tools: Tools the model may call.
            tool_choice: ``auto``, ``none``, ``required`` or a tool name.
            schema: A Pydantic model or JSON schema for structured output.
            tags: Labels for this call's record.

        Returns:
            The result.

        Raises:
            LLMKitError: Any llmkit error; see :mod:`llmkit.errors`.
        """
        call = self._prepare(
            prompt, system, effort, max_tokens, temperature, top_p,
            cache_prefix, tools, tool_choice, schema,
        )  # fmt: skip
        body = self._transport.build(call)
        started = time.monotonic()

        async def attempt() -> Reply:
            raw = await self._transport.asend(self._clients.async_, body)
            return self._transport.parse(call, raw)

        try:
            reply, attempts = await acall_with_retries(
                attempt,
                provider=self.provider,
                max_retries=self.max_retries,
                sleep=self._asleep,
                rng=self._rng,
            )
        except Exception as err:
            self._emit(
                call, None, err, getattr(err, "attempts", 1), _elapsed_ms(started), tags
            )
            raise
        return self._finish(call, reply, attempts, _elapsed_ms(started), schema, tags)


__all__ = ["LLM", "Prompt", "Schema"]
