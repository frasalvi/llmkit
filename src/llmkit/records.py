"""Review tier: plumbing.

One record per call, handed to ``on_call`` hooks after the final attempt, success or
failure. :class:`JsonlLog` is the ready-made hook: one JSON line per call, safe across
threads and async tasks in one process (use one file per process).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._version import __version__
from .transports.base import Call
from .types import Image, Message, Result, parts

log = logging.getLogger("llmkit.records")

_CONTENT_FIELDS = ("system", "messages", "text", "thinking", "tool_calls")


@dataclass
class CallRecord:
    """Everything known about one call.

    Attributes:
        timestamp: UTC ISO-8601 time the record was built.
        call_id: Random hex id.
        llmkit_version: The library version that made the call.
        model: Requested model name.
        provider: Provider that served it.
        transport: Wire format used.
        deployment: Name sent to the provider.
        effort: Requested rung, or ``None``.
        max_tokens: Output cap.
        temperature: Sampling temperature sent, or ``None``.
        top_p: Nucleus cutoff sent, or ``None``.
        tools: Names of offered tools.
        schema: Name of the requested output schema, or ``None``.
        system: System prompt.
        messages: The request conversation.
        text: Reply text.
        thinking: Reasoning text.
        tool_calls: Requested tool calls.
        stop_reason: Normalised stop reason, ``None`` on failure.
        usage: Tokens and cost, ``None`` on failure.
        latency_ms: Wall-clock duration including retries.
        attempts: Attempts made.
        error_type: Exception class name on failure.
        error: Exception message on failure.
        tags: Caller-supplied labels.
    """

    timestamp: str
    call_id: str
    llmkit_version: str
    model: str
    provider: str
    transport: str
    deployment: str
    effort: str | None
    max_tokens: int
    temperature: float | None
    top_p: float | None
    tools: list[str]
    schema: str | None
    system: str
    messages: list[dict[str, Any]]
    text: str
    thinking: str
    tool_calls: list[dict[str, Any]]
    stop_reason: str | None
    usage: dict[str, Any] | None
    latency_ms: int
    attempts: int
    error_type: str | None
    error: str | None
    tags: dict[str, Any]

    def to_dict(self, *, content: bool = True) -> dict[str, Any]:
        """Return a JSON-ready dict.

        Args:
            content: Keep prompts and responses; when False they are set to ``None``.

        Returns:
            The record.
        """
        out = asdict(self)
        if not content:
            for name in _CONTENT_FIELDS:
                out[name] = None
        return out


Hook = Callable[[CallRecord], None]


def message_to_dict(message: Message) -> dict[str, Any]:
    """Render a message for a record, hashing images instead of storing them.

    Args:
        message: The message.

    Returns:
        A JSON-ready dict.
    """
    content: str | list[dict[str, Any]]
    if isinstance(message.content, str):
        content = message.content
    else:
        content = [
            {
                "type": "image",
                "media_type": p.media_type,
                "sha256": hashlib.sha256(p.data).hexdigest(),
            }
            if isinstance(p, Image)
            else {"type": "text", "text": p.text}
            for p in parts(message.content)
        ]
    return {
        "role": message.role,
        "content": content,
        "tool_calls": [asdict(tc) for tc in message.tool_calls],
        "tool_call_id": message.tool_call_id,
        "is_error": message.is_error,
    }


def build_record(
    call: Call,
    *,
    result: Result | None,
    error: BaseException | None,
    attempts: int,
    latency_ms: int,
    tags: Mapping[str, Any],
) -> CallRecord:
    """Assemble the record for one call.

    Args:
        call: The request.
        result: The result, or ``None`` on failure.
        error: The failure, or ``None``.
        attempts: Attempts made.
        latency_ms: Wall-clock duration.
        tags: Caller labels.

    Returns:
        The record.
    """
    route = call.route
    return CallRecord(
        timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
        call_id=uuid.uuid4().hex,
        llmkit_version=__version__,
        model=route.model,
        provider=route.provider,
        transport=route.transport,
        deployment=route.deployment,
        effort=call.effort,
        max_tokens=call.max_tokens,
        temperature=call.temperature,
        top_p=call.top_p,
        tools=[t.name for t in call.tools],
        schema=call.schema_name if call.schema is not None else None,
        system=call.system,
        messages=[message_to_dict(m) for m in call.messages],
        text=result.text if result else "",
        thinking=result.thinking if result else "",
        tool_calls=[asdict(tc) for tc in result.tool_calls] if result else [],
        stop_reason=result.stop_reason if result else None,
        usage=asdict(result.usage) if result else None,
        latency_ms=latency_ms,
        attempts=attempts,
        error_type=type(error).__name__ if error else None,
        error=str(error) if error else None,
        tags=dict(tags),
    )


class JsonlLog:
    """An ``on_call`` hook that appends one JSON line per call."""

    def __init__(self, path: str | Path, *, content: bool = True) -> None:
        """Open (lazily) a log file.

        Args:
            path: The JSONL file; parent directories are created.
            content: Keep prompts and responses in each line.
        """
        self.path = Path(path)
        self.content = content
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._tail_checked = False

    def _drop_torn_tail(self) -> None:
        """Remove a partial last line left by a crash, so appends start on a clean line."""
        if not self.path.exists():
            return
        with self.path.open("rb+") as handle:
            size = handle.seek(0, os.SEEK_END)
            end = size
            while end > 0:
                start = max(0, end - 65536)
                handle.seek(start)
                block = handle.read(end - start)
                if end == size and block.endswith(b"\n"):
                    return
                newline = block.rfind(b"\n")
                if newline >= 0:
                    keep = start + newline + 1
                    break
                end = start
            else:
                keep = 0
            handle.truncate(keep)
        log.warning("dropped %d torn trailing bytes from %s", size - keep, self.path)

    def __call__(self, record: CallRecord) -> None:
        """Append *record* as one line.

        Args:
            record: The call record.
        """
        line = json.dumps(
            record.to_dict(content=self.content), ensure_ascii=False, default=str
        )
        with self._lock:
            if not self._tail_checked:
                self._drop_torn_tail()
                self._tail_checked = True
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()


def read(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL call log, skipping a torn final line from a crash mid-write.

    Args:
        path: The log file.

    Returns:
        The records.

    Raises:
        ValueError: If any line other than the last is not valid JSON.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if number == len(lines):
                log.warning("skipping torn last line %d of %s", number, path)
                continue
            raise ValueError(f"corrupt JSON on line {number} of {path}") from exc
    return rows
