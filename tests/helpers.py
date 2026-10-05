"""Test helpers shared across modules."""

from __future__ import annotations

from typing import Any

import httpx
import openai


def status_error(
    status: int, *, headers: dict[str, str] | None = None, body: Any = None
) -> openai.APIStatusError:
    """Build an OpenAI SDK status error as the SDK would raise it."""
    request = httpx.Request("POST", "https://example.test/v1/x")
    response = httpx.Response(status, request=request, headers=headers or {})
    return openai.APIStatusError(f"HTTP {status}", response=response, body=body)
