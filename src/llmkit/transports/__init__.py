"""Review tier: plumbing.

The four wire formats llmkit speaks, keyed by transport name.
"""

from .anthropic import AnthropicTransport
from .base import Transport
from .chat import ChatTransport
from .gemini import GeminiTransport
from .responses import ResponsesTransport

TRANSPORTS: dict[str, Transport] = {
    "responses": ResponsesTransport(),
    "anthropic": AnthropicTransport(),
    "gemini": GeminiTransport(),
    "chat": ChatTransport(),
}

__all__ = ["TRANSPORTS", "Transport"]
