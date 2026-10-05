"""Review tier: plumbing.

llmkit: one client for the LLMs the research projects call.
"""

from ._version import __version__
from .client import LLM
from .errors import (
    ContentFiltered,
    FatalRequest,
    LLMKitError,
    MissingCredential,
    RequestError,
    RequestTimeout,
    RetriesExhausted,
    SchemaError,
    TransientError,
    UnknownModel,
    UnsupportedEffort,
    UnsupportedFeature,
)
from .pricing import UnpricedModelWarning
from .records import CallRecord, JsonlLog
from .types import (
    LADDER,
    Done,
    Event,
    Image,
    Message,
    Result,
    Text,
    TextDelta,
    ThinkingDelta,
    Tool,
    ToolCall,
    ToolCallDelta,
    Usage,
)

__all__ = [
    "LADDER",
    "LLM",
    "CallRecord",
    "ContentFiltered",
    "Done",
    "Event",
    "FatalRequest",
    "Image",
    "JsonlLog",
    "LLMKitError",
    "Message",
    "MissingCredential",
    "RequestError",
    "RequestTimeout",
    "Result",
    "RetriesExhausted",
    "SchemaError",
    "Text",
    "TextDelta",
    "ThinkingDelta",
    "Tool",
    "ToolCall",
    "ToolCallDelta",
    "TransientError",
    "UnknownModel",
    "UnpricedModelWarning",
    "UnsupportedEffort",
    "UnsupportedFeature",
    "Usage",
    "__version__",
]
