"""Review tier: plumbing.

llmkit: one client for the LLMs the research projects call.
"""

from ._version import __version__
from .azure.source import AzureInventoryWarning
from .azure.target import AzureEnvKeyWarning, AzureRoutingWarning
from .batch import Report, aquery, query
from .cache import Cache
from .client import LLM
from .errors import (
    AzureConfigError,
    BudgetExceeded,
    ContentFiltered,
    FatalRequest,
    LLMKitError,
    MissingCredential,
    RequestError,
    RequestTimeout,
    RetriesExhausted,
    SchemaError,
    SpendUnavailable,
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
    "AzureConfigError",
    "AzureEnvKeyWarning",
    "AzureInventoryWarning",
    "AzureRoutingWarning",
    "BudgetExceeded",
    "Cache",
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
    "Report",
    "RequestTimeout",
    "Result",
    "RetriesExhausted",
    "SchemaError",
    "SpendUnavailable",
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
    "aquery",
    "query",
]
