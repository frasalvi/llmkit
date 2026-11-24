"""Review tier: plumbing.

Azure resource discovery, key retrieval, spend lookup and the ``llmkit azure`` CLI
support.
"""

from llmkit.azure.inventory import Deployment, Resource, Spend
from llmkit.azure.source import Azure, AzureSource, default_azure

__all__ = [
    "Azure",
    "AzureSource",
    "Deployment",
    "Resource",
    "Spend",
    "default_azure",
]
