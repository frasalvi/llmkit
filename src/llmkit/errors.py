"""Review tier: plumbing.

Every exception llmkit raises. Errors raised before a request is sent describe a
misuse (unknown model, missing credential, unsupported option); errors raised by a
request carry the provider's status, request id and body so a failure can be traced
to the provider's own logs.
"""

from __future__ import annotations

from typing import Any


class LLMKitError(Exception):
    """Base class for every error llmkit raises."""


class UnknownModel(LLMKitError):
    """The model name matches no registered model and no family rule."""


class MissingCredential(LLMKitError):
    """A provider's required settings are absent."""

    def __init__(self, provider: str, missing: list[str]) -> None:
        """Name the provider and exactly which settings are missing.

        Args:
            provider: The provider that cannot be reached.
            missing: The setting names that are unset or empty.
        """
        self.provider = provider
        self.missing = list(missing)
        super().__init__(
            f"{provider} needs {', '.join(self.missing)}; set them in .env or the "
            "environment"
        )


class UnsupportedEffort(LLMKitError):
    """The requested effort rung is not served by this model."""


class UnsupportedFeature(LLMKitError):
    """The request asks for a capability this route cannot serve."""


class RequestError(LLMKitError):
    """A request reached the provider, or tried to, and failed."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status: int | None = None,
        request_id: str | None = None,
        body: Any = None,
        retry_after: float | None = None,
    ) -> None:
        """Record what the provider reported.

        Args:
            message: Human-readable description.
            provider: The provider that failed.
            status: HTTP status, when there was a response.
            request_id: The provider's request id, when it sent one.
            body: The parsed error body, when there was one.
            retry_after: Seconds the provider asked us to wait, when it said.
        """
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.request_id = request_id
        self.body = body
        self.retry_after = retry_after
        self.attempts = 1


class ContentFiltered(RequestError):
    """The provider's content filter rejected the request or the response."""

    def __init__(
        self, message: str, *, categories: list[str] | None = None, **kwargs: Any
    ) -> None:
        """Record which filter categories fired.

        Args:
            message: Human-readable description.
            categories: Filter categories reported as triggered.
            **kwargs: Passed to :class:`RequestError`.
        """
        super().__init__(message, **kwargs)
        self.categories = list(categories or [])


class FatalRequest(RequestError):
    """A failure that will not succeed on retry (400, 401, 403, 404, other 4xx)."""


class RequestTimeout(RequestError):
    """One attempt exceeded its timeout."""


class TransientError(RequestError):
    """A failure worth retrying: 408, 429, 5xx or a dropped connection."""


class RetriesExhausted(RequestError):
    """Every attempt failed with a retryable error."""

    def __init__(self, last: RequestError, attempts: int) -> None:
        """Wrap the final retryable error.

        Args:
            last: The error from the final attempt.
            attempts: How many attempts were made.
        """
        super().__init__(
            f"gave up after {attempts} attempts: {last}",
            provider=last.provider,
            status=last.status,
            request_id=last.request_id,
            body=last.body,
        )
        self.last = last
        self.attempts = attempts


class SchemaError(LLMKitError):
    """The model's output did not match the requested schema."""

    def __init__(self, message: str, *, raw: str) -> None:
        """Keep the raw output for inspection.

        Args:
            message: What failed to validate.
            raw: The text the model returned.
        """
        super().__init__(message)
        self.raw = raw
