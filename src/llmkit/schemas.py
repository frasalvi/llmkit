"""Review tier: plumbing.

JSON-schema helpers shared by tools and structured output. Provider strict modes need
every object closed (``additionalProperties: false``); OpenAI's strict mode also needs
every property required, so strictness is requested only when that holds.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from .errors import SchemaError


def is_model(source: Any) -> bool:
    """Return whether *source* is a Pydantic model class."""
    return isinstance(source, type) and issubclass(source, BaseModel)


def to_json_schema(source: dict[str, Any] | type[BaseModel]) -> dict[str, Any]:
    """Return a JSON schema for a Pydantic model or a schema dict.

    Args:
        source: A model class or a schema.

    Returns:
        A fresh schema dict.

    Raises:
        TypeError: If *source* is neither.
    """
    if is_model(source):
        return source.model_json_schema()  # type: ignore[union-attr]
    if isinstance(source, dict):
        return copy.deepcopy(source)
    raise TypeError(f"expected a JSON schema dict or a Pydantic model, got {source!r}")


def _close(node: Any) -> None:
    """Set ``additionalProperties: false`` on every object node, in place."""
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("additionalProperties", False)
        for value in node.values():
            _close(value)
    elif isinstance(node, list):
        for value in node:
            _close(value)


def close_objects(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *schema* with every object closed to extra properties.

    Args:
        schema: A JSON schema.

    Returns:
        The closed copy.
    """
    closed = copy.deepcopy(schema)
    _close(closed)
    return closed


def all_required(schema: dict[str, Any]) -> bool:
    """Return whether every object node requires all of its properties.

    Args:
        schema: A JSON schema.

    Returns:
        True when the schema satisfies OpenAI's strict mode on this point.
    """

    def walk(node: Any) -> bool:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and set(node.get("required", [])) != set(props):
                return False
            return all(walk(v) for v in node.values())
        if isinstance(node, list):
            return all(walk(v) for v in node)
        return True

    return walk(schema)


def schema_name(source: dict[str, Any] | type[BaseModel]) -> str:
    """Return a provider-safe name for a schema.

    Args:
        source: A model class or a schema.

    Returns:
        Letters, digits, ``_`` and ``-`` only, at most 64 characters.
    """
    raw = (
        source.__name__  # type: ignore[union-attr]
        if is_model(source)
        else str(source.get("title") or "output")  # type: ignore[union-attr]
    )
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:64] or "output"


def parse_output(source: dict[str, Any] | type[BaseModel], text: str) -> Any:
    """Parse model output against the requested schema.

    A Pydantic model is fully validated. A schema dict is checked only for valid
    JSON; the provider's strict mode is relied on for the shape.

    Args:
        source: The schema the call requested.
        text: The model's reply.

    Returns:
        A model instance, or the decoded JSON value.

    Raises:
        SchemaError: If the text does not validate.
    """
    if is_model(source):
        try:
            return source.model_validate_json(text)  # type: ignore[union-attr]
        except ValidationError as exc:
            # Without input values: they are model output, kept only in .raw.
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}"
                for e in exc.errors(include_input=False, include_context=False)
            )
            raise SchemaError(
                f"output does not match {schema_name(source)}: {problems}", raw=text
            ) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"output is not valid JSON: {exc}", raw=text) from exc
