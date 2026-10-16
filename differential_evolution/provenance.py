"""JSON-safe descriptions of initial optimizer configuration.

Descriptions of arbitrary user objects are deliberately marked incomplete;
object representations, closures, data and executable code are not serialized.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any


def describe(value: Any, depth: int = 0) -> Any:
    """Describe public constructor fields without traversing runtime state."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth > 12:
        return {"complete": False, "reason": "configuration nesting limit"}
    if isinstance(value, (tuple, list)):
        return [describe(item, depth + 1) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: describe(item, depth + 1) for key, item in value.items()}
    cls = type(value)
    identifier = f"{cls.__module__}.{cls.__qualname__}"
    if is_dataclass(value) and not isinstance(value, type):
        return {"type": identifier, "parameters": {
            item.name: describe(getattr(value, item.name), depth + 1)
            for item in fields(value) if item.init and not item.name.startswith("_")
        }}
    if callable(value):
        identifier = f"{getattr(value, '__module__', cls.__module__)}.{getattr(value, '__qualname__', cls.__qualname__)}"
    return {"type": identifier, "complete": False,
            "reason": "preserve custom implementation and data separately"}
