"""Canonical tool names and JSON Schema normalization.

The MCP surface exposes ``tool_name`` values that are stable, lowercase,
underscore-delimited identifiers.  The registry keeps the historical ``id``
value for internal compatibility, while both values resolve to the same
``ToolMeta`` record.

Tool configuration files may still use the compact ``{field: type}`` notation.
This module turns that notation into a JSON Schema object and adds a small,
explicit contract envelope without changing tool-specific prediction payloads.
"""

from __future__ import annotations

import copy
import re
from typing import Any


TOOL_SCHEMA_VERSION = "1.0.0"


def slugify(value: Any) -> str:
    """Return a stable lowercase identifier fragment."""

    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "tool"


def canonical_tool_name(data: dict[str, Any]) -> str:
    """Build the public no-colon name for one normalized config item.

    An explicit ``tool_name`` takes precedence. Otherwise, append a supplied
    ``tool_model_id`` to distinguish model variants in any package.
    """

    explicit = data.get("tool_name")
    if explicit:
        return slugify(explicit)

    package = slugify(data.get("package") or data.get("name") or "tool")
    variant = slugify(data.get("variant")) if data.get("variant") else ""
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    model_id = (
        data.get("tool_model_id")
        or model.get("tool_model_id")
        or params.get("tool_model_id")
    )
    parts = [package]
    if variant:
        parts.append(variant)
    if model_id:
        parts.append(slugify(model_id))
    return "_".join(parts)


def legacy_tool_ids(data: dict[str, Any]) -> list[str]:
    """Return historical IDs accepted as registry aliases."""

    package = data.get("package")
    variant = data.get("variant")
    category = data.get("category")
    aliases: list[str] = []
    if package and variant:
        short = f"{package}:{variant}"
        aliases.append(short)
        if category:
            aliases.append(f"{category}.{short}")
    elif package:
        aliases.append(str(package))
        if category:
            aliases.append(f"{category}.{package}")
    return list(dict.fromkeys(aliases))


def _type_schema(type_name: str) -> dict[str, Any]:
    nullable = type_name.endswith("?")
    type_name = type_name.rstrip("?")
    if "|" in type_name:
        return {
            "oneOf": [_type_schema(part.strip()) for part in type_name.split("|")],
            **({"nullable": True} if nullable else {}),
        }
    mapping = {
        "string": {"type": "string"},
        "path": {"type": "string", "format": "path"},
        "number": {"type": "number"},
        "float": {"type": "number"},
        "integer": {"type": "integer"},
        "bool": {"type": "boolean"},
        "boolean": {"type": "boolean"},
        "object": {"type": "object"},
        "array": {"type": "array"},
        "null": {"type": "null"},
    }
    schema = copy.deepcopy(mapping.get(type_name, {"type": "string"}))
    if nullable:
        schema["nullable"] = True
    return schema


def _field_description(name: str, *, output: bool) -> str:
    descriptions = {
        "image_path": "Path to one input image.",
        "image_paths": "Ordered paths for a multimodal image group.",
        "modality": "Normalized ophthalmic imaging modality.",
        "labels": "Optional labels or prompts to run.",
        "task": "Tool task or variant identifier.",
        "query": "Question or retrieval query.",
        "language": "Report language code.",
        "prediction": "Primary model prediction.",
        "probabilities": "Probability map keyed by model label.",
        "uncertainty": "Normalized uncertainty score when provided by the model.",
        "low_confidence": "Whether the model output is below its confidence threshold.",
        "detections": "Per-instance detections with labels, scores, boxes, and masks.",
        "masks": "Raw per-instance mask records for client-side rendering.",
        "raw_masks": "Raw mask payload and encoding metadata.",
        "output_paths": "Optional legacy artifacts; clients should prefer structured data.",
        "inference_time": "Inference duration in seconds.",
    }
    return descriptions.get(
        name,
        f"{'Model output' if output else 'Tool input'} field: {name}.",
    )


def _normalize_property(name: str, value: Any, *, output: bool) -> dict[str, Any]:
    if isinstance(value, str):
        schema = _type_schema(value)
    elif isinstance(value, dict):
        schema = copy.deepcopy(value)
        if "type" not in schema and not any(key in schema for key in ("oneOf", "anyOf", "$ref")):
            schema = {"type": "object", "properties": schema}
    else:
        schema = {"type": "string"}
    schema.setdefault("description", _field_description(name, output=output))
    return schema


def normalize_input_schema(raw: Any) -> dict[str, Any]:
    """Normalize compact or full input schema declarations."""

    if isinstance(raw, dict) and raw.get("type"):
        schema = copy.deepcopy(raw)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            schema["properties"] = {
                key: _normalize_property(key, value, output=False)
                for key, value in properties.items()
            }
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        schema.setdefault("additionalProperties", False)
        return schema

    raw_map = raw if isinstance(raw, dict) else {}
    properties = {
        key: _normalize_property(key, value, output=False)
        for key, value in raw_map.items()
    }
    required_defaults = {
        "image_path",
        "query",
        "chart_type",
        "action",
    }
    required = [
        key
        for key, value in raw_map.items()
        if key in required_defaults and not (isinstance(value, str) and value.endswith("?"))
    ]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def normalize_output_schema(raw: Any) -> dict[str, Any]:
    """Normalize compact or full output schema declarations."""

    if isinstance(raw, dict) and raw.get("type"):
        schema = copy.deepcopy(raw)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            schema["properties"] = {
                key: _normalize_property(key, value, output=True)
                for key, value in properties.items()
            }
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return schema

    raw_map = raw if isinstance(raw, dict) else {}
    return {
        "type": "object",
        "properties": {
            key: _normalize_property(key, value, output=True)
            for key, value in raw_map.items()
        },
        "additionalProperties": True,
    }


def normalize_io_schema(
    io: Any,
    *,
    description: str,
    tool_name: str,
) -> dict[str, Any]:
    """Return the canonical input/output schema object stored in ``ToolMeta``."""

    normalized = copy.deepcopy(io) if isinstance(io, dict) else {}
    normalized["schema_version"] = TOOL_SCHEMA_VERSION
    normalized["description"] = description
    normalized["tool_name"] = tool_name
    normalized["input_schema"] = normalize_input_schema(
        normalized.get("input_schema", normalized.get("input", {}))
    )
    normalized["output_schema"] = normalize_output_schema(
        normalized.get("output_schema", {})
    )
    normalized.pop("input", None)
    normalized["contract"] = {
        "schema_version": TOOL_SCHEMA_VERSION,
        "tool_name": tool_name,
        "description": description,
        "input": normalized["input_schema"],
        "output": normalized["output_schema"],
    }
    return normalized


def is_canonical_tool_name(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*", str(value)))


__all__ = [
    "TOOL_SCHEMA_VERSION",
    "canonical_tool_name",
    "is_canonical_tool_name",
    "legacy_tool_ids",
    "normalize_input_schema",
    "normalize_io_schema",
    "normalize_output_schema",
    "slugify",
]
