"""Image input normalization for tool requests."""

from __future__ import annotations

import base64
import binascii
import re
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any


IMAGE_PATH_KEYS = {"image_path", "original_image_path", "segmentation_image_path"}


def is_image_path_key(key: str) -> bool:
    return key in IMAGE_PATH_KEYS or key.endswith("_image_path")


def save_image_bytes(data: bytes, filename: str | None = None) -> str:
    suffix = _suffix_from_filename(filename)
    upload_dir = Path(tempfile.gettempdir()) / "eyetools_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(data)
    return str(path)


def normalize_image_source(value: Any) -> Any:
    if isinstance(value, dict):
        encoded = value.get("base64") or value.get("data")
        if isinstance(encoded, str):
            return save_image_bytes(
                _decode_base64_payload(encoded), value.get("filename")
            )
        return {
            key: (
                normalize_image_source(item)
                if key in {"original", "segmentation"}
                else item
            )
            for key, item in value.items()
        }
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value:
            header, encoded = value.split(",", 1)
            media_type = header.split(";", 1)[0].removeprefix("data:")
            return save_image_bytes(
                _decode_base64_payload(encoded), _filename_for_media_type(media_type)
            )
        raw = _try_decode_raw_base64_image(value)
        if raw is not None:
            return save_image_bytes(raw, "image.png")
    return value


def normalize_request_images(request: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(request)
    inputs = _inputs_dict(normalized)
    _promote_legacy_image_fields(inputs)
    for key, value in list(inputs.items()):
        if key == "image_paths":
            inputs.pop(key, None)
            continue
        if key in {"original_image_path", "segmentation_image_path"}:
            continue
        if is_image_path_key(key):
            if isinstance(value, list):
                inputs[key] = [normalize_image_source(item) for item in value]
            else:
                inputs[key] = normalize_image_source(value)
    _add_legacy_vessel_fields(inputs)
    return normalized


def expand_image_path_batches(request: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = normalize_request_images(request)
    inputs = _inputs_dict(normalized)
    list_fields = {
        key: value
        for key, value in inputs.items()
        if is_image_path_key(key) and isinstance(value, list)
    }
    if not list_fields:
        return [normalized]

    lengths = {len(value) for value in list_fields.values()}
    if 0 in lengths:
        raise ValueError("image_path lists must not be empty")
    if len(lengths) != 1:
        raise ValueError("multiple image_path list fields must have the same length")

    count = lengths.pop()
    expanded = []
    for index in range(count):
        item = deepcopy(normalized)
        item_inputs = _inputs_dict(item)
        for key, values in list_fields.items():
            item_inputs[key] = values[index]
        _add_legacy_vessel_fields(item_inputs)
        expanded.append(item)
    return expanded


def _inputs_dict(request: dict[str, Any]) -> dict[str, Any]:
    inputs = request.get("inputs") if isinstance(request, dict) else None
    return inputs if isinstance(inputs, dict) else request


def _promote_legacy_image_fields(inputs: dict[str, Any]) -> None:
    if "image_path" not in inputs and "image_paths" in inputs:
        inputs["image_path"] = inputs["image_paths"]
    if "image_path" not in inputs:
        original = inputs.get("original_image_path")
        segmentation = inputs.get("segmentation_image_path")
        if original is not None or segmentation is not None:
            inputs["image_path"] = {
                "original": original,
                "segmentation": segmentation,
            }


def _add_legacy_vessel_fields(inputs: dict[str, Any]) -> None:
    image_path = inputs.get("image_path")
    if not isinstance(image_path, dict):
        return
    original = image_path.get("original")
    segmentation = image_path.get("segmentation")
    if original is not None:
        inputs["original_image_path"] = original
    if segmentation is not None:
        inputs["segmentation_image_path"] = segmentation


def _decode_base64_payload(value: str) -> bytes:
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    return base64.b64decode(value, validate=True)


def _try_decode_raw_base64_image(value: str) -> bytes | None:
    compact = value.strip()
    if len(compact) < 64 or not re.fullmatch(r"[A-Za-z0-9+/=\s]+", compact):
        return None
    try:
        data = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError):
        return None
    if data.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a")):
        return data
    return None


def _suffix_from_filename(filename: str | None) -> str:
    if not filename:
        return ".png"
    suffix = Path(filename).suffix.lower()
    return (
        suffix
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
        else ".png"
    )


def _filename_for_media_type(media_type: str) -> str:
    mapping = {
        "image/jpeg": "image.jpg",
        "image/jpg": "image.jpg",
        "image/png": "image.png",
        "image/bmp": "image.bmp",
        "image/gif": "image.gif",
        "image/tiff": "image.tiff",
    }
    return mapping.get(media_type.lower(), "image.png")
