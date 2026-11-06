from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Iterable
import time

import yaml

from .config_loader import parse_config_file
from .registry import ToolMeta

PRECISION_FACTORS = {
    "auto": 1.0,
    "fp32": 1.0,
    "float32": 1.0,
    "fp16": 0.5,
    "float16": 0.5,
    "bf16": 0.5,
    "int8": 0.25,
}

DEFAULT_ESTIMATION_PROFILE = {"runtime_multiplier": 2.5, "activation_margin_mb": 512}


@dataclass(frozen=True)
class WeightResolution:
    files: list[Path]
    gpu_required: bool = True


def estimate_from_files(
    files: Iterable[Path],
    precision: str = "auto",
    runtime_multiplier: float = 2.5,
    activation_margin_mb: int = 512,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    workspace_root = (workspace_root or Path.cwd()).resolve()
    resolved_files = [Path(path).resolve() for path in files]
    factor = PRECISION_FACTORS.get(precision, 1.0)
    file_entries = []
    total_bytes = 0
    for path in resolved_files:
        stat = path.stat()
        total_bytes += stat.st_size
        try:
            display_path = str(path.relative_to(workspace_root))
        except ValueError:
            display_path = str(path)
        file_entries.append(
            {
                "path": display_path,
                "size_bytes": stat.st_size,
                "size_mb": ceil(stat.st_size / (1024 * 1024)),
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    weight_size_mb = ceil(total_bytes / (1024 * 1024))
    param_vram_mb = ceil(weight_size_mb * factor)
    required_vram_mb = ceil(param_vram_mb * runtime_multiplier + activation_margin_mb)
    return {
        "source": "weight_size",
        "weight_size_mb": weight_size_mb,
        "precision": precision,
        "precision_factor": factor,
        "param_vram_mb": param_vram_mb,
        "runtime_multiplier": runtime_multiplier,
        "activation_margin_mb": activation_margin_mb,
        "required_vram_mb": required_vram_mb,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": file_entries,
    }


def is_cached_estimate_valid(
    estimate: dict[str, Any],
    workspace_root: Path | None = None,
    *,
    expected_precision: str | None = None,
    expected_runtime_multiplier: float | None = None,
    expected_activation_margin_mb: int | None = None,
) -> bool:
    workspace_root = (workspace_root or Path.cwd()).resolve()
    if not isinstance(estimate, dict):
        return False
    source = estimate.get("source")
    if source not in {"weight_size", "cpu_only"}:
        return False
    required_vram_mb = estimate.get("required_vram_mb")
    if not _is_number(required_vram_mb):
        return False
    files = estimate.get("files", [])
    if not isinstance(files, list):
        return False
    if source == "cpu_only":
        if required_vram_mb != 0:
            return False
        if estimate.get("gpu_required") is not False:
            return False
        return files == []
    for key in [
        "weight_size_mb",
        "precision_factor",
        "param_vram_mb",
        "runtime_multiplier",
        "activation_margin_mb",
    ]:
        if not _is_number(estimate.get(key)):
            return False
    if (
        expected_precision is not None
        and estimate.get("precision") != expected_precision
    ):
        return False
    if expected_runtime_multiplier is not None and not _numeric_equal(
        estimate["runtime_multiplier"], expected_runtime_multiplier
    ):
        return False
    if (
        expected_activation_margin_mb is not None
        and estimate["activation_margin_mb"] != expected_activation_margin_mb
    ):
        return False
    expected_precision_factor = PRECISION_FACTORS.get(
        str(estimate.get("precision")), 1.0
    )
    if estimate["precision_factor"] != expected_precision_factor:
        return False
    if not files:
        return False
    total_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            return False
        path_value = item.get("path")
        if not path_value:
            return False
        if not isinstance(path_value, (str, Path)):
            return False
        if not _is_int(item.get("size_mb")):
            return False
        if not _is_int(item.get("size_bytes")):
            return False
        if not _is_int(item.get("mtime_ns")):
            return False
        try:
            path = Path(path_value)
            if not path.is_absolute():
                path = workspace_root / path
            if not path.exists():
                return False
            stat = path.stat()
        except (OSError, RuntimeError, ValueError):
            return False
        total_bytes += stat.st_size
        if stat.st_size != item.get("size_bytes"):
            return False
        if ceil(stat.st_size / (1024 * 1024)) != item.get("size_mb"):
            return False
        if stat.st_mtime_ns != item.get("mtime_ns"):
            return False
    expected_weight_size_mb = ceil(total_bytes / (1024 * 1024))
    expected_param_vram_mb = ceil(
        expected_weight_size_mb * estimate["precision_factor"]
    )
    expected_required_vram_mb = ceil(
        expected_param_vram_mb * estimate["runtime_multiplier"]
        + estimate["activation_margin_mb"]
    )
    if estimate["weight_size_mb"] != expected_weight_size_mb:
        return False
    if estimate["param_vram_mb"] != expected_param_vram_mb:
        return False
    if estimate["required_vram_mb"] != expected_required_vram_mb:
        return False
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _numeric_equal(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-9


def estimate_tool_vram(
    meta: ToolMeta, workspace_root: Path | None = None
) -> dict[str, Any]:
    workspace_root = (workspace_root or Path.cwd()).resolve()
    resolution = resolve_weight_files(meta, workspace_root)
    if not resolution.gpu_required:
        return {
            "source": "cpu_only",
            "gpu_required": False,
            "required_vram_mb": 0,
            "files": [],
        }
    if not resolution.files:
        raise FileNotFoundError(
            f"No weight files resolved for tool {meta.id} "
            f"(package={meta.package or 'unknown'})"
        )
    profile = DEFAULT_ESTIMATION_PROFILE
    model_estimate_cfg = (
        meta.model.get("vram_estimate", {}) if isinstance(meta.model, dict) else {}
    )
    runtime_multiplier = float(
        model_estimate_cfg.get("runtime_multiplier", profile["runtime_multiplier"])
    )
    activation_margin_mb = int(
        model_estimate_cfg.get("activation_margin_mb", profile["activation_margin_mb"])
    )
    precision = str(meta.model.get("precision", "auto"))
    estimate = estimate_from_files(
        resolution.files,
        precision=precision,
        runtime_multiplier=runtime_multiplier,
        activation_margin_mb=activation_margin_mb,
        workspace_root=workspace_root,
    )
    estimate["gpu_required"] = True
    return estimate


def build_estimate_report(
    metas: Iterable[ToolMeta], workspace_root: Path | None = None
) -> dict[str, Any]:
    items = []
    for meta in metas:
        item: dict[str, Any] = {
            "tool_id": meta.id,
            "package": meta.package,
            "variant": meta.variant,
        }
        try:
            estimate = estimate_tool_vram(meta, workspace_root=workspace_root)
            files = estimate.get("files", [])
            gpu_required = estimate.get("gpu_required", True)
            item["status"] = (
                "ok" if gpu_required is False or bool(files) else "missing_weights"
            )
            item["estimate"] = estimate
        except FileNotFoundError as exc:
            item["status"] = "missing_weights"
            item["error"] = str(exc)
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)
        items.append(item)
    return {"count": len(items), "items": items}


def _metas_for_config(config_path: Path) -> list[ToolMeta]:
    tool_dir = config_path.parent
    metas = []
    for config in parse_config_file(config_path):
        metas.append(
            ToolMeta(
                id=config["id"],
                entry=config["entry"],
                version=config.get("version", "0.1.0"),
                package=config.get("package"),
                variant=config.get("variant"),
                runtime=config.get("runtime", {}),
                model=config.get("model", {}),
                params=config.get("params", {}),
                warmup=config.get("warmup", {}),
                io=config.get("io", {}),
                tags=config.get("tags", []),
                root_dir=str(tool_dir),
                extra_requires=config.get("extra_requires", []),
                python=config.get("python") or config.get("shared", {}).get("python"),
                category=config.get("category"),
                environment_ref=(
                    config.get("environment_ref")
                    or config.get("shared", {}).get("environment_ref")
                ),
            )
        )
    return metas


def write_estimates_to_config(
    config_path: Path, workspace_root: Path | None = None
) -> bool:
    config_path = Path(config_path)
    workspace_root = (workspace_root or Path.cwd()).resolve()
    original = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(original) or {}
    metas = _metas_for_config(config_path)

    changed = False
    if isinstance(data.get("variants"), list):
        variants_by_name = _variants_by_unique_name(data["variants"], config_path)
        for meta in metas:
            variant_config = variants_by_name.get(meta.variant)
            if variant_config is None:
                continue
            changed = (
                _write_meta_estimate(
                    variant_config, meta, workspace_root=workspace_root
                )
                or changed
            )
    elif isinstance(data.get("tools"), list):
        for tool_config, meta in zip(data["tools"], metas):
            if not isinstance(tool_config, dict):
                continue
            changed = (
                _write_meta_estimate(tool_config, meta, workspace_root=workspace_root)
                or changed
            )
    else:
        for meta in metas:
            changed = (
                _write_meta_estimate(data, meta, workspace_root=workspace_root)
                or changed
            )

    if not changed:
        return False

    serialized = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if serialized == original:
        return False
    config_path.write_text(serialized, encoding="utf-8")
    return True


def _variants_by_unique_name(
    variants: list[Any], config_path: Path
) -> dict[Any, dict[str, Any]]:
    variants_by_name = {}
    seen = set()
    duplicates = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        name = variant.get("variant")
        if name in seen:
            duplicates.append("<missing>" if name is None else str(name))
            continue
        seen.add(name)
        variants_by_name[name] = variant
    if duplicates:
        duplicate_names = ", ".join(sorted(set(duplicates)))
        raise ValueError(
            f"Duplicate variant name(s) in {config_path}: {duplicate_names}"
        )
    return variants_by_name


def _write_meta_estimate(
    config: dict[str, Any], meta: ToolMeta, workspace_root: Path
) -> bool:
    try:
        estimate = estimate_tool_vram(meta, workspace_root=workspace_root)
    except FileNotFoundError:
        return False

    model = config.setdefault("model", {})
    if not isinstance(model, dict):
        model = {}
        config["model"] = model
    weight_paths = [file_info["path"] for file_info in estimate.get("files", [])]

    changed = False
    existing_estimate = model.get("vram_estimate")
    if not _estimates_equivalent(existing_estimate, estimate):
        model["vram_estimate"] = estimate
        changed = True
    if model.get("weight_paths") != weight_paths:
        model["weight_paths"] = weight_paths
        changed = True
    return changed


def _estimates_equivalent(left: Any, right: dict[str, Any]) -> bool:
    if not isinstance(left, dict):
        return False
    left_without_timestamp = {
        key: value for key, value in left.items() if key != "computed_at"
    }
    right_without_timestamp = {
        key: value for key, value in right.items() if key != "computed_at"
    }
    return left_without_timestamp == right_without_timestamp


def resolve_weight_files(meta: ToolMeta, workspace_root: Path) -> WeightResolution:
    if _is_cpu_only(meta):
        return WeightResolution([], gpu_required=False)
    files = [Path(path) for path in (meta.model or {}).get("weight_paths", [])]
    return WeightResolution(
        [path if path.is_absolute() else workspace_root / path for path in files]
    )


def _is_cpu_only(meta: ToolMeta) -> bool:
    params = meta.params or {}
    runtime = meta.runtime or {}
    model = meta.model or {}
    if params.get("cuda_provider") is False:
        return True
    if params.get("device") == "cpu":
        return True
    if runtime.get("device") == "cpu":
        return True
    if runtime.get("device_policy") in {"cpu", "cpu_only"}:
        return True
    return model.get("device") == "cpu"
