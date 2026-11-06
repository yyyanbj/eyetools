from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
from typing import Any, Iterable


@dataclass(frozen=True)
class GpuInfo:
    index: int
    total_mb: int
    free_mb: int
    used_mb: int
    name: str | None = None
    physical_id: int | None = None

    def __post_init__(self) -> None:
        if self.physical_id is None:
            object.__setattr__(self, "physical_id", self.index)


@dataclass(frozen=True)
class DeviceAllocation:
    logical_id: int
    physical_id: int
    device: str
    visible_device: str
    required_mb: int
    free_mb_at_allocation: int
    total_mb: int
    source: str


class GpuAllocationError(RuntimeError):
    pass


def _as_int_mb(value: Any) -> int:
    return int(round(float(value)))


def _parse_cuda_visible_devices(value: str | None) -> dict[int, int] | None:
    if value is None:
        return None
    if not value.strip():
        return {}

    mapping: dict[int, int] = {}
    for logical_id, raw_physical_id in enumerate(value.split(",")):
        raw_physical_id = raw_physical_id.strip()
        if not raw_physical_id:
            return {}
        try:
            physical_id = int(raw_physical_id)
        except ValueError:
            return {}
        if physical_id < 0:
            return {}
        mapping[logical_id] = physical_id
    return mapping


def _apply_visible_device_mapping(
    gpus: Iterable[GpuInfo],
    visible_devices: str | None = None,
) -> list[GpuInfo]:
    mapping = _parse_cuda_visible_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices is None
        else visible_devices
    )
    inventory = list(gpus)
    if mapping is None:
        return inventory
    if not mapping:
        return []

    physical_to_logical = {
        physical_id: logical_id for logical_id, physical_id in mapping.items()
    }
    out: list[GpuInfo] = []
    for gpu in inventory:
        physical_id = gpu.physical_id if gpu.physical_id is not None else gpu.index
        logical_id = physical_to_logical.get(physical_id)
        if logical_id is None:
            continue
        out.append(
            GpuInfo(
                index=logical_id,
                physical_id=physical_id,
                total_mb=gpu.total_mb,
                free_mb=gpu.free_mb,
                used_mb=gpu.used_mb,
                name=gpu.name,
            )
        )
    return out


def list_gpus() -> list[GpuInfo]:
    for provider in (_list_gpus_nvml, _list_gpus_nvidia_smi, _list_gpus_torch):
        try:
            gpus = _apply_visible_device_mapping(provider())
        except Exception:
            continue
        if gpus:
            return gpus
    return []


def _list_gpus_nvml() -> list[GpuInfo]:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    try:
        out: list[GpuInfo] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = (
                raw_name.decode("utf-8")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            total_mb = _as_int_mb(mem.total / (1024 * 1024))
            free_mb = _as_int_mb(mem.free / (1024 * 1024))
            used_mb = _as_int_mb(mem.used / (1024 * 1024))
            out.append(
                GpuInfo(
                    index=index,
                    physical_id=index,
                    total_mb=total_mb,
                    free_mb=free_mb,
                    used_mb=used_mb,
                    name=name,
                )
            )
        return out
    finally:
        pynvml.nvmlShutdown()


def _list_gpus_nvidia_smi() -> list[GpuInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5)
    out: list[GpuInfo] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, total_mb, free_mb, used_mb = parts
        out.append(
            GpuInfo(
                index=int(index),
                physical_id=int(index),
                name=name,
                total_mb=_as_int_mb(total_mb),
                free_mb=_as_int_mb(free_mb),
                used_mb=_as_int_mb(used_mb),
            )
        )
    return out


def _list_gpus_torch() -> list[GpuInfo]:
    import torch  # type: ignore

    if not torch.cuda.is_available():
        return []
    out: list[GpuInfo] = []
    visible_devices = _parse_cuda_visible_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    visible_devices = visible_devices or {}
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        total_mb = _as_int_mb(total_bytes / (1024 * 1024))
        free_mb = _as_int_mb(free_bytes / (1024 * 1024))
        out.append(
            GpuInfo(
                index=index,
                physical_id=visible_devices.get(index, index),
                total_mb=total_mb,
                free_mb=free_mb,
                used_mb=max(total_mb - free_mb, 0),
                name=torch.cuda.get_device_name(index),
            )
        )
    return out


def select_gpu(
    required_mb: int,
    gpus: Iterable[GpuInfo] | None = None,
    exclude: set[int] | None = None,
    source: str = "auto",
) -> DeviceAllocation:
    exclude = exclude or set()
    inventory = list(gpus) if gpus is not None else list_gpus()
    candidates = [gpu for gpu in inventory if gpu.physical_id not in exclude]
    fitting = [gpu for gpu in candidates if gpu.free_mb >= required_mb]
    if not fitting:
        available = ", ".join(
            f"gpu={gpu.index} free_mb={gpu.free_mb} total_mb={gpu.total_mb} "
            f"physical_id={gpu.physical_id}"
            for gpu in candidates
        )
        raise GpuAllocationError(
            f"No GPU satisfies required_vram_mb={required_mb}. "
            f"Available: {available or 'none'}"
        )
    selected = max(fitting, key=lambda gpu: (gpu.free_mb, gpu.total_mb, -gpu.index))
    return DeviceAllocation(
        logical_id=selected.index,
        physical_id=(
            selected.physical_id if selected.physical_id is not None else selected.index
        ),
        device=f"cuda:{selected.index}",
        visible_device="cuda:0",
        required_mb=required_mb,
        free_mb_at_allocation=selected.free_mb,
        total_mb=selected.total_mb,
        source=source,
    )


def _text_indicates_cuda_oom(text: str) -> bool:
    lowered = text.lower()
    return (
        "cuda out of memory" in lowered
        or "cuda error: out of memory" in lowered
        or "torch.cuda.outofmemoryerror" in lowered
    )


def is_cuda_oom(exc_or_response: Any) -> bool:
    if isinstance(exc_or_response, dict):
        text = " ".join(
            str(exc_or_response.get(key, "")) for key in ("error", "message", "trace")
        )
        return _text_indicates_cuda_oom(text)

    seen: set[int] = set()
    stack = [exc_or_response]
    while stack:
        exc = stack.pop()
        exc_id = id(exc)
        if exc_id in seen:
            continue
        seen.add(exc_id)

        exc_type = type(exc)
        text = f"{exc_type.__module__}.{exc_type.__name__}: {exc}"
        if _text_indicates_cuda_oom(text):
            return True

        cause = getattr(exc, "__cause__", None)
        context = getattr(exc, "__context__", None)
        if cause is not None:
            stack.append(cause)
        if context is not None:
            stack.append(context)
    return False
