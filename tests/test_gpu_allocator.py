import pytest

from eyetools.core.errors import PredictionError
from eyetools.core.gpu_allocator import (
    GpuAllocationError,
    GpuInfo,
    _apply_visible_device_mapping,
    _parse_cuda_visible_devices,
    is_cuda_oom,
    select_gpu,
)


def test_select_gpu_chooses_largest_free_gpu_that_fits():
    gpus = [
        GpuInfo(index=0, total_mb=24000, free_mb=12000, used_mb=12000, name="A"),
        GpuInfo(index=1, total_mb=24000, free_mb=18000, used_mb=6000, name="B"),
        GpuInfo(index=2, total_mb=24000, free_mb=8000, used_mb=16000, name="C"),
    ]

    allocation = select_gpu(required_mb=10000, gpus=gpus)

    assert allocation.physical_id == 1
    assert allocation.device == "cuda:1"
    assert allocation.free_mb_at_allocation == 18000


def test_select_gpu_excludes_failed_gpu():
    gpus = [
        GpuInfo(index=0, total_mb=24000, free_mb=20000, used_mb=4000, name="A"),
        GpuInfo(index=1, total_mb=24000, free_mb=16000, used_mb=8000, name="B"),
    ]

    allocation = select_gpu(required_mb=10000, gpus=gpus, exclude={0})

    assert allocation.physical_id == 1


def test_select_gpu_keeps_logical_and_physical_ids_separate():
    gpus = [
        GpuInfo(
            index=0,
            physical_id=2,
            total_mb=24000,
            free_mb=20000,
            used_mb=4000,
            name="A",
        ),
        GpuInfo(
            index=1,
            physical_id=4,
            total_mb=24000,
            free_mb=16000,
            used_mb=8000,
            name="B",
        ),
    ]

    allocation = select_gpu(required_mb=10000, gpus=gpus)

    assert allocation.logical_id == 0
    assert allocation.physical_id == 2
    assert allocation.device == "cuda:0"
    assert allocation.visible_device == "cuda:0"


def test_select_gpu_excludes_by_physical_id():
    gpus = [
        GpuInfo(
            index=0,
            physical_id=2,
            total_mb=24000,
            free_mb=20000,
            used_mb=4000,
            name="A",
        ),
        GpuInfo(
            index=1,
            physical_id=4,
            total_mb=24000,
            free_mb=16000,
            used_mb=8000,
            name="B",
        ),
    ]

    allocation = select_gpu(required_mb=10000, gpus=gpus, exclude={2})

    assert allocation.logical_id == 1
    assert allocation.physical_id == 4
    assert allocation.device == "cuda:1"


def test_parse_cuda_visible_devices_maps_logical_to_physical_ids():
    assert _parse_cuda_visible_devices("2,4") == {0: 2, 1: 4}
    assert _parse_cuda_visible_devices(None) is None
    assert _parse_cuda_visible_devices("") == {}
    assert _parse_cuda_visible_devices("-1") == {}
    assert _parse_cuda_visible_devices("GPU-abc") == {}


def test_apply_visible_device_mapping_leaves_inventory_unmasked(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus = [
        GpuInfo(index=2, physical_id=2, total_mb=24000, free_mb=12000, used_mb=12000),
        GpuInfo(index=4, physical_id=4, total_mb=24000, free_mb=18000, used_mb=6000),
    ]

    assert _apply_visible_device_mapping(gpus) == gpus


def test_apply_visible_device_mapping_hides_invalid_or_disabled_masks():
    gpus = [
        GpuInfo(index=2, physical_id=2, total_mb=24000, free_mb=12000, used_mb=12000),
        GpuInfo(index=4, physical_id=4, total_mb=24000, free_mb=18000, used_mb=6000),
    ]

    assert _apply_visible_device_mapping(gpus, "") == []
    assert _apply_visible_device_mapping(gpus, "-1") == []
    assert _apply_visible_device_mapping(gpus, "GPU-abc") == []


def test_apply_visible_device_mapping_normalizes_physical_provider_ids():
    gpus = [
        GpuInfo(index=2, physical_id=2, total_mb=24000, free_mb=12000, used_mb=12000),
        GpuInfo(index=4, physical_id=4, total_mb=24000, free_mb=18000, used_mb=6000),
        GpuInfo(index=5, physical_id=5, total_mb=24000, free_mb=8000, used_mb=16000),
    ]

    mapped = _apply_visible_device_mapping(gpus, "2,4")

    assert [(gpu.index, gpu.physical_id) for gpu in mapped] == [(0, 2), (1, 4)]
    assert [gpu.free_mb for gpu in mapped] == [12000, 18000]


def test_select_gpu_raises_when_no_gpu_has_enough_memory():
    gpus = [GpuInfo(index=0, total_mb=24000, free_mb=4000, used_mb=20000, name="A")]

    with pytest.raises(GpuAllocationError) as exc:
        select_gpu(required_mb=8000, gpus=gpus)

    assert "required_vram_mb=8000" in str(exc.value)
    assert "gpu=0 free_mb=4000" in str(exc.value)


def test_is_cuda_oom_detects_torch_and_worker_messages():
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB"))
    assert is_cuda_oom({"error": "Model load failed: torch.cuda.OutOfMemoryError"})
    assert is_cuda_oom(
        {
            "error": "Model load failed",
            "trace": "... torch.cuda.OutOfMemoryError: CUDA out of memory ...",
        }
    )
    assert not is_cuda_oom({"error": "java.lang.OutOfMemoryError: Java heap space"})
    assert not is_cuda_oom(
        {
            "error": "java.lang.OutOfMemoryError: Java heap space",
            "trace": "/tmp/cuda_worker.py",
        }
    )
    assert not is_cuda_oom(RuntimeError("file missing"))


def test_is_cuda_oom_detects_prediction_error_cause_chain():
    try:
        try:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        except RuntimeError as exc:
            raise PredictionError("prediction failed") from exc
    except PredictionError as exc:
        assert is_cuda_oom(exc)


def test_is_cuda_oom_ignores_non_cuda_prediction_error_cause_chain():
    try:
        try:
            raise RuntimeError("java.lang.OutOfMemoryError: Java heap space")
        except RuntimeError as exc:
            raise PredictionError("prediction failed") from exc
    except PredictionError as exc:
        assert not is_cuda_oom(exc)
