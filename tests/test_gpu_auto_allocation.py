from pathlib import Path
import sys

import pytest

from eyetools.core import gpu_allocator
from eyetools.core.errors import PredictionError
from eyetools.core.registry import ToolMeta, ToolRegistry
from eyetools.core.tool_manager import ToolManager


def _write_tool_module(root: Path, module_name: str) -> None:
    (root / f"{module_name}.py").write_text(
        """
class DeviceEchoTool:
    def __init__(self, meta, params):
        self.meta = meta
        self.params = params

    def predict(self, request):
        return {"device": self.params.get("device")}


class CapturedDeviceTool:
    def __init__(self, meta, params):
        self.meta = meta
        self.params = params
        self.initial_device = params.get("device")

    def predict(self, request):
        return {
            "initial_device": self.initial_device,
            "current_device": self.params.get("device"),
        }


class CapturedLoadDeviceTool:
    def __init__(self, meta, params):
        self.meta = meta
        self.params = params
        self.initial_device = params.get("device")
        self.load_device = None

    def ensure_model_loaded(self):
        self.load_device = self.params.get("device")

    def predict(self, request):
        return {
            "initial_device": self.initial_device,
            "load_device": self.load_device,
            "current_device": self.params.get("device"),
        }


class CountingCapturedDeviceTool:
    created = 0

    def __init__(self, meta, params):
        self.meta = meta
        self.params = params
        type(self).created += 1
        self.created = type(self).created
        self.initial_device = params.get("device")

    def predict(self, request):
        return {
            "created": self.created,
            "initial_device": self.initial_device,
            "current_device": self.params.get("device"),
        }


class OomOnceTool:
    attempts = 0
    created = 0

    def __init__(self, meta, params):
        self.meta = meta
        self.params = params
        type(self).created += 1

    def predict(self, request):
        type(self).attempts += 1
        if type(self).attempts == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return {
            "device": self.params.get("device"),
            "attempts": type(self).attempts,
            "created": type(self).created,
        }

    def shutdown(self):
        pass
""",
        encoding="utf-8",
    )


def _registry_with(meta: ToolMeta) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(meta)
    return registry


def test_inproc_auto_allocation_selects_largest_satisfying_gpu(tmp_path, monkeypatch):
    module_name = "fake_gpu_tools_echo"
    _write_tool_module(tmp_path, module_name)
    meta = ToolMeta(
        id="fake:echo",
        entry=f"{module_name}:DeviceEchoTool",
        version="1",
        runtime={"load_mode": "inproc"},
        model={"vram_estimate": {"required_vram_mb": 10_000}},
        params={},
        root_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        gpu_allocator,
        "list_gpus",
        lambda: [
            gpu_allocator.GpuInfo(
                index=0, total_mb=24_000, free_mb=12_000, used_mb=12_000
            ),
            gpu_allocator.GpuInfo(
                index=1, total_mb=32_000, free_mb=24_000, used_mb=8_000
            ),
            gpu_allocator.GpuInfo(
                index=2, total_mb=16_000, free_mb=8_000, used_mb=8_000
            ),
        ],
    )

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)

    result = tm.predict(meta.id, {"inputs": {}})

    assert result == {"device": "cuda:1"}
    assert meta.params["device"] == "cuda:1"
    allocation = tm._gpu_allocations[meta.id]
    assert allocation.device == "cuda:1"
    assert allocation.physical_id == 1
    assert allocation.required_mb == 10_000


def test_inproc_persistent_instance_reuses_existing_gpu_allocation(
    tmp_path, monkeypatch
):
    module_name = "fake_gpu_tools_persistent"
    _write_tool_module(tmp_path, module_name)
    meta = ToolMeta(
        id="fake:persistent",
        entry=f"{module_name}:CapturedDeviceTool",
        version="1",
        runtime={"load_mode": "inproc"},
        model={"vram_estimate": {"required_vram_mb": 10_000}},
        params={},
        root_dir=str(tmp_path),
    )
    calls = []

    def list_gpus():
        calls.append(None)
        if len(calls) == 1:
            return [
                gpu_allocator.GpuInfo(
                    index=0, total_mb=24_000, free_mb=20_000, used_mb=4_000
                ),
                gpu_allocator.GpuInfo(
                    index=1, total_mb=24_000, free_mb=12_000, used_mb=12_000
                ),
            ]
        return [
            gpu_allocator.GpuInfo(
                index=0, total_mb=24_000, free_mb=12_000, used_mb=12_000
            ),
            gpu_allocator.GpuInfo(
                index=1, total_mb=24_000, free_mb=20_000, used_mb=4_000
            ),
        ]

    monkeypatch.setattr(gpu_allocator, "list_gpus", list_gpus)

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)

    first = tm.predict(meta.id, {"inputs": {}})
    second = tm.predict(meta.id, {"inputs": {}})

    assert first == {"initial_device": "cuda:0", "current_device": "cuda:0"}
    assert second == {"initial_device": "cuda:0", "current_device": "cuda:0"}
    assert len(calls) == 1
    assert meta.params["device"] == "cuda:0"
    assert tm._gpu_allocations[meta.id].physical_id == 0


def test_subprocess_predict_passes_gpu_allocation_and_reuses_live_worker(
    tmp_path, monkeypatch
):
    meta = ToolMeta(
        id="fake:subprocess",
        entry="fake_module:FakeTool",
        version="1",
        runtime={"load_mode": "subprocess"},
        model={"vram_estimate": {"required_vram_mb": 10_000}},
        params={},
        root_dir=str(tmp_path),
    )
    gpu_queries = []

    def list_gpus():
        gpu_queries.append(None)
        if len(gpu_queries) == 1:
            return [
                gpu_allocator.GpuInfo(
                    index=0, total_mb=24_000, free_mb=12_000, used_mb=12_000
                ),
                gpu_allocator.GpuInfo(
                    index=2, total_mb=32_000, free_mb=24_000, used_mb=8_000
                ),
            ]
        raise AssertionError("live subprocess worker should reuse allocation")

    class LiveProcess:
        def poll(self):
            return None

    class LiveWorker:
        process = LiveProcess()

    class FakeProcessManager:
        def __init__(self):
            self._workers = {}
            self.init_payloads = []

        def ensure_init(self, tool_meta, gpu_allocation=None):
            self.init_payloads.append(gpu_allocation)
            self._workers.setdefault(tool_meta.id, LiveWorker())

        def request(self, tool_id, payload, timeout=None):
            return {"ok": True, "data": {"device": meta.params.get("device")}}

    monkeypatch.setattr(gpu_allocator, "list_gpus", list_gpus)
    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)
    fake_pm = FakeProcessManager()
    tm._process_manager = fake_pm

    first = tm.predict(meta.id, {"inputs": {}})
    second = tm.predict(meta.id, {"inputs": {}})

    assert first == {"device": "cuda:0"}
    assert second == {"device": "cuda:0"}
    assert fake_pm.init_payloads == [
        {"physical_id": 2, "visible_device": "cuda:0"},
        {"physical_id": 2, "visible_device": "cuda:0"},
    ]
    assert len(gpu_queries) == 1
    assert meta.params["device"] == "cuda:0"
    assert tm._gpu_allocations[meta.id].physical_id == 2


def test_subprocess_ensure_loaded_allocates_before_worker_start_and_predict_reuses(
    tmp_path, monkeypatch
):
    meta = ToolMeta(
        id="fake:subprocess-ensure-loaded",
        entry="fake_module:FakeTool",
        version="1",
        runtime={"load_mode": "subprocess"},
        model={"vram_estimate": {"required_vram_mb": 10_000}},
        params={},
        root_dir=str(tmp_path),
    )
    gpu_queries = []

    def list_gpus():
        gpu_queries.append(None)
        if len(gpu_queries) == 1:
            return [
                gpu_allocator.GpuInfo(
                    index=0, total_mb=24_000, free_mb=18_000, used_mb=6_000
                )
            ]
        raise AssertionError("predict should reuse ensure_loaded allocation")

    class LiveProcess:
        def poll(self):
            return None

    class LiveWorker:
        process = LiveProcess()

    class FakeProcessManager:
        def __init__(self):
            self._workers = {}
            self.init_payloads = []

        def ensure_init(self, tool_meta, gpu_allocation=None):
            self.init_payloads.append(gpu_allocation)
            self._workers.setdefault(tool_meta.id, LiveWorker())

        def request(self, tool_id, payload, timeout=None):
            return {"ok": True, "data": {"device": meta.params.get("device")}}

    monkeypatch.setattr(gpu_allocator, "list_gpus", list_gpus)
    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)
    fake_pm = FakeProcessManager()
    tm._process_manager = fake_pm

    loaded = tm.ensure_loaded(meta.id, include_model=False)
    result = tm.predict(meta.id, {"inputs": {}})

    assert loaded == {
        "tool_id": meta.id,
        "mode": "subprocess",
        "status": "worker_started",
    }
    assert result == {"device": "cuda:0"}
    assert fake_pm.init_payloads == [
        {"physical_id": 0, "visible_device": "cuda:0"},
        {"physical_id": 0, "visible_device": "cuda:0"},
    ]
    assert len(gpu_queries) == 1
    assert tm._gpu_allocations[meta.id].physical_id == 0


def test_subprocess_predict_restarts_live_gpu_worker_missing_allocation(
    tmp_path, monkeypatch
):
    meta = ToolMeta(
        id="fake:subprocess-missing-allocation",
        entry="fake_module:FakeTool",
        version="1",
        runtime={"load_mode": "subprocess"},
        model={"vram_estimate": {"required_vram_mb": 10_000}},
        params={},
        root_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        gpu_allocator,
        "list_gpus",
        lambda: [
            gpu_allocator.GpuInfo(
                index=2, total_mb=32_000, free_mb=24_000, used_mb=8_000
            )
        ],
    )

    class LiveProcess:
        def poll(self):
            return None

    class LiveWorker:
        process = LiveProcess()

    class FakeProcessManager:
        def __init__(self):
            self._workers = {meta.id: LiveWorker()}
            self.stopped = []
            self.init_payloads = []

        def stop(self, tool_id):
            self.stopped.append(tool_id)
            self._workers.pop(tool_id, None)

        def ensure_init(self, tool_meta, gpu_allocation=None):
            self.init_payloads.append(gpu_allocation)
            self._workers[tool_meta.id] = LiveWorker()

        def request(self, tool_id, payload, timeout=None):
            return {"ok": True, "data": {"device": meta.params.get("device")}}

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)
    fake_pm = FakeProcessManager()
    tm._process_manager = fake_pm

    result = tm.predict(meta.id, {"inputs": {}})

    assert result == {"device": "cuda:0"}
    assert fake_pm.stopped == [meta.id]
    assert fake_pm.init_payloads == [{"physical_id": 2, "visible_device": "cuda:0"}]
    assert tm._gpu_allocations[meta.id].physical_id == 2


def test_subprocess_immediate_cleanup_clears_gpu_allocation(tmp_path, monkeypatch):
    meta = ToolMeta(
        id="fake:subprocess-immediate",
        entry="fake_module:FakeTool",
        version="1",
        runtime={"load_mode": "subprocess", "cleanup_policy": "immediate"},
        model={"vram_estimate": {"required_vram_mb": 10_000}},
        params={},
        root_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        gpu_allocator,
        "list_gpus",
        lambda: [
            gpu_allocator.GpuInfo(
                index=0, total_mb=24_000, free_mb=18_000, used_mb=6_000
            )
        ],
    )

    class LiveProcess:
        def poll(self):
            return None

    class LiveWorker:
        process = LiveProcess()

    class FakeProcessManager:
        def __init__(self):
            self._workers = {}
            self.stopped = []

        def ensure_init(self, tool_meta, gpu_allocation=None):
            self._workers[tool_meta.id] = LiveWorker()

        def request(self, tool_id, payload, timeout=None):
            return {"ok": True, "data": {"device": meta.params.get("device")}}

        def stop(self, tool_id):
            self.stopped.append(tool_id)
            self._workers.pop(tool_id, None)

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)
    fake_pm = FakeProcessManager()
    tm._process_manager = fake_pm

    result = tm.predict(meta.id, {"inputs": {}})

    assert result == {"device": "cuda:0"}
    assert fake_pm.stopped == [meta.id]
    assert meta.id not in tm._gpu_allocations


def test_persistent_subprocess_predict_marks_lifecycle_loaded(tmp_path):
    meta = ToolMeta(
        id="fake:subprocess-persist",
        entry="fake_module:FakeTool",
        version="1",
        runtime={"load_mode": "subprocess", "cleanup_policy": "persist"},
        model={},
        params={},
        root_dir=str(tmp_path),
    )

    class LiveProcess:
        def poll(self):
            return None

    class LiveWorker:
        process = LiveProcess()

    class FakeProcessManager:
        def __init__(self):
            self._workers = {}
            self.stopped = []

        def ensure_init(self, tool_meta, gpu_allocation=None):
            self._workers[tool_meta.id] = LiveWorker()

        def request(self, tool_id, payload, timeout=None):
            return {"ok": True, "data": {"ok": True}}

        def stop(self, tool_id):
            self.stopped.append(tool_id)
            self._workers.pop(tool_id, None)

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)
    fake_pm = FakeProcessManager()
    tm._process_manager = fake_pm

    assert tm.predict(meta.id, {"inputs": {}}) == {"ok": True}
    assert tm.lifecycle_states()[meta.id] == "LOADED"
    assert tm.lifecycle_states_detailed()[meta.id]["loaded"] is True
    assert fake_pm.stopped == []


def test_maintenance_unloads_idle_subprocess_worker(tmp_path):
    meta = ToolMeta(
        id="fake:idle-subprocess",
        entry="fake_module:FakeTool",
        version="1",
        runtime={"load_mode": "subprocess", "cleanup_policy": "persist"},
        model={},
        params={},
        root_dir=str(tmp_path),
    )

    class LiveProcess:
        def poll(self):
            return None

    class LiveWorker:
        process = LiveProcess()
        last_used = 100.0

    class FakeProcessManager:
        def __init__(self):
            self._workers = {meta.id: LiveWorker()}
            self.stopped = []

        def stop(self, tool_id):
            self.stopped.append(tool_id)
            self._workers.pop(tool_id, None)

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)
    fake_pm = FakeProcessManager()
    tm._process_manager = fake_pm
    tm._lifecycle[meta.id] = "LOADED"

    result = tm.maintenance(mark_idle_s=1, unload_idle_s=1)

    assert fake_pm.stopped == [meta.id]
    assert result["actions"]["unloaded"] == [meta.id]
    assert result["after"][meta.id] == "UNLOADED"


def test_subprocess_cuda_oom_retries_on_different_gpu(tmp_path, monkeypatch):
    meta = ToolMeta(
        id="fake:subprocess-oom",
        entry="fake_module:FakeTool",
        version="1",
        runtime={
            "load_mode": "subprocess",
            "device_policy": "auto",
            "oom_retry": True,
            "oom_retry_max": 1,
        },
        model={"vram_estimate": {"required_vram_mb": 10_000}},
        params={},
        root_dir=str(tmp_path),
    )

    monkeypatch.setattr(
        gpu_allocator,
        "list_gpus",
        lambda: [
            gpu_allocator.GpuInfo(
                index=0, total_mb=24_000, free_mb=22_000, used_mb=2_000
            ),
            gpu_allocator.GpuInfo(
                index=1, total_mb=24_000, free_mb=18_000, used_mb=6_000
            ),
        ],
    )

    class LiveProcess:
        def poll(self):
            return None

    class LiveWorker:
        process = LiveProcess()

    class FakeProcessManager:
        def __init__(self):
            self._workers = {}
            self.init_payloads = []
            self.requests = []
            self.stopped = []

        def ensure_init(self, tool_meta, gpu_allocation=None):
            self.init_payloads.append(gpu_allocation)
            self._workers[tool_meta.id] = LiveWorker()

        def request(self, tool_id, payload, timeout=None):
            self.requests.append((tool_id, payload))
            if len(self.requests) == 1:
                return {"ok": False, "error": "CUDA out of memory"}
            return {"ok": True, "data": {"status": "ok"}}

        def stop(self, tool_id):
            self.stopped.append(tool_id)
            self._workers.pop(tool_id, None)

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)
    fake_pm = FakeProcessManager()
    tm._process_manager = fake_pm

    result = tm.predict(meta.id, {"inputs": {}})

    assert result == {"status": "ok"}
    assert fake_pm.init_payloads == [
        {"physical_id": 0, "visible_device": "cuda:0"},
        {"physical_id": 1, "visible_device": "cuda:0"},
    ]
    assert fake_pm.stopped == [meta.id]
    assert tm.get_metrics(meta.id)["predict_count"] == 1


def test_subprocess_cuda_oom_after_ensure_loaded_retries_on_different_gpu(
    tmp_path, monkeypatch
):
    meta = ToolMeta(
        id="fake:subprocess-oom-preloaded",
        entry="fake_module:FakeTool",
        version="1",
        runtime={
            "load_mode": "subprocess",
            "device_policy": "auto",
            "oom_retry": True,
            "oom_retry_max": 1,
        },
        model={"vram_estimate": {"required_vram_mb": 10_000}},
        params={},
        root_dir=str(tmp_path),
    )

    monkeypatch.setattr(
        gpu_allocator,
        "list_gpus",
        lambda: [
            gpu_allocator.GpuInfo(
                index=0, total_mb=24_000, free_mb=22_000, used_mb=2_000
            ),
            gpu_allocator.GpuInfo(
                index=1, total_mb=24_000, free_mb=18_000, used_mb=6_000
            ),
        ],
    )

    class LiveProcess:
        def poll(self):
            return None

    class LiveWorker:
        process = LiveProcess()

    class FakeProcessManager:
        def __init__(self):
            self._workers = {}
            self.init_payloads = []
            self.requests = []
            self.stopped = []

        def ensure_init(self, tool_meta, gpu_allocation=None):
            self.init_payloads.append(gpu_allocation)
            self._workers[tool_meta.id] = LiveWorker()

        def request(self, tool_id, payload, timeout=None):
            self.requests.append((tool_id, payload))
            if len(self.requests) == 1:
                return {"ok": False, "error": "CUDA out of memory"}
            return {"ok": True, "data": {"status": "ok"}}

        def stop(self, tool_id):
            self.stopped.append(tool_id)
            self._workers.pop(tool_id, None)

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)
    fake_pm = FakeProcessManager()
    tm._process_manager = fake_pm

    loaded = tm.ensure_loaded(meta.id)
    result = tm.predict(meta.id, {"inputs": {}})

    assert loaded == {
        "tool_id": meta.id,
        "mode": "subprocess",
        "status": "worker_started",
    }
    assert result == {"status": "ok"}
    assert fake_pm.init_payloads == [
        {"physical_id": 0, "visible_device": "cuda:0"},
        {"physical_id": 0, "visible_device": "cuda:0"},
        {"physical_id": 1, "visible_device": "cuda:0"},
    ]
    assert fake_pm.stopped == [meta.id]
    assert tm._gpu_allocations[meta.id].physical_id == 1
    assert tm.get_metrics(meta.id)["predict_count"] == 1


def test_subprocess_fixed_policy_cuda_oom_does_not_retry(tmp_path, monkeypatch):
    meta = ToolMeta(
        id="fake:subprocess-fixed-oom",
        entry="fake_module:FakeTool",
        version="1",
        runtime={
            "load_mode": "subprocess",
            "device_policy": "fixed",
            "device": "cuda:3",
            "oom_retry": True,
        },
        model={"vram_estimate": {"required_vram_mb": 10_000}},
        params={},
        root_dir=str(tmp_path),
    )

    def fail_select_gpu(**kwargs):
        raise AssertionError("select_gpu should not be called")

    class LiveProcess:
        def poll(self):
            return None

    class LiveWorker:
        process = LiveProcess()

    class FakeProcessManager:
        def __init__(self):
            self._workers = {}
            self.requests = 0
            self.stopped = []

        def ensure_init(self, tool_meta, gpu_allocation=None):
            assert gpu_allocation is None
            self._workers[tool_meta.id] = LiveWorker()

        def request(self, tool_id, payload, timeout=None):
            self.requests += 1
            return {"ok": False, "error": "CUDA out of memory"}

        def stop(self, tool_id):
            self.stopped.append(tool_id)
            self._workers.pop(tool_id, None)

    monkeypatch.setattr(gpu_allocator, "select_gpu", fail_select_gpu)
    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)
    fake_pm = FakeProcessManager()
    tm._process_manager = fake_pm

    with pytest.raises(PredictionError, match="CUDA out of memory"):
        tm.predict(meta.id, {"inputs": {}})

    assert fake_pm.requests == 1
    assert fake_pm.stopped == []
    assert meta.params["device"] == "cuda:3"


def test_subprocess_gpu_required_false_cuda_oom_does_not_retry(tmp_path, monkeypatch):
    meta = ToolMeta(
        id="fake:subprocess-cpu-oom",
        entry="fake_module:FakeTool",
        version="1",
        runtime={"load_mode": "subprocess", "oom_retry": True},
        model={
            "vram_estimate": {
                "gpu_required": False,
                "required_vram_mb": 10_000,
            }
        },
        params={},
        root_dir=str(tmp_path),
    )

    def fail_select_gpu(**kwargs):
        raise AssertionError("select_gpu should not be called")

    class LiveProcess:
        def poll(self):
            return None

    class LiveWorker:
        process = LiveProcess()

    class FakeProcessManager:
        def __init__(self):
            self._workers = {}
            self.requests = 0
            self.stopped = []

        def ensure_init(self, tool_meta, gpu_allocation=None):
            assert gpu_allocation is None
            self._workers[tool_meta.id] = LiveWorker()

        def request(self, tool_id, payload, timeout=None):
            self.requests += 1
            return {"ok": False, "error": "CUDA out of memory"}

        def stop(self, tool_id):
            self.stopped.append(tool_id)
            self._workers.pop(tool_id, None)

    monkeypatch.setattr(gpu_allocator, "select_gpu", fail_select_gpu)
    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)
    fake_pm = FakeProcessManager()
    tm._process_manager = fake_pm

    with pytest.raises(PredictionError, match="CUDA out of memory"):
        tm.predict(meta.id, {"inputs": {}})

    assert fake_pm.requests == 1
    assert fake_pm.stopped == []
    assert meta.id not in tm._gpu_allocations


def test_inproc_predict_recreates_preloaded_instance_missing_gpu_allocation(
    tmp_path, monkeypatch
):
    module_name = "fake_gpu_tools_preloaded"
    _write_tool_module(tmp_path, module_name)
    meta = ToolMeta(
        id="fake:preloaded",
        entry=f"{module_name}:CountingCapturedDeviceTool",
        version="1",
        runtime={"load_mode": "inproc"},
        model={"vram_estimate": {"required_vram_mb": 10_000}},
        params={},
        root_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        gpu_allocator,
        "list_gpus",
        lambda: [
            gpu_allocator.GpuInfo(
                index=0, total_mb=24_000, free_mb=18_000, used_mb=6_000
            )
        ],
    )
    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)

    loaded = tm.ensure_loaded(meta.id, include_model=False)
    assert loaded == {"tool_id": meta.id, "mode": "inproc", "status": "loaded"}
    assert meta.id in tm._inproc_instances
    assert tm._gpu_allocations[meta.id].physical_id == 0
    tm._gpu_allocations.pop(meta.id)

    result = tm.predict(meta.id, {"inputs": {}})

    assert result == {
        "created": 2,
        "initial_device": "cuda:0",
        "current_device": "cuda:0",
    }
    assert meta.params["device"] == "cuda:0"
    assert tm._gpu_allocations[meta.id].physical_id == 0


def test_inproc_ensure_loaded_allocates_before_model_load(tmp_path, monkeypatch):
    module_name = "fake_gpu_tools_ensure_loaded"
    _write_tool_module(tmp_path, module_name)
    meta = ToolMeta(
        id="fake:ensure-loaded",
        entry=f"{module_name}:CapturedLoadDeviceTool",
        version="1",
        runtime={"load_mode": "inproc"},
        model={"lazy": True, "vram_estimate": {"required_vram_mb": 10_000}},
        params={},
        root_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        gpu_allocator,
        "list_gpus",
        lambda: [
            gpu_allocator.GpuInfo(
                index=1, total_mb=24_000, free_mb=18_000, used_mb=6_000
            )
        ],
    )
    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)

    result = tm.ensure_loaded(meta.id, include_model=True)

    inst = tm._inproc_instances[meta.id]
    assert result == {"tool_id": meta.id, "mode": "inproc", "status": "loaded"}
    assert inst.initial_device == "cuda:1"
    assert inst.load_device == "cuda:1"
    assert meta.params["device"] == "cuda:1"
    assert tm._gpu_allocations[meta.id].physical_id == 1


def test_inproc_auto_gpu_missing_vram_estimate_skips_preallocation(
    tmp_path, monkeypatch
):
    module_name = "fake_gpu_tools_missing_estimate"
    _write_tool_module(tmp_path, module_name)
    meta = ToolMeta(
        id="fake:missing-estimate",
        entry=f"{module_name}:DeviceEchoTool",
        version="1",
        runtime={"load_mode": "inproc", "device_policy": "auto"},
        model={},
        params={},
        root_dir=str(tmp_path),
    )
    monkeypatch.delenv("EYETOOLS_GPU_DEFAULT_REQUIRED_MB", raising=False)

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)

    result = tm.predict(meta.id, {"inputs": {}})

    assert result == {"device": None}
    assert meta.id in tm._inproc_instances
    assert meta.id not in tm._gpu_allocations


def test_subprocess_auto_gpu_missing_vram_estimate_skips_preallocation(
    tmp_path, monkeypatch
):
    meta = ToolMeta(
        id="fake:subprocess-missing-estimate",
        entry="fake_module:FakeTool",
        version="1",
        runtime={"load_mode": "subprocess", "device_policy": "auto"},
        model={},
        params={},
        root_dir=str(tmp_path),
    )
    monkeypatch.delenv("EYETOOLS_GPU_DEFAULT_REQUIRED_MB", raising=False)

    class FakeProcessManager:
        _workers = {}
        allocation = "unset"

        def ensure_init(self, tool_meta, gpu_allocation=None):
            self.allocation = gpu_allocation

        def request(self, tool_id, payload, timeout=None):
            return {"ok": True, "data": {"device": meta.params.get("device")}}

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)
    fake_pm = FakeProcessManager()
    tm._process_manager = fake_pm

    result = tm.predict(meta.id, {"inputs": {}})

    assert result == {"device": None}
    assert fake_pm.allocation is None
    assert meta.id not in tm._gpu_allocations


def test_auto_gpu_explicit_gpu_required_missing_vram_estimate_fails(
    tmp_path, monkeypatch
):
    module_name = "fake_gpu_tools_gpu_required_missing_estimate"
    _write_tool_module(tmp_path, module_name)
    meta = ToolMeta(
        id="fake:gpu-required-missing-estimate",
        entry=f"{module_name}:DeviceEchoTool",
        version="1",
        runtime={
            "load_mode": "inproc",
            "device_policy": "auto",
            "gpu_required": True,
        },
        model={},
        params={},
        root_dir=str(tmp_path),
    )
    monkeypatch.delenv("EYETOOLS_GPU_DEFAULT_REQUIRED_MB", raising=False)

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)

    with pytest.raises(
        ValueError,
        match=r"fake:gpu-required-missing-estimate.*required_vram_mb",
    ):
        tm.predict(meta.id, {"inputs": {}})


def test_inproc_auto_gpu_missing_vram_estimate_uses_positive_env_fallback(
    tmp_path, monkeypatch
):
    module_name = "fake_gpu_tools_env_fallback"
    _write_tool_module(tmp_path, module_name)
    meta = ToolMeta(
        id="fake:env-fallback",
        entry=f"{module_name}:DeviceEchoTool",
        version="1",
        runtime={"load_mode": "inproc", "device_policy": "auto"},
        model={},
        params={},
        root_dir=str(tmp_path),
    )
    monkeypatch.setenv("EYETOOLS_GPU_DEFAULT_REQUIRED_MB", "1000")
    monkeypatch.setattr(
        gpu_allocator,
        "list_gpus",
        lambda: [
            gpu_allocator.GpuInfo(
                index=0, total_mb=24_000, free_mb=12_000, used_mb=12_000
            )
        ],
    )

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)

    result = tm.predict(meta.id, {"inputs": {}})

    assert result == {"device": "cuda:0"}
    assert tm._gpu_allocations[meta.id].required_mb == 1000


def test_inproc_vram_estimate_gpu_required_false_skips_allocation(
    tmp_path, monkeypatch
):
    module_name = "fake_gpu_tools_cpu_only"
    _write_tool_module(tmp_path, module_name)
    meta = ToolMeta(
        id="fake:cpu-only",
        entry=f"{module_name}:DeviceEchoTool",
        version="1",
        runtime={"load_mode": "inproc"},
        model={
            "vram_estimate": {
                "gpu_required": False,
                "required_vram_mb": 1234,
            }
        },
        params={},
        root_dir=str(tmp_path),
    )

    def fail_select_gpu(**kwargs):
        raise AssertionError("select_gpu should not be called")

    monkeypatch.setattr(gpu_allocator, "select_gpu", fail_select_gpu)

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)

    result = tm.predict(meta.id, {"inputs": {}})

    assert result == {"device": None}
    assert "device" not in meta.params
    assert meta.id not in tm._gpu_allocations


def test_inproc_cpu_only_policy_skips_gpu_allocation(tmp_path, monkeypatch):
    module_name = "fake_gpu_tools_cpu_only_policy"
    _write_tool_module(tmp_path, module_name)
    meta = ToolMeta(
        id="fake:cpu-only-policy",
        entry=f"{module_name}:DeviceEchoTool",
        version="1",
        runtime={"load_mode": "inproc", "device_policy": "cpu_only"},
        model={"vram_estimate": {"required_vram_mb": 1234}},
        params={},
        root_dir=str(tmp_path),
    )

    def fail_select_gpu(**kwargs):
        raise AssertionError("select_gpu should not be called")

    monkeypatch.setenv("EYETOOLS_GPU_DEFAULT_REQUIRED_MB", "9999")
    monkeypatch.setattr(gpu_allocator, "select_gpu", fail_select_gpu)

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)

    result = tm.predict(meta.id, {"inputs": {}})

    assert result == {"device": "cpu"}
    assert meta.params["device"] == "cpu"
    assert meta.id not in tm._gpu_allocations


def test_inproc_fixed_policy_cuda_oom_does_not_retry(tmp_path, monkeypatch):
    module_name = "fake_gpu_tools_fixed_oom"
    _write_tool_module(tmp_path, module_name)
    meta = ToolMeta(
        id="fake:fixed-oom",
        entry=f"{module_name}:OomOnceTool",
        version="1",
        runtime={
            "load_mode": "inproc",
            "device_policy": "fixed",
            "device": "cuda:3",
            "oom_retry": True,
        },
        model={"vram_estimate": {"required_vram_mb": 10_000}},
        params={},
        root_dir=str(tmp_path),
    )

    def fail_select_gpu(**kwargs):
        raise AssertionError("select_gpu should not be called")

    monkeypatch.setattr(gpu_allocator, "select_gpu", fail_select_gpu)
    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)

    with pytest.raises(PredictionError, match="CUDA out of memory"):
        tm.predict(meta.id, {"inputs": {}})

    assert sys.modules[module_name].OomOnceTool.attempts == 1
    assert meta.params["device"] == "cuda:3"
    assert meta.id not in tm._gpu_allocations


def test_inproc_cuda_oom_retries_on_different_gpu_and_recreates_tool(
    tmp_path, monkeypatch
):
    module_name = "fake_gpu_tools_oom"
    _write_tool_module(tmp_path, module_name)
    meta = ToolMeta(
        id="fake:oom-once",
        entry=f"{module_name}:OomOnceTool",
        version="1",
        runtime={"load_mode": "inproc", "oom_retry": True},
        model={"vram_estimate": {"required_vram_mb": 10_000}},
        params={},
        root_dir=str(tmp_path),
    )

    def list_gpus():
        return [
            gpu_allocator.GpuInfo(
                index=0, total_mb=24_000, free_mb=12_000, used_mb=12_000
            ),
            gpu_allocator.GpuInfo(
                index=1, total_mb=32_000, free_mb=24_000, used_mb=8_000
            ),
        ]

    monkeypatch.setattr(gpu_allocator, "list_gpus", list_gpus)

    tm = ToolManager(_registry_with(meta), workspace_root=tmp_path)

    result = tm.predict(meta.id, {"inputs": {}})

    assert result == {"device": "cuda:0", "attempts": 2, "created": 2}
    assert meta.params["device"] == "cuda:0"
    allocation = tm._gpu_allocations[meta.id]
    assert allocation.device == "cuda:0"
    assert allocation.physical_id == 0
