"""ToolManager: orchestrates lifecycle & dispatch (inproc + subprocess).

Adds lightweight lifecycle state tracking aligned with documented phases:
DISCOVERED -> REGISTERED -> LOADED -> IDLE -> UNLOADED (evicted) -> RELOADED.
"""

from __future__ import annotations
from typing import Dict, Any, Literal
from importlib import import_module
import os
from .registry import ToolRegistry, ToolMeta
from .process_manager import DEFAULT_MODEL_TIMEOUT_S, ProcessManager
from .env_manager import EnvManager
from .errors import ToolNotFoundError, PredictionError
from .image_inputs import expand_image_path_batches, normalize_request_images
from .logger import summarize_for_log
from . import gpu_allocator
import time
import sys
from pathlib import Path

from loguru import logger

LifecycleState = Literal["DISCOVERED", "REGISTERED", "LOADED", "IDLE", "UNLOADED"]
CPU_DEVICE_POLICIES = {"cpu", "cpu_only"}


class ToolManager:
    def __init__(
        self,
        registry: ToolRegistry,
        workspace_root=None,
        max_inproc: int = 16,
        model_timeout_s: float = DEFAULT_MODEL_TIMEOUT_S,
    ):
        self.registry = registry
        self._inproc_instances: Dict[str, Any] = {}
        self._workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self._env_manager = EnvManager(self._workspace_root)
        self._model_timeout_s = float(model_timeout_s)
        self._process_manager = ProcessManager(
            self._workspace_root,
            env_manager=self._env_manager,
            default_timeout_s=self._model_timeout_s,
        )
        self._max_inproc = max_inproc
        self._inproc_lru: Dict[str, float] = {}
        # metrics: per tool id
        self._metrics: Dict[str, Dict[str, Any]] = {}
        # lifecycle states per tool id
        self._lifecycle: Dict[str, LifecycleState] = {}
        # last used timestamp
        self._last_used = {}
        # telemetry collected from subprocess LOAD_MODEL responses
        self._subprocess_telemetry = {}
        self._gpu_allocations: Dict[str, gpu_allocator.DeviceAllocation] = {}
        # last warmup timestamps
        self._last_warmup = {}
        # Track inserted sys.path entries for cleanup (tool_id -> inserted path)
        self._syspath_entries: Dict[str, str] = {}
        self._syspath_parent_entries: Dict[str, str] = {}
        # Track dynamically loaded modules for cleanup (tool_id -> module name)
        self._dynamic_modules: Dict[str, str] = {}

    def resource_snapshot(self) -> Dict[str, Any]:
        """Return a lightweight resource usage snapshot.
        Includes:
          - counts: discovered, loaded_inproc, subprocess_workers
          - memory (RSS in MB) if psutil available
          - aggregate metrics summary
        GPU memory is omitted unless torch + cuda available; then reports total + allocated.
        """
        snap: Dict[str, Any] = {}
        snap["counts"] = {
            "discovered": len(self.registry.list()),
            "loaded_inproc": len(self._inproc_instances),
            "subprocess_workers": len(getattr(self._process_manager, "_workers", {})),
        }
        # attach subprocess telemetry (non-huge scalar values only)
        if getattr(self, "_subprocess_telemetry", None):
            telem_out = {}
            for k, v in self._subprocess_telemetry.items():
                if isinstance(v, dict):
                    compact = {
                        sk: sv
                        for sk, sv in v.items()
                        if isinstance(sv, (int, float, str, type(None), bool))
                    }
                    telem_out[k] = compact
            # add last warmup timestamps (seconds since epoch)
            warm = {
                k: self._last_warmup.get(k)
                for k in telem_out.keys()
                if k in self._last_warmup
            }
            snap["subprocess"] = {"telemetry": telem_out, "last_warmup": warm}
        # metrics aggregate
        agg = self.get_metrics().get("__aggregate__", {})
        snap["metrics"] = agg
        # process RSS
        try:  # pragma: no cover - psutil optional
            import psutil  # type: ignore

            proc = psutil.Process()
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            snap["memory"] = {"rss_mb": round(rss_mb, 2)}
        except Exception:  # noqa
            snap["memory"] = {"rss_mb": None}
        # GPU (best-effort)
        try:  # pragma: no cover - optional
            import torch  # type: ignore

            if torch.cuda.is_available():
                snap["gpu"] = {
                    "device_count": torch.cuda.device_count(),
                    "allocated_mb": round(
                        torch.cuda.memory_allocated() / (1024 * 1024), 2
                    ),
                    "reserved_mb": round(
                        torch.cuda.memory_reserved() / (1024 * 1024), 2
                    ),
                }
        except Exception as gpu_err:  # noqa
            logger.debug(f"GPU snapshot collection failed: {gpu_err}")
        return snap

    def _import_entry(self, meta: ToolMeta):
        entry = meta.entry
        module_name, class_name = entry.split(":", 1)
        # Always ensure tool root dir precedes import to stabilize relative imports
        root = Path(meta.root_dir)
        root_str = str(root)

        # Track sys.path insertion for later cleanup
        inserted_path = None
        inserted_parent = None

        # Add parent directory (tools/) to sys.path to enable imports like "classification.xxx"
        parent = root.parent
        parent_str = str(parent)
        if parent_str not in sys.path:
            insertion_index = 1 if sys.path and sys.path[0] == "" else 0
            sys.path.insert(insertion_index, parent_str)
            inserted_parent = parent_str
            logger.debug(
                f"[import] inserted tool parent into sys.path index={insertion_index} parent={parent} module={module_name}"
            )

        if root_str not in sys.path:
            # place early but after cwd to avoid shadowing top-level packages
            insertion_index = 1 if sys.path and sys.path[0] == "" else 0
            sys.path.insert(insertion_index, root_str)
            inserted_path = root_str
            logger.debug(
                f"[import] inserted tool root into sys.path index={insertion_index} root={root} module={module_name}"
            )
        else:
            logger.debug(
                f"[import] tool root already in sys.path root={root} module={module_name}"
            )

        try:
            mod = import_module(module_name)
        except ModuleNotFoundError:
            # fallback: if module name is a simple file (e.g., tool_impl) load via spec directly later
            # This branch keeps previous behavior but root already inserted.
            mod = import_module(module_name)

        if not hasattr(mod, class_name):
            # Fallback: load by explicit file path (avoid name collision like multiple tool_impl.py on sys.path)
            candidate = Path(meta.root_dir) / f"{module_name}.py"
            if candidate.exists():
                import importlib.util as _ilu

                safe_name = f"eyetools_dyn_{meta.id.replace(':','_').replace('-','_')}"
                spec = _ilu.spec_from_file_location(safe_name, str(candidate))
                if spec and spec.loader:
                    dyn_mod = _ilu.module_from_spec(spec)
                    sys.modules[safe_name] = dyn_mod
                    spec.loader.exec_module(dyn_mod)  # type: ignore
                    # Track dynamic module for cleanup
                    self._dynamic_modules[meta.id] = safe_name
                    logger.debug(
                        f"[import] loaded dynamic module {safe_name} for tool {meta.id}"
                    )
                    if hasattr(dyn_mod, class_name):
                        # Track sys.path insertion for this tool
                        if inserted_path:
                            self._syspath_entries[meta.id] = inserted_path
                        if inserted_parent:
                            self._syspath_parent_entries[meta.id] = inserted_parent
                        return getattr(dyn_mod, class_name)

        # Track sys.path insertion for this tool (only if we actually inserted it)
        if inserted_path:
            self._syspath_entries[meta.id] = inserted_path
        if inserted_parent:
            self._syspath_parent_entries[meta.id] = inserted_parent
        return getattr(mod, class_name)

    def get_or_create_inproc(self, meta: ToolMeta):
        inst = self._inproc_instances.get(meta.id)
        if inst:
            # mark as loaded if previously unloaded/idle
            prev = self._lifecycle.get(meta.id)
            if prev in ("IDLE", "UNLOADED", "REGISTERED"):
                self._lifecycle[meta.id] = "LOADED"
            return inst
        logger.debug(f"[inproc] create instance tool_id={meta.id} entry={meta.entry}")
        ToolCls = self._import_entry(meta)
        # Pass meta dict form & params
        inst = ToolCls(meta.__dict__, meta.params)
        # prepare lazy; model loaded on first predict
        if hasattr(inst, "prepare"):
            try:
                inst.prepare()
            except Exception as e:  # noqa
                logger.exception(f"prepare failed tool_id={meta.id}: {e}")
                raise
        logger.debug(f"[inproc] ready tool_id={meta.id}")
        self._inproc_instances[meta.id] = inst
        # first time load transition
        self._lifecycle[meta.id] = "LOADED"
        return inst

    # Lightweight metadata hook: ask tool class to describe outputs without loading models
    def get_output_details(self, tool_id: str) -> dict | None:
        meta = self.registry.get(tool_id)
        if not meta:
            return None
        try:
            ToolCls = self._import_entry(meta)
        except Exception:
            return None
        try:
            # Prefer classmethod/staticmethod 'describe_outputs(meta, params)'
            if hasattr(ToolCls, "describe_outputs"):
                fn = getattr(ToolCls, "describe_outputs")
                return fn(meta.__dict__, meta.params)  # type: ignore[misc]
        except Exception:
            return None
        return None

    def _required_vram_mb(self, meta: ToolMeta) -> int:
        model = meta.model or {}
        estimate = model.get("vram_estimate") or {}
        if (
            estimate.get("gpu_required") is False
            or meta.runtime.get("gpu_required") is False
        ):
            return 0
        fallback = os.environ.get("EYETOOLS_GPU_DEFAULT_REQUIRED_MB")
        fallback_mb = None
        if fallback is not None:
            try:
                fallback_mb = int(fallback)
            except ValueError:
                logger.warning("invalid EYETOOLS_GPU_DEFAULT_REQUIRED_MB; ignoring")
            else:
                if fallback_mb <= 0:
                    fallback_mb = None
        try:
            required = estimate.get("required_vram_mb")
            if required is not None:
                return int(required)
        except (TypeError, ValueError):
            logger.warning(f"invalid required_vram_mb tool_id={meta.id}; using default")
        if fallback_mb is not None:
            return fallback_mb
        strict_required = (
            meta.runtime.get("gpu_required") is True
            or estimate.get("gpu_required") is True
        )
        if not strict_required:
            logger.debug(
                "tool_id={} has no VRAM estimate; skipping manager GPU preallocation",
                meta.id,
            )
            return 0
        raise ValueError(
            f"Tool {meta.id} with device_policy=auto requires "
            "model.vram_estimate.required_vram_mb or positive "
            "EYETOOLS_GPU_DEFAULT_REQUIRED_MB"
        )

    def _allocate_device_for_meta(
        self,
        meta: ToolMeta,
        exclude: set[int] | None = None,
        subprocess: bool = False,
    ) -> gpu_allocator.DeviceAllocation | None:
        device_policy = meta.runtime.get("device_policy")
        if device_policy in CPU_DEVICE_POLICIES:
            meta.params["device"] = "cpu"
            return None
        if device_policy == "fixed":
            meta.params["device"] = (
                meta.runtime.get("device") or meta.model.get("device") or "cuda:0"
            )
            return None

        required_mb = self._required_vram_mb(meta)
        if required_mb <= 0:
            return None

        allocation = gpu_allocator.select_gpu(
            required_mb=required_mb,
            exclude=exclude,
            source=meta.id,
        )
        meta.params["device"] = (
            allocation.visible_device if subprocess else allocation.device
        )
        self._gpu_allocations[meta.id] = allocation
        logger.info(
            "allocated gpu tool_id={} device={} visible_device={} logical_id={} "
            "physical_id={} required_mb={} free_mb={} total_mb={}",
            meta.id,
            allocation.device,
            allocation.visible_device,
            allocation.logical_id,
            allocation.physical_id,
            allocation.required_mb,
            allocation.free_mb_at_allocation,
            allocation.total_mb,
        )
        return allocation

    def _requires_auto_gpu_allocation(self, meta: ToolMeta) -> bool:
        device_policy = meta.runtime.get("device_policy")
        return (
            device_policy not in CPU_DEVICE_POLICIES
            and device_policy != "fixed"
            and self._required_vram_mb(meta) > 0
        )

    def _clear_cuda_cache_best_effort(self) -> None:
        try:  # pragma: no cover - torch optional
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:  # noqa
            logger.debug(f"clear cuda cache failed: {e}")

    def _allocation_for_inproc_attempt(
        self,
        meta: ToolMeta,
        exclude: set[int] | None = None,
    ) -> gpu_allocator.DeviceAllocation | None:
        if meta.id in self._inproc_instances:
            allocation = self._gpu_allocations.get(meta.id)
            if allocation is not None:
                return allocation
            if self._requires_auto_gpu_allocation(meta):
                logger.warning(
                    "gpu-required inproc instance has no allocation; "
                    "recreating before predict tool_id={}",
                    meta.id,
                )
                self.unload(meta.id)
                return self._allocate_device_for_meta(
                    meta,
                    exclude=exclude,
                    subprocess=False,
                )
            return None
        return self._allocate_device_for_meta(
            meta,
            exclude=exclude,
            subprocess=False,
        )

    def _allocation_for_subprocess_worker(
        self,
        meta: ToolMeta,
        exclude: set[int] | None = None,
    ) -> gpu_allocator.DeviceAllocation | None:
        return self._allocate_device_for_meta(
            meta,
            exclude=exclude,
            subprocess=True,
        )

    def _subprocess_worker_is_live(self, tool_id: str) -> bool:
        workers = getattr(self._process_manager, "_workers", {})
        worker = workers.get(tool_id)
        return worker is not None and worker.process.poll() is None

    def _stop_subprocess_worker(self, tool_id: str) -> None:
        try:
            self._process_manager.stop(tool_id)
        finally:
            self._gpu_allocations.pop(tool_id, None)

    def _ensure_subprocess_initialized(
        self,
        meta: ToolMeta,
        exclude: set[int] | None = None,
    ):
        allocation = self._gpu_allocations.get(meta.id)
        if self._subprocess_worker_is_live(meta.id):
            if allocation is not None:
                if exclude and allocation.physical_id in exclude:
                    logger.warning(
                        "subprocess worker allocation is excluded; "
                        "restarting before use tool_id={} physical_id={}",
                        meta.id,
                        allocation.physical_id,
                    )
                    self._stop_subprocess_worker(meta.id)
                else:
                    allocation_payload = self._subprocess_allocation_payload(allocation)
                    return self._process_manager.ensure_init(
                        meta,
                        gpu_allocation=allocation_payload,
                    )
            elif self._requires_auto_gpu_allocation(meta):
                logger.warning(
                    "gpu-required subprocess worker has no allocation; "
                    "restarting before use tool_id={}",
                    meta.id,
                )
                self._stop_subprocess_worker(meta.id)
            else:
                allocation_payload = self._subprocess_allocation_payload(allocation)
                return self._process_manager.ensure_init(
                    meta,
                    gpu_allocation=allocation_payload,
                )
        else:
            self._gpu_allocations.pop(meta.id, None)

        allocation = self._allocation_for_subprocess_worker(meta, exclude=exclude)
        allocation_payload = self._subprocess_allocation_payload(allocation)
        return self._process_manager.ensure_init(
            meta,
            gpu_allocation=allocation_payload,
        )

    @staticmethod
    def _subprocess_allocation_payload(
        allocation: gpu_allocator.DeviceAllocation | None,
    ) -> dict | None:
        if allocation is None:
            return None
        return {
            "physical_id": allocation.physical_id,
            "visible_device": allocation.visible_device,
        }

    def _predict_inproc_once(self, meta: ToolMeta, request: Dict[str, Any]):
        # ensure lifecycle registration baseline
        self._lifecycle.setdefault(meta.id, "REGISTERED")
        inst = self.get_or_create_inproc(meta)
        if meta.model.get("lazy", True) and hasattr(inst, "ensure_model_loaded"):
            before_load = time.time()
            inst.ensure_model_loaded()
            logger.debug(
                f"lazy model load tool_id={meta.id} took={(time.time()-before_load)*1000:.1f}ms"
            )
        self._inproc_lru[meta.id] = time.time()
        self._last_used[meta.id] = self._inproc_lru[meta.id]
        # LRU eviction if over limit
        if len(self._inproc_instances) > self._max_inproc:
            # evict oldest non-pinned
            oldest = sorted(self._inproc_lru.items(), key=lambda x: x[1])
            for tid, _ts in oldest:
                if tid != meta.id:  # keep current
                    # Use full unload logic for proper cleanup
                    self.unload(tid)
                    logger.debug(f"[lru eviction] unloaded tool {tid}")
                    break
        start = time.time()
        try:
            result = inst.predict(request)
        except Exception as e:  # noqa
            raise PredictionError(str(e)) from e
        finally:
            # 根据cleanup_policy决定是否立即清理GPU资源
            cleanup_policy = meta.runtime.get("cleanup_policy", "persist")
            if cleanup_policy == "immediate" and hasattr(inst, "shutdown"):
                try:
                    inst.shutdown()
                    # immediate模式：从缓存中移除实例，下次重新创建
                    self._inproc_instances.pop(meta.id, None)
                    self._inproc_lru.pop(meta.id, None)
                    self._gpu_allocations.pop(meta.id, None)
                    self._lifecycle[meta.id] = "UNLOADED"
                    logger.debug(
                        f"[predict cleanup] {meta.id} instance removed from cache (immediate mode)"
                    )
                except Exception as shutdown_err:
                    logger.warning(
                        f"[predict cleanup] {meta.id} shutdown failed: {shutdown_err}"
                    )
        latency = (time.time() - start) * 1000.0
        try:
            preview = summarize_for_log(result)
        except Exception:
            preview = {"type": "unprintable"}
        logger.info(
            f"predict result tool_id={meta.id} mode=inproc latency_ms={latency:.2f} preview={preview}"
        )
        m = self._metrics.setdefault(
            meta.id,
            {"predict_count": 0, "total_latency_ms": 0.0, "last_latency_ms": 0.0},
        )
        m["predict_count"] += 1
        m["total_latency_ms"] += latency
        m["last_latency_ms"] = latency
        m["avg_latency_ms"] = m["total_latency_ms"] / m["predict_count"]
        # mark idle baseline (caller of maintenance can flip to IDLE later)
        self._lifecycle[meta.id] = "LOADED"
        return result

    def _predict_inproc_with_gpu_retry(self, meta: ToolMeta, request: Dict[str, Any]):
        max_retries = int(
            meta.runtime.get(
                "oom_retry_max",
                1 if meta.runtime.get("oom_retry", True) else 0,
            )
        )
        excluded: set[int] = set()
        attempt = 0
        while True:
            allocation = self._allocation_for_inproc_attempt(
                meta,
                exclude=excluded,
            )
            try:
                return self._predict_inproc_once(meta, request)
            except Exception as exc:
                if attempt >= max_retries or not gpu_allocator.is_cuda_oom(exc):
                    raise
                failed_allocation = allocation or self._gpu_allocations.get(meta.id)
                if (
                    not self._requires_auto_gpu_allocation(meta)
                    or failed_allocation is None
                ):
                    raise
                excluded.add(failed_allocation.physical_id)
                failed_gpu = failed_allocation.physical_id
                logger.warning(
                    "cuda oom during inproc predict tool_id={} attempt={} "
                    "failed_physical_id={} retrying",
                    meta.id,
                    attempt + 1,
                    failed_gpu,
                )
                self.unload(meta.id)
                self._clear_cuda_cache_best_effort()
                attempt += 1

    def _predict_subprocess_with_gpu_retry(
        self, meta: ToolMeta, request: Dict[str, Any]
    ):
        max_retries = int(
            meta.runtime.get(
                "oom_retry_max",
                1 if meta.runtime.get("oom_retry", True) else 0,
            )
        )
        excluded: set[int] = set()
        attempt = 0
        while True:
            self._ensure_subprocess_initialized(meta, exclude=excluded)
            self._lifecycle[meta.id] = "LOADED"
            failed_allocation = self._gpu_allocations.get(meta.id)
            start = time.time()
            resp = self._process_manager.request(
                meta.id,
                {"cmd": "PREDICT", "request": request},
                timeout=self._model_timeout_s,
            )
            if not isinstance(resp, dict):
                raise PredictionError(
                    "Invalid subprocess response "
                    f"type={type(resp).__name__} value={resp!r}"
                )
            if not resp.get("ok"):
                if attempt >= max_retries or not gpu_allocator.is_cuda_oom(resp):
                    if meta.runtime.get("cleanup_policy") == "immediate":
                        self._stop_subprocess_worker(meta.id)
                    raise PredictionError(resp.get("error", "Unknown error"))
                if (
                    not self._requires_auto_gpu_allocation(meta)
                    or failed_allocation is None
                ):
                    if meta.runtime.get("cleanup_policy") == "immediate":
                        self._stop_subprocess_worker(meta.id)
                    raise PredictionError(resp.get("error", "Unknown error"))
                excluded.add(failed_allocation.physical_id)
                logger.warning(
                    "cuda oom during subprocess predict tool_id={} attempt={} "
                    "failed_physical_id={} retrying",
                    meta.id,
                    attempt + 1,
                    failed_allocation.physical_id,
                )
                self._stop_subprocess_worker(meta.id)
                self._clear_cuda_cache_best_effort()
                attempt += 1
                continue

            # 根据cleanup_policy决定是否立即清理子进程
            cleanup_policy = meta.runtime.get("cleanup_policy", "persist")
            if cleanup_policy == "immediate":
                # immediate模式：直接终止进程以彻底释放GPU资源
                self._stop_subprocess_worker(meta.id)
                # 下次predict会自动重新启动进程
                logger.debug(
                    f"[predict cleanup] {meta.id} subprocess terminated (immediate mode)"
                )

            latency = (time.time() - start) * 1000.0
            data = resp.get("data")
            self._last_used[meta.id] = time.time()
            try:
                preview = summarize_for_log(data)
            except Exception:
                preview = {"type": "unprintable"}
            logger.info(
                f"predict result tool_id={meta.id} mode=subprocess latency_ms={latency:.2f} preview={preview}"
            )
            m = self._metrics.setdefault(
                meta.id,
                {"predict_count": 0, "total_latency_ms": 0.0, "last_latency_ms": 0.0},
            )
            m["predict_count"] += 1
            m["total_latency_ms"] += latency
            m["last_latency_ms"] = latency
            m["avg_latency_ms"] = m["total_latency_ms"] / m["predict_count"]
            return data

    def predict(self, tool_id: str, request: Dict[str, Any]):
        meta = self.registry.get(tool_id)
        if not meta:
            raise ToolNotFoundError(f"Tool {tool_id} not found")
        normalized_request = normalize_request_images(request)
        if not self._supports_image_inputs(meta):
            return self._predict_single(meta, normalized_request)
        if self._is_multi_image_group(meta, normalized_request):
            output = self._predict_single(meta, normalized_request)
            return self._wrap_single_image_output(
                normalized_request, output, batch_mode="multi_image_group"
            )
        requests = expand_image_path_batches(normalized_request)
        if requests and self._request_has_image_path(requests[0]):
            if len(requests) == 1:
                output = self._predict_single(meta, requests[0])
                return self._wrap_single_image_output(requests[0], output)
            if meta.io.get("batchable") is True:
                try:
                    output = self._predict_single(meta, normalized_request)
                    native_result = self._split_native_batch_output(output, requests)
                    if native_result is not None:
                        return native_result
                    logger.warning(
                        f"native batch output was not splittable tool_id={tool_id}; "
                        "falling back to sequential image_path dispatch"
                    )
                except Exception:  # noqa
                    logger.exception(
                        f"native batch predict failed tool_id={tool_id}; "
                        "falling back to sequential image_path dispatch"
                    )
            return self._predict_sequential_batch(tool_id, meta, requests)
        request = requests[0]
        return self._predict_single(meta, request)

    def _predict_sequential_batch(
        self, tool_id: str, meta: ToolMeta, requests: list[Dict[str, Any]]
    ):
        results = []
        for index, item_request in enumerate(requests):
            try:
                output = self._predict_single(meta, item_request)
                results.append(
                    {
                        "index": index,
                        "output": output,
                    }
                )
            except Exception as exc:  # noqa
                logger.exception(f"batch item failed tool_id={tool_id} index={index}")
                results.append(
                    {
                        "index": index,
                        "error": str(exc),
                    }
                )
        return {
            "batch": True,
            "batch_mode": "sequential",
            "count": len(results),
            "results": results,
            "metadata": self._request_metadata(requests),
        }

    def _split_native_batch_output(
        self, output: Any, requests: list[Dict[str, Any]]
    ) -> Dict[str, Any] | None:
        items = None
        if isinstance(output, list):
            items = output
        elif isinstance(output, dict):
            for key in ("results", "outputs", "predictions"):
                value = output.get(key)
                if isinstance(value, list):
                    items = value
                    break
        if not isinstance(items, list) or len(items) != len(requests):
            return None

        return {
            "batch": True,
            "batch_mode": "native",
            "count": len(items),
            "results": [
                {
                    "index": index,
                    "output": item,
                }
                for index, item in enumerate(items)
            ],
            "metadata": self._request_metadata(requests),
        }

    def _wrap_single_image_output(
        self, request: Dict[str, Any], output: Any, batch_mode: str = "single"
    ):
        return {
            "batch": False,
            "batch_mode": batch_mode,
            "count": 1,
            "results": [
                {
                    "index": 0,
                    "output": output,
                }
            ],
            "metadata": self._request_metadata([request]),
        }

    @staticmethod
    def _request_metadata(requests: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Keep normalized request provenance outside prediction results."""
        return {
            "inputs": [request.get("inputs", request) for request in requests],
        }

    def _request_has_image_path(self, request: Dict[str, Any]) -> bool:
        inputs = request.get("inputs") if isinstance(request, dict) else None
        source = inputs if isinstance(inputs, dict) else request
        return isinstance(source, dict) and "image_path" in source

    def _supports_image_inputs(self, meta: ToolMeta) -> bool:
        input_schema = (meta.io or {}).get("input_schema")
        if not isinstance(input_schema, dict):
            return False
        image_keys = {
            "image_path",
            "image_paths",
            "original_image_path",
            "segmentation_image_path",
        }
        return any(
            key in image_keys or key.endswith("_image_path") for key in input_schema
        )

    def _is_multi_image_group(self, meta: ToolMeta, request: Dict[str, Any]) -> bool:
        if meta.io.get("image_input_mode") != "group":
            return False
        image_path = self._request_image_path(request)
        return isinstance(image_path, list)

    def _request_image_path(self, request: Dict[str, Any]):
        inputs = request.get("inputs") if isinstance(request, dict) else None
        source = inputs if isinstance(inputs, dict) else request
        if isinstance(source, dict):
            return source.get("image_path")
        return None

    def _predict_single(self, meta: ToolMeta, request: Dict[str, Any]):
        load_mode = meta.runtime.get("load_mode", "auto")
        # For now treat auto as inproc
        if load_mode in ("auto", "inproc"):
            return self._predict_inproc_with_gpu_retry(meta, request)
        elif load_mode == "subprocess":
            return self._predict_subprocess_with_gpu_retry(meta, request)
        else:
            raise ValueError(f"Unknown load_mode {load_mode}")

    # ------------------------------------------------------------------
    # Manual warmup
    def warmup_tool(self, tool_id: str) -> Dict[str, Any]:
        """Manually warm up a tool.

        Subprocess: send WARMUP command (worker handles absence gracefully).
        Inproc: call .warmup() if available else ensure_model_loaded().
        Returns status + optional telemetry.
        """
        meta = self.registry.get(tool_id)
        if not meta:
            raise ToolNotFoundError(f"Tool {tool_id} not found")
        mode = meta.runtime.get("load_mode", "auto")
        result: Dict[str, Any] = {"tool_id": tool_id, "runtime": mode}
        try:
            if mode == "subprocess":
                self._ensure_subprocess_initialized(meta)
                resp = self._process_manager.request(
                    tool_id, {"cmd": "WARMUP"}, timeout=self._model_timeout_s
                )
                if resp.get("ok"):
                    data = resp.get("data") or {}
                    if isinstance(data, dict):
                        tele = self._subprocess_telemetry.setdefault(tool_id, {})
                        tele.update(
                            {
                                k: v
                                for k, v in data.items()
                                if isinstance(v, (int, float, str, type(None), bool))
                            }
                        )
                        self._last_warmup[tool_id] = time.time()
                    result.update({"status": "ok", "warmup": data})
                    try:
                        logger.info(
                            f"warmup result tool_id={tool_id} mode=subprocess preview={summarize_for_log(data)}"
                        )
                    except Exception as log_err:
                        logger.error(
                            f"Failed to log subprocess warmup for tool_id={tool_id}: {log_err}"
                        )
                else:
                    result.update({"status": "error", "error": resp.get("error")})
            else:  # inproc
                self._allocation_for_inproc_attempt(meta)
                inst = self.get_or_create_inproc(meta)
                tele: Dict[str, Any] = {}
                if hasattr(inst, "warmup"):
                    try:
                        data = inst.warmup()  # type: ignore
                        if isinstance(data, dict):
                            tele.update(
                                {
                                    k: v
                                    for k, v in data.items()
                                    if isinstance(
                                        v, (int, float, str, type(None), bool)
                                    )
                                }
                            )
                            self._last_warmup[tool_id] = time.time()
                        try:
                            logger.info(
                                f"warmup result tool_id={tool_id} mode=inproc preview={summarize_for_log(data)}"
                            )
                        except Exception as log_err:
                            logger.error(
                                f"Failed to log inproc warmup for tool_id={tool_id}: {log_err}"
                            )
                    except Exception as e:  # noqa
                        result.update({"status": "error", "error": str(e)})
                        return result
                else:
                    if hasattr(inst, "ensure_model_loaded"):
                        inst.ensure_model_loaded()
                    tele["warmed_up"] = True
                    self._last_warmup[tool_id] = time.time()
                result.update({"status": "ok", "warmup": tele})
        except Exception as e:  # noqa
            result.update({"status": "error", "error": str(e)})
        return result

    def warmup_all(self, runtime: str | None = None) -> Dict[str, Any]:
        """Warm up all tools optionally filtered by runtime ('subprocess' or 'inproc')."""
        results = []
        for meta in self.registry.list():
            rmode = meta.runtime.get("load_mode", "auto")
            if (
                runtime
                and runtime != rmode
                and not (runtime == "inproc" and rmode in ("auto", "inproc"))
            ):
                continue
            results.append(self.warmup_tool(meta.id))
        return {"count": len(results), "results": results}

    def lifecycle_states(self) -> Dict[str, LifecycleState]:
        """Return a snapshot of lifecycle state per tool."""
        return self._lifecycle.copy()

    def lifecycle_states_detailed(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        workers = getattr(self._process_manager, "_workers", {})
        for tid, state in self._lifecycle.items():
            worker = workers.get(tid)
            subprocess_loaded = worker is not None and worker.process.poll() is None
            out[tid] = {
                "state": state,
                "last_used": self._last_used.get(tid),
                "loaded": tid in self._inproc_instances or subprocess_loaded,
            }
        return out

    def mark_idle(self, older_than_s: float = 300.0):
        """Mark LOADED tools as IDLE if last used before threshold (no unload)."""
        now = time.time()
        for tid, last_ts in list(self._inproc_lru.items()):
            if now - last_ts > older_than_s and self._lifecycle.get(tid) == "LOADED":
                self._lifecycle[tid] = "IDLE"

    def unload_idle(self, older_than_s: float = 900.0):
        """Unload IDLE tools whose last use exceeds threshold to free memory.

        Uses full unload logic for proper cleanup of sys.path, sys.modules, and GPU resources.
        """
        now = time.time()
        for tid, last_ts in list(self._inproc_lru.items()):
            if now - last_ts > older_than_s:
                self.unload(tid)
                logger.debug(f"[unload_idle] unloaded tool {tid}")

    def unload(self, tool_id: str) -> bool:
        """Explicitly unload a single inproc tool. Returns True if unloaded.

        Also cleans up:
        - sys.path entries inserted for this tool
        - Dynamic modules loaded for this tool
        - Shared encoder resources if tool uses FundusEncoderMixin
        """
        inst = self._inproc_instances.pop(tool_id, None)
        if not inst:
            # Still clean up sys.path and sys.modules even if instance not found
            pass

        # Call release if available
        if inst and hasattr(inst, "release"):
            try:
                inst.release()
            except Exception:  # noqa
                logger.exception(f"release failed tool_id={tool_id}")

        # Call shutdown for GPU cleanup if tool uses shared encoder
        if inst:
            try:
                # Check if tool class has clear_shared_resources classmethod (FundusEncoderMixin)
                tool_class = type(inst)
                if hasattr(tool_class, "clear_shared_resources"):
                    logger.debug(
                        f"[unload] clearing shared resources for tool {tool_id}"
                    )
                    tool_class.clear_shared_resources()
                # Also call instance shutdown for pipeline cleanup
                if hasattr(inst, "shutdown"):
                    inst.shutdown()
            except Exception:  # noqa
                logger.exception(f"shutdown/cleanup failed tool_id={tool_id}")

        # Clean up sys.path entries inserted for this tool
        inserted_path = self._syspath_entries.pop(tool_id, None)
        if inserted_path and inserted_path in sys.path:
            sys.path.remove(inserted_path)
            logger.debug(
                f"[unload] removed sys.path entry for tool {tool_id}: {inserted_path}"
            )

        inserted_parent = self._syspath_parent_entries.pop(tool_id, None)
        if inserted_parent and inserted_parent in sys.path:
            sys.path.remove(inserted_parent)
            logger.debug(
                f"[unload] removed sys.path parent entry for tool {tool_id}: {inserted_parent}"
            )

        # Clean up dynamic module from sys.modules
        dyn_module_name = self._dynamic_modules.pop(tool_id, None)
        if dyn_module_name and dyn_module_name in sys.modules:
            del sys.modules[dyn_module_name]
            logger.debug(
                f"[unload] removed dynamic module for tool {tool_id}: {dyn_module_name}"
            )

        self._inproc_lru.pop(tool_id, None)
        self._gpu_allocations.pop(tool_id, None)
        self._lifecycle[tool_id] = "UNLOADED"
        return True

    def unload_any(self, tool_id: str) -> bool:
        """Unload either an inproc tool or a subprocess worker.

        Returns True if something was unloaded/terminated, False otherwise.

        For inproc tools: calls release(), shutdown(), clears shared resources.
        For subprocess tools: sends SHUTDOWN command and terminates worker.
        Also cleans up sys.path and sys.modules entries for inproc tools.
        """
        meta = self.registry.get(tool_id)
        if not meta:
            return False
        mode = meta.runtime.get("load_mode", "auto")
        if mode in ("auto", "inproc"):
            return self.unload(tool_id)
        elif mode == "subprocess":
            # terminate worker if running
            workers = getattr(self._process_manager, "_workers", {})
            if tool_id in workers:
                try:
                    self._stop_subprocess_worker(tool_id)
                except Exception:  # noqa
                    logger.exception(f"stop worker failed tool_id={tool_id}")
                # Clean up sys.path and sys.modules for subprocess tools too
                # (in case they were preloaded with model info)
                inserted_path = self._syspath_entries.pop(tool_id, None)
                if inserted_path and inserted_path in sys.path:
                    sys.path.remove(inserted_path)
                inserted_parent = self._syspath_parent_entries.pop(tool_id, None)
                if inserted_parent and inserted_parent in sys.path:
                    sys.path.remove(inserted_parent)
                dyn_module_name = self._dynamic_modules.pop(tool_id, None)
                if dyn_module_name and dyn_module_name in sys.modules:
                    del sys.modules[dyn_module_name]
                self._lifecycle[tool_id] = "UNLOADED"
                return True
            return False
        return False

    def get_metrics(self, tool_id: str | None = None):
        """Return metrics for a single tool or all tools.
        If tool_id is None returns dict of all metrics plus aggregate summary under key '__aggregate__'.
        """
        if tool_id:
            return self._metrics.get(tool_id, {}).copy()
        # build aggregate
        aggregate = {
            "tools": len(self._metrics),
            "predict_total": 0,
            "avg_latency_ms": 0.0,
        }
        total_latency = 0.0
        for mid, m in self._metrics.items():
            aggregate["predict_total"] += m.get("predict_count", 0)
            total_latency += m.get("total_latency_ms", 0.0)
        if aggregate["predict_total"]:
            aggregate["avg_latency_ms"] = total_latency / aggregate["predict_total"]
        all_metrics = {tid: data.copy() for tid, data in self._metrics.items()}
        all_metrics["__aggregate__"] = aggregate
        return all_metrics

    def ensure_loaded(self, tool_id: str, include_model: bool = True):
        """Ensure a tool is loaded/initialized based on its load_mode.

        For inproc/auto: instantiate (if needed) and optionally force lazy model load.
        For subprocess: spawn worker & perform INIT (model load is deferred unless protocol extends).
        Returns dict with {mode, tool_id, status}.
        """
        meta = self.registry.get(tool_id)
        if not meta:
            raise ToolNotFoundError(f"Tool {tool_id} not found")
        mode = meta.runtime.get("load_mode", "auto")
        if mode in ("auto", "inproc"):
            self._allocation_for_inproc_attempt(meta)
            inst = self.get_or_create_inproc(meta)
            if (
                include_model
                and meta.model.get("lazy", True)
                and hasattr(inst, "ensure_model_loaded")
            ):
                inst.ensure_model_loaded()
            return {"tool_id": tool_id, "mode": mode, "status": "loaded"}
        elif mode == "subprocess":
            self._ensure_subprocess_initialized(meta)
            # Optionally could add LOAD_MODEL command in future if protocol extended
            self._lifecycle.setdefault(tool_id, "REGISTERED")
            self._lifecycle[tool_id] = "LOADED"
            return {"tool_id": tool_id, "mode": mode, "status": "worker_started"}
        else:
            raise ValueError(f"Unknown load_mode {mode}")

    def maintenance(
        self, mark_idle_s: float | None = None, unload_idle_s: float | None = None
    ) -> Dict[str, Any]:
        """Run maintenance operations (mark idle, unload idle) and return a summary.

        mark_idle_s: if provided, tools not used for > seconds are transitioned LOADED->IDLE.
        unload_idle_s: if provided, tools not used for > seconds are fully unloaded (inproc) or subprocess workers stopped.
        """
        before_states = self.lifecycle_states().copy()
        now = time.time()
        actions: Dict[str, Any] = {"marked_idle": [], "unloaded": []}
        workers = getattr(self._process_manager, "_workers", {})
        if mark_idle_s is not None:
            # replicate mark_idle logic with explicit threshold
            for tid, last_ts in list(self._inproc_lru.items()):
                if (now - last_ts) > mark_idle_s and self._lifecycle.get(
                    tid
                ) == "LOADED":
                    self._lifecycle[tid] = "IDLE"
                    actions["marked_idle"].append(tid)
            for tid, worker in list(workers.items()):
                last_ts = getattr(worker, "last_used", None)
                if last_ts is None:
                    continue
                self._last_used[tid] = last_ts
                if (now - last_ts) > mark_idle_s and self._lifecycle.get(
                    tid
                ) == "LOADED":
                    self._lifecycle[tid] = "IDLE"
                    actions["marked_idle"].append(tid)
        if unload_idle_s is not None:
            unloaded = set()
            for tid, last_ts in list(self._inproc_lru.items()):
                if (now - last_ts) > unload_idle_s:
                    self.unload(tid)
                    unloaded.add(tid)
                    actions["unloaded"].append(tid)
            for tid, worker in list(workers.items()):
                if tid in unloaded:
                    continue
                last_ts = getattr(worker, "last_used", None)
                if last_ts is None:
                    continue
                self._last_used[tid] = last_ts
                if (now - last_ts) > unload_idle_s and self.unload_any(tid):
                    actions["unloaded"].append(tid)
        after_states = self.lifecycle_states().copy()
        return {"before": before_states, "after": after_states, "actions": actions}

    # ------------------------------------------------------------------
    def preload_all(
        self,
        include_subprocess: bool = False,
        parallel_subprocess: bool = False,
        max_workers: int = 4,
    ):
        """Instantiate all tools and (if lazy) load their models.

        include_subprocess: if True also spin up subprocess tools (may be expensive).
        For subprocess tools we currently only INIT the worker; model weights may still
        be lazily loaded on first PREDICT unless/until a future LOAD_MODEL protocol
        extension is issued. (A placeholder will be added if handler exists.)
        Exceptions during individual tool preload are logged and skipped so one
        broken tool doesn't block server startup.
        """
        from loguru import logger as _logger

        count = 0
        subprocess_metas = []
        regular_metas = []
        for meta in self.registry.list():
            if meta.runtime.get("load_mode", "auto") == "subprocess":
                subprocess_metas.append(meta)
            else:
                regular_metas.append(meta)

        # Optionally handle subprocess tools in parallel
        if include_subprocess and parallel_subprocess and subprocess_metas:
            from concurrent.futures import (
                ThreadPoolExecutor,
                as_completed,
            )  # pragma: no cover (timing sensitive)

            def _load_subproc(meta):
                from loguru import logger as _plog

                try:
                    self._ensure_subprocess_initialized(meta)
                    self._lifecycle[meta.id] = "LOADED"
                    try:
                        resp = self._process_manager.request(
                            meta.id,
                            {"cmd": "LOAD_MODEL"},
                            timeout=self._model_timeout_s,
                        )
                        if resp.get("ok") is False:
                            _plog.warning(
                                f"[preload] LOAD_MODEL failed tool_id={meta.id} resp={resp}"
                            )
                        else:
                            data = resp.get("data") or {}
                            if isinstance(data, dict):
                                loaded = data.get("loaded", True)
                                if loaded:
                                    self._subprocess_telemetry[meta.id] = data
                                    _plog.info(
                                        f"[preload] subprocess model loaded tool_id={meta.id} (parallel) cuda={data.get('cuda')} device={data.get('device_name')} mem_delta={data.get('mem_delta')} reserved_delta={data.get('reserved_delta')}",
                                        meta.id,
                                        data.get("cuda"),
                                        data.get("device_name"),
                                        data.get("mem_delta"),
                                        data.get("reserved_delta"),
                                    )
                                    if data.get("cuda") and (
                                        data.get("mem_delta") in (0, None)
                                    ):
                                        _plog.warning(
                                            f"[preload] tool_id={meta.id} CUDA loaded but mem_delta={data.get('mem_delta')} (may allocate later)",
                                        )
                                else:
                                    _plog.warning(
                                        f"[preload] subprocess model NOT loaded tool_id={meta.id}"
                                    )
                            else:
                                _plog.info(
                                    f"[preload] subprocess model loaded tool_id={meta.id} (parallel unstructured)"
                                )
                    except Exception as e:  # noqa
                        _plog.debug(
                            f"[preload] subprocess model lazy-load later tool_id={meta.id} error={e}"
                        )
                    return True
                except Exception as e:  # noqa
                    _plog.exception(
                        f"[preload] worker init failed tool_id={meta.id} error={e}"
                    )
                    return False

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_load_subproc, m): m for m in subprocess_metas}
                for f in as_completed(futures):  # pragma: no cover
                    if f.result():
                        count += 1

        # Sequential handling (remaining subprocess if not parallel, and all regular)
        seq_list = []
        if include_subprocess and not parallel_subprocess:
            seq_list.extend(subprocess_metas)
        seq_list.extend(regular_metas)
        for meta in seq_list:
            mode = meta.runtime.get("load_mode", "auto")
            if mode == "subprocess":
                if not include_subprocess:
                    _logger.info(
                        f"[preload] skip subprocess tool_id={meta.id} (include_subprocess=False)"
                    )
                    continue
                # start worker (INIT)
                try:
                    self._ensure_subprocess_initialized(meta)
                    self._lifecycle[meta.id] = "LOADED"
                    # Attempt forced model load if protocol supports it
                    try:  # pragma: no cover (depends on model availability)
                        resp = self._process_manager.request(
                            meta.id,
                            {"cmd": "LOAD_MODEL"},
                            timeout=self._model_timeout_s,
                        )
                        if resp.get("ok") is False:
                            _logger.warning(
                                f"[preload] LOAD_MODEL failed tool_id={meta.id} resp={resp}"
                            )
                        else:
                            data = resp.get("data") or {}
                            if isinstance(data, dict):
                                loaded = data.get("loaded", True)
                                cuda_flag = data.get("cuda")
                                device_name = data.get("device_name")
                                mem_delta = data.get("mem_delta")
                                mem_before = data.get("mem_before")
                                mem_after = data.get("mem_after")
                                if loaded:
                                    # store telemetry
                                    self._subprocess_telemetry[meta.id] = data
                                    _logger.info(
                                        f"[preload] subprocess model loaded tool_id={meta.id} cuda={cuda_flag} device={device_name} mem_delta={mem_delta} bytes(before={mem_before} after={mem_after}) reserved_delta={data.get('reserved_delta')} first_param_device={data.get('first_param_device')} params={data.get('param_count')} cuda_param_bytes={data.get('cuda_param_bytes')}",
                                    )
                                    if cuda_flag and (
                                        mem_delta is None
                                        or (
                                            isinstance(mem_delta, (int, float))
                                            and mem_delta <= 0
                                        )
                                    ):
                                        _logger.warning(
                                            f"[preload] tool_id={meta.id} reported loaded on CUDA but mem_delta={mem_delta} (may allocate later during first predict)",
                                            meta.id,
                                            mem_delta,
                                        )
                                else:
                                    _logger.warning(
                                        f"[preload] subprocess model NOT loaded (weights missing?) tool_id={meta.id}"
                                    )
                            else:
                                _logger.info(
                                    f"[preload] subprocess model loaded tool_id={meta.id} (unstructured data)"
                                )
                    except Exception as e:  # noqa
                        _logger.debug(
                            f"[preload] subprocess model lazy-load will occur on first predict tool_id={meta.id} error={e}"
                        )
                    count += 1
                except Exception as e:  # noqa
                    _logger.exception(
                        f"[preload] worker init failed tool_id={meta.id} error={e}"
                    )
                continue
            try:
                if mode in ("auto", "inproc"):
                    self._allocation_for_inproc_attempt(meta)
                    inst = self.get_or_create_inproc(meta)
                else:
                    inst = None
                if (
                    inst
                    and meta.model.get("lazy", True)
                    and hasattr(inst, "ensure_model_loaded")
                ):
                    inst.ensure_model_loaded()
                # Optional warmup request
                warm = meta.warmup.get("request") if hasattr(meta, "warmup") else None
                if warm and inst:
                    try:
                        inst.predict(warm)
                    except Exception:  # pragma: no cover
                        _logger.exception(f"[preload] warmup failed tool_id={meta.id}")
                count += 1
            except Exception as e:  # noqa
                _logger.exception(f"[preload] failed tool_id={meta.id} error={e}")
        _logger.info(
            f"[preload] completed count={count} parallel_subprocess={parallel_subprocess}"
        )
        return count


__all__ = ["ToolManager"]
