"""Process manager for out-of-process tool workers (with uv support)."""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import atexit
import os
from typing import Deque, Dict, Optional, List
import subprocess
import sys
import threading
import json
import time
import select
from pathlib import Path
from .env_manager import EnvManager
from .errors import WorkerInitError, EnvironmentError
from loguru import logger

DEFAULT_MODEL_TIMEOUT_S = 900.0


@dataclass
class WorkerInfo:
    tool_id: str
    process: subprocess.Popen
    started: float
    last_used: float
    status: str = "starting"
    stderr_tail: Deque[str] = field(default_factory=lambda: deque(maxlen=200))
    request_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class ProcessManager:
    def __init__(
        self,
        workspace_root: Path,
        env_manager: Optional[EnvManager] = None,
        default_timeout_s: float = DEFAULT_MODEL_TIMEOUT_S,
    ):
        self.workspace_root = workspace_root
        self.env_manager = env_manager or EnvManager(workspace_root)
        self.default_timeout_s = float(default_timeout_s)
        self._workers: Dict[str, WorkerInfo] = {}
        self._init_done: Dict[str, bool] = {}
        self._lock = threading.RLock()
        # 注册退出时清理所有子进程
        atexit.register(self._cleanup_all_workers)

    def spawn(self, tool_id: str, meta=None, gpu_allocation: dict | None = None):
        with self._lock:
            if tool_id in self._workers:
                wi = self._workers[tool_id]
                # If the worker died, drop it so we can respawn and re-init.
                if wi.process.poll() is not None:
                    self._workers.pop(tool_id, None)
                    self._init_done.pop(tool_id, None)
                else:
                    return wi
            # Build uv command if environment_ref/python/extra requires specified
            entrypoint: List[str]
            env: Dict[str, str] | None = None
            if meta and (
                getattr(meta, "environment_ref", None)
                or getattr(meta, "python", None)
                or getattr(meta, "extra_requires", None)
            ):
                entrypoint = ["uv", "run"]
                deps: List[str] = []
                py_exec: str | None = None
                try:
                    # Use EnvManager resolution to collect deps & python tag if environment_ref present
                    if getattr(meta, "environment_ref", None):
                        py_tag, deps, uv_project = self.env_manager._resolve_env_ref(meta.__dict__)  # type: ignore[attr-defined]
                        shorthand_map = {
                            f"py3{py_ver}": f"python3.{py_ver}"
                            for py_ver in range(6, 14)
                        }
                        py_exec = shorthand_map.get(py_tag, py_tag)
                        if uv_project:
                            env = os.environ.copy()
                            env["UV_PROJECT"] = str(uv_project)
                except Exception:  # pragma: no cover
                    import traceback

                    traceback.print_exc()
                    pass
                # Merge explicit extra_requires if any (avoid duplicates)
                extra = list(getattr(meta, "extra_requires", []) or [])
                for e in extra:
                    if e not in deps:
                        deps.append(e)
                # Append --with flags
                for dep in deps:
                    entrypoint += ["--with", dep]
                # Python selection precedence: explicit meta.python overrides env_ref resolution
                if getattr(meta, "python", None):
                    py_exec = getattr(meta, "python")
                if py_exec:
                    entrypoint += [f"--python={py_exec}"]
                entrypoint += ["-m", "eyetools.core.worker_entry"]
            else:
                entrypoint = [sys.executable, "-m", "eyetools.core.worker_entry"]
            if gpu_allocation:
                if env is None:
                    env = os.environ.copy()
                else:
                    env = env.copy()
                physical_id = str(gpu_allocation["physical_id"])
                env["CUDA_VISIBLE_DEVICES"] = physical_id
                env["EYETOOLS_SELECTED_DEVICE"] = str(
                    gpu_allocation.get("visible_device", "cuda:0")
                )
                env["EYETOOLS_PHYSICAL_GPU_ID"] = physical_id
            logger.debug(f"spawn worker tool_id={tool_id} cmd={' '.join(entrypoint)}")
            try:
                proc = subprocess.Popen(
                    entrypoint,
                    cwd=str(self.workspace_root),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
            except Exception as e:  # noqa
                raise EnvironmentError(
                    f"Failed spawning worker for {tool_id}: {e}"
                ) from e
            wi = WorkerInfo(
                tool_id=tool_id,
                process=proc,
                started=time.time(),
                last_used=time.time(),
            )
            self._start_stderr_drain(wi)
            self._workers[tool_id] = wi
            return wi

    def _start_stderr_drain(self, wi: WorkerInfo) -> None:
        """Continuously drain worker stderr so verbose model logs cannot block it."""

        def _drain() -> None:
            stream = getattr(wi.process, "stderr", None)
            if stream is None:
                return
            try:
                for line in stream:
                    if line:
                        wi.stderr_tail.append(line.rstrip())
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "stderr drain stopped tool_id={} error={}: {}",
                    wi.tool_id,
                    type(e).__name__,
                    e,
                )

        threading.Thread(
            target=_drain,
            name=f"eyetools-stderr-{wi.tool_id}",
            daemon=True,
        ).start()

    def _send(self, wi: WorkerInfo, payload: Dict):
        line = json.dumps(payload) + "\n"
        assert wi.process.stdin is not None
        wi.process.stdin.write(line)
        wi.process.stdin.flush()

    def _recv(self, wi: WorkerInfo, timeout: float | None = None) -> Dict:
        assert wi.process.stdout is not None
        timeout_s = self.default_timeout_s if timeout is None else float(timeout)
        start = time.time()
        while True:
            if (time.time() - start) > timeout_s:
                raise TimeoutError("Timeout waiting for worker response")
            if wi.process.poll() is not None:
                line = wi.process.stdout.readline()
                if not line:
                    stderr = "\n".join(wi.stderr_tail).strip()
                    if not stderr:
                        stream = getattr(wi.process, "stderr", None)
                        if stream is not None and hasattr(stream, "read"):
                            try:
                                stderr = stream.read().strip()
                            except Exception:  # noqa: BLE001
                                stderr = ""
                    detail = (
                        f": {stderr}" if stderr else " without producing a response"
                    )
                    raise WorkerInitError(
                        f"Worker for {wi.tool_id} exited with code "
                        f"{wi.process.returncode}{detail}"
                    )
            if hasattr(wi.process.stdout, "fileno"):
                ready, _, _ = select.select([wi.process.stdout], [], [], 0.05)
                if not ready:
                    continue
            line = wi.process.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # Skip lines that are not valid JSON (e.g., 'Built package==version')
                if line.startswith(
                    ("Built ", "Installed ", "Downloading", "[", "Preparing")
                ):
                    continue
                # As a safeguard, ignore any other stray stdout
                continue
            time.sleep(0.01)

    def request(
        self, tool_id: str, payload: Dict, timeout: float | None = None
    ) -> Dict:
        wi = self._workers.get(tool_id)
        if not wi:
            raise WorkerInitError(f"Worker for {tool_id} not running")
        with wi.request_lock:
            if self._workers.get(tool_id) is not wi:
                raise WorkerInitError(f"Worker for {tool_id} stopped before request")
            self._send(wi, payload)
            wi.last_used = time.time()
            timeout_s = self.default_timeout_s if timeout is None else float(timeout)
            return self._recv(wi, timeout_s)

    def ensure_init(self, tool_meta, gpu_allocation: dict | None = None):
        while True:
            wi = self.spawn(
                tool_meta.id, meta=tool_meta, gpu_allocation=gpu_allocation
            )
            with wi.request_lock:
                if self._workers.get(tool_meta.id) is not wi:
                    continue
                if not self._init_done.get(tool_meta.id):
                    logger.debug(f"init worker tool_id={tool_meta.id}")
                    resp = self.request(
                        tool_meta.id, {"cmd": "INIT", "meta": tool_meta.__dict__}
                    )
                    if not isinstance(resp, dict):
                        raise WorkerInitError(
                            "INIT returned non-dict response: "
                            f"{type(resp).__name__} = {resp!r}"
                        )
                    if not resp.get("ok", False):
                        self.stop(tool_meta.id)
                        raise WorkerInitError(f"INIT failed: {resp}")
                    self._init_done[tool_meta.id] = True
                return wi

    def stop(self, tool_id: str):
        """停止worker进程。用于immediate模式或手动清理。"""
        wi = self._workers.get(tool_id)
        if not wi:
            # Preserve previous cleanup behavior for stale init state.
            self._init_done.pop(tool_id, None)
            return
        with wi.request_lock:
            if self._workers.get(tool_id) is not wi:
                return
            if wi.process.poll() is None:
                try:
                    self.request(tool_id, {"cmd": "SHUTDOWN"}, timeout=5)
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        f"Worker {tool_id} SHUTDOWN command failed (expected if worker unresponsive): {type(e).__name__}: {e}"
                    )
                finally:
                    self._workers.pop(tool_id, None)
                    self._init_done.pop(tool_id, None)
                wi.process.terminate()
                try:
                    wi.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"Worker {tool_id} didn't terminate gracefully, killing..."
                    )
                    wi.process.kill()
                    try:
                        wi.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.error(
                            f"Worker {tool_id} didn't exit after kill(), resource leak possible"
                        )
            else:
                self._workers.pop(tool_id, None)
                self._init_done.pop(tool_id, None)
                try:
                    wi.process.wait(timeout=0)
                except Exception:  # noqa: BLE001
                    pass

    def cleanup_idle(self, idle_seconds: float):
        now = time.time()
        to_stop = []
        for tid, wi in self._workers.items():
            if (now - wi.last_used) > idle_seconds:
                to_stop.append(tid)
        for tid in to_stop:
            self.stop(tid)

    def _cleanup_all_workers(self):
        """清理所有worker进程。在程序退出时调用，避免僵尸进程。"""
        worker_ids = list(self._workers.keys())
        for tool_id in worker_ids:
            try:
                self.stop(tool_id)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error stopping worker {tool_id}: {e}")

    def __del__(self):
        """析构时清理所有子进程。"""
        try:
            self._cleanup_all_workers()
        except Exception:  # noqa: BLE001, S110
            pass  # 析构中不抛出异常


__all__ = ["DEFAULT_MODEL_TIMEOUT_S", "ProcessManager", "WorkerInfo"]
