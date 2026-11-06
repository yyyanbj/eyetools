from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from eyetools.core.process_manager import ProcessManager, WorkerInfo


def test_worker_requests_keep_send_and_receive_serialized(tmp_path: Path):
    manager = ProcessManager(tmp_path)
    worker = WorkerInfo(
        "classifier",
        process=SimpleNamespace(poll=lambda: 1, wait=lambda timeout=None: None),
        started=0,
        last_used=0,
    )
    manager._workers[worker.tool_id] = worker
    local = threading.local()
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def send(_worker, payload):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        local.request_id = payload["id"]

    def receive(_worker, _timeout):
        nonlocal active
        time.sleep(0.02)
        with state_lock:
            active -= 1
        return {"id": local.request_id}

    manager._send = send
    manager._recv = receive
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(
            pool.map(lambda item: manager.request("classifier", {"id": item}), range(8))
        )

    assert responses == [{"id": item} for item in range(8)]
    assert max_active == 1
