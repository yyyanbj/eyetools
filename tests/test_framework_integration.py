import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fastmcp import Client

from eyetools.core.registry import ToolMeta
from eyetools.core.vram_estimator import estimate_tool_vram
from eyetools.mcp_server import create_app


@pytest.fixture
def tool_root(tmp_path):
    root = tmp_path / "tools"
    package = root / "echo"
    package.mkdir(parents=True)
    (package / "config.yaml").write_text(
        """id: echo
package: echo
entry: smoke_echo:Echo
runtime:
  load_mode: inproc
  device_policy: cpu_only
model:
  device: cpu
io:
  description: A temporary tool for integration tests.
  input_schema:
    type: object
    properties:
      message:
        type: string
    required: [message]
""",
        encoding="utf-8",
    )
    (package / "smoke_echo.py").write_text(
        """from pathlib import Path

class Echo:
    def __init__(self, meta, params):
        self.meta = meta

    def predict(self, request):
        return {"message": request["inputs"]["message"]}

    def shutdown(self):
        Path(self.meta["root_dir"]).joinpath("shutdown.marker").write_text("done")
""",
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize("mode", ["inproc", "subprocess"])
def test_added_tool_is_discovered_and_called_over_http(tool_root, mode):
    config = tool_root / "echo" / "config.yaml"
    config.write_text(config.read_text().replace("load_mode: inproc", f"load_mode: {mode}"))
    app = create_app(tool_paths=[str(tool_root)], lifecycle_mode="lazy")
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "tools": 1}
        tool = client.get("/tools").json()["tools"][0]
        assert tool["id"] == "echo"
        response = client.post(
            "/predict", json={"tool_id": "echo", "request": {"inputs": {"message": mode}}}
        )
        assert response.status_code == 200, response.text
        assert response.json()["output"] == {"message": mode}
        assert client.post("/admin/evict", params={"tool_id": "echo"}).status_code == 200
        assert (tool_root / "echo" / "shutdown.marker").read_text() == "done"


def test_added_tool_is_callable_over_mcp(tool_root):
    app = create_app(tool_paths=[str(tool_root)], lifecycle_mode="lazy")

    async def exercise():
        async with Client(app.state.fastmcp) as client:
            tools = await client.list_tools()
            assert len(tools) == 1
            result = await client.call_tool(tools[0].name, {"message": "hello"})
            assert result.data["output"] == {"message": "hello"}

    asyncio.run(exercise())


def test_vram_estimation_uses_declared_weight_paths(tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"model")
    meta = ToolMeta(id="custom", entry="sample:Tool", version="1", model={"weight_paths": ["model.pt"]})
    estimate = estimate_tool_vram(meta, tmp_path)
    assert estimate["files"][0]["path"] == "model.pt"
    assert estimate["gpu_required"] is True


def test_vram_estimation_does_not_guess_private_tool_paths(tmp_path):
    weights = tmp_path / "weights" / "report"
    weights.mkdir(parents=True)
    (weights / "checkpoint.pth").write_bytes(b"model")
    meta = ToolMeta(id="report", package="report", entry="sample:Tool", version="1")
    with pytest.raises(FileNotFoundError, match="No weight files resolved"):
        estimate_tool_vram(meta, tmp_path)


def test_model_id_names_are_not_tied_to_a_specific_package():
    from eyetools.core.schema import canonical_tool_name

    assert canonical_tool_name(
        {"package": "custom", "variant": "example", "model": {"tool_model_id": "v2"}}
    ) == "custom_example_v2"


@pytest.mark.parametrize("source", ["direct", "uv_sources"])
def test_subprocess_tool_runs_in_its_referenced_environment(tool_root, tmp_path, monkeypatch, source):
    package = tool_root / "echo"
    config = package / "config.yaml"
    config.write_text(
        config.read_text().replace("load_mode: inproc", "load_mode: subprocess")
        + "\nenvironment_ref: echo-env\n"
    )
    implementation = package / "smoke_echo.py"
    implementation.write_text(
        "import simplejson\n" + implementation.read_text().replace(
            'return {"message": request["inputs"]["message"]}',
            'return simplejson.loads(simplejson.dumps({"message": request["inputs"]["message"]}))',
        )
    )
    env = tmp_path / "envs" / "echo-env"
    env.mkdir(parents=True)
    framework = Path(__file__).resolve().parents[1]
    requirement = f"eyetools @ {framework.as_uri()}" if source == "direct" else "eyetools"
    uv_sources = f'\n[tool.uv.sources]\neyetools = {{path = "{framework}", editable = true}}\n' if source == "uv_sources" else ""
    (env / "pyproject.toml").write_text(
        f'''[project]
name = "echo-test-env"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["{requirement}", "simplejson>=3.19"]
''' + uv_sources
    )
    monkeypatch.chdir(tmp_path)
    app = create_app(tool_paths=[str(tool_root)], lifecycle_mode="lazy", enable_mcp=False)
    with TestClient(app) as client:
        response = client.post(
            "/predict", json={"tool_id": "echo", "request": {"inputs": {"message": "isolated"}}}
        )
        assert response.status_code == 200, response.text
        assert response.json()["output"] == {"message": "isolated"}
        assert client.post("/admin/evict", params={"tool_id": "echo"}).status_code == 200


def test_empty_project_starts_without_tools(tmp_path, monkeypatch):
    (tmp_path / "tools").mkdir()
    monkeypatch.chdir(tmp_path)
    app = create_app(lifecycle_mode="lazy")
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "tools": 0}
        assert client.get("/tools").json() == {"tools": []}

    async def exercise():
        async with Client(app.state.fastmcp) as client:
            assert await client.list_tools() == []

    asyncio.run(exercise())


def test_subprocess_toolbase_loads_model_only_once(tool_root):
    config = tool_root / "echo" / "config.yaml"
    config.write_text(config.read_text().replace("load_mode: inproc", "load_mode: subprocess"))
    (tool_root / "echo" / "smoke_echo.py").write_text(
        '''from eyetools import ToolBase

class Echo(ToolBase):
    def __init__(self, meta, params):
        super().__init__(meta, params)
        self.loads = 0

    def load_model(self):
        self.loads += 1
        super().load_model()

    def predict(self, request):
        return {"loads": self.loads}
'''
    )
    app = create_app(tool_paths=[str(tool_root)], lifecycle_mode="lazy", enable_mcp=False)
    with TestClient(app) as client:
        for call in range(2):
            response = client.post(
                "/predict", json={"tool_id": "echo", "request": {"inputs": {"message": "x"}}}
            )
            assert response.status_code == 200, response.text
            assert response.json()["output"]["loads"] == 1
            if call == 0:
                worker = app.state.tool_manager._process_manager
                assert worker.request("echo", {"cmd": "LOAD_MODEL"})["data"]["loaded"] is True
        client.post("/admin/evict", params={"tool_id": "echo"})
