# EyeTools MCP Framework

EyeTools is a Python MCP server that dynamically discovers tools from their configuration and exposes them over MCP and HTTP. This distribution contains the server and its core infrastructure; it ships with no domain tool packages, model weights, or datasets.

Read the documentation in [English](https://yyyanbj.github.io/eyetools/) or [中文](https://yyyanbj.github.io/eyetools/zh/).

## Requirements and installation

Python 3.12 or later and [uv](https://docs.astral.sh/uv/) are required. From the project root:

```bash
uv sync
uv run eyetools-mcp serve --host 127.0.0.1 --port 8000 --tools-dir tools
```

The empty server reports `{"status":"ok","tools":0}` at `http://127.0.0.1:8000/health`. HTTP API documentation is available at `/docs`; the MCP streamable HTTP endpoint is `/mcp`. The API has no built-in authentication; restrict access with an authenticated proxy if exposing it remotely.

The project includes the existing Apache 2.0 license. It does not install PyTorch or other model libraries in the server environment. Install tool-specific packages in their tool environment as described below.

## Adding a tool

Place a directory containing `config.yaml` in a scanned tool root. The entry is `module:Class`, relative to the tool directory. Each class receives `(meta, params)` and implements `predict(request)`; the server passes the tool input in `request["inputs"]`. Return a JSON-serializable value. Optional hooks include `prepare`, `load_model`, `warmup`, and `shutdown`. When subclassing `ToolBase`, call `super().load_model()` after loading a model so the framework tracks its loaded state.

The following minimal example can be created locally to test the interface; no tool packages are included in this distribution.

```text
tools/
└── echo/
    ├── config.yaml
    └── echo_impl.py
```

`tools/echo/config.yaml`:

```yaml
id: echo
package: echo
entry: echo_impl:EchoTool
runtime:
  load_mode: inproc
  device_policy: cpu_only
model:
  device: cpu
io:
  description: Return the supplied message.
  input_schema:
    type: object
    properties:
      message:
        type: string
    required: [message]
```

`tools/echo/echo_impl.py`:

```python
class EchoTool:
    def __init__(self, meta, params):
        self.meta = meta
        self.params = params

    def predict(self, request):
        return {"message": request["inputs"]["message"]}
```

Start or restart the server after adding the files. Check discovery and invoke the tool through HTTP:

```bash
curl http://127.0.0.1:8000/tools
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"tool_id":"echo","request":{"inputs":{"message":"hello"}}}'
```

The response contains `"output":{"message":"hello"}`. MCP clients can list and call the `echo` tool through `http://127.0.0.1:8000/mcp`. The `io.input_schema` describes the exposed MCP arguments. The configuration also supports multiple `variants` and `tools` per package; see `eyetools/core/config_loader.py` for the supported fields.

### Subprocesses and tool environments

Set `runtime.load_mode: subprocess` to run a tool in a worker process. Tools can select their own `envs/<name>/pyproject.toml` with `environment_ref: <name>`. For a tool in this checkout, create its environment and install the framework and tool-specific dependencies:

```bash
mkdir -p envs/my-tool
cd envs/my-tool
uv init --no-package --no-workspace --python 3.12
uv add --no-workspace --editable ../..
uv add --no-workspace <tool-dependency>
cd ../..
```

Set `environment_ref: my-tool` in the tool configuration. Use a Python version compatible with both the tool and the framework. Worker execution uses `uv`; the server working directory is the workspace root used to resolve `envs/` and relative paths.

Tools requiring GPU allocation can declare the relevant runtime policy and `model.weight_paths`. `uv run eyetools-mcp estimate-vram --tool-path tools` estimates memory using declared files. No tool-specific weight paths or profiles are bundled.

## Server architecture

- `eyetools/core/config_loader.py`, `schema.py`, `loader.py`, and `registry.py`: configuration normalization, discovery, and metadata.
- `tool_manager.py`, `process_manager.py`, and `worker_entry.py`: execution, lifecycle, worker IPC, and resource management.
- `env_manager.py`, `gpu_allocator.py`, and `vram_estimator.py`: isolated environments and optional device allocation.
- `eyetools/mcp_server.py` and `eyetools/cli.py`: MCP/HTTP API and CLI.

Use `--lifecycle-mode lazy` to load on first use, `eager` to preload, or `dynamic` for idle eviction. Management endpoints include `/lifecycle`, `/metrics`, `/admin/resources`, `/admin/preload/{tool_id}`, and `/admin/evict`.

## Development

```bash
uv run pytest
uv build
```

The integration tests generate temporary, non-domain tools and exercise discovery, in-process and subprocess execution, and MCP calls. No tool or model is needed to run the tests.
