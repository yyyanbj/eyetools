"""MCP Server (FastAPI) wrapping ToolManager.

CLI will import this module and call create_app().
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from loguru import logger
from pydantic import BaseModel

from .core.registry import ToolRegistry
from .core.loader import discover_tools
from .core.tool_manager import ToolManager
from .core.process_manager import DEFAULT_MODEL_TIMEOUT_S
from .core.description_manager import DescriptionManager
from .core.image_inputs import save_image_bytes

# Optional fastmcp import. We keep it optional so existing users without MCP needs are not blocked.
try:  # pragma: no cover - import side effect
    from fastmcp import FastMCP  # type: ignore
except Exception:  # noqa
    FastMCP = None  # type: ignore

"""Utilities"""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


class PredictRequest(BaseModel):
    tool_id: str
    request: Dict[str, Any] = {}


def _configure_external_logging():
    """Configure logging for FastAPI/Uvicorn and FastMCP to DEBUG with unified format."""

    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Loguru handlers are not stdlib logging.Handler instances. Use a standard
    # handler here so libraries such as httpx can emit request logs safely.
    base_handlers = []
    target_loggers = [
        # Framework
        "fastapi",
        "starlette",
        # Uvicorn family
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "uvicorn.asgi",
        # FastMCP and friends
        "fastmcp",
        # "mcp",
        # Useful libraries often used by servers
        "anyio",
        "httpx",
    ]

    for lname in target_loggers:
        std_logger = logging.getLogger(lname)
        std_logger.setLevel(logging.DEBUG)
        # Clear existing handlers
        std_logger.handlers = []
        # Reuse eyetools.core handlers if any
        if base_handlers:
            for h in base_handlers:
                std_logger.addHandler(h)
        else:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            std_logger.addHandler(ch)


def _compute_tool_paths(include_examples: bool, extra_paths: List[str]) -> List[Path]:
    paths: List[Path] = []
    # Extra paths (CLI or env)
    for p in extra_paths:
        if p:
            paths.append(Path(p).resolve())
    # Examples only if explicitly requested
    if include_examples:
        examples_dir = Path(__file__).resolve().parent / "examples"
        if examples_dir.exists():
            paths.append(examples_dir)
    # Deduplicate
    dedup = []
    seen = set()
    for p in paths:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup


def create_app(
    include_examples: bool = False,
    tool_paths: Optional[List[str]] = None,
    preload: bool = False,
    preload_subprocess: bool = False,
    # Default lifecycle to eager to load all tools at startup as requested
    lifecycle_mode: Optional[str] = "eager",
    # Set large thresholds to effectively avoid IDLE and UNLOAD transitions unless explicitly called
    dynamic_mark_idle_s: float = 365 * 24 * 3600,
    dynamic_unload_s: float = 365 * 24 * 3600,
    dynamic_interval_s: float = 60.0,
    parallel_subprocess: bool = False,
    parallel_subprocess_workers: int = 4,
    enable_mcp: bool = True,
    # Revert default back to root for backward compatibility; can override via param or EYETOOLS_MCP_MOUNT_PATH env.
    mcp_mount_path: str = "/",
    # Optional descriptions config path; can also be provided via env EYETOOLS_DESCRIPTION_CONFIG
    description_config_path: Optional[str] = None,
    # Exclude packages by category (e.g., ['testing', 'web_search'])
    exclude_packages: Optional[List[str]] = None,
    model_timeout_s: float = DEFAULT_MODEL_TIMEOUT_S,
):
    # Ensure consistent DEBUG-level logging and format for FastAPI/FastMCP and friends
    _configure_external_logging()
    # Normalize trailing slash (except keep root as '/')
    if mcp_mount_path != "/" and mcp_mount_path.endswith("/"):
        mcp_mount_path = mcp_mount_path[:-1]
    registry = ToolRegistry()
    # Two env vars supported for flexibility: legacy EYETOOLS_TOOL_PATHS and new EYETOOLS_EXTRA_TOOL_PATHS
    combined_paths = tool_paths or []
    # If user did not specify any tool path flags or env vars, but a local ./tools directory exists, include it by default.
    if not combined_paths:
        default_dir = Path.cwd() / "tools"
        if default_dir.exists():
            combined_paths.append(str(default_dir))
    path_objs = _compute_tool_paths(include_examples, combined_paths)
    errors = discover_tools(path_objs, registry)
    if errors:
        for e in errors:
            logger.warning(f"discover error: {e}")
    # Filter out excluded packages
    if exclude_packages:
        exclude_set = set(exclude_packages)
        all_metas = registry.list()
        filtered_metas = [m for m in all_metas if m.package not in exclude_set]
        registry = ToolRegistry()
        for meta in filtered_metas:
            registry.register(meta)
        logger.info(f"Excluded packages: {exclude_set}")

    # Log discovered tools summary
    if registry.list():
        logger.info(f"Discovered tools (count={len(registry.list())}):")
        for meta in registry.list():
            logger.info(
                f" - id={meta.id} package={meta.package} variant={meta.variant} runtime={meta.runtime.get('load_mode', 'auto')} lazy={meta.model.get('lazy', True)} root={meta.root_dir}",
            )
    else:
        logger.warning(
            "No tools discovered. Check --tools-dir paths or environment variables."
        )
    tm = ToolManager(registry, model_timeout_s=model_timeout_s)
    # Initialize descriptions manager
    # Precedence: explicit arg > None
    desc_path = description_config_path
    description_manager = DescriptionManager(desc_path)

    app = FastAPI(title="eyetools-mcp", version="0.1.0")

    # ---- MCP Integration (refactored) ----------------------------------------
    def _setup_mcp():
        if not enable_mcp:
            return None
        if FastMCP is None:  # pragma: no cover - optional path
            logger.warning("[mcp] fastmcp not installed; skipping MCP endpoints")
            return None
        try:
            mcp_instance = FastMCP()
        except Exception as e:  # noqa
            logger.exception(f"[mcp] init failed: {e}")
            return None

        prefer_explicit = True  # could make this configurable later
        param_styles: Dict[str, str] = {}

        def _normalize_type(type_str: str) -> Dict[str, Any]:
            """Normalize custom type strings to JSON Schema."""
            tmap = {
                "string": {"type": "string"},
                "path": {"type": "string", "format": "path"},
                "number": {"type": "number"},
                "integer": {"type": "integer"},
                "float": {"type": "number"},
                "bool": {"type": "boolean"},
                "boolean": {"type": "boolean"},
                "object": {"type": "object"},
                "array": {"type": "array"},
            }
            # Handle union types like "string|number"
            if "|" in type_str:
                types = [t.strip() for t in type_str.split("|")]
                return {"oneOf": [tmap.get(t, {"type": "string"}) for t in types]}
            return tmap.get(type_str, {"type": "string"})

        def build_input_schema(meta) -> Dict[str, Any]:
            """Derive a JSON schema for tool inputs.

            Preference order:
            - io.input_schema (explicit schema defined by tool config)
            - io.input (legacy key)
            - else: best-effort object schema with no fixed properties

            Automatically normalizes shorthand types and adds required fields.
            """
            if isinstance(meta.io, dict):
                # Prefer explicit input_schema if provided by config
                if isinstance(meta.io.get("input_schema"), dict):
                    sch = meta.io["input_schema"]
                    # Normalize to a proper JSON schema object if needed
                    if isinstance(sch, dict) and "type" not in sch:
                        # Interpret as shorthand props map {name: type}
                        props = {}
                        required = []
                        for k, v in sch.items():
                            if isinstance(v, str):
                                props[k] = _normalize_type(v)
                                # Fields without nullable marker are required by default
                                if not v.endswith("?"):
                                    required.append(k)
                            else:
                                props[k] = (
                                    v
                                    if isinstance(v, dict)
                                    else _normalize_type(str(v))
                                )
                        result = {
                            "type": "object",
                            "properties": props,
                            "additionalProperties": True,
                        }
                        if required:
                            result["required"] = required
                        return result
                    return sch
                # Legacy key support
                if isinstance(meta.io.get("input"), dict):
                    return meta.io["input"]
            return {"type": "object", "properties": {}, "additionalProperties": True}

        def build_output_schema(meta) -> Dict[str, Any]:
            """Derive a JSON schema for tool outputs.

            Normalizes shorthand output_schema from config and adds field descriptions.
            """
            if isinstance(meta.io, dict):
                output_schema = meta.io.get("output_schema", {})
                if isinstance(output_schema, dict) and "type" not in output_schema:
                    # Normalize shorthand {field: type} to proper JSON Schema
                    props = {}
                    for k, v in output_schema.items():
                        if isinstance(v, str):
                            props[k] = _normalize_type(
                                v.rstrip("?")
                            )  # strip nullable marker
                        else:
                            props[k] = v if isinstance(v, dict) else {"type": "string"}
                    return {"type": "object", "properties": props}
                elif isinstance(output_schema, dict):
                    return output_schema
            return {"type": "object", "properties": {}}

        def extract_param_names(input_schema: Dict[str, Any], meta) -> List[str]:
            """Extract parameter names for explicit wrapper.

            Use object property names from the resolved input_schema when available.
            Avoid leaking meta.io keys like 'input_schema'/'output_schema' into the signature.
            """
            names: List[str] = []
            if isinstance(input_schema, dict) and input_schema.get("type") == "object":
                props = input_schema.get("properties", {})
                if isinstance(props, dict):
                    names.extend([k for k in props.keys()])
            # Fallback: no explicit params
            return list(dict.fromkeys(names))

        def register_single(meta):  # noqa: C901 complexity acceptable here
            # Keep the historical registry id internally, but expose the
            # canonical no-colon name through MCP.
            tool_id = meta.tool_name or meta.id
            input_schema = build_input_schema(meta)
            # Resolve human-friendly description for this tool
            mcp_desc = description_manager.get(tool_id, meta)

            # Attempt explicit signature - use dynamic function creation with proper signature
            def make_explicit():
                params = extract_param_names(input_schema, meta)
                if not params:
                    raise ValueError("no_params")
                import re
                import inspect

                mapping = {}
                used = set()
                san_list = []
                for p in params:
                    san = re.sub(r"[^0-9a-zA-Z_]+", "_", p) or "p"
                    if san[0].isdigit():
                        san = "p_" + san
                    base = san
                    i = 1
                    while san in used:
                        san = f"{base}_{i}"
                        i += 1
                    used.add(san)
                    mapping[san] = p
                    san_list.append(san)

                # Create async function with explicit signature using FunctionType
                # This allows FastMCP 2.x to introspect the actual parameters
                def _make_wrapper(tm, tid, mapping):
                    """Factory that creates a wrapper function with explicit params."""
                    param_names = list(mapping.keys())

                    # Create the async function body
                    async def _explicit_wrapper_impl(**kwargs):
                        """Wrapper with explicit params - signature will be modified below."""
                        try:
                            inputs = {}
                            for san, orig in mapping.items():
                                val = kwargs.get(san)
                                if val is not None:
                                    inputs[orig] = val
                            _raw = tm.predict(tid, {"inputs": inputs} if inputs else {})
                            return {
                                "output": _raw,
                                "__meta__": {
                                    "tool_id": tid,
                                    "inputs": inputs,
                                    "ts": time.time(),
                                },
                            }
                        except Exception as e:
                            import traceback

                            traceback.print_exc()
                            return {
                                "error": str(e),
                                "__meta__": {
                                    "tool_id": tid,
                                    "inputs": inputs if "inputs" in locals() else {},
                                    "ts": time.time(),
                                },
                            }

                    # Modify the signature to have explicit parameters
                    # This is key for FastMCP 2.x to recognize the tool params
                    required_params = set(input_schema.get("required", [])) if isinstance(input_schema, dict) else set()
                    new_params = [
                        inspect.Parameter(
                            name,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            default=(
                                inspect.Parameter.empty
                                if mapping[name] in required_params
                                else None
                            ),
                        )
                        for name in param_names
                    ]
                    new_sig = inspect.Signature(parameters=new_params)
                    _explicit_wrapper_impl.__signature__ = new_sig
                    _explicit_wrapper_impl.__name__ = (
                        f"tool_{tid.replace(':', '_')}_explicit"
                    )
                    _explicit_wrapper_impl.__doc__ = f"MCP wrapper explicit params for '{tid}' params={list(mapping.values())}"

                    return _explicit_wrapper_impl

                fn = _make_wrapper(tm, tool_id, mapping)
                return fn

            async def func_kwargs(**kwargs):  # type: ignore
                try:
                    _raw = tm.predict(tool_id, {"inputs": kwargs} if kwargs else {})
                    return {
                        "output": _raw,
                        "__meta__": {
                            "tool_id": tool_id,
                            "inputs": kwargs or {},
                            "ts": time.time(),
                        },
                    }
                except Exception as e:  # noqa
                    return {
                        "error": str(e),
                        "__meta__": {
                            "tool_id": tool_id,
                            "inputs": kwargs or {},
                            "ts": time.time(),
                        },
                    }

            func_kwargs.__name__ = f"tool_{tool_id.replace(':','_')}"
            func_kwargs.__doc__ = f"MCP wrapper for eyetools tool '{tool_id}' (kwargs)"

            async def func_payload(payload: Dict[str, Any] | None = None):  # type: ignore
                try:
                    _raw = tm.predict(
                        tool_id, {"inputs": (payload or {})} if payload else {}
                    )
                    return {
                        "output": _raw,
                        "__meta__": {
                            "tool_id": tool_id,
                            "inputs": (payload or {}),
                            "ts": time.time(),
                        },
                    }
                except Exception as e:  # noqa
                    return {
                        "error": str(e),
                        "__meta__": {
                            "tool_id": tool_id,
                            "inputs": (payload or {}),
                            "ts": time.time(),
                        },
                    }

            func_payload.__name__ = f"tool_{tool_id.replace(':','_')}_payload"
            func_payload.__doc__ = (
                f"MCP wrapper for eyetools tool '{tool_id}' (payload)"
            )

            # Decorator acquisition (handles FastMCP 1.x vs 2.x API variance)
            # FastMCP 2.x: uses output_schema param, input_schema auto-generated from function signature
            # FastMCP 1.x: supports input_schema param
            try:
                # Build output schema for FastMCP 2.x
                output_schema = build_output_schema(meta)

                # Try FastMCP 2.x style with output_schema
                try:
                    deco = mcp_instance.tool(
                        name=tool_id,
                        description=mcp_desc,
                        output_schema=output_schema,  # FastMCP 2.x uses output_schema
                    )  # type: ignore
                    logger.debug(
                        f"[mcp] FastMCP 2.x detected: using output_schema (tool={tool_id})"
                    )
                except TypeError as te:
                    if "output_schema" in str(te) or "input_schema" in str(te):
                        # FastMCP 1.x: use input_schema param
                        deco = mcp_instance.tool(name=tool_id, description=mcp_desc, input_schema=input_schema)  # type: ignore
                        logger.debug(
                            f"[mcp] FastMCP 1.x detected: using input_schema (tool={tool_id})"
                        )
                    else:
                        raise
                # explicit -> kwargs -> payload fallback
                if prefer_explicit:
                    try:
                        explicit_fn = make_explicit()
                        deco(explicit_fn)
                        param_styles[tool_id] = "explicit"
                        return
                    except Exception as e:  # noqa
                        logger.debug(
                            f"[mcp] explicit build failed tool={tool_id} reason={e}"
                        )
                try:
                    deco(func_kwargs)
                    param_styles[tool_id] = "kwargs"
                except Exception as e:  # noqa
                    if "**kwargs" in str(e) or "kwargs" in str(e):
                        deco2 = mcp_instance.tool(name=tool_id, description=mcp_desc)  # type: ignore
                        deco2(func_payload)
                        param_styles[tool_id] = "payload"
                        logger.info(f"[mcp] payload fallback tool={tool_id}")
                    else:
                        raise
            except Exception as e:  # noqa
                logger.warning(f"[mcp] failed to register tool id={tool_id} error={e}")

        for meta in registry.list():
            register_single(meta)

        # Mount HTTP app
        try:
            mcp_app = mcp_instance.http_app(transport="streamable-http")  # type: ignore
            if mcp_mount_path == "/":
                for r in mcp_app.routes:
                    if not any(
                        getattr(er, "path", None) == r.path for er in app.routes
                    ):
                        app.router.routes.append(r)
                app.router.lifespan_context = mcp_app.router.lifespan_context  # type: ignore
                logger.info("[mcp] routes merged at root")
            else:
                mount_path = mcp_mount_path.rstrip("/") or "/mcp"
                app.mount(mount_path, mcp_app)
                logger.info(f"[mcp] mounted at {mount_path}")
        except Exception as e:  # noqa
            logger.exception(f"[mcp] attach routes failed: {e}")

        # Final summary log
        names = sorted(param_styles.keys())
        logger.info(f"[mcp] available tools (count={len(names)}): {', '.join(names)}")
        app.state.fastmcp = mcp_instance
        app.state._mcp_param_styles = param_styles  # type: ignore

        def re_register_all():
            # clear and re-run registration (simple strategy: new instance)
            logger.info("[mcp] re-registering tools after reload...")
            app.state.fastmcp = None
            return _setup_mcp()

        app.state._mcp_reregister = re_register_all  # type: ignore
        return mcp_instance

    _setup_mcp()

    @app.get("/health")
    def health():
        return {"status": "ok", "tools": len(registry.list())}

    @app.get("/tools")
    def list_tools():
        # When auto selection, include detailed descriptions per tool
        metas_by_id = {m.id: m for m in registry.list()}
        tools_out = []
        for tid in metas_by_id.keys():
            m = metas_by_id.get(tid)
            if not m:
                continue
            out: Dict[str, Any] = {
                "id": m.id,
                "tool_name": m.tool_name or m.id,
                "package": m.package,
                "variant": m.variant,
                "category": m.category,
                "runtime": m.runtime,
                "model": m.model,
                "io": m.io,
                "tags": m.tags,
                "environment_ref": m.environment_ref,
                "python": m.python,
                "extra_requires": m.extra_requires,
            }
            if isinstance(m.io, dict):
                out["schema_version"] = m.io.get("schema_version", "1.0.0")
                out["input_schema"] = m.io.get("input_schema", {})
                out["output_schema"] = m.io.get("output_schema", {})
                out["description_schema"] = {
                    "schema_version": m.io.get("schema_version", "1.0.0"),
                    "tool_name": m.tool_name or m.id,
                    "description": m.io.get("description") or m.tool_name or m.id,
                }
            try:
                out["description"] = description_manager.get(m.tool_name or m.id, m)
            except Exception as desc_err:
                logger.error(
                    f"Failed to fetch description for tool_id={m.id}: {desc_err}"
                )
            tools_out.append(out)
        return {"tools": tools_out}

    @app.post("/predict")
    def predict(req: PredictRequest):
        try:
            raw = tm.predict(req.tool_id, req.request)
        except Exception as e:  # noqa
            import traceback

            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        meta = {
            "tool_id": req.tool_id,
            "inputs": (
                req.request.get("inputs") if isinstance(req.request, dict) else None
            )
            or req.request,
            "ts": time.time(),
        }
        return {"output": raw, "__meta__": meta}

    @app.post("/predict/upload")
    async def predict_upload(
        http_request: Request,
        tool_id: str = Form(...),
        image_field: str = Form("image_path"),
        files: List[UploadFile] = File(...),
    ):
        try:
            form = await http_request.form()
            request_payload: Dict[str, Any] = {"inputs": {}}
            raw_request = form.get("request")
            if isinstance(raw_request, str) and raw_request.strip():
                parsed = json.loads(raw_request)
                if not isinstance(parsed, dict):
                    raise ValueError("request form field must be a JSON object")
                request_payload = parsed
            inputs = request_payload.setdefault("inputs", {})
            if not isinstance(inputs, dict):
                raise ValueError("request.inputs must be an object")

            for key, value in form.multi_items():
                if key in {"tool_id", "image_field", "files", "request"}:
                    continue
                if isinstance(value, str):
                    inputs[key] = value

            uploaded_paths = []
            for upload in files:
                uploaded_paths.append(
                    save_image_bytes(await upload.read(), upload.filename)
                )
            inputs[image_field] = (
                uploaded_paths[0] if len(uploaded_paths) == 1 else uploaded_paths
            )
            raw = tm.predict(tool_id, request_payload)
        except Exception as e:  # noqa
            import traceback

            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        return {
            "output": raw,
            "__meta__": {
                "tool_id": tool_id,
                "inputs": request_payload.get("inputs", request_payload),
                "ts": time.time(),
            },
        }

    @app.get("/metrics")
    def metrics(tool_id: Optional[str] = Query(None)):
        return tm.get_metrics(tool_id=tool_id)

    @app.get("/lifecycle")
    def lifecycle(detailed: bool = Query(False)):
        return tm.lifecycle_states_detailed() if detailed else tm.lifecycle_states()

    @app.post("/admin/reload")
    def admin_reload(preload_models: bool = False):
        """Rediscover tools using original discovery context.
        Optionally preload models again.
        """
        ctx = app.state.discovery_ctx
        registry.clear()
        errors = discover_tools(
            _compute_tool_paths(ctx["include_examples"], (ctx["tool_paths"] or [])),
            registry,
        )
        if errors:
            for e in errors:
                logger.warning(f"reload discover error: {e}")
        if preload_models:
            tm.preload_all()
        # Re-register MCP tools if integration enabled
        if enable_mcp and getattr(
            app.state, "_mcp_reregister", None
        ):  # pragma: no cover
            try:
                app.state._mcp_reregister()  # type: ignore
            except Exception as e:  # noqa
                logger.warning(f"[mcp] MCP re-registration failed: {e}")
        return {
            "tools": [m.id for m in registry.list()],
            "errors": [str(e) for e in errors],
        }

    @app.post("/admin/evict")
    def admin_evict(tool_id: str = Query(...)):
        # unified unload supporting inproc or subprocess workers
        ok = tm.unload_any(tool_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Tool not loaded or not found")
        return {"tool_id": tool_id, "status": "unloaded"}

    @app.post("/admin/preload/{tool_id}")
    def admin_preload_one(tool_id: str):
        meta = registry.get(tool_id)
        if not meta:
            raise HTTPException(status_code=404, detail="Tool not found")
        try:
            result = tm.ensure_loaded(tool_id)
        except Exception as e:  # noqa
            raise HTTPException(status_code=500, detail=str(e))
        return result

    @app.get("/admin/resources")
    def admin_resources():
        return tm.resource_snapshot()

    @app.post("/admin/maintenance")
    def admin_maintenance(
        mark_idle_s: float | None = Query(None),
        unload_idle_s: float | None = Query(None),
    ):
        return tm.maintenance(mark_idle_s=mark_idle_s, unload_idle_s=unload_idle_s)

    @app.post("/admin/warmup")
    def admin_warmup(tool_id: str = Query(...)):
        try:
            return tm.warmup_tool(tool_id)
        except Exception as e:  # noqa
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/admin/warmup_all")
    def admin_warmup_all(
        runtime: str | None = Query(None, description="Filter: subprocess|inproc")
    ):
        if runtime not in (None, "subprocess", "inproc"):
            raise HTTPException(
                status_code=400, detail="runtime must be subprocess or inproc"
            )
        try:
            return tm.warmup_all(runtime=runtime)
        except Exception as e:  # noqa
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/admin/config")
    def admin_config():
        ctx = app.state.discovery_ctx
        return {
            "lifecycle_mode": ctx.get("lifecycle_mode"),
            "dynamic_mark_idle_s": ctx.get("dynamic_mark_idle_s"),
            "dynamic_unload_s": ctx.get("dynamic_unload_s"),
            "dynamic_interval_s": ctx.get("dynamic_interval_s"),
            "model_timeout_s": ctx.get("model_timeout_s"),
            "tool_paths": ctx.get("tool_paths"),
            "include_examples": ctx.get("include_examples"),
            "description_config_path": ctx.get("description_config_path"),
        }

    # Descriptions admin endpoints
    @app.get("/admin/descriptions")
    def admin_descriptions(include_tools: bool = Query(False)):
        summary = description_manager.summary()
        if include_tools:
            items = []
            for meta in registry.list():
                try:
                    items.append(
                        {
                            "id": meta.id,
                            "tool_name": meta.tool_name or meta.id,
                            "aliases": list(meta.aliases),
                            "description": description_manager.get(meta.tool_name or meta.id, meta),
                        }
                    )
                except Exception:
                    items.append(
                        {
                            "id": meta.id,
                            "tool_name": meta.tool_name or meta.id,
                            "aliases": list(meta.aliases),
                            "description": meta.tool_name or meta.id,
                        }
                    )
            summary["tools"] = items
        return summary

    @app.post("/admin/descriptions/reload")
    def admin_descriptions_reload(path: Optional[str] = Query(None)):
        if path:
            # Update path then reload
            description_manager.set_path(path)
        else:
            description_manager.reload()
        # Re-register MCP tools to update descriptions exposed to clients
        if enable_mcp and getattr(
            app.state, "_mcp_reregister", None
        ):  # pragma: no cover
            try:
                app.state._mcp_reregister()  # type: ignore
            except Exception as e:  # noqa
                logger.warning(
                    f"[mcp] MCP re-registration after descriptions reload failed: {e}"
                )
        return {"status": "ok", "summary": description_manager.summary()}

    # Debug / introspection endpoint for MCP (lists tool IDs FastMCP has registered)
    if enable_mcp:

        @app.get("/mcp/tools")  # pragma: no cover - simple passthrough
        async def mcp_tools():
            inst = getattr(app.state, "fastmcp", None)
            if not inst:
                return {"enabled": False, "tools": [], "tool_names": [], "count": 0}
            names: list[str] = []
            detailed: List[Dict[str, Any]] = []
            lifecycle_map = tm.lifecycle_states()
            style_map: Dict[str, str] = getattr(app.state, "_mcp_param_styles", {})

            # Try FastMCP 2.x API first (uses _tool_manager, async)
            tool_mgr = getattr(inst, "_tool_manager", None)
            if tool_mgr:
                try:
                    import asyncio

                    # Get tools from tool manager (async in FastMCP 2.x)
                    tools_result = tool_mgr.get_tools()
                    if asyncio.iscoroutine(tools_result):
                        tools_result = await tools_result
                    # FastMCP 2.x returns dict of name -> FunctionTool
                    if isinstance(tools_result, dict):
                        for k, v in tools_result.items():
                            names.append(k)
                            # FastMCP 2.x uses 'parameters' instead of 'input_schema'
                            schema = getattr(v, "parameters", None) or getattr(
                                v, "input_schema", {}
                            )
                            desc = getattr(v, "description", "")
                            detailed.append(
                                {
                                    "name": k,
                                    "description": desc,
                                    "input_schema": schema if schema else {},
                                    "param_style": style_map.get(k, "unknown"),
                                    "lifecycle_state": lifecycle_map.get(k),
                                }
                            )
                except Exception as e:  # noqa
                    logger.warning(f"[mcp_tools] FastMCP 2.x introspection failed: {e}")

            # Fallback: FastMCP 1.x API (_tools dict)
            if not detailed:
                try:
                    maybe = getattr(inst, "_tools", None) or getattr(
                        inst, "tools", None
                    )
                    if isinstance(maybe, dict):
                        for k, v in maybe.items():
                            names.append(k)
                            desc = getattr(v, "description", "")
                            schema = getattr(v, "input_schema", {})
                            detailed.append(
                                {
                                    "name": k,
                                    "description": desc,
                                    "input_schema": schema,
                                    "param_style": style_map.get(k, "unknown"),
                                    "lifecycle_state": lifecycle_map.get(k),
                                }
                            )
                except Exception:  # noqa
                    pass

            names = sorted(names)
            detailed = sorted(detailed, key=lambda x: x["name"])
            return {
                "enabled": True,
                "mount_path": mcp_mount_path,
                "tool_names": names,
                "tools": detailed,
                "count": len(names),
            }

        @app.post("/mcp/refresh")  # pragma: no cover - light endpoint
        def mcp_refresh():
            rer = getattr(app.state, "_mcp_reregister", None)
            if not rer:
                raise HTTPException(status_code=400, detail="MCP not enabled")
            rer()
            tools_after = getattr(app.state, "_mcp_param_styles", {})
            return {
                "status": "ok",
                "count": len(tools_after),
                "tools": sorted(tools_after.keys()),
            }

    # Lifecycle mode handling precedence:
    # 1. lifecycle_mode if provided (eager|lazy|dynamic)
    # 2. fallback to legacy --preload flags
    if lifecycle_mode:
        app.state.lifecycle_mode = lifecycle_mode
        if lifecycle_mode == "eager":
            # In eager mode we always include subprocess tools regardless of legacy preload_subprocess flag
            logger.info(
                f"[lifecycle] eager mode: preloading all tools (include_subprocess=True parallel_subprocess={parallel_subprocess})"
            )
            tm.preload_all(
                include_subprocess=True,
                parallel_subprocess=parallel_subprocess,
                max_workers=parallel_subprocess_workers,
            )  # eager always includes subprocess
        elif lifecycle_mode == "lazy":
            logger.info("[lifecycle] lazy mode: models load on first request")
            # nothing extra
        elif lifecycle_mode == "dynamic":
            logger.info(
                f"[lifecycle] dynamic mode: lazy load + background maintenance idle={dynamic_mark_idle_s}s unload={dynamic_unload_s}s interval={dynamic_interval_s}s"
            )
            # Start background maintenance task
            import asyncio

            async def _maintenance_loop():  # pragma: no cover (timing loop)
                await asyncio.sleep(dynamic_interval_s)
                while True:
                    try:
                        tm.maintenance(
                            mark_idle_s=dynamic_mark_idle_s,
                            unload_idle_s=dynamic_unload_s,
                        )
                    except Exception as e:  # noqa
                        logger.exception(f"maintenance loop error: {e}")
                    await asyncio.sleep(dynamic_interval_s)

            # Use FastAPI lifespan to manage startup/shutdown without deprecated on_event
            # Properly compose with any existing lifespan context to avoid conflicts
            existing_lifespan = app.router.lifespan_context

            @asynccontextmanager
            async def _dynamic_lifespan(app_: FastAPI):  # pragma: no cover
                """Dynamic mode lifespan that chains with existing lifespan contexts."""

                # Helper to start background maintenance
                async def _start():
                    if getattr(app_.state, "_maintenance_started", False):
                        return
                    app_.state._maintenance_task = asyncio.create_task(
                        _maintenance_loop()
                    )
                    app_.state._maintenance_started = True

                async def _stop():
                    task = getattr(app_.state, "_maintenance_task", None)
                    if task:
                        task.cancel()
                        try:
                            await task
                        except Exception:  # pragma: no cover - task cancellation
                            pass
                    app_.state._maintenance_started = False

                await _start()
                try:
                    yield
                finally:
                    await _stop()

            # Compose lifespans properly: if there's an existing lifespan, wrap it
            if existing_lifespan and existing_lifespan is not _dynamic_lifespan:

                @asynccontextmanager
                async def _composed_lifespan(app_: FastAPI):
                    """Chain dynamic maintenance with existing lifespan (e.g., FastMCP)."""
                    async with existing_lifespan(app_):
                        async with _dynamic_lifespan(app_) as lifespan_result:
                            yield lifespan_result

            else:
                _composed_lifespan = _dynamic_lifespan

            app.router.lifespan_context = _composed_lifespan
        else:
            logger.warning(f"Unknown lifecycle_mode={lifecycle_mode} (ignored)")
    else:
        # Legacy preload behavior
        if preload:
            logger.info(f"Preloading all tools (subprocess={preload_subprocess})...")
            tm.preload_all(include_subprocess=preload_subprocess)

    # Expose references for instrumentation/introspection
    app.state.tool_manager = tm
    app.state.registry = registry
    app.state.discovery_ctx = {
        "include_examples": include_examples,
        "tool_paths": tool_paths,
        "lifecycle_mode": lifecycle_mode,
        "dynamic_mark_idle_s": dynamic_mark_idle_s,
        "dynamic_unload_s": dynamic_unload_s,
        "dynamic_interval_s": dynamic_interval_s,
        "model_timeout_s": model_timeout_s,
        "description_config_path": desc_path,
    }

    # attach state for external access
    app.state.description_manager = description_manager

    return app


__all__ = ["create_app"]
