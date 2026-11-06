"""CLI entrypoint for eyetools MCP server."""

from __future__ import annotations
import argparse
import json
import sys
import uvicorn
from pathlib import Path
from dotenv import load_dotenv
from .mcp_server import create_app

from eyetools.core.logger import setup_logger
from eyetools.core.process_manager import DEFAULT_MODEL_TIMEOUT_S
from eyetools.core.registry import ToolRegistry
from eyetools.core.loader import discover_tools
from eyetools.core.tool_manager import ToolManager


def build_parser():
    p = argparse.ArgumentParser(prog="eyetools-mcp", description="Eytools MCP Server")
    sub = p.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="Start MCP server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    # Backward/alternative flags: --tool-path (repeatable) and --tools-dir (single or repeatable)
    serve.add_argument(
        "--tool-path",
        action="append",
        dest="tool_paths",
        default=[],
        help="Additional tool root path (repeatable). Use --tools-dir as an alias.",
    )
    serve.add_argument(
        "--tools-dir",
        action="append",
        dest="tool_paths",
        help="Alias for --tool-path (repeatable)",
    )
    serve.add_argument(
        "--include-examples", action="store_true", help="Include packaged examples"
    )
    serve.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (dev only)"
    )
    serve.add_argument(
        "--preload",
        action="store_true",
        help="Instantiate and load (lazy) models at startup",
    )
    serve.add_argument(
        "--preload-subprocess",
        action="store_true",
        help="Also preload subprocess tools (if any)",
    )
    serve.add_argument(
        "--log-dir",
        help="Directory to write log file (eyetools.log). If not set, only stdout is used.",
    )
    serve.add_argument(
        "--log-level",
        default="DEBUG",
        help="Logging level (overrides EYETOOLS_LOG_LEVEL env var)",
    )
    # New lifecycle modes: eager (all models upfront), lazy (on-demand), dynamic (idle eviction)
    serve.add_argument(
        "--lifecycle-mode",
        choices=["eager", "lazy", "dynamic"],
        help="Model/tool lifecycle strategy: eager=preload all (+subprocess optional), lazy=load on first use, dynamic=lazy plus auto idle/unload",
    )
    serve.add_argument(
        "--dynamic-mark-idle-s",
        type=float,
        default=300.0,
        help="(dynamic mode) seconds of inactivity before marking a tool IDLE",
    )
    serve.add_argument(
        "--dynamic-unload-s",
        type=float,
        default=900.0,
        help="(dynamic mode) seconds of inactivity before unloading an IDLE tool",
    )
    serve.add_argument(
        "--dynamic-interval-s",
        type=float,
        default=60.0,
        help="(dynamic mode) maintenance loop interval in seconds",
    )
    serve.add_argument(
        "--model-timeout-s",
        type=float,
        default=DEFAULT_MODEL_TIMEOUT_S,
        help="Default subprocess model request/load/warmup timeout in seconds",
    )
    serve.add_argument(
        "--parallel-subprocess",
        action="store_true",
        help="Preload subprocess tools in parallel threads (eager mode recommended)",
    )
    serve.add_argument(
        "--parallel-subprocess-workers",
        type=int,
        default=4,
        help="Max worker threads for parallel subprocess preload",
    )
    serve.add_argument(
        "--description-config",
        help="Path to YAML/JSON file mapping tool IDs/patterns to descriptions",
    )
    serve.add_argument(
        "--exclude",
        action="append",
        dest="exclude_packages",
        default=[],
        help="Exclude tool packages by category (repeatable). Example: --exclude testing --exclude web_search",
    )

    probe = sub.add_parser(
        "probe", help="Run a single tool package without launching the server"
    )
    probe.add_argument(
        "--package-path",
        required=True,
        help="Path to a single tool package directory containing config.yaml",
    )
    probe.add_argument(
        "--variant", help="Variant name to select when multiple variants exist"
    )
    probe.add_argument(
        "--tool-id", help="Explicit tool id override (takes precedence over variant)"
    )
    probe.add_argument(
        "--list",
        action="store_true",
        help="List discovered tools under the package and exit",
    )
    probe.add_argument(
        "--ensure", action="store_true", help="Ensure tool is loaded before predict"
    )
    probe.add_argument(
        "--warmup", action="store_true", help="Run warmup hook before predict"
    )
    probe.add_argument("--input", help="Path to JSON request file for predict")
    probe.add_argument("--input-json", help="Inline JSON request for predict")
    probe.add_argument(
        "--workspace-root",
        help="Workspace root for environment resolution (defaults to current directory)",
    )
    probe.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for probe command (overrides EYETOOLS_LOG_LEVEL env var)",
    )

    estimate_vram = sub.add_parser(
        "estimate-vram", help="Estimate VRAM requirements for discovered tools"
    )
    estimate_vram.add_argument(
        "--tool-path",
        action="append",
        dest="tool_paths",
        default=[],
        help="Tool root path to scan (repeatable; defaults to ./tools).",
    )
    estimate_vram.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root used to resolve relative weight paths.",
    )
    estimate_vram.add_argument(
        "--write-config",
        action="store_true",
        help="Write estimated VRAM metadata back to tool config files.",
    )
    return p


def _select_target_tool(metas, variant=None, tool_id=None):
    candidates = metas
    if tool_id:
        candidates = [m for m in candidates if m.id == tool_id]
    if variant:
        candidates = [m for m in candidates if m.variant == variant]
    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) == 0:
        return None, "No tool matched the given filters (variant/tool-id)"
    return None, "Multiple tools matched; provide --variant or --tool-id"


def _load_json_input(path_arg: str | None, inline_json: str | None):
    if path_arg:
        return json.loads(Path(path_arg).read_text(encoding="utf-8"))
    if inline_json:
        return json.loads(inline_json)
    return None


def run_probe(args) -> int:
    # Load environment variables from .env if present so probe commands can pick up API keys
    try:
        load_dotenv()
    except Exception as dotenv_err:
        print(f"[warn] Failed to load .env: {dotenv_err}", file=sys.stderr)
    setup_logger(log_level=args.log_level)
    package_path = Path(args.package_path).expanduser().resolve()
    if not package_path.exists():
        print(f"[error] package path not found: {package_path}", file=sys.stderr)
        return 2

    registry = ToolRegistry()
    errors = discover_tools([package_path], registry, [])
    metas = registry.list()
    if errors:
        for err in errors:
            print(f"[discover] error: {err}", file=sys.stderr)

    if not metas:
        print(f"[error] no tools discovered under {package_path}", file=sys.stderr)
        return 2

    if args.list:
        print(f"[discover] found {len(metas)} tool(s) under {package_path}")
        for m in metas:
            print(f" - {m.id} (variant={m.variant}, entry={m.entry})")
        return 0

    target, reason = _select_target_tool(
        metas, variant=args.variant, tool_id=args.tool_id
    )
    if not target:
        print(f"[error] {reason}", file=sys.stderr)
        print("Candidates:")
        for m in metas:
            print(f" - {m.id} (variant={m.variant})")
        return 3

    workspace_root = (
        Path(args.workspace_root).expanduser().resolve()
        if args.workspace_root
        else Path.cwd()
    )
    manager = ToolManager(registry=registry, workspace_root=workspace_root)

    if args.ensure:
        res = manager.ensure_loaded(target.id)
        print(json.dumps({"ensure_loaded": res}, ensure_ascii=False))

    if args.warmup:
        res = manager.warmup_tool(target.id)
        print(json.dumps({"warmup": res}, ensure_ascii=False))

    req = _load_json_input(args.input, args.input_json)
    if req is not None:
        out = manager.predict(target.id, req)
        print(json.dumps({"output": out}, ensure_ascii=False))

    if not (args.ensure or args.warmup or req is not None):
        print(
            "No action requested. Use --ensure, --warmup, or --input/--input-json for predict."
        )
    return 0


def run_estimate_vram(args) -> int:
    from eyetools.core.vram_estimator import (
        build_estimate_report,
        write_estimates_to_config,
    )

    workspace_root = Path(args.workspace_root).expanduser().resolve()
    registry = ToolRegistry()
    tool_paths = (
        [Path(path).expanduser().resolve() for path in args.tool_paths]
        if args.tool_paths
        else [workspace_root / "tools"]
    )
    errors = []
    if args.tool_paths:
        errors.extend(
            f"tool path not found: {path}" for path in tool_paths if not path.exists()
        )
    try:
        errors = discover_tools(tool_paths, registry, errors)
    except Exception as exc:
        errors.append(f"tool discovery failed: {exc}")
    report = build_estimate_report(registry.list(), workspace_root=workspace_root)
    report["discover_errors"] = errors
    report["write_config"] = bool(args.write_config)
    if args.write_config:
        changed_configs = []
        write_errors = []
        if not errors:
            for config_path in _estimate_vram_config_paths(tool_paths):
                try:
                    if write_estimates_to_config(
                        config_path, workspace_root=workspace_root
                    ):
                        changed_configs.append(str(config_path))
                except Exception as exc:
                    write_errors.append(f"{config_path}: {exc}")
        report["changed_configs"] = changed_configs
        if write_errors:
            report["write_errors"] = write_errors
            errors.extend(write_errors)
            report["discover_errors"] = errors
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if errors else 0


def _estimate_vram_config_paths(tool_paths: list[Path]) -> list[Path]:
    paths = set()
    for tool_path in tool_paths:
        if not tool_path.exists():
            continue
        if tool_path.is_file() and tool_path.name in {"config.yaml", "config.yml"}:
            paths.add(tool_path)
            continue
        for name in ("config.yaml", "config.yml"):
            paths.update(tool_path.rglob(name))
    return sorted(paths)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "probe":
        return run_probe(args)
    if args.command == "estimate-vram":
        return run_estimate_vram(args)
    if args.command != "serve":
        parser.print_help()
        return 1

    # Setup log directory env before app creation (logging module reads env once)
    if getattr(args, "log_dir", None):
        log_dir_path = Path(args.log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        setup_logger(log_dir=args.log_dir, log_level=args.log_level)
    else:
        setup_logger(log_level=args.log_level)

    app = create_app(
        include_examples=args.include_examples,
        tool_paths=args.tool_paths,
        preload=args.preload,
        preload_subprocess=args.preload_subprocess,
        lifecycle_mode=getattr(args, "lifecycle_mode", None),
        dynamic_mark_idle_s=getattr(args, "dynamic_mark_idle_s", 300.0),
        dynamic_unload_s=getattr(args, "dynamic_unload_s", 900.0),
        dynamic_interval_s=getattr(args, "dynamic_interval_s", 60.0),
        parallel_subprocess=getattr(args, "parallel_subprocess", False),
        parallel_subprocess_workers=getattr(args, "parallel_subprocess_workers", 4),
        description_config_path=getattr(args, "description_config", None),
        exclude_packages=getattr(args, "exclude_packages", []),
        model_timeout_s=getattr(args, "model_timeout_s", DEFAULT_MODEL_TIMEOUT_S),
    )

    # Configure uvicorn logging with timestamps
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s %(levelprefix)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }

    uvicorn.run(
        app, host=args.host, port=args.port, reload=args.reload, log_config=log_config
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
