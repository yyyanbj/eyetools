"""Core framework components for EyeTools.

Modules:
  config_loader: Parse YAML/Python tool configs into normalized tool definitions.
  description_manager: Manage tool natural language descriptions for retrieval and matching.
  env_manager: Manage uv-based environment resolution and execution wrappers.
  errors: Custom exceptions for core framework.
  loader: Filesystem discovery of tool packages and registration.
  logger: Structured logging and request tracing utilities.
  process_manager: Manage subprocess workers for tools running out-of-process.
  registry: In-memory registry of tool metadata.
  tool_base: New abstract base API (thin; can wrap legacy base if needed).
  tool_manager: Instantiate and manage tool instances (in-process and out-of-process).
  utils: Helper functions (hashing, logging, resource usage, path ops).
  worker_entry: Entrypoint for out-of-process tool workers.
"""

from .registry import ToolRegistry  # noqa: F401
from .tool_base import ToolBase  # noqa: F401