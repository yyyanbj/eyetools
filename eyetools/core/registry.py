"""In-memory tool registry."""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import threading


@dataclass
class ToolMeta:
    id: str
    entry: str
    version: str
    package: Optional[str] = None
    variant: Optional[str] = None
    runtime: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    warmup: Dict[str, Any] = field(default_factory=dict)
    io: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    root_dir: Optional[str] = None
    extra_requires: List[str] = field(default_factory=list)
    python: Optional[str] = None
    category: Optional[str] = None
    environment_ref: Optional[str] = None
    tool_name: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(self):
        self._lock = threading.RLock()
        self._tools: Dict[str, ToolMeta] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, meta: ToolMeta):
        with self._lock:
            if meta.id in self._tools:
                raise ValueError(f"Duplicate tool id: {meta.id}")
            self._tools[meta.id] = meta
            aliases = list(meta.aliases)
            if meta.tool_name:
                aliases.append(meta.tool_name)
            for alias in dict.fromkeys(aliases):
                if alias == meta.id:
                    continue
                previous = self._aliases.get(alias)
                if previous is not None and previous != meta.id:
                    raise ValueError(f"Duplicate tool alias: {alias}")
                self._aliases[alias] = meta.id

    def get(self, tool_id: str) -> Optional[ToolMeta]:
        canonical_id = self._aliases.get(tool_id, tool_id)
        return self._tools.get(canonical_id)

    def list(self) -> List[ToolMeta]:
        return list(self._tools.values())

    def remove(self, tool_id: str):
        with self._lock:
            canonical_id = self._aliases.get(tool_id, tool_id)
            meta = self._tools.pop(canonical_id, None)
            if meta is not None:
                for alias in list(self._aliases):
                    if self._aliases[alias] == canonical_id:
                        self._aliases.pop(alias, None)

    def clear(self):
        with self._lock:
            self._tools.clear()
            self._aliases.clear()


__all__ = ["ToolRegistry", "ToolMeta"]
