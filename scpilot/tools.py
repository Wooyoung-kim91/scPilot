"""Tool registry — scpilot plan A5 (minimal) → C1 (full).

Single source mapping a *stage name* to a callable that the CLI ``step``, the MCP
server, and ``replay`` all dispatch through. Each tool follows one calling
convention so the three drivers stay identical::

    fn(session: Session, **params) -> schemas.ToolResult

The tool reads what it needs from the ``session`` (input path / cached AnnData),
does its work, checkpoints via the session when mutating, and returns a
``ToolResult``. Tools register themselves at import time via ``@register``.

A5 scope: the registry + ``inspect`` (read-only). B-tools register as they land.
C1 adds the job interface (start/get_job_status/get_job_result/cancel) for the
long-running tools (scVI/Harmony/scib/CNV) on top of this same registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from scpilot import schemas as S


@dataclass
class ToolSpec:
    name: str
    fn: Callable               # fn(session, **params) -> ToolResult
    mutating: bool = False     # writes a checkpoint / changes session.adata
    long_running: bool = False # gets a job interface in C1
    description: str = ""


REGISTRY: dict[str, ToolSpec] = {}


def register(name: str, *, mutating: bool = False, long_running: bool = False,
             description: str = "") -> Callable:
    """Decorator: register ``fn(session, **params) -> ToolResult`` under ``name``."""
    def deco(fn: Callable) -> Callable:
        REGISTRY[name] = ToolSpec(name=name, fn=fn, mutating=mutating,
                                  long_running=long_running, description=description or (fn.__doc__ or "").strip())
        return fn
    return deco


def get(name: str) -> ToolSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown tool '{name}'. registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


def list_tools() -> list[dict]:
    return [{"name": s.name, "mutating": s.mutating, "long_running": s.long_running,
             "description": s.description} for s in REGISTRY.values()]


def run(name: str, session, **params) -> S.ToolResult:
    """Dispatch one tool through the registry (used by CLI step / replay / MCP)."""
    return get(name).fn(session, **params)


# --------------------------------------------------------------------------- #
# Registered tools
# --------------------------------------------------------------------------- #
@register("inspect", mutating=False, description="Read-only summary of the session input h5ad.")
def _inspect(session, **params) -> S.ToolResult:
    from scpilot.core.io import inspect_h5ad
    path = params.get("path") or session.manifest.input.get("path")
    if not path:
        return S.error("inspect", "missing_input", "no input path in session or params", recoverable=False)
    return inspect_h5ad(path)
