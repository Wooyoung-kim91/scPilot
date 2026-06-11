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


_LOADED = False


def _ensure_loaded() -> None:
    """Import core tool modules so their ``@register`` side-effects populate REGISTRY.

    Lazy (call-time) import avoids a module-load cycle: core modules import
    ``register`` from here, and we import them only once the registry is defined.
    """
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from scpilot.core import (ingest, io, state, qc, preprocess, cluster,  # noqa: F401
                              markers, plots, annotate, integrate, benchmark)


def get(name: str) -> ToolSpec:
    _ensure_loaded()
    if name not in REGISTRY:
        raise KeyError(f"unknown tool '{name}'. registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


def list_tools() -> list[dict]:
    _ensure_loaded()
    return [{"name": s.name, "mutating": s.mutating, "long_running": s.long_running,
             "description": s.description} for s in REGISTRY.values()]


def all_specs() -> list[ToolSpec]:
    _ensure_loaded()
    return list(REGISTRY.values())


def run(name: str, session, **params) -> S.ToolResult:
    """Dispatch one tool through the registry (used by CLI step / replay / MCP)."""
    return get(name).fn(session, **params)


def make_replay_executor(session) -> Callable[[dict], dict]:
    """Build a stateful ``executor(run_log_record) -> new_summary`` for ``repro.replay_session``.

    Re-runs each recorded tool through the registry with its recorded params on a SHARED
    replay ``session``, so mutating tools accumulate state (checkpoint → next tool reads it)
    exactly as the original run did — no LLM in the loop. Raises on a failed re-run so the
    replay report records it as a mismatch.
    """
    def executor(rec: dict) -> dict:
        tool = rec["tool"]
        params = rec.get("params", {}) or {}
        res = run(tool, session, **params)
        if res.status != "success":
            raise RuntimeError(f"replay of '{tool}' failed: {res.error_code}: {res.error}")
        return res.summary
    return executor
