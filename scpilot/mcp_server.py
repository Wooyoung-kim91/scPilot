"""FastMCP (stdio) server — scpilot plan A6/C2.

A6 spike: expose ONE read-only tool (``inspect_h5ad``) over stdio so we can verify
the transport works from both Claude Code and Codex CLI (tool discovery, a short
call, stderr/stdout hygiene). The full registry (all tools + job model) lands in C2.

Hard rules (stdio MCP):
- **stdout carries ONLY protocol JSON** — every log/warning goes to stderr.
- Run ``init_runtime()`` before importing/using numba-backed code (scanpy/umap)
  so a detached session's numba cache does not break imports.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings


def _select_specs(specs, lg):
    """F5: optionally restrict which registry tools the MCP server exposes.

    The server exposes EVERY registered tool by default (the primary integration is a trusted
    local host such as Claude Code). For tighter deployments, two env vars gate the surface by
    tool name (without the ``_tool`` suffix):
      - ``SCPILOT_MCP_ENABLE_TOOLS`` — comma-separated allowlist (only these are exposed).
      - ``SCPILOT_MCP_DISABLE_TOOLS`` — comma-separated denylist (these are removed).
    Allowlist is applied first, then denylist. Unknown names are ignored (logged)."""
    def _names(var):
        return {n.strip() for n in os.environ.get(var, "").split(",") if n.strip()}

    enable, disable = _names("SCPILOT_MCP_ENABLE_TOOLS"), _names("SCPILOT_MCP_DISABLE_TOOLS")
    selected = list(specs)
    if enable:
        selected = [s for s in selected if s.name in enable]
        lg.info("MCP allowlist active (SCPILOT_MCP_ENABLE_TOOLS): %s", ", ".join(sorted(enable)))
    if disable:
        selected = [s for s in selected if s.name not in disable]
        lg.info("MCP denylist active (SCPILOT_MCP_DISABLE_TOOLS): %s", ", ".join(sorted(disable)))
    return selected


def _configure_io() -> logging.Logger:
    """Keep stdout clean for the protocol; route logs + warnings to stderr."""
    # numba/matplotlib caches + njit patch (detached-session safety)
    from scpilot.vendor.harness import init_runtime
    init_runtime()

    # Python warnings -> stderr (never stdout)
    logging.captureWarnings(True)
    warnings.simplefilter("default")

    lg = logging.getLogger("scpilot.mcp")
    lg.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [scpilot.mcp] %(message)s"))
    lg.handlers.clear()
    lg.addHandler(handler)
    lg.propagate = False
    return lg


def build_server():
    """Create the FastMCP server, exposing EVERY tool in the registry dynamically.

    Tools register themselves in ``scpilot.tools.REGISTRY`` at import time; this
    function wraps each as an MCP tool that opens/creates a Session and dispatches
    through the registry. So adding a B-tool (``@register(...)``) auto-exposes it
    over MCP — no edits here. Session working dir comes from a ``workdir`` arg
    (defaults next to the input) so MCP callers get the same on-disk session model.
    """
    from pathlib import Path

    from mcp.server.fastmcp import FastMCP

    from scpilot import __version__, tools
    from scpilot.session import Session

    specs = tools.all_specs()  # triggers registration of all core tool modules

    lg = _configure_io()
    specs = _select_specs(specs, lg)   # F5: optional env-gated allow/deny filter
    mcp = FastMCP("scpilot")

    def _make_handler(name: str):
        def handler(input: str, workdir: str = "", params: dict | None = None,
                    seed: int = 0) -> dict:
            from scpilot import schemas as S
            from scpilot.repro import set_global_seed
            from scpilot.session import DEFAULT_RUN_DIR
            lg.info("tool=%s input=%s seed=%s", name, input, seed)
            params = dict(params or {})
            # optional LLM narration for reasoning_log.md (not a tool param)
            reasoning = params.pop("reasoning", None)
            try:
                # pin RNGs per call so mode-1 (MCP) is reproducible like the CLI (plan A1).
                set_global_seed(seed)
                wd = workdir or DEFAULT_RUN_DIR
                session = Session.create(wd, input_path=input)
                result = tools.run(name, session, **params)
                # result-plot rule + run_log.jsonl + reasoning_log.md via the shared chokepoint
                # (plan C1): IDENTICAL to the CLI `step` path, so mode-1 runs are fully
                # replayable and the records cannot drift between drivers.
                try:
                    session.record_tool_run(result, params=params, seed=seed,
                                            reasoning=reasoning)
                except Exception:  # noqa: BLE001 — logging must never break the tool result
                    lg.exception("run/reasoning logging failed for %s", name)
                return result.to_dict()
            except Exception as exc:  # noqa: BLE001 — MCP must return a structured error, not throw
                lg.exception("tool %s failed", name)
                return S.error(name, "internal", f"{type(exc).__name__}: {exc}").to_dict()
        return handler

    for spec in specs:
        handler = _make_handler(spec.name)
        handler.__name__ = f"{spec.name}_tool"
        handler.__doc__ = (
            f"{spec.description}\n\n"
            "Args:\n"
            "    input: absolute path to a .h5ad on the server filesystem.\n"
            "    workdir: optional session directory (defaults next to input).\n"
            "    params: optional tool parameters (dict). May include a 'reasoning' "
            "string (the WHY for this step) — it is stripped from tool params and "
            "recorded in reasoning_log.md, not passed to the tool.\n"
            "    seed: global RNG seed pinned before the call (default 0) — recorded "
            "in run_log.jsonl so the run is reproducible/replayable."
        )
        mcp.tool(name=f"{spec.name}_tool")(handler)

    @mcp.tool()
    def scpilot_version() -> dict:
        """Return the scpilot version (cheap connectivity check)."""
        return {"scpilot_version": __version__}

    names = [f"{s.name}_tool" for s in specs] + ["scpilot_version"]
    lg.info("scpilot MCP server ready (v%s) — tools: %s", __version__, ", ".join(names))
    return mcp


def main() -> None:
    """Entry point for ``scpilot mcp`` — run the stdio server."""
    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
