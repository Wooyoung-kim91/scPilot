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
import sys
import warnings


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

    # ensure core tool modules are imported so they register (A5: inspect; B-tools later)
    import scpilot.core.io  # noqa: F401

    lg = _configure_io()
    mcp = FastMCP("scpilot")

    def _make_handler(name: str):
        def handler(input: str, workdir: str = "", params: dict | None = None) -> dict:
            from scpilot import schemas as S
            lg.info("tool=%s input=%s", name, input)
            try:
                wd = workdir or str(Path(input).resolve().parent / f"{Path(input).stem}_scpilot")
                session = Session.create(wd, input_path=input)
                return tools.run(name, session, **(params or {})).to_dict()
            except Exception as exc:  # noqa: BLE001 — MCP must return a structured error, not throw
                lg.exception("tool %s failed", name)
                return S.error(name, "internal", f"{type(exc).__name__}: {exc}").to_dict()
        return handler

    for spec in tools.REGISTRY.values():
        handler = _make_handler(spec.name)
        handler.__name__ = f"{spec.name}_tool"
        handler.__doc__ = (
            f"{spec.description}\n\n"
            "Args:\n"
            "    input: absolute path to a .h5ad on the server filesystem.\n"
            "    workdir: optional session directory (defaults next to input).\n"
            "    params: optional tool parameters (dict)."
        )
        mcp.tool(name=f"{spec.name}_tool")(handler)

    @mcp.tool()
    def scpilot_version() -> dict:
        """Return the scpilot version (cheap connectivity check)."""
        return {"scpilot_version": __version__}

    names = [f"{s.name}_tool" for s in tools.REGISTRY.values()] + ["scpilot_version"]
    lg.info("scpilot MCP server ready (v%s) — tools: %s", __version__, ", ".join(names))
    return mcp


def main() -> None:
    """Entry point for ``scpilot mcp`` — run the stdio server."""
    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
