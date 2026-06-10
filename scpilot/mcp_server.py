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
    """Create and return the FastMCP server with the A6 read-only tool registered."""
    from mcp.server.fastmcp import FastMCP

    from scpilot import __version__
    from scpilot.core.io import inspect_h5ad

    lg = _configure_io()
    mcp = FastMCP("scpilot")

    @mcp.tool()
    def inspect_h5ad_tool(path: str) -> dict:
        """Read-only summary of an .h5ad file: shape, layers, obs schema, embeddings,
        uns keys, genomic-coordinate readiness. Does not load the X matrix. Returns
        the scpilot ToolResult as JSON.

        Args:
            path: absolute path to a .h5ad file on the server's filesystem.
        """
        lg.info("inspect_h5ad path=%s", path)
        result = inspect_h5ad(path)
        return result.to_dict()

    @mcp.tool()
    def scpilot_version() -> dict:
        """Return the scpilot version (cheap connectivity check)."""
        return {"scpilot_version": __version__}

    lg.info("scpilot MCP server ready (v%s) — tools: inspect_h5ad_tool, scpilot_version", __version__)
    return mcp


def main() -> None:
    """Entry point for ``scpilot mcp`` — run the stdio server."""
    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
