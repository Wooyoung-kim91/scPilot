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

    specs = tools.all_specs()  # triggers registration of all core tool modules

    lg = _configure_io()
    mcp = FastMCP("scpilot")

    def _make_handler(name: str):
        def handler(input: str, workdir: str = "", params: dict | None = None) -> dict:
            from scpilot import schemas as S
            from scpilot.session import DEFAULT_RUN_DIR
            lg.info("tool=%s input=%s", name, input)
            params = dict(params or {})
            # optional LLM narration for reasoning_log.md (not a tool param)
            reasoning = params.pop("reasoning", None)
            try:
                wd = workdir or DEFAULT_RUN_DIR
                session = Session.create(wd, input_path=input)
                result = tools.run(name, session, **params)
                # result-plot rule: render a stage-appropriate figure for THIS step and
                # attach it, so every step returns a plot (not just numbers).
                if result.status == "success":
                    try:
                        from scpilot.core.autoplot import auto_plots
                        extra = auto_plots(session, name, result.summary)
                        if extra:
                            result.artifacts = list(result.artifacts or []) + extra
                    except Exception:  # noqa: BLE001
                        lg.exception("auto-plot failed for %s", name)
                # PARITY WITH CLI `step`: the MCP path must also populate run_log.jsonl
                # + reasoning_log.md, else mode-1 (MCP) runs leave no replay/run record.
                try:
                    session.log_run(S.RunLogRecord(
                        tool=name, status=result.status, params=params, summary=result.summary,
                        determinism_grade=result.determinism_grade,
                        output_checkpoint=result.checkpoint, error_code=result.error_code,
                        duration_s=result.duration_s,
                    ).to_dict())
                    plot_paths = [a.path for a in (result.artifacts or [])
                                  if getattr(a, "kind", None) == "png"]
                    session.log_reasoning(tool=name, params=params, summary=result.summary,
                                          reasoning=reasoning, status=result.status,
                                          checkpoint=result.checkpoint, plots=plot_paths)
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
            "recorded in reasoning_log.md, not passed to the tool."
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
