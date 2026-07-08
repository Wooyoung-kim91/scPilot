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
from pathlib import Path




def default_workdir_for_input(input_path: str) -> str:
    """Per-input session dir. Thin re-export of the single source in ``scpilot.session`` (I-12) — kept
    at module level (public API) but imported lazily so importing this module does not pull the
    scientific stack before ``init_runtime()`` runs."""
    from scpilot.session import default_workdir_for_input as _f
    return _f(input_path)


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
    import anyio
    from mcp.server.fastmcp import FastMCP

    from scpilot import __version__, tools
    from scpilot.llm import prompts
    from scpilot.session import Session

    specs = tools.all_specs()  # triggers registration of all core tool modules

    lg = _configure_io()
    specs = _select_specs(specs, lg)   # F5: optional env-gated allow/deny filter

    # Why offload: FastMCP calls a *sync* tool handler directly on the asyncio event loop
    # (mcp/server/fastmcp func_metadata: `return fn(...)` when the handler is not a coroutine).
    # So a long tool (ingest 100k+ cells / train_scvi / benchmark / cnv) blocks the loop for
    # minutes, the server cannot answer the client's protocol PINGs, and the client tears the
    # stdio connection down mid-run ("works for a while then disconnects"). We instead run the
    # blocking body in a worker thread so the loop stays free to answer pings for the full
    # duration. Capacity 1 serializes tool execution: the body pins the *process-global* RNG
    # (set_global_seed) and mutates scanpy's global settings, so two concurrent tools would
    # corrupt each other's determinism — the reproducibility invariant requires one-at-a-time.
    _tool_limiter = anyio.CapacityLimiter(1)
    # Model-agnostic guidance: ship the orchestration brief in the MCP `initialize` handshake so
    # EVERY client's LLM (Claude Code, Codex, a local model) — not just Claude — sees how to drive
    # the pipeline. The full canonical flow is fetchable via prompt/resource/tool below.
    mcp = FastMCP("scpilot", instructions=prompts.MCP_INSTRUCTIONS)

    def _make_handler(name: str):
        def _run_blocking(input: str, workdir: str, params: dict | None, seed: int) -> dict:
            """The synchronous tool body — run in a worker thread (see _tool_limiter above)."""
            from scpilot import schemas as S
            from scpilot.repro import set_global_seed
            lg.info("tool=%s input=%s seed=%s", name, input, seed)
            params = dict(params or {})
            # optional LLM narration for reasoning_log.md (not a tool param)
            reasoning = params.pop("reasoning", None)
            try:
                # pin RNGs per call so mode-1 (MCP) is reproducible like the CLI (plan A1).
                set_global_seed(seed)
                wd = workdir or default_workdir_for_input(input)
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

        async def handler(input: str, workdir: str = "", params: dict | None = None,
                          seed: int = 0) -> dict:
            # async handler => FastMCP awaits it, keeping the event loop responsive to protocol
            # pings while the blocking body runs off-loop in a worker thread (serialized by
            # _tool_limiter). This is what stops the client from dropping the connection during
            # long tools. Signature is unchanged so the exposed tool schema is identical.
            return await anyio.to_thread.run_sync(
                _run_blocking, input, workdir, params, seed, limiter=_tool_limiter)
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

    # --- model-agnostic workflow guidance, delivered over EVERY MCP channel so any client can
    # reach it regardless of which capabilities its LLM host supports (prompt / resource / tool) ---
    @mcp.tool()
    def scpilot_guidance() -> dict:
        """Return scpilot's full canonical analysis workflow (the same pipeline mode-2 follows):
        golden rules + step-by-step flow (QC -> preprocess -> cluster/markers -> marker-DB-free
        Tier-1 -> integration + per-embedding annotation -> harmonize/benchmark/best -> Tier-2
        subtype -> CNV/malignancy -> finalize/report) + annotation & malignancy hard rules.
        Call this once at the start of a run if your client did not surface the server instructions."""
        return {"workflow": prompts.full_workflow_guidance()}

    @mcp.prompt(name="scpilot_workflow",
                description="scpilot's full canonical scRNA-seq analysis pipeline + golden rules.")
    def scpilot_workflow() -> str:
        return prompts.full_workflow_guidance()

    @mcp.resource("scpilot://workflow", name="scpilot_workflow",
                  description="scpilot canonical analysis workflow (model-agnostic).",
                  mime_type="text/markdown")
    def scpilot_workflow_resource() -> str:
        return prompts.full_workflow_guidance()

    names = [f"{s.name}_tool" for s in specs] + ["scpilot_version", "scpilot_guidance"]
    lg.info("scpilot MCP server ready (v%s) — tools: %s", __version__, ", ".join(names))
    return mcp


def main() -> None:
    """Entry point for ``scpilot mcp`` — run the stdio server."""
    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
