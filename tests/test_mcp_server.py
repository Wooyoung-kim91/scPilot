"""Integration test for the MCP stdio server — scpilot plan A6 spike.

Spawns ``python -m scpilot mcp`` as a real stdio subprocess and drives it with the
MCP client SDK (the same protocol Claude Code / Codex use): initialize ->
tools/list -> tools/call (success + error path). Validates transport + ToolResult
serialization over MCP. Uses a tiny fixture h5ad (no large I/O).
"""

import asyncio
import json
import sys

import anndata as ad
import numpy as np
from scipy import sparse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _parse(r):
    if getattr(r, "structuredContent", None):
        sc = r.structuredContent
        return sc.get("result", sc)
    for c in (r.content or []):
        if getattr(c, "text", None):
            return json.loads(c.text)
    return {}


def _tiny_h5ad(path):
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(1.0, size=(40, 30)).astype("float32"))
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(40)]
    a.var_names = [f"G{i}" for i in range(30)]
    a.layers["counts"] = a.X.copy()
    a.obs["sample_id"] = (["s1"] * 20 + ["s2"] * 20)
    a.write_h5ad(path)


async def _drive(h5ad_path, workdir):
    params = StdioServerParameters(command=sys.executable, args=["-m", "scpilot", "mcp"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            ver = _parse(await session.call_tool("scpilot_version", {}))
            ok = _parse(await session.call_tool(
                "inspect_tool", {"input": h5ad_path, "workdir": f"{workdir}/ok"}))
            err = _parse(await session.call_tool(
                "inspect_tool", {"input": f"{workdir}/missing.h5ad", "workdir": f"{workdir}/err"}))
            return names, ver, ok, err


async def _drive_guidance():
    """Fetch the model-agnostic workflow guidance over every MCP channel."""
    params = StdioServerParameters(command=sys.executable, args=["-m", "scpilot", "mcp"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            tool_names = {t.name for t in (await session.list_tools()).tools}
            prompt_names = {p.name for p in (await session.list_prompts()).prompts}
            res_uris = {str(r.uri) for r in (await session.list_resources()).resources}
            guidance = _parse(await session.call_tool("scpilot_guidance", {}))
            return init.instructions, tool_names, prompt_names, res_uris, guidance


def test_mcp_workflow_guidance_is_model_agnostic(tmp_path):
    # ANY MCP client (Claude Code, Codex, a local LLM) must receive the pipeline guidance:
    # in the initialize handshake AND fetchable via prompt / resource / tool.
    instructions, tool_names, prompt_names, res_uris, guidance = asyncio.run(_drive_guidance())
    assert instructions and "summary-in" in instructions          # shipped in `initialize`
    assert "scpilot_guidance" in instructions                      # fallback directive for any client
    assert "scpilot_workflow" in prompt_names                      # MCP prompt channel
    assert "scpilot://workflow" in res_uris                        # MCP resource channel
    assert "scpilot_guidance" in tool_names                        # tool channel (tool-only clients)
    wf = guidance["workflow"]
    assert "CANONICAL FLOW" in wf and "detect_state" in wf         # full pipeline present
    assert "malignant" in wf                                        # incl. annotation/CNV hard rules
    assert "Tier-4 consistency" in wf                              # incl. the independent audit/critique


def test_registry_tool_handlers_are_async_offloaded():
    # Regression guard: every registry-driven MCP tool handler MUST be async so FastMCP awaits
    # it and offloads the blocking body to a worker thread (see mcp_server._make_handler). If a
    # handler is sync, FastMCP runs it directly on the asyncio event loop, which freezes the loop
    # for the whole tool duration — the server then can't answer protocol pings and the client
    # drops the connection mid-run on long tools (ingest / train_scvi / benchmark / cnv).
    from scpilot.mcp_server import build_server

    srv = build_server()
    tools = srv._tool_manager.list_tools()
    registry_handlers = [t for t in tools if t.name.endswith("_tool")]
    assert registry_handlers, "expected registry tools to be exposed"
    offending = [t.name for t in registry_handlers if not t.is_async]
    assert not offending, f"these handlers are sync and would freeze the event loop: {offending}"
    # signature/schema must be unchanged by the async wrapper (input/workdir/params/seed)
    ingest = next(t for t in registry_handlers if t.name == "ingest_tool")
    assert {"input", "workdir", "params", "seed"} <= set(ingest.parameters["properties"])


def test_mcp_stdio_tool_discovery_and_call(tmp_path):
    h5ad = tmp_path / "tiny.h5ad"
    _tiny_h5ad(h5ad)
    names, ver, ok, err = asyncio.run(_drive(str(h5ad), str(tmp_path)))

    # registry-driven tool discovery (inspect auto-exposed as inspect_tool)
    assert {"inspect_tool", "scpilot_version"} <= names
    # short call
    assert ver["scpilot_version"]
    # read-only inspect returns a valid ToolResult summary
    assert ok["status"] == "success"
    assert ok["summary"]["n_obs"] == 40
    assert ok["summary"]["n_vars"] == 30
    assert "counts" in ok["summary"]["layers"]
    assert ok["summary"]["has_genomic_coords"] is False
    assert "sample_id" in ok["summary"]["categoricals"]
    # error path surfaces as a structured error, not an exception
    assert err["status"] == "error"
    assert err["error_code"] == "missing_input"


def test_init_fork_safety_sets_forkserver():
    """Regression guard for the cnv_score fork-deadlock fix: the MCP server must run
    tool process-pools under a fork-safe start method. Forking the long-lived,
    multi-threaded server (default 'fork') let pool workers inherit a locked mutex and
    deadlock at 0% CPU (infercnvpy ProcessPoolExecutor). _init_fork_safety() flips the
    default start method to 'forkserver' so workers fork from a clean server process."""
    import multiprocessing as mp
    import sys as _sys

    from scpilot.mcp_server import _init_fork_safety

    _init_fork_safety()
    if _sys.platform == "win32":
        return  # POSIX-only; Windows already spawns
    assert mp.get_start_method(allow_none=False) == "forkserver"


def test_numba_cache_env_set_before_forkserver_warmup(monkeypatch):
    """Regression for the numba-cache-permission fix: NUMBA_CACHE_DIR / MPLCONFIGDIR must be set
    BEFORE main() warms the forkserver daemon, or forkserver-spawned infercnvpy CNV workers inherit
    an environment with no writable numba cache (the setdefaults used to land later, inside
    build_server()->init_runtime, AFTER warmup). Capture the env AT the warmup call site."""
    import os
    import types

    import scpilot.mcp_server as mcp

    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    captured = {}

    def fake_fork_safety(lg=None):
        captured["numba"] = os.environ.get("NUMBA_CACHE_DIR")
        captured["mpl"] = os.environ.get("MPLCONFIGDIR")

    fake_server = types.SimpleNamespace(_scpilot_cleanup=lambda: None,
                                        run=lambda **kw: None)
    monkeypatch.setattr(mcp, "_init_fork_safety", fake_fork_safety)
    monkeypatch.setattr(mcp, "build_server", lambda: fake_server)
    monkeypatch.setattr(mcp, "_install_cleanup_handlers", lambda c: None)

    mcp.main()

    # env present (and pointing at a real, writable dir) at the moment the daemon was warmed
    assert captured["numba"] and os.path.isdir(captured["numba"])
    assert captured["mpl"] and os.path.isdir(captured["mpl"])
