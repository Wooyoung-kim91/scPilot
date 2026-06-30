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
    assert "scpilot_workflow" in prompt_names                      # MCP prompt channel
    assert "scpilot://workflow" in res_uris                        # MCP resource channel
    assert "scpilot_guidance" in tool_names                        # tool channel (tool-only clients)
    wf = guidance["workflow"]
    assert "CANONICAL FLOW" in wf and "detect_state" in wf         # full pipeline present
    assert "malignant" in wf                                        # incl. annotation/CNV hard rules
    assert "Tier-4 consistency" in wf                              # incl. the independent audit/critique


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
