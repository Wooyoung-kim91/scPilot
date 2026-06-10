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


async def _drive(h5ad_path):
    params = StdioServerParameters(command=sys.executable, args=["-m", "scpilot", "mcp"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            ver = _parse(await session.call_tool("scpilot_version", {}))
            ok = _parse(await session.call_tool("inspect_h5ad_tool", {"path": h5ad_path}))
            err = _parse(await session.call_tool("inspect_h5ad_tool", {"path": "/no/such.h5ad"}))
            return names, ver, ok, err


def test_mcp_stdio_tool_discovery_and_call(tmp_path):
    h5ad = tmp_path / "tiny.h5ad"
    _tiny_h5ad(h5ad)
    names, ver, ok, err = asyncio.run(_drive(str(h5ad)))

    # tool discovery
    assert {"inspect_h5ad_tool", "scpilot_version"} <= names
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
