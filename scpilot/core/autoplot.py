"""Per-step auto-plot hook — scpilot result-plot rule.

Every orchestrated step (MCP handler / CLI ``step``) should return a result PLOT, not
just numbers. Rather than editing each tool, this central hook inspects the tool name +
its ToolResult summary and renders the stage-appropriate figure(s) through the existing
``plots`` tool (same vendored fit_and_save engine), then hands the artifacts back so they
are attached to the result and recorded in reasoning_log.md.

Design notes:
- Lives in the ORCHESTRATION layer (not inside tools.run), so direct unit-test calls to a
  tool function are unaffected — only MCP/CLI runs get auto-plots.
- Fully defensive: any plotting failure is swallowed (a missing plot must never fail a step).
- Pure transforms that produce an embedding but no fresh 2D layout (integrate_*/train_scvi/
  compartment_subset) are intentionally skipped — the immediately-following ``cluster`` step
  renders their UMAP. This is logged so the gap is explicit, not silent.
"""

from __future__ import annotations

_UMAP_PREFERENCE = ("X_umap_scvi", "X_umap_harmony", "X_umap")
_QC_KEYS = ("n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo")


def _best_basis(obsm) -> str:
    for b in _UMAP_PREFERENCE:
        if b in obsm:
            return b
    return "X_umap"


def plan_autoplots(tool: str, summary: dict, *, obs: set, obsm: set) -> list[dict]:
    """Return a list of ``plots``-tool kwargs appropriate for the just-run ``tool``."""
    summary = summary or {}
    specs: list[dict] = []
    if tool in ("qc_metrics", "qc_filter"):
        keys = [k for k in _QC_KEYS if k in obs]
        if keys:
            specs.append({"kind": "qc_violin", "keys": keys})
    elif tool == "preprocess":
        specs += [{"kind": "pca_variance"}, {"kind": "hvg"}]
    elif tool == "cluster":
        ck = summary.get("cluster_key")
        bk = summary.get("umap_key", "X_umap")
        if ck and ck in obs and bk in obsm:
            specs.append({"kind": "umap", "color": ck, "basis": bk})
    elif tool == "apply_annotation":
        key = summary.get("label_key")
        if key and key in obs:
            specs.append({"kind": "umap", "color": key, "basis": _best_basis(obsm)})
    elif tool == "apply_fine_annotation":
        key = summary.get("fine_key", "fine_cell_type")
        if key in obs:
            specs.append({"kind": "umap", "color": key, "basis": _best_basis(obsm)})
    elif tool == "consensus_annotation":
        key = summary.get("out_key")
        if key and key in obs:
            specs.append({"kind": "umap", "color": key, "basis": _best_basis(obsm)})
    return specs


def auto_plots(session, tool: str, summary: dict) -> list:
    """Render + return the stage-appropriate plot artifacts (never raises)."""
    try:
        adata = session.adata
    except Exception:  # noqa: BLE001
        return []
    if adata is None:
        return []
    obs = set(adata.obs.columns)
    obsm = set(adata.obsm.keys())
    specs = plan_autoplots(tool, summary, obs=obs, obsm=obsm)
    if not specs:
        return []
    try:  # RISK #16: a plots-module import failure must not escape (never-raises contract)
        from scpilot.core.plots import plots as _plots  # registered fn, callable directly
    except Exception:  # noqa: BLE001
        return []
    arts: list = []
    for sp in specs:
        try:
            res = _plots(session, **sp)
            arts += list(res.artifacts or [])
        except Exception:  # noqa: BLE001 — a missing plot must never break the step
            continue
    return arts
