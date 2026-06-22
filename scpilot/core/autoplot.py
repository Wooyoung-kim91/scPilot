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

# sample/condition-like obs columns to auto-color UMAPs by (name hints + low-cardinality
# categoricals); QC numerics, doublet flags and leiden keys are excluded.
_META_NAME_HINTS = ("sample", "sample_id", "condition", "tissue", "batch", "patient", "donor",
                    "group", "treatment", "response", "status", "timepoint", "origin",
                    "disease", "subject")
_META_EXCLUDE = {"predicted_doublet", "mixed_lineage_flag", "mt", "ribo"}


def _best_basis(obsm) -> str:
    for b in _UMAP_PREFERENCE:
        if b in obsm:
            return b
    return "X_umap"


def _metadata_color_keys(adata, *, cap: int = 6, max_card: int = 30) -> list:
    """sample/condition-like obs columns worth coloring a UMAP by (enforced metadata UMAPs).

    Picks name-hinted columns first, then any low-cardinality (2..max_card) categorical/bool
    column; excludes leiden keys, QC numerics and doublet/artifact flags. Capped at ``cap``.
    """
    import pandas as pd

    obs = adata.obs

    def ok(col: str, picked: list) -> bool:
        if col in picked or col.startswith("leiden") or col in _META_EXCLUDE:
            return False
        s = obs[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return False                                   # QC metrics etc.
        try:
            return 2 <= int(s.nunique(dropna=True)) <= max_card
        except Exception:  # noqa: BLE001
            return False

    picked: list = []
    for h in _META_NAME_HINTS:                              # name-hinted, in priority order
        for col in obs.columns:
            if col.lower() == h and ok(col, picked):
                picked.append(col)
    for col in obs.columns:                                 # then any low-card categorical
        if len(picked) >= cap:
            break
        if ok(col, picked):
            picked.append(col)
    return picked[:cap]


def plan_autoplots(tool: str, summary: dict, *, obs: set, obsm: set,
                   meta_keys: list | None = None) -> list[dict]:
    """Return a list of ``plots``-tool kwargs appropriate for the just-run ``tool``."""
    summary = summary or {}
    specs: list[dict] = []
    has_scatter = {"total_counts", "n_genes_by_counts"} <= obs
    if tool == "qc_metrics":
        specs.append({"kind": "qc_violin", "tag": "pre"})       # before-filter snapshot
        if has_scatter:
            specs.append({"kind": "scatter", "tag": "pre"})
    elif tool == "qc_filter":
        specs.append({"kind": "qc_violin", "tag": "post"})      # after-filter (distinct file)
        if has_scatter:
            specs.append({"kind": "scatter", "tag": "post"})
        # the parameter-justification figure: chosen cutoffs overlaid on the distributions
        specs.append({"kind": "qc_thresholds", "tag": "post",
                      "cutoffs": summary.get("cutoffs", {})})
    elif tool == "preprocess":
        specs += [{"kind": "pca_variance"}, {"kind": "hvg"}]
    elif tool == "cluster_sweep":
        sw = summary.get("sweep")
        if sw:
            specs.append({"kind": "resolution_sweep", "sweep": sw,
                          "suggested": summary.get("suggested_resolution")})
    elif tool == "cluster":
        ck = summary.get("cluster_key")
        bk = summary.get("umap_key", "X_umap")
        if ck and ck in obs and bk in obsm:
            specs.append({"kind": "umap", "color": ck, "basis": bk})        # leiden
            for mk in (meta_keys or []):                                    # sample/condition/tissue/…
                if mk in obs:
                    specs.append({"kind": "umap", "color": mk, "basis": bk})
    elif tool == "apply_annotation":
        key = summary.get("label_key")
        if key and key in obs:
            specs.append({"kind": "umap", "color": key, "basis": _best_basis(obsm)})
            # broad annotation-evidence dotplot (cell types × their marker set). Prefer the LLM's
            # recorded >=3-gene marker_sets; else derive cell-type panels from the leiden DE via the
            # cluster->label map. Rows are family-contiguous (staircase) by default.
            ms = summary.get("marker_sets") or {}
            if ms:
                specs.append({"kind": "dotplot", "groupby": key, "marker_groups": ms})
            elif summary.get("groupby") and summary.get("labels"):
                specs.append({"kind": "dotplot", "groupby": key,
                              "cluster_key": summary["groupby"], "label_map": summary["labels"]})
    elif tool == "apply_fine_annotation":
        key = summary.get("fine_key", "fine_cell_type")
        if key in obs:
            specs.append({"kind": "umap", "color": key, "basis": _best_basis(obsm)})
        # dotplot of per-subcluster marker evidence (DE was just computed on the subcluster
        # key by `markers`, so derive from it); gives every compartment a marker dotplot too.
        gb = summary.get("groupby")
        if gb and gb in obs:
            specs.append({"kind": "dotplot", "groupby": gb})
    elif tool == "consensus_annotation":
        key = summary.get("out_key")
        if key and key in obs:
            specs.append({"kind": "umap", "color": key, "basis": _best_basis(obsm)})
    elif tool == "merge_fine_annotations":
        key = summary.get("fine_key", "fine_cell_type")
        if key in obs:
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
    meta_keys = _metadata_color_keys(adata) if tool == "cluster" else None
    specs = plan_autoplots(tool, summary, obs=obs, obsm=obsm, meta_keys=meta_keys)
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
