"""Clustering + embedding: neighbors → leiden → umap — scpilot plan B6.

Operates on the PCA embedding (or a chosen ``use_rep`` like an integration
embedding, so the same tool serves baseline and post-integration clustering).
Returns cluster counts/sizes as structural invariants (regression checks compare
cluster count within tolerance — determinism grade B for leiden/umap).
"""

from __future__ import annotations

import time

from scpilot import schemas as S
from scpilot.tools import register

# Default leiden resolution applied at EVERY clustering stage (baseline / harmony / scVI /
# compartment subsets) when the caller does not pass one. The user may still override per call
# (or per embedding via `scpilot run --resolution`). Single source of truth for the default.
DEFAULT_RESOLUTION = 0.25


def _model_suffix(use_rep: str) -> str:
    """Derive a per-model suffix from the embedding key (X_pca→'' baseline; X_scVI→'scvi')."""
    if use_rep in ("X_pca", "X_PCA"):
        return ""                       # baseline keeps the canonical names
    return use_rep.removeprefix("X_").lower()


@register("cluster", mutating=True,
          description="neighbors → leiden → umap on a PCA/integration embedding. resolution DEFAULTS to 0.25 at every "
                      "clustering stage; the user may override per call (or per embedding via `scpilot run --resolution`). "
                      "ALL reductions kept per-model (baseline X_umap/leiden; X_umap_<model>/leiden_<model>) — never overwritten (plan B6).")
def cluster(session, *, use_rep: str = "X_pca", n_neighbors: int = 15, n_pcs: int | None = None,
            resolution: float | None = None, seed: int = 0, key_added: str | None = None,
            key_suffix: str | None = None, **params) -> S.ToolResult:
    import scanpy as sc

    t0 = time.time()
    adata = session.adata
    if use_rep not in adata.obsm:
        return S.error("cluster", "invalid_state",
                       f"embedding '{use_rep}' absent in obsm{sorted(adata.obsm)} — run preprocess/integrate first",
                       recoverable=True, suggested_next_tools=["preprocess"])
    # resolution defaults to 0.25 at every stage; an explicit value (per call / per embedding) wins.
    resolution_defaulted = resolution is None
    if resolution_defaulted:
        resolution = DEFAULT_RESOLUTION

    # per-model namespacing so integration-before/after reductions all coexist
    suf = key_suffix if key_suffix is not None else _model_suffix(use_rep)
    leiden_key = key_added or (f"leiden_{suf}" if suf else "leiden")
    umap_key = f"X_umap_{suf}" if suf else "X_umap"
    nkey = f"neighbors_{suf}" if suf else None    # None → scanpy default graph keys (baseline)

    rep_dim = adata.obsm[use_rep].shape[1]
    use_pcs = min(n_pcs, rep_dim) if n_pcs else rep_dim
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=use_pcs, use_rep=use_rep,
                    random_state=seed, key_added=nkey)
    sc.tl.leiden(adata, resolution=resolution, key_added=leiden_key, flavor="igraph",
                 n_iterations=2, directed=False, random_state=seed, neighbors_key=nkey)
    # umap always writes obsm["X_umap"]; preserve any baseline by moving the result
    prev_umap = adata.obsm["X_umap"].copy() if "X_umap" in adata.obsm else None
    sc.tl.umap(adata, random_state=seed, neighbors_key=nkey)
    if umap_key != "X_umap":
        adata.obsm[umap_key] = adata.obsm["X_umap"]
        if prev_umap is not None:
            adata.obsm["X_umap"] = prev_umap        # restore baseline
        else:
            del adata.obsm["X_umap"]

    sizes = adata.obs[leiden_key].value_counts().sort_index()
    summary = {
        "n_cells": int(adata.n_obs),
        "cluster_key": leiden_key, "umap_key": umap_key, "neighbors_key": nkey,
        "model_suffix": suf or "baseline", "use_rep": use_rep,
        "n_clusters": int(sizes.shape[0]),
        "resolution": resolution, "resolution_defaulted": resolution_defaulted,
        "n_neighbors": n_neighbors, "n_pcs": use_pcs,
        "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
        "smallest_cluster": int(sizes.min()), "largest_cluster": int(sizes.max()),
        "embeddings_present": sorted(adata.obsm.keys()),
    }
    cp = session.checkpoint("cluster", x_state=session.manifest.x_state,
                            params={"use_rep": use_rep, "n_neighbors": n_neighbors,
                                    "n_pcs": use_pcs, "resolution": resolution, "seed": seed,
                                    "key_added": leiden_key, "umap_key": umap_key, "neighbors_key": nkey})
    return S.success("cluster", summary=summary, checkpoint=cp.path, determinism_grade="B",
                     duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["markers"])


def _suggest_resolution(sweep: list, *, jump_ratio: float) -> tuple[float, str]:
    """Pick the resolution JUST BEFORE n_clusters jumps by ≥ ``jump_ratio`` (the knee —
    "급증 직전"). No abrupt jump → the lowest (conservative) resolution. ``sweep`` is an
    ordered list of (resolution, n_clusters)."""
    for i in range(len(sweep) - 1):
        r_i, n_i = sweep[i]
        r_next, n_next = sweep[i + 1]
        if n_next >= max(n_i, 1) * jump_ratio:
            return r_i, (f"n_clusters jumps {n_i}→{n_next} at res {r_next} (≥{jump_ratio}×); "
                         f"chose res {r_i} just before the jump")
    return sweep[0][0], (f"no abrupt jump (≥{jump_ratio}×) over res {sweep[0][0]}–{sweep[-1][0]}; "
                         f"chose conservative lowest res {sweep[0][0]}")


@register("cluster_sweep", mutating=False,
          description="Sweep leiden resolution (default 0.1–0.5 step 0.1) on an embedding and return the "
                      "n_clusters-vs-resolution curve + a suggested resolution at the KNEE (the value JUST "
                      "BEFORE n_clusters jumps by ≥jump_ratio). Evidence for choosing resolution — the LLM "
                      "judges/overrides, then calls cluster(use_rep, resolution=chosen). Non-mutating; the "
                      "auto-plot is the resolution_sweep justification figure (plan: dynamic resolution).")
def cluster_sweep(session, *, use_rep: str = "X_pca", res_min: float = 0.1, res_max: float = 0.5,
                  res_step: float = 0.1, n_neighbors: int = 15, n_pcs: int | None = None,
                  jump_ratio: float = 1.5, seed: int = 0, **params) -> S.ToolResult:
    import pandas as pd
    import scanpy as sc

    t0 = time.time()
    adata = session.adata
    if use_rep not in adata.obsm:
        return S.error("cluster_sweep", "invalid_state",
                       f"embedding '{use_rep}' absent in obsm{sorted(adata.obsm)} — run preprocess/integrate first",
                       recoverable=True, suggested_next_tools=["preprocess"])

    rep_dim = adata.obsm[use_rep].shape[1]
    use_pcs = min(n_pcs, rep_dim) if n_pcs else rep_dim
    n_steps = int(round((res_max - res_min) / res_step)) + 1
    grid = [round(res_min + i * res_step, 4) for i in range(max(1, n_steps))]

    nkey, lkey = "_sweep_nbr", "_sweep_leiden"
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=use_pcs, use_rep=use_rep,
                    random_state=seed, key_added=nkey)
    sweep: list = []
    try:
        for r in grid:
            sc.tl.leiden(adata, resolution=r, key_added=lkey, flavor="igraph",
                         n_iterations=2, directed=False, random_state=seed, neighbors_key=nkey)
            sweep.append((r, int(adata.obs[lkey].nunique())))
    finally:
        # drop the throwaway sweep keys so the NEXT real `cluster` checkpoint isn't polluted
        adata.obs.drop(columns=[lkey], errors="ignore", inplace=True)
        adata.uns.pop(nkey, None)
        for k in (f"{nkey}_distances", f"{nkey}_connectivities"):
            if k in adata.obsp:
                del adata.obsp[k]

    suggested, rationale = _suggest_resolution(sweep, jump_ratio=jump_ratio)
    df = pd.DataFrame([{"resolution": r, "n_clusters": n} for r, n in sweep])
    summary = {
        "use_rep": use_rep,
        "sweep": [{"resolution": r, "n_clusters": n} for r, n in sweep],
        "suggested_resolution": suggested,
        "jump_ratio": jump_ratio,
        "rationale": rationale,
        "n_steps": len(sweep),
    }
    return S.success("cluster_sweep", summary=summary, tables={"sweep": S.table_preview(df)},
                     determinism_grade="B", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["cluster"])
