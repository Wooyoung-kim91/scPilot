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


def _model_suffix(use_rep: str) -> str:
    """Derive a per-model suffix from the embedding key (X_pca→'' baseline; X_scVI→'scvi')."""
    if use_rep in ("X_pca", "X_PCA"):
        return ""                       # baseline keeps the canonical names
    return use_rep.removeprefix("X_").lower()


@register("cluster", mutating=True,
          description="neighbors → leiden → umap on a PCA/integration embedding. resolution is HUMAN-IN-THE-LOOP "
                      "and REQUIRED (no auto-default) — the user sets it per clustering. ALL reductions kept "
                      "per-model (baseline X_umap/leiden; X_umap_<model>/leiden_<model>) — never overwritten (plan B6).")
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
    # resolution is a human decision (plan: human-in-the-loop) — never silently defaulted.
    if resolution is None:
        return S.error("cluster", "missing_input",
                       "clustering 'resolution' must be set explicitly by the user (human-in-the-loop) — "
                       "pass resolution=<float>; scpilot does not auto-choose it.",
                       recoverable=True)

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
        "resolution": resolution, "n_neighbors": n_neighbors, "n_pcs": use_pcs,
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
