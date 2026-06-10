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


@register("cluster", mutating=True,
          description="neighbors → leiden → umap on a PCA/integration embedding; returns cluster sizes (plan B6).")
def cluster(session, *, use_rep: str = "X_pca", n_neighbors: int = 15, n_pcs: int | None = None,
            resolution: float = 0.5, seed: int = 0, key_added: str = "leiden",
            **params) -> S.ToolResult:
    import scanpy as sc

    t0 = time.time()
    adata = session.adata
    if use_rep not in adata.obsm:
        return S.error("cluster", "invalid_state",
                       f"embedding '{use_rep}' absent in obsm{sorted(adata.obsm)} — run preprocess/integrate first",
                       recoverable=True, suggested_next_tools=["preprocess"])

    rep_dim = adata.obsm[use_rep].shape[1]
    use_pcs = min(n_pcs, rep_dim) if n_pcs else rep_dim
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=use_pcs, use_rep=use_rep, random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, key_added=key_added, flavor="igraph",
                 n_iterations=2, directed=False, random_state=seed)
    sc.tl.umap(adata, random_state=seed)

    sizes = adata.obs[key_added].value_counts().sort_index()
    summary = {
        "n_cells": int(adata.n_obs),
        "cluster_key": key_added,
        "n_clusters": int(sizes.shape[0]),
        "resolution": resolution,
        "use_rep": use_rep, "n_neighbors": n_neighbors, "n_pcs": use_pcs,
        "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
        "smallest_cluster": int(sizes.min()), "largest_cluster": int(sizes.max()),
        "has_umap": "X_umap" in adata.obsm,
    }
    cp = session.checkpoint("cluster", x_state=session.manifest.x_state,
                            params={"use_rep": use_rep, "n_neighbors": n_neighbors,
                                    "n_pcs": use_pcs, "resolution": resolution, "seed": seed})
    return S.success("cluster", summary=summary, checkpoint=cp.path, determinism_grade="B",
                     duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["markers"])
