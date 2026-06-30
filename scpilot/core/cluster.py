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
    from .. import recipes

    t0 = time.time()
    adata = session.adata
    if use_rep not in adata.obsm:
        return S.error("cluster", "invalid_state",
                       f"embedding '{use_rep}' absent in obsm{sorted(adata.obsm)} — run preprocess/integrate first",
                       recoverable=True, suggested_next_tools=["preprocess"])
    # resolution defaults to 0.25 at every stage; an explicit value (per call / per embedding) wins.
    resolution_defaulted = resolution is None

    # core neighbors → leiden → umap lives in scpilot.recipes (scpilot-free) so the generated
    # standalone tutorial script inlines the SAME source — logic-identical by construction.
    adata, info = recipes.cluster(
        adata, use_rep=use_rep, n_neighbors=n_neighbors, n_pcs=n_pcs, resolution=resolution,
        seed=seed, key_added=key_added, key_suffix=key_suffix)
    leiden_key = info["leiden_key"]
    umap_key = info["umap_key"]
    nkey = info["neighbors_key"]
    use_pcs = info["use_pcs"]
    resolution = info["resolution"]
    sizes = adata.obs[leiden_key].value_counts().sort_index()
    summary = {
        "n_cells": int(adata.n_obs),
        "cluster_key": leiden_key, "umap_key": umap_key, "neighbors_key": nkey,
        "model_suffix": info["model_suffix"], "use_rep": use_rep,
        "n_clusters": info["n_clusters"],
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

    from .. import recipes

    t0 = time.time()
    adata = session.adata
    if use_rep not in adata.obsm:
        return S.error("cluster_sweep", "invalid_state",
                       f"embedding '{use_rep}' absent in obsm{sorted(adata.obsm)} — run preprocess/integrate first",
                       recoverable=True, suggested_next_tools=["preprocess"])

    # the (non-mutating) resolution sweep + knee pick live in scpilot.recipes (scpilot-free) so the
    # generated standalone tutorial script inlines the SAME source — logic-identical by construction.
    sweep = recipes.cluster_sweep(adata, use_rep=use_rep, res_min=res_min, res_max=res_max,
                                  res_step=res_step, n_neighbors=n_neighbors, n_pcs=n_pcs, seed=seed)
    suggested, rationale = recipes.suggest_resolution(sweep, jump_ratio=jump_ratio)
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
