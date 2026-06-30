"""Preprocessing: normalize → log1p → HVG (seurat_v3) → PCA — scpilot plan B4.

Starts from the immutable ``counts`` layer (so the step is reproducible regardless
of what X currently holds), writes log-normalized values to X + a ``scale.data``
layer (project convention; kept for marker/annotation use), selects HVGs with
``seurat_v3`` (counts-based,
needs scikit-misc; batch-aware via ``hvg_batch_key``), and runs PCA on the HVGs.

No global ``sc.pp.scale`` — densifying 40k genes × 180k cells is infeasible and
PCA centers internally; the merged already carries a ``scale.data`` layer if needed.
Returns the PCA variance-ratio + an elbow suggestion so the LLM can pick n_pcs.
"""

from __future__ import annotations

import time

from scpilot import schemas as S
from scpilot.tools import register


@register("preprocess", mutating=True, long_running=False,
          description="normalize_total → log1p → HVG(seurat_v3, batch-aware) → PCA from the counts layer; "
                      "returns variance-ratio + HVG/elbow summary for choosing n_pcs (plan B4).")
def preprocess(session, *, target_sum: float = 1e4, n_top_genes: int = 2000,
               hvg_batch_key: str | None = None, min_cells_per_batch: int = 1000,
               n_pcs: int = 50, seed: int = 0, **params) -> S.ToolResult:
    import numpy as np

    from .. import recipes

    t0 = time.time()
    adata = session.adata

    if "counts" not in adata.layers:
        return S.error("preprocess", "invalid_state",
                       "no 'counts' layer — seurat_v3 HVG + reproducible normalize need raw counts",
                       recoverable=False)

    # core transform (normalize → log1p → HVG → PCA) lives in scpilot.recipes (scpilot-free) so the
    # generated standalone tutorial script inlines the SAME source — logic-identical by construction.
    adata, info = recipes.preprocess(
        adata, target_sum=target_sum, n_top_genes=n_top_genes, hvg_batch_key=hvg_batch_key,
        min_cells_per_batch=min_cells_per_batch, n_pcs=n_pcs, seed=seed)
    warnings = info["warnings"]
    batch_key = info["batch_key"]
    n_hvg = info["n_hvg"]
    max_comps = info["n_pcs"]
    vr = info["variance_ratio"]
    cum = float(np.cumsum(vr)[-1]) if vr else 0.0
    suggested = info["suggested_n_pcs_elbow"]

    session.manifest.x_state = "log1p"
    summary = {
        "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
        "n_hvg": n_hvg, "hvg_flavor": "seurat_v3", "hvg_batch_key": batch_key,
        "n_pcs": max_comps,
        "variance_ratio": vr[:50],
        "cumulative_variance_at_n_pcs": round(cum, 4),
        "suggested_n_pcs_elbow": suggested,
        "x_state": "log1p", "normalized_layer": "scale.data",
    }
    cp = session.checkpoint("preprocess", x_state="log1p",
                            params={"target_sum": target_sum,
                                    "n_top_genes": min(n_top_genes, adata.n_vars),
                                    "hvg_batch_key": batch_key, "n_pcs": max_comps, "seed": seed})
    return S.success("preprocess", summary=summary, warnings=warnings, checkpoint=cp.path,
                     determinism_grade="B", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["cluster"])
