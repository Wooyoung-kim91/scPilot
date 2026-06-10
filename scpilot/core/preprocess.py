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


def _suggest_n_pcs(variance_ratio, *, floor: float = 0.01) -> int:
    """Simple elbow: last PC whose individual variance ratio exceeds ``floor``."""
    import numpy as np
    vr = np.asarray(variance_ratio, dtype=float)
    above = np.where(vr >= floor)[0]
    return int(above[-1] + 1) if above.size else min(10, len(vr))


@register("preprocess", mutating=True, long_running=False,
          description="normalize_total → log1p → HVG(seurat_v3, batch-aware) → PCA from the counts layer; "
                      "returns variance-ratio + HVG/elbow summary for choosing n_pcs (plan B4).")
def preprocess(session, *, target_sum: float = 1e4, n_top_genes: int = 2000,
               hvg_batch_key: str | None = None, n_pcs: int = 50, seed: int = 0,
               **params) -> S.ToolResult:
    import numpy as np
    import scanpy as sc

    t0 = time.time()
    adata = session.adata
    warnings: list[str] = []

    if "counts" not in adata.layers:
        return S.error("preprocess", "invalid_state",
                       "no 'counts' layer — seurat_v3 HVG + reproducible normalize need raw counts",
                       recoverable=False)

    # --- normalize + log1p from counts (reproducible regardless of current X) ---
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    # project convention (matches scqc merged): raw counts stay in `counts`,
    # normalize_total+log1p values are stored in `scale.data` (kept for markers/annotation)
    adata.layers["scale.data"] = adata.X.copy()

    # --- HVG (seurat_v3, counts-based; batch-aware if a valid key is given) ---
    batch_key = hvg_batch_key
    if batch_key and batch_key not in adata.obs.columns:
        warnings.append(f"hvg_batch_key '{batch_key}' absent — HVG computed without batch")
        batch_key = None
    n_top = min(n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top,
                                layer="counts", batch_key=batch_key)
    n_hvg = int(adata.var["highly_variable"].sum())

    # --- PCA on HVGs (no global scale; svd centers) ---
    max_comps = max(1, min(n_pcs, adata.n_vars - 1, adata.n_obs - 1))
    sc.pp.pca(adata, n_comps=max_comps, mask_var="highly_variable", random_state=seed)
    vr = [round(float(x), 5) for x in adata.uns["pca"]["variance_ratio"]]
    cum = float(np.cumsum(vr)[-1])
    suggested = _suggest_n_pcs(vr)

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
                            params={"target_sum": target_sum, "n_top_genes": n_top,
                                    "hvg_batch_key": batch_key, "n_pcs": max_comps, "seed": seed})
    return S.success("preprocess", summary=summary, warnings=warnings, checkpoint=cp.path,
                     determinism_grade="B", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["cluster"])
