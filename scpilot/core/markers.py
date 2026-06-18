"""Marker genes: rank_genes_groups per cluster — scpilot plan B7.

Runs Wilcoxon rank-sum on the log-normalized layer per cluster and returns a
capped per-cluster marker table (top positive markers) inline, with the full
ranking written to a CSV artifact. Also reports each cluster's size and how many
samples contribute (batch-awareness signal for downstream annotation/review).
"""

from __future__ import annotations

import time

from scpilot import schemas as S
from scpilot.tools import register


@register("markers", mutating=True,
          description="rank_genes_groups (Wilcoxon) per cluster → top markers table + per-cluster size/sample "
                      "spread; full ranking as CSV artifact (plan B7).")
def markers(session, *, groupby: str = "leiden", n_genes: int = 25, layer: str | None = "scale.data",
            sample_key: str = "sample_id", **params) -> S.ToolResult:
    import numpy as np
    import pandas as pd
    import scanpy as sc

    t0 = time.time()
    adata = session.adata
    if groupby not in adata.obs.columns:
        return S.error("markers", "invalid_state",
                       f"grouping '{groupby}' absent — run cluster first", recoverable=True,
                       suggested_next_tools=["cluster"])
    use_layer = layer if (layer and layer in adata.layers) else None
    warnings = [] if use_layer else [f"layer '{layer}' absent — ranking on X"]

    # rank ALL genes so the CSV artifact is a genuine full ranking (Codex review 1.5);
    # the inline preview is capped separately by n_genes.
    # pts=True → per-cluster expressed fractions (pct_in / pct_out), consumed by
    # annotation_review as DE evidence for the marker-DB-free LLM annotation.
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon", layer=use_layer,
                            use_raw=False, n_genes=adata.n_vars, pts=True)
    rg = adata.uns["rank_genes_groups"]
    groups = list(rg["names"].dtype.names)
    preview_n = min(n_genes, adata.n_vars)

    # full long-form table + capped per-cluster top markers
    rows, top_by_cluster = [], []
    sizes = adata.obs[groupby].value_counts()
    for g in groups:
        names = rg["names"][g]
        lfc = rg["logfoldchanges"][g]
        pvals = rg["pvals_adj"][g]
        for rank, (gene, l, p) in enumerate(zip(names, lfc, pvals)):
            rows.append({"cluster": g, "rank": rank, "gene": str(gene),
                         "logfoldchange": float(l), "pval_adj": float(p)})
        # sample spread for this cluster
        mask = (adata.obs[groupby] == g).values
        n_samp = int(adata.obs[sample_key][mask].nunique()) if sample_key in adata.obs else None
        top_by_cluster.append({"cluster": g, "n_cells": int(sizes.get(g, 0)),
                               "n_samples": n_samp, "top_genes": [str(x) for x in names[:10]]})

    full_df = pd.DataFrame(rows)
    # key the CSV by the clustering it ranks so per-reduction DE (leiden / leiden_harmony /
    # leiden_scvi) ALL persist — a fixed name would overwrite earlier reductions' DE.
    import re as _re
    safe_gb = _re.sub(r"[^0-9A-Za-z]+", "_", str(groupby)).strip("_") or "groupby"
    csv_path = session.artifacts_dir / f"markers_{safe_gb}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(csv_path, index=False)

    top_df = pd.DataFrame(top_by_cluster)
    summary = {
        "groupby": groupby, "n_clusters": len(groups), "method": "wilcoxon",
        "preview_genes_per_cluster": preview_n,
        "csv_is_full_ranking": True, "n_genes_ranked": int(adata.n_vars),
        "layer": use_layer,
        "single_sample_dominated_clusters": [
            r["cluster"] for r in top_by_cluster if r["n_samples"] == 1],
    }
    tables = {"top_markers": S.table_preview(top_df, max_rows=50)}
    artifacts = [S.artifact_csv(str(csv_path), n_rows=len(full_df), n_cols=full_df.shape[1],
                                description="full rank_genes_groups ranking")]

    cp = session.checkpoint("markers", x_state=session.manifest.x_state,
                            params={"groupby": groupby, "n_genes": n_genes, "layer": use_layer,
                                    "sample_key": sample_key})
    return S.success("markers", summary=summary, tables=tables, artifacts=artifacts,
                     warnings=warnings, checkpoint=cp.path, determinism_grade="A",
                     duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["annotation_review"])
