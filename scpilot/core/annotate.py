"""Annotation — Tier 1 broad (B8); Tier 3 fine lands later (B13).

Tier 1 (``annotate_broad``) assigns ``obs['major_cell_type']`` — the broad
compartment that becomes the scib benchmark ``label_key``. Per de-risk ① (PoC
2026-06-10): unintegrated leiden clusters are heavily GSE-fragmented (27/35
single-GSE), so deriving the label from clusters alone is CIRCULAR. The anchor is
therefore **per-cell marker scores** (batch-agnostic biology of EPCAM/CD3D/...),
aggregated to clusters by majority vote with a confidence = agreement fraction.
Single-GSE-dominated clusters are flagged ``circular_risk`` so the benchmark step
can exclude/penalize label-based metrics that would reward over-correction.

Broad panel mirrors ``cancer_scrnaseq_annotation_strategy.md`` Tier 1 (single source).
"""

from __future__ import annotations

import time

from scpilot import schemas as S
from scpilot.tools import register

# Tier 1 broad lineage panels (from cancer_scrnaseq_annotation_strategy.md).
BROAD_MARKERS = {
    "Epithelial":  ["EPCAM", "KRT8", "KRT18", "KRT19"],
    "T_NK":        ["CD3D", "CD3E", "TRAC", "NKG7", "GNLY", "KLRD1"],
    "B_Plasma":    ["MS4A1", "CD79A", "CD79B", "MZB1", "JCHAIN", "XBP1"],
    "Myeloid":     ["LYZ", "CD68", "C1QA", "C1QB", "C1QC", "FCGR3A", "S100A8"],
    "Stromal":     ["COL1A1", "COL1A2", "DCN", "LUM", "ACTA2", "PDGFRB"],
    "Endothelial": ["PECAM1", "VWF", "KDR", "CLDN5"],
    "Mast":        ["TPSAB1", "TPSB2", "CPA3", "KIT"],
}
UNS_ANNO = "scpilot_annotation"


@register("annotate_broad", mutating=True,
          description="Tier 1 broad cell type → obs['major_cell_type'] from per-cell marker scores "
                      "(batch-agnostic anchor) aggregated to clusters; flags circular-risk / marker-conflict (plan B8).")
def annotate_broad(session, *, groupby: str = "leiden", batch_key: str = "GSE",
                   layer: str | None = "lognorm", min_confidence: float = 0.5,
                   conflict_margin: float = 0.1, circular_frac: float = 0.8,
                   **params) -> S.ToolResult:
    import numpy as np
    import pandas as pd
    import scanpy as sc

    t0 = time.time()
    adata = session.adata
    warnings: list[str] = []

    # --- per-cell marker scores (batch-agnostic anchor) ---
    use_layer = layer if (layer and layer in adata.layers) else None
    if use_layer is None and layer:
        warnings.append(f"layer '{layer}' absent — scoring on X")
    present = {k: [g for g in v if g in adata.var_names] for k, v in BROAD_MARKERS.items()}
    missing = [k for k, v in present.items() if not v]
    lineages = [k for k in BROAD_MARKERS if present[k]]
    if len(lineages) < 2:
        return S.error("annotate_broad", "data_gate_failed",
                       f"too few broad-marker genes present ({present}) — wrong var_names?",
                       recoverable=False)
    if missing:
        warnings.append(f"no markers for lineages {missing} — excluded")

    score_cols = []
    for k in lineages:
        col = f"_sc_{k}"
        sc.tl.score_genes(adata, present[k], score_name=col, layer=use_layer, use_raw=False)
        score_cols.append(col)
    scores = adata.obs[score_cols].to_numpy()
    order = np.argsort(-scores, axis=1)
    cell_lineage = np.array(lineages)[order[:, 0]]
    # margin between top-1 and top-2 lineage score (per cell) → conflict signal
    top1 = np.take_along_axis(scores, order[:, :1], axis=1)[:, 0]
    top2 = np.take_along_axis(scores, order[:, 1:2], axis=1)[:, 0]
    adata.obs["_cell_lineage"] = cell_lineage

    has_groups = groupby in adata.obs.columns
    has_batch = batch_key in adata.obs.columns
    tier1_evidence = {}

    if has_groups:
        # --- aggregate to clusters: majority vote + confidence (agreement frac) ---
        cluster_label, cluster_conf = {}, {}
        conflict_clusters, circular_clusters = [], []
        for cl, sub in adata.obs.groupby(groupby, observed=True):
            vc = sub["_cell_lineage"].value_counts(normalize=True)
            top_lab, conf = vc.index[0], float(vc.iloc[0])
            label = top_lab if conf >= min_confidence else "Mixed-Artifact"
            cluster_label[cl] = label
            cluster_conf[cl] = conf
            # marker conflict: 2nd lineage also substantial
            second = float(vc.iloc[1]) if len(vc) > 1 else 0.0
            is_conflict = (conf - second) < conflict_margin
            # circular risk: cluster dominated by one batch
            bz = None
            if has_batch:
                bz = float(sub[batch_key].astype(str).value_counts(normalize=True).iloc[0])
                if bz >= circular_frac:
                    circular_clusters.append(str(cl))
            if is_conflict:
                conflict_clusters.append(str(cl))
            tier1_evidence[str(cl)] = {
                "label": label, "confidence": round(conf, 3),
                "second_fraction": round(second, 3), "marker_conflict": bool(is_conflict),
                "batch_dominance": round(bz, 3) if bz is not None else None,
                "circular_risk": bool(bz is not None and bz >= circular_frac),
                "n_cells": int(len(sub)),
            }
        adata.obs["major_cell_type"] = adata.obs[groupby].map(cluster_label).astype(str)
        adata.obs["major_confidence"] = adata.obs[groupby].map(cluster_conf).astype(float)
        label_basis = f"cluster-majority over '{groupby}'"
    else:
        # no clustering → per-cell marker label directly (less robust)
        warnings.append(f"groupby '{groupby}' absent — using per-cell marker labels (no cluster smoothing)")
        margin = (top1 - top2)
        labels = np.where(margin >= conflict_margin, cell_lineage, "Mixed-Artifact")
        adata.obs["major_cell_type"] = labels
        adata.obs["major_confidence"] = np.clip(margin, 0, None)
        conflict_clusters, circular_clusters = [], []
        label_basis = "per-cell marker score"

    # cleanup scratch score columns
    adata.obs.drop(columns=score_cols + ["_cell_lineage"], inplace=True, errors="ignore")

    # evidence into uns (annotation tree)
    adata.uns.setdefault(UNS_ANNO, {})
    adata.uns[UNS_ANNO]["tier1"] = {
        "label_basis": label_basis, "anchor": "per_cell_marker_score",
        "groupby": groupby if has_groups else None, "batch_key": batch_key if has_batch else None,
        "clusters": tier1_evidence,
    }

    dist = adata.obs["major_cell_type"].value_counts().to_dict()
    summary = {
        "label_key": "major_cell_type",
        "label_basis": label_basis,
        "lineages_scored": lineages,
        "label_distribution": {str(k): int(v) for k, v in dist.items()},
        "n_clusters": int(adata.obs[groupby].nunique()) if has_groups else None,
        "mean_confidence": round(float(adata.obs["major_confidence"].mean()), 3),
        "marker_conflict_clusters": conflict_clusters,
        "circular_risk_clusters": circular_clusters,
        "circular_risk_note": ("label-based scib metrics on circular-risk clusters can reward "
                               "batch over-correction — flag/exclude in benchmark (de-risk ①)"),
    }
    if circular_clusters:
        warnings.append(f"{len(circular_clusters)} cluster(s) single-batch dominated → circular-risk for scib label_key")

    cp = session.checkpoint("annotate_broad", x_state=session.manifest.x_state,
                            params={"groupby": groupby, "batch_key": batch_key, "layer": use_layer,
                                    "min_confidence": min_confidence, "conflict_margin": conflict_margin,
                                    "circular_frac": circular_frac})
    return S.success("annotate_broad", summary=summary, warnings=warnings, checkpoint=cp.path,
                     determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["integrate", "benchmark"])
