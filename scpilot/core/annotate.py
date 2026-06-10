"""Annotation — Tier 1 broad (B8); Tier 3 fine lands later (B13).

Tier 1 (``annotate_broad``) assigns ``obs['major_cell_type']`` — the broad
compartment that becomes the scib benchmark ``label_key``.

LOGIC (user-confirmed 2026-06-10):
1. **leiden-cluster DE** is the basis: ``rank_genes_groups`` (Wilcoxon, pts=True).
2. A gene is a **marker** for a cluster iff ``pct_in_group >= min_pct(0.25)`` AND
   ``logfoldchange >= min_lfc(1.0)``.
3. Annotate each cluster by the **combination** of its markers vs the broad
   cell-type panels (``BROAD_MARKERS``, from cancer_scrnaseq_annotation_strategy.md).
4. A cell-type call requires **>= min_markers(3)** of that type's panel among the
   cluster's significant markers (else 'Unknown' = insufficient evidence).
5. **Sample provenance is considered**: per-cluster sample/condition composition;
   single-sample / single-batch dominated clusters are flagged (a marker match
   from one sample is weaker evidence; also the de-risk ① circular-label signal).
6. Result layout (separate ``plots`` calls): UMAP(color=major_cell_type) + dotplot
   (``sc.pl.dotplot`` with the marker panels as a dict → cell-type brackets/labels
   above the x-axis).

Evidence (matched markers, candidates, provenance, flags) → ``.uns['scpilot_annotation']['tier1']``.
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
          description="Tier 1 broad cell type → obs['major_cell_type'] via leiden-cluster DE markers "
                      "(pct>=0.25 & LFC>=1) matched to cell-type panels (>=3 markers), sample-provenance aware (plan B8).")
def annotate_broad(session, *, groupby: str = "leiden", min_pct: float = 0.25, min_lfc: float = 1.0,
                   min_markers: int = 3, sample_key: str = "sample_id", batch_key: str = "GSE",
                   layer: str | None = "scale.data", single_source_frac: float = 0.8,
                   **params) -> S.ToolResult:
    import numpy as np
    import scanpy as sc

    t0 = time.time()
    adata = session.adata
    warnings: list[str] = []
    if groupby not in adata.obs.columns:
        return S.error("annotate_broad", "invalid_state",
                       f"clustering '{groupby}' absent — run cluster first", recoverable=True,
                       suggested_next_tools=["cluster"])

    # present panels (genes actually in the data)
    panels = {ct: [g for g in gs if g in adata.var_names] for ct, gs in BROAD_MARKERS.items()}
    panels = {ct: gs for ct, gs in panels.items() if gs}
    if not panels:
        return S.error("annotate_broad", "data_gate_failed",
                       "no broad-marker genes present in var_names", recoverable=False)

    # 1) leiden-cluster DE (Wilcoxon) with per-group expressed fraction (pts)
    use_layer = layer if (layer and layer in adata.layers) else None
    if use_layer is None and layer:
        warnings.append(f"layer '{layer}' absent — DE on X")
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon", pts=True,
                            layer=use_layer, use_raw=False)
    de = sc.get.rank_genes_groups_df(adata, group=None)   # group,names,logfoldchanges,pct_nz_group,...

    # 2) significant markers per cluster: pct>=min_pct AND lfc>=min_lfc
    sig = de[(de["pct_nz_group"] >= min_pct) & (de["logfoldchanges"] >= min_lfc)]
    sig_by_cluster = {str(g): set(sub["names"]) for g, sub in sig.groupby("group", observed=True)}

    obs_g = adata.obs[groupby].astype(str)
    has_sample = sample_key in adata.obs.columns
    has_batch = batch_key in adata.obs.columns
    cluster_label, cluster_conf, evidence = {}, {}, {}
    conflict_clusters, single_source_clusters, unknown_clusters = [], [], []

    for cl in obs_g.cat.categories if hasattr(obs_g, "cat") else sorted(set(obs_g)):
        cl = str(cl)
        sigset = sig_by_cluster.get(cl, set())
        # 3-4) match marker combination to each panel; require >= min_markers
        scored = []
        for ct, gs in panels.items():
            matched = [g for g in gs if g in sigset]
            if len(matched) >= min_markers:
                scored.append((ct, matched))
        scored.sort(key=lambda x: len(x[1]), reverse=True)

        mask = (obs_g == cl).values
        n_cells = int(mask.sum())
        # 5) sample provenance
        prov = {"n_cells": n_cells}
        if has_sample:
            sv = adata.obs[sample_key].astype(str)[mask].value_counts(normalize=True)
            prov["n_samples"] = int(adata.obs[sample_key].astype(str)[mask].nunique())
            prov["top_sample_frac"] = round(float(sv.iloc[0]), 3)
            if sv.iloc[0] >= single_source_frac:
                single_source_clusters.append(cl)
        if has_batch:
            bv = adata.obs[batch_key].astype(str)[mask].value_counts(normalize=True)
            prov["top_batch_frac"] = round(float(bv.iloc[0]), 3)

        if not scored:
            label, conf, matched = "Unknown", 0.0, []
            unknown_clusters.append(cl)
        else:
            ct, matched = scored[0]
            label = ct
            conf = round(len(matched) / len(panels[ct]), 3)   # marker completeness of the call
            if len(scored) > 1 and len(scored[1][1]) >= min_markers:
                conflict_clusters.append(cl)
        cluster_label[cl] = label
        cluster_conf[cl] = conf
        evidence[cl] = {
            "label": label, "confidence": conf, "n_markers_matched": len(matched),
            "matched_markers": matched,
            "candidates": {ct: m for ct, m in scored},
            "single_source": cl in single_source_clusters,
            "marker_conflict": cl in conflict_clusters,
            **prov,
        }

    adata.obs["major_cell_type"] = obs_g.map(cluster_label).astype(str)
    adata.obs["major_confidence"] = obs_g.map(cluster_conf).astype(float)
    adata.uns.setdefault(UNS_ANNO, {})
    adata.uns[UNS_ANNO]["tier1"] = {
        "method": "leiden_DE_marker_combination",
        "groupby": groupby, "min_pct": min_pct, "min_lfc": min_lfc, "min_markers": min_markers,
        "sample_key": sample_key if has_sample else None, "batch_key": batch_key if has_batch else None,
        "panels_used": {ct: gs for ct, gs in panels.items()},
        "clusters": evidence,
    }

    if single_source_clusters:
        warnings.append(f"{len(single_source_clusters)} cluster(s) single-sample dominated (provenance flag)")
    if conflict_clusters:
        warnings.append(f"{len(conflict_clusters)} cluster(s) match >=2 cell types (marker conflict)")
    if unknown_clusters:
        warnings.append(f"{len(unknown_clusters)} cluster(s) Unknown (<{min_markers} panel markers)")

    dist = adata.obs["major_cell_type"].value_counts().to_dict()
    summary = {
        "label_key": "major_cell_type",
        "method": "leiden_DE_marker_combination (pct>=%.2f, LFC>=%.1f, >=%d markers)" % (min_pct, min_lfc, min_markers),
        "n_clusters": int(obs_g.nunique()),
        "label_distribution": {str(k): int(v) for k, v in dist.items()},
        "mean_confidence": round(float(adata.obs["major_confidence"].mean()), 3),
        "unknown_clusters": unknown_clusters,
        "marker_conflict_clusters": conflict_clusters,
        "single_source_clusters": single_source_clusters,
    }
    cp = session.checkpoint("annotate_broad", x_state=session.manifest.x_state,
                            params={"groupby": groupby, "min_pct": min_pct, "min_lfc": min_lfc,
                                    "min_markers": min_markers, "sample_key": sample_key,
                                    "batch_key": batch_key, "layer": use_layer})
    return S.success("annotate_broad", summary=summary, warnings=warnings, checkpoint=cp.path,
                     determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["plots", "integrate_scvi", "benchmark"])
