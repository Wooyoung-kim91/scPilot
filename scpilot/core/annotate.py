"""Annotation — Tier 1 broad (B8) + Tier 3 fine (B13).

Tier 1 (``annotate_broad``) assigns ``obs['major_cell_type']`` — the broad
compartment that becomes the scib benchmark ``label_key``. Tier 3 (bottom of file:
``fine_annotation_review`` → ``apply_fine_annotation``) refines WITHIN a compartment
to ``obs['fine_cell_type']`` + ``obs['facs_style_label']`` (evidence→LLM→apply, same
marker-DB-free split), recording authority/evidence in ``uns[...]['annotation_tree']``.

LOGIC (user-confirmed 2026-06-10, extended 2026-06-11):
1. **leiden-cluster DE** is the basis: ``rank_genes_groups`` (Wilcoxon, pts=True).
2. A gene is a **positive marker** for a cluster iff ``pct_in_group >= min_pct(0.25)``
   AND ``logfoldchange >= min_lfc(1.0)``; only the **top ``top_n_markers``(30)**
   threshold-passing genes (by DE score) per cluster are considered (rule 5) — a panel
   gene buried deep in the ranking is not evidence.
3. Annotate each cluster by the **combination** of its markers vs the broad cell-type
   panels (``BROAD_MARKERS``, from cancer_scrnaseq_annotation_strategy.md).
4. A cell-type call requires **>= min_markers(3)** of that type's panel among the
   cluster's top significant markers (else 'Unknown' = insufficient evidence).
5. **Negative markers / mutual exclusivity** (rule 3): a candidate compartment is
   penalized when its lineage-incompatible genes (``NEG_MARKERS``, e.g. PTPRC for a
   structural call, EPCAM for an immune call) are themselves significant up-markers of
   the cluster — used to break ties between competing panels.
6. **Mixed/Artifact** (rule 1): if >=2 panels survive as strong candidates (genuine
   co-expression of incompatible lineages) OR the cluster is doublet-dominated
   (``predicted_doublet`` fraction high), the cluster is labelled 'Mixed/Artifact'
   instead of being forced into a single type.
7. **Low-quality QC gating** (rule 4): a cluster dominated by high-%MT / low-complexity
   cells is labelled 'Low_quality' (Tier-0 flavour) and marked for review.
8. **Pan-immune (PTPRC/CD45) consistency** (rule 2): immune compartments
   (T_NK/B_Plasma/Myeloid/Mast) are expected PTPRC+, structural compartments
   (Epithelial/Stromal/Endothelial) PTPRC- — a mismatch penalizes confidence and sets
   ``review_required``.
9. **Sample provenance** is considered: per-cluster sample/condition composition;
   single-sample / single-batch dominated clusters are flagged (a marker match from one
   sample is weaker evidence; also the de-risk ① circular-label signal).
10. Result layout (separate ``plots`` calls): UMAP(color=major_cell_type) + dotplot
    (``sc.pl.dotplot`` with the marker panels as a dict → cell-type brackets/labels).

Every rule SKIPS gracefully when its input gene/obs column is absent, so the same tool
runs on a minimal AnnData and on the full PDAC object.

Evidence (matched markers, candidates, negatives, PTPRC, QC, provenance, flags) →
``.uns['scpilot_annotation']['tier1']``.
"""

from __future__ import annotations

import time

from scpilot import schemas as S
from scpilot.tools import register

# Tier 1 broad lineage panels (positive markers; from cancer_scrnaseq_annotation_strategy.md).
BROAD_MARKERS = {
    "Epithelial":  ["EPCAM", "KRT8", "KRT18", "KRT19"],
    "T_NK":        ["CD3D", "CD3E", "TRAC", "NKG7", "GNLY", "KLRD1"],
    "B_Plasma":    ["MS4A1", "CD79A", "CD79B", "MZB1", "JCHAIN", "XBP1"],
    "Myeloid":     ["LYZ", "CD68", "C1QA", "C1QB", "C1QC", "FCGR3A", "S100A8"],
    "Stromal":     ["COL1A1", "COL1A2", "DCN", "LUM", "ACTA2", "PDGFRB"],
    "Endothelial": ["PECAM1", "VWF", "KDR", "CLDN5"],
    "Mast":        ["TPSAB1", "TPSB2", "CPA3", "KIT"],
}

# Negative markers (rule 3): genes a compartment should NOT express. Used only as a
# tie-breaker — a candidate loses a point per negative gene that is a significant
# up-marker of the cluster. Kept to the strongest cross-compartment discriminators.
NEG_MARKERS = {
    "Epithelial":  ["PTPRC"],                  # CD45- (not immune)
    "T_NK":        ["EPCAM", "LYZ"],           # not epithelial / myeloid
    "B_Plasma":    ["EPCAM", "CD3D"],
    "Myeloid":     ["EPCAM", "CD3D"],
    "Stromal":     ["PTPRC", "EPCAM"],         # CD45- EpCAM- mesenchyme
    "Endothelial": ["PTPRC", "EPCAM"],
    "Mast":        ["EPCAM", "CD3D"],
}

IMMUNE_COMPARTMENTS = {"T_NK", "B_Plasma", "Myeloid", "Mast"}
NONIMMUNE_COMPARTMENTS = {"Epithelial", "Stromal", "Endothelial"}
MIXED_LABEL = "Mixed/Artifact"
LOWQ_LABEL = "Low_quality"
UNKNOWN_LABEL = "Unknown"
AMBIGUOUS_LABEL = "ambiguous"   # consensus_annotation sentinel: methods disagree
# tool-produced NON-BIOLOGICAL sentinels (NOT cell types) — excluded from the scib
# bio-conservation label set downstream. (Dataset/tissue-specific labels are NEVER added
# here; drop those per-run via the benchmark `drop_labels` param.)
ARTIFACT_LABELS = {UNKNOWN_LABEL, MIXED_LABEL, LOWQ_LABEL, AMBIGUOUS_LABEL}
UNS_ANNO = "scpilot_annotation"

# ---- Tier 3 (fine) constants (B13) ----
UNS_TREE = "annotation_tree"            # nested under UNS_ANNO: per-subcluster authority + evidence
FINE_UNRESOLVED = "unresolved"          # merge fallback label for under-powered subclusters
# Confounder obs columns surfaced per subcluster — these are SCORE/QC columns produced upstream
# (NOT gene panels): reading them is evidence, not a hardcoded marker list. Genes for on-the-fly
# scoring are caller-supplied via `confounder_genes` (no built-in panel).
CONFOUNDER_KEYS = ("cell_cycle_score", "S_score", "G2M_score", "stress_score",
                   "interferon_score", "activation_score", "doublet_score", "pct_counts_mt")
FINE_DOUBLET_KEY = "predicted_doublet"


def _expressed_fraction(adata, gene: str):
    """Per-cell boolean: is ``gene`` expressed (X>0)? None if the gene is absent."""
    if gene not in adata.var_names:
        return None
    import numpy as np
    import scipy.sparse as sp
    col = adata[:, gene].X
    col = col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel()
    return col > 0


@register("annotate_broad", mutating=True,
          description="LEGACY / opt-in (FIXED marker panel). Deterministic Tier-1 via leiden-cluster DE matched "
                      "to hardcoded BROAD_MARKERS panels — organism/tissue-biased, misses panel-absent types. "
                      "The PRIMARY Tier-1 path is now marker-DB-FREE: markers → annotation_review → apply_annotation "
                      "(LLM infers types from DE). Use this only as a quick fixed-panel sanity check.")
def annotate_broad(session, *, groupby: str = "leiden", min_pct: float = 0.25, min_lfc: float = 1.0,
                   min_markers: int = 3, top_n_markers: int = 30,
                   sample_key: str = "sample_id", batch_key: str = "GSE",
                   layer: str | None = "scale.data", single_source_frac: float = 0.8,
                   # rule 2 (PTPRC consistency)
                   ptprc_gene: str = "PTPRC", immune_ptprc_min: float = 0.20,
                   nonimmune_ptprc_max: float = 0.50, ptprc_penalty: float = 0.7,
                   # rule 1 (Mixed/Artifact)
                   doublet_key: str = "predicted_doublet", doublet_frac: float = 0.5,
                   # rule 4 (low-quality QC gating)
                   qc_gate: bool = True, mt_key: str = "pct_counts_mt", max_pct_mt: float = 25.0,
                   genes_key: str = "n_genes_by_counts", min_genes: float = 300.0,
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
    neg_panels = {ct: [g for g in NEG_MARKERS.get(ct, []) if g in adata.var_names] for ct in panels}

    # 1) leiden-cluster DE (Wilcoxon) with per-group expressed fraction (pts)
    use_layer = layer if (layer and layer in adata.layers) else None
    if use_layer is None and layer:
        warnings.append(f"layer '{layer}' absent — DE on X")
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon", pts=True,
                            layer=use_layer, use_raw=False)
    de = sc.get.rank_genes_groups_df(adata, group=None)   # ordered by score desc within group

    # 2) significant markers per cluster, restricted to the top-N by score (rule 5)
    sig = de[(de["pct_nz_group"] >= min_pct) & (de["logfoldchanges"] >= min_lfc)]
    sig_top = sig.groupby("group", observed=True).head(top_n_markers)
    sig_by_cluster = {str(g): set(sub["names"]) for g, sub in sig_top.groupby("group", observed=True)}

    obs_g = adata.obs[groupby].astype(str)
    has_sample = sample_key in adata.obs.columns
    has_batch = batch_key in adata.obs.columns

    # per-cell PTPRC expression (rule 2) + per-cluster QC stats (rule 4), vectorized
    ptprc_expr = _expressed_fraction(adata, ptprc_gene)
    has_doublet = doublet_key in adata.obs.columns
    has_mt = qc_gate and mt_key in adata.obs.columns
    has_genes = qc_gate and genes_key in adata.obs.columns

    cluster_label, cluster_conf, cluster_review = {}, {}, {}
    evidence = {}
    conflict_clusters, single_source_clusters, unknown_clusters = [], [], []
    mixed_clusters, lowq_clusters, ptprc_bad_clusters = [], [], []

    for cl in (obs_g.cat.categories if hasattr(obs_g, "cat") else sorted(set(obs_g))):
        cl = str(cl)
        sigset = sig_by_cluster.get(cl, set())
        mask = (obs_g == cl).values
        n_cells = int(mask.sum())

        # ----- positive panel matches (top-N significant up-markers) -----
        pos = {ct: [g for g in gs if g in sigset] for ct, gs in panels.items()}
        candidates = {ct: m for ct, m in pos.items() if len(m) >= min_markers}

        # ----- negative-marker adjustment (rule 3): tie-break competing panels -----
        neg_present = {ct: [g for g in neg_panels[ct] if g in sigset] for ct in candidates}
        adj = {ct: len(candidates[ct]) - len(neg_present[ct]) for ct in candidates}
        adj_candidates = {ct: candidates[ct] for ct in candidates if adj[ct] >= min_markers}

        # ----- QC + doublet evidence -----
        qc = {}
        if has_mt:
            qc["median_pct_mt"] = round(float(np.median(adata.obs[mt_key].values[mask])), 2)
        if has_genes:
            qc["median_n_genes"] = round(float(np.median(adata.obs[genes_key].values[mask])), 1)
        if has_doublet:
            dv = adata.obs[doublet_key].values[mask]
            qc["doublet_frac"] = round(float(np.asarray(dv, dtype=float).mean()), 3)
        ptprc_frac = round(float(ptprc_expr[mask].mean()), 3) if ptprc_expr is not None else None

        # ----- provenance (rule 9) -----
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

        # ----- decision (precedence: Low_quality → Mixed/Artifact → biological → Unknown) -----
        marker_conflict = len(adj_candidates) >= 2
        if marker_conflict:
            conflict_clusters.append(cl)
        doublet_dominated = has_doublet and qc.get("doublet_frac", 0.0) >= doublet_frac

        label, conf, reason, review = UNKNOWN_LABEL, 0.0, "no panel reached min_markers", False
        ptprc_consistent = None

        if has_mt and qc.get("median_pct_mt", 0.0) > max_pct_mt:
            label, reason, review = LOWQ_LABEL, f"median %MT {qc['median_pct_mt']} > {max_pct_mt}", True
            lowq_clusters.append(cl)
        elif has_genes and qc.get("median_n_genes", 1e9) < min_genes:
            label, reason, review = LOWQ_LABEL, f"median n_genes {qc['median_n_genes']} < {min_genes}", True
            lowq_clusters.append(cl)
        elif doublet_dominated or marker_conflict:
            label, reason, review = MIXED_LABEL, (
                f"doublet_frac {qc.get('doublet_frac')} >= {doublet_frac}" if doublet_dominated
                else f"co-expression of {sorted(adj_candidates)} (>=2 incompatible panels)"), True
            mixed_clusters.append(cl)
        elif adj_candidates or candidates:
            # winner: highest negative-adjusted score, then most positive markers
            pool = adj_candidates or candidates
            ct = max(pool, key=lambda c: (adj.get(c, len(pool[c])), len(pool[c])))
            matched = candidates[ct]
            label = ct
            conf = round(len(matched) / len(panels[ct]), 3)
            reason = f"{len(matched)}/{len(panels[ct])} {ct} markers (adj {adj.get(ct)})"
            # rule 2: PTPRC consistency
            if ptprc_frac is not None:
                if ct in IMMUNE_COMPARTMENTS and ptprc_frac < immune_ptprc_min:
                    ptprc_consistent = False
                elif ct in NONIMMUNE_COMPARTMENTS and ptprc_frac > nonimmune_ptprc_max:
                    ptprc_consistent = False
                else:
                    ptprc_consistent = True
                if ptprc_consistent is False:
                    conf = round(conf * ptprc_penalty, 3)
                    review = True
                    ptprc_bad_clusters.append(cl)
                    reason += f"; PTPRC {ptprc_frac} inconsistent with {ct}"
        else:
            unknown_clusters.append(cl)

        cluster_label[cl] = label
        cluster_conf[cl] = conf
        cluster_review[cl] = review
        evidence[cl] = {
            "label": label, "confidence": conf, "review_required": review, "decision_reason": reason,
            "n_markers_matched": len(candidates.get(label, [])),
            "matched_markers": candidates.get(label, []),
            "candidates": {ct: m for ct, m in candidates.items()},
            "candidates_adjusted": {ct: adj[ct] for ct in candidates},
            "negative_markers_present": {ct: neg_present[ct] for ct in candidates if neg_present[ct]},
            "ptprc_frac": ptprc_frac, "ptprc_consistent": ptprc_consistent,
            "qc": qc, "doublet_dominated": doublet_dominated,
            "single_source": cl in single_source_clusters,
            "marker_conflict": marker_conflict,
            **prov,
        }

    adata.obs["major_cell_type"] = obs_g.map(cluster_label).astype(str)
    adata.obs["major_confidence"] = obs_g.map(cluster_conf).astype(float)
    adata.obs["major_review_required"] = obs_g.map(cluster_review).astype(bool)
    adata.uns.setdefault(UNS_ANNO, {})
    adata.uns[UNS_ANNO]["tier1"] = {
        "method": "leiden_DE_marker_combination",
        "groupby": groupby, "min_pct": min_pct, "min_lfc": min_lfc, "min_markers": min_markers,
        "top_n_markers": top_n_markers,
        "rules": ["positive_top_n", "negative_marker_tiebreak", "mixed_artifact",
                  "low_quality_qc", "ptprc_consistency", "sample_provenance"],
        "ptprc_gene": ptprc_gene if ptprc_expr is not None else None,
        "qc_gate": bool(has_mt or has_genes),
        "sample_key": sample_key if has_sample else None, "batch_key": batch_key if has_batch else None,
        "panels_used": {ct: gs for ct, gs in panels.items()},
        "neg_panels_used": {ct: gs for ct, gs in neg_panels.items() if gs},
        "clusters": evidence,
    }

    if single_source_clusters:
        warnings.append(f"{len(single_source_clusters)} cluster(s) single-sample dominated (provenance flag)")
    if conflict_clusters:
        warnings.append(f"{len(conflict_clusters)} cluster(s) match >=2 cell types (marker conflict)")
    if mixed_clusters:
        warnings.append(f"{len(mixed_clusters)} cluster(s) → {MIXED_LABEL} (doublet/co-expression)")
    if lowq_clusters:
        warnings.append(f"{len(lowq_clusters)} cluster(s) → {LOWQ_LABEL} (high %MT / low complexity)")
    if ptprc_bad_clusters:
        warnings.append(f"{len(ptprc_bad_clusters)} cluster(s) PTPRC-inconsistent (confidence penalized)")
    if unknown_clusters:
        warnings.append(f"{len(unknown_clusters)} cluster(s) Unknown (<{min_markers} panel markers)")

    dist = adata.obs["major_cell_type"].value_counts().to_dict()
    summary = {
        "label_key": "major_cell_type",
        "method": "leiden_DE_marker_combination (pct>=%.2f, LFC>=%.1f, top-%d, >=%d markers)"
                  % (min_pct, min_lfc, top_n_markers, min_markers),
        "n_clusters": int(obs_g.nunique()),
        "label_distribution": {str(k): int(v) for k, v in dist.items()},
        "mean_confidence": round(float(adata.obs["major_confidence"].mean()), 3),
        "n_review_required": int(adata.obs["major_review_required"].sum()),
        "unknown_clusters": unknown_clusters,
        "marker_conflict_clusters": conflict_clusters,
        "mixed_artifact_clusters": mixed_clusters,
        "low_quality_clusters": lowq_clusters,
        "ptprc_inconsistent_clusters": ptprc_bad_clusters,
        "single_source_clusters": single_source_clusters,
    }
    cp = session.checkpoint("annotate_broad", x_state=session.manifest.x_state,
                            params={"groupby": groupby, "min_pct": min_pct, "min_lfc": min_lfc,
                                    "min_markers": min_markers, "top_n_markers": top_n_markers,
                                    "sample_key": sample_key, "batch_key": batch_key, "layer": use_layer,
                                    "ptprc_gene": ptprc_gene, "qc_gate": qc_gate,
                                    "max_pct_mt": max_pct_mt, "min_genes": min_genes,
                                    "doublet_frac": doublet_frac})
    return S.success("annotate_broad", summary=summary, warnings=warnings, checkpoint=cp.path,
                     determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["plots", "integrate_scvi", "benchmark"])


@register("annotation_review", mutating=False,
          description="PRIMARY Tier-1 annotation evidence (read-only, NO fixed marker panel): packages each "
                      "cluster's top-N SIGNIFICANT (padj<0.05, configurable) ranked DE (logFC/padj/pct_in/pct_out) "
                      "+ cluster size + sample distribution + QC + panel-free artifact flags for the LLM. "
                      "The LLM (host agent / mode-2) infers each cluster's cell type FROM THE DE ITSELF — no "
                      "marker database — applies tissue= as a soft prior to flag implausible calls, then writes "
                      "the calls via apply_annotation. Run after markers (which produces the pts DE). "
                      "See llm/prompts.py ANNOTATION_REVIEW_PROMPT + TISSUE_CONTEXT_GUIDANCE.")
def annotation_review(session, *, groupby: str = "leiden", top_n: int = 50, padj_max: float = 0.05,
                      min_specificity: float = 0.1, max_pct_out: float = 0.5, min_specific_markers: int = 3,
                      sample_key: str = "sample_id", tissue: str | None = None, max_samples_reported: int = 8,
                      doublet_key: str = "predicted_doublet", doublet_frac: float = 0.5,
                      single_source_frac: float = 0.8, mt_key: str = "pct_counts_mt", max_pct_mt: float = 25.0,
                      genes_key: str = "n_genes_by_counts", min_genes: float = 300.0, min_cells: int = 20,
                      **params) -> S.ToolResult:
    """Deterministic, marker-DB-FREE evidence packager — the boundary between replayable tools
    (this) and non-deterministic LLM reasoning (the cell-type call). It reuses the per-cluster
    DE that ``markers`` computed (with ``pts``); it does NOT match against any panel and emits
    no candidate label. Only SIGNIFICANT up-markers (``pvals_adj < padj_max``, default 0.05) are
    exposed, and the top-N of those per cluster. The only deterministic signals it adds are
    panel-independent QC/artifact flags (high %MT, low complexity, doublet-dominated, single-sample)."""
    import json

    import numpy as np
    import scanpy as sc

    t0 = time.time()
    adata = session.adata
    if groupby not in adata.obs.columns:
        return S.error("annotation_review", "invalid_state",
                       f"clustering '{groupby}' absent — run cluster first", recoverable=True,
                       suggested_next_tools=["cluster"])
    rg = adata.uns.get("rank_genes_groups")
    if not rg or rg.get("params", {}).get("groupby") != groupby:
        return S.error("annotation_review", "invalid_state",
                       f"per-cluster DE for '{groupby}' absent — run markers (groupby={groupby}) first",
                       recoverable=True, suggested_next_tools=["markers"])
    de = sc.get.rank_genes_groups_df(adata, group=None)
    if "pct_nz_group" not in de.columns:
        return S.error("annotation_review", "invalid_state",
                       "DE lacks expressed-fraction (pct) — re-run markers (it sets pts=True)",
                       recoverable=True, suggested_next_tools=["markers"])
    # explicit significance gate: keep only significant up-markers (padj < padj_max).
    n_de_total = int(len(de))
    if "pvals_adj" in de.columns:
        de = de[de["pvals_adj"] < padj_max]
    n_sig_total = int(len(de))
    # SPECIFICITY axis added to the DE step (spec = pct_in - pct_out): a marker must be ENRICHED
    # in-cluster, not broadly expressed. A high-pct_out gene (MALAT1, ribosomal, ...) is weak identity
    # evidence even at high logFC, so the list the LLM reads is significant AND specific — the data
    # sets specificity, no marker DB. spec is surfaced per gene and the surfaced top-N is the most
    # specific markers (so the cluster's own identity leads).
    has_spec = {"pct_nz_group", "pct_nz_reference"}.issubset(de.columns)
    if has_spec:
        de = de.assign(spec=(de["pct_nz_group"] - de["pct_nz_reference"]).round(4))
        _spec_ok = (de["spec"] >= min_specificity) & (de["pct_nz_reference"] <= max_pct_out)
    n_specific_total = int(_spec_ok.sum()) if has_spec else n_sig_total

    de_cols = {"names": "gene", "logfoldchanges": "logFC", "pvals_adj": "padj",
               "pct_nz_group": "pct_in", "pct_nz_reference": "pct_out", "spec": "spec", "scores": "score"}
    present_cols = [c for c in de_cols if c in de.columns]
    obs_g = adata.obs[groupby].astype(str)
    has_sample = sample_key in adata.obs.columns
    has_doublet = doublet_key in adata.obs.columns
    has_mt = mt_key in adata.obs.columns
    has_genes = genes_key in adata.obs.columns

    payloads, rows = [], []
    status_counts = {"clean": 0, "review": 0, "artifact_suspected": 0}
    for cl, sub in de.groupby("group", observed=True):
        cl = str(cl)
        n_sig = int(len(sub))                          # significant up-markers for this cluster
        # significant AND specific (delta + out-group ceiling), most-specific first.
        spec_sub = sub[(sub["spec"] >= min_specificity) & (sub["pct_nz_reference"] <= max_pct_out)] \
            .sort_values("spec", ascending=False) if has_spec else sub
        n_specific = int(len(spec_sub))
        top = spec_sub.head(top_n)                     # top-N SPECIFIC markers
        de_table = [{de_cols[c]: (round(float(r[c]), 4) if c != "names" else str(r[c]))
                     for c in present_cols} for _, r in top.iterrows()]
        mask = (obs_g == cl).values
        n_cells = int(mask.sum())

        sample_dist, single_source = {}, False
        if has_sample:
            sv = adata.obs[sample_key].astype(str)[mask].value_counts(normalize=True)
            sample_dist = {str(k): round(float(v), 3) for k, v in sv.head(max_samples_reported).items()}
            single_source = bool(sv.iloc[0] >= single_source_frac)
        qc = {}
        if has_mt:
            qc["median_pct_mt"] = round(float(np.median(adata.obs[mt_key].values[mask])), 2)
        if has_genes:
            qc["median_n_genes"] = round(float(np.median(adata.obs[genes_key].values[mask])), 1)
        if has_doublet:
            qc["doublet_frac"] = round(float(np.asarray(adata.obs[doublet_key].values[mask], dtype=float).mean()), 3)

        # panel-FREE artifact baseline (QC + provenance only — never a cell-type call)
        doublet_dominated = has_doublet and qc.get("doublet_frac", 0.0) >= doublet_frac
        low_quality = (has_mt and qc.get("median_pct_mt", 0) > max_pct_mt) or \
                      (has_genes and qc.get("median_n_genes", 1e9) < min_genes)
        risk = []
        if doublet_dominated or low_quality:
            status = "artifact_suspected"
            if doublet_dominated: risk.append("doublet_dominated")
            if low_quality: risk.append("low_quality_qc")
        elif single_source or n_cells < min_cells or n_specific < min_specific_markers:
            status = "review"
            if single_source: risk.append("single_source")
            if n_cells < min_cells: risk.append("tiny_cluster")
            if n_specific < min_specific_markers: risk.append("few_specific_markers")
        else:
            status = "clean"
        status_counts[status] += 1

        payloads.append({
            "cluster_id": cl, "cluster_size": n_cells, "n_significant_markers": n_sig,
            "n_specific_markers": n_specific,
            "review_status": status, "risk_signals": risk,
            "deterministic_flags": {"doublet_dominated": doublet_dominated,
                                    "low_quality_qc": low_quality, "single_source": single_source},
            "sample_distribution": sample_dist, "qc_metrics": qc,
            "de_table": de_table,
        })
        rows.append({"cluster": cl, "n_cells": n_cells, "review_status": status,
                     "risk_signals": ",".join(risk),
                     "top_de": ",".join(d["gene"] for d in de_table[:8])})

    art_dir = session.artifacts_dir
    art_dir.mkdir(parents=True, exist_ok=True)
    json_path = art_dir / "annotation_review.json"
    json_path.write_text(json.dumps(
        {"groupby": groupby, "top_n": top_n, "tissue_context": tissue,
         "significance_filter": f"pvals_adj < {padj_max}",
         "specificity_filter": f"(pct_in - pct_out) >= {min_specificity} AND pct_out <= {max_pct_out}",
         "n_de_total": n_de_total, "n_significant_total": n_sig_total,
         "n_specific_total": n_specific_total,
         "marker_db_used": False, "candidate_labels_provided": False,
         "instruction": f"de_table = top-{top_n} markers per cluster that are SIGNIFICANT (padj<{padj_max}) "
                        f"AND SPECIFIC (spec = pct_in - pct_out >= {min_specificity}), most-specific first. "
                        "Each gene shows spec so you can weigh enrichment vs broad expression. Infer each "
                        "cluster's broad cell type from its de_table (no marker DB); use tissue_context as a "
                        "soft prior; clusters flagged 'few_specific_markers' lack a distinct identity (likely "
                        "low-quality/ambient/doublet) — treat with caution. Then call apply_annotation with the "
                        "cluster->label map.",
         "clusters": payloads}, indent=2, default=str))

    import pandas as pd
    table = S.table_preview(pd.DataFrame(rows), max_rows=len(rows))
    flagged = [p["cluster_id"] for p in payloads if p["review_status"] != "clean"]
    summary = {
        "groupby": groupby, "top_n": top_n, "n_clusters": len(payloads),
        "significance_filter": f"pvals_adj < {padj_max}", "padj_max": padj_max,
        "specificity_filter": f"(pct_in - pct_out) >= {min_specificity} AND pct_out <= {max_pct_out}",
        "min_specificity": min_specificity, "max_pct_out": max_pct_out,
        "n_significant_total": n_sig_total, "n_specific_total": n_specific_total, "n_de_total": n_de_total,
        "tissue_context": tissue, "marker_db_used": False,
        "status_counts": status_counts, "flagged_clusters": flagged,
        "review_input": str(json_path),
        "note": f"DE-based annotation: de_table = top-{top_n} markers that are SIGNIFICANT (padj<{padj_max}) "
                f"AND SPECIFIC (spec=pct_in-pct_out>={min_specificity}), most-specific first; the LLM assigns "
                "cell types WITHOUT a marker panel (tissue_context = soft prior), then calls apply_annotation. "
                "'few_specific_markers' flags clusters lacking a distinct identity. Flags are panel-free QC only.",
    }
    warnings = [] if not flagged else [f"{len(flagged)} cluster(s) flagged (QC/provenance): {flagged}"]
    return S.success("annotation_review", summary=summary, tables={"review": table},
                     artifacts=[S.Artifact(path=str(json_path), kind="json",
                                           description="per-cluster DE evidence for marker-DB-free LLM annotation")],
                     warnings=warnings, determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["apply_annotation"])


@register("apply_annotation", mutating=True,
          description="Write the LLM's DE-based Tier-1 calls into obs['major_cell_type'] — the cluster->label "
                      "map the LLM inferred from annotation_review evidence (NO fixed marker panel). Deterministic "
                      "given the map, so the LLM's judgment is fully replayable. Optionally takes per-cluster "
                      "confidence / review_required / cell_state maps. This is the PRIMARY Tier-1 result.")
def apply_annotation(session, *, groupby: str = "leiden", labels: dict | None = None,
                     key: str = "major_cell_type", confidence: dict | None = None,
                     review_required: dict | None = None, cell_state: dict | None = None,
                     tissue: str | None = None, method: str = "DE_LLM_marker_free",
                     unassigned: str = "Unassigned", **params) -> S.ToolResult:
    t0 = time.time()
    adata = session.adata
    if groupby not in adata.obs.columns:
        return S.error("apply_annotation", "invalid_state",
                       f"clustering '{groupby}' absent — run cluster first", recoverable=True,
                       suggested_next_tools=["cluster"])
    if not labels:
        return S.error("apply_annotation", "missing_input",
                       "no 'labels' map given (expected {cluster_id: cell_type} from the LLM)",
                       recoverable=True, suggested_next_tools=["annotation_review"])

    obs_g = adata.obs[groupby].astype(str)
    clusters = set(obs_g.unique())
    lab = {str(k): str(v) for k, v in labels.items()}
    missing = sorted(clusters - set(lab))                 # clusters the LLM did not label
    adata.obs[key] = obs_g.map(lambda c: lab.get(c, unassigned)).astype("category")
    if confidence:
        cf = {str(k): float(v) for k, v in confidence.items()}
        adata.obs[f"{key}_confidence"] = obs_g.map(lambda c: cf.get(c, float("nan"))).astype(float)
    if review_required:
        rv = {str(k): bool(v) for k, v in review_required.items()}
        adata.obs[f"{key}_review_required"] = obs_g.map(lambda c: rv.get(c, False)).astype(bool)
    if cell_state:
        cs = {str(k): str(v) for k, v in cell_state.items()}
        adata.obs["cell_state"] = obs_g.map(lambda c: cs.get(c, "")).astype("category")

    adata.uns.setdefault(UNS_ANNO, {})
    adata.uns[UNS_ANNO]["tier1_llm"] = {
        "method": method, "groupby": groupby, "label_key": key, "tissue_context": tissue,
        "labels": lab, "marker_db_used": False,
    }
    dist = adata.obs[key].value_counts().to_dict()
    n_unassigned = int((adata.obs[key].astype(str) == unassigned).sum())
    try:
        session.log_decision(S.DecisionEvent(
            decision_type="tier1_llm_labels", choice=lab, candidates=[],
            rationale=f"DE-based marker-free Tier-1 labels (tissue={tissue})",
            stage="apply_annotation", params={"groupby": groupby, "key": key}).to_dict())
    except Exception:  # noqa: BLE001
        pass

    summary = {
        "label_key": key, "method": method, "groupby": groupby, "tissue_context": tissue,
        "marker_db_used": False, "n_clusters_labeled": len(lab),
        "unlabeled_clusters": missing, "n_unassigned_cells": n_unassigned,
        "label_distribution": {str(k): int(v) for k, v in dist.items()},
    }
    warnings = [] if not missing else [f"{len(missing)} cluster(s) not in labels → '{unassigned}': {missing}"]
    cp = session.checkpoint("apply_annotation", x_state=session.manifest.x_state,
                            params={"groupby": groupby, "key": key, "method": method})
    return S.success("apply_annotation", summary=summary, warnings=warnings, checkpoint=cp.path,
                     determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["plots", "consensus_annotation", "integrate_scvi", "benchmark"])


@register("consensus_annotation", mutating=True,
          description="Build a per-cell CONSENSUS cell-type label by majority vote across SEVERAL annotation "
                      "columns you pass in `keys` (e.g. the per-integration-method annotations). Purpose: an "
                      "UNBIASED benchmark label_key that is NOT derived from any single embedding — so scib "
                      "bio-conservation is not circular (de-risk ①). Cells where no label reaches `min_agreement` "
                      "(or a tie) are marked '" + AMBIGUOUS_LABEL + "' and excluded downstream. NO hardcoding: the "
                      "annotation columns AND any labels to exclude are caller-supplied; no built-in vocabulary.")
def consensus_annotation(session, *, keys: list | None = None, out_key: str = "celltype_consensus",
                         min_agreement: float = 0.5, ambiguous_label: str = AMBIGUOUS_LABEL,
                         **params) -> S.ToolResult:
    """Per-cell majority vote across the given annotation columns. A label must be held by a
    fraction > ``min_agreement`` of the columns AND be a unique winner (no tie) to become the
    consensus; otherwise the cell is ``ambiguous_label``. Embedding-independent by construction
    (it combines per-method calls), so it is the recommended ``benchmark`` label_key."""
    import numpy as np
    import pandas as pd

    t0 = time.time()
    adata = session.adata
    keys = list(keys or [])
    if len(keys) < 2:
        return S.error("consensus_annotation", "missing_input",
                       "pass >=2 annotation columns in 'keys' to vote across (e.g. per-method labels)",
                       recoverable=True, suggested_next_tools=["apply_annotation"])
    missing = [k for k in keys if k not in adata.obs.columns]
    if missing:
        return S.error("consensus_annotation", "missing_input",
                       f"annotation column(s) absent in obs: {missing}", recoverable=True,
                       suggested_next_tools=["apply_annotation"])

    n_keys = len(keys)
    cats = pd.unique(np.concatenate([adata.obs[k].astype(str).values for k in keys]))
    code = {c: i for i, c in enumerate(cats)}
    codes = np.column_stack([
        pd.Categorical(adata.obs[k].astype(str).values, categories=cats).codes for k in keys])  # (n, n_keys)

    out = np.empty(adata.n_obs, dtype=object)
    agree = np.zeros(adata.n_obs, dtype=float)
    for i in range(adata.n_obs):
        counts = np.bincount(codes[i], minlength=len(cats))
        mx = counts.max()
        winners = np.flatnonzero(counts == mx)
        agree[i] = mx / n_keys
        out[i] = cats[winners[0]] if (winners.size == 1 and mx / n_keys > min_agreement) else ambiguous_label

    adata.obs[out_key] = pd.Categorical(out)
    adata.obs[f"{out_key}_agreement"] = agree.astype("float32")

    # pairwise agreement between the input annotations (how concordant the methods are)
    pairwise = {}
    for a in range(n_keys):
        for b in range(a + 1, n_keys):
            frac = float((codes[:, a] == codes[:, b]).mean())
            pairwise[f"{keys[a]}__vs__{keys[b]}"] = round(frac, 3)

    dist = adata.obs[out_key].value_counts().to_dict()
    n_amb = int((adata.obs[out_key].astype(str) == ambiguous_label).sum())
    adata.uns.setdefault(UNS_ANNO, {})
    adata.uns[UNS_ANNO]["consensus"] = {
        "out_key": out_key, "source_keys": keys, "min_agreement": min_agreement,
        "ambiguous_label": ambiguous_label, "n_ambiguous": n_amb, "pairwise_agreement": pairwise,
    }
    try:
        session.log_decision(S.DecisionEvent(
            decision_type="consensus_label", choice={"out_key": out_key, "keys": keys},
            candidates=[], rationale=f"majority vote across {keys} (min_agreement={min_agreement})",
            stage="consensus_annotation", params={"out_key": out_key}).to_dict())
    except Exception:  # noqa: BLE001
        pass

    summary = {
        "out_key": out_key, "source_keys": keys, "n_keys": n_keys, "min_agreement": min_agreement,
        "n_ambiguous": n_amb, "ambiguous_frac": round(n_amb / adata.n_obs, 3),
        "mean_agreement": round(float(agree.mean()), 3), "pairwise_agreement": pairwise,
        "label_distribution": {str(k): int(v) for k, v in dist.items()},
        "note": f"Embedding-independent consensus → use as benchmark label_key. '{ambiguous_label}' + other "
                "non-cell-type labels should be passed to benchmark drop_labels (caller-chosen, not hardcoded).",
    }
    warnings = [f"{n_amb} cells '{ambiguous_label}' (methods disagree)"] if n_amb else []
    cp = session.checkpoint("consensus_annotation", x_state=session.manifest.x_state,
                            params={"keys": keys, "out_key": out_key, "min_agreement": min_agreement})
    return S.success("consensus_annotation", summary=summary, warnings=warnings, checkpoint=cp.path,
                     determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["benchmark", "plots"])


# --------------------------------------------------------------------------- #
# dotplot marker derivation — the markers ARE the Tier-1 annotation evidence
# --------------------------------------------------------------------------- #
# OPTIONAL biological-compartment ROW/PANEL order (display layout only — NOT marker selection).
# top→bottom: parenchymal → vascular/stromal/neural → immune → proliferating → artificial.
# Pass as ``order=``; absent types are skipped. Leave None to order by cell-type abundance.
COMPARTMENT_ORDER = [
    "Epithelial", "Endothelial", "Acinar", "Endocrine", "Ductal",
    "Stromal", "Schwann",
    "T_NK", "B", "Plasma", "Myeloid", "Mast",
    "Cycling",
    "Erythrocyte", "Stress", "Low_quality",
]


def derive_dotplot_markers(adata, *, cluster_key, label_map, top_k=5,
                           min_pct=0.25, min_lfc=1.0, padj_max=0.05, min_specificity=0.1,
                           max_pct_out=0.5, order=None):
    """Per-cell-type dotplot markers that ARE the Tier-1 annotation EVIDENCE — the same DE-selected
    significant markers the LLM read to make the call, shown per assigned cell type. No external
    marker DB, no lineage prior: the marker SELECTION itself is the evidence (that is the point).

    PHILOSOPHY: DE selection → LLM picks the combination → label. This builds the dotplot panels
    from EXACTLY that DE selection: for each cluster the LLM mapped to a type (``label_map``:
    cluster→label), take the cluster's Tier-1 positive markers from ``adata.uns['rank_genes_groups']``
    — ``pvals_adj < padj_max`` AND ``pct_nz_group >= min_pct`` AND ``logfoldchanges >= min_lfc``
    (the annotate_broad rule-2 gate) — ranked by the rank_genes_groups SCORE (the same order
    annotation_review exposes to the LLM). The top ``top_k`` per cell type become its panel; a
    shared gene stays with whichever cell type ranks it strongest. NOTHING is injected that the
    DE didn't select — the figure shows the evidence, not a curated panel.

    ``order`` (e.g. ``COMPARTMENT_ORDER``) only fixes the panel/row LAYOUT (display), never the
    marker content; without it, cell types are laid out by abundance.

    Returns ``{cell_type: [genes]}`` ready to pass as ``plots(kind='dotplot', marker_groups=...)``.
    """
    from collections import defaultdict

    import scanpy as sc

    rg = adata.uns.get("rank_genes_groups")
    if not rg or rg.get("params", {}).get("groupby") != cluster_key:
        raise ValueError(f"per-cluster DE for '{cluster_key}' absent — run markers(groupby={cluster_key}) first")
    de = sc.get.rank_genes_groups_df(adata, group=None)
    for col in ("pvals_adj", "pct_nz_group", "logfoldchanges", "scores"):
        if col not in de.columns:
            raise ValueError(f"DE lacks '{col}' — re-run markers with pts=True")
    # Tier-1 positive-marker gate (annotate_broad rule 2) — the DATA selects the evidence. The
    # SPECIFICITY gate has TWO parts so a near-saturated gene (pct_in≈1) can't slip through on the
    # delta alone: (i) enrichment delta pct_in-pct_out >= min_specificity, AND (ii) an ABSOLUTE
    # out-group ceiling pct_out <= max_pct_out — a gene expressed in a large fraction of OTHER cell
    # types (HLA-DRA, ribosomal, FCER1G, ...) is shared, not identity, and is dropped here.
    de = de[(de["pvals_adj"] < padj_max) & (de["pct_nz_group"] >= min_pct)
            & (de["logfoldchanges"] >= min_lfc)]
    if "pct_nz_reference" in de.columns:
        de = de[((de["pct_nz_group"] - de["pct_nz_reference"]) >= min_specificity)
                & (de["pct_nz_reference"] <= max_pct_out)]

    sizes = adata.obs[cluster_key].astype(str).value_counts().to_dict()
    best = defaultdict(dict)           # cell type -> {gene: best DE score across its clusters}
    ctsize = defaultdict(int)
    for cl, sub in de.groupby("group", observed=True):
        ct = label_map.get(str(cl))
        if ct is None:
            continue
        ctsize[ct] += sizes.get(str(cl), 0)
        for _, r in sub.iterrows():
            g = str(r["names"])
            sc_ = float(r["scores"])
            if g not in best[ct] or sc_ > best[ct][g]:    # keep the cluster where it's strongest
                best[ct][g] = sc_

    # Ownership of a SHARED gene is decided by the evidence ALONE — it goes to the cell type that
    # scores it strongest — so neither `order` nor abundance (display concerns) can change which
    # markers land in which panel. (A tie keeps the first cluster-iteration encounter, still
    # order-independent.) This keeps `order` a pure LAYOUT knob.
    owner = {}                         # gene -> (cell_type, best score anywhere)
    for ct, gs in best.items():
        for g, s in gs.items():
            if g not in owner or s > owner[g][1]:
                owner[g] = (ct, s)
    owned = defaultdict(dict)          # cell type -> {gene it owns: score}
    for g, (ct, s) in owner.items():
        owned[ct][g] = s

    if order:
        pos = {ct: i for i, ct in enumerate(order)}
        ct_iter = sorted(owned, key=lambda c: pos.get(c, len(order)))     # LAYOUT only
    else:
        ct_iter = sorted(owned, key=lambda c: -ctsize[c])

    panels = {}
    for ct in ct_iter:
        genes = [g for g, _ in sorted(owned[ct].items(), key=lambda kv: -kv[1])][:top_k]
        if genes:
            panels[ct] = genes
    return panels


# =========================================================================== #
# Tier 3 — fine annotation (B13): evidence → LLM → apply (marker-DB-free)
# =========================================================================== #
# Same two-step split as Tier-1/Tier-2: a deterministic tool packages per-SUBCLUSTER
# evidence WITHIN a compartment (from compartment_subset → cluster → markers), the LLM
# infers fine_cell_type + a FACS-style display label from the DE itself (no fixed panel),
# and apply_fine_annotation records the calls with deterministic HARD RULES (tiny-cluster
# merge, insufficient-evidence → review). Evidence + authority live in
# uns['scpilot_annotation']['annotation_tree']; cell type / state stay in SEPARATE columns.


def _fine_score_genes(adata, genes, name):
    """Per-cell sc.tl.score_genes for caller-supplied confounder genes (None-safe, local)."""
    import scanpy as sc
    present = [g for g in (genes or []) if g in adata.var_names]
    if not present:
        return None, []
    sc.tl.score_genes(adata, present, score_name=name, ctrl_size=min(50, max(10, len(present) * 5)))
    return adata.obs[name].astype(float), present


@register("fine_annotation_review", mutating=False,
          description="Tier-3 FINE annotation EVIDENCE (read-only, NO fixed marker panel). Run WITHIN a "
                      "compartment subset (compartment_subset → cluster → markers): per subcluster it packages the "
                      "top-N SIGNIFICANT ranked DE (logFC/padj/pct), size, sample distribution + single-patient "
                      "dominance, the dominant parent compartment (major_cell_type) and malignancy composition (if "
                      "present), and CONFOUNDER signals — existing obs score columns (cell_cycle/stress/IFN/"
                      "activation/doublet) plus OPTIONAL caller confounder_genes scored on the fly (no built-in "
                      "panel). The LLM infers fine_cell_type + a FACS-style display label FROM the DE (see "
                      "FINE_ANNOTATION_PROMPT), then calls apply_fine_annotation.")
def fine_annotation_review(session, *, groupby: str = "leiden", compartment: str | None = None,
                           top_n: int = 40, padj_max: float = 0.05,
                           min_specificity: float = 0.1, max_pct_out: float = 0.5, min_specific_markers: int = 3,
                           sample_key: str = "sample_id",
                           major_key: str = "major_cell_type", malignancy_key: str = "malignancy",
                           confounder_keys: list | None = None, confounder_genes: dict | None = None,
                           single_source_frac: float = 0.8, min_cells: int = 20,
                           max_samples_reported: int = 8, **params) -> S.ToolResult:
    import json

    import numpy as np
    import scanpy as sc

    t0 = time.time()
    adata = session.adata
    if groupby not in adata.obs.columns:
        return S.error("fine_annotation_review", "invalid_state",
                       f"subcluster key '{groupby}' absent — run cluster on the compartment subset first",
                       recoverable=True, suggested_next_tools=["cluster"])
    rg = adata.uns.get("rank_genes_groups")
    if not rg or rg.get("params", {}).get("groupby") != groupby:
        return S.error("fine_annotation_review", "invalid_state",
                       f"per-subcluster DE for '{groupby}' absent — run markers(groupby={groupby}) first",
                       recoverable=True, suggested_next_tools=["markers"])
    de = sc.get.rank_genes_groups_df(adata, group=None)
    if "pct_nz_group" not in de.columns:
        return S.error("fine_annotation_review", "invalid_state",
                       "DE lacks expressed-fraction (pct) — re-run markers (it sets pts=True)",
                       recoverable=True, suggested_next_tools=["markers"])
    if "pvals_adj" in de.columns:
        de = de[de["pvals_adj"] < padj_max]
    # SPECIFICITY axis (spec = pct_in - pct_out): within a compartment the broadly-shared epithelial
    # genes (and ambient MALAT1/ribosomal) are NOT identity evidence; the surfaced list is significant
    # AND specific so fine subtypes are told apart by genes they actually concentrate.
    has_spec = {"pct_nz_group", "pct_nz_reference"}.issubset(de.columns)
    if has_spec:
        de = de.assign(spec=(de["pct_nz_group"] - de["pct_nz_reference"]).round(4))

    de_cols = {"names": "gene", "logfoldchanges": "logFC", "pvals_adj": "padj",
               "pct_nz_group": "pct_in", "pct_nz_reference": "pct_out", "spec": "spec", "scores": "score"}
    present_cols = [c for c in de_cols if c in de.columns]

    obs_g = adata.obs[groupby].astype(str)
    has_sample = sample_key in adata.obs.columns
    has_major = major_key in adata.obs.columns
    has_malig = malignancy_key in adata.obs.columns
    # overall compartment context: the parent major label when the subset is one compartment
    overall_comp = compartment
    if overall_comp is None and has_major:
        majs = adata.obs[major_key].astype(str).unique()
        overall_comp = str(majs[0]) if len(majs) == 1 else None

    # confounder evidence — READ existing upstream score columns (numeric); never a gene panel.
    ckeys = list(confounder_keys) if confounder_keys is not None else list(CONFOUNDER_KEYS)
    conf_cols = [k for k in ckeys if k in adata.obs.columns
                 and np.issubdtype(adata.obs[k].dtype, np.number)]
    has_dbl = FINE_DOUBLET_KEY in adata.obs.columns
    # OPTIONAL caller-supplied confounder gene sets scored on the fly (no built-in panel)
    scored, scored_used, scratch = {}, {}, []
    for sname, genes in (confounder_genes or {}).items():
        col = f"_fine_conf_{sname}"
        s, used = _fine_score_genes(adata, genes, col)
        if s is not None:
            scored[sname] = s.values
            scored_used[sname] = used
            scratch.append(col)

    payloads, rows = [], []
    for cl in sorted(obs_g.unique()):
        cl = str(cl)
        sub = de[de["group"].astype(str) == cl]
        # significant AND specific (delta + out-group ceiling), most-specific first.
        spec_sub = sub[(sub["spec"] >= min_specificity) & (sub["pct_nz_reference"] <= max_pct_out)] \
            .sort_values("spec", ascending=False) if has_spec else sub
        n_specific = int(len(spec_sub))
        top = spec_sub.head(top_n)
        de_table = [{de_cols[c]: (round(float(r[c]), 4) if c != "names" else str(r[c]))
                     for c in present_cols} for _, r in top.iterrows()]
        mask = (obs_g == cl).values
        n_cells = int(mask.sum())

        ev: dict = {"subcluster": cl, "n_cells": n_cells, "n_significant_markers": int(len(sub)),
                    "n_specific_markers": n_specific, "de_table": de_table}
        if has_major:
            mv = adata.obs[major_key].astype(str)[mask].value_counts(normalize=True)
            ev["compartment"] = str(mv.index[0])
            ev["compartment_purity"] = round(float(mv.iloc[0]), 3)
        if has_malig:
            mc = adata.obs[malignancy_key].astype(str)[mask].value_counts(normalize=True)
            ev["malignancy_composition"] = {str(k): round(float(v), 3) for k, v in mc.head(4).items()}
        if has_sample:
            sv = adata.obs[sample_key].astype(str)[mask].value_counts(normalize=True)
            ev["sample_distribution"] = {str(k): round(float(v), 3) for k, v in sv.head(max_samples_reported).items()}
            ev["single_patient_dominated"] = bool(sv.iloc[0] >= single_source_frac)
        conf = {}
        for k in conf_cols:
            conf[k] = round(float(np.nanmean(adata.obs[k].values[mask])), 4)
        if has_dbl:
            conf["doublet_frac"] = round(float(np.asarray(adata.obs[FINE_DOUBLET_KEY].values[mask],
                                                          dtype=float).mean()), 3)
        for sname, vals in scored.items():
            conf[sname] = round(float(np.nanmean(vals[mask])), 4)
        if conf:
            ev["confounders"] = conf
        flags = []
        if n_cells < min_cells:
            flags.append("tiny_subcluster")
        if n_specific < min_specific_markers:
            flags.append("few_specific_markers")     # no distinct identity (low-quality/ambient/doublet)
        if flags:
            ev["flag"] = ",".join(flags)
        payloads.append(ev)
        rows.append({"subcluster": cl, "n_cells": n_cells,
                     "compartment": ev.get("compartment"),
                     "n_sig_markers": int(len(sub)), "n_specific": n_specific,
                     "single_patient": ev.get("single_patient_dominated"),
                     "top_markers": ", ".join(d["gene"] for d in de_table[:6])})

    for c in scratch:                          # read-only contract: drop scratch score columns
        if c in adata.obs.columns:
            del adata.obs[c]

    import pandas as pd
    df = pd.DataFrame(rows)
    art_dir = session.artifacts_dir
    art_dir.mkdir(parents=True, exist_ok=True)
    json_path = art_dir / "fine_annotation_evidence.json"
    json_path.write_text(json.dumps({
        "groupby": groupby, "compartment": overall_comp,
        "padj_max": padj_max,
        "specificity_filter": f"(pct_in - pct_out) >= {min_specificity} AND pct_out <= {max_pct_out}",
        "confounder_keys_used": conf_cols,
        "confounder_genes_used": scored_used,
        "instruction": "de_table = markers that are SIGNIFICANT (padj) AND SPECIFIC (spec=pct_in-pct_out, "
                       "most-specific first); 'few_specific_markers' flags a subcluster with no distinct "
                       "identity (low-quality/ambient/doublet). Within this compartment, infer each subcluster's "
                       "fine_cell_type FROM the ranked DE (no marker database) and a FACS-style display label "
                       "(e.g. 'CD8+ PD-1+ T cells'). Keep "
                       "cell TYPE separate from cell STATE. Weigh confounders (cell-cycle/stress/IFN/activation/"
                       "doublet, single-patient dominance) — a state program (cycling/stressed) is NOT a cell "
                       "type. Provide evidence_for / evidence_against / confounders per subcluster + confidence "
                       "[0,1] + review_required. Tiny subclusters (n_cells < merge floor) and calls with no "
                       "evidence_for are auto-flagged for review by apply_fine_annotation. Then call "
                       "apply_fine_annotation(groupby, fine_labels, facs_labels, ...).",
        "subclusters": payloads}, indent=2, default=str))

    warnings = []
    if not has_major:
        warnings.append(f"major_cell_type key '{major_key}' absent — no compartment-purity context.")
    if not has_sample:
        warnings.append(f"sample_key '{sample_key}' absent — no single-patient-dominance signal.")
    if not conf_cols and not scored:
        warnings.append("no confounder score columns found and none supplied — state/confound signals limited "
                        "(pass confounder_genes={name:[genes]} to score cycle/stress/IFN on the fly).")

    summary = {
        "groupby": groupby, "compartment": overall_comp, "n_subclusters": len(payloads),
        "specificity_filter": f"(pct_in - pct_out) >= {min_specificity} AND pct_out <= {max_pct_out}",
        "min_specificity": min_specificity, "max_pct_out": max_pct_out,
        "confounder_keys_used": conf_cols, "confounder_genes_used": scored_used,
        "evidence_input": str(json_path),
        "note": "Evidence only — no fine call. de_table is SIGNIFICANT AND SPECIFIC markers (spec=pct_in-pct_out, "
                "most-specific first); the LLM infers fine_cell_type + FACS label from the DE, then writes them "
                "via apply_fine_annotation (tiny-cluster merge + insufficient-evidence enforced).",
    }
    return S.success("fine_annotation_review", summary=summary,
                     tables={"subclusters": S.table_preview(df, max_rows=len(df))},
                     artifacts=[S.Artifact(path=str(json_path), kind="json",
                                           description="per-subcluster fine-annotation evidence for the LLM")],
                     warnings=warnings, determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["apply_fine_annotation"])


@register("apply_fine_annotation", mutating=True,
          description="Write the LLM's Tier-3 FINE calls: obs['fine_cell_type'] + obs['facs_style_label'] (display, "
                      "e.g. 'CD8+ PD-1+ T cells') + optional obs['cell_state'], keyed on the subcluster map the LLM "
                      "inferred from fine_annotation_review. Cell TYPE and STATE stay in separate columns. "
                      "Deterministic HARD RULES: a subcluster below merge_min_cells is MERGED to a fallback label "
                      "(<compartment>_unresolved) + review_required; a call with no evidence_for is forced "
                      "review_required. Per-subcluster authority + evidence_for/against + confounders are recorded "
                      "in uns['scpilot_annotation']['annotation_tree'].")
def apply_fine_annotation(session, *, groupby: str = "leiden", fine_labels: dict | None = None,
                          facs_labels: dict | None = None, cell_state: dict | None = None,
                          confidence: dict | None = None, review_required: dict | None = None,
                          evidence_for: dict | None = None, evidence_against: dict | None = None,
                          confounders: dict | None = None, compartment: str | None = None,
                          major_key: str = "major_cell_type", fine_key: str = "fine_cell_type",
                          facs_key: str = "facs_style_label", merge_min_cells: int = 20,
                          min_evidence: int = 1, merge_label: str | None = None,
                          method: str = "DE_LLM_fine_marker_free", unassigned: str = "Unassigned",
                          **params) -> S.ToolResult:
    t0 = time.time()
    adata = session.adata
    if groupby not in adata.obs.columns:
        return S.error("apply_fine_annotation", "invalid_state",
                       f"subcluster key '{groupby}' absent — run cluster first", recoverable=True,
                       suggested_next_tools=["cluster"])
    if not fine_labels:
        return S.error("apply_fine_annotation", "missing_input",
                       "no 'fine_labels' map given (expected {subcluster_id: fine_cell_type} from the LLM)",
                       recoverable=True, suggested_next_tools=["fine_annotation_review"])

    obs_g = adata.obs[groupby].astype(str)
    sizes = obs_g.value_counts().to_dict()
    clusters = sorted(obs_g.unique())
    fine = {str(k): str(v) for k, v in fine_labels.items()}
    facs = {str(k): str(v) for k, v in (facs_labels or {}).items()}
    state = {str(k): str(v) for k, v in (cell_state or {}).items()}
    cf = {str(k): float(v) for k, v in (confidence or {}).items()}
    rv = {str(k): bool(v) for k, v in (review_required or {}).items()}
    ef = {str(k): list(v) for k, v in (evidence_for or {}).items()}
    eg = {str(k): list(v) for k, v in (evidence_against or {}).items()}
    conf_map = {str(k): list(v) for k, v in (confounders or {}).items()}

    has_major = major_key in adata.obs.columns
    # global compartment for the merge fallback label (parent major, when the subset is uniform)
    overall_comp = compartment
    if overall_comp is None and has_major:
        majs = adata.obs[major_key].astype(str).unique()
        overall_comp = str(majs[0]) if len(majs) == 1 else None
    fallback = merge_label or (f"{overall_comp}_{FINE_UNRESOLVED}" if overall_comp else FINE_UNRESOLVED)

    fine_final, facs_final, state_final, conf_final, review_final = {}, {}, {}, {}, {}
    tree: dict = {}
    n_merged = n_insufficient = 0
    for cl in clusters:
        n_cells = int(sizes.get(cl, 0))
        proposed = fine.get(cl)
        reasons: list[str] = []
        merged = False
        forced_review = False

        if proposed is None:
            final = unassigned
            reasons.append("not_labeled")
            forced_review = True
        elif n_cells < merge_min_cells:                 # HARD RULE: tiny subcluster → merge + review
            final = fallback
            merged = True
            forced_review = True
            n_merged += 1
            reasons.append(f"below_merge_min_cells(<{merge_min_cells})")
        else:
            final = proposed

        ev_for = ef.get(cl, [])
        if proposed is not None and len(ev_for) < min_evidence:   # HARD RULE: no evidence → review
            forced_review = True
            n_insufficient += 1
            reasons.append("insufficient_evidence")

        review = bool(rv.get(cl, False) or forced_review)
        # per-subcluster compartment context
        comp_cl = overall_comp
        if has_major:
            comp_cl = str(adata.obs[major_key].astype(str)[(obs_g == cl).values].value_counts().index[0])

        fine_final[cl] = final
        facs_final[cl] = facs.get(cl, "")
        state_final[cl] = state.get(cl, "")
        conf_final[cl] = cf.get(cl, float("nan"))
        review_final[cl] = review
        tree[cl] = {
            "n_cells": n_cells, "major_cell_type": comp_cl,
            "fine_cell_type": final, "proposed_fine_cell_type": proposed,
            "facs_style_label": facs.get(cl, ""), "cell_state": state.get(cl, ""),
            "confidence": cf.get(cl), "review_required": review, "merged": merged,
            "evidence_for": ev_for, "evidence_against": eg.get(cl, []),
            "confounders": conf_map.get(cl, []), "review_reasons": reasons,
        }

    adata.obs[fine_key] = obs_g.map(lambda c: fine_final.get(c, unassigned)).astype("category")
    adata.obs[facs_key] = obs_g.map(lambda c: facs_final.get(c, "")).astype("category")
    adata.obs[f"{fine_key}_confidence"] = obs_g.map(lambda c: conf_final.get(c, float("nan"))).astype(float)
    adata.obs[f"{fine_key}_review_required"] = obs_g.map(lambda c: review_final.get(c, False)).astype(bool)
    if any(state_final.values()):
        adata.obs["cell_state"] = obs_g.map(lambda c: state_final.get(c, "")).astype("category")

    adata.uns.setdefault(UNS_ANNO, {})
    adata.uns[UNS_ANNO][UNS_TREE] = {
        "method": method, "groupby": groupby, "compartment": overall_comp,
        "fine_key": fine_key, "facs_key": facs_key, "marker_db_used": False,
        "merge_min_cells": merge_min_cells, "merge_label": fallback, "min_evidence": min_evidence,
        "subclusters": tree,
    }
    try:
        session.log_decision(S.DecisionEvent(
            decision_type="fine_llm_labels",
            choice={cl: fine_final[cl] for cl in clusters}, candidates=[],
            rationale=f"DE-based marker-free Tier-3 fine labels (compartment={overall_comp})",
            stage="apply_fine_annotation", params={"groupby": groupby, "fine_key": fine_key}).to_dict())
    except Exception:  # noqa: BLE001
        pass

    dist = adata.obs[fine_key].value_counts().to_dict()
    n_review = int(sum(1 for v in review_final.values() if v))
    n_unassigned = int((adata.obs[fine_key].astype(str) == unassigned).sum())
    summary = {
        "fine_key": fine_key, "facs_key": facs_key, "groupby": groupby, "compartment": overall_comp,
        "method": method, "marker_db_used": False, "n_subclusters": len(clusters),
        "n_merged_subclusters": n_merged, "merge_label": fallback,
        "n_insufficient_evidence": n_insufficient, "n_review_required_subclusters": n_review,
        "n_unassigned_cells": n_unassigned,
        "label_distribution": {str(k): int(v) for k, v in dist.items()},
    }
    warnings = []
    if n_merged:
        warnings.append(f"{n_merged} subcluster(s) below merge_min_cells={merge_min_cells} → '{fallback}' + review")
    if n_insufficient:
        warnings.append(f"{n_insufficient} subcluster(s) had no evidence_for → review_required forced")
    cp = session.checkpoint("apply_fine_annotation", x_state=session.manifest.x_state,
                            params={"groupby": groupby, "fine_key": fine_key, "method": method})
    return S.success("apply_fine_annotation", summary=summary, warnings=warnings, checkpoint=cp.path,
                     determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["plots", "trajectory", "report"])
