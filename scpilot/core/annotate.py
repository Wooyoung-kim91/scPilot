"""Annotation — Tier 1 broad (B8); Tier 3 fine lands later (B13).

Tier 1 (``annotate_broad``) assigns ``obs['major_cell_type']`` — the broad
compartment that becomes the scib benchmark ``label_key``.

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
# non-biological labels — excluded from the scib bio-conservation label set downstream.
ARTIFACT_LABELS = {UNKNOWN_LABEL, MIXED_LABEL, LOWQ_LABEL}
UNS_ANNO = "scpilot_annotation"


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
          description="Tier 1 broad cell type → obs['major_cell_type'] via leiden-cluster DE markers "
                      "(pct>=0.25 & LFC>=1, top-30 per cluster) matched to cell-type panels (>=3 markers), "
                      "with negative-marker tie-break, Mixed/Artifact + Low_quality gating, PTPRC consistency, "
                      "and sample-provenance flags (plan B8).")
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
                     suggested_next_tools=["annotation_review", "plots", "integrate_scvi", "benchmark"])


# review_status thresholds (deterministic baseline; the LLM reviewer refines these).
_REVIEW_CONF_MIN = 0.5


@register("annotation_review", mutating=False,
          description="Package per-cluster Tier-1 review EVIDENCE for an LLM reviewer (read-only, no LLM "
                      "call here): full top-N ranked DE (logFC/padj/pct_in/pct_out), cluster size, sample "
                      "distribution, QC, the candidate major_cell_type + deterministic rule flags, and a "
                      "deterministic review_status baseline. The LLM (host agent / mode-2) then infers "
                      "transcriptional programs from the raw DE WITHOUT a marker database, detects "
                      "conflicting programs / artifacts, and adjusts confidence — see llm/prompts.py "
                      "ANNOTATION_REVIEW_PROMPT. Pass tissue= (e.g. 'human pancreas, PDAC') as a soft "
                      "prior so the reviewer flags tissue-implausible calls. Run after annotate_broad (plan B8).")
def annotation_review(session, *, top_n: int = 50, sample_key: str = "sample_id",
                      conf_min: float = _REVIEW_CONF_MIN, max_samples_reported: int = 8,
                      tissue: str | None = None, **params) -> S.ToolResult:
    """Deterministic evidence packager — the boundary the plan draws between replayable
    tools (this) and non-deterministic LLM reasoning (the reviewer). It re-uses the
    ``rank_genes_groups`` and ``scpilot_annotation['tier1']`` that ``annotate_broad``
    already produced (no DE recompute), so it is cheap and adds no analysis decisions of
    its own beyond a transparent ``review_status`` heuristic over the existing flags."""
    import json

    import numpy as np
    import scanpy as sc

    t0 = time.time()
    adata = session.adata
    if "major_cell_type" not in adata.obs.columns or UNS_ANNO not in adata.uns:
        return S.error("annotation_review", "invalid_state",
                       "no Tier-1 annotation — run annotate_broad first", recoverable=True,
                       suggested_next_tools=["annotate_broad"])
    tier1 = adata.uns[UNS_ANNO].get("tier1")
    if not tier1 or "rank_genes_groups" not in adata.uns:
        return S.error("annotation_review", "invalid_state",
                       "annotate_broad evidence (tier1 + rank_genes_groups) absent — re-run annotate_broad",
                       recoverable=True, suggested_next_tools=["annotate_broad"])

    groupby = tier1.get("groupby", "leiden")
    de = sc.get.rank_genes_groups_df(adata, group=None)
    de_cols = {"names": "gene", "logfoldchanges": "logFC", "pvals_adj": "padj",
               "pct_nz_group": "pct_in", "pct_nz_reference": "pct_out", "scores": "score"}
    present_cols = [c for c in de_cols if c in de.columns]
    obs_g = adata.obs[groupby].astype(str)
    has_sample = sample_key in adata.obs.columns
    clusters_ev = tier1.get("clusters", {})

    payloads, rows = [], []
    status_counts = {"ok": 0, "low_confidence": 0, "artifact_suspected": 0}
    for cl, sub in de.groupby("group", observed=True):
        cl = str(cl)
        ev = clusters_ev.get(cl, {})
        top = sub.head(top_n)
        de_table = [{de_cols[c]: (round(float(r[c]), 4) if c != "names" else str(r[c]))
                     for c in present_cols} for _, r in top.iterrows()]

        mask = (obs_g == cl).values
        sample_dist = {}
        if has_sample:
            sv = adata.obs[sample_key].astype(str)[mask].value_counts(normalize=True)
            sample_dist = {str(k): round(float(v), 3) for k, v in sv.head(max_samples_reported).items()}

        label = ev.get("label", "Unknown")
        conf = float(ev.get("confidence", 0.0))
        flags = {
            "marker_conflict": bool(ev.get("marker_conflict", False)),
            "doublet_dominated": bool(ev.get("doublet_dominated", False)),
            "ptprc_consistent": ev.get("ptprc_consistent"),
            "single_source": bool(ev.get("single_source", False)),
        }
        # deterministic review_status baseline (the LLM reviewer refines it). Mixed/Artifact
        # and Low_quality (and the conflict/doublet flags) are genuine artifact risk; plain
        # Unknown is insufficient-evidence → low_confidence (the LLM's program inference may
        # still resolve it), not an artifact.
        risk = []
        if label in (MIXED_LABEL, LOWQ_LABEL) or flags["marker_conflict"] or flags["doublet_dominated"]:
            status = "artifact_suspected"
            risk = [k for k in ("marker_conflict", "doublet_dominated") if flags[k]] or [f"label={label}"]
        elif label == UNKNOWN_LABEL or conf < conf_min or flags["ptprc_consistent"] is False \
                or flags["single_source"]:
            status = "low_confidence"
            risk = (["unknown"] if label == UNKNOWN_LABEL else []) \
                + ([f"confidence<{conf_min}"] if (label != UNKNOWN_LABEL and conf < conf_min) else []) \
                + (["ptprc_inconsistent"] if flags["ptprc_consistent"] is False else []) \
                + (["single_source"] if flags["single_source"] else [])
        else:
            status = "ok"
        status_counts[status] += 1

        payloads.append({
            "cluster_id": cl, "candidate_annotation": label, "candidate_confidence": conf,
            "cluster_size": int(mask.sum()), "deterministic_flags": flags,
            "review_status": status, "risk_signals": risk,
            "sample_distribution": sample_dist, "qc_metrics": ev.get("qc", {}),
            "de_table": de_table,
        })
        rows.append({"cluster": cl, "candidate": label, "confidence": conf,
                     "review_status": status, "risk_signals": ",".join(risk),
                     "top_de": ",".join(d["gene"] for d in de_table[:6])})

    art_dir = session.artifacts_dir
    art_dir.mkdir(parents=True, exist_ok=True)
    json_path = art_dir / "annotation_review.json"
    json_path.write_text(json.dumps(
        {"groupby": groupby, "top_n": top_n, "label_key": "major_cell_type",
         "tissue_context": tissue, "marker_db_used_for_review": False,
         "clusters": payloads}, indent=2, default=str))

    import pandas as pd
    table = S.table_preview(pd.DataFrame(rows), max_rows=len(rows))
    flagged = [p["cluster_id"] for p in payloads if p["review_status"] != "ok"]
    summary = {
        "groupby": groupby, "top_n": top_n, "n_clusters": len(payloads),
        "tissue_context": tissue,
        "status_counts": status_counts, "flagged_clusters": flagged,
        "review_input": str(json_path),
        "note": "LLM reviewer infers programs from de_table WITHOUT a marker DB, but uses "
                "tissue_context as a soft prior (flags tissue-implausible calls). See "
                "llm/prompts.py ANNOTATION_REVIEW_PROMPT + TISSUE_CONTEXT_GUIDANCE. "
                "review_status here is the deterministic baseline.",
    }
    warnings = [] if not flagged else [f"{len(flagged)} cluster(s) flagged for review: {flagged}"]
    return S.success("annotation_review", summary=summary, tables={"review": table},
                     artifacts=[S.Artifact(path=str(json_path), kind="json",
                                           description="per-cluster Tier-1 review evidence (LLM reviewer input)")],
                     warnings=warnings, determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["plots", "integrate_scvi"])
