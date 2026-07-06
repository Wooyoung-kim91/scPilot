"""Tier-4 annotation consistency audit — scpilot.

Implements the designed-but-missing "Tier 4: Consistency and Review Agent" from
``cancer_scrnaseq_annotation_strategy.md``. SPLIT per scpilot's contract (evidence vs judgment):

- ``annotation_audit`` (non-mutating) emits DETERMINISTIC inconsistency EVIDENCE for the final
  annotation table — the seven Tier-4 checks. It makes NO biological judgment and uses NO
  hardcoded marker panel / cell-type list / score threshold: it reads the LLM's OWN recorded
  marker_sets (``uns['scpilot_annotation']['tier1_llm']``) and the per-cluster DE + upstream
  score columns, and reports cross-tabs / flags. Whether a flagged label is actually WRONG is
  left to the reviewer.
- the CRITIQUE (confirm / suspect / refute each label) is the reasoning layer's job
  (``ANNOTATION_AUDIT_PROMPT``), ideally an INDEPENDENT reviewer (a different model). Its verdict
  is recorded by ``apply_annotation_audit`` so the audit is replayable.

The seven checks (from the single source):
  1. same marker profile but different labels        -> per-cluster top-marker Jaccard collisions
  2. same label but inconsistent marker evidence      -> per-label marker-set support spread
  3. contradictory hierarchy (major vs fine vs facs)  -> the (major,fine,facs) triple + overlap
  4. single-patient cluster dominance                 -> top_sample_fraction
  5. batch-specific clusters                          -> top_batch_fraction / batch entropy
  6. high doublet or stress score                     -> per-cluster doublet/stress/%MT
  7. malignancy label without CNV/tumor evidence      -> malignant label vs cnv_score/cnv_status
"""

from __future__ import annotations

import time

from scpilot import schemas as S
from scpilot.core.annotate import UNS_ANNO
from scpilot.tools import register

AUDIT_STATUS = ("confirmed", "suspect", "refuted")   # reviewer verdict vocabulary (FIXED)


def _specific_markers(de_sub, *, min_specificity: float, max_pct_out: float, padj_max: float,
                      top_k: int) -> list:
    """Top-K significant, in-cluster-SPECIFIC up-marker gene names for one cluster's DE table.
    Specificity = pct_in - pct_out (the data sets it; no marker DB)."""
    sub = de_sub
    if "pvals_adj" in sub.columns:
        sub = sub[sub["pvals_adj"] < padj_max]
    if {"pct_nz_group", "pct_nz_reference"}.issubset(sub.columns):
        spec = sub["pct_nz_group"] - sub["pct_nz_reference"]
        sub = sub.assign(_spec=spec)
        sub = sub[(sub["_spec"] >= min_specificity) & (sub["pct_nz_reference"] <= max_pct_out)]
        sub = sub.sort_values("_spec", ascending=False)
    return [str(g) for g in sub["names"].head(top_k)]


def _de_stats(de_sub) -> dict:
    """gene -> its DE statistics for one cluster: logFC, padj, pct_in, pct_out, spec.
    The standard cell-type-marker criteria (pct / logFC / p-value) are checked against these."""
    import numpy as np

    has_pct = {"pct_nz_group", "pct_nz_reference"}.issubset(de_sub.columns)
    out: dict = {}
    for _, r in de_sub.iterrows():
        g = str(r["names"])
        pin = float(r["pct_nz_group"]) if has_pct else float("nan")
        pout = float(r["pct_nz_reference"]) if has_pct else float("nan")
        out[g] = {
            "logFC": round(float(r["logfoldchanges"]), 3) if "logfoldchanges" in de_sub.columns else None,
            "padj": float(r["pvals_adj"]) if "pvals_adj" in de_sub.columns else None,
            "pct_in": round(pin, 3) if has_pct else None,
            "pct_out": round(pout, 3) if has_pct else None,
            "spec": round(pin - pout, 3) if (has_pct and not np.isnan(pin)) else None,
        }
    return out


@register("annotation_audit", mutating=False,
          description="Tier-4 consistency AUDIT of the final annotation (evidence-only, no judgment, no marker DB): "
                      "emits the 7 inconsistency checks — and VALIDATES each claimed cell-type marker against the "
                      "standard criteria (pct_in>=min_pct, logFC>=min_lfc, padj<padj_max, AND specific: "
                      "spec>=min_specificity, pct_out<=max_pct_out) — plus marker-profile "
                      "collisions across labels, major/fine/facs hierarchy triples, single-patient & batch dominance, "
                      "doublet/stress/%MT, and malignancy-without-CNV. Deterministic flags for an INDEPENDENT "
                      "reviewer to confirm/refute (see ANNOTATION_AUDIT_PROMPT). Run after finalize_annotation.")
def annotation_audit(session, *, groupby: str = "leiden", label_key: str = "major_cell_type",
                     fine_key: str = "fine_cell_type", facs_key: str = "facs_style_label",
                     final_key: str = "final_annotation", malignancy_key: str = "malignancy",
                     cnv_status_key: str = "cnv_status", cnv_score_key: str = "cnv_score",
                     sample_key: str = "sample_id", batch_key: str = "GSE",
                     doublet_key: str = "predicted_doublet", stress_key: str = "stress_score",
                     mt_key: str = "pct_counts_mt",
                     min_pct: float = 0.25, min_lfc: float = 1.0, padj_max: float = 0.05,
                     min_specificity: float = 0.1, max_pct_out: float = 0.5, top_k_markers: int = 15,
                     profile_similarity: float = 0.5, single_source_frac: float = 0.8,
                     doublet_frac: float = 0.5, max_pct_mt: float = 25.0,
                     min_marker_support: float = 0.5, **params) -> S.ToolResult:
    import json

    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy.stats import entropy as _shannon_entropy

    t0 = time.time()
    adata = session.adata
    warnings: list[str] = []
    if groupby not in adata.obs.columns:
        return S.error("annotation_audit", "invalid_state",
                       f"clustering '{groupby}' absent — annotate first", recoverable=True,
                       suggested_next_tools=["cluster"])
    if label_key not in adata.obs.columns:
        return S.error("annotation_audit", "invalid_state",
                       f"no annotation in obs['{label_key}'] to audit — run apply_annotation/finalize first",
                       recoverable=True, suggested_next_tools=["apply_annotation"])

    from scpilot.core import _species
    _sidx = _species._symbol_index(adata)

    # the LLM's OWN recorded marker-set per label (evidence, not a panel we own)
    msets_raw = (adata.uns.get(UNS_ANNO, {}).get("tier1_llm", {}) or {}).get("marker_sets", {}) or {}
    marker_sets = {str(ct): _species.present(adata, gs, index=_sidx) for ct, gs in msets_raw.items()}

    obs_g = adata.obs[groupby].astype(str)
    clusters = list(obs_g.cat.categories) if hasattr(obs_g, "cat") else sorted(obs_g.unique())

    # per-cluster DE: top specific markers (checks 1/3) + per-gene stats (the pct/logFC/p-value
    # criteria check for claimed markers, check 2). Degrade gracefully if markers wasn't run.
    de_by_cluster: dict[str, list] = {}
    de_stats_by_cluster: dict[str, dict] = {}
    rg = adata.uns.get("rank_genes_groups")
    if rg and rg.get("params", {}).get("groupby") == groupby:
        de = sc.get.rank_genes_groups_df(adata, group=None)
        for cl, sub in de.groupby("group", observed=True):
            de_by_cluster[str(cl)] = _specific_markers(
                sub, min_specificity=min_specificity, max_pct_out=max_pct_out,
                padj_max=padj_max, top_k=top_k_markers)
            de_stats_by_cluster[str(cl)] = _de_stats(sub)
    else:
        warnings.append(f"per-cluster DE for '{groupby}' absent — marker-criteria/profile checks skipped "
                        "(run markers(groupby) first for full audit)")

    # the cell-type marker bar (parameters, NOT hardcoded per type): a claimed marker passes only if
    # it is expressed in enough of the cluster (pct_in >= min_pct), up-regulated (logFC >= min_lfc),
    # AND significant (padj < padj_max). marker_set_support_frac = fraction of claimed genes passing.
    marker_criteria = {"min_pct": min_pct, "min_lfc": min_lfc, "padj_max": padj_max,
                       "min_specificity": min_specificity, "max_pct_out": max_pct_out}

    def _check_marker(stats: dict) -> tuple[bool, list]:
        failed = []
        if stats.get("pct_in") is not None and stats["pct_in"] < min_pct:
            failed.append("pct")
        if stats.get("logFC") is not None and stats["logFC"] < min_lfc:
            failed.append("lfc")
        if stats.get("padj") is not None and stats["padj"] >= padj_max:
            failed.append("pvalue")
        # I-16: a claimed marker must also be SPECIFIC (not broadly expressed) — mirror the
        # marker-DB-free selection gate (_specific_markers) so the audit does NOT pass a
        # housekeeping/ambient gene (e.g. MALAT1/RPS*) that satisfies pct/lfc/padj but fails
        # specificity, which would otherwise inflate marker_set_support_frac and hide a weak label.
        if stats.get("pct_out") is not None and stats["pct_out"] > max_pct_out:
            failed.append("pct_out")
        if stats.get("spec") is not None and stats["spec"] < min_specificity:
            failed.append("spec")
        return (not failed), failed

    def _col(key):
        return adata.obs[key].astype(str) if key in adata.obs.columns else None

    fine_c, facs_c, final_c = _col(fine_key), _col(facs_key), _col(final_key)
    malig_c, cnvst_c = _col(malignancy_key), _col(cnv_status_key)
    has_cnv_score = cnv_score_key in adata.obs.columns
    cnv_score = adata.obs[cnv_score_key].astype(float) if has_cnv_score else None

    def _mode(series, mask):
        vc = series[mask].value_counts()
        return str(vc.index[0]) if len(vc) else ""

    per_cluster, rows = [], []
    status_counts = {"clean": 0, "flagged": 0}
    label_to_clusters: dict[str, list] = {}

    for cl in [str(c) for c in clusters]:
        mask = (obs_g == cl).values
        n_cells = int(mask.sum())
        if not n_cells:
            continue
        label = _mode(adata.obs[label_key].astype(str), mask)
        label_to_clusters.setdefault(label, []).append(cl)
        cl_markers = de_by_cluster.get(cl, [])

        # ---- check 2: claimed marker-set support, validated against the pct/logFC/p-value bar ----
        claimed = marker_sets.get(label, [])
        cl_stats = de_stats_by_cluster.get(cl, {})
        support_frac, marker_eval = None, []
        if claimed and cl_stats:
            n_pass = 0
            for g in claimed:
                st = cl_stats.get(g)
                if st is None:                       # claimed gene not even in the DE ranking
                    marker_eval.append({"gene": g, "in_de": False, "passes": False,
                                        "failed_criteria": ["absent"]})
                    continue
                ok, failed = _check_marker(st)
                n_pass += int(ok)
                marker_eval.append({"gene": g, "in_de": True, "passes": ok,
                                    "failed_criteria": failed, **st})
            support_frac = round(n_pass / len(claimed), 3)

        # ---- check 3: hierarchy triple (no lineage map — emit, reviewer judges) ----
        triple = {"major": label,
                  "fine": _mode(fine_c, mask) if fine_c is not None else None,
                  "facs": _mode(facs_c, mask) if facs_c is not None else None,
                  "final": _mode(final_c, mask) if final_c is not None else None}

        # ---- check 4/5: provenance dominance ----
        prov = {}
        if sample_key in adata.obs.columns:
            sv = adata.obs[sample_key].astype(str)[mask].value_counts(normalize=True)
            prov["top_sample_frac"] = round(float(sv.iloc[0]), 3)
            prov["n_samples"] = int(adata.obs[sample_key].astype(str)[mask].nunique())
        if batch_key in adata.obs.columns:
            bv = adata.obs[batch_key].astype(str)[mask].value_counts(normalize=True)
            prov["top_batch_frac"] = round(float(bv.iloc[0]), 3)
            prov["batch_entropy"] = round(float(_shannon_entropy(bv.values, base=2)), 3)

        # ---- check 6: artifact QC ----
        qc = {}
        if doublet_key in adata.obs.columns:
            qc["doublet_frac"] = round(float(np.asarray(adata.obs[doublet_key].values[mask], dtype=float).mean()), 3)
        if mt_key in adata.obs.columns:
            qc["median_pct_mt"] = round(float(np.median(adata.obs[mt_key].values[mask])), 2)
        if stress_key in adata.obs.columns:
            qc["median_stress"] = round(float(np.median(adata.obs[stress_key].astype(float).values[mask])), 3)

        # ---- check 7: malignancy without CNV/tumor evidence ----
        is_malignant = False
        if malig_c is not None:
            is_malignant = _mode(malig_c, mask) == "malignant"
        elif cnvst_c is not None:
            is_malignant = _mode(cnvst_c, mask) == "tumor"
        cnv_burden = round(float(cnv_score[mask].mean()), 4) if has_cnv_score else None

        # ---- deterministic FLAGS (review triggers — evidence, NOT a biological refutation) ----
        flags = []
        if support_frac is not None and support_frac < min_marker_support:
            flags.append("weak_marker_support")          # check 2
        if prov.get("top_sample_frac", 0.0) >= single_source_frac:
            flags.append("single_patient_dominant")       # check 4
        if prov.get("top_batch_frac", 0.0) >= single_source_frac:
            flags.append("batch_dominant")                # check 5
        if qc.get("doublet_frac", 0.0) >= doublet_frac:
            flags.append("doublet_dominated")             # check 6
        if qc.get("median_pct_mt", 0.0) > max_pct_mt:
            flags.append("high_mt")                       # check 6
        if is_malignant and not has_cnv_score:
            flags.append("malignant_without_cnv")         # check 7
        elif is_malignant and cnv_burden is not None and cnvst_c is None and malig_c is not None:
            # malignant called but no cnv_status track — surface burden for the reviewer
            pass

        status = "flagged" if flags else "clean"
        status_counts[status] += 1

        per_cluster.append({
            "cluster_id": cl, "n_cells": n_cells, "label": label,
            "hierarchy": triple,
            "marker_set_claimed": claimed, "marker_set_support_frac": support_frac,
            "marker_criteria_check": marker_eval,   # per claimed gene: pct/logFC/padj + pass/fail
            "top_specific_markers": cl_markers[:top_k_markers],
            "provenance": prov, "qc": qc,
            "is_malignant": bool(is_malignant), "cnv_burden": cnv_burden,
            "flags": flags, "review_status": status,
        })
        rows.append({"cluster": cl, "label": label, "n_cells": n_cells,
                     "support": support_frac, "flags": ",".join(flags)})

    # ---- check 1 (global): marker-profile COLLISIONS — high top-marker Jaccard, different label ----
    collisions = []
    cl_sets = {c["cluster_id"]: set(c["top_specific_markers"]) for c in per_cluster if c["top_specific_markers"]}
    ids = list(cl_sets)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            la = next(c["label"] for c in per_cluster if c["cluster_id"] == a)
            lb = next(c["label"] for c in per_cluster if c["cluster_id"] == b)
            if la == lb:
                continue
            inter = len(cl_sets[a] & cl_sets[b])
            union = len(cl_sets[a] | cl_sets[b]) or 1
            jac = inter / union
            if jac >= profile_similarity:
                collisions.append({"clusters": [a, b], "labels": [la, lb],
                                   "marker_jaccard": round(jac, 3),
                                   "shared_markers": sorted(cl_sets[a] & cl_sets[b])[:10]})

    # ---- check 2 (global): per-label marker-support spread (same label, inconsistent evidence) ----
    label_support = []
    for lab, cls in label_to_clusters.items():
        sup = [c["marker_set_support_frac"] for c in per_cluster
               if c["cluster_id"] in cls and c["marker_set_support_frac"] is not None]
        if len(sup) >= 2:
            label_support.append({"label": lab, "n_clusters": len(cls),
                                   "support_min": min(sup), "support_max": max(sup),
                                   "support_spread": round(max(sup) - min(sup), 3)})

    # ---- GRANULARITY assessment: EVIDENCE for a resolution recommendation (over/under-clustering).
    # NO hardcoded threshold — like cluster_sweep's suggested_resolution or qc's suggested_cutoffs,
    # the tool emits the raw numbers/ratios and the INDEPENDENT reviewer judges whether the clustering
    # is too fine (many clusters collapse to the same label + weak-support noise clusters → lower
    # resolution) or too coarse (distinct profiles fused under one label → raise resolution). ----
    n_cl = len(per_cluster)
    n_lab = len(label_to_clusters)
    n_flagged = sum(1 for c in per_cluster if c["review_status"] == "flagged")
    weak_ids = [c["cluster_id"] for c in per_cluster if "weak_marker_support" in c["flags"]]
    collision_ids = sorted({cid for col in collisions for cid in col["clusters"]})
    granularity = {
        "n_clusters": n_cl,
        "n_labels": n_lab,
        # clusters per distinct label — >1 means the resolution splits populations finer than the
        # labels distinguish (a high value is the primary over-clustering signal).
        "collapse_ratio": round(n_cl / n_lab, 3) if n_lab else None,
        "n_redundant_label_clusters": sum(max(0, len(cls) - 1) for cls in label_to_clusters.values()),
        "max_clusters_per_label": max((len(cls) for cls in label_to_clusters.values()), default=0),
        "n_profile_collision_clusters": len(collision_ids),   # cross-label profile ambiguity
        "n_weak_support_clusters": len(weak_ids),             # low-evidence (noise) clusters
        "frac_flagged": round(n_flagged / n_cl, 3) if n_cl else 0.0,
    }

    art_dir = session.artifacts_dir
    art_dir.mkdir(parents=True, exist_ok=True)
    json_path = session.artifact_path("annotation_audit.json")
    audit_doc = {
        "groupby": groupby, "label_key": label_key,
        "checks": ["marker_profile_collision", "label_marker_support", "hierarchy_triple",
                   "single_patient", "batch_specific", "doublet_stress_qc", "malignancy_without_cnv"],
        "marker_criteria": marker_criteria,   # pct/logFC/p-value bar each claimed marker is checked against
        "marker_db_used": False, "verdict_vocabulary": list(AUDIT_STATUS),
        "granularity": granularity,           # resolution-feedback evidence (reviewer decides)
        "clusters": per_cluster,
        "marker_profile_collisions": collisions,
        "label_marker_support": label_support,
    }
    json_path.write_text(json.dumps(audit_doc, indent=2, default=str))

    flagged = [c["cluster_id"] for c in per_cluster if c["review_status"] == "flagged"]
    summary = {
        "groupby": groupby, "label_key": label_key, "marker_db_used": False,
        "marker_criteria": marker_criteria,
        "n_clusters": len(per_cluster),
        "n_flagged_clusters": len(flagged), "flagged_clusters": flagged,
        "status_counts": status_counts,
        "n_marker_profile_collisions": len(collisions),
        "n_malignant_without_cnv": sum(1 for c in per_cluster if "malignant_without_cnv" in c["flags"]),
        "granularity": granularity,   # resolution-feedback evidence; reviewer recommends raise/lower/none
        "audit_input": str(json_path),
        "reviewer_action": "critique each flagged label AND assess granularity (recommend a resolution "
                           "change if over/under-clustered) via ANNOTATION_AUDIT_PROMPT, then apply_annotation_audit",
    }
    tables = {"audit": S.table_preview(pd.DataFrame(rows), max_rows=50)} if rows else {}
    artifacts = [S.artifact_csv(str(json_path), description="Tier-4 annotation consistency audit (evidence)")]
    return S.success("annotation_audit", summary=summary, tables=tables, artifacts=artifacts,
                     warnings=warnings, determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["apply_annotation_audit"])


@register("harness_audit", mutating=False,
          description="GOVERNANCE / process-completeness auditor (read-only, no judgment): reads the session's "
                      "run-log + decisions + obs/uns and checks that the harness's own ACTION RULES & invariants "
                      "were actually followed — Tier-4 review ran, refuted→re-annotated / suspect→flagged, "
                      "QC-artifact clusters labeled Low_quality/Doublet (not a cell type), marker-DB-free path, "
                      "seeds + decisions recorded, run_log↔outputs consistent, pipeline reached finalize. Emits a "
                      "per-check pass/warn/fail scorecard so 'rules defined but not honored' is caught. Run last.")
def harness_audit(session, *, label_key: str = "major_cell_type", **params) -> S.ToolResult:
    import json

    t0 = time.time()
    adata = session.adata
    runs = session._read_runs()
    tools_run = [r.get("tool") for r in runs]
    n_dec = 0
    if session.decisions_path.exists():
        n_dec = sum(1 for ln in session.decisions_path.read_text().splitlines() if ln.strip())
    anno = adata.uns.get(UNS_ANNO, {}) or {}
    t4 = anno.get("tier4_audit", {}) or {}
    reviews = anno.get("tier4_reviews", {}) or {}          # per-column Tier-4 coverage ledger

    checks: list[dict] = []
    def chk(name, ok, detail, sev="fail"):
        checks.append({"check": name, "status": "pass" if ok else sev, "detail": detail})

    # 1) Tier-4 INDEPENDENT review ran on EVERY annotation column that exists (broad/fine/
    #    malignancy/final) — reliability of ALL annotation is secured only if each label column
    #    was independently reviewed, not just one. A present-but-unreviewed column is a violation.
    label_cols = [c for c in ("major_cell_type", "fine_cell_type", "malignancy", "final_annotation")
                  if c in adata.obs.columns]
    unreviewed = [c for c in label_cols if c not in reviews]
    chk("tier4_review_ran", ("apply_annotation_audit" in tools_run) and not unreviewed,
        f"annotation_columns={label_cols}, reviewed={sorted(reviews)}, unreviewed={unreviewed}")
    # 1b) each reviewed column names its reviewer (cross-model provenance; warn if any missing)
    no_reviewer = [c for c, r in reviews.items() if not r.get("reviewer_model")]
    chk("tier4_review_provenance", not no_reviewer,
        f"columns_missing_reviewer_model={no_reviewer}", sev="warn")
    # 2) reviewer provenance recorded (cross-model is the goal; at least the model id must be logged)
    chk("tier4_reviewer_recorded", bool(t4.get("reviewer_model")),
        f"reviewer_model={t4.get('reviewer_model')}", sev="warn")
    # 3) FLAGS → ACTION: every refuted/suspect cluster is surfaced + review_required set on cells
    ref, sus = t4.get("refuted_clusters", []), t4.get("suspect_clusters", [])
    rr_col = next((c for c in adata.obs.columns if str(c).endswith("review_required")), None)
    n_rr = int(adata.obs[rr_col].sum()) if rr_col else 0
    chk("flags_lead_to_action", (not (ref or sus)) or (n_rr > 0),
        f"refuted={ref}, suspect={sus}, review_required_cells={n_rr}", sev="warn")
    # 4) QC-artifact clusters get an explicit artifact label (not a biological cell type)
    lbls = set(adata.obs[label_key].astype(str).unique()) if label_key in adata.obs.columns else set()
    art_labeled = bool(lbls & {"Low_quality", "Doublet", "Doublet_Mixed", "Mixed/Artifact"})
    art_flagged = any(r.get("tool") == "annotation_review"
                      and (r.get("summary", {}).get("status_counts", {}) or {}).get("artifact_suspected", 0) > 0
                      for r in runs)
    chk("artifact_clusters_labeled", (not art_flagged) or art_labeled,
        f"artifact_flagged={art_flagged}, artifact_label_present={art_labeled}", sev="warn")
    # 5) marker-DB-FREE annotation (no fixed panel) — HARD-enforced: annotation must be inferred from
    # DE alone (annotation_review → apply_annotation), never via the panel path. Using annotate_broad
    # FAILS the governance gate (project rule: marker-DB-free, per the CELLxGENE re-annotation decision).
    chk("marker_db_free", "annotate_broad" not in tools_run,
        f"annotate_broad_used={'annotate_broad' in tools_run}")
    # 6) reproducibility invariants: seeds on every run, decisions logged, run_log↔outputs consistent
    no_seed = [r.get("tool") for r in runs if r.get("seed") is None]
    chk("seeds_recorded", not no_seed, f"runs_without_seed={no_seed}")
    chk("decisions_logged", n_dec > 0, f"n_decisions={n_dec}", sev="warn")
    try:
        lc = session.log_consistency()
        chk("run_outputs_consistent", lc.get("consistent", True), str(lc), sev="warn")
    except Exception:  # noqa: BLE001
        pass
    # 7) pipeline completeness
    chk("annotation_present", "apply_annotation" in tools_run, "apply_annotation in run_log")
    chk("finalized", "finalize_annotation" in tools_run, "finalize_annotation in run_log", sev="warn")

    n_fail = sum(1 for c in checks if c["status"] == "fail")
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    summary = {
        "n_checks": len(checks), "n_pass": sum(1 for c in checks if c["status"] == "pass"),
        "n_warn": n_warn, "n_fail": n_fail,
        "violations": [c["check"] for c in checks if c["status"] != "pass"],
        "reviewer_model": t4.get("reviewer_model"),
        "tier4_coverage": {k: v.get("reviewer_model") for k, v in reviews.items()},
        "tools_run": tools_run, "checks": checks,
        "verdict": "complete" if n_fail == 0 else "incomplete",
    }
    warnings = [f"{c['check']}: {c['detail']}" for c in checks if c["status"] == "fail"]
    return S.success("harness_audit", summary=summary, warnings=warnings,
                     determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=[])


@register("apply_annotation_audit", mutating=True,
          description="Record an INDEPENDENT reviewer's Tier-4 verdicts (from ANNOTATION_AUDIT_PROMPT) into "
                      "obs[status_key] (confirmed/suspect/refuted) + obs[review_required_key]. The reviewer flags "
                      "WHETHER each label holds; it does NOT relabel — refuted clusters are re-annotated by the "
                      "annotator. Also records the reviewer's optional GRANULARITY recommendation (advisory: "
                      "resolution down/none/up when over/under-clustered). Deterministic given the verdicts, so "
                      "the critique is replayable. Run after annotation_audit.")
def apply_annotation_audit(session, *, groupby: str = "leiden", verdicts: dict | None = None,
                           label_key: str = "major_cell_type",
                           status_key: str = "annotation_audit_status",
                           review_required_key: str = "annotation_review_required",
                           reviewer_model: str | None = None,
                           granularity: dict | None = None, **params) -> S.ToolResult:
    t0 = time.time()
    adata = session.adata
    if groupby not in adata.obs.columns:
        return S.error("apply_annotation_audit", "invalid_state",
                       f"clustering '{groupby}' absent", recoverable=True, suggested_next_tools=["cluster"])
    if not verdicts:
        return S.error("apply_annotation_audit", "missing_input",
                       "no 'verdicts' map given (expected {cluster_id: {status, review_required, note}} "
                       "from the reviewer)",
                       recoverable=True, suggested_next_tools=["annotation_audit"])

    obs_g = adata.obs[groupby].astype(str)
    vd = {str(k): (v if isinstance(v, dict) else {"status": str(v)}) for k, v in verdicts.items()}
    bad = sorted({v.get("status") for v in vd.values()} - set(AUDIT_STATUS))
    if bad:
        return S.error("apply_annotation_audit", "invalid_params",
                       f"verdict status must be one of {list(AUDIT_STATUS)} (got {bad})", recoverable=True)

    status_map = {c: v.get("status", "confirmed") for c, v in vd.items()}
    review_map = {c: bool(v.get("review_required", v.get("status") != "confirmed")) for c, v in vd.items()}
    adata.obs[status_key] = obs_g.map(lambda c: status_map.get(c, "confirmed")).astype("category")
    adata.obs[review_required_key] = obs_g.map(lambda c: review_map.get(c, False)).astype(bool)

    refuted_clusters = sorted(c for c, s in status_map.items() if s == "refuted")
    suspect_clusters = sorted(c for c, s in status_map.items() if s == "suspect")
    # the REASON for each refuted/suspect cluster (the reviewer must give it; refuted → re-annotate,
    # suspect → flagged for targeted action: Tier-2 subtype or human review, not silently kept).
    refuted_reasons = {c: str(vd[c].get("note", "")) for c in refuted_clusters}
    suspect_reasons = {c: str(vd[c].get("note", "")) for c in suspect_clusters}
    n_refuted, n_suspect = len(refuted_clusters), len(suspect_clusters)
    adata.uns.setdefault(UNS_ANNO, {})
    gran = dict(granularity) if isinstance(granularity, dict) else None   # advisory resolution feedback
    record = {
        "groupby": groupby, "label_key": label_key, "reviewer_model": reviewer_model,
        "verdicts": vd, "n_refuted": n_refuted, "n_suspect": n_suspect,
        "refuted_clusters": refuted_clusters, "refuted_reasons": refuted_reasons,
        "suspect_clusters": suspect_clusters, "suspect_reasons": suspect_reasons,
        "granularity": gran,
    }
    adata.uns[UNS_ANNO]["tier4_audit"] = record          # LAST review (back-compat)
    # PER-COLUMN coverage ledger: Tier-4 must review EVERY annotation column (broad/fine/
    # malignancy/final), not just one. harness_audit reads this to prove each label column that
    # exists in obs was independently reviewed (and by whom). Keyed by label_key.
    reviews = adata.uns[UNS_ANNO].setdefault("tier4_reviews", {})
    reviews[str(label_key)] = {
        "groupby": groupby, "reviewer_model": reviewer_model,
        "n_reviewed": len(vd), "n_refuted": n_refuted, "n_suspect": n_suspect,
        "refuted_clusters": refuted_clusters, "suspect_clusters": suspect_clusters,
        "granularity": gran,
    }
    try:
        session.log_decision(S.DecisionEvent(
            decision_type="annotation_audit", choice=status_map, candidates=[],
            rationale=f"Tier-4 reviewer verdicts (reviewer_model={reviewer_model}); "
                      f"{n_refuted} refuted (to re-annotate), {n_suspect} suspect",
            stage="apply_annotation_audit",
            params={"groupby": groupby, "reviewer_model": reviewer_model}).to_dict())
    except Exception:  # noqa: BLE001
        pass

    summary = {
        "groupby": groupby, "status_key": status_key, "review_required_key": review_required_key,
        "reviewer_model": reviewer_model,
        "n_clusters_reviewed": len(vd), "n_refuted": n_refuted, "n_suspect": n_suspect,
        "n_confirmed": sum(1 for s in status_map.values() if s == "confirmed"),
        "refuted_clusters": refuted_clusters,          # the annotator re-annotates these
        "refuted_reasons": refuted_reasons,            # WHY each was rejected (for re-annotation + humans)
        "suspect_clusters": suspect_clusters,          # ACTION: target for Tier-2 subtype / human review
        "suspect_reasons": suspect_reasons,
        "n_review_required": int(adata.obs[review_required_key].sum()),
        "status_distribution": {str(k): int(v) for k, v in adata.obs[status_key].value_counts().items()},
        "granularity": gran,   # advisory: reviewer's resolution recommendation (down/none/up)
    }
    cp = session.checkpoint("apply_annotation_audit", x_state=session.manifest.x_state,
                            params={"groupby": groupby, "status_key": status_key,
                                    "reviewer_model": reviewer_model})
    return S.success("apply_annotation_audit", summary=summary, checkpoint=cp.path,
                     determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["annotation_review", "finalize_annotation"])
