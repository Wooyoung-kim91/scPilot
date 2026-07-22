"""scib Benchmarker comparison of integration embeddings — scpilot plan B10.

Compares the unintegrated PCA baseline vs Harmony vs scVI on the batch-correction
vs bio-conservation trade-off (scib-metrics ``Benchmarker``).

Design decisions (de-risk ①):
- ``label_key`` = marker-anchored ``major_cell_type``, NOT leiden cluster labels —
  using clusters derived FROM an integrated embedding as that embedding's own
  bio-conservation label is circular. major_cell_type is batch-agnostic.
- ``Unknown`` (and labels below ``min_label_cells``) are dropped so a grab-bag /
  tiny class does not pollute the bio-conservation metrics.
- Large datasets are stratified-subsampled (by label) to keep kNN / silhouette /
  LISI tractable; the subsample is reported, never silent.

Contract: AnnData in → scored summary out (non-mutating). Writes the scib results
table (CSV) + scib summary-table figure as artifacts.
"""

from __future__ import annotations

import time
from pathlib import Path

from scpilot import schemas as S
from scpilot.tools import register, require_capability

_AGG_COLS = ["Batch correction", "Bio conservation", "Total"]


@register("benchmark", mutating=False, long_running=True,
          description="scib-metrics comparison of integration embeddings (PCA/Harmony/scVI) on batch-correction "
                      "vs bio-conservation. CRITICAL: label_key must be an EMBEDDING-INDEPENDENT cell-type set — "
                      "use the cross-method consensus_annotation output (recommended) or one consistent label "
                      "for all embeddings; NEVER each embedding's own clustering-derived labels (circular, de-risk "
                      "①). drop_labels (caller-set) removes non-cell-type/sentinel labels. Returns a scored table "
                      "+ scib summary-table figure (plan B10).")
def benchmark(session, *, label_key: str = "major_cell_type", batch_key: str = "sample_id",
              embeddings: list | None = None, drop_labels: list | None = None,
              min_label_cells: int = 10, subsample: int | None = 60000, seed: int = 0,
              **params) -> S.ToolResult:
    if (err := require_capability("benchmark")) is not None:
        return err
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scib_metrics.benchmark import Benchmarker

    t0 = time.time()
    adata = session.adata
    # default-drop the non-biological Tier-1 labels (Unknown + the rule-1/4 artifact
    # buckets) so a grab-bag / doublet / low-quality class can't pollute bio-conservation.
    from scpilot.core.annotate import ARTIFACT_LABELS
    drop_labels = sorted(ARTIFACT_LABELS) if drop_labels is None else [str(x) for x in drop_labels]

    candidates = embeddings or ["X_pca", "X_harmony", "X_scVI"]
    embs = [e for e in candidates if e in adata.obsm]
    absent = [e for e in candidates if e not in adata.obsm]
    if len(embs) < 2:
        return S.error("benchmark", "data_gate_failed",
                       f"need >=2 embeddings present; have {embs} (absent {absent}) in obsm{sorted(adata.obsm)}",
                       recoverable=True, suggested_next_tools=["integrate_harmony", "train_scvi", "cluster"])
    for k in (label_key, batch_key):
        if k not in adata.obs.columns:
            return S.error("benchmark", "data_gate_failed",
                           f"'{k}' absent in obs{list(adata.obs.columns)[:12]}...", recoverable=True)

    warnings: list[str] = []
    if absent:
        warnings.append(f"embeddings absent, skipped: {absent}")

    # --- cell mask: drop blacklisted (Unknown) + tiny labels --------------------
    lab = adata.obs[label_key].astype(str)
    keep = ~lab.isin(drop_labels)
    n_drop_black = int((~keep).sum())
    vc = lab[keep].value_counts()
    small = sorted(vc[vc < min_label_cells].index)
    if small:
        keep = keep & ~lab.isin(small)
        warnings.append(f"dropped {len(small)} label(s) with <{min_label_cells} cells: {small}")
    idx = np.where(keep.values)[0]
    if idx.size == 0 or lab.values[idx].astype(str).__len__() == 0:
        return S.error("benchmark", "data_gate_failed", "no cells left after label filtering", recoverable=True)

    # --- stratified subsample (by label) for tractability -----------------------
    rng = np.random.default_rng(seed)
    n_kept = int(idx.size)
    subsampled_to = None
    if subsample and n_kept > subsample:
        labs_kept = lab.values[idx]
        sel = []
        for L in np.unique(labs_kept):
            li = idx[labs_kept == L]
            take = min(li.size, max(1, int(round(subsample * li.size / n_kept))))
            sel.append(rng.choice(li, size=take, replace=False))
        idx = np.sort(np.concatenate(sel))
        subsampled_to = int(idx.size)
        warnings.append(f"stratified-subsampled {n_kept} -> {idx.size} cells "
                        f"(subsample={subsample}) for scib tractability; re-run with subsample=None for full")

    bdata = adata[idx].copy()
    bdata.obs[label_key] = bdata.obs[label_key].astype(str).astype("category")
    bdata.obs[batch_key] = bdata.obs[batch_key].astype(str).astype("category")
    n_labels = int(bdata.obs[label_key].nunique())
    n_batches = int(bdata.obs[batch_key].nunique())

    pre = "X_pca" if "X_pca" in embs else None
    bm = Benchmarker(bdata, batch_key=batch_key, label_key=label_key,
                     embedding_obsm_keys=embs, pre_integrated_embedding_obsm_key=pre,
                     n_jobs=-1, progress_bar=False)
    bm.benchmark()
    res = bm.get_results(min_max_scale=False, clean_names=False)   # index: embeddings + "Metric Type"

    # --- artifacts: results CSV + scib summary-table figure ---------------------
    art_dir = session.artifacts_dir
    art_dir.mkdir(parents=True, exist_ok=True)
    csv_path = session.artifact_path("benchmark_scib.csv")   # no-overwrite (P1-2)
    res.to_csv(csv_path)
    arts = [S.artifact_csv(str(csv_path), n_rows=int(res.shape[0]), n_cols=int(res.shape[1]),
                           description="scib-metrics results (per-embedding metrics + aggregate scores)")]
    png_path = art_dir / "benchmark_scib_table.png"
    try:
        bm.plot_results_table(min_max_scale=False, show=False)
        plt.savefig(png_path, bbox_inches="tight", dpi=200)
        plt.close("all")
        arts.append(S.Artifact(path=str(png_path), kind="png",
                               description="scib summary table (rows sorted by Total)"))
    except Exception as exc:   # noqa: BLE001 — figure is a nicety, never fail the benchmark on it
        warnings.append(f"plot_results_table failed: {type(exc).__name__}: {exc}")
        plt.close("all")

    # --- scored summary ---------------------------------------------------------
    res_num = res.drop(index="Metric Type", errors="ignore")
    scores = {}
    for emb in res_num.index:
        row = res_num.loc[emb]
        scores[str(emb)] = {c: (float(row[c]) if c in row and row[c] == row[c] else None) for c in _AGG_COLS}
    ranked = sorted((e for e in scores if scores[e].get("Total") is not None),
                    key=lambda e: scores[e]["Total"], reverse=True)
    best = ranked[0] if ranked else None
    if best:
        # Record the best-embedding pick as a first-class DECISION EVENT (replayable, joined to
        # this step's evidence via recipe_hash) so it is NOT a silent, unlogged state mutation.
        # Downstream readers (autoplot / report / export_final) still consume
        # uns['scpilot']['best_embedding'], so we ALSO expose it there — but the authoritative,
        # logged record is the decision event, not the bare uns write.
        try:
            session.log_decision(S.DecisionEvent(
                decision_type="integration_method", choice=str(best), candidates=ranked,
                rationale=f"highest scib Total among {ranked} -> best embedding for downstream "
                          "annotation/finalize/plots (not a fixed scVI-first default)",
                stage="benchmark", params={"best_embedding": str(best)}))
        except Exception:  # noqa: BLE001 — logging must never fail the read-only benchmark
            pass
        try:
            adata.uns.setdefault("scpilot", {})["best_embedding"] = str(best)
        except Exception:  # noqa: BLE001
            pass

    # over-correction flag: an embedding that wins batch-correction but loses bio-conservation
    overcorrection = []
    if "Batch correction" in res_num and "Bio conservation" in res_num:
        bc_rank = res_num["Batch correction"].rank(ascending=False)
        bio_rank = res_num["Bio conservation"].rank(ascending=True)   # 1 = worst bio
        for emb in res_num.index:
            if bc_rank.get(emb, 99) == 1 and bio_rank.get(emb, 99) == 1 and len(res_num) > 1:
                overcorrection.append(str(emb))
    if overcorrection:
        warnings.append(f"possible over-correction (best batch-mixing but worst bio-conservation): {overcorrection}")

    table = S.table_preview(res.reset_index().rename(columns={"index": "embedding"}),
                            full_csv=str(csv_path))
    summary = {
        "label_key": label_key, "batch_key": batch_key,
        "embeddings": embs, "n_labels": n_labels, "n_batches": n_batches,
        "n_cells_evaluated": int(bdata.n_obs), "n_cells_total": int(adata.n_obs),
        "dropped_unknown_or_blacklisted": n_drop_black, "subsampled_to": subsampled_to,
        "scores": scores, "ranking_by_total": ranked, "best": best,
        "overcorrection_flag": overcorrection,
    }
    return S.success("benchmark", summary=summary, tables={"scib_results": table}, artifacts=arts,
                     warnings=warnings, determinism_grade="B", duration_s=round(time.time() - t0, 1),
                     params={"label_key": label_key, "batch_key": batch_key, "embeddings": embs,
                             "drop_labels": drop_labels, "min_label_cells": min_label_cells,
                             "subsample": subsample, "seed": seed},
                     suggested_next_tools=["cluster", "annotate_broad", "report"])


@register("benchmark_reference", mutating=False,
          description="Score a PREDICTED annotation column against a TRUSTED REFERENCE obs column on the SAME cells "
                      "(read-only; scores existing labels, does NOT annotate). Metrics: ARI, AMI (partition "
                      "agreement, label-name-invariant), per-cell accuracy, macro-F1, per-class precision/recall/F1, "
                      "and a confusion matrix. LABEL-SPACE HONESTY: predicted labels are free-text, the reference is "
                      "a fixed vocabulary — apply the caller-supplied label_map (predicted->reference; NEVER "
                      "fabricated), then case-normalized EXACT match; the fraction of predicted cells with no "
                      "reference counterpart is REPORTED, and BOTH a strict score (unmatched=wrong) AND a "
                      "matched-only score (cells whose label maps to a known reference class) are returned, clearly "
                      "labeled. The reference column is user-supplied ground truth — no biology is baked in. Writes a "
                      "metrics JSON + confusion CSV. Run after annotation when a reference is available.")
def benchmark_reference(session, *, pred_key: str, ref_key: str,
                        label_map: dict | None = None, **params) -> S.ToolResult:
    import json

    import numpy as np
    import pandas as pd
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 confusion_matrix, precision_recall_fscore_support)

    t0 = time.time()
    adata = session.adata
    warnings: list[str] = []

    # --- guards: columns present, same length, non-empty ------------------------
    if pred_key not in adata.obs.columns:
        return S.error("benchmark_reference", "invalid_state",
                       f"predicted column '{pred_key}' absent in obs — annotate first "
                       f"(have {list(adata.obs.columns)[:12]}...)",
                       recoverable=True, suggested_next_tools=["apply_annotation", "finalize_annotation"])
    if ref_key not in adata.obs.columns:
        return S.error("benchmark_reference", "invalid_state",
                       f"reference column '{ref_key}' absent in obs — supply a trusted ground-truth column "
                       f"(have {list(adata.obs.columns)[:12]}...)",
                       recoverable=True)
    pred_raw = adata.obs[pred_key]
    ref_raw = adata.obs[ref_key]
    if len(pred_raw) != len(ref_raw):   # same adata → same length; guard anyway
        return S.error("benchmark_reference", "data_gate_failed",
                       f"length mismatch pred={len(pred_raw)} vs ref={len(ref_raw)}", recoverable=False)

    # --- drop cells missing EITHER label (NaN ground truth cannot be scored) -----
    valid = pred_raw.notna().values & ref_raw.notna().values
    n_total = int(len(valid))
    n_dropped = int((~valid).sum())
    if valid.sum() == 0:
        return S.error("benchmark_reference", "data_gate_failed",
                       f"no cells with BOTH a predicted ('{pred_key}') and reference ('{ref_key}') label "
                       f"(all NaN in at least one column)", recoverable=True)
    if n_dropped:
        warnings.append(f"dropped {n_dropped}/{n_total} cell(s) missing a predicted or reference label "
                        "(NaN ground truth cannot be scored)")

    pred_s = pred_raw[valid].astype(str).to_numpy()
    ref_s = ref_raw[valid].astype(str).to_numpy()

    # --- label reconciliation: label_map (never fabricated) then case-normalized exact match ----
    lmap = {str(k): str(v) for k, v in (label_map or {}).items()}
    pred_mapped = np.array([lmap.get(p, p) for p in pred_s], dtype=object)

    # reference vocabulary is FIXED (the ground truth). Case-variant spellings of the SAME label
    # are FOLDED to one canonical spelling (first-seen) for BOTH y_true and the prediction rewrite,
    # so a reference cell spelled in the non-canonical case is still scorable — otherwise a
    # case-normalized prediction could never match it and that reference variant would be
    # permanently unscorable (issue #2).
    ref_canon: dict[str, str] = {}
    for r in ref_s:
        ref_canon.setdefault(r.casefold(), r)
    n_ref_raw_spellings = len(set(ref_s))
    ref_s = np.array([ref_canon[r.casefold()] for r in ref_s], dtype=object)
    reference_classes = sorted(set(ref_s))
    if len(reference_classes) < n_ref_raw_spellings:
        warnings.append(f"reference case-variant spellings folded to {len(reference_classes)} "
                        f"canonical label(s) (from {n_ref_raw_spellings}) so no case-variant is unscorable")

    matched = np.array([p.casefold() in ref_canon for p in pred_mapped], dtype=bool)
    n_matched = int(matched.sum())
    n_unmatched = int((~matched).sum())
    unmatched_frac = round(n_unmatched / len(pred_mapped), 4)
    unmatched_labels = sorted({str(p) for p, m in zip(pred_mapped, matched) if not m})
    if unmatched_labels:
        warnings.append(f"{n_unmatched}/{len(pred_mapped)} predicted cell(s) "
                        f"({unmatched_frac:.1%}) have a label with NO reference counterpart: "
                        f"{unmatched_labels[:10]} — counted WRONG in the strict score, EXCLUDED from matched-only")

    # strict prediction array: matched → canonical reference label; unmatched → a distinct sentinel
    # (guaranteed absent from the reference vocab, so it is honestly WRONG, never accidentally correct).
    y_true = ref_s
    y_pred_strict = np.array([ref_canon[p.casefold()] if m else f"UNMATCHED::{p}"
                              for p, m in zip(pred_mapped, matched)], dtype=object)

    def _scores(yt, yp) -> dict:
        """accuracy + macro-F1 + per-class P/R/F1 over the REFERENCE classes present in yt."""
        classes = sorted(set(yt))
        acc = float((np.asarray(yt) == np.asarray(yp)).mean())
        prec, rec, f1, sup = precision_recall_fscore_support(
            yt, yp, labels=classes, average=None, zero_division=0)
        macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
        per_class = {str(c): {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                              "f1": round(float(f1[i]), 4), "support": int(sup[i])}
                     for i, c in enumerate(classes)}
        return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
                "n_classes": len(classes), "per_class": per_class}

    # ARI/AMI: partition agreement, invariant to label NAMES — computed on the reconciled labeling
    # (strict, with unmatched cells forming their own group) vs the reference partition.
    ari = round(float(adjusted_rand_score(y_true, y_pred_strict)), 4)
    ami = round(float(adjusted_mutual_info_score(y_true, y_pred_strict)), 4)

    strict = _scores(y_true, y_pred_strict)

    # matched-only: restrict to cells whose predicted label maps to a KNOWN reference class
    matched_only: dict | None
    if n_matched == 0:
        matched_only = None
        warnings.append("matched-only score undefined: NO predicted cell mapped to a reference class")
    else:
        mt = _scores(y_true[matched], y_pred_strict[matched])
        mt["n_cells"] = n_matched
        matched_only = mt

    if len(reference_classes) < 2:
        warnings.append(f"single reference class {reference_classes} — ARI/macro-F1 are degenerate here")

    # --- artifacts: confusion CSV (rows=reference, cols=predicted) + metrics JSON ---
    art_dir = session.artifacts_dir
    art_dir.mkdir(parents=True, exist_ok=True)
    conf_labels = sorted(set(y_true) | set(y_pred_strict))
    cm = confusion_matrix(y_true, y_pred_strict, labels=conf_labels)
    cm_df = pd.DataFrame(cm, index=[f"ref::{c}" for c in conf_labels],
                         columns=[f"pred::{c}" for c in conf_labels])
    conf_path = session.artifact_path("benchmark_reference_confusion.csv")
    cm_df.to_csv(conf_path)

    metrics = {
        "pred_key": pred_key, "ref_key": ref_key,
        "label_map_applied": bool(lmap), "label_map": lmap,
        "n_cells_total": n_total, "n_cells_scored": int(len(pred_mapped)), "n_dropped_missing": n_dropped,
        "reference_classes": reference_classes, "n_reference_classes": len(reference_classes),
        "predicted_labels_mapped": sorted({str(p) for p in pred_mapped}),
        "unmatched_predicted_fraction": unmatched_frac, "n_unmatched_cells": n_unmatched,
        "unmatched_predicted_labels": unmatched_labels,
        "ari": ari, "ami": ami,
        "strict": strict, "matched_only": matched_only,
        "confusion_csv": str(conf_path),
        "note": "strict = unmatched predicted labels counted WRONG; matched_only = computed only on cells "
                "whose predicted label maps to a known reference class. ARI/AMI are label-name-invariant.",
    }
    json_path = session.artifact_path("benchmark_reference.json")
    json_path.write_text(json.dumps(metrics, indent=2, default=str))

    artifacts = [
        S.Artifact(path=str(json_path), kind="json",
                   description="reference-label benchmark metrics (ARI/AMI/accuracy/F1)"),
        S.artifact_csv(str(conf_path), n_rows=int(cm_df.shape[0]), n_cols=int(cm_df.shape[1]),
                       description="confusion matrix (rows=reference, cols=predicted; UNMATCHED::* = no ref counterpart)"),
    ]
    summary = {
        "pred_key": pred_key, "ref_key": ref_key,
        "label_map_applied": bool(lmap),
        "n_cells_scored": int(len(pred_mapped)), "n_dropped_missing": n_dropped,
        "n_reference_classes": len(reference_classes),
        "unmatched_predicted_fraction": unmatched_frac, "n_unmatched_cells": n_unmatched,
        "unmatched_predicted_labels": unmatched_labels,
        "ari": ari, "ami": ami,
        "strict_accuracy": strict["accuracy"], "strict_macro_f1": strict["macro_f1"],
        "matched_only_accuracy": (matched_only or {}).get("accuracy"),
        "matched_only_macro_f1": (matched_only or {}).get("macro_f1"),
        "strict": strict, "matched_only": matched_only,
        "metrics_json": str(json_path),
    }
    tab = pd.DataFrame([{"reference_class": c, **strict["per_class"][c]} for c in strict["per_class"]])
    # No standalone full CSV for this preview: do NOT point full_csv at the metrics JSON — a
    # consumer downloading a table's "full CSV" must never receive JSON (issue #1). The CSV
    # artifact is the confusion matrix; the full per-class metrics live in the JSON artifact.
    tables = {"per_class_strict": S.table_preview(tab)} if len(tab) else {}
    return S.success("benchmark_reference", summary=summary, tables=tables, artifacts=artifacts,
                     warnings=warnings, determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     params={"pred_key": pred_key, "ref_key": ref_key, "label_map": lmap},
                     suggested_next_tools=["report"])
