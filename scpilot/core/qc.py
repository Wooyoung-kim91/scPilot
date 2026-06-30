"""QC + Tier-0 artifact detection — scpilot plan B3 (scpilot-owned, decision #1).

scpilot enters at the merged h5ad, so per-sample doublet detection is done here by
**grouping the merged object by ``sample_id`` and running scrublet per group**
(NOT one global scrublet — doublet distributions differ per library), writing the
scores back onto the merged object so per-sample semantics are preserved.

Two tools:
- ``qc_metrics``  — compute QC metrics (%MT/%ribo, counts/genes) on the *counts*
  layer, per-sample scrublet, a mixed-lineage (EPCAM+CD3D) flag, and return a
  **batch-aware distribution summary** the LLM uses to choose cutoffs. Mutating
  (adds obs columns), checkpoints.
- ``qc_filter`` — apply chosen cutoffs (min_genes / max_pct_mt / max_doublet) and
  subset cells. Mutating, checkpoints, reports kept/removed per sample.
"""

from __future__ import annotations

import time

from scpilot import schemas as S
from scpilot.tools import register

_RIBO_PREFIXES = ("RPS", "RPL")
# EXAMPLE only — NOT a default. The mixed-lineage flag is opt-in: the caller (agent) supplies the
# co-expression gene pair appropriate to the tissue (e.g. epithelial+T-cell => doublet-like). No
# gene names are hardcoded into the call (no-hardcoding rule).
_EXAMPLE_MIXED_LINEAGE_GENES = ("EPCAM", "CD3D")
_QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def _dist(series) -> dict:
    import numpy as np
    a = np.asarray(series, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    qs = np.quantile(a, _QUANTILES)
    return {"min": float(qs[0]), "q05": float(qs[1]), "q25": float(qs[2]), "median": float(qs[3]),
            "q75": float(qs[4]), "q95": float(qs[5]), "max": float(qs[6]), "mean": float(a.mean())}


def _med_mad(a) -> tuple[float, float]:
    """(median, scaled MAD) — MAD×1.4826 ≈ σ for normal data (no scipy dependency)."""
    import numpy as np
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 0.0
    med = float(np.median(a))
    return med, float(np.median(np.abs(a - med)) * 1.4826)


def _suggest_cutoffs(obs, *, n_mads: float) -> dict:
    """MAD-based suggested QC cutoffs (sc-best-practices): lower bounds on counts/genes
    (computed on log1p then mapped back to count space) + an upper %MT bound, plus optional
    doublet-side upper bounds. This is EVIDENCE the LLM judges/overrides per tissue — it is
    NOT auto-applied (mirrors preprocess's suggested_n_pcs_elbow; see no-hardcoding principle).
    """
    import numpy as np
    out: dict = {}
    for metric, lo_key, hi_key in (("n_genes_by_counts", "min_genes", "max_genes"),
                                   ("total_counts", "min_counts", "max_counts")):
        if metric in obs:
            med, mad = _med_mad(np.log1p(np.asarray(obs[metric], dtype=float)))
            out[lo_key] = int(max(0, round(np.expm1(med - n_mads * mad))))
            out[hi_key] = int(max(0, round(np.expm1(med + n_mads * mad))))
    if "pct_counts_mt" in obs:
        med, mad = _med_mad(obs["pct_counts_mt"])
        out["max_pct_mt"] = round(float(med + n_mads * mad), 2)
    return out


def _per_sample_scrublet(adata, sample_key: str, *, seed: int = 0):
    """Run scrublet per sample on the counts layer; return (scores, preds, skipped).

    Builds a MINIMAL temporary AnnData from the counts slice only (no extra
    layers/obs/var) so 35× per-sample copies don't duplicate the full 40k-gene
    object on the 180k-cell merge (Codex review 2.2).
    """
    import anndata as ad
    import numpy as np
    import scanpy as sc

    n = adata.n_obs
    scores = np.full(n, np.nan, dtype=float)
    preds = np.zeros(n, dtype=bool)
    skipped = []
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    sample_vals = adata.obs[sample_key].astype(str).values
    for sid in np.unique(sample_vals):
        idx = np.where(sample_vals == sid)[0]
        if idx.size < 30:                       # scrublet needs enough cells
            skipped.append({"sample": str(sid), "n_cells": int(idx.size), "reason": "too_few_cells"})
            continue
        sub = ad.AnnData(X=counts[idx, :].copy())   # minimal: counts slice only
        try:
            sc.pp.scrublet(sub, random_state=seed)
            scores[idx] = sub.obs["doublet_score"].to_numpy()
            preds[idx] = sub.obs["predicted_doublet"].to_numpy()
        except Exception as exc:  # noqa: BLE001 — degrade gracefully per sample
            skipped.append({"sample": str(sid), "n_cells": int(idx.size), "reason": f"{type(exc).__name__}"})
    return scores, preds, skipped


@register("qc_metrics", mutating=True,
          description="Compute QC metrics (%MT/%ribo), per-sample scrublet doublets, mixed-lineage flag, "
                      "and a batch-aware distribution summary for cutoff selection (plan B3).")
def qc_metrics(session, *, sample_key: str = "sample_id", mito_prefix: str | None = None,
               mixed_lineage_genes: tuple[str, ...] | list[str] | None = None,
               run_scrublet: bool = True, n_mads: float = 5.0, seed: int = 0,
               **params) -> S.ToolResult:
    import numpy as np
    import pandas as pd
    import scanpy as sc

    from scpilot.core import _species

    t0 = time.time()
    adata = session.adata
    warnings: list[str] = []

    # --- organism is DETECTED from the data, never assumed (no-hardcoding rule) ---
    org = _species.detect_organism(adata)
    if mito_prefix is None:                      # caller didn't pin it → use detected style
        mito_prefix = org["mito_prefix"]
        warnings.append(f"organism={org['organism']} ({org['evidence']}); "
                        f"mito_prefix auto-set to '{mito_prefix}'")

    # --- QC metrics on COUNTS (X may be normalized on the merged object) ---
    # exception-safe swap: restore X in finally so a failure can't leave X bound
    # to the counts layer with silently-wrong downstream semantics (Codex review 1.2)
    up = adata.var_names.str.upper()
    adata.var["mt"] = up.str.startswith(mito_prefix.upper())
    adata.var["ribo"] = up.str.startswith(_RIBO_PREFIXES)
    has_counts = "counts" in adata.layers
    if not has_counts:
        warnings.append("no 'counts' layer — QC metrics computed on X (assumed counts)")
    x_backup = adata.X if has_counts else None
    try:
        if has_counts:
            adata.X = adata.layers["counts"]
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], inplace=True,
                                   percent_top=None, log1p=False)
    finally:
        if x_backup is not None:
            adata.X = x_backup

    # --- mixed-lineage co-expression flag — OPT-IN (caller supplies the gene pair) ---
    # No gene names are hardcoded: the flag is only computed when mixed_lineage_genes is given.
    # Supplied symbols are resolved to the data's actual casing (EPCAM->Epcam in mouse) so the
    # caller's pair works cross-organism.
    if not mixed_lineage_genes:
        adata.obs["mixed_lineage_flag"] = False
        mixed_frac = None
        warnings.append("mixed-lineage flag skipped (opt-in — pass mixed_lineage_genes=[g1,g2] to enable)")
    else:
        resolved = _species.resolve(adata, mixed_lineage_genes)
        present = [v for v in resolved.values() if v is not None]
        if len(present) == len(list(mixed_lineage_genes)):
            src = adata.layers["counts"] if "counts" in adata.layers else adata.X
            cols = [adata.var_names.get_loc(g) for g in present]
            sub = src[:, cols]
            dense = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
            adata.obs["mixed_lineage_flag"] = (dense > 0).all(axis=1)
            mixed_frac = float(adata.obs["mixed_lineage_flag"].mean())
            if present != list(mixed_lineage_genes):
                warnings.append(f"mixed-lineage genes resolved to data casing: {present}")
        else:
            adata.obs["mixed_lineage_flag"] = False
            mixed_frac = None
            missing = [k for k, v in resolved.items() if v is None]
            warnings.append(f"mixed-lineage genes absent ({missing}); flag set False")

    # --- per-sample scrublet ---
    scrublet_skipped = []
    have_sample = sample_key in adata.obs.columns
    if run_scrublet and have_sample:
        scores, preds, scrublet_skipped = _per_sample_scrublet(adata, sample_key, seed=seed)
        adata.obs["doublet_score"] = scores
        adata.obs["predicted_doublet"] = preds
        doublet_rate = float(np.nanmean(preds.astype(float))) if np.isfinite(scores).any() else None
    else:
        doublet_rate = None
        if run_scrublet and not have_sample:
            warnings.append(f"no '{sample_key}' column — skipped per-sample scrublet")

    # --- batch-aware distribution summary ---
    metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo"]
    if "doublet_score" in adata.obs:
        metrics.append("doublet_score")
    global_dist = {m: _dist(adata.obs[m]) for m in metrics if m in adata.obs}

    per_sample_rows = []
    if have_sample:
        g = adata.obs.groupby(sample_key, observed=True)
        for sid, sub in g:
            row = {"sample": str(sid), "n_cells": int(len(sub)),
                   "median_genes": float(sub["n_genes_by_counts"].median()),
                   "median_total": float(sub["total_counts"].median()),
                   "median_pct_mt": float(sub["pct_counts_mt"].median()),
                   "median_pct_ribo": float(sub["pct_counts_ribo"].median())}
            if "predicted_doublet" in sub:
                row["doublet_rate"] = float(sub["predicted_doublet"].mean())
            per_sample_rows.append(row)
    per_sample_df = pd.DataFrame(per_sample_rows)

    # MAD-based suggested cutoffs (evidence for the LLM to judge — NOT auto-applied)
    suggested = {"global": _suggest_cutoffs(adata.obs, n_mads=n_mads)}
    cutoffs_artifact = None
    if have_sample:
        # Per-sample cutoffs are still computed in FULL (identical analysis); but to keep the tool
        # SUMMARY O(1) in sample count (it is re-sent every agent turn — token cost), the full
        # per-sample map goes to a CSV artifact and the summary carries the global cutoffs + a
        # compact min/median/max SPREAD across samples (the batch-aware signal, descriptive only —
        # no new threshold/judgment). Full detail remains available via the artifact path.
        per_sample_cut = {str(sid): _suggest_cutoffs(sub, n_mads=n_mads)
                          for sid, sub in adata.obs.groupby(sample_key, observed=True)}
        cut_df = pd.DataFrame(per_sample_cut).T.sort_index()
        cutoffs_artifact = session.artifact_path("qc_suggested_cutoffs_per_sample.csv")
        cutoffs_artifact.parent.mkdir(parents=True, exist_ok=True)
        cut_df.to_csv(cutoffs_artifact)
        suggested["per_sample_spread"] = {
            str(k): {"min": round(float(cut_df[k].min()), 3),
                     "median": round(float(cut_df[k].median()), 3),
                     "max": round(float(cut_df[k].max()), 3)}
            for k in cut_df.columns}
        suggested["per_sample_artifact"] = str(cutoffs_artifact)
        suggested["n_samples"] = int(cut_df.shape[0])

    summary = {
        "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
        "organism": org["organism"], "organism_evidence": org["evidence"],
        "mito_prefix": mito_prefix,
        "sample_key": sample_key if have_sample else None,
        "n_samples": int(per_sample_df.shape[0]) if have_sample else None,
        "qc_metrics": metrics,
        "global_distributions": global_dist,
        "suggested_cutoffs": suggested,
        "n_mads": n_mads,
        "doublet_rate_overall": doublet_rate,
        "mixed_lineage_frac": mixed_frac,
        "scrublet_skipped_samples": scrublet_skipped,
    }
    tables = {}
    if not per_sample_df.empty:
        tables["per_sample_qc"] = S.table_preview(per_sample_df, max_rows=50)
    artifacts = []
    if cutoffs_artifact is not None:
        artifacts.append(S.artifact_csv(str(cutoffs_artifact), n_rows=int(cut_df.shape[0]),
                                        n_cols=int(cut_df.shape[1]),
                                        description="per-sample MAD suggested cutoffs (full; "
                                                    "summary carries global + spread)"))

    cp = session.checkpoint("qc_metrics", x_state=session.manifest.x_state,
                            params={"sample_key": sample_key, "mito_prefix": mito_prefix,
                                    "run_scrublet": run_scrublet, "n_mads": n_mads, "seed": seed})
    return S.success("qc_metrics", summary=summary, tables=tables, artifacts=artifacts,
                     warnings=warnings, checkpoint=cp.path,
                     determinism_grade="B" if run_scrublet else "A",
                     duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["qc_filter"])


@register("qc_filter", mutating=True,
          description="Apply QC cutoffs (min_genes / max_pct_mt / max_doublet_score) and subset cells (plan B3).")
def qc_filter(session, *, min_genes: int = 200, max_pct_mt: float = 20.0,
              min_counts: int = 0, max_doublet_score: float | None = None,
              drop_predicted_doublets: bool = False, sample_key: str = "sample_id",
              **params) -> S.ToolResult:
    import numpy as np

    t0 = time.time()
    adata = session.adata
    if "n_genes_by_counts" not in adata.obs:
        return S.error("qc_filter", "invalid_state",
                       "QC metrics absent — run qc_metrics first", recoverable=True,
                       suggested_next_tools=["qc_metrics"])

    n0 = adata.n_obs
    keep = (adata.obs["n_genes_by_counts"] >= min_genes) & (adata.obs["pct_counts_mt"] <= max_pct_mt)
    if min_counts > 0 and "total_counts" in adata.obs:
        keep &= adata.obs["total_counts"] >= min_counts
    if max_doublet_score is not None and "doublet_score" in adata.obs:
        ds = adata.obs["doublet_score"]
        keep &= (ds <= max_doublet_score) | ds.isna()      # keep cells with no score
    if drop_predicted_doublets and "predicted_doublet" in adata.obs:
        keep &= ~adata.obs["predicted_doublet"].fillna(False).astype(bool)

    # guard: cutoffs that remove EVERYTHING must not checkpoint an empty object
    # (downstream PCA/clustering would fail opaquely) — Codex review 2.3
    if not bool(keep.values.any()):
        return S.error("qc_filter", "convergence_failed",
                       "all cells removed by cutoffs — relax thresholds", recoverable=True,
                       summary={"n_cells_before": int(n0),
                                "cutoffs": {"min_genes": min_genes, "max_pct_mt": max_pct_mt,
                                            "min_counts": min_counts, "max_doublet_score": max_doublet_score,
                                            "drop_predicted_doublets": drop_predicted_doublets}},
                       suggested_next_tools=["qc_metrics"])

    # per-sample kept/removed (batch-aware reporting)
    per_sample = {}
    if sample_key in adata.obs.columns:
        before = adata.obs[sample_key].astype(str).value_counts().to_dict()
        after = adata.obs[sample_key].astype(str)[keep.values].value_counts().to_dict()
        per_sample = {s: {"before": int(before.get(s, 0)), "after": int(after.get(s, 0))}
                      for s in before}

    adata._inplace_subset_obs(keep.values)
    # invariants are now enforced centrally in session.checkpoint() (plan B1) — no per-tool call.

    summary = {
        "n_cells_before": int(n0), "n_cells_after": int(adata.n_obs),
        "n_removed": int(n0 - adata.n_obs),
        "frac_removed": round((n0 - adata.n_obs) / n0, 4) if n0 else 0.0,
        "cutoffs": {"min_genes": min_genes, "max_pct_mt": max_pct_mt, "min_counts": min_counts,
                    "max_doublet_score": max_doublet_score,
                    "drop_predicted_doublets": drop_predicted_doublets},
        "per_sample": per_sample,
    }
    warnings = []
    empty = [s for s, v in per_sample.items() if v["after"] == 0]
    if empty:
        warnings.append(f"{len(empty)} sample(s) fully removed by cutoffs: {empty[:5]}")

    cp = session.checkpoint("qc_filter", x_state=session.manifest.x_state,
                            params={**summary["cutoffs"], "sample_key": sample_key})
    return S.success("qc_filter", summary=summary, warnings=warnings, checkpoint=cp.path,
                     determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["preprocess"])
