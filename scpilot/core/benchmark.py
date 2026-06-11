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
from scpilot.tools import register

_AGG_COLS = ["Batch correction", "Bio conservation", "Total"]


@register("benchmark", mutating=False, long_running=True,
          description="scib-metrics comparison of integration embeddings (PCA/Harmony/scVI) on batch-correction "
                      "vs bio-conservation; label_key=major_cell_type (marker-anchored, not leiden — avoids "
                      "circularity). Returns a scored table + scib summary-table figure (plan B10).")
def benchmark(session, *, label_key: str = "major_cell_type", batch_key: str = "sample_id",
              embeddings: list | None = None, drop_labels: list | None = None,
              min_label_cells: int = 10, subsample: int | None = 60000, seed: int = 0,
              **params) -> S.ToolResult:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scib_metrics.benchmark import Benchmarker

    t0 = time.time()
    adata = session.adata
    drop_labels = ["Unknown"] if drop_labels is None else [str(x) for x in drop_labels]

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
    csv_path = art_dir / "benchmark_scib.csv"
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
