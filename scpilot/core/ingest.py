"""Upstream ingest: raw 10x → merged h5ad — scpilot plan B0 (end-to-end, decision A 2026-06-10).

scpilot is now end-to-end: instead of consuming a pre-built merged h5ad, ``ingest``
builds it from a dataset **profile** (the scqc-style YAML pointing at per-sample 10x
matrices + a metadata CSV). Reuses the vendored upstream primitives:
metadata harmonize/filter/derive (``vendor.metaschema``), robust 10x reader +
per-sample QC-metric attach (``vendor.io_10x``), then concat + gene filter +
normalize (counts immutable; log-norm → ``scale.data``).

After ``ingest`` the session holds the merged AnnData and the normal downstream
tools (qc_metrics → preprocess → ...) run as usual. (For data that already has a
merged h5ad, skip ingest and create the session on that file directly.)
"""

from __future__ import annotations

import time

from scpilot import schemas as S
from scpilot.tools import register


def _add_lognorm(adata, *, normalized_layer="scale.data", target_sum=1e4):
    """normalize_total + log1p into X, mirrored to ``normalized_layer`` (scqc convention)."""
    import scanpy as sc
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.layers[normalized_layer] = adata.X.copy()
    return adata


@register("ingest", mutating=True, long_running=True,
          description="Build the merged AnnData from a dataset profile: metadata harmonize → per-sample 10x "
                      "read + cell QC → merge → normalize (counts + scale.data). Raw-10x entry point (plan B0).")
def ingest(session, *, profile: str | None = None, min_genes: int | None = None,
           max_pct_mt: float | None = None, min_cells: int | None = None,
           target_sum: float | None = None, **params) -> S.ToolResult:
    import anndata as ad
    import scanpy as sc
    from scipy import sparse

    from scpilot.vendor.config import PipelineConfig
    from scpilot.vendor.io_10x import read_one_sample
    from scpilot.vendor.metaschema import build_metadata

    t0 = time.time()
    prof = profile or session.manifest.input.get("path")
    if not prof:
        return S.error("ingest", "missing_input", "no profile YAML given (param 'profile' or session input)",
                       recoverable=False)
    # the profile drives QC/merge knobs; tool params override ONLY when explicitly given
    overrides = {k: v for k, v in {"min_genes": min_genes, "max_pct_mt": max_pct_mt,
                                   "min_cells": min_cells, "target_sum": target_sum}.items() if v is not None}
    cfg = PipelineConfig.from_profile(prof, overrides)
    min_genes, max_pct_mt = cfg.min_genes, cfg.max_pct_mt
    min_cells, target_sum = cfg.min_cells, cfg.target_sum

    # 1) harmonized + filtered + derived per-sample metadata
    df, info = build_metadata(cfg)
    warnings: list[str] = []
    if info.get("unmapped_total"):
        warnings.append(f"{info['unmapped_total']} unmapped harmonize value(s) — see profile.harmonize")
    if df.empty:
        return S.error("ingest", "data_gate_failed", "no samples after metadata filters", recoverable=True)

    # 2) per-sample 10x read + cell QC filter
    adatas, keys, per_sample, n_fail = [], [], [], 0
    for _, row in df.iterrows():
        row = row.to_dict()
        sid = str(row[cfg.sample_id_col])
        try:
            a = read_one_sample(row, input_root=cfg.input_root_path, sample_id_col=cfg.sample_id_col,
                                matrix_dir_col=cfg.matrix_dir_col, batch_col=cfg.batch_col,
                                mito_prefix=cfg.mito_prefix)
        except Exception as exc:  # noqa: BLE001 — skip unreadable sample, record
            per_sample.append({"sample": sid, "status": f"read_fail:{type(exc).__name__}"}); n_fail += 1
            continue
        raw_n = a.n_obs
        keep = (a.obs["n_genes_by_counts"] >= min_genes) & (a.obs["pct_counts_mt"] <= max_pct_mt)
        a = a[keep].copy()
        per_sample.append({"sample": sid, "n_raw": int(raw_n), "n_kept": int(a.n_obs)})
        if a.n_obs > 0:
            adatas.append(a); keys.append(sid)
    if not adatas:
        return S.error("ingest", "convergence_failed", "all samples empty after cell QC", recoverable=True)

    # 3) merge (deterministic sample order) + gene filter + normalize
    merged = ad.concat(adatas, join="outer", label="sample_id_from_concat", keys=keys,
                       index_unique=None, merge="same", fill_value=0)
    merged.var_names_make_unique()
    if sparse.issparse(merged.X):
        merged.X = merged.X.tocsr()
    merged.layers["counts"] = merged.X.copy()
    sc.pp.filter_genes(merged, min_cells=min_cells)
    merged.layers["counts"] = merged.X.copy()
    merged = _add_lognorm(merged, normalized_layer=cfg.normalized_layer or "scale.data", target_sum=target_sum)

    session.set_adata(merged)
    session.manifest.counts_fingerprint = None  # (re)compute from the freshly built merged
    summary = {
        "n_cells": int(merged.n_obs), "n_genes": int(merged.n_vars),
        "n_samples_merged": len(keys), "n_samples_failed": n_fail,
        "layers": sorted(merged.layers.keys()),
        "condition_counts": (merged.obs["condition"].astype(str).value_counts().to_dict()
                             if "condition" in merged.obs else {}),
        "profile": str(prof),
    }
    cp = session.checkpoint("ingest", x_state="log1p",
                            params={"profile": str(prof), "min_genes": min_genes,
                                    "max_pct_mt": max_pct_mt, "min_cells": min_cells, "target_sum": target_sum})
    summary["per_sample"] = per_sample[:50]
    return S.success("ingest", summary=summary, warnings=warnings, checkpoint=cp.path,
                     determinism_grade="A", duration_s=round(time.time() - t0, 1),
                     suggested_next_tools=["qc_metrics", "preprocess"])
