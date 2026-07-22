"""Pure, scpilot-FREE recipes — the actual scanpy/anndata logic of each deterministic tool.

Each function uses ONLY scanpy / anndata / numpy / pandas (no scpilot imports, no Session, no
registry), so its SOURCE can be inlined verbatim into a standalone tutorial script (see
``scpilot.scriptgen``). The scpilot tool calls the SAME recipe, so the generated standalone script
and the tool are logic-identical BY CONSTRUCTION — the equivalence test only has to confirm the
read/write wiring, not re-verify the math.

Contract: a recipe takes an AnnData (+ keyword params with the tool's defaults) and either returns a
boolean keep-mask (subset tools) or returns/mutates-and-returns the AnnData (transform tools). It
never reads/writes files and never touches a session.
"""

from __future__ import annotations


def qc_filter_mask(adata, *, min_genes: int = 200, max_pct_mt: float = 20.0,
                   min_counts: int = 0, max_doublet_score: float | None = None,
                   drop_predicted_doublets: bool = False):
    """Boolean keep-mask for QC cutoffs (cells to RETAIN). Pure scanpy/anndata.

    Keeps cells with enough genes (``n_genes_by_counts >= min_genes``) and low mito
    (``pct_counts_mt <= max_pct_mt``); optional total-count floor, doublet-score ceiling
    (cells with no score are kept), and dropping predicted doublets.
    """
    keep = (adata.obs["n_genes_by_counts"] >= min_genes) & (adata.obs["pct_counts_mt"] <= max_pct_mt)
    if min_counts > 0 and "total_counts" in adata.obs:
        keep &= adata.obs["total_counts"] >= min_counts
    if max_doublet_score is not None and "doublet_score" in adata.obs:
        ds = adata.obs["doublet_score"]
        keep &= (ds <= max_doublet_score) | ds.isna()      # keep cells with no score
    if drop_predicted_doublets and "predicted_doublet" in adata.obs:
        keep &= ~adata.obs["predicted_doublet"].fillna(False).astype(bool)
    return keep.values


def preprocess(adata, *, target_sum: float = 1e4, n_top_genes: int = 2000,
               hvg_batch_key: str | None = None, min_cells_per_batch: int = 1000,
               n_pcs: int = 50, seed: int = 0):
    """normalize_total → log1p → HVG(seurat_v3, batch-aware) → PCA from the counts layer (in place).

    Pure scanpy/anndata. Returns ``(adata, info)`` where info carries the resolved batch key, HVG
    count, PCA variance ratio, elbow suggestion, and any warnings. Requires a ``counts`` layer.
    """
    import numpy as np
    import scanpy as sc

    warnings: list = []
    _BATCH_OFF = {"none", "", "off", "false", "no"}

    # normalize + log1p from counts (reproducible regardless of current X)
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    # I-14: no 'scale.data' layer — it was a byte-for-byte duplicate of X (both log1p; no z-scaling
    # is done here), so it doubled RAM + every checkpoint for no gain. X IS the log-norm layer;
    # markers/annotation read X by default (layer=None) and fall back to X if 'scale.data' is absent.

    # HVG batch-key resolution (explicit OFF token → global; else auto-detect a sample-like column)
    batch_off = isinstance(hvg_batch_key, str) and hvg_batch_key.strip().lower() in _BATCH_OFF
    batch_key = None if batch_off else hvg_batch_key
    if batch_off:
        warnings.append("hvg_batch_key disabled — global (non-batch-aware) HVG")
    if batch_key and batch_key not in adata.obs.columns:
        warnings.append(f"hvg_batch_key '{batch_key}' absent — HVG computed without batch")
        batch_key = None
    if batch_key is None and not batch_off:
        for cand in ("sample_id", "sample", "batch", "donor", "patient"):
            if cand in adata.obs.columns:
                try:
                    nu = int(adata.obs[cand].nunique(dropna=True))
                except Exception:  # noqa: BLE001
                    continue
                if 2 <= nu <= 200:
                    batch_key = cand
                    warnings.append(f"hvg_batch_key auto-detected: '{cand}' (n={nu}); pass hvg_batch_key to override")
                    break
    # tiny-batch guard: seurat_v3 fits a per-batch loess; too-small batches make it singular
    if batch_key is not None:
        vc = adata.obs[batch_key].value_counts()
        tiny = vc[vc < min_cells_per_batch]
        if len(tiny):
            warnings.append(
                f"hvg_batch_key '{batch_key}' has {len(tiny)} batch(es) < {min_cells_per_batch} cells "
                f"(smallest={int(vc.min())}: {list(tiny.index[:3])}) — batch-aware HVG disabled")
            batch_key = None
    n_top = min(n_top_genes, adata.n_vars)
    try:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top,
                                    layer="counts", batch_key=batch_key)
    except (ValueError, np.linalg.LinAlgError) as exc:
        if batch_key is None:
            raise
        warnings.append(f"batch-aware HVG failed ({type(exc).__name__}: {exc}); retried without batch")
        batch_key = None
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top,
                                    layer="counts", batch_key=None)
    n_hvg = int(adata.var["highly_variable"].sum())

    # PCA on HVGs (no global scale; svd centers)
    max_comps = max(1, min(n_pcs, adata.n_vars - 1, adata.n_obs - 1))
    sc.pp.pca(adata, n_comps=max_comps, mask_var="highly_variable", random_state=seed)
    vr = [round(float(x), 5) for x in adata.uns["pca"]["variance_ratio"]]
    _vr = np.asarray(vr, dtype=float)                 # elbow: last PC with variance ratio >= 0.01
    _above = np.where(_vr >= 0.01)[0]
    suggested = int(_above[-1] + 1) if _above.size else min(10, len(vr))

    info = {"batch_key": batch_key, "n_hvg": n_hvg, "n_pcs": max_comps,
            "variance_ratio": vr, "suggested_n_pcs_elbow": suggested, "warnings": warnings}
    return adata, info


def cluster(adata, *, use_rep: str = "X_pca", n_neighbors: int = 15, n_pcs: int | None = None,
            resolution: float | None = None, seed: int = 0, key_added: str | None = None,
            key_suffix: str | None = None):
    """neighbors → leiden → umap on an embedding (in place). Pure scanpy. Returns ``(adata, info)``.

    Per-model key namespacing (baseline X_pca → leiden/X_umap; else leiden_<suf>/X_umap_<suf>) so
    baseline and integration reductions coexist. ``resolution`` defaults to 0.25.
    """
    import scanpy as sc

    if resolution is None:
        resolution = 0.25
    suf = use_rep.removeprefix("X_").lower() if use_rep not in ("X_pca", "X_PCA") else ""
    if key_suffix is not None:
        suf = key_suffix
    leiden_key = key_added or (f"leiden_{suf}" if suf else "leiden")
    umap_key = f"X_umap_{suf}" if suf else "X_umap"
    nkey = f"neighbors_{suf}" if suf else None        # None → scanpy default graph keys (baseline)

    rep_dim = adata.obsm[use_rep].shape[1]
    use_pcs = min(n_pcs, rep_dim) if n_pcs else rep_dim
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=use_pcs, use_rep=use_rep,
                    random_state=seed, key_added=nkey)
    sc.tl.leiden(adata, resolution=resolution, key_added=leiden_key, flavor="igraph",
                 n_iterations=2, directed=False, random_state=seed, neighbors_key=nkey)
    prev_umap = adata.obsm["X_umap"].copy() if "X_umap" in adata.obsm else None
    sc.tl.umap(adata, random_state=seed, neighbors_key=nkey)
    if umap_key != "X_umap":
        adata.obsm[umap_key] = adata.obsm["X_umap"]
        if prev_umap is not None:
            adata.obsm["X_umap"] = prev_umap          # restore baseline
        else:
            del adata.obsm["X_umap"]

    sizes = adata.obs[leiden_key].value_counts().sort_index()
    info = {"leiden_key": leiden_key, "umap_key": umap_key, "neighbors_key": nkey,
            "model_suffix": suf or "baseline", "use_rep": use_rep, "use_pcs": use_pcs,
            "resolution": resolution, "n_clusters": int(sizes.shape[0]),
            "cluster_sizes": {str(k): int(v) for k, v in sizes.items()}}
    return adata, info


def markers_rank(adata, *, groupby: str = "leiden", layer: str | None = None,
                 max_genes_ranked: int | None = 5000):
    """Wilcoxon ``rank_genes_groups`` per cluster (pts=True), in place into ``uns``. Pure scanpy.

    The DE method is FIXED to Wilcoxon — this is the deterministic marker evidence for annotation,
    not a tunable. Large-data tractability is the output cap ``max_genes_ranked``. Returns
    ``(adata, n_genes_ranked)``.
    """
    import scanpy as sc

    use_layer = layer if (layer and layer in adata.layers) else None
    rank_n = int(adata.n_vars if max_genes_ranked is None else min(max_genes_ranked, adata.n_vars))
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon", layer=use_layer,
                            use_raw=False, n_genes=rank_n, pts=True)
    return adata, rank_n


def apply_annotation(adata, *, groupby: str = "leiden", labels: dict | None = None,
                     key: str = "major_cell_type", confidence: dict | None = None,
                     review_required: dict | None = None, cell_state: dict | None = None,
                     marker_sets: dict | None = None, tissue: str | None = None,
                     method: str = "DE_LLM_marker_free", unassigned: str = "Unassigned"):
    """Write the LLM's cluster→label map into ``obs[key]`` (+ optional confidence / review_required /
    cell_state columns, and the chosen marker_sets into ``uns['scpilot_annotation']['tier1_llm']``).

    Deterministic given the map (the LLM's judgment is fully replayable). Pure pandas/anndata.
    Returns ``(adata, info)`` with label distribution / unlabeled clusters / unassigned count.
    """
    obs_g = adata.obs[groupby].astype(str)
    clusters = set(obs_g.unique())
    lab = {str(k): str(v) for k, v in (labels or {}).items()}
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

    msets = {str(k): [str(g) for g in v] for k, v in (marker_sets or {}).items()}
    adata.uns.setdefault("scpilot_annotation", {})
    adata.uns["scpilot_annotation"]["tier1_llm"] = {
        "method": method, "groupby": groupby, "label_key": key, "tissue_context": tissue,
        "labels": lab, "marker_sets": msets, "marker_db_used": False,
    }
    dist = adata.obs[key].value_counts().to_dict()
    n_unassigned = int((adata.obs[key].astype(str) == unassigned).sum())
    info = {"label_key": key, "labels": lab, "marker_sets": msets, "unlabeled_clusters": missing,
            "n_unassigned_cells": n_unassigned,
            "label_distribution": {str(k): int(v) for k, v in dist.items()}}
    return adata, info


def suggest_resolution(sweep, *, jump_ratio: float = 1.5):
    """Pick a resolution from the sweep (I-22). Entries are ``(resolution, n_clusters[, silhouette])``.

    If a SEPARABILITY score (embedding silhouette) is present, choose the resolution with the BEST
    separation among those with ≥2 clusters — this reflects whether clusters ACTUALLY separate, not
    just how many there are: over-clustering a noisy/sparse region lowers silhouette (so it is not
    chosen), and merging distinct populations also lowers it. If no silhouette is available (legacy
    2-tuples), fall back to the n_clusters KNEE (value just before an n_clusters jump ≥ ``jump_ratio``).
    Returns ``(resolution, rationale)``. Pure python."""
    res = [e[0] for e in sweep]
    ncl = [e[1] for e in sweep]
    sil = [(e[2] if len(e) > 2 else None) for e in sweep]

    scored = [(r, s) for r, n, s in zip(res, ncl, sil) if s is not None and n >= 2]
    if scored:
        best_r, best_s = max(scored, key=lambda x: x[1])
        return best_r, (f"chose res {best_r} — highest embedding separability (silhouette {best_s:.3f}) "
                        f"among resolutions with ≥2 clusters; reflects real cluster separation, not raw "
                        f"n_clusters (guards against over-clustering sparse/noisy regions)")
    # fallback: n_clusters knee (no separability signal available)
    for i in range(len(sweep) - 1):
        if ncl[i + 1] >= max(ncl[i], 1) * jump_ratio:
            return res[i], (f"n_clusters jumps {ncl[i]}→{ncl[i + 1]} at res {res[i + 1]} (≥{jump_ratio}×); "
                            f"chose res {res[i]} just before the jump")
    return res[0], (f"no abrupt jump (≥{jump_ratio}×) over res {res[0]}–{res[-1]}; "
                    f"chose conservative lowest res {res[0]}")


def cluster_sweep(adata, *, use_rep: str = "X_pca", res_min: float = 0.1, res_max: float = 0.5,
                  res_step: float = 0.1, n_neighbors: int = 15, n_pcs: int | None = None,
                  seed: int = 0, silhouette_subsample: int = 5000):
    """Sweep leiden resolution on an embedding → ordered list of ``(resolution, n_clusters, silhouette)``.

    The per-resolution SILHOUETTE (mean over a seeded subsample of ``use_rep``) is a SEPARABILITY score
    (I-22): it says whether the clusters at that resolution actually separate in the embedding, so
    resolution can be chosen by real structure rather than raw cluster count. ``silhouette`` is ``None``
    when n_clusters < 2 (undefined). NON-MUTATING: temporary neighbor/leiden keys are removed. Pure scanpy.
    (Pair with ``suggest_resolution``.)
    """
    import numpy as np
    import scanpy as sc
    from sklearn.metrics import silhouette_score

    rep = adata.obsm[use_rep]
    rep_dim = rep.shape[1]
    use_pcs = min(n_pcs, rep_dim) if n_pcs else rep_dim
    n_steps = int(round((res_max - res_min) / res_step)) + 1
    grid = [round(res_min + i * res_step, 4) for i in range(max(1, n_steps))]
    # one seeded subsample reused across resolutions → tractable + comparable silhouettes at scale
    n = adata.n_obs
    if n <= silhouette_subsample:
        sidx = np.arange(n)
    else:
        sidx = np.sort(np.random.default_rng(seed).choice(n, silhouette_subsample, replace=False))
    Xsub = np.asarray(rep[sidx, :use_pcs])
    nkey, lkey = "_sweep_nbr", "_sweep_leiden"
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=use_pcs, use_rep=use_rep,
                    random_state=seed, key_added=nkey)
    sweep: list = []
    try:
        for r in grid:
            sc.tl.leiden(adata, resolution=r, key_added=lkey, flavor="igraph",
                         n_iterations=2, directed=False, random_state=seed, neighbors_key=nkey)
            n_cl = int(adata.obs[lkey].nunique())
            sil = None
            if n_cl >= 2:
                lab_sub = adata.obs[lkey].to_numpy()[sidx]
                if len(set(lab_sub)) >= 2:
                    try:
                        sil = round(float(silhouette_score(Xsub, lab_sub)), 4)
                    except Exception:  # noqa: BLE001 — silhouette must never break the sweep
                        sil = None
            sweep.append((r, n_cl, sil))
    finally:
        adata.obs.drop(columns=[lkey], errors="ignore", inplace=True)
        adata.uns.pop(nkey, None)
        adata.uns.pop(lkey, None)      # sc.tl.leiden(key_added=lkey) also writes uns[lkey] (params dict)
        for k in (f"{nkey}_distances", f"{nkey}_connectivities"):
            if k in adata.obsp:
                del adata.obsp[k]
    return sweep


def integrate_harmony(adata, *, batch_key: str = "GSM", use_rep: str = "X_pca",
                      out_key: str = "X_harmony", seed: int = 0):
    """Harmony batch integration via harmonypy (external lib, not scpilot) → ``obsm[out_key]``.

    Pure: needs only harmonypy + numpy. Deterministic under a fixed ``seed``. Returns
    ``(adata, info)`` with the output key + corrected dimensionality.
    """
    import harmonypy
    import numpy as np

    ho = harmonypy.run_harmony(adata.obsm[use_rep], adata.obs, [batch_key], random_state=seed)
    Z = np.asarray(ho.Z_corr)                          # harmonypy 0.2.0 torch → numpy
    Z = Z.T if Z.shape[0] == adata.obsm[use_rep].shape[1] else Z   # (cells × dims)
    adata.obsm[out_key] = Z
    return adata, {"out_key": out_key, "n_dims": int(Z.shape[1])}


def majority_vote(adata, keys, *, min_agreement: float = 0.5, ambiguous_label: str = "ambiguous"):
    """Per-cell majority vote across obs annotation columns ``keys``. Returns (out, agree, pairwise):
    ``out[i]`` = the unique-winner label held by a fraction > ``min_agreement`` of the columns
    (else ``ambiguous_label``); ``agree[i]`` = winner fraction; ``pairwise`` = per-pair concordance.
    Pure numpy/pandas."""
    import numpy as np
    import pandas as pd

    n_keys = len(keys)
    cats = pd.unique(np.concatenate([adata.obs[k].astype(str).values for k in keys]))
    codes = np.column_stack([
        pd.Categorical(adata.obs[k].astype(str).values, categories=cats).codes for k in keys])
    # I-15: vectorized per-cell majority (was an O(n_obs) Python loop — minutes-to-hours at 5M cells).
    # n_keys is small (# annotation columns being voted, ~2–5), so the (n, k, k) equality tensor is cheap.
    # freq[i,j] = how many of the k columns share column j's label for cell i.
    eq = codes[:, :, None] == codes[:, None, :]           # (n_obs, k, k) bool
    freq = eq.sum(axis=2)                                 # (n_obs, k)
    mx = freq.max(axis=1)                                 # top label frequency per cell
    agree = mx / n_keys
    is_max = freq == mx[:, None]
    # a UNIQUE winner ⇔ all columns achieving mx carry the SAME label (min==max of their codes)
    wmin = np.where(is_max, codes, codes.max() + 1).min(axis=1)
    wmax = np.where(is_max, codes, -1).max(axis=1)
    win_ok = (wmin == wmax) & (agree > min_agreement)
    out = np.where(win_ok, cats[wmin], ambiguous_label).astype(object)
    pairwise = {}
    for a in range(n_keys):
        for b in range(a + 1, n_keys):
            pairwise[f"{keys[a]}__vs__{keys[b]}"] = round(float((codes[:, a] == codes[:, b]).mean()), 3)
    return out, agree, pairwise


def consensus_vote(adata, *, keys, out_key: str = "celltype_consensus", min_agreement: float = 0.5,
                   ambiguous_label: str = "ambiguous"):
    """Embedding-independent consensus label by majority vote across ``keys`` → writes
    ``obs[out_key]`` (+ ``obs[out_key+'_agreement']``). Pure numpy/pandas; uses ``majority_vote``.
    Returns ``(adata, info)`` with distribution / n_ambiguous / pairwise / mean_agreement."""
    import pandas as pd

    out, agree, pairwise = majority_vote(adata, keys, min_agreement=min_agreement,
                                         ambiguous_label=ambiguous_label)
    adata.obs[out_key] = pd.Categorical(out)
    adata.obs[f"{out_key}_agreement"] = agree.astype("float32")
    dist = adata.obs[out_key].value_counts().to_dict()
    n_amb = int((adata.obs[out_key].astype(str) == ambiguous_label).sum())
    info = {"out_key": out_key, "source_keys": list(keys), "pairwise": pairwise,
            "n_ambiguous": n_amb, "mean_agreement": round(float(agree.mean()), 3),
            "label_distribution": {str(k): int(v) for k, v in dist.items()}}
    return adata, info
