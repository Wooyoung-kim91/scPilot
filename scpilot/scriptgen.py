"""Standalone (scpilot-FREE) emission for generated per-step tutorial scripts.

Each ``code/NN_<stage>.py`` reads like a hand-written tutorial (cf. ``…/Atherosclerosis/.../code/``):
the actual scanpy/anndata/pandas operations written DIRECTLY in the script body — no ``tools.run``,
no ``Session``, and NO helper-function wrapper that just re-packages a tool. Parameters are baked in
(or assigned as plain editable variables at the top), and the step reads the previous step's
``standalone_data/NN_<stage>.h5ad`` and writes its own, so the files chain when run IN ORDER.

The direct code is a transcription of the corresponding ``scpilot.recipes`` logic (which the scpilot
tool calls). The two are kept in lock-step by the regression tests in
``tests/test_scriptgen_equivalence.py``, which run each generated script as a real subprocess and
assert its result equals the scpilot tool's — the user-chosen "verify by regression test" contract.

Rollout: ``EMITTERS`` maps a stage to its body-emitter. Stages without an emitter fall back (in
``session._step_script``) to a thin scpilot-backed step that reads/writes the SAME h5ad chain, so
converting a tool to standalone is a drop-in replacement and the chain never breaks mid-rollout.
"""

from __future__ import annotations


def _g(params: dict, key, default):
    return (params or {}).get(key, default)


# ---- per-stage body emitters: DIRECT plain-scanpy code (no function wrapper) ----------------
# Convention: the header has already set `IN`, `OUT`, `_DATA` and imported sc / np / pd. Each
# emitter returns statements that read IN, operate, write OUT, and print a one-line summary.

def _qc_filter(cid: str, params: dict) -> str:
    mg = _g(params, "min_genes", 200)
    mpct = _g(params, "max_pct_mt", 20.0)
    mc = _g(params, "min_counts", 0)
    mds = _g(params, "max_doublet_score", None)
    dpd = _g(params, "drop_predicted_doublets", False)
    L = ["adata = sc.read_h5ad(IN)",
         "_n0 = adata.n_obs",
         "",
         "# keep cells passing the QC cutoffs (cells to RETAIN)",
         f'keep = (adata.obs["n_genes_by_counts"] >= {mg}) & (adata.obs["pct_counts_mt"] <= {mpct})']
    if mc and mc > 0:
        L.append(f'keep &= adata.obs["total_counts"] >= {mc}')
    if mds is not None:
        L.append('if "doublet_score" in adata.obs:')
        L.append(f'    _ds = adata.obs["doublet_score"]')
        L.append(f'    keep &= (_ds <= {mds}) | _ds.isna()        # cells with no doublet score are kept')
    if dpd:
        L.append('if "predicted_doublet" in adata.obs:')
        L.append('    keep &= ~adata.obs["predicted_doublet"].fillna(False).astype(bool)')
    L += ["",
          "adata = adata[keep.values].copy()",
          "adata.write_h5ad(OUT)",
          f'print("[{cid}] qc_filter: kept %d/%d cells -> %s" % (adata.n_obs, _n0, OUT.name))']
    return "\n".join(L) + "\n"


def _preprocess(cid: str, params: dict) -> str:
    return f'''adata = sc.read_h5ad(IN)

# parameters (edit freely)
target_sum = {_g(params, "target_sum", 1e4)!r}
n_top_genes = {_g(params, "n_top_genes", 2000)!r}
hvg_batch_key = {_g(params, "hvg_batch_key", None)!r}
min_cells_per_batch = {_g(params, "min_cells_per_batch", 1000)!r}
n_pcs = {_g(params, "n_pcs", 50)!r}
seed = {_g(params, "seed", 0)!r}

# normalize_total -> log1p from the raw counts layer (reproducible regardless of current X)
adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata, target_sum=target_sum)
sc.pp.log1p(adata)
# X holds the log-normalized values (no duplicate 'scale.data' layer); markers/annotation read X

# resolve the HVG batch key: explicit OFF token -> global; else auto-detect a sample-like column
batch_off = isinstance(hvg_batch_key, str) and hvg_batch_key.strip().lower() in {{"none", "", "off", "false", "no"}}
batch_key = None if batch_off else hvg_batch_key
if batch_key and batch_key not in adata.obs.columns:
    batch_key = None
if batch_key is None and not batch_off:
    for _cand in ("sample_id", "sample", "batch", "donor", "patient"):
        if _cand in adata.obs.columns and 2 <= int(adata.obs[_cand].nunique(dropna=True)) <= 200:
            batch_key = _cand
            break
# tiny-batch guard: seurat_v3 fits a per-batch loess; too-small batches make it singular
if batch_key is not None and (adata.obs[batch_key].value_counts() < min_cells_per_batch).any():
    print("    note: batch-aware HVG disabled (a batch has <", min_cells_per_batch, "cells)")
    batch_key = None

n_top = min(n_top_genes, adata.n_vars)
try:
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top, layer="counts", batch_key=batch_key)
except Exception:                                  # singular per-batch loess -> retry global
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top, layer="counts", batch_key=None)
    batch_key = None

# PCA on the HVGs (no global scale; SVD centers internally)
n_comps = max(1, min(n_pcs, adata.n_vars - 1, adata.n_obs - 1))
sc.pp.pca(adata, n_comps=n_comps, mask_var="highly_variable", random_state=seed)

adata.write_h5ad(OUT)
print("[{cid}] preprocess: n_hvg=%d n_pcs=%d batch_key=%s -> %s"
      % (int(adata.var["highly_variable"].sum()), n_comps, batch_key, OUT.name))
'''


def _cluster(cid: str, params: dict) -> str:
    suffix_line = (f'suf = {_g(params, "key_suffix", None)!r}'
                   if _g(params, "key_suffix", None) is not None else "# (no key_suffix override)")
    return f'''adata = sc.read_h5ad(IN)

# parameters (edit freely)
use_rep = {_g(params, "use_rep", "X_pca")!r}
resolution = {_g(params, "resolution", None)!r}
if resolution is None:
    resolution = 0.25                              # default leiden resolution at every stage
n_neighbors = {_g(params, "n_neighbors", 15)!r}
n_pcs = {_g(params, "n_pcs", None)!r}
seed = {_g(params, "seed", 0)!r}
key_added = {_g(params, "key_added", None)!r}

# per-model key namespacing so baseline (X_pca) + integration reductions coexist
suf = "" if use_rep in ("X_pca", "X_PCA") else use_rep.removeprefix("X_").lower()
{suffix_line}
leiden_key = key_added or ("leiden_" + suf if suf else "leiden")
umap_key = "X_umap_" + suf if suf else "X_umap"
nkey = "neighbors_" + suf if suf else None         # None -> scanpy default graph keys (baseline)

rep_dim = adata.obsm[use_rep].shape[1]
use_pcs = min(n_pcs, rep_dim) if n_pcs else rep_dim
sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=use_pcs, use_rep=use_rep,
                random_state=seed, key_added=nkey)
sc.tl.leiden(adata, resolution=resolution, key_added=leiden_key, flavor="igraph",
             n_iterations=2, directed=False, random_state=seed, neighbors_key=nkey)
# umap always writes obsm["X_umap"]; preserve any baseline by moving the result
_prev_umap = adata.obsm["X_umap"].copy() if "X_umap" in adata.obsm else None
sc.tl.umap(adata, random_state=seed, neighbors_key=nkey)
if umap_key != "X_umap":
    adata.obsm[umap_key] = adata.obsm["X_umap"]
    if _prev_umap is not None:
        adata.obsm["X_umap"] = _prev_umap
    else:
        del adata.obsm["X_umap"]

adata.write_h5ad(OUT)
print("[{cid}] cluster: %d clusters (key=%s) -> %s"
      % (adata.obs[leiden_key].nunique(), leiden_key, OUT.name))
'''


def _cluster_sweep(cid: str, params: dict) -> str:
    return f'''adata = sc.read_h5ad(IN)

# parameters (edit freely)
use_rep = {_g(params, "use_rep", "X_pca")!r}
res_min, res_max, res_step = {_g(params, "res_min", 0.1)!r}, {_g(params, "res_max", 0.5)!r}, {_g(params, "res_step", 0.1)!r}
n_neighbors = {_g(params, "n_neighbors", 15)!r}
n_pcs = {_g(params, "n_pcs", None)!r}
jump_ratio = {_g(params, "jump_ratio", 1.5)!r}
seed = {_g(params, "seed", 0)!r}

# sweep leiden resolution and record n_clusters at each (NON-MUTATING: temp keys removed at the end)
rep_dim = adata.obsm[use_rep].shape[1]
use_pcs = min(n_pcs, rep_dim) if n_pcs else rep_dim
grid = [round(res_min + i * res_step, 4) for i in range(max(1, int(round((res_max - res_min) / res_step)) + 1))]
sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=use_pcs, use_rep=use_rep,
                random_state=seed, key_added="_sweep_nbr")
sweep = []
for r in grid:
    sc.tl.leiden(adata, resolution=r, key_added="_sweep_leiden", flavor="igraph",
                 n_iterations=2, directed=False, random_state=seed, neighbors_key="_sweep_nbr")
    sweep.append((r, int(adata.obs["_sweep_leiden"].nunique())))

# knee: the resolution JUST BEFORE n_clusters jumps by >= jump_ratio (else the lowest)
suggested = sweep[0][0]
for i in range(len(sweep) - 1):
    if sweep[i + 1][1] >= max(sweep[i][1], 1) * jump_ratio:
        suggested = sweep[i][0]
        break

adata.obs.drop(columns=["_sweep_leiden"], errors="ignore", inplace=True)
adata.uns.pop("_sweep_nbr", None)
for _k in ("_sweep_nbr_distances", "_sweep_nbr_connectivities"):
    if _k in adata.obsp:
        del adata.obsp[_k]
pd.DataFrame(sweep, columns=["resolution", "n_clusters"]).to_csv(_DATA / (OUT.stem + "_sweep.csv"), index=False)
adata.write_h5ad(OUT)
print("[{cid}] cluster_sweep: suggested resolution=%s -> %s" % (suggested, OUT.name))
'''


def _markers(cid: str, params: dict) -> str:
    return f'''adata = sc.read_h5ad(IN)

# parameters (edit freely)
groupby = {_g(params, "groupby", "leiden")!r}
layer = {_g(params, "layer", None)!r}
max_genes_ranked = {_g(params, "max_genes_ranked", 5000)!r}

# Wilcoxon rank-sum DE per cluster (FIXED method for marker genes; pts=True -> expressed fractions)
use_layer = layer if (layer and layer in adata.layers) else None
rank_n = adata.n_vars if max_genes_ranked is None else min(max_genes_ranked, adata.n_vars)
sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon", layer=use_layer,
                        use_raw=False, n_genes=int(rank_n), pts=True)
de = sc.get.rank_genes_groups_df(adata, group=None)        # long-form ranked DE table
de.to_csv(_DATA / (OUT.stem + "_table.csv"), index=False)
adata.write_h5ad(OUT)
print("[{cid}] markers: ranked %d genes x %d groups -> %s" % (int(rank_n), de["group"].nunique(), OUT.name))
'''


def _apply_annotation(cid: str, params: dict) -> str:
    return f'''adata = sc.read_h5ad(IN)

# the LLM's DE-based calls (edit freely)
groupby = {_g(params, "groupby", "leiden")!r}
key = {_g(params, "key", "major_cell_type")!r}
unassigned = {_g(params, "unassigned", "Unassigned")!r}
labels = {_g(params, "labels", {}) !r}        # cluster -> cell type
confidence = {_g(params, "confidence", None)!r}
review_required = {_g(params, "review_required", None)!r}
cell_state = {_g(params, "cell_state", None)!r}
marker_sets = {_g(params, "marker_sets", None)!r}

_g = adata.obs[groupby].astype(str)
lab = {{str(k): str(v) for k, v in (labels or {{}}).items()}}
adata.obs[key] = _g.map(lambda c: lab.get(c, unassigned)).astype("category")
if confidence:
    _cf = {{str(k): float(v) for k, v in confidence.items()}}
    adata.obs[key + "_confidence"] = _g.map(lambda c: _cf.get(c, float("nan"))).astype(float)
if review_required:
    _rv = {{str(k): bool(v) for k, v in review_required.items()}}
    adata.obs[key + "_review_required"] = _g.map(lambda c: _rv.get(c, False)).astype(bool)
if cell_state:
    _cs = {{str(k): str(v) for k, v in cell_state.items()}}
    adata.obs["cell_state"] = _g.map(lambda c: _cs.get(c, "")).astype("category")
_msets = {{str(k): [str(x) for x in v] for k, v in (marker_sets or {{}}).items()}}
adata.uns.setdefault("scpilot_annotation", {{}})
adata.uns["scpilot_annotation"]["tier1_llm"] = {{
    "method": {_g(params, "method", "DE_LLM_marker_free")!r}, "groupby": groupby, "label_key": key,
    "tissue_context": {_g(params, "tissue", None)!r}, "labels": lab, "marker_sets": _msets,
    "marker_db_used": False,
}}
_n_un = int((adata.obs[key].astype(str) == unassigned).sum())
adata.write_h5ad(OUT)
print("[{cid}] apply_annotation: labeled %d clusters, %d unassigned -> %s" % (len(lab), _n_un, OUT.name))
'''


def _vote_body(cid: str, toolname: str, params: dict, default_out_key: str) -> str:
    """Direct majority-vote consensus (shared by consensus_annotation + harmonize_annotations)."""
    return f'''adata = sc.read_h5ad(IN)

# parameters (edit freely)
keys = {_g(params, "keys", []) !r}
out_key = {_g(params, "out_key", default_out_key)!r}
min_agreement = {_g(params, "min_agreement", 0.5)!r}
ambiguous_label = {_g(params, "ambiguous_label", "ambiguous")!r}

# per-cell majority vote across the annotation columns (embedding-independent)
_cats = pd.unique(np.concatenate([adata.obs[k].astype(str).values for k in keys]))
_codes = np.column_stack([pd.Categorical(adata.obs[k].astype(str).values, categories=_cats).codes for k in keys])
_out = np.empty(adata.n_obs, dtype=object)
_agree = np.zeros(adata.n_obs, dtype=float)
for i in range(adata.n_obs):
    _counts = np.bincount(_codes[i], minlength=len(_cats))
    _mx = _counts.max()
    _win = np.flatnonzero(_counts == _mx)
    _agree[i] = _mx / len(keys)
    _out[i] = _cats[_win[0]] if (_win.size == 1 and _mx / len(keys) > min_agreement) else ambiguous_label
adata.obs[out_key] = pd.Categorical(_out)
adata.obs[out_key + "_agreement"] = _agree.astype("float32")
_n_amb = int((adata.obs[out_key].astype(str) == ambiguous_label).sum())
adata.write_h5ad(OUT)
print("[{cid}] {toolname}: %d ambiguous cells, key=%s -> %s" % (_n_amb, out_key, OUT.name))
'''


def _qc_metrics(cid: str, params: dict) -> str:
    sample_key = _g(params, "sample_key", "sample_id")
    mito_prefix = _g(params, "mito_prefix", None)
    mlg = _g(params, "mixed_lineage_genes", None)
    run_scrublet = _g(params, "run_scrublet", True)
    n_mads = _g(params, "n_mads", 5.0)
    seed = _g(params, "seed", 0)

    if mito_prefix is not None:
        mito_block = f'mito_prefix = {mito_prefix!r}'
    else:
        mito_block = (
            "# detect the mito-gene style from the data (mouse 'mt-' vs human 'MT-'); never assume\n"
            "_v = adata.var_names\n"
            'if _v.str.startswith("mt-").any() and not _v.str.startswith("MT-").any():\n'
            '    mito_prefix = "mt-"\n'
            'elif _v.str.startswith("MT-").any() and not _v.str.startswith("mt-").any():\n'
            '    mito_prefix = "MT-"\n'
            "else:\n"
            '    mito_prefix = "mt-" if float(_v.str.isupper().mean()) <= 0.2 else "MT-"')

    if mlg:
        mixed_block = f'''_want = {list(mlg)!r}
_idx = {{str(n).upper(): n for n in adata.var_names}}
_resolved = [_idx.get(str(g).upper()) for g in _want]
if all(r is not None for r in _resolved):
    _src = adata.layers["counts"] if "counts" in adata.layers else adata.X
    _sub = _src[:, [adata.var_names.get_loc(g) for g in _resolved]]
    _dense = _sub.toarray() if hasattr(_sub, "toarray") else np.asarray(_sub)
    adata.obs["mixed_lineage_flag"] = (_dense > 0).all(axis=1)
else:
    adata.obs["mixed_lineage_flag"] = False'''
    else:
        mixed_block = 'adata.obs["mixed_lineage_flag"] = False        # opt-in; no gene pair given'

    scrublet_block = (f'''import anndata as ad
_counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
_sv = adata.obs[{sample_key!r}].astype(str).values
_scores = np.full(adata.n_obs, np.nan, dtype=float)
_preds = np.zeros(adata.n_obs, dtype=bool)
for _sid in np.unique(_sv):
    _ix = np.where(_sv == _sid)[0]
    if _ix.size < 30:                              # scrublet needs enough cells
        continue
    _ss = ad.AnnData(X=_counts[_ix, :].copy())
    try:
        sc.pp.scrublet(_ss, random_state={seed!r})
        _scores[_ix] = _ss.obs["doublet_score"].to_numpy()
        _preds[_ix] = _ss.obs["predicted_doublet"].to_numpy()
    except Exception:
        pass
adata.obs["doublet_score"] = _scores
adata.obs["predicted_doublet"] = _preds''' if run_scrublet else
        '# run_scrublet=False: doublet scoring skipped')

    return f'''adata = sc.read_h5ad(IN)

# parameters (edit freely)
sample_key = {sample_key!r}
n_mads = {n_mads!r}

{mito_block}

# QC metrics on the COUNTS layer (X may already be normalized on the merged object)
_up = adata.var_names.str.upper()
adata.var["mt"] = _up.str.startswith(mito_prefix.upper())
adata.var["ribo"] = _up.str.startswith(("RPS", "RPL"))
_has_counts = "counts" in adata.layers
_xb = adata.X
if _has_counts:
    adata.X = adata.layers["counts"]
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], inplace=True, percent_top=None, log1p=False)
if _has_counts:
    adata.X = _xb

# mixed-lineage co-expression flag (opt-in)
{mixed_block}

# per-sample scrublet doublet scores (per library, on the counts layer)
{scrublet_block}

# MAD-based suggested cutoffs — EVIDENCE for choosing qc_filter thresholds (not auto-applied)
_cut = {{}}
for _m, _lo, _hi in (("n_genes_by_counts", "min_genes", "max_genes"), ("total_counts", "min_counts", "max_counts")):
    if _m in adata.obs:
        _a = np.log1p(adata.obs[_m].to_numpy(dtype=float)); _a = _a[np.isfinite(_a)]
        _md = np.median(_a); _mad = np.median(np.abs(_a - _md)) * 1.4826
        _cut[_lo] = int(max(0, round(np.expm1(_md - n_mads * _mad))))
        _cut[_hi] = int(max(0, round(np.expm1(_md + n_mads * _mad))))
if "pct_counts_mt" in adata.obs:
    _a = adata.obs["pct_counts_mt"].to_numpy(dtype=float); _a = _a[np.isfinite(_a)]
    _md = np.median(_a); _mad = np.median(np.abs(_a - _md)) * 1.4826
    _cut["max_pct_mt"] = round(float(_md + n_mads * _mad), 2)

adata.write_h5ad(OUT)
print("[{cid}] qc_metrics: mito_prefix=%s -> %s" % (mito_prefix, OUT.name))
print("    suggested cutoffs (MAD, choose qc_filter thresholds from these):", _cut)
'''


def _annotation_audit(cid: str, params: dict) -> str:
    return f'''import json
adata = sc.read_h5ad(IN)

# parameters (edit freely) — marker-criteria bar + flag thresholds
groupby = {_g(params, "groupby", "leiden")!r}
label_key = {_g(params, "label_key", "major_cell_type")!r}
fine_key, facs_key, final_key = {_g(params, "fine_key", "fine_cell_type")!r}, {_g(params, "facs_key", "facs_style_label")!r}, {_g(params, "final_key", "final_annotation")!r}
malignancy_key, cnv_status_key, cnv_score_key = {_g(params, "malignancy_key", "malignancy")!r}, {_g(params, "cnv_status_key", "cnv_status")!r}, {_g(params, "cnv_score_key", "cnv_score")!r}
sample_key, batch_key = {_g(params, "sample_key", "sample_id")!r}, {_g(params, "batch_key", "GSE")!r}
doublet_key, stress_key, mt_key = {_g(params, "doublet_key", "predicted_doublet")!r}, {_g(params, "stress_key", "stress_score")!r}, {_g(params, "mt_key", "pct_counts_mt")!r}
min_pct, min_lfc, padj_max = {_g(params, "min_pct", 0.25)!r}, {_g(params, "min_lfc", 1.0)!r}, {_g(params, "padj_max", 0.05)!r}
min_specificity, max_pct_out, top_k_markers = {_g(params, "min_specificity", 0.1)!r}, {_g(params, "max_pct_out", 0.5)!r}, {_g(params, "top_k_markers", 15)!r}
profile_similarity, single_source_frac = {_g(params, "profile_similarity", 0.5)!r}, {_g(params, "single_source_frac", 0.8)!r}
doublet_frac, max_pct_mt, min_marker_support = {_g(params, "doublet_frac", 0.5)!r}, {_g(params, "max_pct_mt", 25.0)!r}, {_g(params, "min_marker_support", 0.5)!r}

# case-insensitive symbol index; the LLM's OWN recorded marker-sets (no marker DB we own)
_sidx = {{}}
for _n in adata.var_names:
    _sidx.setdefault(str(_n).upper(), _n)
_msets_raw = (adata.uns.get("scpilot_annotation", {{}}).get("tier1_llm", {{}}) or {{}}).get("marker_sets", {{}}) or {{}}
marker_sets = {{str(ct): [_sidx[str(g).upper()] for g in gs if str(g).upper() in _sidx] for ct, gs in _msets_raw.items()}}

obs_g = adata.obs[groupby].astype(str)
clusters = list(obs_g.cat.categories) if hasattr(obs_g, "cat") else sorted(obs_g.unique())

# per-cluster DE: top specific markers + per-gene stats (the marker-criteria check)
de_by_cluster, de_stats_by_cluster = {{}}, {{}}
_rg = adata.uns.get("rank_genes_groups")
if _rg and _rg.get("params", {{}}).get("groupby") == groupby:
    _de = sc.get.rank_genes_groups_df(adata, group=None)
    for _cl, _sub in _de.groupby("group", observed=True):
        _s = _sub[_sub["pvals_adj"] < padj_max] if "pvals_adj" in _sub.columns else _sub
        if {{"pct_nz_group", "pct_nz_reference"}}.issubset(_s.columns):
            _spec = _s["pct_nz_group"] - _s["pct_nz_reference"]
            _s2 = _s.assign(_spec=_spec)
            _s2 = _s2[(_s2["_spec"] >= min_specificity) & (_s2["pct_nz_reference"] <= max_pct_out)].sort_values("_spec", ascending=False)
            de_by_cluster[str(_cl)] = [str(g) for g in _s2["names"].head(top_k_markers)]
        else:
            de_by_cluster[str(_cl)] = [str(g) for g in _s["names"].head(top_k_markers)]
        _stats = {{}}
        _haspct = {{"pct_nz_group", "pct_nz_reference"}}.issubset(_sub.columns)
        for _, _r in _sub.iterrows():
            _pin = float(_r["pct_nz_group"]) if _haspct else float("nan")
            _pout = float(_r["pct_nz_reference"]) if _haspct else float("nan")
            _stats[str(_r["names"])] = {{
                "logFC": round(float(_r["logfoldchanges"]), 3) if "logfoldchanges" in _sub.columns else None,
                "padj": float(_r["pvals_adj"]) if "pvals_adj" in _sub.columns else None,
                "pct_in": round(_pin, 3) if _haspct else None, "pct_out": round(_pout, 3) if _haspct else None,
                "spec": round(_pin - _pout, 3) if (_haspct and not np.isnan(_pin)) else None}}
        de_stats_by_cluster[str(_cl)] = _stats

def _check_marker(st):
    f = []
    if st.get("pct_in") is not None and st["pct_in"] < min_pct: f.append("pct")
    if st.get("logFC") is not None and st["logFC"] < min_lfc: f.append("lfc")
    if st.get("padj") is not None and st["padj"] >= padj_max: f.append("pvalue")
    return (not f), f

def _col(k): return adata.obs[k].astype(str) if k in adata.obs.columns else None
def _mode(s, m):
    vc = s[m].value_counts()
    return str(vc.index[0]) if len(vc) else ""

fine_c, facs_c, final_c = _col(fine_key), _col(facs_key), _col(final_key)
malig_c, cnvst_c = _col(malignancy_key), _col(cnv_status_key)
has_cnv = cnv_score_key in adata.obs.columns
cnv_score = adata.obs[cnv_score_key].astype(float) if has_cnv else None

per_cluster, status_counts, label_to_clusters = [], {{"clean": 0, "flagged": 0}}, {{}}
for cl in [str(c) for c in clusters]:
    mask = (obs_g == cl).values
    n_cells = int(mask.sum())
    if not n_cells: continue
    label = _mode(adata.obs[label_key].astype(str), mask)
    label_to_clusters.setdefault(label, []).append(cl)
    claimed = marker_sets.get(label, [])
    cl_stats = de_stats_by_cluster.get(cl, {{}})
    support_frac, marker_eval = None, []
    if claimed and cl_stats:
        n_pass = 0
        for g in claimed:
            st = cl_stats.get(g)
            if st is None:
                marker_eval.append({{"gene": g, "in_de": False, "passes": False, "failed_criteria": ["absent"]}}); continue
            ok, failed = _check_marker(st); n_pass += int(ok)
            marker_eval.append({{"gene": g, "in_de": True, "passes": ok, "failed_criteria": failed, **st}})
        support_frac = round(n_pass / len(claimed), 3)
    triple = {{"major": label, "fine": _mode(fine_c, mask) if fine_c is not None else None,
              "facs": _mode(facs_c, mask) if facs_c is not None else None,
              "final": _mode(final_c, mask) if final_c is not None else None}}
    prov = {{}}
    if sample_key in adata.obs.columns:
        sv = adata.obs[sample_key].astype(str)[mask].value_counts(normalize=True)
        prov["top_sample_frac"] = round(float(sv.iloc[0]), 3); prov["n_samples"] = int(adata.obs[sample_key].astype(str)[mask].nunique())
    if batch_key in adata.obs.columns:
        bv = adata.obs[batch_key].astype(str)[mask].value_counts(normalize=True)
        prov["top_batch_frac"] = round(float(bv.iloc[0]), 3)
        _p = bv.values; prov["batch_entropy"] = round(float(-(_p * np.log2(_p + 1e-12)).sum()), 3)
    qc = {{}}
    if doublet_key in adata.obs.columns:
        qc["doublet_frac"] = round(float(np.asarray(adata.obs[doublet_key].values[mask], dtype=float).mean()), 3)
    if mt_key in adata.obs.columns:
        qc["median_pct_mt"] = round(float(np.median(adata.obs[mt_key].values[mask])), 2)
    if stress_key in adata.obs.columns:
        qc["median_stress"] = round(float(np.median(adata.obs[stress_key].astype(float).values[mask])), 3)
    is_malignant = (_mode(malig_c, mask) == "malignant") if malig_c is not None else ((_mode(cnvst_c, mask) == "tumor") if cnvst_c is not None else False)
    cnv_burden = round(float(cnv_score[mask].mean()), 4) if has_cnv else None
    flags = []
    if support_frac is not None and support_frac < min_marker_support: flags.append("weak_marker_support")
    if prov.get("top_sample_frac", 0.0) >= single_source_frac: flags.append("single_patient_dominant")
    if prov.get("top_batch_frac", 0.0) >= single_source_frac: flags.append("batch_dominant")
    if qc.get("doublet_frac", 0.0) >= doublet_frac: flags.append("doublet_dominated")
    if qc.get("median_pct_mt", 0.0) > max_pct_mt: flags.append("high_mt")
    if is_malignant and not has_cnv: flags.append("malignant_without_cnv")
    status = "flagged" if flags else "clean"; status_counts[status] += 1
    per_cluster.append({{"cluster_id": cl, "n_cells": n_cells, "label": label, "hierarchy": triple,
        "marker_set_claimed": claimed, "marker_set_support_frac": support_frac, "marker_criteria_check": marker_eval,
        "top_specific_markers": de_by_cluster.get(cl, [])[:top_k_markers], "provenance": prov, "qc": qc,
        "is_malignant": bool(is_malignant), "cnv_burden": cnv_burden, "flags": flags, "review_status": status}})

# check 1: marker-profile collisions (high top-marker Jaccard, different label)
collisions = []
cl_sets = {{c["cluster_id"]: set(c["top_specific_markers"]) for c in per_cluster if c["top_specific_markers"]}}
ids = list(cl_sets)
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        a, b = ids[i], ids[j]
        la = next(c["label"] for c in per_cluster if c["cluster_id"] == a)
        lb = next(c["label"] for c in per_cluster if c["cluster_id"] == b)
        if la == lb: continue
        inter = len(cl_sets[a] & cl_sets[b]); union = len(cl_sets[a] | cl_sets[b]) or 1
        if inter / union >= profile_similarity:
            collisions.append({{"clusters": [a, b], "labels": [la, lb], "marker_jaccard": round(inter / union, 3),
                               "shared_markers": sorted(cl_sets[a] & cl_sets[b])[:10]}})

flagged = [c["cluster_id"] for c in per_cluster if c["review_status"] == "flagged"]
_audit = {{"groupby": groupby, "label_key": label_key, "clusters": per_cluster,
          "marker_profile_collisions": collisions, "status_counts": status_counts, "flagged_clusters": flagged,
          "n_marker_profile_collisions": len(collisions)}}
(_DATA / (OUT.stem + "_audit.json")).write_text(json.dumps(_audit, indent=2, default=str))
adata.write_h5ad(OUT)                                   # non-mutating: pass-through for the chain
print("[{cid}] annotation_audit: %d clusters, flagged %s, %d collisions -> %s"
      % (len(per_cluster), flagged, len(collisions), OUT.name))
'''


def _apply_annotation_audit(cid: str, params: dict) -> str:
    return f'''adata = sc.read_h5ad(IN)

# the INDEPENDENT reviewer's Tier-4 verdicts (edit freely)
groupby = {_g(params, "groupby", "leiden")!r}
label_key = {_g(params, "label_key", "major_cell_type")!r}
status_key = {_g(params, "status_key", "annotation_audit_status")!r}
review_required_key = {_g(params, "review_required_key", "annotation_review_required")!r}
reviewer_model = {_g(params, "reviewer_model", None)!r}
verdicts = {_g(params, "verdicts", {}) !r}

obs_g = adata.obs[groupby].astype(str)
vd = {{str(k): (v if isinstance(v, dict) else {{"status": str(v)}}) for k, v in verdicts.items()}}
status_map = {{c: v.get("status", "confirmed") for c, v in vd.items()}}
review_map = {{c: bool(v.get("review_required", v.get("status") != "confirmed")) for c, v in vd.items()}}
adata.obs[status_key] = obs_g.map(lambda c: status_map.get(c, "confirmed")).astype("category")
adata.obs[review_required_key] = obs_g.map(lambda c: review_map.get(c, False)).astype(bool)
refuted_clusters = sorted(c for c, s in status_map.items() if s == "refuted")
suspect_clusters = sorted(c for c, s in status_map.items() if s == "suspect")
adata.uns.setdefault("scpilot_annotation", {{}})
adata.uns["scpilot_annotation"]["tier4_audit"] = {{
    "groupby": groupby, "label_key": label_key, "reviewer_model": reviewer_model, "verdicts": vd,
    "n_refuted": len(refuted_clusters), "n_suspect": len(suspect_clusters),
    "refuted_clusters": refuted_clusters,
    "refuted_reasons": {{c: str(vd[c].get("note", "")) for c in refuted_clusters}},
    "suspect_clusters": suspect_clusters,
    "suspect_reasons": {{c: str(vd[c].get("note", "")) for c in suspect_clusters}},
}}
adata.write_h5ad(OUT)
print("[{cid}] apply_annotation_audit: %d refuted, %d suspect -> %s"
      % (len(refuted_clusters), len(suspect_clusters), OUT.name))
'''


def _annotation_review(cid: str, params: dict) -> str:
    return f'''import json
adata = sc.read_h5ad(IN)

# parameters (edit freely) — marker-quality + significance thresholds
groupby = {_g(params, "groupby", "leiden")!r}
top_n = {_g(params, "top_n", 50)!r}
padj_max = {_g(params, "padj_max", 0.05)!r}
min_in_group_fraction = {_g(params, "min_in_group_fraction", 0.25)!r}
max_out_group_fraction = {_g(params, "max_out_group_fraction", 0.10)!r}
min_fold_change = {_g(params, "min_fold_change", 1.5)!r}
min_specific_markers = {_g(params, "min_specific_markers", 3)!r}
sample_key = {_g(params, "sample_key", "sample_id")!r}
tissue = {_g(params, "tissue", None)!r}
max_samples_reported = {_g(params, "max_samples_reported", 8)!r}
single_source_frac = {_g(params, "single_source_frac", 0.8)!r}
mt_key, max_pct_mt = {_g(params, "mt_key", "pct_counts_mt")!r}, {_g(params, "max_pct_mt", 25.0)!r}
genes_key, min_genes = {_g(params, "genes_key", "n_genes_by_counts")!r}, {_g(params, "min_genes", 300.0)!r}
doublet_key, doublet_frac = {_g(params, "doublet_key", "predicted_doublet")!r}, {_g(params, "doublet_frac", 0.5)!r}
min_cells = {_g(params, "min_cells", 20)!r}

# reuse the per-cluster DE that `markers` computed (with pts); keep only SIGNIFICANT up-markers
de = sc.get.rank_genes_groups_df(adata, group=None)
de = de[de["pvals_adj"] < padj_max]
# marker-quality filter — scanpy's canonical filter_rank_genes_groups (pct_in/pct_out/fold-change)
sc.tl.filter_rank_genes_groups(adata, key="rank_genes_groups", key_added="_rgg_filt",
                               min_in_group_fraction=min_in_group_fraction,
                               max_out_group_fraction=max_out_group_fraction,
                               min_fold_change=min_fold_change)
_q = sc.get.rank_genes_groups_df(adata, group=None, key="_rgg_filt")
_qpairs = set(zip(_q["group"].astype(str), _q["names"].astype(str)))
adata.uns.pop("_rgg_filt", None)
de = de.assign(spec=(de["pct_nz_group"] - de["pct_nz_reference"]).round(4))
de_q = de[[(g, n) in _qpairs for g, n in zip(de["group"].astype(str), de["names"].astype(str))]]

obs_g = adata.obs[groupby].astype(str)
_cols = {{"names": "gene", "logfoldchanges": "logFC", "pvals_adj": "padj",
          "pct_nz_group": "pct_in", "pct_nz_reference": "pct_out", "spec": "spec", "scores": "score"}}
payloads = []
status_counts = {{"clean": 0, "review": 0, "artifact_suspected": 0}}
for cl, sub in de.groupby("group", observed=True):
    cl = str(cl)
    n_sig = int(len(sub))
    spec_sub = de_q[de_q["group"].astype(str) == cl].sort_values("spec", ascending=False)
    n_specific = int(len(spec_sub))
    top = spec_sub.head(top_n)
    mask = (obs_g == cl).values
    n_cells = int(mask.sum())
    _genes = [str(g) for g in top["names"] if str(g) in adata.var_names]
    _mean = {{}}
    if _genes and n_cells:
        _vals = np.asarray(adata[:, _genes].X[mask].mean(axis=0)).ravel()
        _mean = {{_genes[k]: round(float(_vals[k]), 4) for k in range(len(_genes))}}
    de_table = [{{**{{_cols[c]: (round(float(r[c]), 4) if c != "names" else str(r[c]))
                     for c in _cols if c in de.columns}}, "mean_in": _mean.get(str(r["names"]))}}
                for _, r in top.iterrows()]
    sample_dist, single_source = {{}}, False
    if sample_key in adata.obs.columns:
        sv = adata.obs[sample_key].astype(str)[mask].value_counts(normalize=True)
        sample_dist = {{str(k): round(float(v), 3) for k, v in sv.head(max_samples_reported).items()}}
        single_source = bool(sv.iloc[0] >= single_source_frac)
    qc = {{}}
    if mt_key in adata.obs.columns:
        qc["median_pct_mt"] = round(float(np.median(adata.obs[mt_key].values[mask])), 2)
    if genes_key in adata.obs.columns:
        qc["median_n_genes"] = round(float(np.median(adata.obs[genes_key].values[mask])), 1)
    if doublet_key in adata.obs.columns:
        qc["doublet_frac"] = round(float(np.asarray(adata.obs[doublet_key].values[mask], dtype=float).mean()), 3)
    doublet_dominated = doublet_key in adata.obs.columns and qc.get("doublet_frac", 0.0) >= doublet_frac
    low_quality = (mt_key in adata.obs.columns and qc.get("median_pct_mt", 0) > max_pct_mt) or \\
                  (genes_key in adata.obs.columns and qc.get("median_n_genes", 1e9) < min_genes)
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
    payloads.append({{"cluster_id": cl, "cluster_size": n_cells, "n_significant_markers": n_sig,
                      "n_specific_markers": n_specific, "review_status": status, "risk_signals": risk,
                      "sample_distribution": sample_dist, "qc_metrics": qc, "de_table": de_table}})

flagged = [p["cluster_id"] for p in payloads if p["review_status"] != "clean"]
_review = {{"groupby": groupby, "top_n": top_n, "tissue_context": tissue,
           "significance_filter": "pvals_adj < %s" % padj_max,
           "status_counts": status_counts, "flagged_clusters": flagged, "clusters": payloads}}
(_DATA / (OUT.stem + "_review.json")).write_text(json.dumps(_review, indent=2, default=str))
adata.write_h5ad(OUT)                                   # non-mutating: pass-through for the chain
print("[{cid}] annotation_review: %d clusters, flagged %s -> %s"
      % (len(payloads), flagged, OUT.name))
'''


_INGEST_BODY = r'''import yaml
import anndata as ad
from scipy import io, sparse

# ---- read the dataset profile (the entry input) -----------------------------------------------
prof = yaml.safe_load(open(IN)) or {}
input_root = Path(prof["input_root"])
if not input_root.is_absolute():
    input_root = Path(prof.get("out_dir", ".")) / input_root
metadata_csv = prof["metadata_csv"]
sample_id_col = prof.get("sample_id_col", "sample_id")
matrix_dir_col = prof.get("matrix_dir_col", "matrix_dir")
batch_col = prof.get("batch_col", "batch")
min_genes = prof.get("min_genes", 200)
max_pct_mt = prof.get("max_pct_mt", 20.0)
min_cells = prof.get("min_cells", 3)
target_sum = prof.get("target_sum", 1e4)
mito_prefix = prof.get("mito_prefix", "MT-")
normalized_layer = prof.get("normalized_layer", "scale.data")
harmonize = prof.get("harmonize", {}) or {}
harmonize_overrides = prof.get("harmonize_overrides", {}) or {}
filters = prof.get("filters", {}) or {}
derive = prof.get("derive", []) or []

# ---- metadata: load -> harmonize -> filter -> derive (profile-driven) -------------------------
import re as _re
def _norm(v):
    s = "" if v is None else str(v)
    s = _re.sub(r"[^0-9a-z]+", " ", s.strip().lower())
    return _re.sub(r"\s+", " ", s).strip()
def _cond_mask(df, c):
    col, op, val = c["col"], c.get("op", "eq"), c.get("value")
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col].astype(str)
    if op == "eq": return s == str(val)
    if op == "ne": return s != str(val)
    if op == "eq_ci": return s.map(_norm) == _norm(val)
    if op == "ne_ci": return s.map(_norm) != _norm(val)
    if op == "in": return s.isin([str(x) for x in val])
    if op == "not_in": return ~s.isin([str(x) for x in val])
    if op == "in_ci": return s.map(_norm).isin({_norm(x) for x in val})
    if op == "notna": return s.str.len() > 0
    raise ValueError("unknown op: %s" % op)
def _all(df, conds):
    m = pd.Series(True, index=df.index)
    for c in conds: m &= _cond_mask(df, c)
    return m

df = pd.read_csv(metadata_csv, dtype=str).fillna("")
# harmonize raw values -> canonical (preserves <field>__raw)
for field, mapping in harmonize.items():
    if field not in df.columns: continue
    syn = {}
    for canon, raws in mapping.items():
        for raw in raws: syn[_norm(raw)] = canon
        syn[_norm(canon)] = canon
    df["%s__raw" % field] = df[field]
    out = []
    for idx, raw in df[field].items():
        batch = str(df.at[idx, batch_col]) if batch_col in df.columns else ""
        over = harmonize_overrides.get(batch, {}).get(field, {})
        osyn = {_norm(r): canon for canon, raws in over.items() for r in raws}
        out.append(osyn.get(_norm(raw)) or syn.get(_norm(raw)) or raw)
    df[field] = out
# filters: include (keep matching ALL) then exclude (drop matching ANY)
if filters.get("include"):
    df = df[_all(df, filters["include"])].copy()
if filters.get("exclude"):
    drop = pd.Series(False, index=df.index)
    for c in filters["exclude"]: drop |= _cond_mask(df, c)
    df = df[~drop].copy()
# derive: ordered label rules
for rule in derive:
    t = rule.get("type")
    if t == "relabel":
        m = _all(df, rule.get("where", []))
        for col, val in rule["set"].items(): df.loc[m, col] = val
    elif t == "case":
        df[rule["target"]] = rule.get("default", "")
        for case in rule.get("cases", []):
            df.loc[_all(df, case.get("when", [])), rule["target"]] = case["value"]
    elif t == "alias":
        src = batch_col if rule["source"] == "__batch__" else rule["source"]
        df[rule["target"]] = df[src]
    elif t == "const":
        df[rule["target"]] = rule["value"]
    elif t == "isin_flag":
        present = df[rule["source"]].astype(str).isin([str(x) for x in rule["values"]])
        df[rule["target"]] = present.map({True: rule.get("true_value", "True"),
                                          False: rule.get("false_value", "False")})

# ---- per-sample 10x read + cell QC ------------------------------------------------------------
def _first(paths):
    return next((Path(p) for p in paths if Path(p).exists()), None)
def _has_10x(p):
    p = Path(p)
    return (p / "matrix.mtx.gz").exists() or (p / "matrix.mtx").exists() or (p / "filtered_feature_bc_matrix.h5").exists()
def _resolve_matrix_dir(row):
    local = Path(str(row.get(matrix_dir_col, ""))); sid = str(row.get(sample_id_col, ""))
    batch = str(row.get(batch_col, "")) if batch_col else ""
    cands = []
    def add(p):
        p = Path(p)
        if p not in cands: cands.append(p)
    add(input_root / local)
    if local.name == "filtered_feature_bc_matrix": add(input_root / local.parent)
    parts = local.parts
    if len(parts) >= 2:
        rp = input_root / parts[0] / "raw" / Path(*parts[1:]); add(rp)
        if rp.name == "filtered_feature_bc_matrix": add(rp.parent)
    if batch and (input_root / batch / "raw").exists():
        for m in sorted((input_root / batch / "raw").glob(sid + "*")):
            add(m); add(m / "filtered_feature_bc_matrix")
    for c in cands:
        if _has_10x(c): return c
    raise FileNotFoundError("no 10x files for %s; checked %s" % (sid, [str(c) for c in cands[:10]]))
def _read_10x_robust(d):
    d = Path(d)
    mp = _first([d / "matrix.mtx.gz", d / "matrix.mtx"])
    bp = _first([d / "barcodes.tsv.gz", d / "barcodes.tsv"])
    fp = _first([d / "features.tsv.gz", d / "features.tsv", d / "genes.tsv.gz", d / "genes.tsv"])
    X = io.mmread(str(mp)).T.tocsr()
    bc = pd.read_csv(bp, sep="\t", header=None, compression="infer", dtype=str)[0].astype(str).values
    ft = pd.read_csv(fp, sep="\t", header=None, compression="infer", dtype=str)
    syms = (ft.iloc[:, 1] if ft.shape[1] >= 2 else ft.iloc[:, 0]).astype(str).values
    a = ad.AnnData(X=X); a.obs_names = pd.Index(bc, name="barcode")
    a.var_names = pd.Index(syms, name="gene_symbols"); a.var_names_make_unique()
    return a

adatas, keys, per_sample, n_fail = [], [], [], 0
for _, row in df.iterrows():
    row = row.to_dict(); sid = str(row[sample_id_col])
    try:
        mdir = _resolve_matrix_dir(row)
        h5 = mdir / "filtered_feature_bc_matrix.h5"
        has_mtx = (mdir / "matrix.mtx.gz").exists() or (mdir / "matrix.mtx").exists()
        if h5.exists() and not has_mtx:
            a = sc.read_10x_h5(str(h5), gex_only=True)
        else:
            try:
                a = sc.read_10x_mtx(str(mdir), var_names="gene_symbols", make_unique=True)
            except Exception:
                a = _read_10x_robust(mdir)
        a.var_names_make_unique(); a.X = a.X.astype(np.float32)
        a.obs_names = ["%s_%s" % (sid, bc) for bc in a.obs_names.astype(str)]
        a.obs["sample_id"] = sid
        for k, v in row.items():
            a.obs[k] = "" if pd.isna(v) else str(v)
        a.var["mt"] = a.var_names.str.upper().str.startswith(mito_prefix.upper())
        sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
        a.layers["counts"] = a.X.copy()
    except Exception as exc:
        per_sample.append({"sample": sid, "status": "read_fail:%s" % type(exc).__name__}); n_fail += 1
        continue
    raw_n = a.n_obs
    keep = (a.obs["n_genes_by_counts"] >= min_genes) & (a.obs["pct_counts_mt"] <= max_pct_mt)
    a = a[keep].copy()
    per_sample.append({"sample": sid, "n_raw": int(raw_n), "n_kept": int(a.n_obs)})
    if a.n_obs > 0:
        adatas.append(a); keys.append(sid)

# ---- merge (deterministic order) + gene filter + normalize ------------------------------------
merged = ad.concat(adatas, join="outer", label="sample_id_from_concat", keys=keys,
                   index_unique=None, merge="same", fill_value=0)
merged.var_names_make_unique()
if sparse.issparse(merged.X):
    merged.X = merged.X.tocsr()
merged.layers["counts"] = merged.X.copy()
sc.pp.filter_genes(merged, min_cells=min_cells)
merged.layers["counts"] = merged.X.copy()
merged.X = merged.layers["counts"].copy()
sc.pp.normalize_total(merged, target_sum=target_sum)
sc.pp.log1p(merged)
# X holds the log-normalized values (no duplicate normalized layer; markers/annotation read X)

merged.write_h5ad(OUT)
print("[__CID__] ingest: %d cells x %d genes from %d samples (%d failed) -> %s"
      % (merged.n_obs, merged.n_vars, len(keys), n_fail, OUT.name))
'''


def _ingest(cid: str, params: dict) -> str:
    # ingest reads the profile YAML at runtime; nothing to bake (params come from the profile)
    return _INGEST_BODY.replace("__CID__", cid)


def _detect_state(cid: str, params: dict) -> str:
    # non-mutating: classify how far the h5ad has been processed; pass adata through
    return f'''adata = sc.read_h5ad(IN)

_layers = set(adata.layers.keys()); _obsm = set(adata.obsm.keys())
_obsp = set(adata.obsp.keys()); _obs = set(adata.obs.columns); _uns = set(adata.uns.keys())
# best-effort guess of what .X holds
try:
    _a = adata.X[:50]; _a = _a.toarray() if hasattr(_a, "toarray") else np.asarray(_a)
    x_state = "raw_counts" if (_a.size and np.allclose(_a, np.round(_a)) and _a.min() >= 0) else "normalized"
except Exception:
    x_state = "raw_counts" if "counts" in _layers else "unknown"
flags = {{
    "has_counts": "counts" in _layers,
    "normalized": x_state in ("normalized", "log1p") or "scale.data" in _layers,
    "hvg": "highly_variable" in adata.var.columns,
    "pca": "X_pca" in _obsm,
    "neighbors": ("neighbors" in _uns) or ("distances" in _obsp) or ("connectivities" in _obsp),
    "clustered": any(c in _obs for c in ("leiden", "louvain")),
    "umap": "X_umap" in _obsm,
    "annotated": any(c in _obs for c in ("major_cell_type", "fine_cell_type", "facs_style_label", "malignancy")),
}}
_order = ["raw", "normalized", "hvg", "pca", "neighbors", "clustered", "umap", "annotated"]
_sat = [s for s in _order if (s == "raw" and flags["has_counts"]) or flags.get(s)]
stage = _sat[-1] if _sat else "raw"
(_DATA / (OUT.stem + "_state.txt")).write_text(stage)
adata.write_h5ad(OUT)
print("[{cid}] detect_state: stage=%s x_state=%s -> %s" % (stage, x_state, OUT.name))
'''


def _compartment_plan(cid: str, params: dict) -> str:
    # non-mutating Tier-1->Tier-2 bridge evidence; pass adata through
    return f'''import json
adata = sc.read_h5ad(IN)

# parameters (edit freely)
groupby = {_g(params, "groupby", None)!r}
batch_key = {_g(params, "batch_key", None)!r}
sample_key = {_g(params, "sample_key", "sample_id")!r}
min_cells = {_g(params, "min_cells", 50)!r}
min_samples = {_g(params, "min_samples", 2)!r}
single_source_frac = {_g(params, "single_source_frac", 0.8)!r}

def _resolve(key, cands):
    if key is not None:
        return key if key in adata.obs.columns else None
    for c in cands:
        if c in adata.obs.columns:
            return c
    return None
def _norm_entropy(counts, n_global):
    p = np.asarray(counts, dtype=float); p = p[p > 0]; total = p.sum()
    if total <= 0 or n_global <= 1: return 0.0
    p = p / total
    return round(float(-(p * np.log(p)).sum()) / np.log(n_global), 4)

gkey = _resolve(groupby, ("major_cell_type", "celltype_consensus", "leiden"))
bkey = _resolve(batch_key, ("GSE", "batch", "sample_id"))
has_sample = sample_key in adata.obs.columns
obs_g = adata.obs[gkey].astype(str)
n_batches_global = int(adata.obs[bkey].astype(str).nunique()) if bkey else 0
n_samples_global = int(adata.obs[sample_key].astype(str).nunique()) if has_sample else 0

rows, branchable, blocked = [], [], []
for comp in sorted(obs_g.unique()):
    comp = str(comp); mask = (obs_g == comp).values; n_cells = int(mask.sum())
    reasons = []; n_samples = 0; top_sample_frac = None; ent = None
    if has_sample:
        sv = adata.obs[sample_key].astype(str)[mask].value_counts(normalize=True)
        n_samples = int(sv.size); top_sample_frac = round(float(sv.iloc[0]), 3)
    if bkey:
        bv = adata.obs[bkey].astype(str)[mask].value_counts()
        ent = _norm_entropy(bv.values, n_batches_global)
        if int(bv.size) == 1 and n_batches_global > 1: reasons.append("single_batch")
        elif ent < 0.3 and int(bv.size) > 1: reasons.append("low_batch_mixing")
    powered = n_cells >= min_cells and (n_samples >= min_samples if has_sample else True)
    if n_cells < min_cells: reasons.append("below_min_cells(<%d)" % min_cells)
    if has_sample and n_samples < min_samples: reasons.append("below_min_samples(<%d)" % min_samples)
    (branchable if powered else blocked).append(comp)
    rows.append({{"compartment": comp, "n_cells": n_cells, "n_samples": n_samples,
                 "top_sample_frac": top_sample_frac, "batch_entropy_norm": ent,
                 "branch_recommended": bool(powered), "block_reasons": reasons}})

(_DATA / (OUT.stem + "_plan.json")).write_text(json.dumps(
    {{"groupby": gkey, "batch_key": bkey, "branchable": branchable, "blocked": blocked, "compartments": rows}},
    indent=2, default=str))
adata.write_h5ad(OUT)                                   # non-mutating: pass-through for the chain
print("[{cid}] compartment_plan: branchable=%s blocked=%s -> %s" % (branchable, blocked, OUT.name))
'''


def _benchmark(cid: str, params: dict) -> str:
    # non-mutating scib comparison of integration embeddings; writes scib results CSV
    return f'''from scib_metrics.benchmark import Benchmarker
adata = sc.read_h5ad(IN)

# parameters (edit freely)
label_key = {_g(params, "label_key", "major_cell_type")!r}
batch_key = {_g(params, "batch_key", "sample_id")!r}
embeddings = {_g(params, "embeddings", None)!r} or ["X_pca", "X_harmony", "X_scVI"]
drop_labels = {_g(params, "drop_labels", None)!r}
if drop_labels is None:
    drop_labels = ["Unknown", "Mixed/Artifact", "Low_quality", "ambiguous"]   # non-biological labels
min_label_cells = {_g(params, "min_label_cells", 10)!r}
subsample = {_g(params, "subsample", 60000)!r}
seed = {_g(params, "seed", 0)!r}

embs = [e for e in embeddings if e in adata.obsm]
lab = adata.obs[label_key].astype(str)
keep = ~lab.isin([str(x) for x in drop_labels])
vc = lab[keep].value_counts()
small = sorted(vc[vc < min_label_cells].index)
if small:
    keep = keep & ~lab.isin(small)
idx = np.where(keep.values)[0]
# stratified subsample (by label) for tractability
rng = np.random.default_rng(seed); n_kept = int(idx.size)
if subsample and n_kept > subsample:
    labs = lab.values[idx]; sel = []
    for L in np.unique(labs):
        li = idx[labs == L]; take = min(li.size, max(1, int(round(subsample * li.size / n_kept))))
        sel.append(rng.choice(li, size=take, replace=False))
    idx = np.sort(np.concatenate(sel))
bdata = adata[idx].copy()
bdata.obs[label_key] = bdata.obs[label_key].astype(str).astype("category")
bdata.obs[batch_key] = bdata.obs[batch_key].astype(str).astype("category")
pre = "X_pca" if "X_pca" in embs else None
bm = Benchmarker(bdata, batch_key=batch_key, label_key=label_key, embedding_obsm_keys=embs,
                 pre_integrated_embedding_obsm_key=pre, n_jobs=-1, progress_bar=False)
bm.benchmark()
res = bm.get_results(min_max_scale=False, clean_names=False)
res.to_csv(_DATA / (OUT.stem + "_scib.csv"))
res_num = res.drop(index="Metric Type", errors="ignore")
ranked = sorted((e for e in res_num.index if res_num.loc[e].get("Total") == res_num.loc[e].get("Total")),
                key=lambda e: float(res_num.loc[e]["Total"]), reverse=True)
adata.write_h5ad(OUT)                                   # non-mutating: pass-through for the chain
print("[{cid}] benchmark: embeddings=%s best=%s -> %s" % (embs, ranked[0] if ranked else None, OUT.name))
'''


def _load(cid: str, params: dict) -> str:
    # entry step: read the input h5ad and pass it on as the chain's first checkpoint
    return (
        "adata = sc.read_h5ad(IN)\n"
        "adata.write_h5ad(OUT)\n"
        f'print("[{cid}] load: %d cells x %d genes -> %s" % (adata.n_obs, adata.n_vars, OUT.name))\n'
    )


def _integrate_harmony(cid: str, params: dict) -> str:
    return f'''import harmonypy
adata = sc.read_h5ad(IN)

# parameters (edit freely)
batch_key = {_g(params, "batch_key", "GSM")!r}
use_rep = {_g(params, "use_rep", "X_pca")!r}
out_key = {_g(params, "out_key", "X_harmony")!r}
seed = {_g(params, "seed", 0)!r}

# Harmony batch integration via harmonypy
ho = harmonypy.run_harmony(adata.obsm[use_rep], adata.obs, [batch_key], random_state=seed)
Z = np.asarray(ho.Z_corr)                          # harmonypy torch -> numpy
Z = Z.T if Z.shape[0] == adata.obsm[use_rep].shape[1] else Z   # (cells x dims)
adata.obsm[out_key] = Z
adata.write_h5ad(OUT)
print("[{cid}] integrate_harmony: %d dims (key=%s) -> %s" % (Z.shape[1], out_key, OUT.name))
'''


def _consensus_annotation(cid: str, params: dict) -> str:
    return _vote_body(cid, "consensus_annotation", params, "celltype_consensus")


def _harmonize_annotations(cid: str, params: dict) -> str:
    # standalone uses the always-available embedding-independent majority vote (the optional
    # cellhint path is scpilot-only and not wired); 'method' is dropped.
    p = {k: v for k, v in (params or {}).items() if k != "method"}
    return _vote_body(cid, "harmonize_annotations", p, "celltype_harmonized")


EMITTERS: dict = {
    "ingest": _ingest,
    "load": _load,
    "detect_state": _detect_state,
    "compartment_plan": _compartment_plan,
    "benchmark": _benchmark,
    "qc_metrics": _qc_metrics,
    "qc_filter": _qc_filter,
    "preprocess": _preprocess,
    "cluster": _cluster,
    "cluster_sweep": _cluster_sweep,
    "markers": _markers,
    "integrate_harmony": _integrate_harmony,
    "annotation_review": _annotation_review,
    "annotation_audit": _annotation_audit,
    "apply_annotation_audit": _apply_annotation_audit,
    "apply_annotation": _apply_annotation,
    "consensus_annotation": _consensus_annotation,
    "harmonize_annotations": _harmonize_annotations,
}


def has_emitter(stage: str) -> bool:
    return stage in EMITTERS


_HEADER = (
    "#!/usr/bin/env python\n"
    '"""{cid}_{stage}.py — {stage} (STANDALONE tutorial step {idx}; needs only scanpy/anndata/numpy/pandas).\n'
    "\n"
    "{why}\n"
    "\n"
    "Run the NN_*.py files IN ORDER — each reads the previous step's .h5ad and writes its own.\n"
    "parameters:\n"
    "{md}\n"
    'outputs: standalone_data/{cid}_{stage}.h5ad\n'
    '"""\n'
    "import scanpy as sc\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "from pathlib import Path\n"
    "\n"
    "_HERE = Path(__file__).resolve().parent\n"
    '_DATA = _HERE.parent / "standalone_data"\n'
    "_DATA.mkdir(exist_ok=True)\n"
    "IN  = {in_expr}\n"
    'OUT = _DATA / "{cid}_{stage}.h5ad"\n'
    "\n"
)


def build(idx: int, cid: str, stage: str, params: dict, in_expr: str,
          reasoning: str | None = None) -> str | None:
    """Full standalone script text for one step, or None if the stage has no emitter yet."""
    emit = EMITTERS.get(stage)
    if emit is None:
        return None
    if params:
        md = "\n".join(f"  - {k} = {v!r}" for k, v in params.items())
    else:
        md = "  - (no parameters)"
    why = ("why: " + " ".join(str(reasoning).split())[:400]) if reasoning else \
        f"deterministic {stage} step, written as plain scanpy."
    header = _HEADER.format(idx=idx, cid=cid, stage=stage, why=why, md=md, in_expr=in_expr)
    return header + emit(cid, params)
