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
adata.layers["scale.data"] = adata.X.copy()        # keep normalized values for markers/annotation

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
layer = {_g(params, "layer", "scale.data")!r}
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
    "load": _load,
    "qc_metrics": _qc_metrics,
    "qc_filter": _qc_filter,
    "preprocess": _preprocess,
    "cluster": _cluster,
    "cluster_sweep": _cluster_sweep,
    "markers": _markers,
    "integrate_harmony": _integrate_harmony,
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
