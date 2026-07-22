"""Unit tests for B4 preprocess, B6 cluster, B7 markers (chained)."""

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from scpilot import tools
from scpilot.session import Session


def _raw(n_obs=300, n_vars=200):
    """Two latent groups so clustering finds structure; counts in X + layer."""
    rng = np.random.default_rng(0)
    base = rng.poisson(0.5, (n_obs, n_vars)).astype("float32")
    half = n_obs // 2
    base[:half, :40] += rng.poisson(4.0, (half, 40)).astype("float32")    # group A program
    base[half:, 40:80] += rng.poisson(4.0, (n_obs - half, 40)).astype("float32")  # group B
    a = ad.AnnData(sparse.csr_matrix(base))
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    a.var_names = [f"G{i}" for i in range(n_vars)]
    a.layers["counts"] = a.X.copy()
    a.obs["sample_id"] = rng.choice(["s1", "s2", "s3"], n_obs)
    return a


def _session(tmp_path):
    a = _raw()
    p = tmp_path / "in.h5ad"
    a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p))
    s.load_input()
    return s


def test_preprocess(tmp_path):
    s = _session(tmp_path)
    r = tools.run("preprocess", s, n_top_genes=100, n_pcs=20, hvg_batch_key="sample_id")
    assert r.status == "success"
    sm = r.summary
    assert sm["n_hvg"] > 0 and sm["n_hvg"] <= 100
    assert sm["x_state"] == "log1p"
    assert "X_pca" in s.adata.obsm
    assert "scale.data" not in s.adata.layers  # I-14: X is the log-norm layer (no duplicate)
    assert len(sm["variance_ratio"]) == sm["n_pcs"]
    assert 1 <= sm["suggested_n_pcs_elbow"] <= sm["n_pcs"]
    assert r.determinism_grade == "B"
    # counts layer preserved (invariant)
    assert "counts" in s.adata.layers


def test_preprocess_hvg_batch_key_disable_token(tmp_path):
    # I-3: an explicit OFF token forces global HVG and must NOT fall through to sample_id auto-detect
    # (the old bug: "none" was ignored and auto-detect re-grabbed sample_id).
    s = _session(tmp_path)
    r = tools.run("preprocess", s, n_top_genes=100, n_pcs=20, hvg_batch_key="none")
    assert r.status == "success"
    assert any("disabled" in w.lower() for w in r.warnings)
    assert not any("auto-detected" in w.lower() for w in r.warnings)
    assert int(s.adata.var["highly_variable"].sum()) > 0


def test_preprocess_tiny_batch_guard_disables_batch_hvg(tmp_path):
    # I-3: batches below min_cells_per_batch make seurat_v3's per-batch loess singular; the guard
    # disables batch-aware HVG (with a warning) instead of crashing. The _raw fixture has ~100
    # cells/sample → all below the default 1000, so an explicit sample_id key trips the guard.
    s = _session(tmp_path)
    r = tools.run("preprocess", s, n_top_genes=100, n_pcs=20, hvg_batch_key="sample_id")
    assert r.status == "success"
    assert any("batch-aware HVG disabled" in w for w in r.warnings)
    assert int(s.adata.var["highly_variable"].sum()) > 0


def test_preprocess_requires_counts(tmp_path):
    a = _raw()
    del a.layers["counts"]
    p = tmp_path / "nc.h5ad"
    a.write_h5ad(p)
    s = Session.create(tmp_path / "s2", input_path=str(p)); s.load_input()
    r = tools.run("preprocess", s)
    assert r.status == "error" and r.error_code == "invalid_state"


def test_cluster_requires_pca(tmp_path):
    s = _session(tmp_path)
    r = tools.run("cluster", s)
    assert r.status == "error" and r.error_code == "invalid_state"


def test_cluster_and_markers_chain(tmp_path):
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=100, n_pcs=20)
    rc = tools.run("cluster", s, resolution=0.5)
    assert rc.status == "success"
    assert rc.summary["n_clusters"] >= 2          # two planted groups
    assert rc.summary["cluster_key"] == "leiden"  # baseline keeps canonical name
    assert "X_umap" in rc.summary["embeddings_present"]
    assert sum(rc.summary["cluster_sizes"].values()) == s.adata.n_obs

    rm = tools.run("markers", s, n_genes=15)
    assert rm.status == "success"
    assert rm.summary["n_clusters"] == rc.summary["n_clusters"]
    assert "top_markers" in rm.tables
    # ranking CSV artifact written, absolute path
    assert rm.artifacts and rm.artifacts[0].kind == "csv"
    from pathlib import Path
    assert Path(rm.artifacts[0].path).exists()
    rm.to_dict()


def test_markers_caps_ranking_wilcoxon(tmp_path):
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=100, n_pcs=20)
    rc = tools.run("cluster", s, resolution=0.5)
    assert rc.status == "success"

    rm = tools.run("markers", s, n_genes=15, max_genes_ranked=7)
    assert rm.status == "success"
    assert rm.summary["method"] == "wilcoxon"          # DE method is fixed to Wilcoxon
    assert rm.summary["n_genes_ranked"] == 7
    assert rm.summary["max_genes_ranked"] == 7
    assert rm.summary["csv_is_full_ranking"] is False
    assert "capped" in rm.artifacts[0].description
    assert rm.artifacts[0].meta["csv_is_full_ranking"] is False
    assert rm.artifacts[0].meta["n_genes_ranked"] == 7
    assert rm.artifacts[0].meta["n_rows"] == rc.summary["n_clusters"] * 7


def test_markers_full_ranking_when_cap_is_none(tmp_path):
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=100, n_pcs=20)
    tools.run("cluster", s, resolution=0.5)

    rm = tools.run("markers", s, max_genes_ranked=None)
    assert rm.status == "success"
    assert rm.summary["csv_is_full_ranking"] is True
    assert rm.summary["n_genes_ranked"] == s.adata.n_vars
    assert rm.artifacts[0].description == "full rank_genes_groups ranking"


def test_cluster_defaults_resolution(tmp_path):
    # resolution defaults to 0.25 at every stage when the caller omits it
    from scpilot.core.cluster import DEFAULT_RESOLUTION
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=100, n_pcs=20)
    r = tools.run("cluster", s, use_rep="X_pca")             # no resolution given
    assert r.status == "success"
    assert r.summary["resolution"] == DEFAULT_RESOLUTION == 0.25
    assert r.summary["resolution_defaulted"] is True
    # an explicit value still wins and is not flagged as defaulted
    r2 = tools.run("cluster", s, use_rep="X_pca", resolution=0.5)
    assert r2.summary["resolution"] == 0.5
    assert r2.summary["resolution_defaulted"] is False


def test_cluster_preserves_reductions_per_model(tmp_path):
    """All reductions kept per model, before+after integration (user requirement)."""
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=100, n_pcs=20)
    tools.run("cluster", s, use_rep="X_pca", resolution=0.5)   # baseline: X_umap / leiden
    # simulate an integration embedding, then cluster on it
    s.adata.obsm["X_scVI"] = s.adata.obsm["X_pca"][:, :10].copy()
    rc = tools.run("cluster", s, use_rep="X_scVI", resolution=0.5)
    a = s.adata
    # baseline reductions still present AND model-specific ones added (not overwritten)
    assert "X_umap" in a.obsm and "X_umap_scvi" in a.obsm
    assert "leiden" in a.obs and "leiden_scvi" in a.obs
    assert rc.summary["umap_key"] == "X_umap_scvi"
    assert rc.summary["cluster_key"] == "leiden_scvi"


def test_cluster_sweep_leaves_no_scratch_state(tmp_path):
    """Bug D: cluster_sweep is mutating=False, so it MUST leave the object's uns/obs/obsp
    byte-identical. sc.tl.leiden(key_added='_sweep_leiden') also writes uns['_sweep_leiden']
    (a params dict) which the finally block must pop, alongside the neighbors key + graphs +
    obs column. A leaked _sweep_* key would get persisted by a later mutating checkpoint,
    making the saved object order-dependent (non-content-addressed)."""
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=100, n_pcs=20)
    a = s.adata
    uns_before = set(a.uns.keys())
    obs_before = set(a.obs.columns)
    obsp_before = set(a.obsp.keys())

    r = tools.run("cluster_sweep", s, use_rep="X_pca")
    assert r.status == "success"

    # no temp scratch keys of ANY kind survive the sweep
    assert not [k for k in a.uns if str(k).startswith("_sweep_")]
    assert not [c for c in a.obs.columns if str(c).startswith("_sweep_")]
    assert not [k for k in a.obsp if str(k).startswith("_sweep_")]
    # object is unchanged: uns/obs/obsp identical to before the (non-mutating) sweep
    assert set(a.uns.keys()) == uns_before
    assert set(a.obs.columns) == obs_before
    assert set(a.obsp.keys()) == obsp_before


def test_registry_has_b4_b7():
    names = {t["name"] for t in tools.list_tools()}
    assert {"preprocess", "cluster", "markers"} <= names
