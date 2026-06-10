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
    assert "lognorm" in s.adata.layers
    assert len(sm["variance_ratio"]) == sm["n_pcs"]
    assert 1 <= sm["suggested_n_pcs_elbow"] <= sm["n_pcs"]
    assert r.determinism_grade == "B"
    # counts layer preserved (invariant)
    assert "counts" in s.adata.layers


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
    assert rc.summary["has_umap"] is True
    assert sum(rc.summary["cluster_sizes"].values()) == s.adata.n_obs

    rm = tools.run("markers", s, n_genes=15)
    assert rm.status == "success"
    assert rm.summary["n_clusters"] == rc.summary["n_clusters"]
    assert "top_markers" in rm.tables
    # full ranking CSV artifact written, absolute path
    assert rm.artifacts and rm.artifacts[0].kind == "csv"
    from pathlib import Path
    assert Path(rm.artifacts[0].path).exists()
    rm.to_dict()


def test_registry_has_b4_b7():
    names = {t["name"] for t in tools.list_tools()}
    assert {"preprocess", "cluster", "markers"} <= names
