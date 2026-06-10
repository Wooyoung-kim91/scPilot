"""Unit tests for B5 plots (vendored auto-fit harness + column-size policy)."""

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

from scpilot import tools
from scpilot.session import Session


def _processed_session(tmp_path):
    """A small clustered AnnData so umap/hvg/pca/qc plots all have inputs."""
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (200, 80)).astype("float32")
    X[:100, :20] += rng.poisson(4.0, (100, 20)).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(80)]
    a.obs["sample_id"] = rng.choice(["s1", "s2"], 200)
    a.layers["counts"] = a.X.copy()
    a.obs["n_genes_by_counts"] = (a.layers["counts"] > 0).sum(1).A1
    a.obs["total_counts"] = a.layers["counts"].sum(1).A1
    a.obs["pct_counts_mt"] = rng.uniform(0, 10, 200)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=40)
    sc.pp.pca(a, n_comps=15)
    sc.pp.neighbors(a, n_neighbors=10)
    sc.tl.leiden(a, flavor="igraph", n_iterations=2, random_state=0)
    sc.tl.umap(a, random_state=0)
    p = tmp_path / "in.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    return s


def _assert_size_policy(r):
    """Saved size must be within {min 0.5×0.5, max 1.5 each, not both >1.0} col."""
    w, h = r.summary["size_col"]
    assert 0.5 <= w <= 1.5 and 0.5 <= h <= 1.5
    assert not (w > 1.0 and h > 1.0), f"both dims >1 col: {(w, h)}"


def test_plot_umap(tmp_path):
    s = _processed_session(tmp_path)
    r = tools.run("plots", s, kind="umap", color="leiden")
    assert r.status == "success"
    assert r.artifacts and r.artifacts[0].kind == "png"
    from pathlib import Path
    png = Path(r.artifacts[0].path)
    assert png.exists() and png.stat().st_size > 0
    assert r.artifacts[0].meta["dpi"] == 300
    _assert_size_policy(r)
    r.to_dict()


def test_plot_qc_violin_and_hvg_and_pca(tmp_path):
    s = _processed_session(tmp_path)
    for kind in ("qc_violin", "hvg", "pca_variance"):
        r = tools.run("plots", s, kind=kind)
        assert r.status == "success", f"{kind}: {r.error}"
        assert r.artifacts
        _assert_size_policy(r)


def test_plot_umap_requires_umap(tmp_path):
    # raw session without umap
    a = ad.AnnData(sparse.csr_matrix(np.random.default_rng(0).poisson(1.0, (50, 20)).astype("float32")))
    a.layers["counts"] = a.X.copy()
    p = tmp_path / "raw.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "s", input_path=str(p)); s.load_input()
    r = tools.run("plots", s, kind="umap")
    assert r.status == "error" and r.error_code == "invalid_state"


def test_size_grid_orientation_filter():
    # the square_limit_col filter forbids both dims exceeding the limit
    from scpilot.vendor.plotting import _size_grid
    p = {"start_col": 0.5, "step_col": 0.25, "max_w_col": 1.5, "max_h_col": 1.5,
         "square_limit_col": 1.0}
    pairs = _size_grid(p)
    assert (1.5, 1.0) in pairs and (1.0, 1.5) in pairs and (1.0, 1.0) in pairs
    assert all(not (w > 1.0 and h > 1.0) for w, h in pairs)
    assert (1.5, 1.5) not in pairs and (1.25, 1.25) not in pairs


def test_registry_has_plots():
    assert "plots" in {t["name"] for t in tools.list_tools()}
