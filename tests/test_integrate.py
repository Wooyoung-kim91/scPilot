"""Unit tests for B9 integration (Harmony fixture + scVI gates).

Real scVI-model load is covered by a separate real-data smoke (needs the
pretrained PDAC model); here we test the Harmony path + scVI precondition gates.
"""

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

from scpilot import tools
from scpilot.session import Session


def _pca_session(tmp_path, with_batch=True):
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (200, 80)).astype("float32")
    X[:100, :20] += rng.poisson(4.0, (100, 20)).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(80)]
    a.layers["counts"] = a.X.copy()
    if with_batch:
        a.obs["GSM"] = rng.choice(["s1", "s2", "s3"], 200)
    p = tmp_path / "in.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    tools.run("preprocess", s, n_top_genes=40, n_pcs=15)
    return s


def test_harmony_integrate(tmp_path):
    s = _pca_session(tmp_path)
    r = tools.run("integrate_harmony", s, batch_key="GSM")
    assert r.status == "success"
    assert "X_harmony" in s.adata.obsm
    # X_pca preserved alongside X_harmony (reductions-preservation convention)
    assert "X_pca" in s.adata.obsm
    assert r.summary["n_dims"] == s.adata.obsm["X_pca"].shape[1]
    assert r.summary["n_cells"] == s.adata.n_obs
    r.to_dict()


def test_harmony_requires_pca(tmp_path):
    a = ad.AnnData(sparse.csr_matrix(np.random.default_rng(0).poisson(1.0, (40, 20)).astype("float32")))
    a.layers["counts"] = a.X.copy(); a.obs["GSM"] = "s1"
    p = tmp_path / "raw.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "s", input_path=str(p)); s.load_input()
    r = tools.run("integrate_harmony", s, batch_key="GSM")
    assert r.status == "error" and r.error_code == "invalid_state"


def test_harmony_requires_batch_key(tmp_path):
    s = _pca_session(tmp_path, with_batch=False)
    r = tools.run("integrate_harmony", s, batch_key="GSM")
    assert r.status == "error" and r.error_code == "data_gate_failed"


def test_scvi_missing_model_dir(tmp_path):
    s = _pca_session(tmp_path)
    r = tools.run("integrate_scvi", s, model_dir=str(tmp_path / "no_model"), batch_key="GSM")
    assert r.status == "error" and r.error_code == "missing_input"


def test_registry_has_integrate():
    names = {t["name"] for t in tools.list_tools()}
    assert {"integrate_scvi", "integrate_harmony"} <= names
