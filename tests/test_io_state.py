"""Unit tests for B1 (io: load/inspect) and B2 (state detection)."""

import anndata as ad
import numpy as np
import scanpy as sc
from pathlib import Path
from scipy import sparse

from scpilot import tools
from scpilot.core.io import load_h5ad, save_h5ad
from scpilot.core.state import detect_state
from scpilot.session import Session


def _raw(n_obs=80, n_vars=60):
    X = sparse.csr_matrix(np.random.default_rng(0).poisson(1.0, (n_obs, n_vars)).astype("float32"))
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    a.var_names = [f"G{i}" for i in range(n_vars)]
    a.layers["counts"] = a.X.copy()
    a.obs["sample_id"] = (["s1"] * (n_obs // 2) + ["s2"] * (n_obs - n_obs // 2))
    return a


def test_io_helpers_roundtrip(tmp_path):
    a = _raw()
    p = save_h5ad(a, tmp_path / "x.h5ad")
    b = load_h5ad(p)
    assert b.n_obs == a.n_obs and "counts" in b.layers
    backed = load_h5ad(p, backed="r")
    assert backed.isbacked


def test_load_tool_registered_and_runs(tmp_path):
    a = _raw()
    save_h5ad(a, tmp_path / "x.h5ad")
    sess = Session.create(tmp_path / "sess", input_path=str(tmp_path / "x.h5ad"))
    r = tools.run("load", sess)
    assert r.status == "success"
    assert r.summary["n_obs"] == 80 and r.summary["has_counts"] is True
    assert r.summary["x_state_guess"] == "raw_counts"
    r.to_dict()  # JSON-serializable


def test_mcp_default_session_uses_requested_checkpoint_input(tmp_path):
    raw = _raw()
    raw_path = tmp_path / "obesity_merged_counts.h5ad"
    raw.write_h5ad(raw_path)

    clustered = raw.copy()
    clustered.obs["leiden"] = (["0"] * (clustered.n_obs // 2)
                               + ["1"] * (clustered.n_obs - clustered.n_obs // 2))
    clustered.layers["scale.data"] = clustered.X.copy()
    checkpoint = tmp_path / "scpilot_obesity_run" / "checkpoints" / "04_cluster.h5ad"
    checkpoint.parent.mkdir(parents=True)
    clustered.write_h5ad(checkpoint)

    # Simulate a stale unrelated default session: the MCP no-workdir path must not reopen it.
    stale = Session.create(tmp_path / "scpilot_run", input_path=str(raw_path))
    stale.load_input()

    from scpilot.mcp_server import default_workdir_for_input

    wd = default_workdir_for_input(str(checkpoint))
    assert Path(wd) != stale.out
    sess = Session.create(wd, input_path=str(checkpoint))
    r = tools.run("markers", sess, max_genes_ranked=5)
    assert r.status == "success"
    assert sess.manifest.input["path"] == str(checkpoint.resolve())
    assert "leiden" in sess.adata.obs
    assert r.summary["n_genes_ranked"] == 5


def test_registry_lists_b1_b2_tools():
    names = {t["name"] for t in tools.list_tools()}
    assert {"inspect", "load", "detect_state"} <= names


def test_detect_state_raw(tmp_path):
    save_h5ad(_raw(), tmp_path / "raw.h5ad")
    r = detect_state(str(tmp_path / "raw.h5ad"))
    assert r.status == "success"
    assert r.summary["stage"] == "raw"
    assert r.summary["reentry_point"] == "preprocess"
    assert r.summary["flags"]["has_counts"] is True


def test_detect_state_clustered(tmp_path):
    a = _raw()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=30)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=10)
    sc.tl.leiden(a, flavor="igraph", n_iterations=2, random_state=0)
    save_h5ad(a, tmp_path / "clust.h5ad")
    r = detect_state(str(tmp_path / "clust.h5ad"))
    s = r.summary
    assert s["flags"]["hvg"] and s["flags"]["pca"] and s["flags"]["clustered"]
    assert s["stage"] in ("clustered", "umap")  # umap not run -> clustered
    assert s["annotation_columns_present"] == []
