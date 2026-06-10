"""Unit tests for B3 QC tools (qc_metrics + qc_filter)."""

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot import tools
from scpilot.session import Session


def _qc_fixture(n_per_sample=90):
    """Two samples; named genes incl. MT-/ribo/EPCAM/CD3D; counts in X + layer."""
    rng = np.random.default_rng(0)
    genes = (["MT-CO1", "MT-ND1", "RPS6", "RPL7", "EPCAM", "CD3D"]
             + [f"G{i}" for i in range(44)])
    n_obs, n_vars = 2 * n_per_sample, len(genes)
    X = rng.poisson(1.0, (n_obs, n_vars)).astype("float32")
    # give some cells MT + co-expression signal
    X[:, 0:2] += rng.poisson(2.0, (n_obs, 2)).astype("float32")
    X[:10, 4] += 5; X[:10, 5] += 5  # first 10 cells: EPCAM+CD3D co-expression
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = genes
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    a.layers["counts"] = a.X.copy()
    a.obs["sample_id"] = (["s1"] * n_per_sample + ["s2"] * n_per_sample)
    return a


def _session_with(adata, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "in.h5ad"
    adata.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p))
    s.load_input()
    return s


def test_qc_metrics_computes_and_summarizes(tmp_path):
    s = _session_with(_qc_fixture(), tmp_path)
    r = tools.run("qc_metrics", s, run_scrublet=True, seed=0)
    assert r.status == "success"
    sm = r.summary
    assert sm["n_samples"] == 2
    assert "pct_counts_mt" in sm["global_distributions"]
    assert "pct_counts_ribo" in sm["global_distributions"]
    assert sm["mixed_lineage_frac"] is not None and sm["mixed_lineage_frac"] > 0
    # per-sample batch-aware table present
    assert "per_sample_qc" in r.tables
    # obs columns written
    assert "pct_counts_mt" in s.adata.obs and "mixed_lineage_flag" in s.adata.obs
    # scrublet either produced scores or recorded a skip — never crashes
    assert ("doublet_score" in s.adata.obs) or sm["scrublet_skipped_samples"]
    assert r.determinism_grade == "B"
    assert (tmp_path / "sess" / "checkpoints").exists()
    r.to_dict()


def test_qc_metrics_without_scrublet_is_grade_a(tmp_path):
    s = _session_with(_qc_fixture(), tmp_path)
    r = tools.run("qc_metrics", s, run_scrublet=False)
    assert r.status == "success"
    assert r.determinism_grade == "A"
    assert "doublet_score" not in r.summary["qc_metrics"]


def test_qc_filter_requires_metrics_first(tmp_path):
    s = _session_with(_qc_fixture(), tmp_path)
    r = tools.run("qc_filter", s)
    assert r.status == "error"
    assert r.error_code == "invalid_state"


def test_qc_filter_subsets_cells(tmp_path):
    s = _session_with(_qc_fixture(), tmp_path)
    tools.run("qc_metrics", s, run_scrublet=False)
    n_before = s.adata.n_obs
    r = tools.run("qc_filter", s, min_genes=5, max_pct_mt=100.0)
    assert r.status == "success"
    assert r.summary["n_cells_before"] == n_before
    assert r.summary["n_cells_after"] <= n_before
    assert set(r.summary["per_sample"]) == {"s1", "s2"}
    # cutoffs that remove everything -> recoverable error, NOT an empty checkpoint
    s2 = _session_with(_qc_fixture(), tmp_path / "b")
    tools.run("qc_metrics", s2, run_scrublet=False)
    r2 = tools.run("qc_filter", s2, min_genes=10_000, max_pct_mt=100.0)
    assert r2.status == "error" and r2.error_code == "convergence_failed"
    assert r2.recoverable is True
