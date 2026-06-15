"""Unit tests for B11 compartment_plan (read-only branch floor) + compartment_subset (2 modes)."""

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot import tools
from scpilot.session import Session


def _compartmented_adata(n=300):
    """Three broad compartments of unequal size + a batch/sample axis.

    Epithelial (big, multi-batch) / T_NK (medium, multi-batch) / Mast (tiny,
    single-batch) — so the floor blocks the tiny single-batch compartment.
    """
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(40)]
    X = rng.poisson(0.5, (n, len(genes))).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = genes
    a.layers["counts"] = a.X.copy()

    comp = np.array(["Epithelial"] * 180 + ["T_NK"] * 110 + ["Mast"] * 10)
    a.obs["major_cell_type"] = comp
    # Epithelial + T_NK spread over 3 GSE/4 samples; Mast confined to one GSE/sample
    gse = np.where(comp == "Mast", "GSEc",
                   rng.choice(["GSEa", "GSEb", "GSEc"], n))
    sample = np.where(comp == "Mast", "s_mast",
                      rng.choice(["s1", "s2", "s3", "s4"], n))
    a.obs["GSE"] = gse
    a.obs["sample_id"] = sample
    # a fake integration embedding (for mode='clustering')
    a.obsm["X_scVI"] = rng.standard_normal((n, 10)).astype("float32")
    return a


def _session(a, tmp_path):
    p = tmp_path / "in.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    return s


# --------------------------------------------------------------------------- #
# compartment_plan
# --------------------------------------------------------------------------- #
def test_compartment_plan_counts_and_floor(tmp_path):
    s = _session(_compartmented_adata(), tmp_path)
    r = tools.run("compartment_plan", s, min_cells=50, min_samples=2)
    assert r.status == "success", r.error
    sm = r.summary
    assert sm["groupby"] == "major_cell_type" and sm["batch_key"] == "GSE"
    assert sm["n_compartments"] == 3
    # big multi-batch compartments clear the floor; the tiny single-batch one is blocked
    assert set(sm["branchable"]) == {"Epithelial", "T_NK"}
    assert sm["blocked"] == ["Mast"]
    r.to_dict()


def test_compartment_plan_batch_mixing_and_dominance(tmp_path):
    s = _session(_compartmented_adata(), tmp_path)
    r = tools.run("compartment_plan", s, min_cells=50)
    ev = {c["compartment"]: c for c in
          __import__("json").loads((s.artifacts_dir / "compartment_plan.json").read_text())["compartments"]}
    # Mast is one batch out of 3 globally → low entropy + single-patient dominated + blocked
    mast = ev["Mast"]
    assert mast["batch_mixing"]["n_batches"] == 1
    assert mast["batch_mixing"]["batch_entropy_norm"] == 0.0
    assert mast["sample_coverage"]["single_patient_dominated"] is True
    assert mast["branch_recommended"] is False
    assert "single_batch" in mast["branch_block_reasons"]
    # Epithelial is well mixed across batches
    assert ev["Epithelial"]["batch_mixing"]["batch_entropy_norm"] > 0.5


def test_compartment_plan_requires_compartment_key(tmp_path):
    a = _compartmented_adata()
    del a.obs["major_cell_type"]
    a.obs.drop(columns=[c for c in ("celltype_consensus", "leiden") if c in a.obs], inplace=True, errors="ignore")
    s = _session(a, tmp_path)
    r = tools.run("compartment_plan", s)
    assert r.status == "error" and r.error_code == "invalid_state"


# --------------------------------------------------------------------------- #
# compartment_subset
# --------------------------------------------------------------------------- #
def test_compartment_subset_clustering_mode_keeps_embedding(tmp_path):
    s = _session(_compartmented_adata(), tmp_path)
    n_parent = int(s.adata.n_obs)
    n_genes = int(s.adata.n_vars)
    r = tools.run("compartment_subset", s, compartment="T_NK", mode="clustering", use_rep="X_scVI")
    assert r.status == "success", r.error
    sm = r.summary
    assert sm["mode"] == "clustering" and sm["n_cells"] == 110
    assert sm["parent_n_cells"] == n_parent and sm["next_use_rep"] == "X_scVI"
    assert sm["recompute_rep"] is None
    sub = s.adata
    assert sub.n_obs == 110                               # subset replaced the working adata
    assert set(sub.obs["major_cell_type"].unique()) == {"T_NK"}
    assert "X_scVI" in sub.obsm                           # integration embedding preserved
    assert sub.n_vars == n_genes and "counts" in sub.layers  # genes never dropped
    assert r.determinism_grade == "A" and r.checkpoint


def test_compartment_subset_markers_mode_recomputes_features(tmp_path):
    s = _session(_compartmented_adata(), tmp_path)
    r = tools.run("compartment_subset", s, compartment="Epithelial", mode="markers",
                  n_top_genes=20, n_pcs=10)
    assert r.status == "success", r.error
    sm = r.summary
    assert sm["mode"] == "markers" and sm["n_cells"] == 180
    assert sm["x_state"] == "log1p" and sm["recompute_rep"] == "X_pca"
    sub = s.adata
    assert "highly_variable" in sub.var and int(sub.var["highly_variable"].sum()) == sm["n_hvg"]
    assert "X_pca" in sub.obsm and "scale.data" in sub.layers
    assert r.determinism_grade == "B"


def test_compartment_subset_unknown_compartment_errors(tmp_path):
    s = _session(_compartmented_adata(), tmp_path)
    r = tools.run("compartment_subset", s, compartment="DoesNotExist", mode="clustering")
    assert r.status == "error" and r.error_code == "data_gate_failed"


def test_compartment_subset_clustering_requires_embedding(tmp_path):
    a = _compartmented_adata()
    del a.obsm["X_scVI"]
    s = _session(a, tmp_path)
    r = tools.run("compartment_subset", s, compartment="T_NK", mode="clustering", use_rep="X_scVI")
    assert r.status == "error" and r.error_code == "invalid_state"


def test_compartment_subset_bad_mode_errors(tmp_path):
    s = _session(_compartmented_adata(), tmp_path)
    r = tools.run("compartment_subset", s, compartment="T_NK", mode="nonsense")
    assert r.status == "error" and r.error_code == "missing_input"
