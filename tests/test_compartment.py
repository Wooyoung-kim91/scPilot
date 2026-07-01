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
    from scpilot.session import Session
    s = _session(_compartmented_adata(), tmp_path)
    n_parent = int(s.adata.n_obs)
    n_genes = int(s.adata.n_vars)
    r = tools.run("compartment_subset", s, compartment="T_NK", mode="clustering", use_rep="X_scVI")
    assert r.status == "success", r.error
    sm = r.summary
    assert sm["mode"] == "clustering" and sm["n_cells"] == 110
    assert sm["parent_n_cells"] == n_parent and sm["next_use_rep"] == "X_scVI"
    assert sm["recompute_rep"] is None
    # DEFAULT: parent is UNTOUCHED; the subset lives in its own child session directory
    assert s.adata.n_obs == n_parent                     # parent working adata not replaced
    assert sm["child_session_dir"] and sm["child_session_dir"].endswith("/compartments/T_NK")
    child = Session.open(sm["child_session_dir"])
    sub = child.adata                                     # loads the child's subset checkpoint
    assert sub.n_obs == 110
    assert set(sub.obs["major_cell_type"].unique()) == {"T_NK"}
    assert "X_scVI" in sub.obsm                           # integration embedding preserved
    assert sub.n_vars == n_genes and "counts" in sub.layers  # genes never dropped
    # provenance pointer back to the parent (reproducibility)
    assert child.manifest.derived_from["compartment"] == "T_NK"
    assert child.manifest.derived_from["parent_session_id"] == s.manifest.session_id
    assert r.determinism_grade == "A" and r.checkpoint


def test_compartment_subset_markers_mode_recomputes_features(tmp_path):
    from scpilot.session import Session
    s = _session(_compartmented_adata(), tmp_path)
    r = tools.run("compartment_subset", s, compartment="Epithelial", mode="markers",
                  n_top_genes=20, n_pcs=10)
    assert r.status == "success", r.error
    sm = r.summary
    assert sm["mode"] == "markers" and sm["n_cells"] == 180
    assert sm["x_state"] == "log1p" and sm["recompute_rep"] == "X_pca"
    sub = Session.open(sm["child_session_dir"]).adata
    assert "highly_variable" in sub.var and int(sub.var["highly_variable"].sum()) == sm["n_hvg"]
    assert "X_pca" in sub.obsm and "scale.data" in sub.layers
    assert r.determinism_grade == "B"


def test_compartment_subset_in_place_legacy(tmp_path):
    # in_place=True keeps the legacy single-session behaviour: subset REPLACES the parent adata.
    s = _session(_compartmented_adata(), tmp_path)
    r = tools.run("compartment_subset", s, compartment="T_NK", mode="clustering",
                  use_rep="X_scVI", in_place=True)
    assert r.status == "success", r.error
    assert r.summary["child_session_dir"] is None
    assert s.adata.n_obs == 110                           # subset replaced the working adata
    assert set(s.adata.obs["major_cell_type"].unique()) == {"T_NK"}


def test_child_session_loop_merges_back_to_parent(tmp_path):
    # END-TO-END: subset two compartments into their own child sessions, annotate each (fine labels),
    # then merge_fine_annotations(compartments_root) reassembles them onto the untouched parent.
    from scpilot.session import Session
    s = _session(_compartmented_adata(), tmp_path)
    n_parent = int(s.adata.n_obs)
    for comp in ("T_NK", "Epithelial"):
        r = tools.run("compartment_subset", s, compartment=comp, mode="clustering", use_rep="X_scVI")
        assert r.status == "success", r.error
        child = Session.open(r.summary["child_session_dir"])
        sub = child.adata
        sub.obs["fine_cell_type"] = f"{comp}_subtypeA"
        sub.obs["facs_style_label"] = f"{comp}+ cells"
        sub.obs["cell_state"] = "resting"
        child.set_adata(sub)
        child.checkpoint("apply_fine_annotation", x_state=child.manifest.x_state, params={})
    # parent never changed by the subsetting
    assert s.adata.n_obs == n_parent and "fine_cell_type" not in s.adata.obs
    m = tools.run("merge_fine_annotations", s, compartments_root=str(tmp_path / "sess" / "compartments"))
    assert m.status == "success", m.error
    sm = m.summary
    assert sm["n_sources"] == 2
    assert "fine_cell_type" in s.adata.obs
    fine = set(s.adata.obs["fine_cell_type"].astype(str).unique())
    assert {"T_NK_subtypeA", "Epithelial_subtypeA"} <= fine
    # the un-subclustered compartment carries its Tier-1 major_cell_type forward
    assert sm["n_carried_terminal"] >= 1


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
