"""Unit tests for B13 Tier-2 fine annotation — fine_annotation_review (evidence) +
apply_fine_annotation (LLM call + deterministic HARD RULES: tiny-cluster merge,
insufficient-evidence → review). Marker-DB-free, same evidence→apply split as Tier-1/2."""

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot import tools
from scpilot.core.annotate import FINE_UNRESOLVED, UNS_ANNO, UNS_TREE
from scpilot.session import Session


def _two_program_adata(n=400):
    """A single compartment (major_cell_type='T_NK') with two distinct DE programs
    (→ two subclusters), plus a confounder score + sample/doublet obs."""
    rng = np.random.default_rng(0)
    prog_a = ["CD8A", "CD8B", "GZMB", "PDCD1"]      # e.g. CD8 / exhausted-like program
    prog_b = ["FOXP3", "IL2RA", "CTLA4", "IKZF2"]   # e.g. Treg-like program
    genes = prog_a + prog_b + [f"G{i}" for i in range(60)]
    X = rng.poisson(0.2, (n, len(genes))).astype("float32")
    half = n // 2
    a_cols = list(range(len(prog_a)))
    b_cols = list(range(len(prog_a), len(prog_a) + len(prog_b)))
    X[:half][:, a_cols] += rng.poisson(8.0, (half, len(prog_a))).astype("float32")
    X[half:][:, b_cols] += rng.poisson(8.0, (n - half, len(prog_b))).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = genes
    a.layers["counts"] = a.X.copy()
    a.obs["major_cell_type"] = "T_NK"               # one compartment (as after compartment_subset)
    a.obs["sample_id"] = rng.choice(["s1", "s2", "s3"], n)
    a.obs["cell_cycle_score"] = rng.standard_normal(n).astype("float32")   # a confounder score
    a.obs["predicted_doublet"] = rng.random(n) < 0.05
    return a


def _prep_subset(tmp_path, a=None):
    a = a if a is not None else _two_program_adata()
    p = tmp_path / "in.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    tools.run("preprocess", s, n_top_genes=40, n_pcs=15)
    tools.run("cluster", s, resolution=0.5)
    tools.run("markers", s, groupby="leiden")
    return s


# --------------------------------------------------------------------------- #
# fine_annotation_review (read-only evidence)
# --------------------------------------------------------------------------- #
def test_fine_review_packages_evidence(tmp_path):
    import json

    s = _prep_subset(tmp_path)
    # this test checks evidence PACKAGING → loosen the marker-quality filter so the synthetic
    # (cross-subcluster-leaky) markers survive; strict fine defaults are covered in test_annotate.
    r = tools.run("fine_annotation_review", s, groupby="leiden",
                  confounder_genes={"ifn": ["GZMB", "PDCD1"]},
                  min_in_group_fraction=0.0, max_out_group_fraction=1.0, min_fold_change=1.0)
    assert r.status == "success", r.error
    sm = r.summary
    assert sm["compartment"] == "T_NK"                       # inferred from uniform major_cell_type
    assert sm["n_subclusters"] >= 2
    assert "cell_cycle_score" in sm["confounder_keys_used"]  # existing obs score surfaced
    assert "ifn" in sm["confounder_genes_used"]              # caller genes scored on the fly
    ev = json.loads((s.artifacts_dir / "fine_annotation_evidence.json").read_text())
    sub0 = ev["subclusters"][0]
    assert sub0["de_table"] and "gene" in sub0["de_table"][0]
    assert sub0["compartment"] == "T_NK" and "confounders" in sub0
    assert "cell_cycle_score" in sub0["confounders"]
    # read-only contract: the on-the-fly score scratch column is cleaned up
    assert "_fine_conf_ifn" not in s.adata.obs.columns
    r.to_dict()


def test_fine_review_needs_markers(tmp_path):
    a = _two_program_adata()
    p = tmp_path / "in.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "s2", input_path=str(p)); s.load_input()
    tools.run("preprocess", s, n_top_genes=40, n_pcs=15)
    tools.run("cluster", s, resolution=0.5)                  # no markers → no DE
    r = tools.run("fine_annotation_review", s, groupby="leiden")
    assert r.status == "error" and r.error_code == "invalid_state"


# --------------------------------------------------------------------------- #
# apply_fine_annotation (LLM call + HARD RULES)
# --------------------------------------------------------------------------- #
def test_apply_fine_writes_columns_and_tree(tmp_path):
    s = _prep_subset(tmp_path)
    clusters = sorted(s.adata.obs["leiden"].astype(str).unique())
    fine = {c: ("Exhausted CD8 T" if i == 0 else "Regulatory T cell") for i, c in enumerate(clusters)}
    facs = {c: ("CD8+ PD-1+ T cells" if i == 0 else "FOXP3+ Tregs") for i, c in enumerate(clusters)}
    ef = {c: ["lineage program clearly present"] for c in clusters}
    r = tools.run("apply_fine_annotation", s, groupby="leiden", fine_labels=fine,
                  facs_labels=facs, evidence_for=ef, merge_min_cells=1,
                  confidence={c: 0.8 for c in clusters})
    assert r.status == "success", r.error
    obs = s.adata.obs
    assert "fine_cell_type" in obs and "facs_style_label" in obs
    assert set(obs["fine_cell_type"].astype(str).unique()) == set(fine.values())
    assert set(obs["facs_style_label"].astype(str).unique()) == set(facs.values())
    tree = s.adata.uns[UNS_ANNO][UNS_TREE]
    assert tree["marker_db_used"] is False and tree["compartment"] == "T_NK"
    for c in clusters:
        node = tree["subclusters"][c]
        assert node["major_cell_type"] == "T_NK" and node["fine_cell_type"] == fine[c]
        assert node["facs_style_label"] == facs[c] and node["merged"] is False
    assert r.summary["n_merged_subclusters"] == 0 and r.summary["n_review_required_subclusters"] == 0


def test_apply_fine_merges_tiny_subclusters(tmp_path):
    # merge_min_cells huge → every subcluster is under the floor → merged + review
    s = _prep_subset(tmp_path)
    clusters = sorted(s.adata.obs["leiden"].astype(str).unique())
    fine = {c: "Some subtype" for c in clusters}
    r = tools.run("apply_fine_annotation", s, groupby="leiden", fine_labels=fine,
                  evidence_for={c: ["x"] for c in clusters}, merge_min_cells=10_000)
    assert r.status == "success", r.error
    assert r.summary["n_merged_subclusters"] == len(clusters)
    assert r.summary["merge_label"] == f"T_NK_{FINE_UNRESOLVED}"
    assert set(s.adata.obs["fine_cell_type"].astype(str).unique()) == {f"T_NK_{FINE_UNRESOLVED}"}
    assert bool(s.adata.obs["fine_cell_type_review_required"].all())     # all flagged for review
    assert any("merge_min_cells" in w for w in r.warnings)


def test_apply_fine_insufficient_evidence_forces_review(tmp_path):
    s = _prep_subset(tmp_path)
    clusters = sorted(s.adata.obs["leiden"].astype(str).unique())
    fine = {c: "Some subtype" for c in clusters}
    # supply evidence_for for all but the first cluster → that one is forced to review
    ef = {c: ["clear program"] for c in clusters[1:]}
    r = tools.run("apply_fine_annotation", s, groupby="leiden", fine_labels=fine,
                  evidence_for=ef, merge_min_cells=1)
    assert r.status == "success", r.error
    assert r.summary["n_insufficient_evidence"] == 1
    tree = s.adata.uns[UNS_ANNO][UNS_TREE]["subclusters"]
    assert tree[clusters[0]]["review_required"] is True
    assert "insufficient_evidence" in tree[clusters[0]]["review_reasons"]


def test_apply_fine_separates_state_from_type(tmp_path):
    s = _prep_subset(tmp_path)
    clusters = sorted(s.adata.obs["leiden"].astype(str).unique())
    fine = {c: "CD8 T cell" for c in clusters}
    state = {clusters[0]: "exhausted"}
    r = tools.run("apply_fine_annotation", s, groupby="leiden", fine_labels=fine,
                  cell_state=state, evidence_for={c: ["x"] for c in clusters}, merge_min_cells=1)
    assert r.status == "success", r.error
    # cell STATE lives in its own column, not folded into fine_cell_type
    assert "cell_state" in s.adata.obs
    assert "exhausted" not in set(s.adata.obs["fine_cell_type"].astype(str).unique())
    assert "exhausted" in set(s.adata.obs["cell_state"].astype(str).unique())


def test_apply_fine_requires_labels(tmp_path):
    s = _prep_subset(tmp_path)
    r = tools.run("apply_fine_annotation", s, groupby="leiden")
    assert r.status == "error" and r.error_code == "missing_input"
