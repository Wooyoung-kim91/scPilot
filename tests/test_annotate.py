"""Unit tests for B8 Tier 1 broad annotation (marker-anchored, de-risk ①)."""

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

from scpilot import tools
from scpilot.core.annotate import BROAD_MARKERS, UNS_ANNO
from scpilot.session import Session


def _two_lineage_adata(n=240):
    """Half cells express Epithelial markers, half express T_NK; + a GSE batch col."""
    rng = np.random.default_rng(0)
    genes = BROAD_MARKERS["Epithelial"] + BROAD_MARKERS["T_NK"] + [f"G{i}" for i in range(40)]
    X = rng.poisson(0.3, (n, len(genes))).astype("float32")
    half = n // 2
    epi_cols = list(range(len(BROAD_MARKERS["Epithelial"])))
    tnk_cols = list(range(len(BROAD_MARKERS["Epithelial"]),
                          len(BROAD_MARKERS["Epithelial"]) + len(BROAD_MARKERS["T_NK"])))
    X[:half][:, epi_cols] += rng.poisson(6.0, (half, len(epi_cols))).astype("float32")
    X[half:][:, tnk_cols] += rng.poisson(6.0, (n - half, len(tnk_cols))).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = genes
    a.layers["counts"] = a.X.copy()
    a.obs["GSE"] = rng.choice(["GSE_A", "GSE_B"], n)   # mixed batches (not circular)
    return a


def _prep(a, tmp_path):
    p = tmp_path / "in.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    tools.run("preprocess", s, n_top_genes=30, n_pcs=15)
    tools.run("cluster", s, resolution=0.5)
    return s


def test_annotate_broad_assigns_marker_anchored_labels(tmp_path):
    s = _prep(_two_lineage_adata(), tmp_path)
    r = tools.run("annotate_broad", s)
    assert r.status == "success"
    sm = r.summary
    assert sm["label_key"] == "major_cell_type"
    # both planted lineages recovered
    labels = set(sm["label_distribution"])
    assert "Epithelial" in labels and "T_NK" in labels
    assert "major_cell_type" in s.adata.obs and "major_confidence" in s.adata.obs
    assert sm["mean_confidence"] > 0.5
    # evidence tree written
    assert UNS_ANNO in s.adata.uns and "tier1" in s.adata.uns[UNS_ANNO]
    # scratch score cols cleaned up
    assert not any(c.startswith("_sc_") for c in s.adata.obs.columns)
    r.to_dict()


def test_circular_risk_flagged_when_cluster_is_single_gse(tmp_path):
    # make GSE perfectly aligned with lineage → each cluster single-GSE → circular risk
    a = _two_lineage_adata()
    a.obs["GSE"] = (["GSE_A"] * (a.n_obs // 2) + ["GSE_B"] * (a.n_obs - a.n_obs // 2))
    s = _prep(a, tmp_path)
    r = tools.run("annotate_broad", s, circular_frac=0.8)
    assert r.status == "success"
    assert r.summary["circular_risk_clusters"]            # at least one flagged
    assert any("circular" in w for w in r.warnings)


def test_annotate_broad_needs_markers(tmp_path):
    # var_names with no broad markers at all
    a = _two_lineage_adata()
    a.var_names = [f"X{i}" for i in range(a.n_vars)]
    s = _prep(a, tmp_path)
    r = tools.run("annotate_broad", s)
    assert r.status == "error" and r.error_code == "data_gate_failed"


def test_registry_has_annotate_broad():
    assert "annotate_broad" in {t["name"] for t in tools.list_tools()}
