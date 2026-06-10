"""Unit tests for B8 Tier 1 broad annotation — leiden-DE marker-combination logic."""

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

from scpilot import tools
from scpilot.core.annotate import BROAD_MARKERS, UNS_ANNO
from scpilot.session import Session


def _two_lineage_adata(n=400):
    """Two groups: one strongly expresses Epithelial markers, the other T_NK."""
    rng = np.random.default_rng(0)
    epi, tnk = BROAD_MARKERS["Epithelial"], BROAD_MARKERS["T_NK"]
    genes = epi + tnk + [f"G{i}" for i in range(60)]
    X = rng.poisson(0.2, (n, len(genes))).astype("float32")
    half = n // 2
    epi_cols = list(range(len(epi)))
    tnk_cols = list(range(len(epi), len(epi) + len(tnk)))
    X[:half][:, epi_cols] += rng.poisson(8.0, (half, len(epi))).astype("float32")
    X[half:][:, tnk_cols] += rng.poisson(8.0, (n - half, len(tnk))).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = genes
    a.layers["counts"] = a.X.copy()
    a.obs["sample_id"] = rng.choice(["s1", "s2", "s3"], n)
    a.obs["GSE"] = rng.choice(["GSEa", "GSEb"], n)
    return a


def _prep(a, tmp_path):
    p = tmp_path / "in.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    tools.run("preprocess", s, n_top_genes=40, n_pcs=15)
    tools.run("cluster", s, resolution=0.5)
    return s


def test_annotate_broad_de_marker_combination(tmp_path):
    s = _prep(_two_lineage_adata(), tmp_path)
    r = tools.run("annotate_broad", s, min_markers=3)
    assert r.status == "success", r.error
    sm = r.summary
    assert "leiden_DE_marker_combination" in sm["method"]
    labels = set(sm["label_distribution"])
    assert "Epithelial" in labels and "T_NK" in labels      # both lineages recovered
    assert "major_cell_type" in s.adata.obs and "major_confidence" in s.adata.obs
    # evidence: each non-Unknown cluster cites >=3 matched markers
    tier1 = s.adata.uns[UNS_ANNO]["tier1"]
    assert tier1["min_markers"] == 3 and tier1["min_pct"] == 0.25 and tier1["min_lfc"] == 1.0
    for cl, ev in tier1["clusters"].items():
        if ev["label"] != "Unknown":
            assert ev["n_markers_matched"] >= 3
            assert len(ev["matched_markers"]) >= 3
            assert "n_samples" in ev          # provenance recorded
    r.to_dict()


def test_annotate_broad_requires_min_3_markers(tmp_path):
    # with min_markers=99, nothing can be called → all Unknown
    s = _prep(_two_lineage_adata(), tmp_path)
    r = tools.run("annotate_broad", s, min_markers=99)
    assert r.status == "success"
    assert set(s.adata.obs["major_cell_type"].unique()) == {"Unknown"}
    assert r.summary["unknown_clusters"]


def test_annotate_broad_flags_single_source(tmp_path):
    # make sample perfectly aligned with lineage → clusters single-sample dominated
    a = _two_lineage_adata()
    a.obs["sample_id"] = (["s1"] * (a.n_obs // 2) + ["s2"] * (a.n_obs - a.n_obs // 2))
    s = _prep(a, tmp_path)
    r = tools.run("annotate_broad", s, single_source_frac=0.8)
    assert r.summary["single_source_clusters"]
    assert any("single-sample" in w for w in r.warnings)


def test_annotate_broad_needs_cluster(tmp_path):
    a = _two_lineage_adata()
    p = tmp_path / "raw.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "s2", input_path=str(p)); s.load_input()
    r = tools.run("annotate_broad", s)            # no leiden yet
    assert r.status == "error" and r.error_code == "invalid_state"


def test_dotplot_with_celltype_brackets(tmp_path):
    s = _prep(_two_lineage_adata(), tmp_path)
    tools.run("annotate_broad", s, min_markers=3)
    r = tools.run("plots", s, kind="dotplot", groupby="major_cell_type")
    assert r.status == "success", r.error
    assert r.artifacts and r.artifacts[0].kind == "png"
    from pathlib import Path
    assert Path(r.artifacts[0].path).exists()


def test_registry_has_annotate_broad():
    assert "annotate_broad" in {t["name"] for t in tools.list_tools()}
