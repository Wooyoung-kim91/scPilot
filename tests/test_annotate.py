"""Unit tests for B8 Tier 1 broad annotation — leiden-DE marker-combination logic."""

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

from scpilot import tools
from scpilot.core.annotate import (ARTIFACT_LABELS, BROAD_MARKERS, LOWQ_LABEL,
                                   MIXED_LABEL, UNS_ANNO)
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


# ---------------------------------------------------------------------------
# new Tier-1 rules (2026-06-11): top-30 positives, negative-marker tie-break,
# Mixed/Artifact, Low_quality QC gate, PTPRC consistency.
# Pre-assigned leiden labels (annotate only needs groupby + expression for DE),
# with a large background cluster so one-vs-rest LFCs aren't diluted by overlap.
# ---------------------------------------------------------------------------
def _ruleset_session(tmp_path):
    rng = np.random.default_rng(0)
    epi, tnk = BROAD_MARKERS["Epithelial"], BROAD_MARKERS["T_NK"]
    genes = epi + tnk + ["PTPRC"] + [f"G{i}" for i in range(40)]
    gi = {g: i for i, g in enumerate(genes)}
    # cluster sizes (bg large so it dominates "rest")
    spec = [("0", 300), ("1", 50), ("2", 50), ("3", 50), ("4", 50), ("5", 50), ("6", 50)]
    leiden = sum(([c] * n for c, n in spec), [])
    N = len(leiden)
    X = rng.poisson(0.1, (N, len(genes))).astype("float32")

    def boost(cl, cols, lam=10.0):
        idx = np.array([i for i, c in enumerate(leiden) if c == cl])
        for col in cols:
            X[np.ix_(idx, [col])] += rng.poisson(lam, (idx.size, 1)).astype("float32")

    boost("0", [gi[f"G{i}"] for i in range(10)])           # background noise → Unknown
    boost("1", [gi[g] for g in epi])                        # Epithelial (PTPRC-)
    boost("2", [gi[g] for g in tnk] + [gi["PTPRC"]])        # T_NK (PTPRC+)  consistent
    boost("3", [gi[g] for g in epi] + [gi[g] for g in tnk]) # epi+tnk co-expr → Mixed (conflict)
    boost("4", [gi[g] for g in epi])                        # Epithelial markers BUT high %MT → Low_quality
    boost("5", [gi[g] for g in tnk])                        # T_NK markers, PTPRC- → ptprc-inconsistent
    boost("6", [gi[g] for g in tnk])                        # T_NK markers, doublet+ → Mixed (doublet path)

    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = genes
    a.obs["leiden"] = leiden
    a.obs["leiden"] = a.obs["leiden"].astype("category")
    a.layers["counts"] = a.X.copy()
    a.obs["sample_id"] = rng.choice(["s1", "s2", "s3"], N)
    a.obs["pct_counts_mt"] = np.where(np.array(leiden) == "4", 40.0, 3.0).astype("float32")
    # constant high complexity → only the %MT gate fires here (low-gene gate tested elsewhere)
    a.obs["n_genes_by_counts"] = np.full(N, 1500.0, dtype="float32")
    a.obs["predicted_doublet"] = (np.array(leiden) == "6")
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    p = tmp_path / "rules.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "rsess", input_path=str(p)); s.load_input()
    return s


def test_annotate_broad_new_rules(tmp_path):
    s = _ruleset_session(tmp_path)
    r = tools.run("annotate_broad", s, groupby="leiden")
    assert r.status == "success", r.error
    ev = s.adata.uns[UNS_ANNO]["tier1"]["clusters"]

    # pure lineage clusters resolve correctly
    assert ev["1"]["label"] == "Epithelial"
    assert ev["2"]["label"] == "T_NK" and ev["2"]["ptprc_consistent"] is True
    # rule 1: co-expression of two incompatible panels → Mixed/Artifact (not a single type)
    assert ev["3"]["label"] == MIXED_LABEL and ev["3"]["marker_conflict"] is True
    # rule 4: high-%MT cluster gated to Low_quality even though epi markers are present
    assert ev["4"]["label"] == LOWQ_LABEL and ev["4"]["review_required"] is True
    # rule 2: immune label but PTPRC- → flagged inconsistent + confidence penalized + review
    assert ev["5"]["label"] == "T_NK" and ev["5"]["ptprc_consistent"] is False
    assert ev["5"]["review_required"] is True
    # rule 1 (doublet path): doublet-dominated cluster → Mixed/Artifact
    assert ev["6"]["label"] == MIXED_LABEL and ev["6"]["doublet_dominated"] is True

    sm = r.summary
    assert sm["mixed_artifact_clusters"] and sm["low_quality_clusters"]
    assert sm["ptprc_inconsistent_clusters"] == ["5"]
    assert "top-30" in sm["method"]
    assert s.adata.uns[UNS_ANNO]["tier1"]["top_n_markers"] == 30
    assert "major_review_required" in s.adata.obs


def test_annotate_broad_top_n_limits_positives(tmp_path):
    # top_n_markers=1 keeps only the single strongest DE gene per cluster → a 3-marker
    # panel call is impossible, so even clean lineages fall back to Unknown.
    s = _ruleset_session(tmp_path)
    r = tools.run("annotate_broad", s, groupby="leiden", top_n_markers=1)
    assert r.status == "success"
    ev = s.adata.uns[UNS_ANNO]["tier1"]["clusters"]
    assert ev["1"]["label"] == "Unknown" and ev["2"]["label"] == "Unknown"


def test_artifact_labels_dropped_by_benchmark_default():
    # the non-biological labels are the benchmark's default drop set (de-risk ①)
    assert {"Unknown", MIXED_LABEL, LOWQ_LABEL} == ARTIFACT_LABELS


def test_registry_has_annotate_broad():
    assert "annotate_broad" in {t["name"] for t in tools.list_tools()}
