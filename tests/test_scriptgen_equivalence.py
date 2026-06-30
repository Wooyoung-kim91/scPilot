"""Regression: each transpiled standalone step (scpilot.scriptgen) reproduces its scpilot tool.

The standalone script inlines the SAME recipe source the tool calls, so equivalence is structural;
these tests confirm (a) the generated script is genuinely scpilot-free and (b) the read/write wiring
round-trips — executed as a real subprocess — to the identical result the scpilot tool produces."""

import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot import scriptgen, tools
from scpilot.session import Session


def _fixture(n=200, g=60):
    """Two clear populations (so clustering finds >1 cluster) + 2 samples; counts in X + layer."""
    rng = np.random.default_rng(0)
    genes = ["MT-CO1", "MT-ND1", "RPS6", "RPL7", "EPCAM", "CD3D"] + [f"G{i}" for i in range(g - 6)]
    X = rng.poisson(1.0, (n, g)).astype("float32")
    X[:, 0:2] += rng.poisson(2.0, (n, 2)).astype("float32")          # MT signal
    X[: n // 2, 6:16] += rng.poisson(5.0, (n // 2, 10)).astype("float32")   # population A
    X[n // 2:, 16:26] += rng.poisson(5.0, (n - n // 2, 10)).astype("float32")  # population B
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = genes
    a.obs_names = [f"c{i}" for i in range(n)]
    a.layers["counts"] = a.X.copy()
    a.obs["sample_id"] = [f"s{i % 2}" for i in range(n)]
    return a


def _run_standalone(tmp_path, cid, stage, params, in_path):
    """Generate the standalone step script, assert it is scpilot-free, run it in a subprocess,
    and return the AnnData it wrote to standalone_data/CID_STAGE.h5ad."""
    code_dir = tmp_path / "code"
    code_dir.mkdir(exist_ok=True)
    script = scriptgen.build(int(cid), cid, stage, params, in_expr=repr(str(in_path)))
    assert script is not None
    for forbidden in ("import scpilot", "from scpilot", "tools.run", "Session"):
        assert forbidden not in script, forbidden
    f = code_dir / f"{cid}_{stage}.py"
    f.write_text(script)
    r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "standalone_data" / f"{cid}_{stage}.h5ad"
    assert out.exists()
    return ad.read_h5ad(out)


def _session(tmp_path):
    p = tmp_path / "in.h5ad"
    _fixture().write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p))
    s.load_input()
    return s


def test_qc_filter_standalone_matches_tool(tmp_path):
    params = {"min_genes": 5, "max_pct_mt": 30.0, "max_doublet_score": 0.25}
    s = _session(tmp_path)
    tools.run("qc_metrics", s, run_scrublet=True, seed=0)
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)                       # standalone INPUT (has QC obs cols)
    tools.run("qc_filter", s, **params)
    tool_kept = set(map(str, s.adata.obs_names))

    out = _run_standalone(tmp_path, "03", "qc_filter", params, snap)
    assert set(map(str, out.obs_names)) == tool_kept and out.n_obs > 0


def test_preprocess_standalone_matches_tool(tmp_path):
    params = {"n_top_genes": 30, "n_pcs": 15, "seed": 0}
    s = _session(tmp_path)
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)
    tools.run("preprocess", s, **params)
    tool_pca = np.asarray(s.adata.obsm["X_pca"])
    tool_hvg = s.adata.var["highly_variable"].to_numpy()

    out = _run_standalone(tmp_path, "01", "preprocess", params, snap)
    assert (out.var["highly_variable"].to_numpy() == tool_hvg).all()
    assert np.allclose(np.asarray(out.obsm["X_pca"]), tool_pca, atol=1e-5)


def test_cluster_standalone_matches_tool(tmp_path):
    params = {"resolution": 0.5, "seed": 0}
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=30, n_pcs=15, seed=0)
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)                       # post-preprocess state (has X_pca)
    tools.run("cluster", s, **params)
    tool_leiden = s.adata.obs["leiden"].astype(str).to_numpy()

    out = _run_standalone(tmp_path, "02", "cluster", params, snap)
    assert (out.obs["leiden"].astype(str).to_numpy() == tool_leiden).all()
    assert len(set(tool_leiden)) >= 2


def test_markers_standalone_matches_tool(tmp_path):
    params = {"groupby": "leiden", "max_genes_ranked": 20}
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=30, n_pcs=15, seed=0)
    tools.run("cluster", s, resolution=0.5, seed=0)
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)                       # post-cluster state (has leiden)
    tools.run("markers", s, **params)
    tool_rg = s.adata.uns["rank_genes_groups"]
    tool_groups = list(tool_rg["names"].dtype.names)

    out = _run_standalone(tmp_path, "03", "markers", params, snap)
    out_rg = out.uns["rank_genes_groups"]
    assert list(out_rg["names"].dtype.names) == tool_groups
    for g in tool_groups:                          # identical ranked gene order per cluster
        assert [str(x) for x in out_rg["names"][g]] == [str(x) for x in tool_rg["names"][g]]


def test_qc_metrics_standalone_matches_tool(tmp_path):
    params = {"run_scrublet": True, "seed": 0}     # mixed_lineage opt-in left off (matches real run)
    s = _session(tmp_path)
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)
    tools.run("qc_metrics", s, **params)
    tool = s.adata.obs

    out = _run_standalone(tmp_path, "01", "qc_metrics", params, snap)
    o = out.obs
    for col in ("n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo"):
        assert np.allclose(o[col].to_numpy(float), tool[col].to_numpy(float), equal_nan=True), col
    if "doublet_score" in tool:                    # per-sample scrublet (seed-pinned -> identical)
        assert np.allclose(o["doublet_score"].to_numpy(float), tool["doublet_score"].to_numpy(float),
                           equal_nan=True)
        assert (o["predicted_doublet"].to_numpy() == tool["predicted_doublet"].to_numpy()).all()


def _write_10x(dir_path, genes, n_cells, sid, rng):
    """Write a cellranger-v3 filtered_feature_bc_matrix dir (gzipped mtx/barcodes/features)."""
    import gzip
    from scipy import io, sparse
    dir_path.mkdir(parents=True, exist_ok=True)
    X = sparse.csr_matrix(rng.poisson(1.0, (n_cells, len(genes))).astype("float32"))
    with gzip.open(dir_path / "matrix.mtx.gz", "wb") as fh:
        io.mmwrite(fh, X.T.tocoo())                 # 10x stores genes × cells
    with gzip.open(dir_path / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(f"{sid}_BC{i}" for i in range(n_cells)) + "\n")
    with gzip.open(dir_path / "features.tsv.gz", "wt") as fh:
        fh.write("\n".join(f"ENSG{i}\t{g}\tGene Expression" for i, g in enumerate(genes)) + "\n")


def _ingest_fixture(tmp_path):
    """A 2-sample raw-10x dataset + metadata CSV + scpilot ingest profile."""
    import pandas as pd
    rng = np.random.default_rng(0)
    genes = ["MT-CO1", "MT-ND1", "EPCAM", "CD3D", "PTPRC", "COL1A1"]
    root = tmp_path / "cellranger"
    rows = []
    for k, sid in enumerate(["S1", "S2"]):
        rel = f"{sid}/outs/filtered_feature_bc_matrix"
        _write_10x(root / rel, genes, 12 + k * 4, sid, rng)
        # NB: no 'sample_id' column — read_one_sample creates obs['sample_id'] from the library value
        rows.append({"library": sid, "mtx_dir": rel, "condition": ["ctrl", "trt"][k]})
    meta_csv = tmp_path / "samples.csv"
    pd.DataFrame(rows).to_csv(meta_csv, index=False)
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        f"profile_name: testfix\ninput_root: {root}\nmetadata_csv: {meta_csv}\nout_dir: {tmp_path}\n"
        "sample_id_col: library\nmatrix_dir_col: mtx_dir\nbatch_col: sample_id\n"
        "min_genes: 0\nmax_pct_mt: 100.0\nmin_cells: 1\ntarget_sum: 10000.0\nmito_prefix: MT-\n"
        "normalized_layer: scale.data\n")
    return profile


def test_ingest_standalone_matches_tool(tmp_path):
    profile = _ingest_fixture(tmp_path)
    s = Session.create(tmp_path / "sess", input_path=str(profile))
    tools.run("ingest", s)
    tool = s.adata
    tool_counts = np.asarray(tool.layers["counts"].todense()) if hasattr(tool.layers["counts"], "todense") \
        else np.asarray(tool.layers["counts"])

    out = _run_standalone(tmp_path, "00", "ingest", {}, profile)
    assert (out.n_obs, out.n_vars) == (tool.n_obs, tool.n_vars)
    assert out.obs["sample_id"].astype(str).value_counts().to_dict() == \
        tool.obs["sample_id"].astype(str).value_counts().to_dict()
    assert sorted(out.var_names) == sorted(tool.var_names)
    assert set(out.layers) >= {"counts", "scale.data"}
    out_counts = np.asarray(out.layers["counts"].todense()) if hasattr(out.layers["counts"], "todense") \
        else np.asarray(out.layers["counts"])
    # align gene/cell order then compare raw counts
    oo = out[tool.obs_names, tool.var_names]
    oc = np.asarray(oo.layers["counts"].todense()) if hasattr(oo.layers["counts"], "todense") \
        else np.asarray(oo.layers["counts"])
    assert np.allclose(oc, tool_counts)


def test_detect_state_standalone_matches_tool(tmp_path):
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=30, n_pcs=15, seed=0)
    tools.run("cluster", s, resolution=0.5, seed=0)
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)
    r = tools.run("detect_state", s, path=str(snap))

    _run_standalone(tmp_path, "02", "detect_state", {}, snap)
    stage = (tmp_path / "standalone_data" / "02_detect_state_state.txt").read_text()
    assert stage == r.summary["stage"]


def test_compartment_plan_standalone_matches_tool(tmp_path):
    import json
    params = {"min_cells": 5, "min_samples": 1}
    s, _ = _annotated_state(tmp_path)               # has major_cell_type
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)
    r = tools.run("compartment_plan", s, **params)

    _run_standalone(tmp_path, "07", "compartment_plan", params, snap)
    plan = json.loads((tmp_path / "standalone_data" / "07_compartment_plan_plan.json").read_text())
    assert sorted(map(str, plan["branchable"])) == sorted(map(str, r.summary["branchable"]))
    assert sorted(map(str, plan["blocked"])) == sorted(map(str, r.summary["blocked"]))


def test_benchmark_standalone_matches_tool(tmp_path):
    import pytest
    pytest.importorskip("scib_metrics")
    pytest.importorskip("harmonypy")
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=30, n_pcs=15, seed=0)
    rh = tools.run("integrate_harmony", s, batch_key="sample_id", use_rep="X_pca", seed=0)
    if rh.status != "success":
        pytest.skip("harmony unavailable")
    s.adata.obs["major_cell_type"] = np.where(np.arange(s.adata.n_obs) % 2 == 0, "A", "B")
    params = {"label_key": "major_cell_type", "batch_key": "sample_id",
              "embeddings": ["X_pca", "X_harmony"], "min_label_cells": 5, "subsample": None, "seed": 0}
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)
    try:                                            # scib needs a sizable dataset; tiny fixture may not
        r = tools.run("benchmark", s, **params)     # support kNN/LISI — skip rather than assert on noise
    except Exception as exc:                        # noqa: BLE001
        pytest.skip(f"scib Benchmarker cannot run on the tiny fixture: {type(exc).__name__}: {exc}")
    if r.status != "success":
        pytest.skip(f"benchmark unavailable on fixture: {r.error}")

    import pandas as pd
    _run_standalone(tmp_path, "08", "benchmark", params, snap)
    std = pd.read_csv(tmp_path / "standalone_data" / "08_benchmark_scib.csv", index_col=0)
    # the standalone scores the same embeddings the tool did (scib seed-pinned on identical inputs)
    assert {"X_pca", "X_harmony"}.issubset(set(std.index))
    assert set(r.summary["scores"].keys()).issubset(set(map(str, std.index)))


def test_load_standalone_matches_tool(tmp_path):
    p = tmp_path / "in.h5ad"
    _fixture().write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p))
    tools.run("load", s)
    tool_shape = (s.adata.n_obs, s.adata.n_vars)

    out = _run_standalone(tmp_path, "00", "load", {}, p)   # step 0 reads the raw input path
    assert (out.n_obs, out.n_vars) == tool_shape
    assert list(out.var_names) == list(s.adata.var_names)


def test_integrate_harmony_standalone_matches_tool(tmp_path):
    import pytest
    pytest.importorskip("harmonypy")
    params = {"batch_key": "sample_id", "use_rep": "X_pca", "seed": 0}
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=30, n_pcs=15, seed=0)
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)
    r = tools.run("integrate_harmony", s, **params)
    if r.status != "success":
        pytest.skip(f"integrate_harmony unavailable: {r.error}")
    tool_Z = np.asarray(s.adata.obsm["X_harmony"])

    out = _run_standalone(tmp_path, "06", "integrate_harmony", params, snap)
    assert np.allclose(np.asarray(out.obsm["X_harmony"]), tool_Z, atol=1e-4)


def test_cluster_sweep_standalone_matches_tool(tmp_path):
    params = {"res_min": 0.1, "res_max": 0.4, "res_step": 0.1, "seed": 0}
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=30, n_pcs=15, seed=0)
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)
    r = tools.run("cluster_sweep", s, **params)
    tool_sweep = [(row["resolution"], row["n_clusters"]) for row in r.summary["sweep"]]

    _run_standalone(tmp_path, "02", "cluster_sweep", params, snap)
    import pandas as pd
    sweep_csv = pd.read_csv(tmp_path / "standalone_data" / "02_cluster_sweep_sweep.csv")
    std_sweep = list(zip(sweep_csv["resolution"].round(4), sweep_csv["n_clusters"]))
    assert std_sweep == [(round(rr, 4), nn) for rr, nn in tool_sweep]


def _two_label_state(tmp_path):
    """A post-cluster session with two per-cell label columns to vote across."""
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=30, n_pcs=15, seed=0)
    tools.run("cluster", s, resolution=0.5, seed=0)
    leid = s.adata.obs["leiden"].astype(str)
    # two annotation columns that mostly agree (so consensus is non-trivial, some ambiguous)
    s.adata.obs["anno_a"] = ("A_" + leid).astype("category")
    flip = np.zeros(s.adata.n_obs, dtype=bool)
    flip[::7] = True                                   # ~1/7 cells disagree
    s.adata.obs["anno_b"] = np.where(flip, "OTHER", "A_" + leid)
    s.adata.obs["anno_b"] = s.adata.obs["anno_b"].astype("category")
    return s


def test_consensus_annotation_standalone_matches_tool(tmp_path):
    params = {"keys": ["anno_a", "anno_b"], "out_key": "celltype_consensus", "min_agreement": 0.5}
    s = _two_label_state(tmp_path)
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)
    tools.run("consensus_annotation", s, **params)
    tool_lab = s.adata.obs["celltype_consensus"].astype(str).to_numpy()

    out = _run_standalone(tmp_path, "03", "consensus_annotation", params, snap)
    assert (out.obs["celltype_consensus"].astype(str).to_numpy() == tool_lab).all()


def test_harmonize_annotations_standalone_matches_tool(tmp_path):
    params = {"keys": ["anno_a", "anno_b"], "out_key": "celltype_harmonized", "method": "auto"}
    s = _two_label_state(tmp_path)
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)
    tools.run("harmonize_annotations", s, **params)
    tool_lab = s.adata.obs["celltype_harmonized"].astype(str).to_numpy()

    out = _run_standalone(tmp_path, "04", "harmonize_annotations", params, snap)
    assert (out.obs["celltype_harmonized"].astype(str).to_numpy() == tool_lab).all()


def test_annotation_review_standalone_matches_tool(tmp_path):
    import json
    params = {"groupby": "leiden", "min_in_group_fraction": 0.1, "max_out_group_fraction": 0.5,
              "min_fold_change": 1.0}      # loose filter so the tiny fixture yields markers
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=30, n_pcs=15, seed=0)
    tools.run("cluster", s, resolution=0.5, seed=0)
    tools.run("markers", s, groupby="leiden", max_genes_ranked=None)
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)
    r = tools.run("annotation_review", s, **params)
    tool_status = r.summary["status_counts"]
    tool_flagged = sorted(map(str, r.summary["flagged_clusters"]))

    _run_standalone(tmp_path, "07", "annotation_review", params, snap)
    review = json.loads((tmp_path / "standalone_data" / "07_annotation_review_review.json").read_text())
    assert review["status_counts"] == tool_status
    assert sorted(map(str, review["flagged_clusters"])) == tool_flagged
    # per-cluster specific-marker counts identical
    tool_json = json.loads(Path(r.summary["review_input"]).read_text())
    std_spec = {p["cluster_id"]: p["n_specific_markers"] for p in review["clusters"]}
    assert {p["cluster_id"]: p["n_specific_markers"] for p in tool_json["clusters"]} == std_spec


def _annotated_state(tmp_path):
    """A post-apply_annotation session (leiden + markers + major_cell_type + marker_sets in uns)."""
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=30, n_pcs=15, seed=0)
    tools.run("cluster", s, resolution=0.5, seed=0)
    tools.run("markers", s, groupby="leiden", max_genes_ranked=None)
    clusters = sorted(s.adata.obs["leiden"].astype(str).unique())
    labels = {c: f"CellType_{c}" for c in clusters}
    # give each label a marker_set from its own top DE genes (so audit's check 2 has input)
    import scanpy as sc
    de = sc.get.rank_genes_groups_df(s.adata, group=None)
    msets = {f"CellType_{c}": [str(g) for g in de[de["group"].astype(str) == c]["names"].head(4)]
             for c in clusters}
    tools.run("apply_annotation", s, groupby="leiden", labels=labels, marker_sets=msets)
    return s, clusters


def test_annotation_audit_standalone_matches_tool(tmp_path):
    import json
    params = {"groupby": "leiden", "label_key": "major_cell_type",
              "min_specificity": 0.0, "max_pct_out": 1.0}    # loose so the tiny fixture yields markers
    s, _ = _annotated_state(tmp_path)
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)
    r = tools.run("annotation_audit", s, **params)
    tool = json.loads(Path(r.summary["audit_input"]).read_text())

    _run_standalone(tmp_path, "08", "annotation_audit", params, snap)
    std = json.loads((tmp_path / "standalone_data" / "08_annotation_audit_audit.json").read_text())
    assert std["status_counts"] == r.summary["status_counts"]
    assert sorted(map(str, std["flagged_clusters"])) == sorted(map(str, r.summary["flagged_clusters"]))
    assert std["n_marker_profile_collisions"] == r.summary["n_marker_profile_collisions"]
    tool_sup = {c["cluster_id"]: c["marker_set_support_frac"] for c in tool["clusters"]}
    std_sup = {c["cluster_id"]: c["marker_set_support_frac"] for c in std["clusters"]}
    assert std_sup == tool_sup


def test_apply_annotation_audit_standalone_matches_tool(tmp_path):
    s, clusters = _annotated_state(tmp_path)
    verdicts = {clusters[0]: {"status": "confirmed"},
                clusters[1]: {"status": "suspect", "review_required": True, "note": "weak"},
                clusters[-1]: {"status": "refuted", "note": "wrong lineage"}}
    params = {"groupby": "leiden", "label_key": "major_cell_type", "verdicts": verdicts,
              "reviewer_model": "claude-opus-4-8"}
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)
    tools.run("apply_annotation_audit", s, **params)
    tool_status = s.adata.obs["annotation_audit_status"].astype(str).to_numpy()
    tool_t4 = s.adata.uns["scpilot_annotation"]["tier4_audit"]

    out = _run_standalone(tmp_path, "09", "apply_annotation_audit", params, snap)
    assert (out.obs["annotation_audit_status"].astype(str).to_numpy() == tool_status).all()
    t4 = out.uns["scpilot_annotation"]["tier4_audit"]
    assert list(map(str, t4["refuted_clusters"])) == list(map(str, tool_t4["refuted_clusters"]))
    assert list(map(str, t4["suspect_clusters"])) == list(map(str, tool_t4["suspect_clusters"]))


def test_apply_annotation_standalone_matches_tool(tmp_path):
    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=30, n_pcs=15, seed=0)
    tools.run("cluster", s, resolution=0.5, seed=0)
    clusters = sorted(s.adata.obs["leiden"].astype(str).unique())
    labels = {c: f"CellType_{c}" for c in clusters}
    labels.pop(clusters[-1])                        # leave one cluster unlabeled → Unassigned path
    params = {"groupby": "leiden", "labels": labels, "key": "major_cell_type",
              "confidence": {clusters[0]: 0.9}, "tissue": "test"}
    snap = tmp_path / "snap.h5ad"
    s.adata.write_h5ad(snap)
    tools.run("apply_annotation", s, **params)
    tool_lab = s.adata.obs["major_cell_type"].astype(str).to_numpy()
    tool_uns = s.adata.uns["scpilot_annotation"]["tier1_llm"]

    out = _run_standalone(tmp_path, "03", "apply_annotation", params, snap)
    assert (out.obs["major_cell_type"].astype(str).to_numpy() == tool_lab).all()
    assert out.uns["scpilot_annotation"]["tier1_llm"]["labels"] == tool_uns["labels"]
    assert "Unassigned" in set(tool_lab)            # the unlabeled-cluster path exercised
