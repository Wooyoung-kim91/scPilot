"""Unit test for the export_final tool (publication-ready slim export).

Builds a MINIMAL session (no pipeline run): a tiny AnnData carrying a final_annotation
column with QC-artifact cells to drop, a best-embedding reduction in obsm, and the
benchmark-recorded uns['scpilot']['best_embedding']. Asserts the tool drops the artifact
cells, keeps only the chosen reduction, writes the standalone .h5ad, and reports it.
"""

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot import tools
from scpilot.session import Session


def _session(tmp_path):
    rng = np.random.default_rng(0)
    n = 12
    X = sparse.csr_matrix(rng.poisson(1.0, (n, 5)).astype("float32"))
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(5)]
    a.layers["counts"] = a.X.copy()
    # final_annotation: 3 QC-artifact cells (dropped) + 9 real cells across 2 labels (kept)
    labels = (["Low_quality"] * 2 + ["Doublet"] * 1
              + ["T cell"] * 5 + ["B cell"] * 4)
    a.obs["final_annotation"] = labels
    # best-embedding reduction (a benchmark RESULT) + its recorded choice
    a.obsm["X_harmony"] = rng.standard_normal((n, 4)).astype("float32")
    a.obsm["X_pca"] = rng.standard_normal((n, 4)).astype("float32")  # alternate, must be dropped
    a.uns["scpilot"] = {"best_embedding": "X_harmony"}
    p = tmp_path / "in.h5ad"
    a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p))
    s.load_input()
    return s


def test_export_final_drops_artifacts_and_writes_file(tmp_path):
    s = _session(tmp_path)
    r = tools.run("export_final", s)
    assert r.status == "success", r.error
    su = r.summary

    # QC-artifact cells dropped; malignant/real cells kept (evidence: label_key value_counts)
    assert su["n_cells_before"] == 12
    assert su["n_cells_after"] == 9
    assert su["n_removed"] == 3
    assert su["removed_by_label"] == {"Low_quality": 2, "Doublet": 1}
    assert su["n_final_labels"] == 2

    # only the benchmark-chosen reduction is kept (alternate X_pca dropped)
    assert su["kept_reduction"] == "X_harmony"
    assert "X_harmony" in su["obsm_kept"]
    assert "X_pca" not in su["obsm_kept"]

    # a NEW standalone .h5ad file was actually written with the slimmed cell count
    out_path = su["out_path"]
    assert out_path.endswith("final_clean.h5ad")
    clean = ad.read_h5ad(out_path)
    assert clean.n_obs == 9
    assert "X_harmony" in clean.obsm
    assert "X_pca" not in clean.obsm
    r.to_dict()


def test_export_final_requires_best_embedding(tmp_path):
    s = _session(tmp_path)
    # remove the recorded best embedding and give no explicit keep_reduction -> no guessing
    s.adata.uns["scpilot"].pop("best_embedding")
    r = tools.run("export_final", s)
    assert r.status == "error"
    assert r.error_code == "invalid_state"
    assert r.suggested_next_tools == ["benchmark"]
