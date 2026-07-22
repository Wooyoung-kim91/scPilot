"""Unit tests for the drop_noisy_cells tool (explicit post-Tier-1 QC-artifact cleanup).

Builds a MINIMAL session (no pipeline run): a tiny AnnData carrying an annotation column with
QC-artifact-labeled cells to drop and malignant + normal biological cells to KEEP. Asserts the
tool subsets the object (mutating + checkpoints), reports the summary, keeps malignant/normal
cells, errors on an absent label_key, and is a no-op success when there is nothing to drop.
"""

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot import tools
from scpilot.session import Session


def _session(tmp_path, labels, key="final_annotation", major=None):
    rng = np.random.default_rng(0)
    n = len(labels)
    X = sparse.csr_matrix(rng.poisson(1.0, (n, 5)).astype("float32"))
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(5)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.layers["counts"] = a.X.copy()
    a.obs[key] = labels
    if major is not None:
        a.obs["major_cell_type"] = major
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "in.h5ad"
    a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p))
    s.load_input()
    return s


def test_drops_qc_artifacts_keeps_malignant_and_normal(tmp_path):
    # 2 Low_quality + 1 Doublet + 1 composite 'Malignant Low_quality' are QC artifacts (dropped);
    # 3 Malignant + 4 T cell are real biology (KEPT — malignant is not dropped for being malignant).
    labels = (["Low_quality"] * 2 + ["Doublet"] * 1 + ["Malignant Low_quality"] * 1
              + ["Malignant"] * 3 + ["T cell"] * 4)
    s = _session(tmp_path, labels)
    r = tools.run("drop_noisy_cells", s)
    assert r.status == "success", r.error
    su = r.summary

    assert su["label_key"] == "final_annotation"
    assert su["n_before"] == 11
    assert su["n_after"] == 7
    assert su["n_dropped"] == 4
    assert su["removed_by_label"] == {"Low_quality": 2, "Doublet": 1, "Malignant Low_quality": 1}

    # malignant + normal biology survived; no QC-artifact label remains in the object
    remaining = s.adata.obs["final_annotation"].astype(str)
    assert (remaining == "Malignant").sum() == 3
    assert (remaining == "T cell").sum() == 4
    assert not remaining.isin(["Low_quality", "Doublet", "Malignant Low_quality"]).any()

    # mutating step: a checkpoint was written
    assert r.checkpoint is not None
    assert (tmp_path / "sess" / "checkpoints").exists()
    r.to_dict()


def test_absent_label_key_errors(tmp_path):
    s = _session(tmp_path, ["T cell"] * 5)
    r = tools.run("drop_noisy_cells", s, label_key="does_not_exist")
    assert r.status == "error"
    assert r.error_code == "invalid_state"
    assert "apply_annotation" in r.suggested_next_tools


def test_nothing_to_drop_is_success_zero(tmp_path):
    s = _session(tmp_path, ["Malignant"] * 3 + ["T cell"] * 4)
    n_before = s.adata.n_obs
    r = tools.run("drop_noisy_cells", s)
    assert r.status == "success"
    assert r.summary["n_dropped"] == 0
    assert r.summary["n_after"] == n_before
    assert s.adata.n_obs == n_before          # object untouched
    assert any("nothing to" in w for w in r.warnings)


def test_default_label_key_falls_back_to_major_cell_type(tmp_path):
    # no final_annotation column -> the default label_key resolves to major_cell_type
    s = _session(tmp_path, ["Low_quality"] * 2 + ["B cell"] * 3,
                 key="major_cell_type")
    r = tools.run("drop_noisy_cells", s)
    assert r.status == "success"
    assert r.summary["label_key"] == "major_cell_type"
    assert r.summary["n_dropped"] == 2
    assert r.summary["n_after"] == 3
