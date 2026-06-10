"""Unit tests for the on-disk Session — scpilot plan A3 (single out_dir)."""

import json

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from scpilot.session import Session, counts_fingerprint, UNS_KEY


def _tiny_adata(n_obs=60, n_vars=40):
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(1.0, size=(n_obs, n_vars)).astype("float32"))
    a = ad.AnnData(X)
    a.obs_names = [f"cell{i}" for i in range(n_obs)]
    a.var_names = [f"G{i}" for i in range(n_vars)]
    a.layers["counts"] = a.X.copy()
    return a


def test_create_open_roundtrip(tmp_path):
    s = Session.create(tmp_path / "sess")
    assert (tmp_path / "sess" / "session.json").exists()
    sid = s.manifest.session_id
    reopened = Session.open(tmp_path / "sess")
    assert reopened.manifest.session_id == sid
    # create on an existing dir returns the same session (exist_ok)
    again = Session.create(tmp_path / "sess")
    assert again.manifest.session_id == sid


def test_load_input_and_checkpoint(tmp_path):
    inp = tmp_path / "input.h5ad"
    _tiny_adata().write_h5ad(inp)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    a = s.load_input()
    assert a.n_obs == 60
    assert s.manifest.counts_fingerprint is not None

    # mutate X (normalize) but keep counts; checkpoint with x_state
    import scanpy as sc
    sc.pp.normalize_total(a, target_sum=1e4)
    cp = s.checkpoint("normalize", x_state="normalized")
    assert cp.x_state == "normalized"
    assert s.manifest.x_state == "normalized"
    assert len(s.manifest.checkpoints) == 1
    from pathlib import Path
    assert Path(cp.path).exists()

    # provenance block was stamped into uns
    assert UNS_KEY in a.uns
    assert a.uns[UNS_KEY]["stage"] == "normalize"
    assert a.uns[UNS_KEY]["x_state"] == "normalized"


def test_reopen_loads_latest_checkpoint(tmp_path):
    inp = tmp_path / "input.h5ad"
    _tiny_adata().write_h5ad(inp)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    s.load_input()
    s.checkpoint("input")
    # fresh Session object, no in-memory adata -> lazily loads latest checkpoint
    s2 = Session.open(tmp_path / "sess")
    a2 = s2.adata
    assert a2.n_obs == 60
    assert "counts" in a2.layers


def test_run_and_decision_logs_append(tmp_path):
    s = Session.create(tmp_path / "sess")
    s.log_run({"tool": "cluster", "status": "success", "params": {"resolution": 0.5}})
    s.log_run({"tool": "markers", "status": "success"})
    s.log_decision({"decision_type": "integration_method", "choice": "harmony",
                    "candidates": ["harmony", "scvi"], "rationale": "best scib bio-conservation",
                    "confidence": 0.8})
    runs = [json.loads(l) for l in s.run_log_path.read_text().splitlines()]
    assert len(runs) == 2
    assert runs[0]["tool"] == "cluster" and "ts" in runs[0]
    assert s.manifest.n_runs == 2
    decs = [json.loads(l) for l in s.decisions_path.read_text().splitlines()]
    assert decs[0]["choice"] == "harmony"


def test_counts_fingerprint_and_invariant(tmp_path):
    a = _tiny_adata()
    fp = counts_fingerprint(a)
    assert fp["shape"] == [60, 40]
    assert fp["nnz"] >= 0
    s = Session.create(tmp_path / "sess")
    s.set_adata(a)
    s.manifest.counts_fingerprint = fp
    # same counts -> ok
    s.assert_invariants(a)
    # missing counts layer -> raises
    b = a.copy()
    del b.layers["counts"]
    with pytest.raises(AssertionError):
        s.assert_invariants(b)


def test_checkpoint_writes_repro_and_source_snapshot(tmp_path):
    inp = tmp_path / "input.h5ad"
    _tiny_adata().write_h5ad(inp)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    s.load_input()
    cp = s.checkpoint("preprocess", params={"n_pcs": 15, "seed": 0})
    # per-step repro script + a pinned source snapshot exist
    assert cp.repro and Path(cp.repro).exists()
    code = tmp_path / "sess" / "code"
    assert (code / "00_preprocess.repro.py").exists()
    snaps = [p for p in code.iterdir() if p.name.startswith("scpilot-")]
    assert snaps and (snaps[0] / "scpilot" / "session.py").exists()  # full package snapshotted
    text = Path(cp.repro).read_text()
    assert 'TOOL = "preprocess"' in text and "INPUT_CHECKPOINT" in text and "n_pcs" in text


def test_invariant_catches_counts_value_drift(tmp_path):
    # Codex review 2.1: value drift that preserves shape/nnz must be caught via content hash
    a = _tiny_adata()
    s = Session.create(tmp_path / "sess")
    s.set_adata(a)
    s.manifest.counts_fingerprint = counts_fingerprint(a)
    s.assert_invariants(a)                       # unchanged -> ok
    # mutate a nonzero count value in place (same shape, same nnz)
    a.layers["counts"].data[0] += 7.0
    with pytest.raises(AssertionError):
        s.assert_invariants(a)
