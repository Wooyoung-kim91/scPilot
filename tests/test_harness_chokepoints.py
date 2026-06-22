"""Phase-1 harness chokepoints — unified run-log helper + checkpoint-boundary invariants.

Covers the strengthening landed for plan A1/A2/B1/C1:
- ``Session.record_run`` always fills seed + recipe_hash + lib_versions (no per-driver drift).
- ``Session.checkpoint`` enforces the AnnData invariants at the single write boundary.
- the real CLI ``step`` driver routes through the shared helper and records all three fields.
"""

import json

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from scpilot import repro
from scpilot import schemas as S
from scpilot.session import Session


def _tiny_adata(n_obs=60, n_vars=40):
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(1.0, size=(n_obs, n_vars)).astype("float32"))
    a = ad.AnnData(X)
    a.obs_names = [f"cell{i}" for i in range(n_obs)]
    a.var_names = [f"G{i}" for i in range(n_vars)]
    a.layers["counts"] = a.X.copy()
    return a


def _result(tool="cluster", **summary):
    return S.success(tool, summary=summary or {"n_clusters": 5},
                     determinism_grade="B", params={}, duration_s=0.1)


def _records(s):
    return [json.loads(l) for l in s.run_log_path.read_text().splitlines()]


# --------------------------------------------------------------------------- #
# C1 + A1 + A2: the unified record_run fills the fields that used to diverge
# --------------------------------------------------------------------------- #
def test_record_run_populates_seed_recipe_lib(tmp_path):
    s = Session.create(tmp_path / "sess")
    s.set_adata(_tiny_adata())
    s.record_run(_result("cluster", n_clusters=5), params={"resolution": 0.5}, seed=0)
    rec = _records(s)[0]
    assert rec["seed"] == 0
    assert rec["recipe_hash"]                       # A2: populated, not None
    assert rec["lib_versions"]                      # A2: real env versions
    assert rec["tool"] == "cluster" and rec["params"] == {"resolution": 0.5}


def test_recipe_hash_is_deterministic_and_param_sensitive(tmp_path):
    """Same data+params → same recipe_hash; different params → different (drift signal)."""
    s = Session.create(tmp_path / "sess")
    s.set_adata(_tiny_adata())
    s.record_run(_result(), params={"resolution": 0.5}, seed=0)
    s.record_run(_result(), params={"resolution": 0.5}, seed=0)
    s.record_run(_result(), params={"resolution": 0.8}, seed=0)
    recs = _records(s)
    assert recs[0]["recipe_hash"] == recs[1]["recipe_hash"]
    assert recs[0]["recipe_hash"] != recs[2]["recipe_hash"]


# --------------------------------------------------------------------------- #
# B1: invariants enforced at the checkpoint write boundary
# --------------------------------------------------------------------------- #
def test_checkpoint_rejects_counts_value_drift(tmp_path):
    inp = tmp_path / "in.h5ad"
    _tiny_adata().write_h5ad(inp)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    a = s.load_input()                          # establishes counts_fingerprint
    s.checkpoint("qc_metrics")                  # clean: genes/counts preserved → OK
    a.layers["counts"].data[0] += 99.0          # corrupt counts in place (same shape/nnz)
    with pytest.raises(AssertionError):
        s.checkpoint("preprocess")              # rejected BEFORE the bad h5ad is written


def test_checkpoint_rejects_gene_count_change(tmp_path):
    inp = tmp_path / "in.h5ad"
    _tiny_adata().write_h5ad(inp)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    a = s.load_input()
    s.set_adata(a[:, :30].copy())               # drop genes — must never happen
    with pytest.raises(AssertionError):
        s.checkpoint("preprocess")


def test_checkpoint_establish_phase_not_tripped(tmp_path):
    """A checkpoint that first CREATES counts (no prior fingerprint) is not blocked."""
    s = Session.create(tmp_path / "sess")
    a = _tiny_adata()
    del a.layers["counts"]                      # pre-counts working state
    s.set_adata(a)
    assert s.manifest.counts_fingerprint is None
    cp = s.checkpoint("ingest", enforce_invariants=True)   # require_counts defaults False here
    assert cp.id.endswith("ingest")


def test_checkpoint_escape_hatch(tmp_path):
    """enforce_invariants=False bypasses the boundary check (rare escape hatch)."""
    inp = tmp_path / "in.h5ad"
    _tiny_adata().write_h5ad(inp)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    a = s.load_input()
    a.layers["counts"].data[0] += 99.0
    cp = s.checkpoint("preprocess", enforce_invariants=False)   # no raise
    assert cp.id.endswith("preprocess")


# --------------------------------------------------------------------------- #
# C1 driver parity: the real CLI `step` path routes through record_run
# --------------------------------------------------------------------------- #
def test_cli_step_records_seed_recipe_lib(tmp_path):
    from typer.testing import CliRunner

    from scpilot.cli import app

    inp = tmp_path / "in.h5ad"
    _tiny_adata().write_h5ad(inp)
    wd = tmp_path / "sess"
    res = CliRunner().invoke(app, ["step", "qc_metrics", str(inp), "-w", str(wd),
                                   "-p", "run_scrublet=false", "--seed", "0"])
    assert res.exit_code == 0, res.output
    rec = _records(Session.open(wd))[-1]
    assert rec["tool"] == "qc_metrics"
    assert rec["seed"] == 0
    assert rec["recipe_hash"]
    assert rec["lib_versions"]


# --------------------------------------------------------------------------- #
# D1: capability gate — missing optional deps become a recoverable error
# --------------------------------------------------------------------------- #
def test_check_capability_no_requirement_is_ok():
    from scpilot import doctor

    assert doctor.check_capability("preprocess") == (True, [])   # unlisted tool → ungated


def test_require_capability_present_and_absent(monkeypatch):
    from scpilot import doctor, tools

    # present: scVI + torch are env deps on this host → gate passes (None)
    assert tools.require_capability("integrate_scvi") is None
    # absent: fabricate a tool needing a nonexistent module
    monkeypatch.setitem(doctor.CAPABILITY_REQUIRES, "_fake_tool", ["totally_missing_pkg_xyz"])
    err = tools.require_capability("_fake_tool")
    assert err is not None
    assert err.status == "error"
    assert err.error_code == "capability_unavailable"
    assert err.recoverable is True
    assert "totally_missing_pkg_xyz" in err.error


# --------------------------------------------------------------------------- #
# E1: replay surfaces forced LLM structured outputs it does NOT re-derive
# --------------------------------------------------------------------------- #
def test_replay_surfaces_skipped_structured_decisions(tmp_path):
    s = Session.create(tmp_path / "sess")
    s.log_run(S.RunLogRecord(tool="apply_annotation", status="success",
                             params={"labels": {"0": "T"}}, summary={}).to_dict())
    s.log_decision(S.DecisionEvent(decision_type="annotation_strategy",
                                   choice={"0": "T cell"}, candidates=[{"0": "T cell"}],
                                   rationale="emit").to_dict())
    s.log_decision(S.DecisionEvent(decision_type="de_design", choice={"method": "pseudobulk"},
                                   candidates=[{}], rationale="emit").to_dict())

    report = repro.replay_session(str(tmp_path / "sess"), executor=lambda rec: {})
    info = report["structured_decisions_not_reexecuted"]
    assert info["count"] == 2
    assert set(info["types"]) == {"annotation_strategy", "de_design"}
