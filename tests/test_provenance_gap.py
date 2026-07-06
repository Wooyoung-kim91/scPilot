"""I-14 (safe slice) — log_consistency detects the checkpoint-vs-run provenance bypass."""

from scpilot.session import Session


def test_normal_session_no_bypass(tmp_path):
    s = Session.create(str(tmp_path / "s"), exist_ok=True)
    s.manifest.n_runs = 5
    s.manifest.n_outputs = 5
    s.manifest.checkpoints = [{"id": str(i)} for i in range(5)]     # each mutating run → 1 checkpoint
    lc = s.log_consistency()
    assert lc["consistent"] is True
    assert lc["checkpoint_bypass_suspected"] is False
    assert lc["n_checkpoints"] == 5 and lc["checkpoint_run_gap"] == 0


def test_detects_ad_hoc_checkpoint_bypass(tmp_path):
    # the observed pancreas case: 25 checkpoints written by ad-hoc scripts, only 2 logged runs
    s = Session.create(str(tmp_path / "s"), exist_ok=True)
    s.manifest.checkpoints = [{"id": str(i)} for i in range(25)]
    s.manifest.n_runs = 2
    s.manifest.n_outputs = 2
    lc = s.log_consistency()
    assert lc["checkpoint_bypass_suspected"] is True
    assert lc["checkpoint_run_gap"] == 23


def test_non_mutating_runs_do_not_falsely_flag(tmp_path):
    # non-mutating tools (annotation_review/benchmark) add runs without checkpoints → negative gap, no flag
    s = Session.create(str(tmp_path / "s"), exist_ok=True)
    s.manifest.checkpoints = [{"id": str(i)} for i in range(3)]
    s.manifest.n_runs = 8
    s.manifest.n_outputs = 8
    lc = s.log_consistency()
    assert lc["checkpoint_bypass_suspected"] is False
    assert lc["checkpoint_run_gap"] == -5
