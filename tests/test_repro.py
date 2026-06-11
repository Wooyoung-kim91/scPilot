"""Unit tests for the reproducibility harness — scpilot plan A7."""

import numpy as np

from scpilot import repro, schemas as S
from scpilot.session import Session


def test_set_global_seed_is_deterministic():
    repro.set_global_seed(0)
    a = np.random.rand(5)
    repro.set_global_seed(0)
    b = np.random.rand(5)
    assert np.allclose(a, b)
    rec = repro.set_global_seed(7)
    assert rec["seed"] == 7 and rec["numpy"] == 7


def test_recipe_hash_stable_and_param_sensitive():
    h1 = repro.recipe_hash(params={"resolution": 0.5}, lib_versions={"scanpy": "1.11.5"})
    h2 = repro.recipe_hash(params={"resolution": 0.5}, lib_versions={"scanpy": "1.11.5"})
    h3 = repro.recipe_hash(params={"resolution": 0.8}, lib_versions={"scanpy": "1.11.5"})
    assert h1 == h2          # deterministic
    assert h1 != h3          # sensitive to params


def test_dataset_fingerprint():
    import anndata as ad
    from scipy import sparse
    X = sparse.csr_matrix(np.random.default_rng(0).poisson(1.0, (10, 8)).astype("float32"))
    a = ad.AnnData(X)
    fp = repro.dataset_fingerprint(a)
    assert fp["shape"] == [10, 8]
    assert "var_names_sha" in fp and "obs_names_sha" in fp


def test_compare_summaries_grade_tolerance():
    ref = {"n_clusters": 12, "silhouette": 0.50, "n_obs": 1000}
    # grade B tolerates small drift: cluster ±1, numbers within 5%
    new_b = {"n_clusters": 13, "silhouette": 0.51, "n_obs": 1000}
    assert repro.compare_summaries(ref, new_b, grade="B")["match"] is True
    # grade A is strict: same drift now fails
    res_a = repro.compare_summaries(ref, new_b, grade="A")
    assert res_a["match"] is False
    # large drift fails even at B
    new_big = {"n_clusters": 20, "silhouette": 0.9, "n_obs": 1000}
    assert repro.compare_summaries(ref, new_big, grade="B")["match"] is False


def test_decision_validation():
    good = S.DecisionEvent(
        decision_type="integration_method", choice="harmony",
        candidates=["harmony", "scvi"], rationale="best bio-conservation", confidence=0.8,
    ).to_dict()
    assert S.validate_decision(good) == []
    bad = {"choice": "harmony"}  # missing decision_type/candidates/rationale
    problems = S.validate_decision(bad)
    assert any("decision_type" in p for p in problems)
    # out-of-range confidence
    assert S.validate_decision({**good, "confidence": 1.5})


def test_replay_dry_run(tmp_path):
    s = Session.create(tmp_path / "sess")
    s.log_run(S.RunLogRecord(tool="cluster", status="success",
                             params={"resolution": 0.5}, determinism_grade="B").to_dict())
    s.log_decision(S.DecisionEvent(decision_type="clustering_resolution", choice=0.5,
                                   candidates=[0.3, 0.5, 0.8], rationale="elbow").to_dict())
    report = repro.replay_session(str(tmp_path / "sess"))
    assert report["mode"] == "dry-run"
    assert report["n_runs"] == 1
    assert report["n_decisions"] == 1
    assert report["steps"][0]["tool"] == "cluster"


def test_replay_with_executor_diffs(tmp_path):
    s = Session.create(tmp_path / "sess")
    # record a run WITH a summary, then replay with an executor that reproduces it (±drift)
    rec = {"tool": "cluster", "status": "success", "determinism_grade": "B",
           "summary": {"n_clusters": 12, "n_obs": 1000}}
    s.log_run(rec)

    def executor(record):
        return {"n_clusters": 13, "n_obs": 1000}  # +1 cluster, tolerated at grade B

    report = repro.replay_session(str(tmp_path / "sess"), executor=executor)
    assert report["mode"] == "executed"
    assert report["all_match"] is True
    assert report["steps"][0]["diff"]["match"] is True


def test_replay_registry_executor_end_to_end(tmp_path):
    """Re-run a real tool chain through the registry executor on a fresh session and
    confirm every summary matches the original within its determinism grade."""
    import anndata as ad
    from scipy import sparse
    from scpilot import tools

    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (300, 80)).astype("float32")
    X[:150, :20] += rng.poisson(5.0, (150, 20)).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(80)]
    a.layers["counts"] = a.X.copy()
    p = tmp_path / "in.h5ad"; a.write_h5ad(p)

    chain = [("preprocess", {"n_top_genes": 40, "n_pcs": 15}), ("cluster", {"resolution": 0.5})]

    repro.set_global_seed(0)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    for tool, params in chain:
        res = tools.run(tool, s, **params)
        assert res.status == "success", res.error
        s.log_run(S.RunLogRecord(tool=tool, status="success", params=params, summary=res.summary,
                                 determinism_grade=res.determinism_grade or "B",
                                 output_checkpoint=res.checkpoint).to_dict())

    repro.set_global_seed(0)
    replay_sess = Session.create(tmp_path / "replay", input_path=str(p))
    report = repro.replay_session(str(tmp_path / "sess"),
                                  executor=tools.make_replay_executor(replay_sess))
    assert report["mode"] == "executed"
    assert report["all_match"] is True, report["steps"]
    assert [st["tool"] for st in report["steps"]] == ["preprocess", "cluster"]
