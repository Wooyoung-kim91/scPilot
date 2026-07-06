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


# --------------------------------------------------------------------------- #
# Outputs harness — per-step OutputRecord binds artifacts(+sha) + reasoning + provenance
# --------------------------------------------------------------------------- #
def test_outputs_jsonl_binds_artifacts_reasoning_provenance(tmp_path):
    from typer.testing import CliRunner

    from scpilot.cli import app

    inp = tmp_path / "in.h5ad"
    _tiny_adata().write_h5ad(inp)
    wd = tmp_path / "sess"
    res = CliRunner().invoke(app, ["step", "qc_metrics", str(inp), "-w", str(wd),
                                   "-p", "run_scrublet=false",
                                   "-p", "reasoning=QC threshold review", "--seed", "0"])
    assert res.exit_code == 0, res.output
    recs = [json.loads(l) for l in (wd / "outputs.jsonl").read_text().splitlines()]
    assert recs, "outputs.jsonl should hold one record per step"
    r = recs[-1]
    assert r["tool"] == "qc_metrics"
    assert r["reasoning"] == "QC threshold review"     # WHY bound to the step
    assert r["recipe_hash"] and r["seed"] == 0          # provenance
    # auto-plot artifact captured with an integrity sha256
    assert r["artifacts"], "expected the qc auto-plot to be cataloged"
    assert any((a.get("meta") or {}).get("sha256") for a in r["artifacts"])


def test_report_links_artifacts_to_producing_step(tmp_path):
    from scpilot import tools

    s = Session.create(tmp_path / "sess")
    s._ensure_dirs()
    (s.artifacts_dir / "markers.csv").write_text("gene,score\nA,1.0\n")
    (s.artifacts_dir / "umap.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    res = S.success("markers", artifacts=[
        S.Artifact(path=str(s.artifacts_dir / "markers.csv"), kind="csv"),
        S.Artifact(path=str(s.artifacts_dir / "umap.png"), kind="png"),
    ])
    s.record_run(res, params={"n_genes": 25}, seed=0, reasoning="rank genes per cluster")

    rep = tools.run("report", s)
    assert rep.status == "success"
    rj = json.loads((s.artifacts_dir / "report.json").read_text())
    arts = rj["artifacts"]
    # CSV table is included (not just PNGs) and carries its producing step + reasoning
    csv = next(a for a in arts if a["kind"] == "csv")
    assert csv["tool"] == "markers" and csv["reasoning"] == "rank genes per cluster"
    assert csv["sha256"]                                  # integrity hash recorded
    assert rep.summary["n_tables"] == 1 and rep.summary["n_figures"] == 1


# --------------------------------------------------------------------------- #
# Pipeline notebook — cell-by-cell execution reproduces the recorded result
# --------------------------------------------------------------------------- #
def test_generated_notebook_reproduces_cell_by_cell(tmp_path):
    import subprocess
    import sys

    from scpilot import tools
    from scpilot.repro import set_global_seed

    # two latent groups so clustering finds real (reproducible) structure
    rng = np.random.default_rng(0)
    base = rng.poisson(0.5, (200, 120)).astype("float32")
    base[:100, :30] += rng.poisson(4.0, (100, 30)).astype("float32")
    base[100:, 30:60] += rng.poisson(4.0, (100, 30)).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(base))
    a.var_names = [f"G{i}" for i in range(120)]
    a.layers["counts"] = a.X.copy()
    inp = tmp_path / "in.h5ad"
    a.write_h5ad(inp)

    s = Session.create(tmp_path / "sess", input_path=str(inp))
    s.load_input()
    set_global_seed(0)
    for stage, params in [("preprocess", {"n_top_genes": 60, "n_pcs": 15}),
                          ("cluster", {"resolution": 0.5})]:
        res = tools.run(stage, s, **params)
        assert res.status == "success", res.error
        s.record_run(res, params=params, seed=0)
    orig_n = int(s.adata.obs["leiden"].nunique())

    nb = s.code_dir / "pipeline_notebook.py"
    text = nb.read_text()
    # every step cell re-pins its seed (top + 2 steps) → self-contained cells
    assert text.count("set_global_seed(") >= 3

    # run the generated notebook as a plain script (executes each cell top-to-bottom)
    r = subprocess.run([sys.executable, str(nb)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    nb_sess = Session.open(tmp_path / "sess" / "repro_notebook")
    assert int(nb_sess.adata.obs["leiden"].nunique()) == orig_n   # identical result


# --------------------------------------------------------------------------- #
# QC dynamic params + enforced plots (MAD suggester, before/after, scatter, thresholds,
# metadata UMAPs, HVG auto-batch)
# --------------------------------------------------------------------------- #
def _qc_adata(n_obs=120, n_vars=60):
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (n_obs, n_vars)).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(n_vars - 3)] + ["MT-A", "MT-B", "MT-C"]
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    a.layers["counts"] = a.X.copy()
    a.obs["sample_id"] = rng.choice(["s1", "s2", "s3"], n_obs)
    a.obs["condition"] = rng.choice(["tumor", "normal"], n_obs)
    return a


def test_qc_metrics_emits_mad_suggested_cutoffs(tmp_path):
    from scpilot import tools

    inp = tmp_path / "in.h5ad"
    _qc_adata().write_h5ad(inp)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    s.load_input()
    res = tools.run("qc_metrics", s, run_scrublet=False)
    assert res.status == "success"
    sc = res.summary["suggested_cutoffs"]
    assert {"min_genes", "max_genes", "max_pct_mt"} <= set(sc["global"])    # MAD evidence
    assert sc["global"]["min_genes"] >= 0
    # per-sample cutoffs: summary stays O(1) — global + a min/median/max SPREAD + artifact path;
    # the full per-sample map is in the CSV artifact (token cost fixed regardless of sample count).
    assert {"min", "median", "max"} <= set(sc["per_sample_spread"]["min_genes"])   # batch-aware spread
    assert sc["n_samples"] >= 3
    from pathlib import Path
    assert Path(sc["per_sample_artifact"]).exists()
    assert "per_sample" not in sc                                            # full map NOT inline


def test_qc_metrics_detects_mouse_and_resolves_mixed_lineage(tmp_path):
    # no-hardcoding rule: organism is DETECTED, and human reference symbols (EPCAM/CD3D)
    # are resolved to the data's mouse casing (Epcam/Cd3d) instead of being missed.
    from scpilot import tools

    a = _qc_adata(n_obs=120, n_vars=60)
    mouse = [f"Gene{i}" for i in range(a.n_vars - 5)] + ["Epcam", "Cd3d", "mt-Nd1", "mt-Co1", "Rps2"]
    a.var_names = mouse
    inp = tmp_path / "in.h5ad"
    a.write_h5ad(inp)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    s.load_input()
    res = tools.run("qc_metrics", s, run_scrublet=False,   # mito_prefix auto-detected
                    mixed_lineage_genes=("EPCAM", "CD3D"))  # opt-in; resolved to mouse Epcam/Cd3d
    assert res.status == "success"
    assert res.summary["organism"] == "mouse"
    assert res.summary["mito_prefix"] == "mt-"
    assert res.summary["mixed_lineage_frac"] is not None   # Epcam/Cd3d resolved, flag computed


def test_preprocess_tiny_batch_guard_disables_batch_hvg(tmp_path):
    # a batch far below min_cells_per_batch would make the seurat_v3 per-batch loess
    # singular; the guard falls back to global HVG instead of crashing.
    from scpilot import tools

    a = _qc_adata(n_obs=200, n_vars=80)
    sid = ["big"] * 195 + ["tiny"] * 5            # one 5-cell batch
    a.obs["sample_id"] = sid
    inp = tmp_path / "in.h5ad"
    a.write_h5ad(inp)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    s.load_input()
    res = tools.run("preprocess", s, n_top_genes=40, n_pcs=15, min_cells_per_batch=50)
    assert res.status == "success"
    assert res.summary["hvg_batch_key"] is None       # guard disabled batch-aware HVG
    assert any("min_cells_per_batch" in w or "batch" in w.lower() for w in res.warnings)


def test_qc_autoplots_before_after_and_thresholds(tmp_path):
    from typer.testing import CliRunner

    from scpilot.cli import app

    inp = tmp_path / "in.h5ad"
    _qc_adata().write_h5ad(inp)
    wd = tmp_path / "sess"
    runner = CliRunner()
    r1 = runner.invoke(app, ["step", "qc_metrics", str(inp), "-w", str(wd), "-p", "run_scrublet=false"])
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(app, ["step", "qc_filter", "-w", str(wd), "-p", "min_genes=3", "-p", "max_pct_mt=80"])
    assert r2.exit_code == 0, r2.output
    names = {p.name for p in (wd / "artifacts").glob("*")}
    # before/after as DISTINCT files (the old fixed-name collision is gone) + justification plot
    assert any(n.startswith("qc_violin_pre") for n in names)
    assert any(n.startswith("qc_violin_post") for n in names)
    assert any(n.startswith("qc_scatter_pre") for n in names)
    assert any(n.startswith("qc_scatter_post") for n in names)
    assert any(n.startswith("qc_thresholds") for n in names)


def test_cluster_autoplots_metadata_umaps(tmp_path):
    from typer.testing import CliRunner

    from scpilot.cli import app

    inp = tmp_path / "in.h5ad"
    _qc_adata(n_obs=200, n_vars=80).write_h5ad(inp)
    wd = tmp_path / "sess"
    runner = CliRunner()
    assert runner.invoke(app, ["step", "preprocess", str(inp), "-w", str(wd),
                               "-p", "n_top_genes=40", "-p", "n_pcs=15"]).exit_code == 0
    assert runner.invoke(app, ["step", "cluster", "-w", str(wd), "-p", "resolution=0.5"]).exit_code == 0
    umaps = {p.name for p in (wd / "artifacts").glob("umap*")}
    assert any("leiden" in n for n in umaps)            # leiden UMAP
    assert any("sample_id" in n for n in umaps)         # enforced metadata UMAP
    assert any("condition" in n for n in umaps)


def test_preprocess_auto_detects_hvg_batch_key(tmp_path):
    from scpilot import tools

    inp = tmp_path / "in.h5ad"
    _qc_adata(n_obs=150, n_vars=80).write_h5ad(inp)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    s.load_input()
    # min_cells_per_batch low so the tiny-batch guard doesn't fire on this small fixture
    res = tools.run("preprocess", s, n_top_genes=40, n_pcs=15, min_cells_per_batch=10)
    assert res.status == "success"
    assert res.summary["hvg_batch_key"] == "sample_id"           # auto-detected


def test_plots_scatter_and_qc_thresholds_kinds(tmp_path):
    from scpilot import tools

    inp = tmp_path / "in.h5ad"
    _qc_adata().write_h5ad(inp)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    s.load_input()
    tools.run("qc_metrics", s, run_scrublet=False)
    assert tools.run("plots", s, kind="scatter").artifacts
    assert tools.run("plots", s, kind="qc_thresholds",
                     cutoffs={"min_genes": 3, "max_pct_mt": 50}).artifacts


# --------------------------------------------------------------------------- #
# Annotation Phase A — dynamic resolution sweep + knee
# --------------------------------------------------------------------------- #
def _two_group_adata(n_obs=200, n_vars=120):
    rng = np.random.default_rng(0)
    base = rng.poisson(0.5, (n_obs, n_vars)).astype("float32")
    h = n_obs // 2
    base[:h, :30] += rng.poisson(4.0, (h, 30)).astype("float32")
    base[h:, 30:60] += rng.poisson(4.0, (n_obs - h, 30)).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(base))
    a.var_names = [f"G{i}" for i in range(n_vars)]
    a.layers["counts"] = a.X.copy()
    return a


def test_suggest_resolution_knee():
    from scpilot.recipes import suggest_resolution

    # n_clusters jumps 6→15 between 0.3 and 0.4 → choose 0.3 (just before the jump)
    jump = [(0.1, 5), (0.2, 6), (0.3, 6), (0.4, 15), (0.5, 16)]
    assert suggest_resolution(jump, jump_ratio=1.5)[0] == 0.3
    # no abrupt jump → conservative lowest resolution
    flat = [(0.1, 4), (0.2, 4), (0.3, 5), (0.4, 5), (0.5, 6)]
    assert suggest_resolution(flat, jump_ratio=1.5)[0] == 0.1


def test_cluster_sweep_curve_and_cleanup(tmp_path):
    from scpilot import tools
    from scpilot.repro import set_global_seed

    inp = tmp_path / "in.h5ad"
    _two_group_adata().write_h5ad(inp)
    set_global_seed(0)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    s.load_input()
    assert tools.run("preprocess", s, n_top_genes=60, n_pcs=15).status == "success"
    res = tools.run("cluster_sweep", s, use_rep="X_pca")
    assert res.status == "success"
    assert [round(d["resolution"], 1) for d in res.summary["sweep"]] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert 0.1 <= res.summary["suggested_resolution"] <= 0.5
    # non-mutating: the throwaway sweep keys must not pollute the working AnnData
    assert "_sweep_leiden" not in s.adata.obs.columns
    assert "_sweep_nbr" not in s.adata.uns


def test_plots_resolution_sweep_kind(tmp_path):
    from scpilot import tools

    s = Session.create(tmp_path / "sess")
    s.set_adata(_tiny_adata())                          # adata present but unused by this kind
    res = tools.run("plots", s, kind="resolution_sweep", suggested=0.2,
                    sweep=[{"resolution": 0.1, "n_clusters": 2},
                           {"resolution": 0.2, "n_clusters": 4},
                           {"resolution": 0.3, "n_clusters": 9}])
    assert res.status == "success" and res.artifacts


def test_cluster_sweep_autoplots_resolution_curve(tmp_path):
    from typer.testing import CliRunner

    from scpilot.cli import app

    inp = tmp_path / "in.h5ad"
    _two_group_adata().write_h5ad(inp)
    wd = tmp_path / "sess"
    runner = CliRunner()
    assert runner.invoke(app, ["step", "preprocess", str(inp), "-w", str(wd),
                               "-p", "n_top_genes=60", "-p", "n_pcs=15"]).exit_code == 0
    assert runner.invoke(app, ["step", "cluster_sweep", "-w", str(wd),
                               "-p", "use_rep=X_pca"]).exit_code == 0
    assert {p.name for p in (wd / "artifacts").glob("resolution_sweep*")}


# --------------------------------------------------------------------------- #
# Harness integrity — run_log↔outputs coupling (C-2) + no-overwrite artifacts (P1-2)
# --------------------------------------------------------------------------- #
def test_log_consistency_tracks_run_and_outputs(tmp_path):
    s = Session.create(tmp_path / "sess")
    s.set_adata(_tiny_adata())
    for _ in range(3):
        s.record_run(_result("cluster", n_clusters=4), params={"resolution": 0.3}, seed=0)
    lc = s.log_consistency()
    assert lc["n_runs"] == 3 and lc["n_outputs"] == 3
    assert lc["log_inconsistencies"] == 0 and lc["consistent"] is True
    # the coupling is detectable: every run_log line has a matching outputs line
    n_runlog = len(s.run_log_path.read_text().splitlines())
    n_out = len(s.outputs_path.read_text().splitlines())
    assert n_runlog == n_out == 3


def test_log_consistency_flags_divergence(tmp_path):
    s = Session.create(tmp_path / "sess")
    s.set_adata(_tiny_adata())
    # force the outputs.jsonl append to fail → divergence must be COUNTED + flagged, not silent
    orig = Session._append_jsonl

    def boom(path, record):
        if str(path).endswith("outputs.jsonl"):
            raise OSError("disk full")
        return orig(path, record)

    import scpilot.session as _sess
    s_cls = _sess.Session
    s_cls._append_jsonl = staticmethod(boom)
    try:
        s.record_run(_result("cluster"), params={}, seed=0)
    finally:
        s_cls._append_jsonl = staticmethod(orig)
    lc = s.log_consistency()
    assert lc["log_inconsistencies"] == 1 and lc["consistent"] is False
    assert lc["n_runs"] == 1 and lc["n_outputs"] == 0


def test_artifact_path_no_overwrite_on_rerun(tmp_path):
    s = Session.create(tmp_path / "sess")
    s._ensure_dirs()
    p1 = s.artifact_path("annotation_review.json")
    assert p1.name == "annotation_review.json"          # first run → plain name
    p1.write_text("{}")
    s.manifest.n_runs = 5                                # simulate a later run
    p2 = s.artifact_path("annotation_review.json")
    assert p2 != p1 and p2.name == "annotation_review.05.json"   # versioned, no overwrite
    assert p1.exists()                                  # prior evidence preserved


def test_derive_dotplot_markers_family_contiguous():
    # subtypes of the same family must stay ADJACENT on the y-axis (not scattered by abundance):
    # sizes here would, by pure abundance, split the two Macrophage subtypes with Monocyte between.
    import scanpy as sc

    from scpilot.core.annotate import derive_dotplot_markers

    rng = np.random.default_rng(0)
    labels = {"0": "Macrophage SPP1+", "1": "Monocyte", "2": "Macrophage C1Q+", "3": "DC"}
    sizes = {"0": 50, "1": 40, "2": 30, "3": 20}
    blocks = {"0": range(0, 8), "1": range(8, 16), "2": range(16, 24), "3": range(24, 32)}
    n_vars, parts, grp = 40, [], []
    for cl, n in sizes.items():
        b = rng.poisson(0.3, (n, n_vars)).astype("float32")
        for j in blocks[cl]:
            b[:, j] += rng.poisson(8.0, n).astype("float32")
        parts.append(b)
        grp += [cl] * n
    a = ad.AnnData(sparse.csr_matrix(np.vstack(parts)))
    a.var_names = [f"G{i}" for i in range(n_vars)]
    a.layers["counts"] = a.X.copy()
    a.obs["grp"] = grp
    a.obs["grp"] = a.obs["grp"].astype("category")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.tl.rank_genes_groups(a, "grp", method="wilcoxon", pts=True)

    panels = derive_dotplot_markers(a, cluster_key="grp", label_map=labels)   # no `order` → family-contiguous
    order = list(panels)
    mac = [i for i, ct in enumerate(order) if ct.startswith("Macrophage")]
    assert len(mac) == 2 and mac[1] == mac[0] + 1, order      # Macrophage* block is contiguous


def test_phase_d_fine_facs_primary(tmp_path):
    # fine (Tier-2 subtype): mean_in parity + FACS-style label is the PRIMARY, always-populated name
    import json

    from scpilot import tools
    from scpilot.repro import set_global_seed

    inp = tmp_path / "in.h5ad"
    _two_group_adata(n_obs=220, n_vars=120).write_h5ad(inp)
    set_global_seed(0)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    s.load_input()
    assert tools.run("preprocess", s, n_top_genes=60, n_pcs=15).status == "success"
    assert tools.run("cluster", s, resolution=0.5).status == "success"
    s.adata.obs["major_cell_type"] = ["T cell"] * s.adata.n_obs                   # compartment context
    assert tools.run("markers", s, groupby="leiden").status == "success"

    # packaging/parity test → loosen the marker-quality filter so synthetic markers survive
    rev = tools.run("fine_annotation_review", s, groupby="leiden", min_in_group_fraction=0.0,
                    max_out_group_fraction=1.0, min_fold_change=1.0)
    assert rev.status == "success"
    art = next(a.path for a in rev.artifacts if a.path.endswith("fine_annotation_evidence.json"))
    data = json.loads(open(art).read())
    de0 = data["subclusters"][0]["de_table"]
    assert de0 and "mean_in" in de0[0]                       # (1) expression exposed (parity with broad)

    subs = [str(p["subcluster"]) for p in data["subclusters"]]
    fine_labels = {c: f"T_sub{c}" for c in subs}
    facs_labels = {subs[0]: "CD8+ T cells"}                  # only ONE FACS label given
    ap = tools.run("apply_fine_annotation", s, groupby="leiden",
                   fine_labels=fine_labels, facs_labels=facs_labels)
    assert ap.status == "success"
    facs_col = s.adata.obs["facs_style_label"].astype(str)
    assert (facs_col.str.len() > 0).all()                    # (2) FACS primary: never empty (fine fallback)
    assert "CD8+ T cells" in set(facs_col)                   # the given FACS label applied
    assert "facs_label_map" in ap.summary

    # (3) FACS dotplot — rows = FACS labels (subcluster DE mapped through subcluster->FACS)
    d = tools.run("plots", s, kind="dotplot", groupby="facs_style_label",
                  cluster_key="leiden", label_map=ap.summary["facs_label_map"])
    assert d.status == "success" and d.artifacts


def test_finalize_annotation_consolidates_facs_like(tmp_path):
    from scpilot import tools

    s = Session.create(tmp_path / "sess")
    a = _tiny_adata()
    n = a.n_obs
    # broad for all; facs (subtype) for the first half only; malignancy on a slice
    a.obs["major_cell_type"] = ["Epithelial" if i < n // 2 else "T cell" for i in range(n)]
    a.obs["facs_style_label"] = ["CD8+ T" if i >= n // 2 else "" for i in range(n)]   # only T half has FACS
    a.obs["malignancy"] = ["malignant" if i < n // 4 else "non_malignant" for i in range(n)]
    s.set_adata(a)

    res = tools.run("finalize_annotation", s)
    assert res.status == "success"
    fa = s.adata.obs["final_annotation"].astype(str)
    # most-specific base: T half uses FACS, Epithelial half uses broad; malignant slice prefixed
    assert set(fa[n // 2:]) == {"CD8+ T"}                     # FACS subtype wins where present
    assert fa.iloc[0] == "Malignant Epithelial"               # broad base + malignancy qualifier
    assert fa.iloc[n // 2 - 1] == "Epithelial"                # non-malignant Epithelial, no FACS
    assert res.summary["n_malignant"] == n // 4
    assert res.summary["label_distribution"]


def test_finalize_annotation_no_malignancy(tmp_path):
    from scpilot import tools

    s = Session.create(tmp_path / "sess")
    a = _tiny_adata()
    a.obs["major_cell_type"] = ["B cell"] * a.n_obs           # broad only, no facs/malignancy
    s.set_adata(a)
    res = tools.run("finalize_annotation", s)
    assert res.status == "success"
    assert set(s.adata.obs["final_annotation"].astype(str)) == {"B cell"}   # base only, no prefix
    assert res.summary["n_malignant"] == 0


def test_harmonize_annotations_consensus_fallback(tmp_path):
    # harmonize_annotations must ALWAYS fall back to the embedding-independent majority vote when the
    # cellhint cross-shard path is unavailable — whether cellhint is absent OR installed-but-unwired
    # (the cross-shard path is still a NotImplementedError stub, I-9 deferred). Either way: graceful
    # fallback, harmonized label written, path + cellhint availability reported (never silent).
    from scpilot import tools
    from scpilot.doctor import _findable

    s = Session.create(tmp_path / "sess")
    a = _tiny_adata()
    n = a.n_obs
    base = ["T" if i % 2 == 0 else "B" for i in range(n)]
    a.obs["major_cell_type"] = base
    a.obs["major_cell_type_harmony"] = base                 # agrees
    a.obs["major_cell_type_scvi"] = ["T"] * n               # disagrees on the B cells (minority)
    s.set_adata(a)

    res = tools.run("harmonize_annotations", s,
                    keys=["major_cell_type", "major_cell_type_harmony", "major_cell_type_scvi"])
    assert res.status == "success"
    assert res.summary["method_used"] == "consensus_fallback"   # cellhint path unavailable → fallback
    # report reflects the env truthfully (cellhint 1.0.0 may be installed) rather than a hardcoded flag
    assert res.summary["cellhint_available"] is bool(_findable("cellhint"))
    if res.summary["cellhint_available"]:
        # installed but the cross-shard path is a stub → a warning must explain the degrade (§3: warn, never silent)
        assert any("cellhint" in w.lower() for w in res.warnings)
    out = s.adata.obs["celltype_harmonized"].astype(str).tolist()
    assert out == base                                          # 2/3 majority resolves every cell
    assert res.summary["n_ambiguous"] == 0


def test_phase_b_annotation_evidence(tmp_path):
    # mean_in exposure + marker_sets recording + broad dotplot (recorded + derived paths)
    import json

    from scpilot import tools
    from scpilot.repro import set_global_seed

    inp = tmp_path / "in.h5ad"
    _two_group_adata(n_obs=220, n_vars=120).write_h5ad(inp)
    set_global_seed(0)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    s.load_input()
    assert tools.run("preprocess", s, n_top_genes=60, n_pcs=15).status == "success"
    assert tools.run("cluster", s, resolution=0.5).status == "success"
    assert tools.run("markers", s, groupby="leiden").status == "success"

    # packaging/exposure test → loosen the marker-quality filter so synthetic markers survive
    rev = tools.run("annotation_review", s, groupby="leiden", min_in_group_fraction=0.0,
                    max_out_group_fraction=1.0, min_fold_change=1.0)
    assert rev.status == "success"
    art = next(a.path for a in rev.artifacts if a.path.endswith("annotation_review.json"))
    data = json.loads(open(art).read())
    de0 = data["clusters"][0]["de_table"]
    assert de0 and "mean_in" in de0[0]                       # (1) expression exposed alongside pct

    clusters = [str(c["cluster_id"]) for c in data["clusters"]]
    labels = {c: ("Macrophage SPP1+" if i == 0 else f"Type{i}") for i, c in enumerate(clusters)}
    ms = {labels[clusters[0]]: ["G0", "G1", "G2"]}
    ap = tools.run("apply_annotation", s, groupby="leiden", labels=labels, marker_sets=ms)
    assert ap.status == "success"
    assert ap.summary["marker_sets"] == {labels[clusters[0]]: ["G0", "G1", "G2"]}   # (2) recorded

    # (3) broad dotplot via the recorded marker_sets (rows = major_cell_type)
    d1 = tools.run("plots", s, kind="dotplot", groupby="major_cell_type", marker_groups=ms)
    assert d1.status == "success" and d1.artifacts
    # (4) broad dotplot via the derived path (leiden DE mapped through label_map)
    d2 = tools.run("plots", s, kind="dotplot", groupby="major_cell_type",
                   cluster_key="leiden", label_map=labels)
    assert d2.status == "success" and d2.artifacts


def test_per_step_tutorial_scripts(tmp_path):
    """The pipeline is emitted as numbered step scripts code/NN_<stage>.py (tutorial form, run IN
    ORDER, h5ad-chained). Transpiled stages (scriptgen.EMITTERS, e.g. qc_filter) emit STANDALONE
    plain-scanpy code; not-yet-transpiled stages fall back to a scpilot-backed step that reads/writes
    the SAME standalone_data/NN_<stage>.h5ad chain so converting a tool is a drop-in replacement."""
    import ast
    from pathlib import Path

    import anndata as ad
    import numpy as np
    from scipy import sparse

    from scpilot.session import Session

    a = ad.AnnData(sparse.csr_matrix(np.random.default_rng(0).poisson(2, (20, 6)).astype("float32")))
    a.var_names = [f"G{i}" for i in range(6)]
    a.layers["counts"] = a.X.copy()
    p = tmp_path / "in.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    s._append_jsonl(s.run_log_path, {"tool": "ingest", "status": "success", "seed": 0, "recipe_hash": "h0"})
    s._append_jsonl(s.run_log_path, {"tool": "qc_filter", "status": "success", "seed": 0,
                                     "recipe_hash": "h1", "params": {"min_genes": 500, "max_pct_mt": 15.0}})
    s._append_jsonl(s.run_log_path, {"tool": "report", "status": "success", "seed": 0,
                                     "recipe_hash": "h2"})   # no emitter -> exercises the fallback path

    written = s._write_step_scripts()
    names = [Path(f).name for f in written]
    assert names == ["00_ingest.py", "01_qc_filter.py", "02_report.py"]   # numbered, one file per step
    for f in written:                                              # every script is valid Python
        ast.parse(Path(f).read_text())

    # step 0 (ingest): transpiled -> STANDALONE (reads the dataset profile, no scpilot)
    ingest = Path(written[0]).read_text()
    assert "import scpilot" not in ingest and "tools.run" not in ingest
    assert "yaml.safe_load" in ingest                             # reads the profile YAML directly
    assert 'OUT = _DATA / "00_ingest.h5ad"' in ingest            # produces the chain file

    # step 1 (qc_filter): transpiled -> STANDALONE plain scanpy, reads step 0's h5ad
    qc = Path(written[1]).read_text()
    assert qc.startswith("#!/usr/bin/env python")
    assert "import scanpy as sc" in qc                            # standalone scientific stack
    assert "import scpilot" not in qc and "tools.run" not in qc   # genuinely scpilot-free
    assert "def " not in qc                                       # DIRECT code, no function wrapper
    assert ">= 500" in qc and "<= 15.0" in qc                     # actual param values, inline in the ops
    assert "IN ORDER" in qc                                       # run-order guidance
    assert 'IN  = _DATA / "00_ingest.h5ad"' in qc                 # chained to the previous step

    # step 2 (report): no emitter yet -> scpilot-backed FALLBACK, but same h5ad chain
    rep = Path(written[2]).read_text()
    assert 'tools.run("report"' in rep                            # fallback still uses the tool
    assert 'IN  = _DATA / "01_qc_filter.h5ad"' in rep             # chained to the previous step
    assert 'OUT = _DATA / "02_report.h5ad"' in rep
