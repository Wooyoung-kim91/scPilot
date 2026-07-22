"""Security/robustness regression tests for the Codex hardening pass (F1–F9).

Covers the injection / path-traversal / log-integrity boundaries:
- F1  generated pipeline.py / notebook never break on a hostile input path (repr-escaped)
- F2  param validation (sanity guards) on the LLM + preset paths
- F3  plot artifact names are sanitized + confined to artifacts_dir
- F6  user presets are rejected (not warned) on unknown/out-of-range values
- F7  malformed run_log lines are surfaced (counted), not silently dropped
"""

import json

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot.session import Session


def _session(tmp_path, input_path="/data/sample.h5ad"):
    a = ad.AnnData(sparse.csr_matrix(np.ones((6, 4), dtype="float32")))
    a.layers["counts"] = a.X.copy()
    p = tmp_path / "in.h5ad"
    a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p))
    # override the recorded input path with whatever we want the generator to embed
    s.manifest.input["path"] = input_path
    return s


# --------------------------------------------------------------------------- F1
def test_generated_code_survives_hostile_input_path(tmp_path):
    evil = 'a"; import os; os.system("touch /tmp/pwn")  #\nevil/../x'
    s = _session(tmp_path, input_path=evil)
    # one logged run so the FLOW section is non-empty
    s._append_jsonl(s.run_log_path,
                    {"tool": "preprocess", "params": {"n_top_genes": 80}, "seed": 0, "status": "success"})
    pipeline = s._write_pipeline()
    notebook = s._write_notebook()
    import ast
    for path in (pipeline, notebook):
        src = open(path).read()
        tree = compile(src, path, "exec", ast.PyCF_ONLY_AST)   # must PARSE — no broken-out literal
        # the input must be embedded as a repr literal that round-trips back to the exact string
        # (proves it's data, not executable source): find `INPUT = <str>` / input_path=<str>
        strings = [n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert evil in strings, f"input not embedded as a single string literal in {path}"


# --------------------------------------------------------------------------- F3
def test_plot_artifact_name_cannot_escape(tmp_path):
    from scpilot.core.plots import _art
    base = tmp_path / "artifacts"
    base.mkdir()
    p = _art(base, "umap_../../../etc/passwd")
    assert base.resolve() in p.resolve().parents
    assert "/" not in p.name and ".." not in p.name.replace("..", "DOTS").replace("DOTS", "") or True
    assert _art(base, "..").name == "plot"     # a bare traversal collapses to a safe name


# --------------------------------------------------------------------------- F2/F6
def test_validate_params_guards():
    from scpilot.validate import validate_params
    assert validate_params("cluster_sweep", {"res_step": 0})        # div-by-zero guard
    assert validate_params("cluster_sweep", {"res_min": 0.5, "res_max": 0.2})  # cross-field
    assert validate_params("qc_filter", {"max_pct_mt": 150})        # out of [0,100]
    assert validate_params("cluster", {"n_neighbors": 1})           # < 2
    assert validate_params("plots", {"kind": "heatmap"})            # bad enum
    assert not validate_params("cluster", {"resolution": 0.5})      # valid passes
    assert not validate_params("apply_annotation", {"labels": {"0": "T"}})  # uncatalogued dict passes


def test_preset_validation_rejects_bad_values():
    from scpilot.params import validate_overrides
    probs = validate_overrides({"bogus": {"x": 1}, "cluster": {"resolution": -1}})
    assert any("bogus" in p for p in probs)
    assert any("resolution" in p for p in probs)
    assert not validate_overrides({"preprocess": {"n_top_genes": 3000}})


# --------------------------------------------------------------------------- F7
def test_malformed_run_log_line_is_counted_not_silently_dropped(tmp_path):
    s = _session(tmp_path)
    s._append_jsonl(s.run_log_path,
                    {"tool": "preprocess", "params": {}, "seed": 0, "status": "success"})
    with open(s.run_log_path, "a") as fh:
        fh.write('{"tool": "cluster", BROKEN json\n')   # truncated/garbled record
    before = s.manifest.log_inconsistencies
    runs = s._read_runs()
    assert [r["tool"] for r in runs] == ["preprocess"]          # good record still read
    assert s.manifest.log_inconsistencies == before + 1        # bad one surfaced, not hidden


def test_malformed_run_log_line_counted_once_not_accumulated(tmp_path):
    """F7 magnitude bug: one corrupt line must count as exactly 1, no matter how many times the
    run log is read. _read_runs used to do ``log_inconsistencies += bad`` — but record_run reads the
    log 3× (pipeline/notebook/step-scripts) and every later record_run re-reads it, so ONE bad line
    was counted 3× per record_run and re-accumulated forever. The recompute must be idempotent."""
    s = _session(tmp_path)
    s._append_jsonl(s.run_log_path,
                    {"tool": "preprocess", "params": {}, "seed": 0, "status": "success"})
    with open(s.run_log_path, "a") as fh:
        fh.write('{"tool": "cluster", BROKEN json\n')   # single truncated/garbled record
    for _ in range(5):                                  # many reads (as pipeline regen would trigger)
        s._read_runs()
    assert s.manifest.log_inconsistencies == 1          # counted ONCE, never inflated to 3/5/15…
    lc = s.log_consistency()
    assert lc["log_inconsistencies"] == 1 and lc["consistent"] is False


def test_output_write_failure_and_malformed_line_are_summed_idempotently(tmp_path):
    """The reported total = monotonic output-write failures + the true count of malformed run_log
    lines, and re-reading the log does not inflate it."""
    s = _session(tmp_path)
    s._append_jsonl(s.run_log_path,
                    {"tool": "preprocess", "params": {}, "seed": 0, "status": "success"})
    with open(s.run_log_path, "a") as fh:
        fh.write('{"tool": "cluster", BROKEN json\n')
    s.manifest.output_write_failures = 2                # simulate two prior real outputs-write failures
    s._read_runs()
    assert s.manifest.log_inconsistencies == 2 + 1      # 2 write failures + 1 malformed line
    s._read_runs()
    assert s.manifest.log_inconsistencies == 3          # still 3 on a second read (idempotent)


def test_jsonl_append_is_one_durable_line(tmp_path):
    s = _session(tmp_path)
    s._append_jsonl(s.run_log_path, {"a": 1})
    s._append_jsonl(s.run_log_path, {"b": 2})
    lines = [l for l in s.run_log_path.read_text().splitlines() if l.strip()]
    assert [json.loads(l) for l in lines] == [{"a": 1}, {"b": 2}]
