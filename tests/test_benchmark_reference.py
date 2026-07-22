"""Unit tests for the reference-label self-benchmark — benchmark_reference (Improvement ③, Part A).

Read-only: scores a PREDICTED annotation column against a TRUSTED REFERENCE obs column on the SAME
cells. Label-space HONESTY: label_map reconciliation, case-normalized exact match, the unmatched
fraction reported, and BOTH a strict (unmatched=wrong) and a matched-only score. No biology baked in —
the reference is user-supplied ground truth. Tiny fixture with hand-computable metrics.
"""

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from scpilot import tools
from scpilot.session import Session

# 10 cells; predicted labels are free-text, the reference is a fixed vocabulary.
_REF = ["Bcell", "Bcell", "Tcell", "Tcell", "Tcell", "NK", "NK", "Mono", "Mono", "Mono"]
_PRED = ["B", "B", "T", "T", "B", "NK", "NK", "Mono", "Mono", "X"]
_MAP = {"B": "Bcell", "T": "Tcell", "NK": "NK", "Mono": "Mono"}   # X deliberately unmapped


def _bench_session(tmp_path, pred=_PRED, ref=_REF):
    n = len(ref)
    X = np.random.default_rng(0).poisson(0.5, (n, 5)).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(5)]
    a.obs["pred"] = pred
    a.obs["ref"] = ref
    a.layers["counts"] = a.X.copy()
    p = tmp_path / "b.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    return s


def test_benchmark_reference_hand_computed_metrics(tmp_path):
    s = _bench_session(tmp_path)
    r = tools.run("benchmark_reference", s, pred_key="pred", ref_key="ref", label_map=_MAP)
    assert r.status == "success", r.error
    sm = r.summary

    # label-space honesty: 'X' has no reference counterpart -> 1/10 unmatched, reported clearly
    assert sm["n_unmatched_cells"] == 1
    assert sm["unmatched_predicted_fraction"] == 0.1
    assert sm["unmatched_predicted_labels"] == ["X"]

    # strict score (unmatched counted WRONG): 8/10 cells agree (cell 5 B->Bcell vs Tcell; cell 10 X)
    assert sm["strict_accuracy"] == 0.8
    assert sm["strict_macro_f1"] == 0.85          # (0.8+0.8+1.0+0.8)/4 over Bcell/Mono/NK/Tcell

    # matched-only score (drop the unmatched cell): 8/9 agree; macro-F1 over the 4 ref classes
    assert sm["matched_only"]["n_cells"] == 9
    assert sm["matched_only_accuracy"] == round(8 / 9, 4)
    assert sm["matched_only_macro_f1"] == 0.9

    # BOTH scores present and clearly labeled; matched-only >= strict here (drops the unmatched wrong)
    assert sm["matched_only_accuracy"] > sm["strict_accuracy"]

    # ARI/AMI are label-name-invariant; recompute independently on the strict reconciled labeling
    y_pred_strict = ["Bcell", "Bcell", "Tcell", "Tcell", "Bcell", "NK", "NK", "Mono", "Mono", "UNMATCHED::X"]
    assert sm["ari"] == round(float(adjusted_rand_score(_REF, y_pred_strict)), 4)
    assert -1.0 <= sm["ari"] <= 1.0 and 0.0 <= sm["ami"] <= 1.0

    # per-class precision/recall/F1 over the reference classes (strict)
    pc = sm["strict"]["per_class"]
    assert set(pc) == {"Bcell", "Tcell", "NK", "Mono"}
    assert pc["Bcell"]["recall"] == 1.0 and pc["Bcell"]["precision"] == round(2 / 3, 4)
    assert pc["NK"]["f1"] == 1.0 and pc["NK"]["support"] == 2
    assert pc["Tcell"]["support"] == 3


def test_benchmark_reference_confusion_matrix(tmp_path):
    import json

    import pandas as pd
    s = _bench_session(tmp_path)
    r = tools.run("benchmark_reference", s, pred_key="pred", ref_key="ref", label_map=_MAP)
    metrics = json.load(open(r.summary["metrics_json"]))
    cm = pd.read_csv(metrics["confusion_csv"], index_col=0)
    assert cm.loc["ref::Bcell", "pred::Bcell"] == 2
    assert cm.loc["ref::Tcell", "pred::Bcell"] == 1          # cell 5 mislabeled
    assert cm.loc["ref::Mono", "pred::UNMATCHED::X"] == 1    # unmatched shown, never silently dropped


def test_benchmark_reference_label_map_reconciles_vocabulary(tmp_path):
    # WITHOUT the map, the free-text predicted labels (B/T) do not match the reference vocabulary
    s = _bench_session(tmp_path)
    r_nomap = tools.run("benchmark_reference", s, pred_key="pred", ref_key="ref")
    # B, T, X all lack a reference counterpart -> 6 cells unmatched (3 B + 2 T + 1 X); NK/Mono match
    assert r_nomap.summary["n_unmatched_cells"] == 6
    assert set(r_nomap.summary["unmatched_predicted_labels"]) == {"B", "T", "X"}
    # WITH the map, only X stays unmatched -> reconciliation demonstrably works
    r_map = tools.run("benchmark_reference", s, pred_key="pred", ref_key="ref", label_map=_MAP)
    assert r_map.summary["n_unmatched_cells"] == 1
    assert r_map.summary["strict_accuracy"] > r_nomap.summary["strict_accuracy"]


def test_benchmark_reference_case_normalized_match(tmp_path):
    # case differences alone must NOT count as a mismatch (case-normalized exact match)
    s = _bench_session(tmp_path, pred=["bcell", "BCELL", "tcell", "tcell", "tcell",
                                       "nk", "nk", "mono", "mono", "mono"])
    r = tools.run("benchmark_reference", s, pred_key="pred", ref_key="ref")
    assert r.summary["n_unmatched_cells"] == 0
    assert r.summary["strict_accuracy"] == 1.0        # identical up to casing
    assert r.summary["ari"] == 1.0


def test_benchmark_reference_missing_column_structured_error(tmp_path):
    s = _bench_session(tmp_path)
    r_pred = tools.run("benchmark_reference", s, pred_key="does_not_exist", ref_key="ref")
    assert r_pred.status == "error" and r_pred.error_code == "invalid_state"
    r_ref = tools.run("benchmark_reference", s, pred_key="pred", ref_key="nope")
    assert r_ref.status == "error" and r_ref.error_code == "invalid_state"


def test_benchmark_reference_all_nan_reference(tmp_path):
    s = _bench_session(tmp_path)
    s.adata.obs["ref"] = np.nan
    r = tools.run("benchmark_reference", s, pred_key="pred", ref_key="ref")
    assert r.status == "error" and r.error_code == "data_gate_failed"


def test_benchmark_reference_drops_missing_cells(tmp_path):
    # NaN ground-truth cells cannot be scored -> dropped + reported, not silently wrong
    ref = list(_REF); ref[0] = None; ref[1] = None
    s = _bench_session(tmp_path, ref=ref)
    r = tools.run("benchmark_reference", s, pred_key="pred", ref_key="ref", label_map=_MAP)
    assert r.status == "success"
    assert r.summary["n_dropped_missing"] == 2
    assert r.summary["n_cells_scored"] == 8


def test_benchmark_reference_single_class(tmp_path):
    # single reference class edge case: must not crash, warns that ARI/macro-F1 are degenerate
    s = _bench_session(tmp_path, pred=["Bcell"] * 10, ref=["Bcell"] * 10)
    r = tools.run("benchmark_reference", s, pred_key="pred", ref_key="ref")
    assert r.status == "success"
    assert r.summary["strict_accuracy"] == 1.0
    assert any("single reference class" in w for w in r.warnings)


def test_benchmark_reference_registered_and_readonly():
    specs = {t["name"]: t for t in tools.list_tools()}
    assert "benchmark_reference" in specs
    assert specs["benchmark_reference"]["mutating"] is False   # read-only: no checkpoint, no mutation
