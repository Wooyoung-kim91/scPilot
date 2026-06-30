"""Unit tests for the Tier-4 consistency audit — annotation_audit (evidence) + apply_annotation_audit.

Deterministic, marker-DB-free: the audit emits the seven inconsistency checks as EVIDENCE; the
verdict (confirmed/suspect/refuted) is the reviewer's, recorded by the apply tool. Tiny fixture.
"""

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

from scpilot import tools
from scpilot.core.annotate import UNS_ANNO
from scpilot.session import Session


def _audit_session(tmp_path):
    """4 leiden clusters with a planted inconsistency in each Tier-4 axis."""
    rng = np.random.default_rng(0)
    A = ["A1", "A2", "A3"]; B = ["B1", "B2", "B3"]
    genes = A + B + [f"G{i}" for i in range(30)]
    gi = {g: i for i, g in enumerate(genes)}
    spec = [("0", 120), ("1", 120), ("2", 120), ("3", 120)]
    leiden = sum(([c] * n for c, n in spec), [])
    N = len(leiden)
    X = rng.poisson(0.1, (N, len(genes))).astype("float32")

    def boost(cl, gs, lam=12.0):
        idx = np.array([i for i, c in enumerate(leiden) if c == cl])
        for g in gs:
            X[np.ix_(idx, [gi[g]])] += rng.poisson(lam, (idx.size, 1)).astype("float32")

    boost("0", A)            # A-markers, labeled TypeA  -> clean, well-supported
    boost("1", A)            # A-markers, labeled TypeB  -> WRONG: collides with 0 + weak support
    boost("2", B)            # B-markers, labeled TypeB  -> well-supported
    boost("3", B)            # B-markers, labeled malignant, NO cnv + single patient

    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = genes
    a.obs["leiden"] = leiden
    a.obs["leiden"] = a.obs["leiden"].astype("category")
    a.layers["counts"] = a.X.copy()
    # cluster 3 is single-patient dominated; the rest are spread across samples
    a.obs["sample_id"] = [("p9" if c == "3" else rng.choice(["s1", "s2", "s3"])) for c in leiden]
    # malignancy: only cluster 3 malignant, but NO cnv_score column exists -> malignant_without_cnv
    a.obs["major_cell_type"] = [{"0": "TypeA", "1": "TypeB", "2": "TypeB", "3": "TypeA"}[c] for c in leiden]
    a.obs["malignancy"] = [("malignant" if c == "3" else "non_malignant") for c in leiden]
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    p = tmp_path / "audit.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    # record the LLM's marker_sets (the audit reads these — it owns no panel)
    s.adata.uns.setdefault(UNS_ANNO, {})
    s.adata.uns[UNS_ANNO]["tier1_llm"] = {
        "marker_sets": {"TypeA": ["A1", "A2", "A3"], "TypeB": ["B1", "B2", "B3"]}}
    tools.run("markers", s, groupby="leiden")     # populates rank_genes_groups (Wilcoxon)
    return s


def test_annotation_audit_emits_seven_checks(tmp_path):
    s = _audit_session(tmp_path)
    r = tools.run("annotation_audit", s, groupby="leiden", label_key="major_cell_type")
    assert r.status == "success", r.error
    sm = r.summary
    assert sm["marker_db_used"] is False                       # no hardcoded panel
    assert sm["n_clusters"] == 4
    # check 1: cluster 0 (TypeA) vs cluster 1 (TypeB) share A-markers -> a profile collision
    assert sm["n_marker_profile_collisions"] >= 1
    # check 7: cluster 3 is malignant with no cnv_score -> flagged
    assert sm["n_malignant_without_cnv"] >= 1
    assert "3" in sm["flagged_clusters"]
    # per-cluster evidence is written and self-describing
    import json
    doc = json.load(open(sm["audit_input"]))
    by = {c["cluster_id"]: c for c in doc["clusters"]}
    # cluster 1 labeled TypeB but expresses A-markers -> low marker-set support
    assert by["1"]["marker_set_support_frac"] is not None
    assert by["1"]["marker_set_support_frac"] < by["0"]["marker_set_support_frac"]
    assert "weak_marker_support" in by["1"]["flags"]
    # marker criteria (pct/logFC/p-value) are validated per claimed gene, and reported
    assert sm["marker_criteria"] == {"min_pct": 0.25, "min_lfc": 1.0, "padj_max": 0.05}
    # cluster 0 (TypeA, A-markers boosted) -> its claimed A genes PASS all criteria
    ev0 = {m["gene"]: m for m in by["0"]["marker_criteria_check"]}
    assert ev0 and all(m["passes"] for m in ev0.values())
    assert all({"pct_in", "logFC", "padj"} <= set(m) for m in ev0.values())   # stats surfaced
    # cluster 1 (TypeB label, B-markers NOT expressed) -> claimed B genes FAIL pct/lfc
    ev1 = {m["gene"]: m for m in by["1"]["marker_criteria_check"]}
    assert not all(m["passes"] for m in ev1.values())
    assert any(("pct" in m["failed_criteria"] or "lfc" in m["failed_criteria"])
               for m in ev1.values() if not m["passes"])
    # check 4: cluster 3 single-patient dominated
    assert "single_patient_dominant" in by["3"]["flags"]
    # hierarchy triple surfaced for the reviewer to judge (no built-in lineage map)
    assert by["0"]["hierarchy"]["major"] == "TypeA"


def test_annotation_audit_needs_labels(tmp_path):
    s = _audit_session(tmp_path)
    r = tools.run("annotation_audit", s, groupby="leiden", label_key="does_not_exist")
    assert r.status == "error" and r.error_code == "invalid_state"


def test_apply_annotation_audit_records_verdicts_and_reasons(tmp_path):
    s = _audit_session(tmp_path)
    tools.run("annotation_audit", s, groupby="leiden", label_key="major_cell_type")
    # the reviewer flags WHETHER each label holds + the REASON — never a replacement cell type
    verdicts = {
        "0": {"status": "confirmed", "review_required": False},
        "1": {"status": "refuted", "review_required": True, "note": "TypeB markers fail pct/lfc; support 0/3"},
        "2": {"status": "confirmed", "review_required": False},
        "3": {"status": "suspect", "review_required": True, "note": "malignant without CNV evidence"},
    }
    r = tools.run("apply_annotation_audit", s, groupby="leiden", verdicts=verdicts,
                  reviewer_model="test-reviewer")
    assert r.status == "success", r.error
    sm = r.summary
    assert sm["n_refuted"] == 1 and sm["n_suspect"] == 1 and sm["n_confirmed"] == 2
    assert sm["reviewer_model"] == "test-reviewer"
    assert sm["refuted_clusters"] == ["1"]
    # the rejection REASON is recorded (for re-annotation + humans); NO replacement label is given
    assert "fail pct/lfc" in sm["refuted_reasons"]["1"]
    assert "annotation_audit_status" in s.adata.obs and "annotation_review_required" in s.adata.obs
    # the audit does NOT relabel — cluster 1 keeps its (refuted) label until the annotator re-infers it
    c1 = s.adata.obs.loc[s.adata.obs["leiden"] == "1", "major_cell_type"].astype(str).unique()
    assert list(c1) == ["TypeB"]
    t4 = s.adata.uns[UNS_ANNO]["tier4_audit"]
    assert t4["n_refuted"] == 1 and t4["refuted_reasons"]["1"]


def test_apply_annotation_audit_rejects_corrected_label_is_gone(tmp_path):
    # the reviewer schema no longer carries a corrected_label; the apply tool ignores stray keys
    s = _audit_session(tmp_path)
    r = tools.run("apply_annotation_audit", s, groupby="leiden",
                  verdicts={"1": {"status": "refuted", "review_required": True, "note": "why"}})
    assert r.status == "success" and r.summary["refuted_clusters"] == ["1"]


def test_apply_annotation_audit_rejects_bad_status(tmp_path):
    s = _audit_session(tmp_path)
    r = tools.run("apply_annotation_audit", s, groupby="leiden",
                  verdicts={"0": {"status": "looks_fine"}})   # not in the fixed vocabulary
    assert r.status == "error" and r.error_code == "invalid_params"


def test_audit_tools_registered():
    names = {t["name"] for t in tools.list_tools()}
    assert {"annotation_audit", "apply_annotation_audit"} <= names


# ---------------------------------------------------------------------------
# Bounded annotation+verification LOOP (mode-2): a scripted reviewer refutes a label,
# the annotator re-infers it (told only the reason, not a replacement), and the loop converges.
# No network — FakeProvider returns fixed structured tool calls.
# ---------------------------------------------------------------------------
from scpilot.llm.provider import LLMResponse, ToolCall   # noqa: E402


class _Fake:
    def __init__(self, script, model="fake-model"):
        self.script = list(script); self.model = model; self.name = "fake"

    def complete(self, messages, *, tools=None, system=None, tool_choice=None, max_tokens=None):
        return self.script.pop(0)

    def tool_result_message(self, call, content):
        return {"role": "tool", "content": content}


def _emit(name, args):
    return LLMResponse(text="", tool_calls=[ToolCall(id="c", name=name, arguments=args)],
                       stop_reason="tool_use", usage={})


def test_review_loop_reannotates_refuted_then_converges(tmp_path):
    from scpilot.llm.agent import run_annotation_review_loop

    s = _audit_session(tmp_path)   # cluster 1 mislabeled TypeB (expresses A-markers)
    # reviewer: round 1 refutes cluster 1 (with a REASON, no replacement); round 2 confirms all
    reviewer = _Fake([
        _emit("emit_annotation_audit", {"reviewer_model": "rev", "verdicts": [
            {"cluster_id": "1", "status": "refuted", "review_required": True,
             "note": "claimed TypeB markers fail pct/lfc; expresses A-markers"},
            {"cluster_id": "0", "status": "confirmed", "review_required": False},
            {"cluster_id": "2", "status": "confirmed", "review_required": False},
            {"cluster_id": "3", "status": "confirmed", "review_required": False}]}),
        _emit("emit_annotation_audit", {"reviewer_model": "rev", "verdicts": [
            {"cluster_id": c, "status": "confirmed", "review_required": False} for c in ["0", "1", "2", "3"]]}),
    ], model="reviewer-model")
    # annotator: re-infers ONLY cluster 1 from the DE evidence -> TypeA (independent of the reviewer)
    annotator = _Fake([
        _emit("emit_annotation_labels", {"clusters": [
            {"cluster_id": "1", "major_cell_type": "TypeA", "malignancy": "non_malignant",
             "confidence": 0.9, "review_required": False}]}),
    ], model="annotator-model")

    res = run_annotation_review_loop(s, annotator, reviewer, groupby="leiden",
                                     label_key="major_cell_type", max_rounds=3, seed=0)
    assert res["status"] == "completed"
    assert res["converged"] is True and res["final_refuted"] == []
    assert res["n_rounds"] == 2                       # refute -> re-annotate -> confirm
    # the refuted cluster was re-annotated by the ANNOTATOR (not the reviewer) to TypeA
    c1 = s.adata.obs.loc[s.adata.obs["leiden"] == "1", "major_cell_type"].astype(str).unique()
    assert list(c1) == ["TypeA"]


def test_run_agent_reviews_after_broad_annotation(tmp_path):
    # the cross-model hook: run_agent fires an INDEPENDENT review right after apply_annotation
    from scpilot.llm.agent import run_agent

    s = _audit_session(tmp_path)
    labels = {"0": "TypeA", "1": "TypeB", "2": "TypeB", "3": "TypeA"}
    msets = {"TypeA": ["A1", "A2", "A3"], "TypeB": ["B1", "B2", "B3"]}
    annotator = _Fake([
        _emit("apply_annotation", {"groupby": "leiden", "labels": labels,
                                   "key": "major_cell_type", "marker_sets": msets}),
        LLMResponse(text="done", tool_calls=[], stop_reason="end_turn", usage={}),
    ], model="annotator-model")
    reviewer = _Fake([   # confirms all -> hook records verdicts, no re-annotation needed
        _emit("emit_annotation_audit", {"reviewer_model": "rev", "verdicts": [
            {"cluster_id": c, "status": "confirmed", "review_required": False} for c in ["0", "1", "2", "3"]]}),
    ], model="reviewer-model")

    run_agent(s, annotator, reviewer_provider=reviewer, review=True, max_iters=5, seed=0)
    # the reviewer hook ran the audit on the broad labels and recorded its verdicts in obs/uns
    assert "annotation_audit_status" in s.adata.obs
    assert s.adata.uns[UNS_ANNO]["tier4_audit"]["reviewer_model"] == "reviewer-model"
