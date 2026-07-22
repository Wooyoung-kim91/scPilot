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
    assert sm["marker_criteria"] == {"min_pct": 0.25, "min_lfc": 1.0, "padj_max": 0.05,
                                     "min_specificity": 0.1, "max_pct_out": 0.5}
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
    # flags→action: suspect clusters are surfaced (not just counted) for targeted review
    assert sm["suspect_clusters"] == ["3"] and "malignant without CNV" in sm["suspect_reasons"]["3"]
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
    assert {"annotation_audit", "apply_annotation_audit", "harness_audit"} <= names


def test_annotation_audit_emits_granularity_evidence(tmp_path):
    # annotation_audit emits granularity EVIDENCE (no hardcoded threshold) so the reviewer can judge
    # whether the resolution is too fine. The fixture splits one profile (A-markers) across clusters
    # 0 and 1 → collapse_ratio > 1 and a profile collision are surfaced as over-clustering signals.
    s = _audit_session(tmp_path)
    r = tools.run("annotation_audit", s, groupby="leiden", label_key="major_cell_type")
    g = r.summary["granularity"]
    assert g["n_clusters"] == 4 and g["n_labels"] >= 1
    assert g["collapse_ratio"] == round(g["n_clusters"] / g["n_labels"], 3)
    # TypeA is assigned to clusters 0 and 3 → at least one redundant (same-label) cluster
    assert g["n_redundant_label_clusters"] >= 1 and g["max_clusters_per_label"] >= 2
    for k in ("n_profile_collision_clusters", "n_weak_support_clusters", "frac_flagged"):
        assert k in g
    # no hardcoded verdict — the tool does NOT decide "over_clustered"; that's the reviewer's call
    assert "assessment" not in g and "recommend_resolution" not in g


def test_apply_annotation_audit_records_granularity_recommendation(tmp_path):
    # the reviewer's advisory resolution recommendation is recorded (uns + summary) for the human/agent.
    from scpilot.core.annotate import UNS_ANNO
    s = _audit_session(tmp_path)
    gran = {"assessment": "over_clustered", "recommend_resolution": "down",
            "rationale": "collapse_ratio 4.0; 3 weak-support clusters"}
    r = tools.run("apply_annotation_audit", s, groupby="leiden", label_key="major_cell_type",
                  verdicts={"0": {"status": "confirmed"}}, reviewer_model="rev-A", granularity=gran)
    assert r.summary["granularity"]["recommend_resolution"] == "down"
    t4 = s.adata.uns[UNS_ANNO]["tier4_audit"]
    assert t4["granularity"]["assessment"] == "over_clustered"
    assert s.adata.uns[UNS_ANNO]["tier4_reviews"]["major_cell_type"]["granularity"]["recommend_resolution"] == "down"


def test_apply_annotation_audit_records_per_column_ledger(tmp_path):
    # apply_annotation_audit must record coverage KEYED BY label_key so harness_audit can prove
    # EVERY annotation column was reviewed. Two reviews on different columns → two ledger entries.
    from scpilot.core.annotate import UNS_ANNO
    s = _audit_session(tmp_path)
    tools.run("apply_annotation_audit", s, groupby="leiden", label_key="major_cell_type",
              verdicts={"0": {"status": "confirmed"}, "1": {"status": "refuted", "note": "weak"}},
              reviewer_model="rev-A")
    tools.run("apply_annotation_audit", s, groupby="leiden", label_key="malignancy",
              verdicts={"3": {"status": "suspect", "note": "no cnv"}}, reviewer_model="rev-B")
    ledger = s.adata.uns[UNS_ANNO]["tier4_reviews"]
    assert set(ledger) == {"major_cell_type", "malignancy"}
    assert ledger["major_cell_type"]["reviewer_model"] == "rev-A"
    assert ledger["malignancy"]["reviewer_model"] == "rev-B"
    assert ledger["major_cell_type"]["refuted_clusters"] == ["1"]


def test_harness_audit_governance_scorecard(tmp_path):
    # the governance/監視 agent: checks the harness's OWN action rules were honored.
    s = _audit_session(tmp_path)
    # COMPLIANT pipeline: marker-DB-free annotation + Tier-4 + finalize, all in the run-log
    for tool in ["markers", "annotation_review", "apply_annotation", "annotation_audit",
                 "apply_annotation_audit", "finalize_annotation"]:
        s._append_jsonl(s.run_log_path, {"tool": tool, "status": "success", "seed": 0, "recipe_hash": tool})
    s.adata.uns[UNS_ANNO]["tier4_audit"] = {"reviewer_model": "rev-model",
                                            "refuted_clusters": [], "suspect_clusters": ["3"]}
    # PER-COLUMN Tier-4 coverage: EVERY label column present in obs must be independently reviewed
    # (here major_cell_type + malignancy). harness_audit now gates on this ledger, not one audit call.
    s.adata.uns[UNS_ANNO]["tier4_reviews"] = {
        "major_cell_type": {"reviewer_model": "rev-model", "refuted_clusters": [], "suspect_clusters": ["3"]},
        "malignancy": {"reviewer_model": "rev-model", "refuted_clusters": [], "suspect_clusters": []},
    }
    s.adata.obs["annotation_review_required"] = (s.adata.obs["leiden"].astype(str) == "3")
    r = tools.run("harness_audit", s)
    assert r.status == "success"
    st = {c["check"]: c["status"] for c in r.summary["checks"]}
    assert st["tier4_review_ran"] == "pass"          # every annotation column reviewed
    assert st["marker_db_free"] == "pass"            # no annotate_broad
    assert st["annotation_present"] == "pass" and st["finalized"] == "pass"
    assert st["flags_lead_to_action"] == "pass"      # suspect cl3 → review_required set
    assert r.summary["verdict"] == "complete"        # no hard failures

    # VIOLATION: a run that used the legacy fixed panel and skipped Tier-4
    (tmp_path / "v2").mkdir(parents=True, exist_ok=True)
    s2 = _audit_session(tmp_path / "v2")
    for tool in ["markers", "annotate_broad"]:
        s2._append_jsonl(s2.run_log_path, {"tool": tool, "status": "success", "seed": 0, "recipe_hash": tool})
    r2 = tools.run("harness_audit", s2)
    st2 = {c["check"]: c["status"] for c in r2.summary["checks"]}
    assert st2["tier4_review_ran"] == "fail"         # no annotation_audit/apply_annotation_audit
    assert st2["marker_db_free"] == "fail"           # annotate_broad used → HARD fail (marker-DB-free rule)
    assert r2.summary["verdict"] == "incomplete"
    assert {"tier4_review_ran", "marker_db_free"} <= set(r2.summary["violations"])


def test_harness_audit_flags_unreviewed_annotation_column(tmp_path):
    # The exact reliability gap: a fine (Tier-2) annotation exists but was NEVER independently
    # reviewed. Even though broad + malignancy were reviewed, the present-but-unreviewed column
    # must fail the per-column coverage gate (reliability of ALL annotation, not just one).
    s = _audit_session(tmp_path)
    for tool in ["markers", "apply_annotation", "annotation_audit", "apply_annotation_audit",
                 "apply_fine_annotation", "finalize_annotation"]:
        s._append_jsonl(s.run_log_path, {"tool": tool, "status": "success", "seed": 0, "recipe_hash": tool})
    # fine_cell_type + final_annotation now exist in obs, but only broad + malignancy were reviewed
    s.adata.obs["fine_cell_type"] = s.adata.obs["major_cell_type"].astype(str) + "_sub"
    s.adata.obs["final_annotation"] = s.adata.obs["major_cell_type"]
    s.adata.uns.setdefault(UNS_ANNO, {})["tier4_reviews"] = {
        "major_cell_type": {"reviewer_model": "rev-model"},
        "malignancy": {"reviewer_model": "rev-model"},
    }
    r = tools.run("harness_audit", s)
    st = {c["check"]: c["status"] for c in r.summary["checks"]}
    assert st["tier4_review_ran"] == "fail"           # fine_cell_type + final_annotation unreviewed
    # the detail names the unreviewed columns so the action item is actionable
    detail = next(c["detail"] for c in r.summary["checks"] if c["check"] == "tier4_review_ran")
    assert "fine_cell_type" in detail and "final_annotation" in detail
    assert r.summary["verdict"] == "incomplete"


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


# ---------------------------------------------------------------------------
# Bug E: a FAILED apply_annotation_audit (verdicts NOT recorded) must NOT be read as
# "reviewed, nothing refuted". run_annotation_critique returns a NON-success status, and
# run_annotation_review_loop must NOT claim convergence — it surfaces the failure instead,
# so an unreviewed label is never shipped as clean (defeats the §4 independent-review invariant).
# ---------------------------------------------------------------------------
def test_critique_returns_non_success_when_apply_fails(tmp_path):
    from scpilot.llm.agent import run_annotation_critique

    s = _audit_session(tmp_path)
    # reviewer OMITS cluster_id on every verdict -> the built verdicts map is empty ->
    # apply_annotation_audit errors (missing_input). The verdicts were NOT recorded.
    reviewer = _Fake([
        _emit("emit_annotation_audit", {"reviewer_model": "rev", "verdicts": [
            {"status": "confirmed", "review_required": False}]}),
    ], model="reviewer-model")

    crit = run_annotation_critique(s, reviewer, groupby="leiden", label_key="major_cell_type",
                                   seed=0, analysis_model="annotator-model")
    # NOT a success, and NOT a success-looking payload with an empty refuted set
    assert crit["status"] != "success"
    assert crit["status"] != "skipped"           # the audit DID run; it's the apply that failed
    assert "refuted_clusters" not in crit         # never a fake "reviewed, nothing refuted"
    assert crit.get("error")                       # the failure reason is surfaced


def test_review_loop_does_not_converge_when_apply_fails(tmp_path):
    from scpilot.llm.agent import run_annotation_review_loop

    s = _audit_session(tmp_path)
    reviewer = _Fake([
        _emit("emit_annotation_audit", {"reviewer_model": "rev", "verdicts": [
            {"status": "confirmed", "review_required": False}]}),
    ], model="reviewer-model")
    annotator = _Fake([], model="annotator-model")   # never reached — the round fails first

    res = run_annotation_review_loop(s, annotator, reviewer, groupby="leiden",
                                     label_key="major_cell_type", max_rounds=3, seed=0)
    # the loop must NOT declare the annotation clean on a failed review round
    assert res.get("converged") is not True
    assert res["status"] != "completed"           # a real failure, not a silent "completed"
    assert res["status"] != "skipped"
    assert res.get("reason")                       # the failure reason is surfaced


def test_review_loop_converges_on_clean_successful_review(tmp_path):
    # Happy path (UNCHANGED): a SUCCESSFUL review that genuinely finds 0 refuted -> converged=True.
    from scpilot.llm.agent import run_annotation_review_loop

    s = _audit_session(tmp_path)
    reviewer = _Fake([
        _emit("emit_annotation_audit", {"reviewer_model": "rev", "verdicts": [
            {"cluster_id": c, "status": "confirmed", "review_required": False}
            for c in ["0", "1", "2", "3"]]}),
    ], model="reviewer-model")
    annotator = _Fake([], model="annotator-model")   # not reached — nothing refuted

    res = run_annotation_review_loop(s, annotator, reviewer, groupby="leiden",
                                     label_key="major_cell_type", max_rounds=3, seed=0)
    assert res["status"] == "completed"
    assert res["converged"] is True and res["final_refuted"] == []
    assert res["n_rounds"] == 1


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


# ---------------------------------------------------------------------------
# Improvement ③ Part B: reviewer INDEPENDENCE is a first-class recorded flag, not a silent
# self-review fallback. apply_annotation_audit records it; harness_audit surfaces the status.
# ---------------------------------------------------------------------------
def test_apply_annotation_audit_records_reviewer_independence(tmp_path):
    s = _audit_session(tmp_path)
    # distinct reviewer vs analysis model -> INDEPENDENT (satisfied)
    r = tools.run("apply_annotation_audit", s, groupby="leiden", label_key="major_cell_type",
                  verdicts={"0": {"status": "confirmed"}},
                  reviewer_model="gpt-reviewer", analysis_model="claude-analysis")
    assert r.summary["reviewer_independent"] is True
    assert r.summary["review_mode"] == "independent"
    ledger = s.adata.uns[UNS_ANNO]["tier4_reviews"]["major_cell_type"]
    assert ledger["reviewer_independent"] is True and ledger["review_mode"] == "independent"


def test_apply_annotation_audit_flags_degraded_self_review(tmp_path):
    s = _audit_session(tmp_path)
    # reviewer model == analysis model (the silent fallback) -> DEGRADED self-review, recorded explicitly
    r = tools.run("apply_annotation_audit", s, groupby="leiden", label_key="major_cell_type",
                  verdicts={"0": {"status": "confirmed"}},
                  reviewer_model="claude-analysis", analysis_model="claude-analysis")
    assert r.summary["reviewer_independent"] is False
    assert r.summary["review_mode"] == "self-review-degraded"
    t4 = s.adata.uns[UNS_ANNO]["tier4_audit"]
    assert t4["reviewer_independent"] is False and t4["review_mode"] == "self-review-degraded"


def test_run_annotation_critique_records_self_review_when_same_model(tmp_path):
    # the self-review fallback: annotator and reviewer are the SAME model -> critique records degraded
    from scpilot.llm.agent import run_annotation_critique
    s = _audit_session(tmp_path)
    reviewer = _Fake([
        _emit("emit_annotation_audit", {"reviewer_model": "same-model", "verdicts": [
            {"cluster_id": c, "status": "confirmed", "review_required": False} for c in ["0", "1", "2", "3"]]}),
    ], model="same-model")
    crit = run_annotation_critique(s, reviewer, groupby="leiden", label_key="major_cell_type",
                                   seed=0, analysis_model="same-model")
    assert crit["reviewer_independent"] is False and crit["review_mode"] == "self-review-degraded"
    assert s.adata.uns[UNS_ANNO]["tier4_reviews"]["major_cell_type"]["reviewer_independent"] is False


def test_harness_audit_surfaces_reviewer_independence(tmp_path):
    s = _audit_session(tmp_path)
    for tool in ["markers", "annotation_review", "apply_annotation", "annotation_audit",
                 "apply_annotation_audit", "finalize_annotation"]:
        s._append_jsonl(s.run_log_path, {"tool": tool, "status": "success", "seed": 0, "recipe_hash": tool})
    s.adata.uns[UNS_ANNO]["tier4_audit"] = {"reviewer_model": "rev-model",
                                            "refuted_clusters": [], "suspect_clusters": []}

    # SATISFIED: distinct reviewer/analysis models on every reviewed column
    s.adata.uns[UNS_ANNO]["tier4_reviews"] = {
        "major_cell_type": {"reviewer_model": "gpt", "analysis_model": "claude",
                            "reviewer_independent": True, "review_mode": "independent"},
    }
    r = tools.run("harness_audit", s)
    st = {c["check"]: c["status"] for c in r.summary["checks"]}
    assert st["tier4_reviewer_independence"] == "pass"
    assert r.summary["reviewer_independence"] == "satisfied"

    # DEGRADED: reviewer fell back to the analysis model -> machine-readable degraded_self_review (advisory warn)
    s.adata.uns[UNS_ANNO]["tier4_reviews"] = {
        "major_cell_type": {"reviewer_model": "claude", "analysis_model": "claude",
                            "reviewer_independent": False, "review_mode": "self-review-degraded"},
    }
    r2 = tools.run("harness_audit", s)
    st2 = {c["check"]: c["status"] for c in r2.summary["checks"]}
    assert st2["tier4_reviewer_independence"] == "warn"      # advisory, does NOT hard-fail the run
    assert r2.summary["reviewer_independence"] == "degraded_self_review"
    assert r2.summary["degraded_self_review_columns"] == ["major_cell_type"]
    # independence is advisory: it contributes a WARN, never a hard FAIL that flips the verdict
    assert "tier4_reviewer_independence" not in [c["check"] for c in r2.summary["checks"]
                                                 if c["status"] == "fail"]
