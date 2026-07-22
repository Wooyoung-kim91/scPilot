"""Bug G regression — in-tool DecisionEvents carry the step's recipe_hash join key.

The actual biological CALLs (cell-type / malignancy / consensus / harmonize / finalize) are
logged from INSIDE the tool via ``session.log_decision`` BEFORE the driver's ``record_run``
computes the step's recipe_hash. Before the fix every such decision had ``recipe_hash=None``,
so ``audit_export`` produced ZERO ``used(decision → evidence)`` edges for them — exactly the
biological-call provenance the PROV bundle exists to expose.

These tests drive a tool through the SAME sequence the drivers use — ``begin_step`` → tool →
``record_tool_run`` (this is precisely what cli.py / mcp_server.py / agent.py do) — and assert:
- the in-tool DecisionEvent's ``recipe_hash`` EQUALS the step's RunLogRecord/OutputRecord hash;
- ``export_audit`` now emits a ``used`` edge from that decision to ``entity:evidence/<recipe_hash>``;
- v1 (no schema_version/provenance) and v2 decisions still validate + the log still loads.
"""

import json

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot import schemas as S
from scpilot import tools
from scpilot.session import Session


def _clustered_adata(n=200):
    """Two clearly-separated groups so leiden finds >=2 clusters with clean DE."""
    rng = np.random.default_rng(0)
    n_genes = 40
    X = rng.poisson(0.2, (n, n_genes)).astype("float32")
    half = n // 2
    # group A over-expresses genes 0..4, group B over-expresses genes 5..9
    X[:half, 0:5] += rng.poisson(10.0, (half, 5)).astype("float32")
    X[half:, 5:10] += rng.poisson(10.0, (n - half, 5)).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(n_genes)]
    a.layers["counts"] = a.X.copy()
    a.obs["sample_id"] = rng.choice(["s1", "s2", "s3"], n)
    return a


def _session_ready_for_annotation(tmp_path):
    p = tmp_path / "in.h5ad"
    _clustered_adata().write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p))
    s.load_input()
    tools.run("preprocess", s, n_top_genes=30, n_pcs=10)
    tools.run("cluster", s, resolution=0.5)
    return s


def _drive(session, name, params, seed=0):
    """Exercise a tool through the DRIVER chokepoint sequence (begin_step → tool →
    record_tool_run) — identical to what CLI ``step`` / the MCP handler / the agent do."""
    session.begin_step(params=params, seed=seed)
    result = tools.get(name).fn(session, **params)
    assert result.status == "success", result.error
    rh = session.record_tool_run(result, params=params, seed=seed)
    return result, rh


def _decisions(session):
    return [json.loads(l) for l in session.decisions_path.read_text().splitlines() if l.strip()]


def _runs(session):
    return [json.loads(l) for l in session.run_log_path.read_text().splitlines() if l.strip()]


def _outputs(session):
    return [json.loads(l) for l in session.outputs_path.read_text().splitlines() if l.strip()]


def test_in_tool_decision_recipe_hash_equals_runlog(tmp_path):
    s = _session_ready_for_annotation(tmp_path)
    clusters = sorted(s.adata.obs["leiden"].astype(str).unique())
    labels = {c: ("Epithelial" if i == 0 else "T_NK") for i, c in enumerate(clusters)}

    _res, rh = _drive(s, "apply_annotation", {"groupby": "leiden", "labels": labels})
    assert rh, "begin_step/record_tool_run should have produced a recipe_hash"

    # the in-tool tier1 cell-type CALL now carries the join key (was None before the fix)
    decs = _decisions(s)
    tier1 = next(d for d in decs if d["decision_type"] == "tier1_llm_labels")
    assert tier1["recipe_hash"] == rh, "in-tool decision must carry the step's recipe_hash"

    # ... and it is the SAME value the step's RunLogRecord AND OutputRecord carry
    apply_run = next(r for r in _runs(s)
                     if r["tool"] == "apply_annotation" and r["status"] == "success")
    apply_out = next(o for o in _outputs(s) if o["tool"] == "apply_annotation")
    assert apply_run["recipe_hash"] == rh
    assert apply_out["recipe_hash"] == rh
    # the JOIN INVARIANT: decision.recipe_hash == runlog.recipe_hash for the same tool run
    assert tier1["recipe_hash"] == apply_run["recipe_hash"] == apply_out["recipe_hash"]

    # a deterministic in-tool call keeps model_id=None (this fix is the recipe_hash join only)
    assert tier1.get("model_id") is None


def test_malignancy_in_tool_decision_carries_recipe_hash(tmp_path):
    # the malignancy_call DecisionEvent (cnv.py) is joined too — build a cnv grouping by hand
    s = _session_ready_for_annotation(tmp_path)
    a = s.adata
    a.obs["cnv_leiden"] = a.obs["leiden"].astype(str).astype("category")
    s.set_adata(a)
    groups = sorted(a.obs["cnv_leiden"].astype(str).unique())
    labels = {g: ("malignant" if i == 0 else "non_malignant") for i, g in enumerate(groups)}

    _res, rh = _drive(s, "apply_malignancy", {"groupby": "cnv_leiden", "labels": labels})
    decs = _decisions(s)
    mal = next(d for d in decs if d["decision_type"] == "malignancy_call")
    mal_run = next(r for r in _runs(s)
                   if r["tool"] == "apply_malignancy" and r["status"] == "success")
    assert mal["recipe_hash"] == rh == mal_run["recipe_hash"]


def test_audit_export_used_edge_is_no_longer_dead(tmp_path):
    s = _session_ready_for_annotation(tmp_path)
    clusters = sorted(s.adata.obs["leiden"].astype(str).unique())
    labels = {c: ("Epithelial" if i == 0 else "T_NK") for i, c in enumerate(clusters)}
    _res, rh = _drive(s, "apply_annotation", {"groupby": "leiden", "labels": labels})

    r = tools.run("export_audit", s)
    assert r.status == "success", r.error
    doc = json.loads((s.out / "audit_bundle.json").read_text())

    # the step's evidence bundle is keyed by the recipe_hash
    ev_id = f"entity:evidence/{rh}"
    assert ev_id in doc["entity"]

    # the in-tool tier1_llm_labels decision USED that evidence (the join that used to be DEAD)
    dec_id = next(a for a, v in doc["activity"].items()
                  if v.get("scpilot:decision_type") == "tier1_llm_labels")
    assert doc["activity"][dec_id]["scpilot:recipe_hash"] == rh
    used_targets = {u["prov:entity"] for u in doc.get("used", {}).values()
                    if u["prov:activity"] == dec_id}
    assert ev_id in used_targets, "in-tool decision must now be joined to its evidence by recipe_hash"


def test_v1_and_v2_decisions_still_validate_and_load(tmp_path):
    # backward-compat: a v1 event (no schema_version, no provenance, no recipe_hash) and a v2 event
    # (with recipe_hash) must BOTH validate, and a decisions.jsonl mixing them must load cleanly.
    v1 = {"decision_type": "qc_cutoff", "choice": {"min_genes": 500},
          "candidates": [], "rationale": "v1 event, no provenance fields"}
    assert S.validate_decision(v1) == []

    v2 = S.DecisionEvent(
        decision_type="tier1_llm_labels", choice={"0": "T cell"}, candidates=[],
        rationale="v2 event with join key", recipe_hash="0123456789abcdef",
        model_id="m", prompt_version="v1", prompt_hash="deadbeefdeadbeef", temperature=0.0,
    ).to_dict()
    assert S.validate_decision(v2) == []
    assert v2["schema_version"] == S.DECISION_SCHEMA_VERSION

    inp = tmp_path / "in.h5ad"
    _clustered_adata(n=40).write_h5ad(inp)
    s = Session.create(tmp_path / "sess_compat", input_path=str(inp))
    s.load_input()
    s.log_decision(v1)                                 # v1 dict (no active hash → stays None)
    s.log_decision(S.DecisionEvent(**{k: v for k, v in {
        "decision_type": "malignancy_call", "choice": {"0": "malignant"},
        "candidates": [], "rationale": "v2"}.items()}))
    loaded = _decisions(s)
    assert [d["decision_type"] for d in loaded] == ["qc_cutoff", "malignancy_call"]
    assert loaded[0].get("recipe_hash") is None        # no begin_step → honest None
