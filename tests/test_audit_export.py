"""Unit tests for the machine-readable audit export — export_audit (Improvement ②).

Builds a MINIMAL session with a couple of decisions (one carrying the recipe_hash join key +
an artifact with sha256, one deterministic-tool decision with no model) plus a Tier-4 reviewer
verdict, runs the export, and asserts the PROV-JSON bundle is well-formed and actually LINKS
evidence → decision → verdict → artifact. No real pipeline / LLM is run — records are synthesized
via the same harness writers the drivers use (record_run / log_decision).
"""

import json

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot import schemas as S
from scpilot import tools
from scpilot.session import Session


def _tiny_adata(n_obs=40, n_vars=20):
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(1.0, size=(n_obs, n_vars)).astype("float32"))
    a = ad.AnnData(X)
    a.obs_names = [f"cell{i}" for i in range(n_obs)]
    a.var_names = [f"G{i}" for i in range(n_vars)]
    a.layers["counts"] = a.X.copy()
    return a


def _built_session(tmp_path):
    """A session with: one recorded tool run producing an artifact (sha256) + its evidence, an
    LLM decision joined to that evidence by recipe_hash, a deterministic-tool decision (no model),
    and a Tier-4 reviewer verdict decision."""
    inp = tmp_path / "input.h5ad"
    _tiny_adata().write_h5ad(inp)
    s = Session.create(tmp_path / "sess", input_path=str(inp))
    s.load_input()

    # a real artifact file on disk so _artifact_records records its sha256
    art_path = s.artifacts_dir / "markers_de.csv"
    art_path.write_text("gene,logfc\nG1,2.0\nG2,1.5\n")
    result = S.success(
        "markers", summary={"n_clusters": 3, "n_genes_ranked": 20},
        artifacts=[S.artifact_csv(str(art_path), n_rows=2, n_cols=2, description="DE table")],
    )
    # record_run writes RunLogRecord + OutputRecord (both carry the SAME recipe_hash) and returns it
    rh = s.record_run(result, params={"groupby": "leiden"}, seed=0)
    assert rh, "expected a computed recipe_hash join key"

    # (1) an LLM decision joined to that step's evidence via recipe_hash, with model provenance
    s.log_decision(S.DecisionEvent(
        decision_type="clustering_resolution", choice={"resolution": 0.25},
        candidates=[{"resolution": 0.25}, {"resolution": 0.5}],
        rationale="knee just before the n_clusters jump", stage="cluster",
        model_id="test-model", prompt_version="v1", prompt_hash="abc123",
        temperature=0.0, recipe_hash=rh,
    ))
    # (2) a DETERMINISTIC-tool decision: no model_id, no recipe_hash → must NOT fabricate an agent
    s.log_decision(S.DecisionEvent(
        decision_type="cnv_reference", choice={"reference": "auto"},
        candidates=[], rationale="deterministic reference pick", stage="cnv_score",
    ))
    # (3) a Tier-4 reviewer verdict (the shape apply_annotation_audit logs)
    s.log_decision(S.DecisionEvent(
        decision_type="annotation_audit",
        choice={"0": "confirmed", "1": "refuted"}, candidates=[],
        rationale="Tier-4 reviewer verdicts (reviewer_model=reviewer-x); 1 refuted",
        stage="apply_annotation_audit",
        params={"groupby": "leiden", "reviewer_model": "reviewer-x"},
    ))
    return s, rh, art_path


def test_export_audit_registered_and_non_mutating():
    specs = {t["name"]: t for t in tools.list_tools()}
    assert "export_audit" in specs
    assert specs["export_audit"]["mutating"] is False   # read-only over the session


def test_export_audit_writes_valid_prov_json(tmp_path):
    s, rh, _ = _built_session(tmp_path)
    r = tools.run("export_audit", s)
    assert r.status == "success", r.error
    # (a) the file exists in the session dir and (a/b) parses as valid JSON
    bundle_path = s.out / "audit_bundle.json"
    assert bundle_path.exists()
    assert r.summary["bundle_path"] == str(bundle_path)
    doc = json.loads(bundle_path.read_text())

    # (b) well-formed PROV-JSON: prefix + the three node maps + a small scpilot header
    assert doc["scpilot"]["prov_format"] == "W3C PROV-JSON"
    assert doc["scpilot"]["session_id"] == s.manifest.session_id
    assert doc["scpilot"]["bundle_schema_version"] == 1
    assert "closed_model_provenance_note" in doc["scpilot"]      # honest closed-model note
    for k in ("prefix", "entity", "activity", "agent"):
        assert k in doc and isinstance(doc[k], dict)
    assert doc["prefix"]["prov"].startswith("http")


def test_decision_linked_to_evidence_and_artifact(tmp_path):
    # (c) a DecisionEvent node is actually LINKED to its evidence via recipe_hash and to an
    #     artifact via sha256 (not merely listed side by side).
    s, rh, art_path = _built_session(tmp_path)
    tools.run("export_audit", s)
    doc = json.loads((s.out / "audit_bundle.json").read_text())

    # the evidence bundle for this step is keyed by the recipe_hash
    ev_id = f"entity:evidence/{rh}"
    assert ev_id in doc["entity"]
    assert doc["entity"][ev_id]["prov:type"] == "prov:Collection"
    assert doc["entity"][ev_id]["scpilot:recipe_hash"] == rh

    # the clustering_resolution decision USED that evidence (the recipe_hash JOIN)
    dec_id = next(a for a, v in doc["activity"].items()
                  if v.get("scpilot:decision_type") == "clustering_resolution")
    used_targets = {u["prov:entity"] for u in doc["used"].values() if u["prov:activity"] == dec_id}
    assert ev_id in used_targets, "decision must be joined to its evidence by recipe_hash"

    # the evidence collection HAS the artifact as a member, and the artifact's identity IS its sha256
    members = [m["prov:entity"] for m in doc["hadMember"].values()
               if m["prov:collection"] == ev_id]
    assert members, "evidence collection must have the artifact member"
    art_id = members[0]
    sha = doc["entity"][art_id]["scpilot:sha256"]
    assert sha and doc["entity"][art_id]["scpilot:sha256_recorded"] is True
    assert art_id == f"entity:artifact/{sha}"               # sha256 is the artifact identity
    # decision --used--> evidence(recipe_hash) --hadMember--> artifact(sha256): the full traversal
    assert art_path.name in doc["entity"][art_id]["scpilot:path"]


def test_deterministic_tool_decision_has_no_fabricated_agent(tmp_path):
    # (d) a deterministic-tool decision (model_id=None) produces NO model Agent and NO association.
    s, rh, _ = _built_session(tmp_path)
    tools.run("export_audit", s)
    doc = json.loads((s.out / "audit_bundle.json").read_text())

    # no agent has a null/empty model id (never fabricated)
    assert all(v.get("scpilot:model_id") for v in doc["agent"].values())
    # the only LLM agent is the recorded "test-model"; the reviewer "reviewer-x" is also an agent
    assert "agent:model/test-model" in doc["agent"]
    assert "agent:model/reviewer-x" in doc["agent"]

    # the deterministic cnv_reference decision exists but has NO wasAssociatedWith edge
    det_id = next(a for a, v in doc["activity"].items()
                  if v.get("scpilot:decision_type") == "cnv_reference")
    assoc_for_det = [w for w in doc.get("wasAssociatedWith", {}).values()
                     if w["prov:activity"] == det_id]
    assert assoc_for_det == []


def test_tier4_verdict_attached_to_reviewer_agent(tmp_path):
    s, rh, _ = _built_session(tmp_path)
    tools.run("export_audit", s)
    doc = json.loads((s.out / "audit_bundle.json").read_text())

    review_id = next(a for a, v in doc["activity"].items()
                     if v.get("prov:type") == "scpilot:Tier4Review")
    v = doc["activity"][review_id]
    assert v["scpilot:reviewer_model"] == "reviewer-x"
    assert json.loads(v["scpilot:verdicts"]) == {"0": "confirmed", "1": "refuted"}
    # the review is associated with the reviewer agent (WHO reviewed)
    rev_assoc = [w for w in doc["wasAssociatedWith"].values() if w["prov:activity"] == review_id]
    assert rev_assoc and rev_assoc[0]["prov:agent"] == "agent:model/reviewer-x"
    assert rev_assoc[0]["prov:role"] == "tier4-reviewer"


def test_export_is_byte_deterministic(tmp_path):
    # (e) re-running the export on the same (unchanged) session yields byte-identical output.
    s, rh, _ = _built_session(tmp_path)
    tools.run("export_audit", s)
    first = (s.out / "audit_bundle.json").read_bytes()
    tools.run("export_audit", s)
    second = (s.out / "audit_bundle.json").read_bytes()
    assert first == second


def test_no_hash_artifacts_same_basename_do_not_collide(tmp_path):
    # Regression (Improvement ② review, Minor #1): two artifacts across DIFFERENT steps that both
    # lack a recorded sha256 (files absent on disk) and share a basename must map to DISTINCT
    # entities — keying on basename alone would collapse them and misattribute the graph.
    inp = tmp_path / "input.h5ad"
    _tiny_adata().write_h5ad(inp)
    s = Session.create(tmp_path / "sess_nohash", input_path=str(inp))
    s.load_input()
    for i in range(2):
        missing = s.artifacts_dir / f"missing_dir_{i}" / "de.csv"   # same basename, different path, absent
        res = S.success("markers", summary={"step": i},
                        artifacts=[S.artifact_csv(str(missing), n_rows=1, n_cols=1, description="absent")])
        s.record_run(res, params={"step": i}, seed=0)
    tools.run("export_audit", s)
    doc = json.loads((s.out / "audit_bundle.json").read_text())

    nohash = [e for e in doc["entity"] if e.startswith("entity:artifact/nohash/")]
    assert len(nohash) == 2, f"expected 2 distinct no-hash artifact entities, got {nohash}"
    for e in nohash:
        assert doc["entity"][e]["scpilot:sha256_recorded"] is False   # honest: no fabricated hash
    # each artifact is a member of its OWN step's evidence collection (no cross-step misattribution)
    members = [m["prov:entity"] for m in doc["hadMember"].values()]
    assert sorted(members) == sorted(nohash)
