"""Machine-readable audit export — W3C PROV-JSON provenance bundle (Improvement ②).

Provenance in scpilot is written across five append-only artifacts (``run_log.jsonl`` /
``decisions.jsonl`` / ``outputs.jsonl`` / ``reasoning_log.md`` / ``uns['scpilot*']`` / Tier-4
verdicts). This tool assembles them into ONE citable, machine-readable graph that links
``evidence → decision → reviewer-verdict → artifact(+sha256)`` so an external auditor consumes a
single file instead of parsing the ad-hoc logs. It is a THIN assembler over what the harness
already recorded: it reads only the session's on-disk records (``session.json`` manifest +
``run_log.jsonl`` + ``decisions.jsonl`` + ``outputs.jsonl``) and NEVER loads or mutates the
AnnData / checkpoints (read-only).

FORMAT: **W3C PROV-JSON** (https://www.w3.org/submissions/prov-json/). Node/edge mapping:

  ``prov:Agent``    ← the LLM model that made a decision (``DecisionEvent.model_id``), the Tier-4
                      reviewer (``reviewer_model``), and any model declared in the manifest
                      ``llm_topology``. Created ONLY when a model id is actually recorded — a
                      deterministic-tool decision (``model_id`` is ``None``) yields NO agent
                      (honest: no fabricated model).
  ``prov:Activity`` ← (a) each tool execution (``RunLogRecord``)        → ``activity:run/<i>/<tool>``
                      (b) each LLM decision (``DecisionEvent``)          → ``activity:decision/<i>/<type>``
                      (c) each Tier-4 review (``annotation_audit`` dec.) → ``activity:review/<i>``
  ``prov:Entity``   ← (a) per-step evidence bundle (``OutputRecord``), a ``prov:Collection`` keyed
                          by ``recipe_hash``                            → ``entity:evidence/<recipe_hash>``
                      (b) each output artifact WITH its sha256 identity → ``entity:artifact/<sha256>``
                      (c) each checkpoint (manifest)                    → ``entity:checkpoint/<id>``

  Relations (the graph LINKS these, it does not list them side by side):
    ``used(decision, evidence)``           — the **recipe_hash** JOIN: a decision USED the exact
                                             evidence bundle carrying the SAME ``recipe_hash``.
    ``wasGeneratedBy(evidence, run)``      — the evidence bundle was produced by its tool run.
    ``hadMember(evidence, artifact)``      — the artifact (**sha256** identity) is a member of the
                                             step's evidence collection.
    ``wasGeneratedBy(artifact, run)``      — the artifact was produced by its tool run.
    ``wasGeneratedBy(checkpoint, run)``    — the checkpoint was produced by its tool run.
    ``wasAssociatedWith(decision, model)`` — WHO decided (the LLM agent), when a model is recorded.
    ``wasAssociatedWith(review, reviewer)``— WHO reviewed (the Tier-4 reviewer agent).

  So ``decision --used--> evidence(recipe_hash) --hadMember--> artifact(sha256)``: a single
  traversal links a decision to BOTH its evidence (by ``recipe_hash``) and its artifact (by
  ``sha256``). Honest representation of gaps: a decision with no ``recipe_hash`` (emit-tool
  decisions) simply gets no ``used`` edge; a deterministic tool (``model_id`` is ``None``) gets no
  agent; an artifact whose ``sha256`` was not recorded is emitted with
  ``scpilot:sha256_recorded = false`` rather than a fabricated identity.

DETERMINISM: the bundle is a pure function of the on-disk records — node/edge maps are emitted
with ``sort_keys=True``, sha256 values are READ from the records (never re-hashed), and no
wall-clock is injected (only the ``ts`` timestamps already present in the records). Re-running the
export on an unchanged session yields byte-identical JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from scpilot import schemas as S
from scpilot.tools import register

# Bundle self-version (independent of the DecisionEvent schema version): bumped only if the
# node/edge mapping above changes. The PROV format tag pins the serialization contract.
BUNDLE_SCHEMA_VERSION = 1
PROV_FORMAT = "W3C PROV-JSON"

_PREFIX = {
    "prov": "http://www.w3.org/ns/prov#",
    "scpilot": "https://scpilot.dev/prov#",
}

# Tier-4 reviewer verdicts are logged as a DecisionEvent of this type (see apply_annotation_audit).
_TIER4_DECISION_TYPE = "annotation_audit"

# Closed-model honesty note carried in the header — the model's weights/outputs are not
# reproducible; only the recorded identifiers (model id / prompt hash / temperature) are captured.
_CLOSED_MODEL_NOTE = (
    "Closed-model LLM provenance is captured AS RECORDED: only the model identifier, prompt hash, "
    "prompt version and temperature from the decision log are represented as the agent. The model's "
    "internal weights and generations are not reproducible and are not fabricated here."
)


def _read_jsonl(path: Path) -> list[dict]:
    """Records IN FILE ORDER from an append-only jsonl (skips blank/malformed lines)."""
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001 — a garbled line must not abort the export
            continue
    return out


def _lit(value) -> str:
    """A deterministic JSON-string literal for a PROV attribute value (objects → sorted JSON)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def build_bundle(session) -> dict:
    """Assemble the PROV-JSON bundle dict from the session's on-disk records (pure/read-only)."""
    man = session.manifest
    runs = _read_jsonl(session.run_log_path)
    outputs = _read_jsonl(session.outputs_path)
    decisions = _read_jsonl(session.decisions_path)

    entity: dict[str, dict] = {}
    activity: dict[str, dict] = {}
    agent: dict[str, dict] = {}
    used: dict[str, dict] = {}
    was_generated_by: dict[str, dict] = {}
    was_associated_with: dict[str, dict] = {}
    had_member: dict[str, dict] = {}

    # ---- agents: one per distinct recorded model id; NEVER fabricated -----------------------
    _agent_roles: dict[str, set] = {}

    def _agent(model_id, role: str) -> str | None:
        if not model_id:
            return None                                   # deterministic tool / unrecorded → no agent
        aid = f"agent:model/{model_id}"
        if aid not in agent:
            agent[aid] = {"prov:type": "prov:SoftwareAgent", "scpilot:model_id": str(model_id)}
            _agent_roles[aid] = set()
        _agent_roles[aid].add(role)
        return aid

    # models declared in the run topology are genuinely recorded provenance — include them
    topo = man.llm_topology or {}
    for trole, entry in sorted(topo.items()):
        if isinstance(entry, dict) and entry.get("model"):
            _agent(entry["model"], f"topology:{trole}")

    # ---- run activities (one per RunLogRecord, in file order) --------------------------------
    run_ids: list[str] = []
    ckpt_path_to_run: dict[str, str] = {}
    for i, r in enumerate(runs):
        rid = f"activity:run/{i:04d}/{r.get('tool')}"
        run_ids.append(rid)
        activity[rid] = {
            "prov:type": "scpilot:ToolExecution",
            "scpilot:tool": r.get("tool"),
            "scpilot:stage": r.get("stage"),
            "scpilot:status": r.get("status"),
            "scpilot:recipe_hash": r.get("recipe_hash"),
            "scpilot:seed": r.get("seed"),
            "scpilot:determinism_grade": r.get("determinism_grade"),
            "scpilot:duration_s": r.get("duration_s"),
            "scpilot:params": _lit(r.get("params", {})),
            "scpilot:ts": r.get("ts"),
        }
        out_cp = r.get("output_checkpoint")
        if out_cp:
            activity[rid]["scpilot:output_checkpoint"] = out_cp
            ckpt_path_to_run[str(out_cp)] = rid

    # ---- checkpoints (manifest) as entities, generated by the run that produced them ---------
    for cp in (man.checkpoints or []):
        cid = f"entity:checkpoint/{cp.get('id')}"
        entity[cid] = {
            "prov:type": "scpilot:Checkpoint",
            "scpilot:stage": cp.get("stage"),
            "scpilot:x_state": cp.get("x_state"),
            "scpilot:path": cp.get("path"),
            "scpilot:fingerprint": _lit(cp.get("fingerprint")),
        }
        rid = ckpt_path_to_run.get(str(cp.get("path")))
        if rid:
            key = f"_:wgb/checkpoint/{cp.get('id')}"
            was_generated_by[key] = {"prov:entity": cid, "prov:activity": rid}

    # ---- evidence bundles (OutputRecord) + artifacts (sha256 identity) -----------------------
    # recipe_hash → evidence entity id: the JOIN a decision uses to reach its exact evidence.
    evidence_by_hash: dict[str, str] = {}
    for j, o in enumerate(outputs):
        rh = o.get("recipe_hash")
        n = o.get("n")
        run_index = (int(n) - 1) if isinstance(n, int) else j     # OutputRecord.n is 1-based run count
        ev_id = f"entity:evidence/{rh}" if rh else f"entity:evidence/idx{j:04d}"
        entity[ev_id] = {
            "prov:type": "prov:Collection",
            "scpilot:kind": "step_evidence",
            "scpilot:tool": o.get("tool"),
            "scpilot:stage": o.get("stage"),
            "scpilot:recipe_hash": rh,
            "scpilot:run_index": run_index,
            "scpilot:seed": o.get("seed"),
            "scpilot:summary": _lit(o.get("summary", {})),
            "scpilot:ts": o.get("ts"),
        }
        if rh:
            evidence_by_hash.setdefault(rh, ev_id)          # first-writer wins (stable/deterministic)
        rid = run_ids[run_index] if 0 <= run_index < len(run_ids) else None
        if rid:
            was_generated_by[f"_:wgb/evidence/{ev_id}"] = {"prov:entity": ev_id, "prov:activity": rid}

        for a in (o.get("artifacts") or []):
            path = a.get("path")
            if not path:
                continue
            meta = a.get("meta") or {}
            sha = meta.get("sha256")
            art_id = f"entity:artifact/{sha}" if sha else f"entity:artifact/nohash/{Path(str(path)).name}"
            entity.setdefault(art_id, {
                "prov:type": "scpilot:Artifact",
                "scpilot:path": path,
                "scpilot:kind": a.get("kind", "other"),
                "scpilot:description": a.get("description", ""),
                "scpilot:sha256": sha,
                "scpilot:sha256_recorded": bool(sha),        # honest: mark un-hashed artifacts
                "scpilot:bytes": meta.get("bytes"),
            })
            # artifact is a MEMBER of the step's evidence collection, and was generated by the run
            had_member[f"_:member/{ev_id}/{art_id}"] = {"prov:collection": ev_id, "prov:entity": art_id}
            if rid:
                was_generated_by[f"_:wgb/artifact/{ev_id}/{art_id}"] = {
                    "prov:entity": art_id, "prov:activity": rid}

    # ---- decisions + Tier-4 reviewer verdicts ------------------------------------------------
    n_tier4 = 0
    for i, d in enumerate(decisions):
        dtype = d.get("decision_type")
        is_review = dtype == _TIER4_DECISION_TYPE
        base_attrs = {
            "scpilot:decision_type": dtype,
            "scpilot:choice": _lit(d.get("choice")),
            "scpilot:rationale": d.get("rationale"),
            "scpilot:confidence": d.get("confidence"),
            "scpilot:stage": d.get("stage"),
            "scpilot:schema_version": d.get("schema_version"),
            "scpilot:recipe_hash": d.get("recipe_hash"),
            "scpilot:model_id": d.get("model_id"),
            "scpilot:prompt_version": d.get("prompt_version"),
            "scpilot:prompt_hash": d.get("prompt_hash"),
            "scpilot:temperature": d.get("temperature"),
            "scpilot:ts": d.get("ts"),
        }
        if is_review:
            n_tier4 += 1
            act_id = f"activity:review/{i:04d}"
            reviewer = (d.get("params") or {}).get("reviewer_model")
            base_attrs["prov:type"] = "scpilot:Tier4Review"
            base_attrs["scpilot:reviewer_model"] = reviewer
            base_attrs["scpilot:verdicts"] = _lit(d.get("choice"))   # the per-cluster verdict map
            agent_id = _agent(reviewer, "tier4-reviewer") or _agent(d.get("model_id"), "tier4-reviewer")
            role = "tier4-reviewer"
        else:
            act_id = f"activity:decision/{i:04d}/{dtype}"
            base_attrs["prov:type"] = "scpilot:Decision"
            agent_id = _agent(d.get("model_id"), "decision-maker")
            role = "decision-maker"
        activity[act_id] = base_attrs
        # WHO: associate the LLM/reviewer agent, only when a model was recorded (honest for det. tools)
        if agent_id:
            was_associated_with[f"_:waw/{act_id}"] = {
                "prov:activity": act_id, "prov:agent": agent_id, "prov:role": role}
        # recipe_hash JOIN: link the decision to its exact evidence bundle when the key is present
        rh = d.get("recipe_hash")
        if rh and rh in evidence_by_hash:
            used[f"_:used/{act_id}"] = {"prov:activity": act_id, "prov:entity": evidence_by_hash[rh]}

    # finalize agent role lists (sorted, deterministic)
    for aid, roles in _agent_roles.items():
        agent[aid]["scpilot:roles"] = sorted(roles)

    header = {
        "scpilot_version": man.scpilot_version,
        "session_id": man.session_id,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "prov_format": PROV_FORMAT,
        "input": (man.input or {}).get("path"),
        "stage_reached": man.stage,
        "llm_topology": topo or None,
        "closed_model_provenance_note": _CLOSED_MODEL_NOTE,
        "counts": {
            "runs": len(runs), "decisions": len(decisions), "outputs": len(outputs),
            "checkpoints": len(man.checkpoints or []), "tier4_reviews": n_tier4,
            "agents": len(agent), "artifacts": sum(1 for e in entity
                                                    if e.startswith("entity:artifact/")),
        },
    }

    bundle: dict = {
        "scpilot": header,          # small top-level header (non-PROV members are ignored by readers)
        "prefix": _PREFIX,
        "entity": entity,
        "activity": activity,
        "agent": agent,
    }
    # include relation maps only when non-empty (keeps the document clean; all are optional in PROV-JSON)
    for name, rel in (("used", used), ("wasGeneratedBy", was_generated_by),
                      ("wasAssociatedWith", was_associated_with), ("hadMember", had_member)):
        if rel:
            bundle[name] = rel
    return bundle


@register("export_audit", mutating=False,
          description="Assemble a single machine-readable AUDIT BUNDLE (W3C PROV-JSON) for the session, "
                      "linking evidence → decision → Tier-4 reviewer-verdict → artifact(+sha256) into one "
                      "provenance graph. Read-only: assembles ONLY from the recorded run_log/decisions/"
                      "outputs + manifest (never loads/mutates the AnnData or checkpoints). Uses recipe_hash "
                      "as the decision→evidence join key and artifact sha256 as artifact identity; represents "
                      "unrecorded provenance (deterministic-tool decisions, emit-tool decisions) honestly with "
                      "no fabricated model agent. Deterministic (byte-identical on re-run). Writes "
                      "audit_bundle.json in the session dir and returns its path.")
def export_audit(session, *, out_name: str = "audit_bundle.json", **params) -> S.ToolResult:
    t0 = time.time()
    bundle = build_bundle(session)

    out_path = session.out / out_name
    # deterministic serialization: sorted keys, no wall-clock injected beyond the records' own ts.
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True, default=str))

    header = bundle["scpilot"]
    summary = {
        "bundle_path": str(out_path),
        "prov_format": PROV_FORMAT,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "session_id": header["session_id"],
        "n_entities": len(bundle["entity"]),
        "n_activities": len(bundle["activity"]),
        "n_agents": len(bundle["agent"]),
        "n_used": len(bundle.get("used", {})),
        "n_was_generated_by": len(bundle.get("wasGeneratedBy", {})),
        "n_had_member": len(bundle.get("hadMember", {})),
        "n_was_associated_with": len(bundle.get("wasAssociatedWith", {})),
        "counts": header["counts"],
    }
    artifacts = [S.Artifact(path=str(out_path), kind="json",
                            description="W3C PROV-JSON audit bundle (evidence→decision→verdict→artifact)")]
    return S.success("export_audit", summary=summary, artifacts=artifacts,
                     determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=[])
