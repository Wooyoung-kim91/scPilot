"""Offline-safe tests for the mode-2 LLM layer (plan D1-D5).

No real API key, no network: a scripted FakeProvider returns a fixed sequence of tool
calls, driving a tiny synthetic h5ad through real registry tools. Asserts that:
- the agent runs tools, logs token/tool-call stats,
- run-log + decision events are written (frozen schema),
- forced structured output (emit_de_design) is recorded as a decision + artifact,
- and `scpilot replay` reproduces the mode-2 session deterministically (no LLM).
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot import tools
from scpilot.llm.agent import build_tool_schemas, run_agent
from scpilot.llm.provider import (LLMResponse, ProviderConfig, ToolCall,
                                   build_provider, probe_backend)
from scpilot.session import Session


# --------------------------------------------------------------------------- #
# tiny synthetic data (two latent groups so clustering finds structure)
# --------------------------------------------------------------------------- #
def _raw(n_obs=240, n_vars=150):
    rng = np.random.default_rng(0)
    base = rng.poisson(0.5, (n_obs, n_vars)).astype("float32")
    half = n_obs // 2
    base[:half, :30] += rng.poisson(4.0, (half, 30)).astype("float32")
    base[half:, 30:60] += rng.poisson(4.0, (n_obs - half, 30)).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(base))
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    a.var_names = [f"G{i}" for i in range(n_vars)]
    a.layers["counts"] = a.X.copy()
    a.obs["sample_id"] = rng.choice(["s1", "s2", "s3"], n_obs)
    return a


def _session(tmp_path):
    a = _raw()
    p = tmp_path / "in.h5ad"
    a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p))
    s.load_input()
    return s


# --------------------------------------------------------------------------- #
# scripted fake provider (no SDK, no network)
# --------------------------------------------------------------------------- #
class FakeProvider:
    """A Provider stand-in that returns a pre-scripted list of LLMResponses."""

    def __init__(self, script):
        self._script = list(script)
        self.name = "fake"
        self.model = "fake-model"
        self.config = ProviderConfig(backend="fake", model="fake-model")
        self.calls = 0

    def complete(self, messages, *, tools=None, system=None, tool_choice=None, max_tokens=None):
        self.calls += 1
        if self._script:
            return self._script.pop(0)
        return LLMResponse(text="done", tool_calls=[], stop_reason="end_turn",
                           usage={"input_tokens": 5, "output_tokens": 5})

    @staticmethod
    def tool_result_message(call, content):
        return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}


def _tc(i, name, args):
    return ToolCall(id=f"call_{i}", name=name, arguments=args)


def _resp(*calls):
    return LLMResponse(text="reasoning", tool_calls=list(calls), stop_reason="tool_use",
                       usage={"input_tokens": 100, "output_tokens": 40})


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_build_tool_schemas_includes_emit_tools():
    names = {s["name"] for s in build_tool_schemas()}
    assert {"preprocess", "cluster", "markers"} <= names
    assert {"emit_annotation_labels", "emit_de_design"} <= names
    # forced-structured schemas carry required keys
    de = next(s for s in build_tool_schemas() if s["name"] == "emit_de_design")
    assert "method" in de["input_schema"]["required"]


def test_invalid_param_is_recoverable_not_fatal(tmp_path):
    # F2: a model-proposed out-of-range param (res_step=0 → would div-by-zero) must come back as a
    # RECOVERABLE invalid_params error, the loop survives, and a corrected retry succeeds.
    s = _session(tmp_path)
    script = [
        _resp(_tc(1, "preprocess", {"n_top_genes": 80, "n_pcs": 20})),
        _resp(_tc(2, "cluster_sweep", {"res_min": 0.1, "res_max": 0.3, "res_step": 0})),  # bad
        _resp(_tc(3, "cluster_sweep", {"res_min": 0.1, "res_max": 0.3, "res_step": 0.1})),  # fixed
        LLMResponse(text="done", tool_calls=[], stop_reason="end_turn", usage={}),
    ]
    res = run_agent(s, FakeProvider(script), seed=0, max_iters=20)
    assert res.stopped_reason == "completed"
    # the bad call was NOT logged as a successful run; the corrected one ran
    runs = [json.loads(l) for l in s.run_log_path.read_text().splitlines() if l.strip()]
    sweeps = [r for r in runs if r["tool"] == "cluster_sweep" and r.get("status") == "success"]
    assert sweeps and all(r["params"]["res_step"] == 0.1 for r in sweeps)


def test_max_iters_is_clamped(tmp_path):
    # F9: a non-positive max_iters must be clamped to >=1 (not skip the loop), and the run proceeds.
    s = _session(tmp_path)
    script = [LLMResponse(text="done", tool_calls=[], stop_reason="end_turn", usage={})]
    res = run_agent(s, FakeProvider(script), seed=0, max_iters=0)
    assert res.stats.llm_turns >= 1


def test_tool_output_wrapped_as_untrusted(tmp_path):
    # F4: tool results handed back to the model are wrapped in the untrusted-data envelope.
    captured = {}

    class CapturingProvider(FakeProvider):
        def tool_result_message(self, call, content):     # noqa: D401 — capture wrapper
            captured["content"] = content
            return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}

    s = _session(tmp_path)
    script = [
        _resp(_tc(1, "detect_state", {})),
        LLMResponse(text="done", tool_calls=[], stop_reason="end_turn", usage={}),
    ]
    run_agent(s, CapturingProvider(script), seed=0, max_iters=20)
    assert captured["content"].startswith("<tool_output_data>")
    assert captured["content"].rstrip().endswith("</tool_output_data>")


def test_agent_drives_tools_logs_and_stats(tmp_path):
    s = _session(tmp_path)
    script = [
        _resp(_tc(1, "detect_state", {})),
        _resp(_tc(2, "preprocess", {"n_top_genes": 80, "n_pcs": 20})),
        _resp(_tc(3, "cluster", {"resolution": 0.5})),
        _resp(_tc(4, "markers", {"n_genes": 10})),
        _resp(_tc(5, "emit_de_design", {
            "method": "pseudobulk", "comparison_axis": "leiden", "group_key": "leiden",
            "groups": ["0", "1"], "rationale": "two planted groups"})),
        LLMResponse(text="Analysis complete.", tool_calls=[], stop_reason="end_turn",
                    usage={"input_tokens": 10, "output_tokens": 8}),
    ]
    res = run_agent(s, FakeProvider(script), goal="cluster the data", seed=0, max_iters=20)

    assert res.final_text == "Analysis complete."
    st = res.stats.to_dict()
    assert st["tool_calls"] == 5
    assert st["tool_calls_by_name"]["preprocess"] == 1
    assert st["total_tokens"] > 0
    assert st["decisions_logged"] >= 2          # preprocess(hvg_npcs)+cluster(res)+de_design

    # run-log written for the mutating tools (replayable recipe)
    runs = [json.loads(l) for l in s.run_log_path.read_text().splitlines() if l.strip()]
    logged = {r["tool"] for r in runs}
    assert {"preprocess", "cluster", "markers"} <= logged

    # decision events written + valid against the frozen schema
    decs = [json.loads(l) for l in s.decisions_path.read_text().splitlines() if l.strip()]
    dtypes = {d["decision_type"] for d in decs}
    assert "hvg_npcs" in dtypes and "clustering_resolution" in dtypes
    assert "de_design" in dtypes               # forced structured output recorded

    # forced-structured artifact written
    assert (s.artifacts_dir / "de_design.json").exists()
    payload = json.loads((s.artifacts_dir / "de_design.json").read_text())
    assert payload["method"] == "pseudobulk"


def _compartmented_session(tmp_path):
    import anndata as ad
    from scipy import sparse
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(40)]
    a = ad.AnnData(sparse.csr_matrix(rng.poisson(0.5, (300, 40)).astype("float32")))
    a.var_names = genes
    a.layers["counts"] = a.X.copy()
    comp = np.array(["Epithelial"] * 180 + ["T_NK"] * 120)
    a.obs["major_cell_type"] = comp
    a.obs["GSE"] = rng.choice(["GSEa", "GSEb"], 300)
    a.obs["sample_id"] = rng.choice(["s1", "s2", "s3"], 300)
    a.obsm["X_scVI"] = rng.standard_normal((300, 10)).astype("float32")
    p = tmp_path / "in.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    return s


def test_mode2_routes_subset_steps_to_child_session(tmp_path):
    # mode-2 child-session routing: compartment_subset spawns a child, and the SUBSEQUENT cluster/
    # markers land in THAT child — not the parent. The parent stays at its broad-annotation state.
    s = _compartmented_session(tmp_path)
    script = [
        _resp(_tc(1, "compartment_subset", {"compartment": "T_NK", "mode": "clustering",
                                            "use_rep": "X_scVI"})),
        _resp(_tc(2, "cluster", {"use_rep": "X_scVI", "resolution": 0.5})),
        _resp(_tc(3, "markers", {"groupby": "leiden_scvi", "n_genes": 10})),
        LLMResponse(text="done", tool_calls=[], stop_reason="end_turn", usage={}),
    ]
    run_agent(s, FakeProvider(script), seed=0, max_iters=20, review=False)

    # parent untouched: full cell count, no clustering written on the parent by the subset steps
    assert s.adata.n_obs == 300 and "leiden_scvi" not in s.adata.obs
    # the child session exists and carries the subset + its clustering/markers (X_scVI → _scvi keys)
    child = Session.open(str(s.out / "compartments" / "T_NK"))
    assert child.adata.n_obs == 120
    assert "leiden_scvi" in child.adata.obs and "rank_genes_groups" in child.adata.uns
    stages = {c["stage"] for c in child.manifest.checkpoints}
    assert "cluster" in stages                          # cluster ran IN the child session
    # the child's run-log (not the parent's) recorded the subset steps
    child_runs = {json.loads(l)["tool"] for l in child.run_log_path.read_text().splitlines() if l.strip()}
    assert {"cluster", "markers"} <= child_runs


def test_mode2_session_replays_deterministically(tmp_path):
    s = _session(tmp_path)
    script = [
        _resp(_tc(1, "preprocess", {"n_top_genes": 80, "n_pcs": 20})),
        _resp(_tc(2, "cluster", {"resolution": 0.5})),
        _resp(_tc(3, "markers", {"n_genes": 10})),
        LLMResponse(text="done", tool_calls=[], stop_reason="end_turn", usage={}),
    ]
    run_agent(s, FakeProvider(script), seed=0, max_iters=20)

    # replay through the registry executor (no LLM) on a fresh session
    from scpilot.repro import replay_session, set_global_seed
    set_global_seed(0)
    replay_dir = tmp_path / "replay"
    rsess = Session.create(replay_dir, input_path=s.manifest.input["path"])
    executor = tools.make_replay_executor(rsess)
    report = replay_session(str(s.out), executor=executor)

    assert report["mode"] == "executed"
    assert report["all_match"] is True, report
    # every successfully-logged step was diffed and matched within grade tolerance
    diffed = [st for st in report["steps"] if "diff" in st]
    assert diffed and all(st["diff"]["match"] for st in diffed)


def test_report_tool_assembles_markdown(tmp_path):
    s = _session(tmp_path)
    # drive via the agent so the run log (which report reads) is populated
    script = [
        _resp(_tc(1, "preprocess", {"n_top_genes": 80, "n_pcs": 20})),
        _resp(_tc(2, "cluster", {"resolution": 0.5})),
        LLMResponse(text="done", tool_calls=[], stop_reason="end_turn", usage={}),
    ]
    run_agent(s, FakeProvider(script), seed=0, max_iters=20)
    rep = tools.run("report", s, interpretation="Two clusters were found.")
    assert rep.status == "success"
    md = next(a.path for a in rep.artifacts if a.path.endswith("report.md"))
    assert Path(md).exists()
    text = Path(md).read_text()
    assert "Two clusters were found." in text
    assert "preprocess" in text and "cluster" in text
    assert (s.artifacts_dir / "report.json").exists()


def test_mode2_writes_reasoning_log_and_outputs(tmp_path):
    # mode-2 now routes through record_tool_run → it must produce the reasoning narrative
    # AND the per-step outputs index (artifacts + reasoning), like mode-1/3 (plan R1/R3).
    s = _session(tmp_path)
    script = [
        _resp(_tc(1, "preprocess", {"n_top_genes": 80, "n_pcs": 20})),
        _resp(_tc(2, "cluster", {"resolution": 0.5})),
        LLMResponse(text="done", tool_calls=[], stop_reason="end_turn", usage={}),
    ]
    run_agent(s, FakeProvider(script), seed=0, max_iters=20)

    # reasoning narrative written (was absent in mode-2 before)
    assert s.reasoning_log_path.exists()
    rlog = s.reasoning_log_path.read_text()
    assert "preprocess" in rlog and "cluster" in rlog

    # per-step outputs index written, with the model's prose bound as the WHY
    orecs = [json.loads(l) for l in s.outputs_path.read_text().splitlines() if l.strip()]
    tools_logged = {r["tool"] for r in orecs}
    assert {"preprocess", "cluster"} <= tools_logged
    assert any(r.get("reasoning") == "reasoning" for r in orecs)   # _resp() prose captured


def test_param_overrides_fix_tool_args(tmp_path):
    # the user pre-fixes cluster resolution=0.7; the model proposes 0.3 — the FIXED value must win
    # and be recorded (human-in-the-loop param preset).
    s = _session(tmp_path)
    script = [
        _resp(_tc(1, "preprocess", {"n_top_genes": 80, "n_pcs": 20})),
        _resp(_tc(2, "cluster", {"resolution": 0.3})),
        LLMResponse(text="done", tool_calls=[], stop_reason="end_turn", usage={}),
    ]
    run_agent(s, FakeProvider(script), seed=0, max_iters=20,
              param_overrides={"cluster": {"resolution": 0.7}})
    runs = [json.loads(l) for l in s.run_log_path.read_text().splitlines() if l.strip()]
    cl = next(r for r in runs if r["tool"] == "cluster")
    assert cl["params"]["resolution"] == 0.7        # user-fixed override won over the model's 0.3


def test_agent_signals_max_iters_when_never_stops(tmp_path):
    # model keeps calling a tool forever → loop must stop and report it (not silently)
    s = _session(tmp_path)
    forever = [_resp(_tc(i, "detect_state", {})) for i in range(10)]
    res = run_agent(s, FakeProvider(forever), seed=0, max_iters=3)
    assert res.stopped_reason == "max_iters"
    assert res.stats.llm_turns == 3


def test_anthropic_groups_consecutive_tool_results():
    # multiple tool results for one assistant turn must collapse into ONE user message
    from scpilot.llm.provider import _to_anthropic_messages
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant_tool_calls", "text": "calling",
         "tool_calls": [ToolCall("a", "t1", {}), ToolCall("b", "t2", {})]},
        {"role": "tool", "tool_call_id": "a", "name": "t1", "content": "r1"},
        {"role": "tool", "tool_call_id": "b", "name": "t2", "content": "r2"},
    ]
    out = _to_anthropic_messages(msgs)
    tool_msgs = [m for m in out if m["role"] == "user" and isinstance(m["content"], list)
                 and m["content"] and m["content"][0].get("type") == "tool_result"]
    assert len(tool_msgs) == 1
    assert [b["tool_use_id"] for b in tool_msgs[0]["content"]] == ["a", "b"]


def test_openai_response_synthesizes_missing_tool_call_id():
    from types import SimpleNamespace
    from scpilot.llm.provider import _from_openai_response
    fn = SimpleNamespace(name="cluster", arguments='{"resolution": 0.5}')
    tc = SimpleNamespace(id=None, function=fn)               # local server omitted the id
    msg = SimpleNamespace(content="", tool_calls=[tc])
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
                           usage=None)
    parsed = _from_openai_response(resp)
    assert parsed.tool_calls[0].id == "call_0"               # deterministic fallback
    assert parsed.tool_calls[0].arguments == {"resolution": 0.5}


def test_provider_config_from_env_and_local_backend(monkeypatch):
    # default backend = anthropic, default model name comes from config (not hardcoded in logic)
    monkeypatch.delenv("SCPILOT_LLM_BACKEND", raising=False)
    monkeypatch.delenv("SCPILOT_LLM_MODEL", raising=False)
    cfg = ProviderConfig.from_env()
    assert cfg.backend == "anthropic"
    assert cfg.resolved_model() == "claude-opus-4-8"

    # point at a local OpenAI-compatible endpoint via env
    monkeypatch.setenv("SCPILOT_LLM_BACKEND", "openai")
    monkeypatch.setenv("SCPILOT_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("SCPILOT_LLM_MODEL", "llama3.1")
    cfg2 = ProviderConfig.from_env()
    assert cfg2.backend == "openai"
    assert cfg2.base_url == "http://localhost:11434/v1"
    assert cfg2.resolved_model() == "llama3.1"


def test_unknown_backend_raises_unavailable(monkeypatch):
    from scpilot.llm.provider import ProviderUnavailable
    import pytest
    monkeypatch.setenv("SCPILOT_LLM_BACKEND", "nope")
    with pytest.raises(ProviderUnavailable):
        build_provider()


def test_probe_backend_is_nonfatal(monkeypatch):
    monkeypatch.setenv("SCPILOT_LLM_BACKEND", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SCPILOT_LLM_API_KEY", raising=False)
    p = probe_backend()
    assert p["backend"] == "anthropic"
    assert p["ready"] is False           # no key -> not ready, but no exception
    assert "reason" in p


def test_anthropic_prompt_caching_shaping():
    """Cost: the Anthropic backend marks the last message (and system) with an ephemeral cache
    breakpoint so the re-sent tools+system+history prefix is a cache hit, not full-price input."""
    from scpilot.llm import provider as P

    # string content -> promoted to a text block carrying cache_control
    m = [{"role": "user", "content": "hi"}]
    P._mark_last_cache(m)
    assert m[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # tool_result block list -> cache_control on the last block
    m2 = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "y"}]}]
    P._mark_last_cache(m2)
    assert m2[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    P._mark_last_cache([])   # empty -> no-op, no crash


# --------------------------------------------------------------------------- #
# Improvement ①: LLM decision provenance (schemas + prompt hash + write site)
# --------------------------------------------------------------------------- #
def test_decision_event_roundtrips_with_provenance_fields():
    # a v2 DecisionEvent carrying the new provenance fields serializes + validates cleanly.
    from scpilot import schemas as S

    ev = S.DecisionEvent(
        decision_type="hvg_npcs", choice={"n_pcs": 20}, candidates=[],
        rationale="elbow at 20", model_id="claude-opus-4-8", prompt_version="1.0",
        prompt_hash="deadbeefcafef00d", temperature=0.0, recipe_hash="0123456789abcdef",
        alternatives_rejected=[{"n_pcs": 30, "why": "past elbow"}])
    d = ev.to_dict()
    assert d["schema_version"] == S.DECISION_SCHEMA_VERSION == 2
    assert d["model_id"] == "claude-opus-4-8" and d["recipe_hash"] == "0123456789abcdef"
    assert d["prompt_version"] == "1.0" and d["prompt_hash"] == "deadbeefcafef00d"
    assert d["temperature"] == 0.0
    assert d["alternatives_rejected"][0]["n_pcs"] == 30
    assert S.validate_decision(d) == []


def test_old_shape_decision_still_validates_and_loads():
    # a v1 event (only the frozen required keys, no provenance / schema_version) must still be
    # accepted so pre-existing decisions.jsonl files keep validating + replaying.
    from scpilot import schemas as S

    old = {"decision_type": "clustering_resolution", "choice": {"resolution": 0.25},
           "candidates": [], "rationale": "knee at 0.25"}
    assert S.validate_decision(old) == []
    # constructs from the v1 keys with provenance defaulting to None (backward compatible)
    ev = S.DecisionEvent(**old)
    assert ev.model_id is None and ev.recipe_hash is None and ev.temperature is None
    assert ev.schema_version == S.DECISION_SCHEMA_VERSION
    # a malformed provenance value (wrong type) is the ONLY thing rejected
    assert "recipe_hash must be a string or null" in S.validate_decision({**old, "recipe_hash": 123})


def test_prompt_hash_is_stable_and_deterministic():
    from scpilot.llm import prompts

    assert prompts.prompt_hash("same text") == prompts.prompt_hash("same text")
    assert prompts.prompt_hash("a") != prompts.prompt_hash("b")
    assert isinstance(prompts.PROMPT_VERSION, str) and prompts.PROMPT_VERSION
    # the orchestration system prompt hashes to a stable 16-char hex key
    h = prompts.prompt_hash(prompts.ORCHESTRATION_PROMPT)
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)


def test_write_site_populates_provenance_and_recipe_hash_join_key(tmp_path):
    # the mode-2 write site stamps the resolved model id + prompt version/hash + temperature on the
    # decision event, and its recipe_hash EQUALS the join key of the same step's run-log record.
    from scpilot.llm import prompts

    s = _session(tmp_path)
    script = [
        _resp(_tc(1, "preprocess", {"n_top_genes": 80, "n_pcs": 20})),
        LLMResponse(text="done", tool_calls=[], stop_reason="end_turn", usage={}),
    ]
    # temperature flows from the provider config (provider.py resolution) onto the event
    prov = FakeProvider(script)
    prov.config = ProviderConfig(backend="fake", model="fake-model", temperature=0.0)
    run_agent(s, prov, seed=0, max_iters=20, review=False)

    decs = [json.loads(l) for l in s.decisions_path.read_text().splitlines() if l.strip()]
    hvg = next(d for d in decs if d["decision_type"] == "hvg_npcs")
    assert hvg["model_id"] == "fake-model"
    assert hvg["prompt_version"] == prompts.PROMPT_VERSION
    assert hvg["prompt_hash"] and len(hvg["prompt_hash"]) == 16
    assert hvg["temperature"] == 0.0
    assert hvg["schema_version"] == 2

    # recipe_hash is the SAME value the preprocess run-log record carries (the evidence join key)
    runs = [json.loads(l) for l in s.run_log_path.read_text().splitlines() if l.strip()]
    pre_run = next(r for r in runs if r["tool"] == "preprocess" and r["status"] == "success")
    assert hvg["recipe_hash"] and hvg["recipe_hash"] == pre_run["recipe_hash"]


def test_in_tool_llm_label_call_is_provenance_stamped(tmp_path):
    # Follow-up #6: the highest-value LLM decision — the tier1 cell-type CALL logged from INSIDE
    # apply_annotation — carries WHO decided + on WHAT prompt basis when routed through the agent,
    # while a DETERMINISTIC in-tool decision (finalize_annotation) keeps provenance None (honest).
    from scpilot.llm import prompts

    s = _session(tmp_path)
    script = [
        _resp(_tc(1, "preprocess", {"n_top_genes": 80, "n_pcs": 20})),
        _resp(_tc(2, "cluster", {"resolution": 0.5})),
        _resp(_tc(3, "apply_annotation",
                  {"groupby": "leiden", "labels": {"0": "T cell", "1": "Epithelial cell"}})),
        _resp(_tc(4, "finalize_annotation", {})),
        LLMResponse(text="done", tool_calls=[], stop_reason="end_turn", usage={}),
    ]
    prov = FakeProvider(script)
    prov.config = ProviderConfig(backend="fake", model="fake-model", temperature=0.0)
    # review=False → no Tier-4 reviewer provider needed; keeps this test provider-only.
    run_agent(s, prov, seed=0, max_iters=20, review=False)

    decs = [json.loads(l) for l in s.decisions_path.read_text().splitlines() if l.strip()]

    # LLM-DRIVEN in-tool CALL → provenance stamped (same field values as the ① orchestrator path)
    call = next(d for d in decs if d["decision_type"] == "tier1_llm_labels")
    assert call["model_id"] == "fake-model"
    assert call["prompt_version"] == prompts.PROMPT_VERSION
    assert call["prompt_hash"] and len(call["prompt_hash"]) == 16
    assert call["temperature"] == 0.0
    assert call["schema_version"] == 2

    # DETERMINISTIC in-tool decision → provenance intentionally None (not fabricated)
    fin = next(d for d in decs if d["decision_type"] == "annotation_finalized")
    assert fin["model_id"] is None and fin["prompt_hash"] is None
    assert fin["prompt_version"] is None and fin["temperature"] is None

    # EXACTLY ONE provenance-bearing decision per LLM label call (no double-stamp: apply_annotation
    # is NOT in _DECISION_TYPE, so the wrapper does not emit a second event for it).
    assert sum(1 for d in decs if d["decision_type"] == "tier1_llm_labels") == 1
    stamped_for_apply = [d for d in decs if d.get("stage") == "apply_annotation"
                         and d.get("model_id") is not None]
    assert len(stamped_for_apply) == 1

    # provenance is NON-replayable: it must NOT leak into the run-log params / recipe (①'s contract).
    runs = [json.loads(l) for l in s.run_log_path.read_text().splitlines() if l.strip()]
    apply_run = next(r for r in runs if r["tool"] == "apply_annotation" and r["status"] == "success")
    for k in ("model_id", "prompt_version", "prompt_hash", "temperature"):
        assert k not in apply_run["params"]

    # all decisions (v2 + the deterministic ones) still validate against the frozen schema
    from scpilot import schemas as S
    assert all(S.validate_decision(d) == [] for d in decs)


def test_direct_in_tool_label_call_leaves_provenance_none(tmp_path):
    # Follow-up #6: a DIRECT (non-agent) apply_annotation call — mode-1 host / CLI / replay — has no
    # provider, so the same in-tool tier1_llm_labels event honestly records provenance as None.
    from scpilot import schemas as S
    from scpilot.core import annotate

    s = _session(tmp_path)
    tools.run("preprocess", s, n_top_genes=80, n_pcs=20)
    tools.run("cluster", s, resolution=0.5)
    res = annotate.apply_annotation(s, groupby="leiden",
                                    labels={"0": "T cell", "1": "Epithelial cell"})
    assert res.status == "success"

    decs = [json.loads(l) for l in s.decisions_path.read_text().splitlines() if l.strip()]
    call = next(d for d in decs if d["decision_type"] == "tier1_llm_labels")
    assert call["model_id"] is None and call["prompt_hash"] is None
    assert call["prompt_version"] is None and call["temperature"] is None
    assert S.validate_decision(call) == []
