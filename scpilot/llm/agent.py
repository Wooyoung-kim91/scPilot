"""Mode-2 self-driving agent loop — scpilot plan D4.

The agent is the LLM substitute for a host agent (Claude Code / Codex). It drives
the SAME deterministic tool registry (``scpilot.tools.run``) that mode 1 (MCP) uses,
reading each tool's JSON summary and deciding the next tool + params (the plan's
"summary-in -> decision-out" loop). It works over the backend-neutral ``Provider``
interface (``llm/provider.py``), so the identical control flow runs on Anthropic OR
a local / OpenAI-compatible LLM.

Key responsibilities:
- Expose the registry tools to the model as tool/function schemas.
- Run the call -> execute -> feed-result loop (the generic equivalent of Anthropic's
  ``client.beta.messages.tool_runner``; we keep one loop so both backends behave the same).
- LOG every mutating tool run to ``run_log.jsonl`` (so ``scpilot replay`` reproduces the
  mode-2 run with NO LLM) and log a ``decision`` event for each consequential choice.
- FORCE structured output on the critical steps (annotation labels, DE design) via a
  dedicated ``emit_*`` tool whose JSON Schema is required (plan D4 / structured output).
- Account token usage + tool-call counts for cost (returned in the run result).

Reproducibility note: the agent does NOT inject any LLM-only state into the session
artifacts. Everything replay needs (tool, params, seed, summary, determinism_grade) is
written through ``session.log_run`` exactly as the deterministic ``scpilot step`` path
does — so a mode-2 session replays identically. The CLI also logs the final ``report``
tool run (with the recorded interpretation text) so replay regenerates the report.

KNOWN BOUNDARY (follow-up): the forced structured LLM outputs (``emit_annotation_labels``
/ ``emit_de_design``) are recorded as decision events + JSON artifacts but are NOT
re-derived by ``scpilot replay`` — they are non-deterministic LLM products, so faithful
replay would RESTORE them from the decision log rather than re-query the model. Today
replay reproduces the deterministic TOOL recipe; restoring recorded LLM decisions into a
replayed session is a planned replay-layer extension (decision-event consumption).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from scpilot import schemas as S
from scpilot import tools
from scpilot.llm import prompts
from scpilot.llm.provider import Provider, ToolCall

# Tools the agent is allowed to drive autonomously (registry names). Long-running /
# job-model tools (train_scvi, ingest, benchmark) and raw inspect are excluded from the
# default autonomous set; integration tools are allowed but optional. This is a policy
# list, not a hardcode of behaviour — the registry remains the single source of truth.
DEFAULT_TOOLSET = [
    "detect_state", "qc_metrics", "qc_filter", "preprocess", "cluster",
    "markers", "annotation_review", "apply_annotation", "plots",
    "integrate_scvi", "integrate_harmony",
]

# Per-tool parameter hints surfaced to the model (kept minimal; the registry's ToolSpec
# description carries the semantics). Empty schema => the model may pass any params, which
# the tool validates. We give explicit, decision-relevant knobs for the common steps.
_PARAM_HINTS: dict[str, dict] = {
    "qc_filter": {
        "min_genes": {"type": "integer", "description": "min genes/cell to keep"},
        "max_pct_mt": {"type": "number", "description": "max %% mito to keep"},
        "min_counts": {"type": "integer"},
        "drop_predicted_doublets": {"type": "boolean"},
    },
    "preprocess": {
        "n_top_genes": {"type": "integer", "description": "number of HVGs"},
        "n_pcs": {"type": "integer", "description": "number of principal components"},
    },
    "cluster": {
        "use_rep": {"type": "string", "description": "X_pca | X_harmony | X_scVI"},
        "resolution": {"type": "number", "description": "leiden resolution"},
        "n_neighbors": {"type": "integer"},
    },
    "markers": {"n_genes": {"type": "integer"}},
    "annotation_review": {"groupby": {"type": "string", "description": "cluster key (leiden)"},
                          "top_n": {"type": "integer", "description": "DE genes per cluster to expose"},
                          "tissue": {"type": "string", "description": "tissue/condition — soft prior"}},
    "apply_annotation": {
        "groupby": {"type": "string", "description": "cluster key the labels are keyed on (leiden)"},
        "labels": {"type": "object", "description": "cluster_id -> broad cell type (inferred from DE, NO panel)"},
        "confidence": {"type": "object", "description": "optional cluster_id -> 0..1 confidence"},
        "review_required": {"type": "object", "description": "optional cluster_id -> bool"},
        "tissue": {"type": "string", "description": "tissue/condition context"}},
    "plots": {"kind": {"type": "string", "description": "umap | qc_violin | hvg | pca_variance | dotplot"}},
}

# Decision-event type per tool (which step the agent's choice maps to in the frozen schema).
_DECISION_TYPE: dict[str, str] = {
    "qc_filter": "qc_cutoff",
    "preprocess": "hvg_npcs",
    "cluster": "clustering_resolution",
    # apply_annotation logs its own tier1_llm_labels decision (it owns the label map) → not here.
    "integrate_scvi": "integration_method",
    "integrate_harmony": "integration_method",
}


@dataclass
class RunStats:
    """Token + tool-call accounting for a mode-2 run (cost visibility, plan D5)."""
    llm_turns: int = 0
    tool_calls: int = 0
    tool_calls_by_name: dict = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    decisions_logged: int = 0
    errors: int = 0

    def add_usage(self, usage: dict) -> None:
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)

    def add_call(self, name: str) -> None:
        self.tool_calls += 1
        self.tool_calls_by_name[name] = self.tool_calls_by_name.get(name, 0) + 1

    def to_dict(self) -> dict:
        return {
            "llm_turns": self.llm_turns,
            "tool_calls": self.tool_calls,
            "tool_calls_by_name": dict(self.tool_calls_by_name),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "decisions_logged": self.decisions_logged,
            "errors": self.errors,
        }


@dataclass
class AgentResult:
    final_text: str
    stats: RunStats
    transcript: list = field(default_factory=list)   # list[{role, ...}] for debugging
    stage: str | None = None
    stopped_reason: str = "completed"                 # "completed" | "max_iters"


def build_tool_schemas(toolset: list[str] | None = None) -> list[dict]:
    """Build LLM tool schemas from the registry (single source of tool truth)."""
    names = toolset or DEFAULT_TOOLSET
    schemas: list[dict] = []
    for name in names:
        try:
            spec = tools.get(name)
        except KeyError:
            continue
        props = _PARAM_HINTS.get(name, {})
        schemas.append({
            "name": name,
            "description": spec.description or (spec.fn.__doc__ or "").strip()[:300],
            "input_schema": {"type": "object", "properties": props},
        })
    # Two forced-structured-output emit tools (the model calls these to record the
    # annotation label set / DE design as machine-readable JSON; plan D4).
    schemas.append({
        "name": "emit_annotation_labels",
        "description": "Record the FINAL per-cluster annotation label set (structured). "
                       "Call after annotation_review to commit major/fine/FACS labels, "
                       "malignancy, confidence, and review flags.",
        "input_schema": prompts.ANNOTATION_LABEL_SCHEMA,
    })
    schemas.append({
        "name": "emit_de_design",
        "description": "Record the differential-expression design (structured) before any "
                       "DE test: method, comparison axis, groups, replicate unit, confounders.",
        "input_schema": prompts.DE_DESIGN_SCHEMA,
    })
    return schemas


def _system_prompt(goal: str | None, tissue: str | None = None,
                   resolutions: dict | None = None) -> str:
    parts = [prompts.ORCHESTRATION_PROMPT, prompts.ANNOTATION_PROMPT,
             prompts.ANNOTATION_REVIEW_PROMPT, prompts.TISSUE_CONTEXT_GUIDANCE,
             prompts.DE_DESIGN_PROMPT]
    if resolutions:
        # human-in-the-loop: the ONLY resolutions the agent may use (per embedding/model).
        res = ", ".join(f"{k}={v}" for k, v in resolutions.items())
        parts.insert(0, f"Human-set clustering resolution (use ONLY these; ask if one is missing): {res}\n")
    if tissue:
        parts.insert(0, f"TISSUE / CONTEXT: {tissue}\n")
    if goal:
        parts.insert(0, f"ANALYSIS GOAL: {goal}\n")
    return "\n\n".join(parts)


def _execute_registry_tool(session, name: str, args: dict, seed: int,
                           stats: RunStats, rationale: str = "") -> S.ToolResult:
    """Run one registry tool and LOG it (run-log + decision) so replay reproduces it.

    ``rationale`` is the model's own prose from the tool-call turn — recorded verbatim
    as the decision rationale (we do NOT fabricate candidates/rationale)."""
    spec = tools.get(name)
    result = spec.fn(session, **args)
    stats.errors += 0 if result.status == "success" else 1

    # run-log record — IDENTICAL shape to the deterministic `scpilot step` path, so a
    # mode-2 session replays with tools.make_replay_executor() and NO LLM.
    in_cp = None
    cps = session.manifest.checkpoints
    if len(cps) >= 2 and result.checkpoint:
        in_cp = cps[-2].get("id")
    session.log_run(S.RunLogRecord(
        tool=name, status=result.status, stage=name, params=args,
        summary=result.summary, seed=seed,
        input_checkpoint=in_cp, output_checkpoint=result.checkpoint,
        determinism_grade=result.determinism_grade, error_code=result.error_code,
        duration_s=result.duration_s,
    ).to_dict())

    # decision event for consequential choices (frozen schema; powers audit + replay note)
    dtype = _DECISION_TYPE.get(name)
    if dtype and result.status == "success" and args:
        try:
            session.log_decision(S.DecisionEvent(
                decision_type=dtype, choice=args,
                # candidates are not enumerated by the model turn-by-turn; record [] rather
                # than fabricating, and keep the model's actual prose as the rationale.
                candidates=[], rationale=(rationale.strip() or f"chose params for {name}"),
                stage=name, params=args,
                input_summary_ref=in_cp,
            ).to_dict())
            stats.decisions_logged += 1
        except Exception:  # noqa: BLE001 — never let a decision-log issue abort the run
            pass
    return result


def _persist_structured(session, name: str, args: dict, stats: RunStats) -> dict:
    """Handle the forced-structured-output emit tools (annotation labels / DE design).

    These do not mutate the AnnData; they record a first-class decision event (so the
    structured choice is auditable + part of the replayable recipe metadata) and write
    a JSON artifact in the session.
    """
    from pathlib import Path

    kind = "annotation_labels" if name == "emit_annotation_labels" else "de_design"
    dtype = "annotation_strategy" if kind == "annotation_labels" else "de_design"
    # validate locally against the tool's JSON Schema required-keys — never trust that the
    # model/API honored the forced schema. Record the validation result, don't crash.
    schema = (prompts.ANNOTATION_LABEL_SCHEMA if name == "emit_annotation_labels"
              else prompts.DE_DESIGN_SCHEMA)
    missing = ([k for k in schema.get("required", []) if k not in args]
               if isinstance(args, dict) and schema.get("type") == "object" else [])
    session.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out = session.artifacts_dir / f"{kind}.json"
    out.write_text(json.dumps(args, indent=2, default=str))
    try:
        session.log_decision(S.DecisionEvent(
            decision_type=dtype, choice=args,
            candidates=[args],
            rationale=f"mode-2 agent emitted structured {kind}",
            stage=name, params={"artifact": str(out)},
        ).to_dict())
        stats.decisions_logged += 1
    except Exception:  # noqa: BLE001
        pass
    payload = {"recorded": kind, "artifact": str(out), "schema_valid": not missing}
    if missing:
        payload["missing_required"] = missing   # surfaced back to the model to re-emit
    return payload


def run_agent(session, provider: Provider, *, goal: str | None = None,
              tissue: str | None = None, resolutions: dict | None = None,
              toolset: list[str] | None = None, seed: int = 0,
              max_iters: int = 40) -> AgentResult:
    """Drive the autonomous tool loop until the model stops calling tools (or max_iters).

    ``tissue`` (e.g. 'human pancreas, PDAC') is a soft annotation prior. ``resolutions`` is the
    human-set clustering resolution(s) per embedding (human-in-the-loop) — the agent must use
    only these and never auto-choose. Returns an ``AgentResult`` (prose, stats, transcript);
    all tool runs are logged for deterministic replay.
    """
    system = _system_prompt(goal, tissue, resolutions)
    tool_schemas = build_tool_schemas(toolset)
    stats = RunStats()
    transcript: list[dict] = []

    user_kick = (
        "Begin the autonomous analysis. Call detect_state first to find the re-entry "
        "point, then proceed through the canonical flow, choosing parameters from each "
        "tool's JSON summary. State each consequential choice (candidates + rationale) in "
        "prose before the tool call. When the goal is met, stop and summarize."
    )
    messages: list[dict] = [{"role": "user", "content": user_kick}]

    final_text = ""
    stopped_reason = "max_iters"
    for _ in range(max_iters):
        resp = provider.complete(messages, tools=tool_schemas, system=system,
                                 tool_choice="auto")
        stats.llm_turns += 1
        stats.add_usage(resp.usage)
        if resp.text:
            transcript.append({"role": "assistant", "text": resp.text})

        if not resp.wants_tools:
            final_text = resp.text or final_text
            stopped_reason = "completed"
            break

        # echo the assistant's tool-call turn back into the conversation
        messages.append({"role": "assistant_tool_calls", "text": resp.text,
                         "tool_calls": resp.tool_calls})

        for call in resp.tool_calls:
            stats.add_call(call.name)
            transcript.append({"role": "tool_call", "name": call.name, "args": call.arguments})
            try:
                if call.name in ("emit_annotation_labels", "emit_de_design"):
                    payload = _persist_structured(session, call.name, call.arguments, stats)
                    content = json.dumps(payload)
                else:
                    result = _execute_registry_tool(session, call.name, call.arguments,
                                                     seed, stats, rationale=resp.text or "")
                    content = json.dumps(result.to_dict(), default=str)
            except KeyError:
                content = json.dumps({"status": "error", "error_code": "unknown_tool",
                                      "error": f"no tool named {call.name}"})
                stats.errors += 1
            except Exception as exc:  # noqa: BLE001 — feed the error back, keep the loop alive
                content = json.dumps({"status": "error", "error_code": "internal",
                                      "error": f"{type(exc).__name__}: {exc}"})
                stats.errors += 1
            messages.append(provider.tool_result_message(call, content))

    return AgentResult(final_text=final_text, stats=stats, transcript=transcript,
                       stage=session.manifest.stage, stopped_reason=stopped_reason)


def force_structured(session, provider: Provider, *, schema_tool: str,
                     context: str, system: str, seed: int = 0) -> dict:
    """One-shot FORCED structured output (plan D4): require the model to call ``schema_tool``.

    Used when a step MUST return machine-readable JSON (annotation labels / DE design).
    Both backends force the specific tool via ``tool_choice=<name>``.
    """
    schema = (prompts.ANNOTATION_LABEL_SCHEMA if schema_tool == "emit_annotation_labels"
              else prompts.DE_DESIGN_SCHEMA)
    tool_schemas = [{"name": schema_tool, "description": f"Emit the {schema_tool} object.",
                     "input_schema": schema}]
    messages = [{"role": "user", "content": context}]
    resp = provider.complete(messages, tools=tool_schemas, system=system,
                             tool_choice=schema_tool)
    if not resp.tool_calls:
        raise RuntimeError(f"model did not emit the forced {schema_tool} object")
    call = resp.tool_calls[0]
    _persist_structured(session, schema_tool, call.arguments, RunStats())
    return call.arguments
