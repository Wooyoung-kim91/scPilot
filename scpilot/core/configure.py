"""configure_run — select the run's LLM topology at MCP-call time (see ``llm/topology.py``).

The host (Claude Code / Codex) calls this ONCE at session start to declare which engine plays each
role and HOW it is invoked (api | cli | host_plugin). The choice is persisted in the session manifest
so every subsequent Tier-4 review routes to the chosen reviewer, and the analysis engine is recorded
for provenance + cross-engine reviewer selection. Non-mutating (does not touch the AnnData).
"""

from __future__ import annotations

from scpilot import schemas as S
from scpilot.llm import topology as T
from scpilot.tools import register


@register("configure_run", mutating=False,
          description="Select the run's LLM topology: which engine plays analysis/reviewer/annotator/"
                      "interpreter, each invoked as api|cli|host_plugin. Call once at session start; "
                      "persisted so Tier-4 review routes to the chosen reviewer. 'analysis' is recorded "
                      "for provenance (in mode-1 the host itself is the analysis engine).")
def _configure_run(session, *, topology: dict | None = None, **params) -> S.ToolResult:
    if not topology:
        return S.error("configure_run", "invalid_params",
                       "no 'topology' given — pass e.g. {\"analysis\": {\"type\": \"cli\", "
                       "\"cli\": \"claude-code\", \"model\": \"claude-opus-4-8\"}, \"reviewer\": "
                       "{\"type\": \"host_plugin\", \"plugin\": \"codex\"}}", recoverable=True)
    norm, problems = T.validate_topology(topology)
    if problems:
        return S.error("configure_run", "invalid_params", "; ".join(problems), recoverable=True)
    avail = T.probe_availability(norm)

    session.manifest.llm_topology = norm
    session.save()

    warnings = [f"{r}: {a['reason']}" for r, a in avail.items()
                if a.get("ready") is False and a.get("reason")]

    # Host directive for a host_plugin reviewer (mode-1): the host must delegate the critique to its
    # plugin (a DIFFERENT engine than the annotator), then feed verdicts back via apply_annotation_audit.
    directives: list[str] = []
    rev = norm.get("reviewer")
    if rev and rev["type"] == "host_plugin":
        directives.append(
            f"Tier-4 reviewer = host plugin '{rev['plugin']}': after annotation_audit emits evidence, "
            f"delegate the adversarial critique to your '{rev['plugin']}' plugin (a different engine "
            f"than the annotator), then call apply_annotation_audit with its verdicts.")
    elif rev and rev["type"] in ("api", "cli"):
        directives.append(
            f"Tier-4 reviewer = {rev['type']} ({rev.get('model') or rev.get('cli') or rev.get('backend')}): "
            f"scpilot will run the critique itself — call the review flow rather than reviewing inline.")

    summary = {
        "topology": norm,
        "availability": avail,
        "host_directives": directives,
        "analysis_is_declaration": "analysis" in norm,
        "roles_defaulting_to_analysis": [r for r in ("annotator", "interpreter") if r not in norm],
    }
    return S.success("configure_run", summary=summary, warnings=warnings,
                     determinism_grade="A", suggested_next_tools=["detect_state"])


def review_routing(topo: dict | None, *, label_key: str = "major_cell_type") -> dict:
    """Decide HOW the Tier-4 review of ``label_key`` should be executed, from the run topology.

    Pure (no side effects) so it is unit-testable. Returns ``{mode, reviewer, directive}`` where mode
    ∈ {host_plugin, host_or_mode2, self}. scpilot does NOT run an LLM critique inside a mode-1 registry
    tool (that would not replay); api/cli reviewers execute in the replay-safe mode-2 ``run_agent`` path
    (or the host may run them and call apply_annotation_audit)."""
    rev = (topo or {}).get("reviewer")
    if not rev:
        return {"mode": "host_or_mode2", "reviewer": None,
                "directive": (f"No reviewer configured — call configure_run. Review '{label_key}' with "
                              "an independent engine (cross-engine preferred), then apply_annotation_audit.")}
    t = rev.get("type")
    if t == "host_plugin":
        return {"mode": "host_plugin", "reviewer": rev,
                "directive": (f"Delegate the adversarial Tier-4 critique of '{label_key}' to your "
                              f"'{rev['plugin']}' plugin (a different engine than the annotator), then "
                              "call apply_annotation_audit with its verdicts.")}
    return {"mode": "host_or_mode2", "reviewer": rev,
            "directive": (f"Reviewer is {t} ({rev.get('model') or rev.get('cli') or rev.get('backend')}): "
                          "scpilot runs it in the replay-safe `scpilot run` (mode-2) path (topology-driven), "
                          "or the host may execute the critique and call apply_annotation_audit.")}


@register("run_review", mutating=False,
          description="Route/prepare a Tier-4 review of a label column using the configured reviewer "
                      "(configure_run topology). Runs annotation_audit (deterministic evidence) and returns "
                      "it plus a routing directive: host_plugin → delegate to the host's plugin; api/cli → "
                      "run via mode-2 or the host. Kept deterministic (no inline LLM) so replay is exact.")
def _run_review(session, *, groupby: str = "leiden", label_key: str = "major_cell_type", **params) -> S.ToolResult:
    from scpilot import tools
    if label_key not in session.adata.obs.columns:
        return S.error("run_review", "invalid_state",
                       f"no obs['{label_key}'] to review — annotate that column first", recoverable=True)
    routing = review_routing(session.manifest.llm_topology, label_key=label_key)
    audit = tools.run("annotation_audit", session, groupby=groupby, label_key=label_key)
    warnings = [] if audit.status == "success" else [f"annotation_audit: {audit.error or 'failed'}"]
    summary = {"label_key": label_key, "groupby": groupby, "routing": routing,
               "audit": getattr(audit, "summary", {}) if audit.status == "success" else {}}
    return S.success("run_review", summary=summary, warnings=warnings, determinism_grade="A",
                     suggested_next_tools=["apply_annotation_audit"])
