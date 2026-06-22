"""Analysis report — figures + tables + interpretation -> Markdown (plan B16 / D5).

The ``report`` tool is deterministic: it gathers what the session already produced
(run log, decision events, checkpoints, artifacts) and writes a Markdown report plus a
machine-readable ``report.json`` manifest. The LLM *interpretation* prose is OPTIONAL
and supplied by the mode-2 CLI (``scpilot run``) via the ``interpretation`` param — the
tool itself makes no LLM call, so it stays replayable. In mode 1 (MCP) the host agent
can pass its own interpretation, or omit it for a numbers-only report.

Contract: ``fn(session, **params) -> ToolResult``. Read-only w.r.t. the AnnData (it does
not mutate or checkpoint); it writes report files as artifacts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from scpilot import schemas as S
from scpilot.tools import register


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _kv(params: dict, n: int = 4) -> str:
    """Compact scalar-param string for an artifact's provenance line."""
    items = [(k, v) for k, v in (params or {}).items()
             if isinstance(v, (int, float, str, bool)) and k != "reasoning"]
    return ", ".join(f"{k}={v}" for k, v in items[:n])


def _artifact_catalog(session) -> list[dict]:
    """Flatten outputs.jsonl into per-artifact rows carrying their PROVENANCE.

    Each row binds an artifact to the step that produced it (tool/params), the WHY
    (reasoning), the recipe hash, and the file sha256 — so the report can show, per
    figure/table, where it came from and why (plan A3). Falls back to a bare directory
    scan only for legacy sessions with no outputs.jsonl.
    """
    rows: list[dict] = []
    for rec in _read_jsonl(session.outputs_path):
        for a in rec.get("artifacts", []) or []:
            rows.append({
                "tool": rec.get("tool"), "stage": rec.get("stage"),
                "params": rec.get("params", {}), "reasoning": rec.get("reasoning"),
                "recipe_hash": rec.get("recipe_hash"),
                "path": a.get("path"), "kind": a.get("kind", "other"),
                "description": a.get("description", ""),
                "sha256": (a.get("meta") or {}).get("sha256"),
            })
    if rows:
        return rows
    # legacy fallback: no outputs index → bare PNG scan, no provenance
    art_dir = session.artifacts_dir
    if art_dir.exists():
        for p in sorted(art_dir.rglob("*.png")):
            rows.append({"tool": None, "stage": None, "params": {}, "reasoning": None,
                         "recipe_hash": None, "path": str(p), "kind": "png",
                         "description": "", "sha256": None})
    return rows


@register("report", mutating=False,
          description="Assemble the analysis report (Markdown + report.json) from the session's run log, "
                      "decision events, and artifacts (figures/tables). Optional LLM `interpretation` "
                      "prose is injected by mode-2; the tool itself makes no LLM call (replayable). "
                      "Run last (plan B16).")
def report(session, *, interpretation: str | None = None, title: str = "scpilot analysis report",
           **params) -> S.ToolResult:
    t0 = time.time()
    runs = _read_jsonl(session.run_log_path)
    decisions = _read_jsonl(session.decisions_path)
    catalog = _artifact_catalog(session)
    figures = [c for c in catalog if c["kind"] in ("png", "svg")]
    tables = [c for c in catalog if c["kind"] == "csv"]
    man = session.manifest

    # ---- assemble the structured manifest ----
    steps = [{"tool": r.get("tool"), "status": r.get("status"),
              "summary": r.get("summary", {})} for r in runs]
    report_json = {
        "session_id": man.session_id,
        "title": title,
        "input": man.input.get("path"),
        "stage_reached": man.stage,
        "n_runs": len(runs),
        "n_decisions": len(decisions),
        "steps": steps,
        "decisions": [{"decision_type": d.get("decision_type"), "choice": d.get("choice"),
                       "rationale": d.get("rationale"), "stage": d.get("stage")}
                      for d in decisions],
        # provenance-bearing artifact catalog (figures + tables + others) — each row links the
        # file to its producing tool/params/reasoning/sha (plan A3). `figures` kept for compat.
        "artifacts": catalog,
        "figures": [c["path"] for c in figures],
        "checkpoints": [cp.get("id") for cp in man.checkpoints],
        "log_consistency": session.log_consistency(),   # run_log ↔ outputs.jsonl coupling (C-2)
    }

    # ---- render Markdown ----
    md: list[str] = [f"# {title}", ""]
    md.append(f"- **Session**: `{man.session_id}`")
    md.append(f"- **Input**: `{man.input.get('path', '?')}`")
    md.append(f"- **Stage reached**: `{man.stage}`")
    md.append(f"- **Tool runs**: {len(runs)}  |  **Decisions logged**: {len(decisions)}")
    md.append("")

    if interpretation:
        md += ["## Interpretation", "", interpretation.strip(), ""]

    md += ["## Pipeline steps", ""]
    for r in runs:
        sm = r.get("summary", {}) or {}
        keys = ", ".join(f"{k}={sm[k]}" for k in list(sm)[:6]
                         if isinstance(sm[k], (int, float, str, bool)))
        md.append(f"- `{r.get('tool')}` — {r.get('status')}" + (f" ({keys})" if keys else ""))
    md.append("")

    if decisions:
        md += ["## Key decisions", ""]
        for d in decisions:
            md.append(f"- **{d.get('decision_type')}**: {d.get('rationale', '')}")
        md.append("")

    def _prov_line(c: dict) -> str:
        bits = []
        if c.get("tool"):
            kv = _kv(c.get("params", {}))
            bits.append(f"from `{c['tool']}`" + (f" ({kv})" if kv else ""))
        if c.get("reasoning"):
            bits.append(f"why: {c['reasoning']}")
        return "  \n  _" + " — ".join(bits) + "_" if bits else ""

    if figures:
        md += ["## Figures", ""]
        for c in figures:
            name = Path(c["path"]).name
            md.append(f"### {name}")
            md.append(f"![{name}]({c['path']})" + _prov_line(c))
            md.append("")

    if tables:
        md += ["## Tables", ""]
        for c in tables:
            name = Path(c["path"]).name
            md.append(f"- `{name}` — `{c['path']}`" + _prov_line(c))
        md.append("")

    session.artifacts_dir.mkdir(parents=True, exist_ok=True)
    md_path = session.artifact_path("report.md")        # no-overwrite on re-run (P1-2)
    json_path = session.artifact_path("report.json")
    md_path.write_text("\n".join(md))
    json_path.write_text(json.dumps(report_json, indent=2, default=str))

    summary = {
        "n_runs": len(runs), "n_decisions": len(decisions),
        "n_figures": len(figures), "n_tables": len(tables),
        "n_artifacts": len(catalog), "stage_reached": man.stage,
        "has_interpretation": bool(interpretation),
    }
    artifacts = [
        S.Artifact(path=str(md_path), kind="txt", description="Markdown report (report.md)"),
        S.Artifact(path=str(json_path), kind="json", description="report manifest (report.json)"),
    ]
    return S.success("report", summary=summary, artifacts=artifacts,
                     determinism_grade="A", duration_s=round(time.time() - t0, 3))
