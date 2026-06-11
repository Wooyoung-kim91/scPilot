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


def _collect_png_artifacts(session) -> list[str]:
    art_dir = session.artifacts_dir
    if not art_dir.exists():
        return []
    return sorted(str(p) for p in art_dir.rglob("*.png"))


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
    pngs = _collect_png_artifacts(session)
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
        "figures": pngs,
        "checkpoints": [cp.get("id") for cp in man.checkpoints],
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

    if pngs:
        md += ["## Figures", ""]
        for p in pngs:
            name = Path(p).name
            md.append(f"### {name}")
            md.append(f"![{name}]({p})")
            md.append("")

    session.artifacts_dir.mkdir(parents=True, exist_ok=True)
    md_path = session.artifacts_dir / "report.md"
    json_path = session.artifacts_dir / "report.json"
    md_path.write_text("\n".join(md))
    json_path.write_text(json.dumps(report_json, indent=2, default=str))

    summary = {
        "n_runs": len(runs), "n_decisions": len(decisions),
        "n_figures": len(pngs), "stage_reached": man.stage,
        "has_interpretation": bool(interpretation),
    }
    artifacts = [
        S.Artifact(path=str(md_path), kind="txt", description="Markdown report (report.md)"),
        S.Artifact(path=str(json_path), kind="json", description="report manifest (report.json)"),
    ]
    return S.success("report", summary=summary, artifacts=artifacts,
                     determinism_grade="A", duration_s=round(time.time() - t0, 3))
