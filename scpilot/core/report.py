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
    # final consolidated annotation distribution (Phase F), if finalize_annotation ran
    _fin = next((r for r in runs if r.get("tool") == "finalize_annotation"), None)
    if _fin:
        report_json["final_annotation"] = (_fin.get("summary", {}) or {}).get("label_distribution", {})

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


@register("export_final", mutating=False,
          description="Export a PUBLICATION-READY slim object: drop QC-artifact cells (default "
                      "Low_quality/Doublet — malignant tumour cells are KEPT) and keep ONLY the "
                      "benchmark-chosen best integration reduction (its embedding + its UMAP as the "
                      "canonical X_umap + its neighbour graph), then render the final UMAP(s). Writes a "
                      "NEW .h5ad FILE (cell count changes, so it is deliberately NOT a pipeline checkpoint) "
                      "plus figures on the best-embedding manifold. Run after finalize_annotation.")
def export_final(session, *, label_key: str = "final_annotation",
                 remove_labels: list | None = None, keep_reduction: str | None = None,
                 out_name: str = "final_clean.h5ad", slim_obs: bool = True,
                 color_keys: list | None = None, **params) -> S.ToolResult:
    """Slim, artifact-free export on the best embedding. Read-only w.r.t. the session (writes a
    standalone file + figures; does not mutate/checkpoint the pipeline object)."""
    t0 = time.time()
    adata = session.adata

    if label_key not in adata.obs.columns:
        return S.error("export_final", "invalid_state",
                       f"'{label_key}' not in obs — run finalize_annotation first",
                       recoverable=True, suggested_next_tools=["finalize_annotation"])

    # --- resolve the reduction to keep — EVIDENCE-BASED, never a hardcoded preference ------------
    # The best reduction is a benchmark RESULT (uns['scpilot']['best_embedding']); if the caller did
    # not override it and benchmark has not recorded one, we do NOT guess — the choice must come from
    # evidence (run benchmark) or be stated explicitly via keep_reduction.
    if not keep_reduction:
        keep_reduction = (adata.uns.get("scpilot", {}) or {}).get("best_embedding")
    if not keep_reduction:
        return S.error("export_final", "invalid_state",
                       "no best reduction to keep: benchmark has not recorded "
                       "uns['scpilot']['best_embedding'] and keep_reduction was not given. Run benchmark "
                       "(evidence-based choice) or pass keep_reduction explicitly — the best reduction is "
                       "never hardcoded.", recoverable=True, suggested_next_tools=["benchmark"])
    if keep_reduction not in adata.obsm:
        return S.error("export_final", "invalid_state",
                       f"reduction '{keep_reduction}' not in obsm{sorted(adata.obsm)}",
                       recoverable=True, suggested_next_tools=["benchmark"])
    from scpilot.core.autoplot import _umap_for_reduction
    umap_key = _umap_for_reduction(keep_reduction)

    # --- drop QC-artifact cells (keep malignant tumour cells) -----------------------------------
    if remove_labels is None:
        remove_labels = ["Low_quality", "Doublet", "Doublet_Mixed", "Mixed/Artifact"]
    fa = adata.obs[label_key].astype(str)
    drop_mask = fa.isin(remove_labels)
    for lab in remove_labels:                                # also catch 'Malignant Low_quality' etc.
        drop_mask = drop_mask | fa.str.contains(lab, regex=False)
    if "major_cell_type" in adata.obs.columns:
        drop_mask = drop_mask | adata.obs["major_cell_type"].astype(str).isin(remove_labels)
    removed_counts = {k: int(v) for k, v in fa[drop_mask].value_counts().items()}
    n0 = adata.n_obs
    clean = adata[~drop_mask.values].copy()

    # --- keep ONLY the chosen reduction (+ its UMAP as canonical X_umap) -------------------------
    if umap_key and umap_key in clean.obsm:
        clean.obsm["X_umap"] = clean.obsm[umap_key]
    keep_obsm = {keep_reduction, "X_umap"}
    for k in [k for k in clean.obsm if k not in keep_obsm]:
        del clean.obsm[k]

    # neighbour graph: promote the chosen reduction's graph to the canonical connectivities/distances
    chosen_nbr = None
    for uv in clean.uns.values():
        if isinstance(uv, dict) and isinstance(uv.get("params"), dict) \
                and uv["params"].get("use_rep") == keep_reduction:
            chosen_nbr = uv
            break
    if chosen_nbr:
        ck, dk = chosen_nbr.get("connectivities_key"), chosen_nbr.get("distances_key")
        if ck in clean.obsp:
            clean.obsp["connectivities"] = clean.obsp[ck]
        if dk in clean.obsp:
            clean.obsp["distances"] = clean.obsp[dk]
        clean.uns["neighbors"] = {"connectivities_key": "connectivities",
                                  "distances_key": "distances", "params": dict(chosen_nbr["params"])}
    for k in [k for k in clean.obsp if k not in ("connectivities", "distances")]:
        del clean.obsp[k]
    for uk in [k for k in clean.uns if k != "neighbors"
               and (str(k).startswith("neighbors_") or str(k).startswith("cnv_neighbors"))]:
        clean.uns.pop(uk, None)

    # --- optionally slim redundant per-alternate-embedding obs columns --------------------------
    if slim_obs:
        suffix = keep_reduction.removeprefix("X_").lower()   # harmony / scvi / pca
        keep_leiden = "leiden" if suffix == "pca" else f"leiden_{suffix}"
        drop_exact = {"celltype_harmonized", "celltype_harmonized_agreement", "sample_id_from_concat"}
        for a_ in ("harmony", "scvi", "pca"):                # per-method major labels: redundant with canonical
            for s2 in ("", "_confidence", "_review_required"):
                drop_exact.add(f"major_cell_type_{a_}{s2}")
        for c in list(clean.obs.columns):
            if c in drop_exact or (c.startswith("leiden") and c != keep_leiden):
                del clean.obs[c]
        for c in clean.obs.columns:
            if str(clean.obs[c].dtype) == "category":
                clean.obs[c] = clean.obs[c].cat.remove_unused_categories()

    clean.uns.setdefault("scpilot", {})
    clean.uns["scpilot"]["final_export"] = {
        "removed_labels": remove_labels, "n_removed": int(n0 - clean.n_obs),
        "kept_reduction": keep_reduction, "canonical_umap": f"X_umap (={umap_key})",
    }

    # --- write the standalone file (NOT a checkpoint: n_obs changed) ----------------------------
    out_path = session.out / out_name
    clean.write_h5ad(out_path, compression="gzip")

    # --- render the final UMAP(s) on the best-embedding manifold (defensive) --------------------
    arts: list = []
    try:
        from scpilot.vendor import plotting as P
        from scpilot.core.plots import _cfg, _art, _artifacts_from_fit
        cfg = _cfg(session)
        art_dir = session.artifacts_dir
        art_dir.mkdir(parents=True, exist_ok=True)
        if color_keys is None:
            color_keys = [label_key] + [c for c in ("major_cell_type", "malignancy")
                                        if c in clean.obs.columns]
        for ck in color_keys:
            if ck not in clean.obs.columns:
                continue
            try:
                fit = P.save_umap(clean, cfg, _art(art_dir, f"final_export_{ck}"),
                                  color=ck, basis="X_umap")
                arts += _artifacts_from_fit(fit, cfg)
            except Exception:  # noqa: BLE001 — a missing plot must never fail the export
                continue
    except Exception:  # noqa: BLE001
        pass

    summary = {
        "out_path": str(out_path), "n_cells_before": int(n0), "n_cells_after": int(clean.n_obs),
        "n_removed": int(n0 - clean.n_obs), "removed_by_label": removed_counts,
        "kept_reduction": keep_reduction, "canonical_umap": umap_key,
        "obsm_kept": sorted(clean.obsm.keys()), "n_final_labels": int(clean.obs[label_key].nunique()),
        "slim_obs": bool(slim_obs),
    }
    return S.success("export_final", summary=summary, artifacts=arts,
                     determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["report"])
