"""Compartment planning + subset reprocessing — scpilot plan B11.

The bridge from Tier-1 broad labels (``obs['major_cell_type']``) to Tier-3 fine
annotation: which broad compartments are worth recursing into, and a reprocessed
subset to recurse on. Two tools, following the project's evidence→decision→apply
split (no hardcoded thresholds drive the call — the gate is a *floor*, the LLM
chooses which branches to take):

1. ``compartment_plan`` (read-only) — per-compartment REAL cell counts, sample/
   batch coverage, single-patient dominance, and a batch-mixing diagnostic
   (normalized batch entropy). A ``min_cells``/``min_samples`` FLOOR blocks
   branches with too little data to subcluster reliably ("임계 미달 분기 차단").
   The LLM reads this and records a ``compartment_branch`` decision.

2. ``compartment_subset`` (mutating) — extract ONE compartment's cells and
   reprocess in one of two modes:
     - ``markers``    : re-normalize from counts → log1p → HVG(seurat_v3) → PCA,
                        so the HVGs/markers are compartment-relevant (fresh
                        feature space for within-compartment DE).
     - ``clustering`` : integration-aware — keep the chosen integration embedding
                        (``use_rep``) subset untouched (preserves batch
                        correction) for neighbors→leiden subclustering.
   Writes a subset checkpoint; the parent dataset stays as the prior checkpoint.

Invariants: counts stays per-cell immutable (subsetting rows is legitimate
filtering; genes are never dropped), and provenance (parent stage, compartment,
mode) is stamped into ``.uns['scpilot']`` via the session checkpoint.
"""

from __future__ import annotations

import re
import time

from scpilot import schemas as S
from scpilot.tools import register

# obs keys tried (in order) when the caller does not pass an explicit compartment key.
_GROUPBY_CANDIDATES = ("major_cell_type", "celltype_consensus", "leiden")
# obs keys tried (in order) as the technical batch axis for the mixing diagnostic.
_BATCH_CANDIDATES = ("GSE", "batch", "sample_id")


def _resolve_key(adata, key: str | None, candidates: tuple[str, ...]) -> str | None:
    if key is not None:
        return key if key in adata.obs.columns else None
    for cand in candidates:
        if cand in adata.obs.columns:
            return cand
    return None


def _norm_entropy(counts, n_global: int) -> float:
    """Shannon entropy of a category distribution, normalized to [0,1] by the
    GLOBAL number of categories — so a compartment present in only one batch
    scores ~0 (single-batch dominated) and one spread evenly over all batches ~1."""
    import numpy as np

    p = np.asarray(counts, dtype=float)
    p = p[p > 0]
    total = p.sum()
    if total <= 0 or n_global <= 1:
        return 0.0
    p = p / total
    H = float(-(p * np.log(p)).sum())
    return round(H / np.log(n_global), 4)


# --------------------------------------------------------------------------- #
# Tool 1 — compartment plan (read-only evidence + branch FLOOR)
# --------------------------------------------------------------------------- #
@register("compartment_plan", mutating=False,
          description="Tier-1→Tier-3 bridge EVIDENCE (read-only, no hardcoded call): per broad compartment "
                      "(obs['major_cell_type'] by default) reports REAL cell counts, sample/batch coverage, "
                      "single-patient dominance, and a batch-mixing diagnostic (normalized batch entropy: ~1 "
                      "well-mixed, ~0 single-batch). A min_cells/min_samples FLOOR blocks under-powered branches "
                      "from fine subclustering. The LLM reads this, records a compartment_branch decision, then "
                      "calls compartment_subset on each branchable compartment.")
def compartment_plan(session, *, groupby: str | None = None, batch_key: str | None = None,
                     sample_key: str = "sample_id", min_cells: int = 50, min_samples: int = 2,
                     single_source_frac: float = 0.8, **params) -> S.ToolResult:
    import json

    import pandas as pd

    t0 = time.time()
    adata = session.adata

    gkey = _resolve_key(adata, groupby, _GROUPBY_CANDIDATES)
    if gkey is None:
        return S.error("compartment_plan", "invalid_state",
                       f"no compartment key in obs (looked for {list(_GROUPBY_CANDIDATES)}) — run "
                       "annotate_broad / apply_annotation first", recoverable=True,
                       suggested_next_tools=["annotation_review", "apply_annotation"])
    bkey = _resolve_key(adata, batch_key, _BATCH_CANDIDATES)
    has_sample = sample_key in adata.obs.columns

    obs_g = adata.obs[gkey].astype(str)
    n_batches_global = int(adata.obs[bkey].astype(str).nunique()) if bkey else 0
    n_samples_global = int(adata.obs[sample_key].astype(str).nunique()) if has_sample else 0

    payloads, rows = [], []
    branchable, blocked = [], []
    for comp in obs_g.groupby(obs_g, observed=True).groups:
        comp = str(comp)
        mask = (obs_g == comp).values
        n_cells = int(mask.sum())

        ev: dict = {"compartment": comp, "n_cells": n_cells}
        reasons: list[str] = []

        n_samples = 0
        top_sample_frac = None
        if has_sample:
            sv = adata.obs[sample_key].astype(str)[mask].value_counts(normalize=True)
            n_samples = int(sv.size)
            top_sample_frac = round(float(sv.iloc[0]), 3)
            ev["sample_coverage"] = {
                "n_samples": n_samples,
                "sample_coverage_frac": round(n_samples / n_samples_global, 3) if n_samples_global else None,
                "top_sample_fraction": top_sample_frac,
                "dominant_sample": str(sv.index[0]),
                "single_patient_dominated": bool(top_sample_frac >= single_source_frac),
            }

        if bkey:
            bv = adata.obs[bkey].astype(str)[mask].value_counts()
            batch_entropy = _norm_entropy(bv.values, n_batches_global)
            dom_batch_frac = round(float(bv.iloc[0] / bv.sum()), 3)
            ev["batch_mixing"] = {
                "batch_key": bkey,
                "n_batches": int(bv.size),
                "batch_coverage_frac": round(int(bv.size) / n_batches_global, 3) if n_batches_global else None,
                "batch_entropy_norm": batch_entropy,     # ~1 well-mixed, ~0 single-batch dominated
                "dominant_batch_fraction": dom_batch_frac,
                "dominant_batch": str(bv.index[0]),
            }
            if int(bv.size) == 1 and n_batches_global > 1:
                reasons.append("single_batch")
            elif batch_entropy < 0.3 and int(bv.size) > 1:
                reasons.append("low_batch_mixing")

        # branch FLOOR — too few cells / samples to subcluster reliably (a gate, not a call)
        powered = n_cells >= min_cells and (n_samples >= min_samples if has_sample else True)
        if n_cells < min_cells:
            reasons.append(f"below_min_cells(<{min_cells})")
        if has_sample and n_samples < min_samples:
            reasons.append(f"below_min_samples(<{min_samples})")
        ev["branch_recommended"] = bool(powered)
        ev["branch_block_reasons"] = reasons
        (branchable if powered else blocked).append(comp)

        payloads.append(ev)
        rows.append({
            "compartment": comp, "n_cells": n_cells, "n_samples": n_samples,
            "top_sample_frac": top_sample_frac,
            "batch_entropy_norm": ev.get("batch_mixing", {}).get("batch_entropy_norm"),
            "branch_recommended": ev["branch_recommended"],
            "block_reasons": ";".join(reasons),
        })

    rows.sort(key=lambda r: r["n_cells"], reverse=True)
    df = pd.DataFrame(rows)

    art_dir = session.artifacts_dir
    art_dir.mkdir(parents=True, exist_ok=True)
    json_path = art_dir / "compartment_plan.json"
    json_path.write_text(json.dumps({
        "groupby": gkey, "batch_key": bkey, "sample_key": sample_key if has_sample else None,
        "floor": {"min_cells": min_cells, "min_samples": min_samples},
        "n_batches_global": n_batches_global, "n_samples_global": n_samples_global,
        "branchable": branchable, "blocked": blocked,
        "instruction": "Choose which compartments to recurse into for Tier-3 fine annotation. The FLOOR "
                       "(min_cells/min_samples) only marks branch_recommended=false for under-powered "
                       "compartments — you still decide. Prefer well-mixed compartments (batch_entropy_norm "
                       "near 1); treat single_patient_dominated / low_batch_mixing as confounds to note, not "
                       "auto-exclude. Record a compartment_branch decision, then call compartment_subset "
                       "(mode='clustering' to subcluster on the integration embedding, mode='markers' to "
                       "re-derive compartment-relevant HVGs/markers).",
        "compartments": payloads}, indent=2, default=str))

    warnings: list[str] = []
    if not bkey:
        warnings.append(f"no batch key in obs (looked for {list(_BATCH_CANDIDATES)}) — no batch-mixing diagnostic.")
    if not has_sample:
        warnings.append(f"sample_key '{sample_key}' absent — no sample-coverage / single-patient signal.")
    if not branchable:
        warnings.append(f"no compartment clears the floor (min_cells={min_cells}, min_samples={min_samples}) — "
                        "lower the floor or revisit Tier-1 labels.")

    summary = {
        "groupby": gkey, "batch_key": bkey,
        "n_compartments": len(payloads),
        "n_branchable": len(branchable), "n_blocked": len(blocked),
        "branchable": branchable, "blocked": blocked,
        "floor": {"min_cells": min_cells, "min_samples": min_samples},
        "evidence_input": str(json_path),
        "note": "Plan only — no subset created. The LLM picks branches (compartment_branch decision) then "
                "calls compartment_subset per chosen compartment.",
    }
    return S.success("compartment_plan", summary=summary,
                     tables={"compartments": S.table_preview(df, max_rows=len(df))},
                     artifacts=[S.Artifact(path=str(json_path), kind="json",
                                           description="per-compartment counts/coverage/batch-mixing + branch floor")],
                     warnings=warnings, determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["compartment_subset"])


# --------------------------------------------------------------------------- #
# Tool 2 — compartment subset + reprocess (two modes)
# --------------------------------------------------------------------------- #
def _safe(label: str) -> str:
    """Filesystem/obs-safe slug for a compartment label (e.g. 'T/NK' → 'T_NK')."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(label)).strip("_") or "compartment"


@register("compartment_subset", mutating=True,
          description="Extract ONE compartment's cells (obs[groupby]==compartment) and reprocess for Tier-3 "
                      "fine annotation. mode='markers' re-normalizes from counts → log1p → HVG(seurat_v3) → PCA "
                      "(compartment-relevant feature space for within-compartment DE). mode='clustering' is "
                      "integration-aware: keeps the use_rep embedding (e.g. X_scVI) subset untouched so batch "
                      "correction is preserved for neighbors→leiden. Counts stay per-cell immutable; genes are "
                      "never dropped. Writes a subset checkpoint (parent stays as the prior checkpoint); run "
                      "cluster (then markers/annotation_review) on the subset next.")
def compartment_subset(session, *, compartment: str | None = None, groupby: str | None = None,
                       mode: str = "clustering", use_rep: str = "X_scVI",
                       n_top_genes: int = 2000, n_pcs: int = 30, target_sum: float = 1e4,
                       hvg_batch_key: str | None = None, seed: int = 0, **params) -> S.ToolResult:
    import scanpy as sc

    t0 = time.time()
    adata = session.adata
    warnings: list[str] = []

    if mode not in ("markers", "clustering"):
        return S.error("compartment_subset", "missing_input",
                       f"mode must be 'markers' or 'clustering' (got '{mode}')", recoverable=True)
    gkey = _resolve_key(adata, groupby, _GROUPBY_CANDIDATES)
    if gkey is None:
        return S.error("compartment_subset", "invalid_state",
                       f"no compartment key in obs (looked for {list(_GROUPBY_CANDIDATES)}) — run "
                       "annotate_broad / apply_annotation first", recoverable=True,
                       suggested_next_tools=["compartment_plan"])
    if not compartment:
        return S.error("compartment_subset", "missing_input",
                       f"'compartment' must name a value of obs['{gkey}'] "
                       f"(available: {sorted(adata.obs[gkey].astype(str).unique())[:20]})", recoverable=True,
                       suggested_next_tools=["compartment_plan"])

    mask = (adata.obs[gkey].astype(str) == str(compartment)).values
    n_sub = int(mask.sum())
    if n_sub == 0:
        return S.error("compartment_subset", "data_gate_failed",
                       f"compartment '{compartment}' not found in obs['{gkey}'] "
                       f"(available: {sorted(adata.obs[gkey].astype(str).unique())[:20]})", recoverable=True,
                       suggested_next_tools=["compartment_plan"])

    if "counts" not in adata.layers:
        return S.error("compartment_subset", "invalid_state",
                       "no 'counts' layer — reproducible subset reprocessing needs raw counts",
                       recoverable=False)

    if mode == "clustering" and use_rep not in adata.obsm:
        return S.error("compartment_subset", "invalid_state",
                       f"mode='clustering' needs integration embedding '{use_rep}' in obsm{sorted(adata.obsm)} — "
                       "run integrate_scvi/integrate_harmony first, or use mode='markers'", recoverable=True,
                       suggested_next_tools=["integrate_scvi"])

    sub = adata[mask].copy()
    safe = _safe(compartment)

    if mode == "markers":
        # re-derive a compartment-relevant feature space from counts (mirrors B4 preprocess)
        sub.X = sub.layers["counts"].copy()
        sc.pp.normalize_total(sub, target_sum=target_sum)
        sc.pp.log1p(sub)
        sub.layers["scale.data"] = sub.X.copy()
        bkey = hvg_batch_key if (hvg_batch_key and hvg_batch_key in sub.obs.columns) else None
        if hvg_batch_key and bkey is None:
            warnings.append(f"hvg_batch_key '{hvg_batch_key}' absent — HVG computed without batch")
        n_top = min(n_top_genes, sub.n_vars)
        sc.pp.highly_variable_genes(sub, flavor="seurat_v3", n_top_genes=n_top,
                                    layer="counts", batch_key=bkey)
        n_hvg = int(sub.var["highly_variable"].sum())
        max_comps = max(1, min(n_pcs, sub.n_vars - 1, sub.n_obs - 1))
        sc.pp.pca(sub, n_comps=max_comps, mask_var="highly_variable", random_state=seed)
        x_state = "log1p"
        grade = "B"
        reprocess = {"n_hvg": n_hvg, "n_pcs": max_comps, "recompute_rep": "X_pca",
                     "hvg_flavor": "seurat_v3", "hvg_batch_key": bkey}
        next_use_rep = "X_pca"
    else:  # clustering — integration-aware: keep use_rep subset untouched, no renormalization
        x_state = session.manifest.x_state
        grade = "A"
        reprocess = {"recompute_rep": None, "use_rep": use_rep,
                     "rep_dim": int(sub.obsm[use_rep].shape[1])}
        next_use_rep = use_rep

    stage = f"compartment_{safe}_{mode}"
    resolved = {"compartment": str(compartment), "groupby": gkey, "mode": mode,
                "use_rep": next_use_rep, "seed": seed, **reprocess}
    session.set_adata(sub)
    cp = session.checkpoint(stage, adata=sub, x_state=x_state, params=resolved)

    summary = {
        "compartment": str(compartment), "groupby": gkey, "mode": mode,
        "parent_n_cells": int(adata.n_obs), "n_cells": n_sub,
        "fraction_of_parent": round(n_sub / int(adata.n_obs), 4),
        "n_genes": int(sub.n_vars), "x_state": x_state,
        "next_use_rep": next_use_rep, "stage": stage,
        **reprocess,
    }
    return S.success("compartment_subset", summary=summary, checkpoint=cp.path,
                     warnings=warnings, determinism_grade=grade, duration_s=round(time.time() - t0, 3),
                     params=resolved,
                     suggested_next_tools=["cluster"] if mode == "clustering" else ["cluster", "markers"])


# --------------------------------------------------------------------------- #
# Tool 3 — assemble Tier-3 fine labels back onto the parent (B11 closing step)
# --------------------------------------------------------------------------- #
def _latest_fine_checkpoint(comp_dir, fine_key: str):
    """Return the highest-numbered checkpoint .h5ad under comp_dir/checkpoints whose
    obs carries ``fine_key`` (the per-compartment fine result), else None."""
    from pathlib import Path
    import anndata as ad
    cps = sorted((Path(comp_dir) / "checkpoints").glob("*.h5ad"), reverse=True)
    for cp in cps:
        try:
            obs = ad.read_h5ad(cp, backed="r").obs
        except Exception:  # noqa: BLE001
            continue
        if fine_key in obs.columns:
            return cp
    return None


@register("merge_fine_annotations", mutating=True,
          description="Assemble Tier-3 FINE labels from per-compartment subset sessions back onto the PARENT "
                      "dataset by cell barcode (obs_names) — the harness-tracked closing step of the compartment "
                      "loop (replaces an out-of-session merge script). Pass compartments_root (globs each "
                      "<root>/<compartment>/checkpoints for its latest fine checkpoint) or explicit "
                      "compartment_checkpoints paths. Cells in no subclustered compartment carry their Tier-1 "
                      "major_cell_type forward (carry_terminal). Writes obs[fine_key]/[facs_key]/[cell_state] + a "
                      "provenance-stamped checkpoint, so the final annotation is fully reproducible.")
def merge_fine_annotations(session, *, compartments_root: str | None = None,
                           compartment_checkpoints: list | None = None,
                           fine_key: str = "fine_cell_type", facs_key: str = "facs_style_label",
                           state_key: str = "cell_state", major_key: str = "major_cell_type",
                           carry_terminal: bool = True, **params) -> S.ToolResult:
    from pathlib import Path

    import anndata as ad
    import pandas as pd

    t0 = time.time()
    adata = session.adata
    if major_key not in adata.obs.columns:
        return S.error("merge_fine_annotations", "invalid_state",
                       f"parent lacks Tier-1 '{major_key}' — run apply_annotation first", recoverable=True,
                       suggested_next_tools=["apply_annotation"])

    # resolve the source per-compartment checkpoints
    sources: list = []
    if compartment_checkpoints:
        sources = [Path(p) for p in compartment_checkpoints]
    elif compartments_root:
        root = Path(compartments_root)
        if not root.exists():
            return S.error("merge_fine_annotations", "missing_input",
                           f"compartments_root not found: {root}", recoverable=True)
        for comp_dir in sorted(d for d in root.iterdir() if d.is_dir()):
            cp = _latest_fine_checkpoint(comp_dir, fine_key)
            if cp is not None:
                sources.append(cp)
    if not sources:
        return S.error("merge_fine_annotations", "missing_input",
                       "no fine-annotation checkpoints found (pass compartments_root or compartment_checkpoints)",
                       recoverable=True, suggested_next_tools=["compartment_subset", "apply_fine_annotation"])

    n_parent = int(adata.n_obs)
    fine = pd.Series(index=adata.obs_names, dtype=object)
    facs = pd.Series(index=adata.obs_names, dtype=object)
    state = pd.Series(index=adata.obs_names, dtype=object)

    per_source, warnings = [], []
    for src in sources:
        try:
            sub_obs = ad.read_h5ad(src, backed="r").obs
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"unreadable source skipped: {src} ({type(exc).__name__})")
            continue
        if fine_key not in sub_obs.columns:
            warnings.append(f"no '{fine_key}' in {src} — skipped")
            continue
        idx = sub_obs.index.intersection(adata.obs_names)         # map by barcode
        n_overlap = int(len(idx))
        # don't let one compartment overwrite another's already-assigned cells
        free = idx[fine.loc[idx].isna().values]
        fine.loc[free] = sub_obs.loc[free, fine_key].astype(str).values
        if facs_key in sub_obs.columns:
            facs.loc[free] = sub_obs.loc[free, facs_key].astype(str).values
        if state_key in sub_obs.columns:
            state.loc[free] = sub_obs.loc[free, state_key].astype(str).values
        per_source.append({"source": str(src), "n_overlap": n_overlap, "n_written": int(len(free))})

    n_from_compartments = int(fine.notna().sum())
    maj = adata.obs[major_key].astype(str)
    n_carried = 0
    if carry_terminal:
        unset = fine.isna()
        n_carried = int(unset.sum())
        fine.loc[unset.values] = maj[unset.values].values
    n_uncovered = int(fine.isna().sum())
    if n_uncovered:
        warnings.append(f"{n_uncovered} cells left unlabeled (carry_terminal={carry_terminal})")

    adata.obs[fine_key] = pd.Categorical(fine.astype(str))
    adata.obs[facs_key] = facs.fillna("").astype(str)
    adata.obs[state_key] = state.fillna("").astype(str)

    dist = adata.obs[fine_key].value_counts()
    resolved = {"compartments_root": str(compartments_root) if compartments_root else None,
                "n_sources": len(per_source), "fine_key": fine_key, "carry_terminal": carry_terminal}
    cp = session.checkpoint("merge_fine_annotations", x_state=session.manifest.x_state, params=resolved)
    summary = {
        "fine_key": fine_key, "n_parent_cells": n_parent, "n_sources": len(per_source),
        "n_from_compartments": n_from_compartments, "n_carried_terminal": n_carried,
        "n_uncovered": n_uncovered, "n_fine_types": int(adata.obs[fine_key].nunique()),
        "per_source": per_source,
        "label_distribution": {str(k): int(v) for k, v in dist.head(40).items()},
    }
    return S.success("merge_fine_annotations", summary=summary, checkpoint=cp.path,
                     warnings=warnings, determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     params=resolved, suggested_next_tools=["plots", "report"])
