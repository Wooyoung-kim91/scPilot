"""Malignancy (CNV) track — genomic-position annotation (B12-pre) + infercnv (B12).

``annotate_genomic_positions`` is the REQUIRED preflight for CNV: the merged PDAC
``var`` carries only gene symbols (no coordinates), so ``infercnvpy.tl.infercnv``
cannot run until ``var[chromosome,start,end]`` is filled.

Design (PoC-validated 2026-06-10, GENCODE v44 basic GRCh38 × PDAC 40,237 symbols):

- **Coordinate source** — a pinned-release GENCODE GRCh38 GTF, downloaded once and
  kept in a sha256 content-addressed cache (determinism grade A). A user ``--gtf``
  (ideally the alignment reference) overrides it. ``gtfparse`` is required —
  infercnvpy raises ImportError without it.
- **2-pass symbol mapping** (safe inverse of ``var_names_make_unique``): pass 1 maps
  raw ``var_names`` on ``gene_name`` (so real hyphenated genes like ``HLA-A`` map
  correctly); pass 2 takes only the still-unmapped names matching ``^(.+)-\\d+$`` and
  remaps the base symbol (recovers make_unique suffixes). A plain trailing-``-\\d+``
  strip would corrupt real hyphenated genes, so it is NOT used.
- **Coverage gate — protein-coding, not overall.** Overall mapping rate is dragged
  down by unmapped lncRNA/clone contigs, so the gate metric is
  ``protein_coding_coverage`` = (GENCODE protein_coding genes the data gave coordinates
  to). >=0.8 ok / 0.6-0.8 warn (symbol drift / build mismatch) / <0.6 strong warn.
- **Invariant**: ``var`` columns are ADDED only (non-destructive); ``layers['counts']``
  and ``.X`` meaning are unchanged. GTF hash/build/coverage go into ``.uns['scpilot']``.

Malignancy (the malignancy CALL) follows the same split as Tier-1 annotation
(``annotation_review`` -> LLM -> ``apply_annotation``):

- ``cnv_score`` produces the per-cell CNV burden (deterministic evidence).
- ``malignancy_evidence`` (read-only) packages per-group MULTI-AXIS evidence for the LLM:
  CNV burden RELATIVE to the in-data non-malignant reference (data-driven, no absolute
  threshold), clonal-expansion signal (single-patient concentration), and OPTIONAL
  caller-supplied tumor/normal marker scores (no hardcoded panel). It emits NO call.
- ``apply_malignancy`` writes the LLM's per-group call into ``obs['malignancy']`` over a
  FIXED vocabulary {malignant, non_malignant, uncertain, not_applicable} + confidence +
  review_required. It enforces the HARD RULE deterministically: a 'malignant' call made
  WITHOUT CNV evidence is forced to review_required.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import urllib.request
from pathlib import Path

from scpilot import schemas as S
from scpilot.tools import register, require_capability

logging.getLogger("infercnvpy").setLevel(logging.ERROR)

# Pinned coordinate source (determinism grade A). GRCh38 = "basic" GENCODE release.
GENCODE = {
    "GRCh38": {
        "url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
               "release_44/gencode.v44.basic.annotation.gtf.gz",
        "filename": "gencode.v44.basic.annotation.gtf.gz",
        "release": "GENCODE v44 basic",
    },
    "GRCh37": {
        "url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
               "release_44/GRCh37_mapping/gencode.v44lift37.basic.annotation.gtf.gz",
        "filename": "gencode.v44lift37.basic.annotation.gtf.gz",
        "release": "GENCODE v44lift37 basic",
    },
}

_RUN = os.environ.get("SCPILOT_RUN_DIR", os.path.expanduser("~/data/scpilot_run"))
GTF_CACHE = Path(os.environ.get("SCPILOT_GTF_CACHE", os.path.join(_RUN, "gtf_cache")))

_DUP_SUFFIX = re.compile(r"^(.+)-\d+$")   # make_unique suffix (NOT a real hyphenated gene)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_gtf(gtf: str | None, genome_build: str) -> tuple[Path, dict]:
    """Return (gtf_path, source_meta). Order: explicit path -> cache -> download."""
    if gtf:
        p = Path(gtf).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--gtf not found: {p}")
        return p, {"type": "user", "name": p.name, "sha256": _sha256(p)[:16]}

    spec = GENCODE.get(genome_build)
    if spec is None:
        raise ValueError(f"unknown genome_build '{genome_build}' (use {sorted(GENCODE)})")
    GTF_CACHE.mkdir(parents=True, exist_ok=True)
    p = GTF_CACHE / spec["filename"]
    downloaded = False
    if not p.exists():
        tmp = p.with_suffix(p.suffix + ".part")
        urllib.request.urlretrieve(spec["url"], tmp)   # noqa: S310 (pinned GENCODE URL)
        tmp.replace(p)
        downloaded = True
    return p, {"type": "gencode", "name": spec["release"], "sha256": _sha256(p)[:16],
               "url": spec["url"], "downloaded": downloaded}


def _protein_coding_names(gtf_path: Path) -> set[str]:
    """gene_names typed protein_coding in the GTF (the CNV-relevant universe)."""
    import gtfparse

    df = gtfparse.read_gtf(str(gtf_path), usecols=["feature", "gene_name", "gene_type"])
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    g = df[df["feature"] == "gene"]
    return set(g.loc[g["gene_type"] == "protein_coding", "gene_name"])


def _classify_unmapped(name: str) -> str:
    """noncoding/clone contigs vs. genuinely unexpected symbols (drift suspects)."""
    if re.match(r"^(MIR|LINC|SNOR|RNU|AC\d|AL\d|AP\d|RP\d|CTD-|CTC-|CTB-|Z\d)", name) \
            or "orf" in name.lower() or "-AS" in name:
        return "noncoding/clone"
    return "other"


@register("annotate_genomic_positions", mutating=True, long_running=True,
          description="CNV preflight (plan B12-pre): fill var[chromosome,start,end] from a pinned GENCODE "
                      "GTF via a 2-pass gene_name map (recovers make_unique suffixes). Gate metric = "
                      "protein_coding_coverage (>=0.8 ok). Non-destructive (var columns added only). "
                      "REQUIRED before cnv_score / infercnv when var lacks coordinates.")
def annotate_genomic_positions(session, *, gtf: str | None = None, genome_build: str = "GRCh38",
                               pc_coverage_warn: float = 0.8, pc_coverage_fail: float = 0.6,
                               **params) -> S.ToolResult:
    if (err := require_capability("annotate_genomic_positions")) is not None:
        return err
    import infercnvpy as cnv
    import numpy as np
    import pandas as pd

    t0 = time.time()
    adata = session.adata
    n_total = int(adata.n_vars)
    if n_total == 0:
        return S.error("annotate_genomic_positions", "invalid_state", "empty var", recoverable=False)

    try:
        gtf_path, source = _resolve_gtf(gtf, genome_build)
    except FileNotFoundError as exc:
        return S.error("annotate_genomic_positions", "missing_input", str(exc), recoverable=True)
    except ValueError as exc:
        return S.error("annotate_genomic_positions", "invalid_state", str(exc), recoverable=False)
    except Exception as exc:  # noqa: BLE001 (network/download failure)
        return S.error("annotate_genomic_positions", "missing_input",
                       f"could not obtain GTF ({exc}); provide one with --gtf <path>", recoverable=True)

    var_names = list(adata.var_names)

    # Pass 1: raw var_names on gene_name (real hyphenated genes map here).
    cnv.io.genomic_position_from_gtf(str(gtf_path), adata, gtf_gene_id="gene_name", inplace=True)
    chrom = adata.var["chromosome"]
    mapped1 = chrom.notna()
    n1 = int(mapped1.sum())

    # Pass 2: still-unmapped names with a make_unique suffix -> remap the base symbol.
    base_of = {}
    for nm in (var_names[i] for i in range(n_total) if not bool(mapped1.iloc[i])):
        m = _DUP_SUFFIX.match(nm)
        if m:
            base_of[nm] = m.group(1)
    recovered = 0
    if base_of:
        import anndata as ad
        bases = sorted(set(base_of.values()))
        shell = ad.AnnData(X=np.zeros((1, len(bases)), dtype="float32"))
        shell.var_names = bases
        bcoord = {}
        try:
            cnv.io.genomic_position_from_gtf(str(gtf_path), shell, gtf_gene_id="gene_name", inplace=True)
            bcoord = {b: shell.var.loc[b, ["chromosome", "start", "end"]]
                      for b in bases if pd.notna(shell.var.loc[b, "chromosome"])}
        except (TypeError, KeyError):
            # infercnvpy adds a 'chr' prefix to a categorical that is all-NaN when ZERO
            # base symbols map -> TypeError. That case == nothing to recover, so skip.
            bcoord = {}
        for nm, base in base_of.items():
            row = bcoord.get(base)
            if row is not None:
                adata.var.loc[nm, ["chromosome", "start", "end"]] = row.values
                adata.var.loc[nm, "gene_name"] = base
                recovered += 1

    chrom = adata.var["chromosome"]
    mapped = chrom.notna()
    n_mapped = int(mapped.sum())

    # Protein-coding coverage = the CNV-relevant gate (overall rate is noise-dominated).
    pc_names = _protein_coding_names(gtf_path)
    data_bases = set(var_names) | {b for b in base_of.values()}
    pc_covered = len(pc_names & data_bases)
    pc_coverage = (pc_covered / len(pc_names)) if pc_names else 0.0

    # chromosome distribution + unmapped breakdown
    per_chrom = {str(k): int(v) for k, v in chrom[mapped].value_counts().items()}
    unmapped_names = [var_names[i] for i in range(n_total) if not bool(mapped.iloc[i])]
    from collections import Counter
    kinds = Counter(_classify_unmapped(n) for n in unmapped_names)
    grade = "A" if source["type"] in ("gencode", "user") else "B"

    warnings = []
    if pc_coverage < pc_coverage_fail:
        warnings.append(f"protein_coding_coverage={pc_coverage:.1%} < {pc_coverage_fail:.0%}: likely genome-build / "
                        "symbol-version mismatch (multi-GSE merge?). Re-check build, resolve symbol aliases, or "
                        "supply the alignment GTF via --gtf before running CNV.")
    elif pc_coverage < pc_coverage_warn:
        warnings.append(f"protein_coding_coverage={pc_coverage:.1%} in [{pc_coverage_fail:.0%},{pc_coverage_warn:.0%}): "
                        "possible old-symbol drift or build mismatch — CNV usable but verify.")

    summary = {
        "n_genes_total": n_total, "n_mapped": n_mapped,
        "overall_fraction": round(n_mapped / n_total, 4),
        "protein_coding_coverage": round(pc_coverage, 4),
        "pc_covered": pc_covered, "pc_total": len(pc_names),
        "pass1_mapped": n1, "make_unique_recovered": recovered,
        "n_unmapped": n_total - n_mapped, "unmapped_kind": dict(kinds),
        "genome_build": genome_build, "source": source,
        "reproducibility_grade": grade,
        "n_chromosomes": len(per_chrom), "per_chromosome_gene_counts": per_chrom,
        "unmapped_preview": unmapped_names[:15],
        "gate_pass": pc_coverage >= pc_coverage_fail,
    }

    cp = session.checkpoint("annotate_genomic_positions", x_state=session.manifest.x_state,
                            params={"genome_build": genome_build, "source": source})
    nxt = ["cnv_score"] if pc_coverage >= pc_coverage_fail else ["inspect"]
    return S.success("annotate_genomic_positions", summary=summary, checkpoint=cp.path,
                     determinism_grade=grade, duration_s=round(time.time() - t0, 3),
                     warnings=warnings, suggested_next_tools=nxt)


def _cnv_save_fig(session, base: str):
    """Save the current matplotlib figure as a no-overwrite PNG artifact (CNV figures are
    genome-wide / panel layouts, so they bypass the journal-column fit harness)."""
    import matplotlib.pyplot as plt

    p = session.artifact_path(f"{base}.png")
    plt.savefig(p, bbox_inches="tight", dpi=200)
    plt.close("all")
    return S.artifact_png(str(p), description=base)


def _save_cnv_plots(session, adata, cnv, *, celltype_key):
    """Emit the infercnvpy CNV evidence figures (chromosome heatmaps + cnv/standard UMAP panels)
    as artifacts. Fully defensive: any plot failure is swallowed (a figure must never fail the
    tool), mirroring benchmark's plot_results_table handling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scanpy as sc

    arts: list = []
    colors = ["cnv_leiden", "cnv_score"] + ([celltype_key] if (celltype_key and celltype_key in adata.obs) else [])

    if celltype_key and celltype_key in adata.obs:
        try:
            cnv.pl.chromosome_heatmap(adata, groupby=celltype_key, show=False)
            arts.append(_cnv_save_fig(session, "cnv_heatmap_celltype"))
        except Exception:  # noqa: BLE001
            plt.close("all")
    try:
        cnv.pl.chromosome_heatmap(adata, groupby="cnv_leiden", dendrogram=True, show=False)
        arts.append(_cnv_save_fig(session, "cnv_heatmap_cnvleiden"))
    except Exception:  # noqa: BLE001
        plt.close("all")
    # cnv-space UMAP panel (cnv_leiden / cnv_score / cell type)
    try:
        _, axes = plt.subplots(2, 2, figsize=(11, 11))
        flat = list(axes.ravel())
        for ax, c in zip(flat, colors):
            cnv.pl.umap(adata, color=c, ax=ax, show=False)
        for ax in flat[len(colors):]:
            ax.axis("off")
        arts.append(_cnv_save_fig(session, "cnv_umap_panel"))
    except Exception:  # noqa: BLE001
        plt.close("all")
    # the same colours on the standard expression UMAP, when present
    if "X_umap" in adata.obsm:
        try:
            _, axes = plt.subplots(2, 2, figsize=(12, 11))
            flat = list(axes.ravel())
            for ax, c in zip(flat, colors):
                sc.pl.umap(adata, color=c, ax=ax, show=False)
            for ax in flat[len(colors):]:
                ax.axis("off")
            arts.append(_cnv_save_fig(session, "umap_cnv_panel"))
        except Exception:  # noqa: BLE001
            plt.close("all")
    return arts


@register("cnv_score", mutating=True, long_running=True,
          description="CNV evidence (malignancy track) (plan B12): infercnvpy tl.infercnv -> cnv-space leiden -> per-cell "
                      "cnv_score. DETERMINISTIC EVIDENCE ONLY (no malignant/non-malignant call here — that is a "
                      "downstream multi-evidence judgment: CNV burden + tumor markers + reference + patient "
                      "expansion). reference_key/reference_cat set the baseline (e.g. condition=Normal, or a known "
                      "non-malignant immune/stromal cell type); None => average-of-all baseline (advisory-only). "
                      "REQUIRES annotate_genomic_positions first (var coordinates).")
def cnv_score(session, *, reference_key: str | None = None, reference_cat: list | None = None,
              layer: str | None = None, groupby: str | None = None, window_size: int = 100,
              step: int = 10, leiden_resolution: float = 1.0, seed: int = 0, **params) -> S.ToolResult:
    if (err := require_capability("cnv_score")) is not None:
        return err
    import infercnvpy as cnv
    import numpy as np
    import pandas as pd

    t0 = time.time()
    adata = session.adata

    if "chromosome" not in adata.var.columns or adata.var["chromosome"].notna().sum() == 0:
        return S.error("cnv_score", "invalid_state",
                       "var has no genomic coordinates — run annotate_genomic_positions first",
                       recoverable=True, suggested_next_tools=["annotate_genomic_positions"])
    if reference_key is not None and reference_key not in adata.obs.columns:
        return S.error("cnv_score", "data_gate_failed",
                       f"reference_key '{reference_key}' absent in obs", recoverable=True)
    if reference_key is not None and reference_cat:
        present = set(adata.obs[reference_key].astype(str).unique())
        missing = [c for c in reference_cat if str(c) not in present]
        if missing:
            return S.error("cnv_score", "data_gate_failed",
                           f"reference_cat {missing} not found in obs['{reference_key}'] "
                           f"(present: {sorted(present)[:10]})", recoverable=True)

    n_coord = int(adata.var["chromosome"].notna().sum())
    advisory = reference_key is None
    warnings = []
    if advisory:
        warnings.append("no reference_key/reference_cat: CNV is scored against the average of ALL cells "
                        "(advisory-only). Provide a known non-malignant reference (e.g. condition=Normal or "
                        "immune/stromal cell type) for a trustworthy malignant/normal contrast.")

    # 1) infercnv on log-normalized expression (genes lacking coordinates are auto-excluded)
    cnv.tl.infercnv(adata, reference_key=reference_key, reference_cat=reference_cat,
                    layer=layer, window_size=window_size, step=step, inplace=True)
    # 2) cnv-space embedding -> neighbors -> leiden clusters
    cnv.tl.pca(adata)
    cnv.pp.neighbors(adata)
    import scanpy as sc
    sc.settings.verbosity = 0
    cnv.tl.leiden(adata, resolution=leiden_resolution, random_state=seed)
    # 3) per-cell + per-cnv-cluster CNV burden + a cnv-space UMAP (for cnv.pl.umap panels)
    cnv.tl.cnv_score(adata, groupby="cnv_leiden")
    try:
        cnv.tl.umap(adata)
    except Exception:  # noqa: BLE001 — UMAP is for plotting only; never fail scoring on it
        pass

    scores = adata.obs["cnv_score"].astype(float)
    per_cnv_leiden = (adata.obs.groupby("cnv_leiden", observed=True)["cnv_score"]
                      .agg(["size", "mean"]).reset_index()
                      .rename(columns={"size": "n_cells", "mean": "mean_cnv_score"})
                      .sort_values("mean_cnv_score", ascending=False))

    # reference vs rest contrast (the key signal for downstream malignancy judgment)
    ref_contrast = None
    if reference_key is not None and reference_cat:
        is_ref = adata.obs[reference_key].astype(str).isin([str(c) for c in reference_cat])
        ref_contrast = {"reference_mean_cnv": round(float(scores[is_ref].mean()), 4),
                        "nonreference_mean_cnv": round(float(scores[~is_ref].mean()), 4),
                        "n_reference": int(is_ref.sum())}

    # cross-tab CNV clusters against an existing cell-type annotation, if present
    celltype_cnv = None
    if groupby is None:
        for cand in ("major_cell_type", "celltype_consensus", "major_cell_type_scvi"):
            if cand in adata.obs.columns:
                groupby = cand
                break
    if groupby and groupby in adata.obs.columns:
        ct = (adata.obs.groupby(groupby, observed=True)["cnv_score"]
              .agg(["size", "mean"]).reset_index()
              .rename(columns={"size": "n_cells", "mean": "mean_cnv_score"})
              .sort_values("mean_cnv_score", ascending=False))
        cnv_csv = session.artifact_path("cnv_by_celltype.csv")   # no-overwrite on re-run (P1-2)
        celltype_cnv = S.table_preview(ct, full_csv=str(cnv_csv))
        ct.to_csv(cnv_csv, index=False)

    summary = {
        "n_cells": int(adata.n_obs), "n_genes_with_coords": n_coord,
        "reference_key": reference_key, "reference_cat": reference_cat,
        "advisory_only": advisory,
        "cnv_score_mean": round(float(scores.mean()), 4),
        "cnv_score_median": round(float(scores.median()), 4),
        "n_cnv_clusters": int(adata.obs["cnv_leiden"].nunique()),
        "reference_contrast": ref_contrast,
        "note": "cnv_score is per-cell CNV burden; HIGH = more aberrant (candidate malignant). "
                "This tool emits EVIDENCE only — the malignant/non-malignant call is downstream.",
    }
    tables = {"cnv_by_cnv_leiden": S.table_preview(per_cnv_leiden)}
    if celltype_cnv is not None:
        tables["cnv_by_celltype"] = celltype_cnv

    # CNV evidence figures (chromosome heatmaps + cnv/standard UMAP panels) — defensive
    cnv_arts = _save_cnv_plots(session, adata, cnv, celltype_key=groupby)

    cp = session.checkpoint("cnv_score", x_state=session.manifest.x_state,
                            params={"reference_key": reference_key, "reference_cat": reference_cat,
                                    "window_size": window_size, "step": step,
                                    "leiden_resolution": leiden_resolution, "seed": seed})
    return S.success("cnv_score", summary=summary, tables=tables, artifacts=cnv_arts, checkpoint=cp.path,
                     determinism_grade="B", duration_s=round(time.time() - t0, 3),
                     warnings=warnings,
                     suggested_next_tools=["malignancy_evidence", "annotation_review", "markers"])


# --------------------------------------------------------------------------- #
# Malignancy CALL — evidence packager + apply (Tier-1-style split)
# --------------------------------------------------------------------------- #
MALIGNANCY_VOCAB = ("malignant", "non_malignant", "uncertain", "not_applicable")
UNS_MAL = "scpilot_malignancy"


def _score_genes(adata, genes, name):
    """Per-cell sc.tl.score_genes for caller-supplied markers (None-safe)."""
    import scanpy as sc
    present = [g for g in (genes or []) if g in adata.var_names]
    if not present:
        return None, []
    sc.tl.score_genes(adata, present, score_name=name, ctrl_size=min(50, max(10, len(present) * 5)))
    return adata.obs[name].astype(float), present


@register("malignancy_evidence", mutating=False,
          description="Malignancy EVIDENCE for the LLM (read-only, no call, no hardcoded panel/threshold). "
                      "Per group (cnv_leiden or a cell-type key) it packages: CNV burden RELATIVE to the in-data "
                      "non-malignant reference (reference_key/reference_cat) — group mean, ratio, and fraction of "
                      "cells above the reference 95th pct (data-driven, NOT an absolute cutoff); clonal-expansion "
                      "signal (single-patient concentration via sample_key); and OPTIONAL caller-supplied "
                      "tumor_markers/normal_markers scores. Requires cnv_score first (obs['cnv_score']). The LLM "
                      "weighs ALL axes (HARD RULE: never epithelial markers alone) then calls apply_malignancy.")
def malignancy_evidence(session, *, groupby: str | None = None, reference_key: str | None = None,
                        reference_cat: list | None = None, sample_key: str = "sample_id",
                        tumor_markers: list | None = None, normal_markers: list | None = None,
                        ref_quantile: float = 0.95, min_cells: int = 20, **params) -> S.ToolResult:
    import json

    import numpy as np
    import pandas as pd

    t0 = time.time()
    adata = session.adata

    has_cnv = "cnv_score" in adata.obs.columns
    if not has_cnv:
        return S.error("malignancy_evidence", "invalid_state",
                       "obs['cnv_score'] absent — run cnv_score first (or, if cnv_available=false, "
                       "assemble marker+reference+expansion evidence and flag review_required).",
                       recoverable=True, suggested_next_tools=["cnv_score"])

    if groupby is None:
        for cand in ("cnv_leiden", "major_cell_type", "celltype_consensus", "leiden"):
            if cand in adata.obs.columns:
                groupby = cand
                break
    if groupby is None or groupby not in adata.obs.columns:
        return S.error("malignancy_evidence", "invalid_state",
                       f"groupby '{groupby}' absent — run cnv_score / cluster first", recoverable=True,
                       suggested_next_tools=["cnv_score"])
    if reference_key is not None and reference_key not in adata.obs.columns:
        return S.error("malignancy_evidence", "data_gate_failed",
                       f"reference_key '{reference_key}' absent in obs", recoverable=True)

    obs_g = adata.obs[groupby].astype(str)
    cnv = adata.obs["cnv_score"].astype(float)

    # reference baseline = the in-data non-malignant cells (data-driven; no absolute cutoff)
    advisory = not (reference_key and reference_cat)
    if advisory:
        ref_mask = np.ones(adata.n_obs, dtype=bool)        # weak baseline = all cells
        ref_desc = "all cells (advisory-only)"
    else:
        ref_mask = adata.obs[reference_key].astype(str).isin([str(c) for c in reference_cat]).values
        ref_desc = f"{reference_key} in {list(reference_cat)}"
    ref_cnv = cnv.values[ref_mask]
    ref_mean = float(ref_cnv.mean()) if ref_cnv.size else float("nan")
    ref_p = float(np.quantile(ref_cnv, ref_quantile)) if ref_cnv.size else float("nan")

    tumor_score, tumor_used = _score_genes(adata, tumor_markers, "_mal_tumor_score")
    normal_score, normal_used = _score_genes(adata, normal_markers, "_mal_normal_score")
    has_sample = sample_key in adata.obs.columns

    payloads, rows = [], []
    for cl, idx in obs_g.groupby(obs_g, observed=True).groups.items():
        mask = obs_g.values == str(cl)
        n_cells = int(mask.sum())
        g_cnv = cnv.values[mask]
        evidence = {
            "group": str(cl), "n_cells": n_cells,
            "cnv_burden": {
                "mean_cnv_score": round(float(g_cnv.mean()), 4),
                "reference_mean": round(ref_mean, 4),
                "ratio_to_reference": (round(float(g_cnv.mean() / ref_mean), 3)
                                       if ref_mean not in (0.0, float("nan")) else None),
                "frac_above_reference_q": round(float((g_cnv > ref_p).mean()), 3),
            },
        }
        # clonal expansion: a malignant clone is typically dominated by ONE patient;
        # immune/stromal compartments are shared across patients.
        if has_sample:
            sv = adata.obs[sample_key].astype(str)[mask].value_counts(normalize=True)
            evidence["clonal_expansion"] = {
                "top_sample_fraction": round(float(sv.iloc[0]), 3),
                "n_samples": int(sv.size),
                "dominant_sample": str(sv.index[0]),
            }
        if tumor_score is not None:
            evidence["tumor_marker_score"] = round(float(tumor_score.values[mask].mean()), 4)
        if normal_score is not None:
            evidence["normal_marker_score"] = round(float(normal_score.values[mask].mean()), 4)
        if n_cells < min_cells:
            evidence["flag"] = "tiny_group"
        payloads.append(evidence)
        rows.append({"group": str(cl), "n_cells": n_cells,
                     "mean_cnv": evidence["cnv_burden"]["mean_cnv_score"],
                     "cnv_ratio_to_ref": evidence["cnv_burden"]["ratio_to_reference"],
                     "frac_above_ref_q": evidence["cnv_burden"]["frac_above_reference_q"],
                     "top_sample_frac": evidence.get("clonal_expansion", {}).get("top_sample_fraction")})

    # clean up scratch obs columns (read-only contract)
    for c in ("_mal_tumor_score", "_mal_normal_score"):
        if c in adata.obs.columns:
            del adata.obs[c]

    art_dir = session.artifacts_dir
    art_dir.mkdir(parents=True, exist_ok=True)
    json_path = session.artifact_path("malignancy_evidence.json")   # no-overwrite (P1-2)
    json_path.write_text(json.dumps({
        "groupby": groupby, "reference": ref_desc, "advisory_only": advisory,
        "reference_quantile": ref_quantile,
        "tumor_markers_used": tumor_used, "normal_markers_used": normal_used,
        "vocabulary": list(MALIGNANCY_VOCAB),
        "instruction": "Weigh ALL axes per group: CNV burden vs the in-data reference (ratio + "
                       "frac_above_reference_q), clonal expansion (top_sample_fraction high => clone-like), "
                       "and any marker scores. HARD RULE: do NOT call malignant on epithelial markers alone — "
                       "require CNV and/or expansion support. Then call apply_malignancy with the group->label "
                       f"map over {list(MALIGNANCY_VOCAB)} + confidence + review_required.",
        "groups": payloads}, indent=2, default=str))

    warnings = []
    if advisory:
        warnings.append("no reference_cat: CNV burden is relative to ALL cells (advisory). Provide a known "
                        "non-malignant reference for a trustworthy contrast.")
    if not has_sample:
        warnings.append(f"sample_key '{sample_key}' absent — no clonal-expansion signal.")
    if not (tumor_used or normal_used):
        warnings.append("no tumor/normal markers supplied — call rests on CNV + expansion only "
                        "(supply caller markers to strengthen, but never as the sole basis).")

    summary = {
        "groupby": groupby, "n_groups": len(payloads), "reference": ref_desc,
        "advisory_only": advisory, "reference_mean_cnv": round(ref_mean, 4),
        "tumor_markers_used": tumor_used, "normal_markers_used": normal_used,
        "vocabulary": list(MALIGNANCY_VOCAB), "evidence_input": str(json_path),
        "note": "Evidence only — no malignancy call. The LLM judges from CNV burden + clonal expansion "
                "+ markers, then writes obs['malignancy'] via apply_malignancy.",
    }
    return S.success("malignancy_evidence", summary=summary,
                     tables={"evidence": S.table_preview(pd.DataFrame(rows), max_rows=len(rows))},
                     artifacts=[S.Artifact(path=str(json_path), kind="json",
                                           description="per-group multi-axis malignancy evidence for the LLM")],
                     warnings=warnings, determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["apply_malignancy"])


def _save_cnv_status_plots(session, adata, *, status_key="cnv_status"):
    """cnv_status figures: a cnv-space + standard UMAP panel and a tumor-only chromosome
    heatmap. Defensive (never raises); needs the cnv-space data from cnv_score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scanpy as sc

    arts: list = []
    try:
        import infercnvpy as cnv
    except Exception:  # noqa: BLE001
        return arts
    try:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        cnv.pl.umap(adata, color=status_key, ax=ax1, show=False)
        if "X_umap" in adata.obsm:
            sc.pl.umap(adata, color=status_key, ax=ax2, show=False)
        else:
            ax2.axis("off")
        arts.append(_cnv_save_fig(session, "cnv_status_panel"))
    except Exception:  # noqa: BLE001
        plt.close("all")
    try:
        if bool((adata.obs[status_key].astype(str) == "tumor").any()):
            cnv.pl.chromosome_heatmap(adata[adata.obs[status_key].astype(str) == "tumor"], show=False)
            arts.append(_cnv_save_fig(session, "cnv_heatmap_tumor"))
    except Exception:  # noqa: BLE001
        plt.close("all")
    return arts


@register("apply_malignancy", mutating=True,
          description="Write the LLM's malignancy call into obs['malignancy'] over the FIXED vocabulary "
                      "{malignant, non_malignant, uncertain, not_applicable} — the group->label map the LLM "
                      "inferred from malignancy_evidence. Deterministic given the map (replayable). Enforces the "
                      "HARD RULE: a 'malignant' call made without CNV evidence (no obs['cnv_score']) is forced to "
                      "review_required. Also writes obs['malignancy_confidence'] / obs['malignancy_review_required'].")
def apply_malignancy(session, *, groupby: str = "cnv_leiden", labels: dict | None = None,
                     confidence: dict | None = None, review_required: dict | None = None,
                     key: str = "malignancy", method: str = "CNV_marker_expansion_LLM",
                     unassigned: str = "not_applicable", **params) -> S.ToolResult:
    t0 = time.time()
    adata = session.adata
    if groupby not in adata.obs.columns:
        return S.error("apply_malignancy", "invalid_state",
                       f"groupby '{groupby}' absent — run cnv_score / cluster first", recoverable=True,
                       suggested_next_tools=["cnv_score"])
    if not labels:
        return S.error("apply_malignancy", "missing_input",
                       "no 'labels' map given (expected {group: malignancy_label} from the LLM)",
                       recoverable=True, suggested_next_tools=["malignancy_evidence"])
    lab = {str(k): str(v) for k, v in labels.items()}
    bad = sorted({v for v in lab.values() if v not in MALIGNANCY_VOCAB})
    if bad:
        return S.error("apply_malignancy", "data_gate_failed",
                       f"labels outside the malignancy vocabulary {list(MALIGNANCY_VOCAB)}: {bad}",
                       recoverable=True)

    obs_g = adata.obs[groupby].astype(str)
    clusters = set(obs_g.unique())
    missing = sorted(clusters - set(lab))
    adata.obs[key] = obs_g.map(lambda c: lab.get(c, unassigned)).astype("category")
    # cnv_status (tumor/normal) DERIVED from the malignancy call — not hardcoded cnv_leiden IDs.
    adata.obs["cnv_status"] = (adata.obs[key].astype(str)
                               .map(lambda m: "tumor" if m == "malignant" else "normal").astype("category"))

    # HARD RULE: 'malignant' without CNV evidence cannot be trusted -> force review.
    has_cnv = "cnv_score" in adata.obs.columns
    rv = {str(k): bool(v) for k, v in (review_required or {}).items()}
    forced = []
    if not has_cnv:
        for c, v in lab.items():
            if v == "malignant" and not rv.get(c, False):
                rv[c] = True
                forced.append(c)
    adata.obs[f"{key}_review_required"] = obs_g.map(lambda c: rv.get(c, False)).astype(bool)
    if confidence:
        cf = {str(k): float(v) for k, v in confidence.items()}
        adata.obs[f"{key}_confidence"] = obs_g.map(lambda c: cf.get(c, float("nan"))).astype(float)

    adata.uns.setdefault(UNS_MAL, {})
    adata.uns[UNS_MAL] = {"method": method, "groupby": groupby, "key": key,
                          "labels": lab, "cnv_evidence_available": has_cnv,
                          "forced_review_no_cnv": forced}
    try:
        session.log_decision(S.DecisionEvent(
            decision_type="malignancy_call", choice=lab, candidates=list(MALIGNANCY_VOCAB),
            rationale=f"malignancy call from CNV+marker+expansion evidence (cnv_evidence={has_cnv})",
            stage="apply_malignancy", params={"groupby": groupby, "key": key}).to_dict())
    except Exception:  # noqa: BLE001
        pass

    dist = {str(k): int(v) for k, v in adata.obs[key].value_counts().items()}
    warnings = []
    if missing:
        warnings.append(f"{len(missing)} group(s) not in labels -> '{unassigned}': {missing}")
    if forced:
        warnings.append(f"{len(forced)} group(s) called malignant WITHOUT CNV evidence -> review_required "
                        f"forced (HARD RULE): {forced}")
    cnv_status_dist = {str(k): int(v) for k, v in adata.obs["cnv_status"].value_counts().items()}
    summary = {
        "key": key, "method": method, "groupby": groupby,
        "vocabulary": list(MALIGNANCY_VOCAB), "cnv_evidence_available": has_cnv,
        "n_groups_labeled": len(lab), "unlabeled_groups": missing,
        "label_distribution": dist, "forced_review_no_cnv": forced,
        "cnv_status_distribution": cnv_status_dist,    # tumor/normal derived from the call
    }
    status_arts = _save_cnv_status_plots(session, adata)
    cp = session.checkpoint("apply_malignancy", x_state=session.manifest.x_state,
                            params={"groupby": groupby, "key": key, "method": method})
    return S.success("apply_malignancy", summary=summary, artifacts=status_arts, warnings=warnings,
                     checkpoint=cp.path, determinism_grade="A", duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["annotation_review", "plots", "report"])
