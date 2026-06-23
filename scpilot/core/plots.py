"""Figure tools — scpilot plan B5 (vendored auto-fit harness).

Each plot is rendered in its package's own tutorial style (scanpy ``sc.pl.*`` via
the vendored ``save_*`` builders; scib / other packages route their own plotters
through the same ``fit_and_save``) and saved through the vendored auto-fit engine,
which searches the smallest journal-column size that has no clipping / no text
overlap / legible legend, actively adjusting font + legend + labels (knob ladder)
on a fixed canvas. Saved size obeys the user-confirmed policy (2026-06-10):

  min 0.5×0.5 col, max orientation-flexible {1×1.5, 1.5×1, 1×1} (never both >1).

Read-only (no checkpoint): writes PNGs under ``session.artifacts_dir`` and returns
them as ``Artifact``s with width/height(in)/dpi metadata.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from scpilot import schemas as S
from scpilot.tools import register

# F3: figure basenames are built from LLM/user-controlled values (color, basis, tag, groupby);
# the vendor save layer creates parent dirs, so an unsanitized "../../etc/x" would escape the
# session. `_art` strips every char outside [0-9A-Za-z_.-], drops leading/trailing separators
# (so a bare ".." can't survive), and asserts the resolved path stays under artifacts_dir.
_SAFE_NAME = re.compile(r"[^0-9A-Za-z_.-]+")


def _art(art_dir, name: str) -> Path:
    safe = _SAFE_NAME.sub("_", str(name)).strip("._") or "plot"
    p = Path(art_dir) / safe
    base = Path(art_dir).resolve()
    rp = p.resolve()
    if rp != base and base not in rp.parents:
        raise ValueError(f"unsafe artifact name: {name!r}")
    return p

# scpilot plotting policy → fed to the vendored plotting_cfg / fit_and_save.
_PLOTTING = {
    "max_w_col": 1.5, "max_h_col": 1.5, "square_limit_col": 1.0,
    "start_col": 0.5, "step_col": 0.25,
    "column_width_in": 3.5, "dpi_save": 300, "min_font_pt": 5, "base_font_pt": 7,
    "formats": ["svg", "png"],     # vector SVG deliverable + PNG preview for every figure
}
_QC_KEYS = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]


def _cfg(session):
    from scpilot.vendor.config import PipelineConfig
    return PipelineConfig(out_dir=str(session.out), plotting=dict(_PLOTTING))


def _artifacts_from_fit(fit, cfg) -> list[S.Artifact]:
    col = cfg.plotting["column_width_in"]
    w, h = fit.size_col
    desc = f"{w}x{h} col, font {fit.font_pt:g}pt"
    out = []
    for p in fit.path:
        if str(p).lower().endswith(".svg"):
            out.append(S.Artifact(path=p, kind="svg", description=desc + " (vector SVG)",
                                  meta={"width_in": round(w * col, 3), "height_in": round(h * col, 3)}))
        else:
            out.append(S.artifact_png(p, width_in=w * col, height_in=h * col,
                                      dpi=cfg.plotting["dpi_save"], description=desc))
    return out


@register("plots", mutating=False,
          description="Render a figure and return the saved PNG(s) (plan B5). "
                      "kind=umap (params: color, basis — e.g. basis=X_umap_harmony / X_umap_scvi "
                      "and color=sample_id/condition/major_cell_type for integration before/after "
                      "comparisons; many-category colors auto-use a generous canvas), "
                      "qc_violin (keys, groupby; tag='pre'/'post' for before/after QC), "
                      "scatter (QC: total_counts × n_genes_by_counts colored by pct_counts_mt), "
                      "qc_thresholds (chosen cutoffs overlaid on QC distributions — the param "
                      "justification figure; pass cutoffs={min_genes,max_pct_mt,...}), "
                      "resolution_sweep (n_clusters vs resolution + chosen knee; pass sweep + suggested), "
                      "hvg, pca_variance, "
                      "dotplot (annotation marker dotplot: groupby=major_cell_type, optional marker_groups; "
                      "cell-type rows ordered as a staircase under their marker brackets, and FAMILY-CONTIGUOUS "
                      "so subtypes stay together — e.g. all Macrophage* in one block (pass family_map to set "
                      "families explicitly, else derived from the label's leading token); vertical gene labels). "
                      "umap/qc_violin/hvg/pca_variance obey the journal-column size policy; the dotplot "
                      "auto-fits to the SMALLEST 0.5–2.0×0.5–2.0 col size with no text/dot overlap and a "
                      "size/colour legend ≤5% of the figure (many-category umap uses a generous canvas).")
def plots(session, *, kind: str = "umap", color: str | None = None,
          basis: str = "X_umap", keys: list | None = None, groupby: str | None = None,
          marker_groups: dict | None = None, order: list | None = None,
          family_map: dict | None = None, cluster_key: str | None = None,
          label_map: dict | None = None,
          tag: str | None = None, cutoffs: dict | None = None, **params) -> S.ToolResult:
    import matplotlib
    matplotlib.use("Agg")  # headless (MCP/CLI: no display)

    from scpilot.vendor import plotting as P

    t0 = time.time()
    adata = session.adata
    cfg = _cfg(session)
    art_dir = session.artifacts_dir
    art_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    try:
        if kind == "umap":
            if basis not in adata.obsm:
                return S.error("plots", "invalid_state",
                               f"no '{basis}' in obsm{sorted(adata.obsm)} — run cluster/integrate first",
                               recoverable=True, suggested_next_tools=["cluster"])
            ck = color or ("leiden" if "leiden" in adata.obs else adata.obs.columns[0])
            if ck not in adata.obs:
                return S.error("plots", "missing_input", f"color key '{ck}' not in obs", recoverable=True)
            suf = "" if basis == "X_umap" else "_" + basis.removeprefix("X_umap_").removeprefix("X_")
            fit = P.save_umap(adata, cfg, _art(art_dir, f"umap{suf}_{ck}"), color=ck, basis=basis)
            label = f"{basis} colored by {ck}"

        elif kind == "qc_violin":
            ks = keys or [k for k in _QC_KEYS if k in adata.obs]
            if not keys and "doublet_score" in adata.obs:
                ks = ks + ["doublet_score"]
            if not ks:
                return S.error("plots", "invalid_state", "no QC metrics — run qc_metrics first",
                               recoverable=True, suggested_next_tools=["qc_metrics"])
            base = _art(art_dir, f"qc_violin_{tag}" if tag else "qc_violin")
            fit = P.save_violin(adata, cfg, base, keys=ks, groupby=groupby)
            label = f"QC violins ({tag or 'all'}): {ks}"

        elif kind == "scatter":
            # QC scatter (default total_counts × n_genes_by_counts, colored by pct_counts_mt)
            x = params.get("x", "total_counts")
            y = params.get("y", "n_genes_by_counts")
            for cn in (x, y):
                if cn not in adata.obs:
                    return S.error("plots", "invalid_state",
                                   f"'{cn}' not in obs — run qc_metrics first",
                                   recoverable=True, suggested_next_tools=["qc_metrics"])
            cc = (color or "pct_counts_mt")
            cc = cc if cc in adata.obs else None
            base = _art(art_dir, f"qc_scatter_{tag}" if tag else "qc_scatter")
            fit = P.save_scatter(adata, cfg, base, x, y, color=cc)
            label = f"scatter {x} vs {y}" + (f" (color {cc})" if cc else "")

        elif kind == "qc_thresholds":
            ks = keys or [k for k in _QC_KEYS if k in adata.obs]
            if not ks:
                return S.error("plots", "invalid_state", "no QC metrics — run qc_metrics first",
                               recoverable=True, suggested_next_tools=["qc_metrics"])
            cut = cutoffs or {}
            # map the flat qc_filter cutoffs to per-metric {min,max} bounds
            bounds = {
                "n_genes_by_counts": {"min": cut.get("min_genes"), "max": cut.get("max_genes")},
                "total_counts": {"min": cut.get("min_counts"), "max": cut.get("max_counts")},
                "pct_counts_mt": {"max": cut.get("max_pct_mt")},
            }
            base = _art(art_dir, f"qc_thresholds_{tag}" if tag else "qc_thresholds")
            fit = P.save_qc_thresholds(adata, cfg, base, keys=ks, cutoffs=bounds)
            label = "QC cutoffs over distributions"

        elif kind == "resolution_sweep":
            sweep = params.get("sweep")
            if not sweep:
                return S.error("plots", "missing_input",
                               "resolution_sweep needs sweep=[{resolution,n_clusters},...]",
                               recoverable=True, suggested_next_tools=["cluster_sweep"])
            base = _art(art_dir, f"resolution_sweep_{tag}" if tag else "resolution_sweep")
            fit = P.save_resolution_sweep(cfg, base, sweep, suggested=params.get("suggested"))
            label = "resolution sweep (n_clusters vs resolution)"

        elif kind == "hvg":
            if "highly_variable" not in adata.var:
                return S.error("plots", "invalid_state", "no HVG — run preprocess first",
                               recoverable=True, suggested_next_tools=["preprocess"])
            fit = P.save_highly_variable_genes(adata, cfg, _art(art_dir, "hvg"))
            label = "highly variable genes"

        elif kind == "pca_variance":
            if "pca" not in adata.uns:
                return S.error("plots", "invalid_state", "no PCA — run preprocess first",
                               recoverable=True, suggested_next_tools=["preprocess"])
            fit = P.save_pca_variance_ratio(adata, cfg, _art(art_dir, "pca_variance"))
            label = "PCA variance ratio"

        elif kind == "dotplot":
            # annotation dotplot: sc.pl.dotplot with marker panels AS A DICT → cell-type
            # brackets + labels above the x-axis (built-in var-group rendering).
            from scpilot.core.annotate import BROAD_MARKERS, derive_dotplot_markers
            gb = groupby or ("major_cell_type" if "major_cell_type" in adata.obs else "leiden")
            if gb not in adata.obs:
                return S.error("plots", "invalid_state",
                               f"groupby '{gb}' absent — run annotate_broad/cluster first",
                               recoverable=True, suggested_next_tools=["annotate_broad"])
            # marker source priority: caller panel > DATA-DRIVEN derivation from this groupby's DE
            # (organism-agnostic — works for mouse fine types) > human BROAD_MARKERS fallback.
            src = marker_groups
            if src is None and cluster_key and label_map:
                # broad dotplot: rows = cell types (gb=major_cell_type) but DE is on the leiden
                # cluster_key — derive cell-type panels by mapping the cluster DE through label_map
                # (no DE recompute on the cell-type key needed).
                try:
                    src = derive_dotplot_markers(adata, cluster_key=cluster_key,
                                                 label_map={str(k): str(v) for k, v in label_map.items()},
                                                 order=order, family_map=family_map)
                except Exception:  # noqa: BLE001 — fall through to gb-DE / fixed panel
                    src = None
            if src is None:
                rg = adata.uns.get("rank_genes_groups")
                if rg and rg.get("params", {}).get("groupby") == gb:
                    try:
                        labels = [str(c) for c in adata.obs[gb].astype("category").cat.categories]
                        # `order` = caller lineage order (LAYOUT only) so rows group by cell
                        # family, not by abundance; staircase then follows this panel order.
                        src = derive_dotplot_markers(adata, cluster_key=gb,
                                                     label_map={lab: lab for lab in labels},
                                                     order=order, family_map=family_map)
                    except Exception:  # noqa: BLE001 — fall through to the fixed panel
                        src = None
                if not src:
                    src = BROAD_MARKERS
            # honour an explicit lineage `order` even when marker_groups was supplied directly
            if order and marker_groups:
                src = {ct: src[ct] for ct in order if ct in src} | {ct: gs for ct, gs in src.items() if ct not in order}
            groups = {ct: [g for g in gs if g in adata.var_names] for ct, gs in src.items()}
            groups = {ct: gs for ct, gs in groups.items() if gs}   # drop empty panels
            if not groups:
                return S.error("plots", "data_gate_failed",
                               "no marker-panel genes present (pass marker_groups or run markers on this groupby)",
                               recoverable=False)

            # y-axis rows follow the (now lineage-ordered) panel order; save_dotplot's staircase
            # keeps each cell type under its own marker bracket in that biological order.
            present = list(adata.obs[gb].astype("category").cat.categories)
            cats_order = [ct for ct in groups if ct in present]
            cats_order += [ct for ct in present if ct not in cats_order]

            fit = P.save_dotplot(adata, cfg, _art(art_dir, f"dotplot_{gb}"), groups, gb,
                                 categories_order=cats_order, logger=None)
            label = f"dotplot (markers grouped by cell type) over {gb}"

        else:
            return S.error("plots", "missing_input",
                           f"unknown kind '{kind}' (umap|qc_violin|scatter|qc_thresholds|"
                           "resolution_sweep|hvg|pca_variance|dotplot)", recoverable=True)
    except Exception as exc:  # noqa: BLE001
        return S.error("plots", "internal", f"{type(exc).__name__}: {exc}")

    artifacts = _artifacts_from_fit(fit, cfg)
    if fit.warnings:
        warnings += fit.warnings
    summary = {
        "kind": kind, "label": label,
        "size_col": list(fit.size_col), "font_pt": fit.font_pt,
        "n_files": len(artifacts),
        "fit_at_max_failed": bool(fit.knobs.get("best_effort")),
    }
    return S.success("plots", summary=summary, artifacts=artifacts, warnings=warnings,
                     determinism_grade="B", duration_s=round(time.time() - t0, 3))
