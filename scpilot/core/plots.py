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

import time

from scpilot import schemas as S
from scpilot.tools import register

# scpilot plotting policy → fed to the vendored plotting_cfg / fit_and_save.
_PLOTTING = {
    "max_w_col": 1.5, "max_h_col": 1.5, "square_limit_col": 1.0,
    "start_col": 0.5, "step_col": 0.25,
    "column_width_in": 3.5, "dpi_save": 300, "min_font_pt": 5, "base_font_pt": 7,
}
_QC_KEYS = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]


def _cfg(session):
    from scpilot.vendor.config import PipelineConfig
    return PipelineConfig(out_dir=str(session.out), plotting=dict(_PLOTTING))


def _artifacts_from_fit(fit, cfg) -> list[S.Artifact]:
    col = cfg.plotting["column_width_in"]
    w, h = fit.size_col
    return [S.artifact_png(p, width_in=w * col, height_in=h * col,
                           dpi=cfg.plotting["dpi_save"],
                           description=f"{w}x{h} col, font {fit.font_pt:g}pt")
            for p in fit.path]


@register("plots", mutating=False,
          description="Render a figure (umap/qc_violin/hvg/pca_variance) via the auto-fit harness; "
                      "returns the saved PNG(s) sized to the column policy (plan B5).")
def plots(session, *, kind: str = "umap", color: str | None = None,
          keys: list | None = None, groupby: str | None = None,
          marker_groups: dict | None = None, **params) -> S.ToolResult:
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
            if "X_umap" not in adata.obsm:
                return S.error("plots", "invalid_state", "no X_umap — run cluster first",
                               recoverable=True, suggested_next_tools=["cluster"])
            ck = color or ("leiden" if "leiden" in adata.obs else adata.obs.columns[0])
            if ck not in adata.obs:
                return S.error("plots", "missing_input", f"color key '{ck}' not in obs", recoverable=True)
            fit = P.save_umap(adata, cfg, art_dir / f"umap_{ck}", color=ck)
            label = f"umap colored by {ck}"

        elif kind == "qc_violin":
            ks = keys or [k for k in _QC_KEYS if k in adata.obs]
            if not ks:
                return S.error("plots", "invalid_state", "no QC metrics — run qc_metrics first",
                               recoverable=True, suggested_next_tools=["qc_metrics"])
            fit = P.save_violin(adata, cfg, art_dir / "qc_violin", keys=ks, groupby=groupby)
            label = f"QC violins: {ks}"

        elif kind == "hvg":
            if "highly_variable" not in adata.var:
                return S.error("plots", "invalid_state", "no HVG — run preprocess first",
                               recoverable=True, suggested_next_tools=["preprocess"])
            fit = P.save_highly_variable_genes(adata, cfg, art_dir / "hvg")
            label = "highly variable genes"

        elif kind == "pca_variance":
            if "pca" not in adata.uns:
                return S.error("plots", "invalid_state", "no PCA — run preprocess first",
                               recoverable=True, suggested_next_tools=["preprocess"])
            fit = P.save_pca_variance_ratio(adata, cfg, art_dir / "pca_variance")
            label = "PCA variance ratio"

        elif kind == "dotplot":
            # annotation dotplot: sc.pl.dotplot with marker panels AS A DICT → cell-type
            # brackets + labels above the x-axis (built-in var-group rendering).
            import scanpy as sc
            from scpilot.core.annotate import BROAD_MARKERS
            gb = groupby or ("major_cell_type" if "major_cell_type" in adata.obs else "leiden")
            if gb not in adata.obs:
                return S.error("plots", "invalid_state",
                               f"groupby '{gb}' absent — run annotate_broad/cluster first",
                               recoverable=True, suggested_next_tools=["annotate_broad"])
            src = marker_groups or BROAD_MARKERS
            groups = {ct: [g for g in gs if g in adata.var_names] for ct, gs in src.items()}
            groups = {ct: gs for ct, gs in groups.items() if gs}   # drop empty panels
            if not groups:
                return S.error("plots", "data_gate_failed", "no marker-panel genes present", recoverable=False)

            def build(size, font, draft=False):
                return P._scanpy_build(
                    lambda: sc.pl.dotplot(adata, groups, groupby=gb, show=False,
                                          dendrogram=False), size)
            fit = P.fit_and_save(build, cfg, art_dir / f"dotplot_{gb}", logger=None)
            label = f"dotplot (markers grouped by cell type) over {gb}"

        else:
            return S.error("plots", "missing_input",
                           f"unknown kind '{kind}' (umap|qc_violin|hvg|pca_variance|dotplot)", recoverable=True)
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
