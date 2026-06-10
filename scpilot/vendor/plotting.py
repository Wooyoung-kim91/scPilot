# =====================================================================
# VENDORED FROM scqc_pipeline @ source_hash debef308904633e1
#   source: /home/wykim/data/PDAC/scqc_pipeline/ (copied 2026-06-10)
# scpilot 베다링 정책: 독립 진화. import 경로·provenance 키·uns 키만
#   scpilot으로 적응했고 로직은 원본 유지. 재동기화 절차/원본 대비 diff는
#   scpilot/vendor/VENDORING.md 참조. scpilot 고유 코드는 여기 두지 말 것.
# =====================================================================
"""Figure style harness + publication-quality auto-fit engine.

All figure stages route through here so size / plot-kind / palette / colormap are
controlled from the profile's `plotting` block. The auto-fit engine searches the
smallest figure size (in journal-column units) that satisfies, for every saved
plot:

  1. size:      width ≤ 1.5 col, height ≤ 1.0 col (search starts 0.5 × 0.5 col)
  2. no clipping:  every artist inside the canvas (get_tightbbox ⊆ fig.bbox)
  3. text:      no overlap among tick labels / axis titles / plot title / legend
  4. elements:  categorical groups (violin/bar/strip) stay distinguishable
  5. font:      every text ≥ 5 pt

Saving uses the fixed canvas (no bbox_inches='tight') so the saved size equals the
chosen size and the no-clipping guarantee is real, not faked by expanding the canvas.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.transforms import Bbox

import scanpy as sc

from scpilot.vendor.config import PipelineConfig


# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
def plotting_cfg(cfg: PipelineConfig) -> dict:
    p = dict(cfg.plotting or {})
    p.setdefault("dpi", 100)
    p.setdefault("dpi_save", 300)
    p.setdefault("formats", ["png"])
    p.setdefault("theme", "whitegrid")
    p.setdefault("facecolor", "white")
    p.setdefault("column_width_in", 3.5)
    p.setdefault("max_w_col", 1.5)
    p.setdefault("max_h_col", 1.0)
    p.setdefault("start_col", 0.5)
    p.setdefault("step_col", 0.25)
    p.setdefault("min_font_pt", 5)
    p.setdefault("base_font_pt", 7)
    p.setdefault("min_category_px", 14)
    p.setdefault("palette", None)           # None → scanpy default categorical palette
    p.setdefault("cmap", None)              # None → scanpy/matplotlib default (viridis)
    p.setdefault("palette_overrides", {})
    p.setdefault("rotate_thresh", 6)        # rotate x tick labels only if longer than this
    p.setdefault("legend_marker_pt", 4)     # small legend dots
    p.setdefault("min_plot_frac", 0.45)     # data axes must keep ≥ this frac of fig width
    p.setdefault("min_plot_in", 1.3)        # ...and this many inches (so legend can't dominate)
    return p


def apply_style(cfg: PipelineConfig) -> None:
    p = plotting_cfg(cfg)
    sns.set_theme(style=p["theme"], context="paper")
    sc.settings.set_figure_params(dpi=p["dpi"], dpi_save=p["dpi_save"],
                                  facecolor=p["facecolor"])


def palette_for(cfg: PipelineConfig, key: str, categories: list) -> list:
    """Stable category→color list (so the same level is the same color everywhere)."""
    p = plotting_cfg(cfg)
    override = p["palette_overrides"].get(key, {})
    base = sns.color_palette(p["palette"], n_colors=max(len(categories), 1))
    colors = []
    for i, c in enumerate(categories):
        if str(c) in override:
            colors.append(override[str(c)])
        else:
            colors.append(matplotlib.colors.to_hex(base[i % len(base)]))
    return colors


@contextmanager
def _font_context(size: float):
    keys = ["font.size", "axes.titlesize", "axes.labelsize",
            "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
            "legend.title_fontsize", "figure.titlesize"]
    with plt.rc_context({k: size for k in keys}):
        yield


# --------------------------------------------------------------------------- #
# measurement / constraint checks
# --------------------------------------------------------------------------- #
def _text_artists(fig) -> list:
    out = []
    if fig._suptitle is not None:
        out.append(fig._suptitle)
    legends = list(getattr(fig, "legends", []))
    for ax in fig.axes:
        out += [ax.title, ax.xaxis.label, ax.yaxis.label]
        out += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        out += list(ax.texts)            # in-plot annotations (e.g. pca_variance_ratio PC labels)
        if ax.get_legend() is not None:
            legends.append(ax.get_legend())
    for leg in legends:                       # figure-level + axes legends
        out += list(leg.get_texts())
        if leg.get_title() is not None:
            out.append(leg.get_title())
    return [t for t in out if t.get_text().strip() and t.get_visible()]


def _data_axes(fig):
    """Primary data axes (exclude colorbar / legend-host axes)."""
    return [ax for ax in fig.axes
            if not getattr(ax, "_scqc_aux", False)
            and ax.get_label() not in ("<colorbar>",)]


def _bbox(artist, renderer):
    try:
        bb = artist.get_window_extent(renderer)
        if bb.width <= 0 or bb.height <= 0:
            return None
        return bb
    except Exception:
        return None


def _overlap_area(a: Bbox, b: Bbox) -> float:
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if (dx > 0 and dy > 0) else 0.0


def check_constraints(fig, p: dict, n_categories: int | None) -> list[str]:
    """Return list of constraint violations ([] = all satisfied)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []

    # 2) clipping — tightbbox must be within the canvas
    tight = fig.get_tightbbox(renderer)
    fb = fig.bbox
    eps = 1.0
    if (tight.x0 < fb.x0 - eps or tight.y0 < fb.y0 - eps
            or tight.x1 > fb.x1 + eps or tight.y1 > fb.y1 + eps):
        problems.append("clipping")

    # 3) text–text overlap
    texts = _text_artists(fig)
    bbs = [(t, _bbox(t, renderer)) for t in texts]
    bbs = [(t, b) for t, b in bbs if b is not None]
    thresh = 2.0  # px² — ignore shared-edge touches
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            if _overlap_area(bbs[i][1], bbs[j][1]) > thresh:
                problems.append("text_overlap")
                break
        if "text_overlap" in problems:
            break

    data_axes = _data_axes(fig) or fig.axes

    # 4) categorical element distinguishability (proxy: per-category pixel width)
    if n_categories and n_categories > 1 and data_axes:
        ax_w = data_axes[0].get_window_extent(renderer).width
        if ax_w / n_categories < p["min_category_px"]:
            problems.append("elements_cramped")

    # 5) minimum plot area — the data axes must not be squished by a legend/colorbar
    if data_axes:
        total_w = sum(ax.get_window_extent(renderer).width for ax in data_axes)
        dpi = fig.dpi
        if (total_w / fig.bbox.width < p["min_plot_frac"]
                or total_w / dpi < p["min_plot_in"]):
            problems.append("plot_too_small")

    # 6) font floor
    for t in texts:
        if t.get_fontsize() < p["min_font_pt"] - 1e-6:
            problems.append("font_too_small")
            break
    return problems


# --------------------------------------------------------------------------- #
# auto-fit search + save
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    path: list
    size_col: tuple
    font_pt: float
    knobs: dict
    warnings: list


def _size_grid(p: dict, cols: int = 1, rows: int = 1):
    """Column-unit grid. The 0.5–1.5w × 0.5–1.0h col range is PER PANEL, so a
    multi-panel composite scales with its panel grid (each panel stays legible)."""
    start, step = p["start_col"], p["step_col"]
    ws = np.round(np.arange(start * cols, p["max_w_col"] * cols + 1e-9, step * cols), 3)
    hs = np.round(np.arange(start * rows, p["max_h_col"] * rows + 1e-9, step * rows), 3)
    pairs = [(float(w), float(h)) for w in ws for h in hs]
    pairs.sort(key=lambda wh: (wh[0] * wh[1], wh[0] + wh[1]))  # smallest area first
    return pairs


def _knob_ladder(p: dict):
    """Escalating mitigations within a size: just reduce font (rotation is content-based)."""
    base, mn = p["base_font_pt"], p["min_font_pt"]
    fonts = sorted({base, max(mn, base - 1), mn}, reverse=True)
    return [{"font": f} for f in fonts]


# --------------------------------------------------------------------------- #
# generic harness "guidance" applied to whatever scanpy produced
# (we do NOT hand-build plots — only steer the scanpy figure: legend markers/font/
#  ncol/placement, colorbar font, and content-based x-tick rotation)
# --------------------------------------------------------------------------- #
import re as _re
_NUM_RE = _re.compile(r"^[\d.,eE+\-−×·%\s]*\d[\d.,eE+\-−×·%\s]*$")


def _all_legends(fig):
    legs = list(getattr(fig, "legends", []))
    for ax in fig.axes:
        if ax.get_legend() is not None:
            legs.append(ax.get_legend())
    return legs


def _legend_cats_colors(adata, key, cfg):
    """Categories + scanpy-assigned colors (read back after a plot), profile may override."""
    cats = list(adata.obs[key].astype("category").cat.categories)
    p = plotting_cfg(cfg)
    if p["palette"] is not None or p["palette_overrides"].get(key):
        return cats, palette_for(cfg, key, cats)
    colors = list(adata.uns.get(f"{key}_colors", []))
    if len(colors) < len(cats):
        colors = colors + ["#333333"] * (len(cats) - len(colors))
    return cats, colors[:len(cats)]


def _add_cat_legend(fig, labels, colors, *, font, marker_pt, fig_h_in):
    """Figure-level categorical legend, outside-right, small dots, auto ncol
    (constrained_layout reserves the space → never clips / never dominates)."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", markersize=marker_pt,
                      markerfacecolor=c, markeredgewidth=0, color=c, label=str(l))
               for l, c in zip(labels, colors)]
    rows_fit = max(1, int((fig_h_in * 72) / (font * 1.7)))
    ncol = max(1, -(-len(labels) // rows_fit))
    fig.legend(handles=handles, loc="outside right upper", frameon=False, fontsize=font,
               ncol=ncol, handletextpad=0.3, columnspacing=0.6, labelspacing=0.25,
               borderaxespad=0.2)


def _add_colorbar(fig, axes, values, cmap, *, font, label=None):
    """Figure-level colorbar, outside-right (constrained reserves space → no intrusion)."""
    import numpy as _np
    cmap = cmap or "viridis"
    vals = _np.asarray(values, dtype=float)
    norm = matplotlib.colors.Normalize(vmin=_np.nanmin(vals), vmax=_np.nanmax(vals))
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, location="right", fraction=0.05, pad=0.02,
                      shrink=0.85, aspect=20)
    if label:
        cb.set_label(label, fontsize=font)
    cb.ax.tick_params(labelsize=font)
    cb.ax._scqc_aux = True
    return cb


def _handle_color(h):
    for getter in ("get_markerfacecolor", "get_facecolor", "get_color"):
        if hasattr(h, getter):
            try:
                c = getattr(h, getter)()
                c = c[0] if hasattr(c, "__len__") and len(c) and hasattr(c[0], "__len__") else c
                return c
            except Exception:
                continue
    return "#333333"


def _tame_legends(fig, p, font, fig_h_in):
    """Set legend/colorbar text to the harness font. Categorical legends and colorbars
    are built by the wrappers (outside, small dots) — here we only normalize font and
    shrink an inline legend's markers (e.g. scanpy's hvg legend)."""
    mk = p["legend_marker_pt"]
    for leg in _all_legends(fig):
        for t in leg.get_texts():
            t.set_fontsize(font)
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(font)
        for h in getattr(leg, "legend_handles", []):
            if hasattr(h, "set_markersize"):
                h.set_markersize(mk)
    for ax in fig.axes:
        if getattr(ax, "_colorbar", None) is not None or ax.get_label() == "<colorbar>":
            ax.tick_params(labelsize=font)
            ax._scqc_aux = True


def _strip_common_prefix(texts):
    """Drop a shared leading word across category labels (e.g. 'Illumina ') so long
    x-labels get shorter. Word-boundary only; numeric/short labels untouched."""
    import os
    vals = [t for t in texts if t.strip()]
    if len(vals) < 2:
        return texts
    pre = os.path.commonprefix(vals)
    sp = pre.rfind(" ")
    if sp <= 0:
        return texts
    pre = pre[:sp + 1]
    return [t[len(pre):] if t.strip() and t.startswith(pre) else t for t in texts]


def _relabel_xaxis(ax, rotate_thresh):
    """Strip common prefix from x labels; rotate long non-numeric ones 45° (not 90°)."""
    from matplotlib.ticker import FixedLocator
    texts = [t.get_text() for t in ax.get_xticklabels()]
    if not any(t.strip() for t in texts):
        return
    stripped = _strip_common_prefix(texts)
    non_numeric = [s for s in stripped if s.strip() and not _NUM_RE.match(s)]
    long = non_numeric and max((len(s) for s in non_numeric), default=0) > rotate_thresh
    if stripped == texts and not long:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0); lbl.set_ha("center")
        return
    ticks = list(ax.get_xticks())
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels(stripped, rotation=45 if long else 0,
                       ha="right" if long else "center")


def _finalize_layout(fig, p, font, fig_h_in):
    """Draw → tame legends/colorbars → resize in-plot annotation text → strip/rotate
    long x-labels → redraw (reflow)."""
    fig.canvas.draw()
    _tame_legends(fig, p, font, fig_h_in)
    # scanpy hardcodes some in-plot annotation sizes (e.g. pca_variance_ratio PC
    # labels at 8pt, ignoring rcParams) — bring them to the harness font.
    for ax in _data_axes(fig):
        for t in ax.texts:
            t.set_fontsize(font)
        # x breathing room so edge labels (e.g. PC1) don't sit on the spine.
        # scanpy sets an explicit xlim, so expand it directly (margins is ignored).
        if ax.texts:
            xs = [t.get_position()[0] for t in ax.texts]
            if xs:
                lo, hi = min(xs), max(xs)
                pad = 0.04 * (hi - lo + 1)
                ax.set_xlim(lo - pad, hi + pad)
    for ax in _data_axes(fig):
        _relabel_xaxis(ax, p["rotate_thresh"])
    fig.canvas.draw()


def fit_and_save(build, cfg: PipelineConfig, base_path: Path, *,
                 n_categories: int | None = None, grid=(1, 1),
                 logger=None) -> FitResult:
    """Search smallest column-size satisfying all constraints, then save (fixed canvas).

    ``build(size_inches, font_pt) -> Figure`` must create and return the figure
    (so plots that own their figure, e.g. multi_panel violin / highly_variable_genes,
    are supported). The engine only sizes / measures / saves it.
    """
    p = plotting_cfg(cfg)
    col = p["column_width_in"]
    base_path = Path(base_path)

    def _final(size, knob):
        """Full (non-draft) render at the chosen size, then save."""
        with _font_context(knob["font"]):
            ff = build(size, knob["font"], draft=False)
            _finalize_layout(ff, p, knob["font"], size[1])
        paths = _save(ff, base_path, p)
        plt.close(ff)
        return paths

    rows, cols = grid
    for (w, h) in _size_grid(p, cols=cols, rows=rows):
        for knob in _knob_ladder(p):
            with _font_context(knob["font"]):
                # cheap draft render for the layout search (no stripplot / subsampled
                # points) — layout/text geometry matches the full render
                fig = build((w * col, h * col), knob["font"], draft=True)
                _finalize_layout(fig, p, knob["font"], h * col)
                problems = check_constraints(fig, p, n_categories)
            plt.close(fig)
            if not problems:
                paths = _final((w * col, h * col), knob)
                if logger:
                    logger.info("fit %s → %.1fx%.1f col, font %.0f",
                                base_path.name, w, h, knob["font"])
                return FitResult(paths, (w, h), knob["font"], knob, [])

    # could not satisfy within max size → best-effort at max, flag it
    max_size = (p["max_w_col"] * col * cols, p["max_h_col"] * col * rows)
    knob = {"font": p["min_font_pt"]}
    with _font_context(p["min_font_pt"]):
        fig = build(max_size, p["min_font_pt"], draft=True)
        _finalize_layout(fig, p, p["min_font_pt"], max_size[1])
        remaining = check_constraints(fig, p, n_categories)
        plt.close(fig)
    paths = _final(max_size, knob)
    warn = [f"fit-at-max-failed: {base_path.name}: {sorted(set(remaining))}"]
    warnings.warn(warn[0])
    if logger:
        logger.warning(warn[0])
    return FitResult(paths, (p["max_w_col"], p["max_h_col"]), p["min_font_pt"],
                     {"best_effort": True}, warn)


def _save(fig, base_path: Path, p: dict) -> list:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in p["formats"]:
        path = base_path.with_suffix(f".{fmt}")
        # fixed canvas: no bbox_inches='tight' (would silently grow beyond size cap)
        fig.savefig(path, dpi=p["dpi_save"], facecolor=p["facecolor"])
        out.append(str(path))
    return out


def _new_fig(size):
    return plt.figure(figsize=size, layout="constrained")


def _draft_view(adata, draft, n=3000):
    """Subsample for cheap draft renders during the size search (full data for final)."""
    if not draft or adata.n_obs <= n:
        return adata
    return sc.pp.subsample(adata, n_obs=n, copy=True, random_state=0)


def _adopt_fig(fig, size):
    """Resize a scanpy/seaborn-owned figure into our canvas with managed layout."""
    fig.set_size_inches(*size)
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        pass
    return fig


# --------------------------------------------------------------------------- #
# scanpy plot wrappers — thin: call the scanpy base function, hand the figure to
# the harness. The harness STEERS (resize / font / rotation / legend marker+ncol /
# colorbar font / constraints); it never hand-builds the plot.
# --------------------------------------------------------------------------- #
def _is_cat(adata, key):
    return key in adata.obs and (adata.obs[key].dtype == object
                                 or str(adata.obs[key].dtype) == "category")


def _default_cmap(cfg):
    return plotting_cfg(cfg)["cmap"] or "viridis"   # scanpy/matplotlib default continuous


def _scanpy_build(call, size):
    """Run a scanpy plotting call that owns its figure (no ax= support, e.g. hvg /
    pca_variance_ratio), then size it. tight_layout reliably reserves room for long
    axis titles (set_layout_engine('constrained') doesn't reflow scanpy's figure)."""
    plt.close("all")
    call()
    fig = plt.gcf()
    fig.set_size_inches(*size)
    try:
        fig.tight_layout(pad=0.6)
    except Exception:
        pass
    return fig


def save_violin(adata, cfg, base_path, keys, groupby=None, logger=None):
    """Harness lays out a constrained 1×N axes grid; scanpy draws each panel via ax=
    (multi_panel FacetGrid ignores constrained_layout and clips labels, so the
    harness provides the grid — data rendering stays 100% scanpy)."""
    keys = list(keys)
    n_cat = adata.obs[groupby].nunique() if groupby else len(keys)

    def build(size, font, draft=False):
        ad = _draft_view(adata, draft)
        fig = _new_fig(size)
        axes = fig.subplots(1, len(keys), squeeze=False)[0]
        for ax, m in zip(axes, keys):
            sc.pl.violin(ad, keys=m, groupby=groupby, ax=ax, show=False,
                         stripplot=not draft, jitter=0.4)
            ax.set_xlabel("")
        return fig
    return fit_and_save(build, cfg, base_path, n_categories=n_cat,
                        grid=(1, len(keys)), logger=logger)


def save_scatter(adata, cfg, base_path, x, y, color=None, logger=None):
    """Generic QC scatter (total_counts vs n_genes, continuous color). sc.pl.scatter's
    colorbar is an intruding inset with no placement param, so the harness renders the
    points with ax.scatter and owns the colorbar outside (viridis, default cmap)."""
    cmap = _default_cmap(cfg)

    def build(size, font, draft=False):
        ad = _draft_view(adata, draft)
        fig = _new_fig(size)
        ax = fig.subplots()
        c = ad.obs[color] if color else None
        ax.scatter(ad.obs[x], ad.obs[y], c=c, cmap=cmap, s=4, linewidths=0)
        ax.set_xlabel(x); ax.set_ylabel(y)
        if color:
            ax.set_title(color)
            _add_colorbar(fig, [ax], ad.obs[color], cmap, font=font, label=None)
        return fig
    return fit_and_save(build, cfg, base_path, logger=logger)


def save_highly_variable_genes(adata, cfg, base_path, logger=None):
    """sc.pl.highly_variable_genes(adata) — harness sizes it."""
    def build(size, font, draft=False):
        return _scanpy_build(lambda: sc.pl.highly_variable_genes(adata, show=False), size)
    return fit_and_save(build, cfg, base_path, logger=logger)


def save_pca_variance_ratio(adata, cfg, base_path, n_pcs=50, logger=None):
    """sc.pl.pca_variance_ratio(adata, n_pcs=..., log=True) — harness sizes it."""
    def build(size, font, draft=False):
        return _scanpy_build(lambda: sc.pl.pca_variance_ratio(
            adata, n_pcs=n_pcs, log=True, show=False), size)
    return fit_and_save(build, cfg, base_path, logger=logger)


def save_pca_diagnostic(adata, cfg, base_path, cat_key, cont_key, size_pt=2, logger=None):
    """Harness lays out a 2×2 grid; scanpy renders each panel (sc.pl.pca ax=) with its
    own legend/colorbar suppressed. The harness adds ONE shared legend (cat) + ONE
    shared colorbar (cont) outside — no redundant/intruding per-panel legends."""
    cmap = _default_cmap(cfg)
    mk = plotting_cfg(cfg)["legend_marker_pt"]

    def build(size, font, draft=False):
        ad = _draft_view(adata, draft)
        fig = _new_fig(size)
        axs = fig.subplots(2, 2)
        for c, dims in enumerate([(0, 1), (2, 3)]):
            sc.pl.pca(ad, color=cat_key, dimensions=dims, ax=axs[0][c], show=False,
                      size=size_pt, legend_loc=None, colorbar_loc=None)
            sc.pl.pca(ad, color=cont_key, dimensions=dims, ax=axs[1][c], show=False,
                      size=size_pt, color_map=cmap, legend_loc=None, colorbar_loc=None)
        cats, colors = _legend_cats_colors(ad, cat_key, cfg)
        _add_cat_legend(fig, cats, colors, font=font, marker_pt=mk, fig_h_in=size[1])
        _add_colorbar(fig, [axs[1][0], axs[1][1]], ad.obs[cont_key], cmap,
                      font=font, label=cont_key)
        return fig
    return fit_and_save(build, cfg, base_path, grid=(2, 2), logger=logger)


def save_umap(adata, cfg, base_path, color, size_pt=2, logger=None):
    """Harness lays out the axes; scanpy renders the points (sc.pl.umap ax=) with its
    legend/colorbar suppressed. The harness owns a small-dot legend (categorical) or a
    colorbar (continuous) outside — controlled, never clipped, never dominating."""
    cmap = _default_cmap(cfg)
    mk = plotting_cfg(cfg)["legend_marker_pt"]
    cat = _is_cat(adata, color)
    n = adata.obs[color].nunique() if cat else None

    def build(size, font, draft=False):
        ad = _draft_view(adata, draft)
        fig = _new_fig(size)
        ax = fig.subplots()
        sc.pl.umap(ad, color=color, size=size_pt, ax=ax, show=False,
                   color_map=cmap, legend_loc=None, colorbar_loc=None)
        if cat:
            cats, colors = _legend_cats_colors(ad, color, cfg)
            _add_cat_legend(fig, cats, colors, font=font, marker_pt=mk, fig_h_in=size[1])
        else:
            _add_colorbar(fig, [ax], ad.obs[color], cmap, font=font, label=color)
        return fig
    nc = n if (n and n <= 12) else None
    return fit_and_save(build, cfg, base_path, n_categories=nc, logger=logger)
