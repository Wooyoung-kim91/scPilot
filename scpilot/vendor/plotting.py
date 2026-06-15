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
  4. elements:  categorical groups (violin/bar/strip) stay distinguishable; a 2-D dot
                grid (``dot_grid``) keeps every cell ≥ ``dot_min_cell_px`` in both dims
  5. font:      every text ≥ 5 pt
  6. legend:    axes flagged ``_scqc_legend`` stay ≤ ``legend_area_frac`` of the figure
                (when set) — e.g. a dotplot's size/colour legend not dominating the panel
  7. dot clip: on a ``dot_grid`` axis, no boundary dot's centre ± radius spills past the
                axis limits (clip_on dots get sliced at the spine, invisible to check 2)

Per-call ``overrides`` (see ``fit_and_save``) can widen the size cap / add the legend +
grid constraints for one plot without touching the profile (the dotplot uses a 0.5–2.0×
0.5–2.0 col square cap + a 5% legend cap). Saving uses the fixed canvas (no
bbox_inches='tight') so the saved size equals the chosen size and the no-clipping
guarantee is real, not faked by expanding the canvas.
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
    # scpilot adaptation: orientation-flexible cap — allow up to max_w/max_h in
    # either dimension but forbid BOTH exceeding this per-panel limit (so the saved
    # size is one of {1×1.5, 1.5×1, 1×1} col, never 1.5×1.5). None = off (scqc default).
    p.setdefault("square_limit_col", None)
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
    # scpilot adaptation: axes flagged ``_scqc_legend`` (e.g. a dotplot's size/colour legend)
    # may not exceed this fraction of the figure area. None = off (no legend-area cap).
    p.setdefault("legend_area_frac", None)
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
    """Return list of constraint violations ([] = all satisfied).

    All pixel thresholds (``min_category_px`` / ``dot_min_cell_px`` / ``dot_min_row_px`` /
    ``text_shrink_px``) are defined at the layout DPI (``p['dpi']``, default 100), so the
    figure DPI is pinned here before measuring. This decouples the *physical* layout checks
    from the raster save DPI (``dpi_save``=300) and from any ambient rcParams figure.dpi —
    otherwise the same panel would 'pass' or 'fail' depending on the environment's DPI."""
    layout_dpi = float(p.get("dpi", 100))
    if abs(fig.get_dpi() - layout_dpi) > 1e-6:
        fig.set_dpi(layout_dpi)          # consistent px ↔ physical-size mapping
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []

    # 2) clipping — tightbbox must be within the canvas. NB ``get_tightbbox`` returns
    #    INCHES while ``fig.bbox`` is display PIXELS; comparing them directly (the old bug)
    #    made this check vacuous (inches ≪ pixels → never flagged), so brackets / rotated
    #    labels spilling off a fixed canvas went undetected. Compare in inches.
    #    Skipped under ``tight_save`` (the crop includes everything → clipping is impossible).
    if not p.get("tight_save"):
        tight = fig.get_tightbbox(renderer)            # inches
        w_in, h_in = fig.get_size_inches()
        eps = 0.02                                     # ~1.4 pt slack
        if (tight.x0 < -eps or tight.y0 < -eps
                or tight.x1 > w_in + eps or tight.y1 > h_in + eps):
            problems.append("clipping")

    # 3) text–text overlap. A rotated label's axis-aligned window_extent is the LINE box
    #    (cap height + asc/descent + line spacing), wider than the visible ink, so adjacent
    #    vertical/45° labels falsely "overlap" by that padding. Deflate each box by
    #    ``text_shrink_px`` (per side) to test ink-vs-ink rather than box-vs-box.
    shrink = float(p.get("text_shrink_px", 0.0))

    def _shrunk(b):
        if shrink <= 0 or b.width <= 2 * shrink or b.height <= 2 * shrink:
            return b
        return Bbox.from_extents(b.x0 + shrink, b.y0 + shrink, b.x1 - shrink, b.y1 - shrink)

    texts = _text_artists(fig)
    bbs = [(t, _bbox(t, renderer)) for t in texts]
    bbs = [(t, _shrunk(b)) for t, b in bbs if b is not None]
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

    # 4b) 2-D grid distinguishability (scpilot dotplot): EVERY cell of the n_x × n_y dot
    #     grid must be ≥ min_category_px in BOTH dims, so dots are visibly separated (not
    #     just mathematically non-overlapping). Prevents a degenerate thin strip.
    grid = p.get("dot_grid")
    if grid and data_axes:
        n_x, n_y = grid
        cell_px = p.get("dot_min_cell_px", p["min_category_px"])     # per-gene column (x)
        row_px = p.get("dot_min_row_px", cell_px)                    # per-cell-type row (y)
        bb = data_axes[0].get_window_extent(renderer)
        if (n_x and bb.width / n_x < cell_px) or (n_y and bb.height / n_y < row_px):
            problems.append("elements_cramped")

    # 4c) dot clipping (scpilot dotplot): boundary-row / -column dots are drawn clip_on=True,
    #     so unless the axis limits reach ≥ one dot RADIUS past the outermost dot CENTRE the
    #     edge dots get sliced at the spine — and because clip_on drops the overflow before
    #     rendering, the generic tight-bbox 'clipping' check (item 2) can't see it. save_dotplot
    #     sizes the limits from an ESTIMATE of the panel fraction; this measures the ACTUAL
    #     rendered geometry so a wrong estimate can never ship clipped dots silently.
    if grid and data_axes:
        dpi = fig.get_dpi()
        for ax in data_axes:
            cols = [c for c in ax.collections if len(c.get_offsets()) and len(c.get_sizes())]
            if not cols:
                continue
            xl = sorted(ax.get_xlim()); yl = sorted(ax.get_ylim())
            x0d, y0d = ax.transData.transform((0.0, 0.0))
            x1d, _ = ax.transData.transform((1.0, 0.0))
            _, y1d = ax.transData.transform((0.0, 1.0))
            px_per_x = abs(x1d - x0d) or 1.0
            px_per_y = abs(y1d - y0d) or 1.0
            for c in cols:
                offs = np.asarray(c.get_offsets())
                diam_px = float(np.sqrt(np.asarray(c.get_sizes()).max())) / 72.0 * dpi
                rdx = diam_px / 2.0 / px_per_x        # dot radius in DATA units, per axis
                rdy = diam_px / 2.0 / px_per_y
                xs = offs[:, 0]; ys = offs[:, 1]
                if (xs.min() - rdx < xl[0] - 1e-6 or xs.max() + rdx > xl[1] + 1e-6
                        or ys.min() - rdy < yl[0] - 1e-6 or ys.max() + rdy > yl[1] + 1e-6):
                    problems.append("dot_clipped")
                    break
            break       # the mainplot is the first data axis carrying the dot grid

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

    # 7) legend area cap (scpilot) — flagged legend axes must stay ≤ legend_area_frac of
    #    the figure (e.g. a dotplot's size + colour legend not dominating the panel).
    frac = p.get("legend_area_frac")
    if frac is not None:
        fig_area = fig.bbox.width * fig.bbox.height
        leg_area = 0.0
        for ax in fig.axes:
            if getattr(ax, "_scqc_legend", False):
                bb = ax.get_window_extent(renderer)
                leg_area += max(0.0, bb.width) * max(0.0, bb.height)
        if fig_area > 0 and leg_area / fig_area > frac + 1e-9:
            problems.append("legend_too_big")
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
    # include the exact max bound so a fit that only appears AT the cap (when the cap is
    # not a multiple of step, e.g. 1.8 with step 0.25) is found, not missed → best-effort.
    ws = np.unique(np.append(ws, round(p["max_w_col"] * cols, 3)))
    hs = np.unique(np.append(hs, round(p["max_h_col"] * rows, 3)))
    pairs = [(float(w), float(h)) for w in ws for h in hs]
    lim = p.get("square_limit_col")
    if lim is not None:                       # forbid both dims exceeding lim (per panel)
        pairs = [(w, h) for (w, h) in pairs
                 if not (w / cols > lim + 1e-9 and h / rows > lim + 1e-9)]
    pairs.sort(key=lambda wh: (wh[0] * wh[1], wh[0] + wh[1]))  # smallest area first
    return pairs


def _knob_ladder(p: dict):
    """Escalating mitigations within a size: reduce font, largest first (rotation is
    content-based). Full integer range base→min so a wide band (e.g. 12→6) picks the
    LARGEST font that avoids overlap, not just {base, base-1, min}."""
    base, mn = int(round(p["base_font_pt"])), int(round(p["min_font_pt"]))
    fonts = list(range(max(base, mn), mn - 1, -1)) or [mn]
    return [{"font": float(f)} for f in fonts]


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


def _add_cat_legend(fig, labels, colors, *, font, marker_pt, fig_h_in, fig_w_in=None,
                    force_right=False):
    """Figure-level categorical legend with small dots (constrained_layout reserves the
    space → never clips / never dominates). Few categories sit outside-right (auto ncol
    by height); many categories (a 31/41-sample batch key) go BELOW the plot in many
    columns instead — a right-side column of 40 labels would otherwise crush the UMAP to
    a sliver."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", markersize=marker_pt,
                      markerfacecolor=c, markeredgewidth=0, color=c, label=str(l))
               for l, c in zip(labels, colors)]
    n = len(labels)
    if n > 12 and not force_right:                # bottom strip, packed in columns
        maxlen = max((len(str(l)) for l in labels), default=4)
        col_in = (maxlen + 3) * font / 72.0       # ~chars × font width + marker/pad
        avail = (fig_w_in or 3.5)
        ncol = int(max(2, min(n, avail / max(col_in, 0.4))))
        fig.legend(handles=handles, loc="outside lower center", frameon=False, fontsize=font,
                   ncol=ncol, handletextpad=0.3, columnspacing=0.6, labelspacing=0.25,
                   borderaxespad=0.2)
        return
    # right-side legend: as many ROWS as the (fixed) height allows, then add columns →
    # long / numerous labels widen the figure rather than taller it.
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
        if getattr(ax, "_scqc_no_relabel", False):
            continue        # axis manages its own x-tick rotation (e.g. dotplot gene labels)
        _relabel_xaxis(ax, p["rotate_thresh"])
    fig.canvas.draw()


def fit_and_save(build, cfg: PipelineConfig, base_path: Path, *,
                 n_categories: int | None = None, grid=(1, 1),
                 logger=None, overrides: dict | None = None) -> FitResult:
    """Search smallest column-size satisfying all constraints, then save (fixed canvas).

    ``build(size_inches, font_pt) -> Figure`` must create and return the figure
    (so plots that own their figure, e.g. multi_panel violin / highly_variable_genes,
    are supported). The engine only sizes / measures / saves it.

    ``overrides`` patches the plotting config FOR THIS CALL ONLY (e.g. a dotplot raising
    the size cap to 2×2 col and adding a legend-area cap) without touching the profile.
    """
    p = plotting_cfg(cfg)
    if overrides:
        p = {**p, **overrides}
    col = p["column_width_in"]
    base_path = Path(base_path)

    def _final(size, knob):
        """Full (non-draft) render at the chosen size: re-VALIDATE on full data (the search
        ran on a subsample, and full data can shift legend tick labels / colour range /
        dot scaling), then save. Returns (paths, residual_problems)."""
        with _font_context(knob["font"]):
            ff = build(size, knob["font"], draft=False)
            _finalize_layout(ff, p, knob["font"], size[1])
            final_problems = check_constraints(ff, p, n_categories)
        paths = _save(ff, base_path, p)
        plt.close(ff)
        return paths, final_problems

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
                paths, final_problems = _final((w * col, h * col), knob)
                warns = []
                if final_problems:
                    warns = [f"final-render-violations({base_path.name}) at {w}x{h} col: "
                             f"{sorted(set(final_problems))} — passed on the draft subsample but "
                             "not on full data"]
                    warnings.warn(warns[0])
                    if logger:
                        logger.warning(warns[0])
                elif logger:
                    logger.info("fit %s → %.1fx%.1f col, font %.0f",
                                base_path.name, w, h, knob["font"])
                return FitResult(paths, (w, h), knob["font"],
                                 {**knob, "final_validated": not final_problems}, warns)

    # could not satisfy within max size → best-effort at max, flag it
    max_size = (p["max_w_col"] * col * cols, p["max_h_col"] * col * rows)
    knob = {"font": p["min_font_pt"]}
    with _font_context(p["min_font_pt"]):
        fig = build(max_size, p["min_font_pt"], draft=True)
        _finalize_layout(fig, p, p["min_font_pt"], max_size[1])
        remaining = check_constraints(fig, p, n_categories)
        plt.close(fig)
    paths, final_problems = _final(max_size, knob)
    remaining = sorted(set(remaining) | set(final_problems))   # draft + full-data residuals
    warn = [f"fit-at-max-failed: {base_path.name}: {remaining}"]
    warnings.warn(warn[0])
    if logger:
        logger.warning(warn[0])
    return FitResult(paths, (p["max_w_col"], p["max_h_col"]), p["min_font_pt"],
                     {"best_effort": True}, warn)


def _save(fig, base_path: Path, p: dict) -> list:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    # Default = fixed canvas (no bbox_inches='tight'), so the saved size equals the chosen
    # size. ``tight_save`` opts a plot (e.g. the dotplot, whose scanpy brackets / rotated
    # labels are drawn OUTSIDE the axes) into a tight crop so those labels are never clipped
    # — at the cost of the exact-size guarantee (saved size = content extent ≥ figsize).
    bbox = "tight" if p.get("tight_save") else None
    out = []
    for fmt in p["formats"]:
        path = base_path.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=p["dpi_save"], facecolor=p["facecolor"], bbox_inches=bbox)
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


def save_violin(adata, cfg, base_path, keys, groupby=None, logger=None, strip_max=2500):
    """Harness lays out a constrained 1×N axes grid; scanpy draws each panel via ax=
    (multi_panel FacetGrid ignores constrained_layout and clips labels, so the
    harness provides the grid — data rendering stays 100% scanpy).

    The stripplot is thinned: at full scale (10⁵–10⁶ cells) an exhaustive strip is a
    solid black mass that hides the violin entirely, so we subsample to ``strip_max``
    points (KDE shape is unchanged) and draw them small + semi-transparent so the
    violin body reads through."""
    keys = list(keys)
    n_cat = adata.obs[groupby].nunique() if groupby else len(keys)

    def build(size, font, draft=False):
        ad = _draft_view(adata, draft) if draft else _draft_view(adata, True, n=strip_max)
        fig = _new_fig(size)
        axes = fig.subplots(1, len(keys), squeeze=False)[0]
        for ax, m in zip(axes, keys):
            sc.pl.violin(ad, keys=m, groupby=groupby, ax=ax, show=False,
                         stripplot=not draft, jitter=0.4, size=1.0)
            ax.set_xlabel("")
            if not draft:                       # de-densify the strip so the violin shows
                for coll in ax.collections:
                    coll.set_alpha(0.4)
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


# Fixed UMAP point size (pt²), HARDCODED so every UMAP — every coloring, every integration
# method — draws dots at exactly one size (the size in artifacts/umap_harmony_sample_id.svg,
# scanpy size=2). Not derived from cell count / figure size, so the plots are comparable.
UMAP_DOT_SIZE_PT = 2.0


def save_umap(adata, cfg, base_path, color, size_pt=UMAP_DOT_SIZE_PT, logger=None, basis="X_umap"):
    """Harness lays out the axes; scanpy renders the points (sc.pl.embedding ax=) with its
    legend/colorbar suppressed. The harness owns a small-dot legend (categorical) or a
    colorbar (continuous) outside — controlled, never clipped, never dominating.

    Point size is HARDCODED to ``UMAP_DOT_SIZE_PT`` (the size in umap_harmony_sample_id.svg)
    so dots are identical across all UMAPs. ``basis`` selects the obsm embedding (default
    X_umap; e.g. X_umap_harmony / X_umap_scvi to render per-integration UMAPs)."""
    size_pt = UMAP_DOT_SIZE_PT     # hardcoded — ignore any caller override
    cmap = _default_cmap(cfg)
    mk = plotting_cfg(cfg)["legend_marker_pt"]
    cat = _is_cat(adata, color)
    n = adata.obs[color].nunique() if cat else None
    emb = basis[2:] if basis.startswith("X_") else basis   # sc.pl.embedding strips the X_ prefix itself

    def build(size, font, draft=False):
        ad = _draft_view(adata, draft)
        fig = _new_fig(size)
        ax = fig.subplots()
        sc.pl.embedding(ad, basis=emb, color=color, size=size_pt, ax=ax, show=False,
                        color_map=cmap, legend_loc=None, colorbar_loc=None)
        if cat:
            cats, colors = _legend_cats_colors(ad, color, cfg)
            _add_cat_legend(fig, cats, colors, font=font, marker_pt=mk,
                            fig_h_in=size[1], fig_w_in=size[0], force_right=True)
        else:
            _add_colorbar(fig, [ax], ad.obs[color], cmap, font=font, label=color)
        return fig

    # CATEGORICAL UMAP policy (user spec): HEIGHT fixed at 1 col, WIDTH 1→2 col — only as
    # wide as the right-side legend's label names need (condition / sample IDs). Fixed
    # height → the square panel + FIXED dot size are identical across integration methods
    # (X_umap / X_umap_harmony / X_umap_scVI), so they are directly comparable; the legend
    # widens the figure (never taller) and the width is hard-capped at 2 col.
    if cat:
        overrides = {"start_col": 1.0, "step_col": 0.25, "max_w_col": 2.0,
                     "max_h_col": 1.0, "square_limit_col": None}
        return fit_and_save(build, cfg, base_path, n_categories=None,
                            overrides=overrides, logger=logger)
    return fit_and_save(build, cfg, base_path, logger=logger)   # continuous → auto-fit


def save_dotplot(adata, cfg, base_path, groups, groupby, *, categories_order=None,
                 logger=None, standard_scale=None, layer=None, swap_axes=False,
                 max_w_col=1.8, max_h_col=1.0, legend_area_frac=0.05, dot_min_cell_px=6.0,
                 dot_min_row_px=9.0, font_min=6.0, font_max=12.0, dot_cell_frac=0.9,
                 var_group_rotation=60.0, smallest_dot_frac=0.0,
                 color_vmax_pct=95.0, color_vmax=None, staircase=True):
    """Annotation dotplot routed through the auto-fit engine (``sc.pl.dotplot`` with the
    marker panels passed AS A DICT, so scanpy draws the cell-type brackets/labels).

    scpilot sizing policy (per user spec): the SAVED canvas is the smallest size in the
    range 0.5×0.5 → ``max_w_col``×``max_h_col`` columns (default 1.8×1.0 — a dotplot is
    wide: genes on x ≫ cell types on y) at which NO text overlaps, NO dots overlap, and
    the size+colour legend stays ≤ ``legend_area_frac`` (5%) of the figure. Font is chosen
    from ``font_min``..``font_max`` (largest that avoids overlap); the largest dot diameter
    is ``dot_cell_frac`` of the grid cell (the no-overlap line — no fixed area ceiling), so
    dots are as big as possible without touching neighbours. ``figsize`` is the whole figure
    for a dotplot (panel + brackets + legend packed in); saved with a tight crop so the
    out-of-axes brackets/labels are never clipped. ``categories_order``
    orders the y-axis cell types (staircase under the marker-group columns).

    Marker contrast (per user spec): the size scale runs from ``smallest_dot_frac`` ×
    the largest dot (default 0.0 → low-fraction off-target dots shrink to nothing, so the
    fraction axis is fully spent on the on-target diagonal); the colour ceiling is the
    ``color_vmax_pct`` percentile (default 95th) of the per-group mean expression — NOT the
    raw max — so a handful of extreme genes don't wash the scale out and the typical
    on-target marker saturates dark instead of sitting at ~30% of the bar. Pass an explicit
    ``color_vmax`` to override the percentile."""
    n_genes = sum(len(v) for v in groups.values())
    cats = list(categories_order) if categories_order is not None \
        else list(adata.obs[groupby].astype("category").cat.categories)
    n_groups = len(cats)

    # Per-group mean-expression matrix M (rows = present cell types, cols = panel genes),
    # computed ONCE — it is size-independent and mirrors scanpy's dot_color_df (mean of X over
    # ALL cells in a group), so it drives both the colour ceiling and the staircase order off
    # the same numbers the dots are drawn from.
    flat_genes = [g for gs in groups.values() for g in gs]
    gidx = {g: i for i, g in enumerate(flat_genes)}
    sub = adata[:, flat_genes]
    grp = adata.obs[groupby].astype("category")
    present_cats, rows = [], []
    for ct in cats:
        mask = (grp == ct).to_numpy()
        if not mask.any():
            continue
        present_cats.append(ct)
        rows.append(np.asarray(sub.X[mask].mean(axis=0)).ravel())
    M = np.vstack(rows) if rows else np.zeros((0, len(flat_genes)))
    rowof = {ct: i for i, ct in enumerate(present_cats)}

    # Colour ceiling: the ``color_vmax_pct`` percentile of M (NOT the raw max) so a few extreme
    # genes don't wash the scale out and the typical on-target marker saturates dark.
    vmax = float(color_vmax) if color_vmax is not None \
        else (float(np.percentile(M, color_vmax_pct)) if M.size else None)

    # STAIRCASE: the y-axis rows follow the marker-PANEL (x) order, so a caller that hands its
    # panels in a fixed biological-compartment order (epithelial → stromal/vascular → immune →
    # artificial) gets that exact top-to-bottom row order, with each cell type sitting under its
    # own marker block on the diagonal; cell types with no panel of their own trail below.
    # VERIFICATION (the staircase invariant): a paneled cell type whose strongest marker BLOCK is
    # NOT its own panel is flagged — its cells express another type's markers more strongly
    # (marker non-specificity or a mislabel), surfaced as a warning rather than silently hidden.
    staircase_msg = None
    if staircase and M.size and groups:
        block_cols = {pn: [gidx[g] for g in gs] for pn, gs in groups.items()}

        def _peak_block(ct):
            r = M[rowof[ct]]
            return max(((pn, float(r[cols].mean())) for pn, cols in block_cols.items()),
                       key=lambda kv: kv[1])

        paneled = [ct for ct in groups if ct in rowof]          # panel (compartment) order
        others = [ct for ct in cats if ct not in groups]        # non-paneled, keep incoming order
        cats = paneled + others
        n_groups = len(cats)
        off = [f"{ct}↛{_peak_block(ct)[0]}" for ct in paneled if _peak_block(ct)[0] != ct]
        if off:
            staircase_msg = (f"staircase: {len(off)}/{len(paneled)} cell types peak off their own "
                             f"marker block ({', '.join(off)}) — marker non-specificity or a "
                             "possible mislabel")
            if logger:
                logger.warning(staircase_msg)

    def build(size, font, draft=False):
        w_in, h_in = size
        view = _draft_view(adata, draft)
        # The draft subsample can drop a rare category entirely; categories_order must only list
        # rows actually present in THIS view, or scanpy raises KeyError. The final render uses the
        # full data (draft=False) so it keeps every category.
        grp_v = view.obs[groupby].astype("category")
        cats_v = [c for c in cats if (grp_v == c).to_numpy().any()]
        n_groups_v = len(cats_v)
        plt.close("all")
        dp = sc.pl.dotplot(view, groups, groupby=groupby, categories_order=cats_v,
                           figsize=(w_in, h_in), var_group_rotation=var_group_rotation,
                           layer=layer, swap_axes=swap_axes, standard_scale=standard_scale,
                           vmax=vmax, dendrogram=False, return_fig=True, show=False)
        # Dot MAX size is bounded ONLY by the no-overlap line: the largest dot diameter is
        # ``dot_cell_frac`` of the grid cell, so adjacent dots never touch — no arbitrary
        # fixed ceiling. The cell is estimated from the panel's share of the figure
        # (legend/brackets take the rest); cells are kept ≥ dot_min_*_px by elements_cramped,
        # so the dots can't become invisibly small either.
        x_cell_in = w_in * 0.6 / max(n_genes, 1)
        y_cell_in = h_in * 0.75 / max(n_groups_v, 1)
        cell_in = min(x_cell_in, y_cell_in)
        diam_pt = cell_in * 72.0 * dot_cell_frac        # ≤ cell → no neighbour overlap
        largest = float(diam_pt ** 2)                   # scanpy dot size = area (pt²)
        # Edge dots are drawn clip_on=True, so the axis must reach ≥ half a dot beyond the
        # outermost dot CENTRE or the boundary rows/cols get sliced at the spine. scanpy's
        # x/y_padding is exactly that centre→edge gap (in cell units) → set it to the dot's
        # half-width in each axis's cell + a small margin. Per-axis (cells are non-square:
        # a dot fills the binding axis but is a smaller fraction of the looser one).
        diam_in = diam_pt / 72.0
        x_padding = diam_in / 2 / x_cell_in + 0.1
        y_padding = diam_in / 2 / y_cell_in + 0.1
        dp.style(largest_dot=largest, smallest_dot=largest * smallest_dot_frac, dot_edge_lw=0.25,
                 dot_edge_color="black", grid=False, x_padding=x_padding, y_padding=y_padding)
        dp.legend(width=min(1.1, max(0.55, w_in * 0.15)))   # compact legend (helps the 5% cap)
        dp.make_figure()
        fig = dp.fig
        fig.canvas.draw()
        # scanpy reserves a fixed legend column and leaves a wide gap on the far right →
        # pack the legend to the right edge and grow the main panel into the freed space,
        # so the genes get the widest possible layout inside the fixed canvas.
        gg = dp.ax_dict.get("gene_group_ax")
        mp = dp.ax_dict.get("mainplot_ax")
        legs = [dp.ax_dict[k] for k in ("size_legend_ax", "color_legend_ax")
                if dp.ax_dict.get(k) is not None]
        if mp is not None and legs:
            pad, gap = 0.012, 0.05
            leg_right = max(ax.get_position().x1 for ax in legs)
            shift = (1.0 - pad) - leg_right
            if shift > 0:
                for ax in legs:
                    q = ax.get_position()
                    ax.set_position([q.x0 + shift, q.y0, q.width, q.height])
            leg_left = min(ax.get_position().x0 for ax in legs)
            for ax in (mp, gg):
                if ax is not None:
                    q = ax.get_position()
                    new_w = max(0.1, (leg_left - gap) - q.x0)
                    ax.set_position([q.x0, q.y0, new_w, q.height])
            fig.canvas.draw()
        # scanpy hardcodes tick-label / legend-title sizes (≈5pt) on the dotplot's own
        # axes, ignoring rcParams → force EVERY text on EVERY dotplot axis to the knob font
        # so nothing dips below the font floor.
        for ax in dp.ax_dict.values():
            ax.tick_params(labelsize=font)
            for t in (list(ax.get_xticklabels()) + list(ax.get_yticklabels())
                      + list(ax.texts) + ([ax.title] if ax.title else [])):
                t.set_fontsize(font)
        # flag the size/colour legend so the engine measures it (≤5%) and excludes it from
        # the data-axes (so the main panel stays dominant); bracket row is aux too.
        for k in ("size_legend_ax", "color_legend_ax"):
            ax = dp.ax_dict.get(k)
            if ax is not None:
                ax._scqc_legend = True
                ax._scqc_aux = True
        gg = dp.ax_dict.get("gene_group_ax")
        if gg is not None:
            gg._scqc_aux = True
        # gene x-labels go VERTICAL (90°) — short gene names packed horizontally collide
        # even on a wide panel; vertical labels are the conventional dotplot style and pack
        # tightest. Flag the axis so the harness's length-based relabel doesn't reset it.
        mp = dp.ax_dict.get("mainplot_ax")
        if mp is not None:
            for lbl in mp.get_xticklabels():
                lbl.set_rotation(90); lbl.set_ha("center"); lbl.set_va("top")
            mp._scqc_no_relabel = True
        return fig

    overrides = {"max_w_col": max_w_col, "max_h_col": max_h_col, "start_col": 0.5,
                 "square_limit_col": None, "legend_area_frac": legend_area_frac,
                 "dot_grid": (n_genes, n_groups), "dot_min_cell_px": dot_min_cell_px,
                 "dot_min_row_px": dot_min_row_px,
                 "min_font_pt": font_min, "base_font_pt": font_max,
                 "text_shrink_px": 2.0,    # ink-vs-ink overlap for rotated tick/bracket labels
                 "tight_save": True,       # scanpy draws brackets/labels OUTSIDE the axes → tight crop (no clip)
                 "formats": ["svg", "png"]}  # SVG = vector deliverable; PNG = quick preview
    fit = fit_and_save(build, cfg, Path(base_path), n_categories=None,
                       logger=logger, overrides=overrides)
    fit.knobs["row_order"] = list(cats)        # resolved y-axis order (top→bottom), for callers/tests
    if staircase_msg:
        fit.warnings.append(staircase_msg)
    return fit
