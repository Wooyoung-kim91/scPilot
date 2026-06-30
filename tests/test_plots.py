"""Unit tests for B5 plots (vendored auto-fit harness + column-size policy)."""

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

from scpilot import tools
from scpilot.session import Session


def _processed_session(tmp_path):
    """A small clustered AnnData so umap/hvg/pca/qc plots all have inputs."""
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (200, 80)).astype("float32")
    X[:100, :20] += rng.poisson(4.0, (100, 20)).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(80)]
    a.obs["sample_id"] = rng.choice(["s1", "s2"], 200)
    a.layers["counts"] = a.X.copy()
    a.obs["n_genes_by_counts"] = (a.layers["counts"] > 0).sum(1).A1
    a.obs["total_counts"] = a.layers["counts"].sum(1).A1
    a.obs["pct_counts_mt"] = rng.uniform(0, 10, 200)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=40)
    sc.pp.pca(a, n_comps=15)
    sc.pp.neighbors(a, n_neighbors=10)
    sc.tl.leiden(a, flavor="igraph", n_iterations=2, random_state=0)
    sc.tl.umap(a, random_state=0)
    p = tmp_path / "in.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    return s


def _assert_size_policy(r):
    """Saved size must be within {min 0.5×0.5, max 1.5 each, not both >1.0} col."""
    w, h = r.summary["size_col"]
    assert 0.5 <= w <= 1.5 and 0.5 <= h <= 1.5
    assert not (w > 1.0 and h > 1.0), f"both dims >1 col: {(w, h)}"


def test_plot_umap(tmp_path):
    s = _processed_session(tmp_path)
    r = tools.run("plots", s, kind="umap", color="leiden")
    assert r.status == "success"
    # every figure writes a vector SVG (deliverable) + a PNG (preview)
    kinds = {a.kind for a in r.artifacts}
    assert {"svg", "png"} <= kinds
    from pathlib import Path
    assert all(Path(a.path).exists() and Path(a.path).stat().st_size > 0 for a in r.artifacts)
    # categorical UMAP: height fixed at 1 col; width 1→2 col (only as wide as the legend needs)
    w, h = r.summary["size_col"]
    assert h == 1.0 and 1.0 <= w <= 2.0
    r.to_dict()


def test_plot_qc_violin_and_hvg_and_pca(tmp_path):
    s = _processed_session(tmp_path)
    for kind in ("qc_violin", "hvg", "pca_variance"):
        r = tools.run("plots", s, kind=kind)
        assert r.status == "success", f"{kind}: {r.error}"
        assert r.artifacts
        _assert_size_policy(r)


def test_plot_umap_requires_umap(tmp_path):
    # raw session without umap
    a = ad.AnnData(sparse.csr_matrix(np.random.default_rng(0).poisson(1.0, (50, 20)).astype("float32")))
    a.layers["counts"] = a.X.copy()
    p = tmp_path / "raw.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "s", input_path=str(p)); s.load_input()
    r = tools.run("plots", s, kind="umap")
    assert r.status == "error" and r.error_code == "invalid_state"


def test_size_grid_orientation_filter():
    # the square_limit_col filter forbids both dims exceeding the limit
    from scpilot.vendor.plotting import _size_grid
    p = {"start_col": 0.5, "step_col": 0.25, "max_w_col": 1.5, "max_h_col": 1.5,
         "square_limit_col": 1.0}
    pairs = _size_grid(p)
    assert (1.5, 1.0) in pairs and (1.0, 1.5) in pairs and (1.0, 1.0) in pairs
    assert all(not (w > 1.0 and h > 1.0) for w, h in pairs)
    assert (1.5, 1.5) not in pairs and (1.25, 1.25) not in pairs


def _annotated_session(tmp_path):
    """Small clustered AnnData carrying broad-marker genes + a major_cell_type label,
    so the annotation dotplot and a many-category UMAP both have inputs."""
    from scpilot.core.annotate import BROAD_MARKERS

    rng = np.random.default_rng(0)
    genes = [g for gs in BROAD_MARKERS.values() for g in gs]
    n = len(genes)
    X = rng.poisson(1.0, (300, n)).astype("float32")
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = genes
    cts = list(BROAD_MARKERS) + ["Unknown"]
    a.obs["major_cell_type"] = rng.choice(cts, 300)
    a.obs["sample_id"] = rng.choice([f"S{i:02d}" for i in range(20)], 300)  # >12 cats
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=10); sc.pp.neighbors(a, n_neighbors=10); sc.tl.umap(a, random_state=0)
    p = tmp_path / "anno.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "asess", input_path=str(p)); s.load_input()
    return s


def test_plot_dotplot_staircase(tmp_path):
    from scpilot.core.annotate import BROAD_MARKERS

    s = _annotated_session(tmp_path)
    r = tools.run("plots", s, kind="dotplot", groupby="major_cell_type")
    assert r.status == "success", r.message
    from pathlib import Path
    png = Path(r.artifacts[0].path)
    assert png.exists() and png.stat().st_size > 0
    # bounded auto-fit (0.5–2.0 col square): a real fit, not the best-effort fallback
    assert r.summary["fit_at_max_failed"] is False
    w, h = r.summary["size_col"]
    assert 0.5 <= w <= 2.0 and 0.5 <= h <= 2.0
    # y-axis order is a staircase: cell types follow the marker-panel column order,
    # with non-panel labels (Unknown) trailing at the bottom.
    cats = list(s.adata.obs["major_cell_type"].astype("category").cat.categories)
    expected = [ct for ct in BROAD_MARKERS if ct in cats] + \
               [ct for ct in cats if ct not in BROAD_MARKERS]
    assert s.adata.obs["major_cell_type"].cat.categories.tolist() == cats  # unchanged in obs
    assert expected[-1] == "Unknown"


def test_dotplot_sizing_policy(tmp_path):
    """The dotplot auto-fits within the dotplot bounds (≤1.8×1.0 col), at a font in
    [6,12], with the saved canvas equal to the chosen size (no best-effort fallback)."""
    from PIL import Image
    from scpilot.core.annotate import BROAD_MARKERS
    from scpilot.vendor import plotting as P
    from scpilot.vendor.config import PipelineConfig

    s = _annotated_session(tmp_path)
    a = s.adata
    groups = {ct: [g for g in gs if g in a.var_names] for ct, gs in BROAD_MARKERS.items()}
    groups = {ct: gs for ct, gs in groups.items() if gs}
    cats = [ct for ct in groups] + [ct for ct in a.obs["major_cell_type"].cat.categories
                                    if ct not in groups]
    fit = P.save_dotplot(a, PipelineConfig(), tmp_path / "dp", groups, "major_cell_type",
                         categories_order=cats)
    w, h = fit.size_col
    assert 0.5 <= w <= 1.8 and 0.5 <= h <= 1.0
    assert 6.0 <= fit.font_pt <= 12.0
    assert not fit.knobs.get("best_effort"), fit.warnings
    # both a vector SVG (deliverable) and a PNG (preview) are written
    assert any(str(p).endswith(".svg") for p in fit.path)
    png = next(p for p in fit.path if str(p).endswith(".png"))
    # tight crop (brackets/labels live outside the axes): saved size ≥ figsize, and not
    # wildly larger (the label margins are small relative to the panel).
    col = P.plotting_cfg(PipelineConfig())["column_width_in"]
    im = Image.open(png)
    assert im.size[0] / 300 >= w * col - 0.1 and im.size[1] / 300 >= h * col - 0.1
    assert im.size[0] / 300 <= w * col + 1.5 and im.size[1] / 300 <= h * col + 1.5


def test_dot_clipped_invariant():
    """check_constraints flags 'dot_clipped' IFF a boundary dot overflows the axis spine.

    Edge dots are drawn clip_on=True, so a sliced dot vanishes before get_tightbbox sees it
    → the generic tight-bbox 'clipping' check can't catch it (and it's skipped under
    tight_save anyway). This invariant measures dot centre ± radius against the axis limits,
    so an under-padded dotplot can't ship clipped boundary rows/cols silently."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scpilot.vendor import plotting as P
    from scpilot.vendor.config import PipelineConfig

    p = {**P.plotting_cfg(PipelineConfig()), "dot_grid": (3, 3), "tight_save": True}

    def make(xlim):
        fig, ax = plt.subplots(figsize=(3, 3), dpi=p["dpi"])
        xs, ys = np.meshgrid([0.5, 1.5, 2.5], [0.5, 1.5, 2.5])
        ax.scatter(xs.ravel(), ys.ravel(), s=400, clip_on=True)  # area pt² → ~20pt dots
        ax.set_xlim(*xlim); ax.set_ylim(0, 3)
        return fig

    tight = make((0.5, 2.5))      # limits hug the edge dot CENTRES → boundary dots overflow
    assert "dot_clipped" in P.check_constraints(tight, p, None)
    plt.close(tight)
    roomy = make((-1.0, 4.0))     # generous limits fully contain every dot
    assert "dot_clipped" not in P.check_constraints(roomy, p, None)
    plt.close(roomy)
    # non-dotplot figures (no dot_grid) are never subject to this check
    plain = make((0.5, 2.5))
    assert "dot_clipped" not in P.check_constraints(plain, {k: v for k, v in p.items()
                                                            if k != "dot_grid"}, None)
    plt.close(plain)


def test_text_overlap_invariant():
    """text_overlap_count / check_constraints flag overlapping LABELS on ink-vs-ink boxes, so a
    dotplot (rotated gene labels + cell-type brackets) can't ship with colliding text silently.
    The harness gates the layout fit on this AND re-checks it on the final render."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scpilot.vendor import plotting as P
    from scpilot.vendor.config import PipelineConfig

    p = P.plotting_cfg(PipelineConfig())

    # two labels stacked at the SAME spot → ink overlaps (1 pair)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=p["dpi"])
    ax.set_xticks([]); ax.set_yticks([])          # only our two texts exist
    ax.text(0.5, 0.5, "OverlappingLabel", ha="center", va="center")
    ax.text(0.5, 0.5, "OverlappingLabel", ha="center", va="center")
    fig.canvas.draw()
    assert P.text_overlap_count(fig, p) >= 1
    assert "text_overlap" in P.check_constraints(fig, p, None)
    plt.close(fig)

    # two labels far apart → no overlap
    fig2, ax2 = plt.subplots(figsize=(6, 3), dpi=p["dpi"])
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.text(0.02, 0.5, "left", ha="left", va="center")
    ax2.text(0.98, 0.5, "right", ha="right", va="center")
    fig2.canvas.draw()
    assert P.text_overlap_count(fig2, p) == 0
    assert "text_overlap" not in P.check_constraints(fig2, p, None)
    plt.close(fig2)


def test_dotplot_staircase_invariant(tmp_path):
    """Staircase: y-axis rows follow the marker-PANEL declaration order (so a caller's fixed
    compartment order is honoured), and the verification flags a cell type whose strongest
    marker block isn't its own panel (non-specificity / mislabel) via a 'staircase:' warning."""
    from scpilot.vendor import plotting as P
    from scpilot.vendor.config import PipelineConfig

    genes = ["gA", "gB", "gC"]
    panels = {"C": ["gC"], "A": ["gA"], "B": ["gB"]}          # deliberately NOT alphabetical

    def adata_where(expr):           # expr[ct] = the gene ct's cells express strongly
        rng = np.random.default_rng(0)
        labels = np.repeat(["A", "B", "C"], 40)
        X = rng.poisson(0.2, (120, 3)).astype("float32")
        for i, ct in enumerate(labels):
            X[i, genes.index(expr[ct])] += 8.0
        a = ad.AnnData(sparse.csr_matrix(X))
        a.var_names = genes
        a.obs["celltype"] = labels
        a.obs["celltype"] = a.obs["celltype"].astype("category")
        sc.pp.log1p(a)
        return a

    clean = adata_where({"A": "gA", "B": "gB", "C": "gC"})    # each peaks on its own panel
    fit = P.save_dotplot(clean, PipelineConfig(), tmp_path / "clean", panels, "celltype")
    # rows follow PANEL order (C, A, B) — not alphabetical, not peak-driven
    assert fit.knobs["row_order"] == ["C", "A", "B"], fit.knobs["row_order"]
    assert not any("staircase:" in w for w in fit.warnings), fit.warnings

    dirty = adata_where({"A": "gA", "B": "gB", "C": "gA"})    # C expresses A's marker
    fit2 = P.save_dotplot(dirty, PipelineConfig(), tmp_path / "dirty", panels, "celltype")
    assert fit2.knobs["row_order"] == ["C", "A", "B"], fit2.knobs["row_order"]   # order unchanged
    assert any("staircase:" in w and "C↛A" in w for w in fit2.warnings), fit2.warnings


def test_plot_umap_many_categories(tmp_path):
    s = _annotated_session(tmp_path)
    r = tools.run("plots", s, kind="umap", color="sample_id")
    assert r.status == "success", r.message
    from pathlib import Path
    assert Path(r.artifacts[0].path).exists()
    # many-category embeddings get the generous fixed canvas, not the column cap
    assert r.summary["fit_at_max_failed"] is False


def test_registry_has_plots():
    assert "plots" in {t["name"] for t in tools.list_tools()}
