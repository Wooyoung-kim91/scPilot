"""Pipeline-stage detection — scpilot plan B2.

``detect_state`` reads an h5ad (backed='r', no X load) and classifies how far it
has been processed, so the orchestrator knows the re-entry point. Stages are
cumulative flags (a clustered object is also normalized + hvg + pca). Returns a
``ToolResult`` summary with the detected stage + evidence + suggested next tool.
"""

from __future__ import annotations

from pathlib import Path

from scpilot import schemas as S
from scpilot.core.io import _guess_x_state
from scpilot.tools import register

# obs columns that signal annotation has happened (plan annotation schema).
_ANNOTATION_COLS = ("major_cell_type", "fine_cell_type", "facs_style_label", "malignancy")
# ordered stages from least to most processed; the highest satisfied one wins.
_STAGE_ORDER = ["raw", "normalized", "hvg", "pca", "neighbors", "clustered", "umap", "annotated"]
_NEXT_TOOL = {
    "raw": "preprocess", "normalized": "preprocess", "hvg": "cluster", "pca": "cluster",
    "neighbors": "cluster", "clustered": "markers", "umap": "annotate", "annotated": "report",
}


def detect_state(path: str) -> S.ToolResult:
    """Classify the most-advanced pipeline stage present in an h5ad."""
    p = Path(path)
    if not p.exists():
        return S.error("detect_state", "missing_input", f"file not found: {p}", recoverable=False)
    try:
        import anndata as ad

        a = ad.read_h5ad(str(p), backed="r")
        layers = set(a.layers.keys())
        obsm = set(a.obsm.keys())
        obsp = set(a.obsp.keys())
        obs_cols = set(a.obs.columns)
        uns = set(a.uns.keys())
        x_state = _guess_x_state(a)

        flags = {
            "has_counts": "counts" in layers,
            "normalized": x_state in ("normalized", "log1p") or "scale.data" in layers,
            "hvg": "highly_variable" in a.var.columns,
            "pca": "X_pca" in obsm,
            "neighbors": ("neighbors" in uns) or ("distances" in obsp) or ("connectivities" in obsp),
            "clustered": any(c in obs_cols for c in ("leiden", "louvain")),
            "umap": "X_umap" in obsm,
            "annotated": any(c in obs_cols for c in _ANNOTATION_COLS),
        }
        # most-advanced satisfied stage
        satisfied = [s for s in _STAGE_ORDER if (s == "raw" and flags["has_counts"]) or flags.get(s)]
        stage = satisfied[-1] if satisfied else "raw"

        present_annos = sorted(c for c in _ANNOTATION_COLS if c in obs_cols)
        summary = {
            "path": str(p.resolve()),
            "n_obs": int(a.n_obs), "n_vars": int(a.n_vars),
            "stage": stage,
            "flags": flags,
            "x_state_guess": x_state,
            "annotation_columns_present": present_annos,
            "reentry_point": _NEXT_TOOL.get(stage, "preprocess"),
            "scpilot_provenance_present": "scpilot" in uns,
        }
        warnings = []
        if not flags["has_counts"]:
            warnings.append("no 'counts' layer — cannot re-run count-based steps (QC/HVG seurat_v3/CNV)")
        try:
            a.file.close()
        except Exception:  # noqa: BLE001
            pass
        return S.success("detect_state", summary=summary, warnings=warnings,
                         determinism_grade="A", suggested_next_tools=[summary["reentry_point"]])
    except Exception as exc:  # noqa: BLE001
        return S.error("detect_state", "internal", f"{type(exc).__name__}: {exc}")


@register("detect_state", mutating=False,
          description="Detect how far an h5ad has been processed (raw/hvg/clustered/annotated) → re-entry point.")
def _detect_state_tool(session, **params) -> S.ToolResult:
    path = params.get("path") or session.manifest.input.get("path")
    if not path:
        return S.error("detect_state", "missing_input", "no input path", recoverable=False)
    return detect_state(path)
