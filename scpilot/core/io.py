"""I/O + read-only inspection — scpilot plan B1 / A6.

scpilot enters at the merged h5ad (load_10x/merge are scqc's job). This module
provides ``inspect_h5ad`` — the read-only tool used for the A6 MCP spike — plus
load/save helpers. ``inspect_h5ad`` reads the file ``backed='r'`` so it never
materializes the (multi-GB) X matrix; it returns a small ``ToolResult`` summary.
"""

from __future__ import annotations

from pathlib import Path

from scpilot import schemas as S

# obs columns with at most this many distinct values get a value_counts breakdown.
_MAX_CATEGORIES = 20
_MAX_VALUE_ROWS = 15  # cap per-column category listing


def _guess_x_state(adata) -> str:
    """Best-effort guess of what .X currently means (counts vs normalized)."""
    layers = set(getattr(adata, "layers", {}).keys())
    # backed read: peek a tiny slice to test integrality without loading all of X
    try:
        import numpy as np
        sub = adata[:50].to_memory().X if hasattr(adata, "isbacked") and adata.isbacked else adata.X[:50]
        arr = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
        if arr.size and np.allclose(arr, np.round(arr)) and arr.min() >= 0:
            return "raw_counts"
        return "normalized"
    except Exception:  # noqa: BLE001
        return "unknown" if "counts" not in layers else "raw_counts"


def inspect_h5ad(path: str, *, max_categories: int = _MAX_CATEGORIES) -> S.ToolResult:
    """Read-only summary of an .h5ad (shape, layers, obs schema, embeddings, uns).

    Loads ``backed='r'`` — does not materialize X. Returns a ``ToolResult`` whose
    ``summary`` is small enough for an LLM to read directly.
    """
    p = Path(path)
    if not p.exists():
        return S.error("inspect_h5ad", "missing_input", f"file not found: {p}", recoverable=False)
    try:
        import anndata as ad
        import pandas as pd

        adata = ad.read_h5ad(str(p), backed="r")
        n_obs, n_vars = int(adata.n_obs), int(adata.n_vars)
        obs = adata.obs

        # per-column obs schema + low-cardinality value_counts
        obs_schema = []
        categoricals = {}
        for col in obs.columns:
            ser = obs[col]
            nun = int(ser.nunique(dropna=True))
            obs_schema.append({"column": str(col), "dtype": str(ser.dtype), "n_unique": nun})
            if nun <= max_categories:
                vc = ser.value_counts(dropna=False).head(_MAX_VALUE_ROWS)
                categoricals[str(col)] = {str(k): int(v) for k, v in vc.items()}

        # genomic-coordinate readiness (for CNV / Tier 2 gate, plan B12-pre)
        var_cols = [str(c) for c in adata.var.columns]
        has_coords = all(c in var_cols for c in ("chromosome", "start", "end"))

        summary = {
            "path": str(p.resolve()),
            "n_obs": n_obs,
            "n_vars": n_vars,
            "layers": sorted(adata.layers.keys()),
            "obsm": sorted(adata.obsm.keys()),
            "varm": sorted(adata.varm.keys()),
            "obsp": sorted(adata.obsp.keys()),
            "uns_keys": sorted(adata.uns.keys()),
            "obs_columns": [s["column"] for s in obs_schema],
            "var_columns": var_cols,
            "x_dtype": str(getattr(adata.X, "dtype", "?")),
            "x_state_guess": _guess_x_state(adata),
            "has_genomic_coords": has_coords,
            "var_names_preview": [str(v) for v in adata.var_names[:10]],
            "categoricals": categoricals,
            "scpilot_provenance_present": "scpilot" in adata.uns,
        }
        # obs schema as a capped table preview
        tables = {"obs_schema": S.table_preview(pd.DataFrame(obs_schema), max_rows=50)}

        warnings = []
        if "counts" not in summary["layers"]:
            warnings.append("no 'counts' layer — raw counts may be unavailable for QC/CNV")
        if not has_coords:
            warnings.append("var has no chromosome/start/end — run annotate_genomic_positions before CNV (B12-pre)")

        try:
            adata.file.close()
        except Exception:  # noqa: BLE001
            pass

        return S.success(
            "inspect_h5ad",
            summary=summary,
            tables=tables,
            warnings=warnings,
            determinism_grade="A",
            suggested_next_tools=["detect_state"],
        )
    except Exception as exc:  # noqa: BLE001
        return S.error("inspect_h5ad", "internal", f"{type(exc).__name__}: {exc}")
