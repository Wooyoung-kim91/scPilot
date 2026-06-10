# =====================================================================
# VENDORED FROM scqc_pipeline @ source_hash debef308904633e1
#   source: /home/wykim/data/PDAC/scqc_pipeline/ (copied 2026-06-10)
# scpilot 베다링 정책: 독립 진화. import 경로·provenance 키·uns 키만
#   scpilot으로 적응했고 로직은 원본 유지. 재동기화 절차/원본 대비 diff는
#   scpilot/vendor/VENDORING.md 참조. scpilot 고유 코드는 여기 두지 말 것.
# =====================================================================
"""Robust 10x matrix reader, ported verbatim-in-spirit from PDAC_scanpy_QC_merge
notebook cell 5, generalized so the sample-id / matrix-dir columns and input root
come from the profile (no GSM/local_matrix_dir hardcoding).

Handles Cell Ranger v2/v3 layouts, gzipped or plain mtx/tsv, and an .h5 fallback.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io


def _read_tsv(path, header=None):
    return pd.read_csv(path, sep="\t", header=header, compression="infer", dtype=str)


def make_unique(values):
    """Make gene symbols unique up-front to avoid AnnData var_names warnings."""
    seen: dict[str, int] = {}
    out = []
    for value in map(str, values):
        if value not in seen:
            seen[value] = 0
            out.append(value)
        else:
            seen[value] += 1
            out.append(f"{value}-{seen[value]}")
    return out


def _first_existing(paths):
    return next((Path(p) for p in paths if Path(p).exists()), None)


def read_10x_mtx_robust(matrix_dir: Path) -> ad.AnnData:
    matrix_dir = Path(matrix_dir)
    matrix_path = _first_existing([matrix_dir / "matrix.mtx.gz", matrix_dir / "matrix.mtx"])
    barcodes_path = _first_existing([matrix_dir / "barcodes.tsv.gz", matrix_dir / "barcodes.tsv"])
    features_path = _first_existing([
        matrix_dir / "features.tsv.gz",
        matrix_dir / "features.tsv",
        matrix_dir / "genes.tsv.gz",
        matrix_dir / "genes.tsv",
    ])

    missing = []
    if matrix_path is None:
        missing.append("matrix.mtx(.gz)")
    if barcodes_path is None:
        missing.append("barcodes.tsv(.gz)")
    if features_path is None:
        missing.append("features.tsv(.gz) or genes.tsv(.gz)")
    if missing:
        raise FileNotFoundError(f"Incomplete 10x mtx files in {matrix_dir}: {', '.join(missing)}")

    matrix = io.mmread(str(matrix_path)).T.tocsr()
    barcodes = _read_tsv(barcodes_path, header=None)[0].astype(str).values
    features = _read_tsv(features_path, header=None)

    if features.shape[1] >= 2:
        gene_ids = features.iloc[:, 0].astype(str).values
        gene_symbols = features.iloc[:, 1].astype(str).values
    else:
        gene_ids = features.iloc[:, 0].astype(str).values
        gene_symbols = gene_ids.copy()

    feature_types = (
        features.iloc[:, 2].astype(str).values
        if features.shape[1] >= 3
        else np.array(["Gene Expression"] * len(gene_symbols))
    )

    adata = ad.AnnData(X=matrix)
    adata.obs_names = pd.Index(barcodes, name="barcode")
    adata.var_names = pd.Index(make_unique(gene_symbols), name="gene_symbols")
    adata.var["gene_ids"] = gene_ids
    adata.var["feature_types"] = feature_types
    return adata


def _sanitize(value):
    if pd.isna(value):
        return ""
    return str(value)


def _path_has_10x_files(path: Path) -> bool:
    path = Path(path)
    has_mtx = (path / "matrix.mtx.gz").exists() or (path / "matrix.mtx").exists()
    has_h5 = (path / "filtered_feature_bc_matrix.h5").exists()
    return has_mtx or has_h5


def resolve_matrix_dir(row: dict, *, input_root: Path, matrix_dir_col: str,
                       sample_id_col: str, batch_col: str | None = None) -> Path:
    """Locate the 10x matrix dir for one sample, trying several layout conventions.

    Returns the first candidate that actually contains 10x files; raises with the
    full candidate list otherwise (consumed by `scqc doctor`).
    """
    local_matrix_dir = Path(str(row.get(matrix_dir_col, "")))
    sample_id = str(row.get(sample_id_col, ""))
    batch = str(row.get(batch_col, "")) if batch_col else ""

    candidates: list[Path] = []

    def add(path):
        path = Path(path)
        if path not in candidates:
            candidates.append(path)

    add(input_root / local_matrix_dir)
    if local_matrix_dir.name == "filtered_feature_bc_matrix":
        add(input_root / local_matrix_dir.parent)

    parts = local_matrix_dir.parts
    if len(parts) >= 2:
        raw_path = input_root / parts[0] / "raw" / Path(*parts[1:])
        add(raw_path)
        if raw_path.name == "filtered_feature_bc_matrix":
            add(raw_path.parent)

    if batch:
        raw_root = input_root / batch / "raw"
        if raw_root.exists():
            for match in sorted(raw_root.glob(f"{sample_id}*")):
                add(match)
                add(match / "filtered_feature_bc_matrix")

    for candidate in candidates:
        if _path_has_10x_files(candidate):
            return candidate

    existing = [str(c) for c in candidates if c.exists()]
    checked = "\n".join(str(c) for c in candidates[:20])
    raise FileNotFoundError(
        f"Could not locate 10x matrix files for {sample_id}.\n"
        f"Existing candidate dirs: {existing}\nChecked:\n{checked}"
    )


def diagnose_matrix_dir(row: dict, *, input_root: Path, matrix_dir_col: str,
                        sample_id_col: str, batch_col: str | None = None) -> dict:
    """Non-raising variant for `scqc doctor`: report what was found / missing."""
    sample_id = str(row.get(sample_id_col, ""))
    try:
        found = resolve_matrix_dir(
            row, input_root=input_root, matrix_dir_col=matrix_dir_col,
            sample_id_col=sample_id_col, batch_col=batch_col,
        )
        return {"sample_id": sample_id, "ok": True, "matrix_dir": str(found)}
    except FileNotFoundError as exc:
        return {"sample_id": sample_id, "ok": False, "error": str(exc)}


def read_one_sample(row: dict, *, input_root: Path, sample_id_col: str,
                    matrix_dir_col: str, batch_col: str | None,
                    mito_prefix: str) -> ad.AnnData:
    """Read one sample, attach all metadata to obs, compute QC metrics, keep raw counts."""
    matrix_dir = resolve_matrix_dir(
        row, input_root=input_root, matrix_dir_col=matrix_dir_col,
        sample_id_col=sample_id_col, batch_col=batch_col,
    )
    sample_id = str(row.get(sample_id_col, ""))
    h5_path = matrix_dir / "filtered_feature_bc_matrix.h5"
    has_mtx = (matrix_dir / "matrix.mtx.gz").exists() or (matrix_dir / "matrix.mtx").exists()

    if h5_path.exists() and not has_mtx:
        sample = sc.read_10x_h5(str(h5_path), gex_only=True)
    else:
        try:
            sample = sc.read_10x_mtx(str(matrix_dir), var_names="gene_symbols", make_unique=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[fallback] {sample_id}: read_10x_mtx failed ({type(exc).__name__}); robust parser")
            sample = read_10x_mtx_robust(matrix_dir)

    sample.var_names_make_unique()
    sample.X = sample.X.astype(np.float32)

    # Prefix barcodes with sample id so obs_names stay unique after merge.
    original_barcodes = sample.obs_names.astype(str)
    sample.obs_names = [f"{sample_id}_{bc}" for bc in original_barcodes]
    sample.obs["barcode"] = original_barcodes
    sample.obs["sample_id"] = sample_id
    sample.obs["matrix_dir"] = str(matrix_dir)

    meta_values = {k: _sanitize(v) for k, v in row.items()}
    meta_df = pd.DataFrame(
        {k: [v] * sample.n_obs for k, v in meta_values.items()},
        index=sample.obs_names,
    )
    sample.obs = pd.concat([sample.obs, meta_df], axis=1)

    sample.var["mt"] = sample.var_names.str.upper().str.startswith(mito_prefix.upper())
    sc.pp.calculate_qc_metrics(sample, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    sample.layers["counts"] = sample.X.copy()
    return sample
