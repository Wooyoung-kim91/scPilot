"""Unit test for B0 ingest (raw 10x → merged) — scpilot end-to-end (decision A).

Builds tiny synthetic 10x sample dirs + a metadata CSV + a profile, then runs the
``ingest`` tool and checks the merged AnnData (counts + scale.data, samples merged).
"""

import gzip

import numpy as np
import pandas as pd
import yaml
from scipy import io as sio, sparse

from scpilot import tools
from scpilot.session import Session


def _gz(path, text):
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def _write_10x(sample_dir, n_genes=30, n_cells=40, seed=0):
    """Write a minimal CellRanger v3 10x dir (gzipped matrix/barcodes/features)."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    M = sparse.csr_matrix(rng.poisson(1.0, (n_genes, n_cells)).astype(int))  # genes × cells
    # mmwrite then gzip (scanpy reads matrix.mtx.gz)
    raw = sample_dir / "matrix.mtx"
    sio.mmwrite(str(raw), M, field="integer")
    _gz(sample_dir / "matrix.mtx.gz", raw.read_text())
    raw.unlink()
    _gz(sample_dir / "barcodes.tsv.gz", "\n".join(f"BC{i}" for i in range(n_cells)) + "\n")
    feats = [("ENSG%05d" % i, ("MT-G%d" % i if i < 3 else "G%d" % i), "Gene Expression")
             for i in range(n_genes)]
    _gz(sample_dir / "features.tsv.gz", "\n".join("\t".join(f) for f in feats) + "\n")


def _make_dataset(tmp_path):
    raw = tmp_path / "raw"
    for sid in ("S1", "S2"):
        _write_10x(raw / sid, seed=hash(sid) % 100)
    # NB: matrix-dir column must NOT be named 'matrix_dir' (read_one_sample hardcodes
    # obs['matrix_dir']); real profiles use 'local_matrix_dir'.
    meta = pd.DataFrame({"GSM": ["S1", "S2"], "local_matrix_dir": ["S1", "S2"],
                         "GSE": ["GSEa", "GSEa"], "condition": ["PDAC", "Normal"]})
    meta.to_csv(tmp_path / "meta.csv", index=False)
    profile = {
        "profile_name": "test", "input_root": str(raw), "metadata_csv": str(tmp_path / "meta.csv"),
        "out_dir": str(tmp_path / "out"), "sample_id_col": "GSM",
        "matrix_dir_col": "local_matrix_dir", "batch_col": "GSE",
        "min_genes": 0, "max_pct_mt": 100.0, "min_cells": 1,
    }
    ppath = tmp_path / "profile.yaml"
    yaml.safe_dump(profile, open(ppath, "w"))
    return str(ppath)


def test_ingest_builds_merged_from_raw_10x(tmp_path):
    profile = _make_dataset(tmp_path)
    s = Session.create(tmp_path / "sess", input_path=profile)
    r = tools.run("ingest", s, profile=profile)
    assert r.status == "success", r.error
    sm = r.summary
    assert sm["n_samples_merged"] == 2
    assert sm["n_cells"] == 80                      # 40 + 40, min_genes=0 keeps all
    assert "counts" in sm["layers"] and "scale.data" in sm["layers"]
    assert sm["condition_counts"] == {"PDAC": 40, "Normal": 40}
    # session now holds the merged + downstream-ready; checkpoint written
    assert "counts" in s.adata.layers and "scale.data" in s.adata.layers
    assert r.checkpoint
    r.to_dict()


def test_ingest_requires_profile(tmp_path):
    s = Session.create(tmp_path / "s2")          # no input/profile
    r = tools.run("ingest", s)
    assert r.status == "error" and r.error_code == "missing_input"


def test_registry_has_ingest():
    assert "ingest" in {t["name"] for t in tools.list_tools()}
