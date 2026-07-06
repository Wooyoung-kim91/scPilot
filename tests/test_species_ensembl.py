"""I-11 — Ensembl-ID var_names entry handling (CELLxGENE).

CELLxGENE stores Ensembl gene IDs as var_names with symbols in a var column
(``feature_name``). Left as-is, ``MT-``/``RPS`` prefix matching finds nothing and
``pct_counts_mt`` is silently 0. These tests pin the evidence-based remap + the
organism-detection guard + the ``session.load_input`` entry-point wiring.
"""

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scpilot.core import _species
from scpilot.session import Session


def _ensembl_adata(n_obs=50, with_symbols=True, organism="human"):
    """AnnData whose var_names are Ensembl IDs; symbols (incl. a mito gene) live in feature_name."""
    prefix = "ENSG" if organism == "human" else "ENSMUSG"
    n_vars = 6
    ids = [f"{prefix}{i:011d}" for i in range(n_vars)]
    mt_sym = "MT-ND1" if organism == "human" else "mt-Nd1"
    symbols = [mt_sym, "CD3D", "EPCAM", "RPS6", "ACTB", "GAPDH"]
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(1.0, size=(n_obs, n_vars)).astype("float32"))
    a = ad.AnnData(X)
    a.obs_names = [f"cell{i}" for i in range(n_obs)]
    a.var_names = ids
    if with_symbols:
        a.var["feature_name"] = symbols
    a.layers["counts"] = a.X.copy()
    return a, ids, symbols


def test_looks_like_ensembl():
    a, ids, _ = _ensembl_adata()
    assert _species.looks_like_ensembl(a.var_names)
    b, _, syms = _ensembl_adata()
    b.var_names = syms
    assert not _species.looks_like_ensembl(b.var_names)


def test_normalize_remaps_and_preserves_ids():
    a, ids, symbols = _ensembl_adata(organism="human")
    ev = _species.normalize_var_symbols(a)
    assert ev["remapped"] is True
    assert ev["symbol_column"] == "feature_name"
    assert list(a.var_names) == symbols                      # var_names are now symbols
    assert list(a.var["gene_ids"]) == ids                    # original IDs preserved
    # the mito gene is now matchable by the human MT- prefix (the whole point)
    assert bool(a.var_names.str.upper().str.startswith("MT-").any())


def test_normalize_is_noop_on_symbols():
    a, _, symbols = _ensembl_adata()
    a.var_names = symbols
    ev = _species.normalize_var_symbols(a)
    assert ev["remapped"] is False and ev["reason"] == "not_ensembl"


def test_normalize_warns_without_symbol_column():
    a, ids, _ = _ensembl_adata(with_symbols=False)
    ev = _species.normalize_var_symbols(a)
    assert ev["remapped"] is False and ev["reason"] == "ensembl_but_no_symbol_column"
    assert list(a.var_names) == ids                          # unchanged — no silent damage


def test_detect_organism_defers_on_ensembl():
    # Ensembl IDs must NOT be misread as "human" by casing.
    for org in ("human", "mouse"):
        a, _, _ = _ensembl_adata(organism=org)
        det = _species.detect_organism(a)
        assert det["organism"] == "unknown"
        assert "Ensembl" in det["evidence"]
    # after remap, organism detection works again
    a, _, _ = _ensembl_adata(organism="mouse")
    _species.normalize_var_symbols(a)
    assert _species.detect_organism(a)["organism"] == "mouse"


def test_load_input_normalizes_at_entry(tmp_path):
    a, ids, symbols = _ensembl_adata(organism="human")
    p = tmp_path / "census_like.h5ad"
    a.write_h5ad(p)
    sess = Session.create(str(tmp_path / "sess"), input_path=str(p), exist_ok=True)
    loaded = sess.load_input(str(p))
    assert list(loaded.var_names) == symbols
    assert "scpilot_var_symbol_normalization" in loaded.uns
    assert loaded.uns["scpilot_var_symbol_normalization"]["remapped"] is True
