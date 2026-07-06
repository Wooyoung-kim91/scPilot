"""I-20 — neutral batch-key resolution (no hardcoded 'GSM') + I-10 knobs present."""

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot.core.integrate import _resolve_batch_key


def _adata(cols):
    n = len(next(iter(cols.values())))
    a = ad.AnnData(sparse.csr_matrix(np.ones((n, 3), dtype="float32")))
    for k, v in cols.items():
        a.obs[k] = v
    return a


def test_explicit_key_used_when_present():
    a = _adata({"donor_id": ["d1", "d1", "d2"]})
    assert _resolve_batch_key(a, "donor_id") == ("donor_id", None)


def test_explicit_missing_key_errors():
    a = _adata({"donor_id": ["d1", "d2", "d2"]})
    key, err = _resolve_batch_key(a, "GSM")
    assert key is None and "absent" in err


def test_autodetect_common_column():
    # no key given → picks the first common batch column present (donor_id before GSM in the list)
    a = _adata({"donor_id": ["d1", "d1", "d2"], "GSM": ["g1", "g2", "g3"]})
    key, err = _resolve_batch_key(a, None)
    assert err is None and key == "donor_id"


def test_autodetect_none_found_errors():
    a = _adata({"random_meta": ["x", "y", "z"]})
    key, err = _resolve_batch_key(a, None)
    assert key is None and "no batch_key given" in err
