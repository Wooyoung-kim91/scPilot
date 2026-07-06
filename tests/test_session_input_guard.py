"""I-12 — per-input workdir default + fingerprint guard against silent input swap (root cause of I-2)."""

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from scpilot.session import InputMismatch, Session, default_workdir_for_input


def _write(path, n_obs):
    a = ad.AnnData(sparse.csr_matrix(np.ones((n_obs, 4), dtype="float32")))
    a.layers["counts"] = a.X.copy()
    a.write_h5ad(path)
    return str(path)


def test_default_workdir_for_input(tmp_path):
    wd = default_workdir_for_input(str(tmp_path / "shard_A.h5ad"))
    assert wd.endswith("shard_A_scpilot_session")
    # different inputs → different session dirs (shards don't collide)
    assert default_workdir_for_input(str(tmp_path / "shard_B.h5ad")) != wd


def test_reuse_same_input_ok(tmp_path):
    inp = _write(tmp_path / "a.h5ad", 10)
    wd = str(tmp_path / "sess")
    Session.create(wd, input_path=inp, exist_ok=True)
    # reopening the SAME workdir with the SAME input must succeed (common resume path)
    s2 = Session.create(wd, input_path=inp, exist_ok=True)
    assert s2.out == Session.create(wd, input_path=inp, exist_ok=True).out


def test_reuse_without_input_ok(tmp_path):
    inp = _write(tmp_path / "a.h5ad", 10)
    wd = str(tmp_path / "sess")
    Session.create(wd, input_path=inp, exist_ok=True)
    # reopening with no input given must not trip the guard
    Session.open(wd)
    Session.create(wd, exist_ok=True)


def test_different_input_rejected(tmp_path):
    a = _write(tmp_path / "a.h5ad", 10)
    b = _write(tmp_path / "b.h5ad", 25)      # different shape → different fingerprint
    wd = str(tmp_path / "sess")
    Session.create(wd, input_path=a, exist_ok=True)
    with pytest.raises(InputMismatch):
        Session.create(wd, input_path=b, exist_ok=True)
