"""Unit tests for the tool registry + step dispatch — scpilot plan A5."""

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from scpilot import tools
from scpilot.cli import _coerce, _parse_params
from scpilot.session import Session


def _tiny_h5ad(path):
    X = sparse.csr_matrix(np.random.default_rng(0).poisson(1.0, (30, 20)).astype("float32"))
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(20)]
    a.layers["counts"] = a.X.copy()
    a.write_h5ad(path)


def test_registry_has_inspect():
    names = {t["name"] for t in tools.list_tools()}
    assert "inspect" in names
    spec = tools.get("inspect")
    assert spec.mutating is False


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        tools.get("does_not_exist")


def test_run_inspect_through_registry(tmp_path):
    h5ad = tmp_path / "tiny.h5ad"
    _tiny_h5ad(h5ad)
    sess = Session.create(tmp_path / "sess", input_path=str(h5ad))
    result = tools.run("inspect", sess)
    assert result.status == "success"
    assert result.summary["n_obs"] == 30
    assert result.summary["n_vars"] == 20
    # JSON-serializable per the frozen contract
    result.to_dict()


def test_param_coercion():
    assert _coerce("0.5") == 0.5
    assert _coerce("30") == 30
    assert _coerce("true") is True
    assert _coerce("harmony") == "harmony"
    assert _parse_params(["resolution=0.5", "n_pcs=30", "batch_aware=true"]) == {
        "resolution": 0.5, "n_pcs": 30, "batch_aware": True,
    }


def test_parse_params_rejects_bad():
    import typer
    with pytest.raises(typer.BadParameter):
        _parse_params(["noequals"])
