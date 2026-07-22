"""Unit test for the scib ``benchmark`` tool's best-embedding side effect (issue #3).

``benchmark`` is a NON-mutating tool. It must NOT silently mutate ``adata.uns`` with the chosen
best embedding as its ONLY record of that choice — a state mutation that bypasses the checkpoint
chokepoint is not content-addressed and is not reproduced on replay. Instead the pick is logged as
a first-class DECISION EVENT (``decisions.jsonl``); the ``uns['scpilot']['best_embedding']`` key is
still exposed so downstream readers (autoplot / report / export_final) keep working.
"""

import json

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from scpilot import tools
from scpilot.session import Session


def _bench_session(tmp_path):
    """Two well-separated embeddings + a batch key + an embedding-independent label key."""
    rng = np.random.default_rng(0)
    n_per, n_lab = 40, 3
    n = n_per * n_lab
    labels, centers = [], rng.normal(0, 8, (n_lab, 10))
    emb = np.zeros((n, 10), dtype="float32")
    for i in range(n_lab):
        emb[i * n_per:(i + 1) * n_per] = centers[i] + rng.normal(0, 1, (n_per, 10))
        labels += [f"type{i}"] * n_per
    X = sparse.csr_matrix(rng.poisson(0.5, (n, 6)).astype("float32"))
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(6)]
    a.layers["counts"] = a.X.copy()
    a.obs["major_cell_type"] = labels
    a.obs["sample_id"] = ([f"s{j % 2}" for j in range(n)])   # 2 batches
    a.obsm["X_pca"] = emb
    a.obsm["X_harmony"] = emb + rng.normal(0, 0.05, emb.shape).astype("float32")
    p = tmp_path / "b.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    return s


def test_benchmark_best_embedding_logged_not_silently_mutated(tmp_path):
    pytest.importorskip("scib_metrics")
    s = _bench_session(tmp_path)
    r = tools.run("benchmark", s, label_key="major_cell_type", batch_key="sample_id",
                  subsample=None, seed=0)
    if r.status != "success":
        pytest.skip(f"benchmark unavailable in this env: {r.error_code} {r.error}")

    best = r.summary["best"]
    assert best in ("X_pca", "X_harmony")

    # (1) the pick is recorded as a first-class DECISION EVENT (not only a silent uns write)
    assert s.decisions_path.exists(), "benchmark did not log a decision event"
    events = [json.loads(l) for l in s.decisions_path.read_text().splitlines() if l.strip()]
    picks = [e for e in events if e.get("decision_type") == "integration_method"
             and e.get("stage") == "benchmark"]
    assert len(picks) == 1, events
    ev = picks[0]
    assert ev["choice"] == best
    assert best in ev["candidates"]
    assert ev["params"]["best_embedding"] == best
    assert ev["rationale"]                                   # a non-empty reason is recorded

    # (2) downstream readers still work: uns exposes the same best embedding
    assert s.adata.uns.get("scpilot", {}).get("best_embedding") == best
