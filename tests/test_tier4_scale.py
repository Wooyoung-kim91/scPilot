"""I-15 (vectorized majority vote) + I-17 (non-silent evidence cap)."""

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot.llm.agent import _cap_evidence
from scpilot.recipes import consensus_vote, majority_vote


def _adata(cols):
    n = len(next(iter(cols.values())))
    a = ad.AnnData(sparse.csr_matrix(np.ones((n, 3), dtype="float32")))
    for k, v in cols.items():
        a.obs[k] = v
    return a


def test_majority_vote_three_keys():
    a = _adata({"k1": ["T", "T", "T", "A"],
                "k2": ["T", "T", "B", "B"],
                "k3": ["T", "B", "C", "C"]})
    out, agree, pairwise = majority_vote(a, ["k1", "k2", "k3"], min_agreement=0.5)
    # cell0 all-T→T(1.0); cell1 T,T,B→T(2/3>0.5); cell2 T,B,C→3-way tie→ambiguous; cell3 A,B,C→tie
    assert list(out) == ["T", "T", "ambiguous", "ambiguous"]
    assert np.allclose(agree, [1.0, 2 / 3, 1 / 3, 1 / 3])
    assert set(pairwise) == {"k1__vs__k2", "k1__vs__k3", "k2__vs__k3"}


def test_majority_vote_two_key_tie_is_ambiguous():
    a = _adata({"k1": ["X", "X"], "k2": ["X", "Y"]})
    out, agree, _ = majority_vote(a, ["k1", "k2"], min_agreement=0.5)
    assert list(out) == ["X", "ambiguous"]        # 1/2 is not > 0.5, and it is a tie
    assert np.allclose(agree, [1.0, 0.5])


def test_majority_vote_ignores_nan_no_string_nan_winner():
    # NaN/empty labels are "no opinion", NOT the literal category "nan". If they were coerced to
    # "nan", that string could WIN the vote and be written as the consensus label.
    a = _adata({"k1": ["T", np.nan, "T"],
                "k2": ["T", np.nan, "B"],
                "k3": [np.nan, np.nan, np.nan]})
    out, agree, pairwise = majority_vote(a, ["k1", "k2", "k3"], min_agreement=0.5)
    # "nan" must never appear as a chosen label
    assert "nan" not in list(out)
    # cell0: T,T over 2 valid → T (2/2 > 0.5). cell1: all missing → ambiguous.
    # cell2: T,B over 2 valid → 1/2, not > 0.5 and tie → ambiguous.
    assert list(out) == ["T", "ambiguous", "ambiguous"]
    assert np.allclose(agree, [1.0, 0.0, 0.5])
    # pairwise concordance is computed only over cells where BOTH columns have a real label:
    # k1 vs k2 co-labelled at cell0 (T==T) and cell2 (T!=B) → 0.5; pairs with fully-missing k3 → 0.0
    assert pairwise["k1__vs__k2"] == 0.5
    assert pairwise["k1__vs__k3"] == 0.0 and pairwise["k2__vs__k3"] == 0.0


def test_majority_vote_nan_never_wins_when_most_frequent():
    # Even when the missing sentinel is the most frequent raw value, it must not become the winner.
    a = _adata({"k1": [np.nan, "T"], "k2": [np.nan, "T"], "k3": ["B", np.nan]})
    out, _, _ = majority_vote(a, ["k1", "k2", "k3"], min_agreement=0.5)
    # cell0: only real label is B (1 valid) → B wins (1/1 > 0.5), NOT "nan"
    # cell1: T,T over 2 valid → T
    assert list(out) == ["B", "T"]


def test_consensus_vote_writes_obs():
    a = _adata({"m1": ["A", "A", "B"], "m2": ["A", "B", "B"]})
    a, info = consensus_vote(a, keys=["m1", "m2"], out_key="cons")
    assert list(a.obs["cons"].astype(str)) == ["A", "ambiguous", "B"]
    assert "cons_agreement" in a.obs.columns
    assert info["n_ambiguous"] == 1 and info["out_key"] == "cons"


def test_cap_evidence_no_truncation():
    text = "x" * 100
    out, warn = _cap_evidence(text, max_chars=200)
    assert out == text and warn is None


def test_cap_evidence_truncates_visibly():
    text = "y" * 500
    out, warn = _cap_evidence(text, max_chars=100, what="audit evidence")
    assert warn is not None and "omitted" in warn                # caller gets a warning (not silent)
    assert "TRUNCATED" in out and "audit evidence" in out         # reviewer SEES the cut
    assert out.startswith("y" * 100)                              # kept prefix + marker
