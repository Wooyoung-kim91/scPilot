"""I-22 — resolution chosen by embedding separability (silhouette), not raw n_clusters."""

from scpilot.recipes import suggest_resolution


def test_prefers_best_silhouette():
    # res 0.3 separates best; 0.4/0.5 add clusters but silhouette DROPS (over-clustering) → not chosen
    sweep = [(0.1, 2, 0.30), (0.2, 3, 0.45), (0.3, 4, 0.62), (0.4, 9, 0.25), (0.5, 14, 0.10)]
    r, why = suggest_resolution(sweep)
    assert r == 0.3 and "silhouette" in why


def test_excludes_singleton_clusterings():
    # a resolution with <2 clusters can't have a meaningful silhouette and is ignored
    sweep = [(0.1, 1, None), (0.2, 2, 0.50), (0.3, 3, 0.41)]
    assert suggest_resolution(sweep)[0] == 0.2


def test_falls_back_to_knee_without_silhouette():
    # legacy 2-tuples (or all-None silhouette) → n_clusters knee, unchanged behavior
    knee = [(0.1, 5), (0.2, 6), (0.3, 6), (0.4, 15), (0.5, 16)]
    assert suggest_resolution(knee, jump_ratio=1.5)[0] == 0.3
    flat = [(0.1, 4, None), (0.2, 4, None), (0.3, 5, None)]
    assert suggest_resolution(flat, jump_ratio=1.5)[0] == 0.1
