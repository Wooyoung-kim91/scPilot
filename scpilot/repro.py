"""Reproducibility harness — scpilot plan A7.

LLM-driven exploration is non-deterministic, so we separate the *deterministic
recipe* (params + decisions) from the LLM run and can replay it. This module:

- ``set_global_seed`` — pin numpy / random / torch / scvi seeds (scanpy uses the
  numpy global + per-call random_state).
- ``dataset_fingerprint`` / ``recipe_hash`` — lightweight hashing (no 6GB content
  hash by default): shape/nnz/name-hashes + params + lib versions + input ckpt id.
- ``compare_summaries`` — structural diff with **per-determinism-grade tolerance**
  (A=params/keys equal, B=structural within tolerance, C=bit-identical).
- ``replay_session`` — re-run a session's run log WITHOUT the LLM (consumes the
  recorded params/decisions) and diff against the recorded summaries.

Replay re-execution is delegated to a pluggable ``executor(record) -> summary``
which the tool registry (plan C1) will provide; until then ``replay_session`` runs
in dry-run mode (validates/parses the log and reports what it would re-run).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable

# Default tolerances per determinism grade (plan A7).
GRADE_TOLERANCE = {
    "A": {"rtol": 0.0, "cluster_tol": 0},     # params/env identical
    "B": {"rtol": 0.05, "cluster_tol": 1},    # structural equivalence within tolerance
    "C": {"rtol": 0.0, "cluster_tol": 0},     # bit-identical when achievable
}


# --------------------------------------------------------------------------- #
# Seed control
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int = 0) -> dict:
    """Pin all RNGs we can and return a record of what was set (for provenance)."""
    rec: dict = {"seed": seed}
    random.seed(seed)
    rec["python_random"] = seed
    try:
        import numpy as np
        np.random.seed(seed)
        rec["numpy"] = seed
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        torch.manual_seed(seed)
        rec["torch"] = seed
    except Exception:  # noqa: BLE001
        pass
    try:
        import scvi
        scvi.settings.seed = seed
        rec["scvi"] = seed
    except Exception:  # noqa: BLE001
        pass
    # scanpy has no global seed; functions take random_state (default 0) + numpy global.
    rec["scanpy"] = "per-call random_state (numpy global pinned)"
    return rec


# --------------------------------------------------------------------------- #
# Lightweight hashing
# --------------------------------------------------------------------------- #
def _hash_names(names) -> str:
    h = hashlib.sha256()
    for n in names:
        h.update(str(n).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def dataset_fingerprint(adata) -> dict:
    """Cheap fingerprint of an AnnData (no full content hash): shape + nnz + name hashes."""
    X = adata.X
    nnz = getattr(X, "nnz", None)
    return {
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "nnz": int(nnz) if nnz is not None else int(getattr(X, "size", 0)),
        "var_names_sha": _hash_names(adata.var_names),
        "obs_names_sha": _hash_names(adata.obs_names),
    }


def recipe_hash(*, params: dict, lib_versions: dict | None = None,
                input_checkpoint_id: str | None = None,
                fingerprint: dict | None = None) -> str:
    """Deterministic hash of the recipe inputs (params + libs + input ckpt + fingerprint)."""
    blob = json.dumps(
        {"params": params, "lib_versions": lib_versions or {},
         "input_checkpoint_id": input_checkpoint_id, "fingerprint": fingerprint or {}},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Structural diff with grade tolerance
# --------------------------------------------------------------------------- #
def compare_summaries(ref: dict, new: dict, *, grade: str = "B",
                      cluster_keys=("n_clusters", "n_leiden", "leiden_n")) -> dict:
    """Compare two summary dicts under a determinism-grade tolerance.

    Returns {match: bool, grade, diffs: [...]}. Numbers compare within ``rtol``;
    keys in ``cluster_keys`` compare within ``cluster_tol`` (integer clusters drift
    a little for leiden/umap). Grades A/C are strict on numbers; B is tolerant.
    """
    tol = GRADE_TOLERANCE.get(grade, GRADE_TOLERANCE["B"])
    rtol, ctol = tol["rtol"], tol["cluster_tol"]
    diffs: list[str] = []

    keys = set(ref) | set(new)
    for k in sorted(keys):
        if k not in ref:
            diffs.append(f"+{k} (only in new)")
            continue
        if k not in new:
            diffs.append(f"-{k} (missing in new)")
            continue
        a, b = ref[k], new[k]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            if k in cluster_keys:
                if abs(a - b) > ctol:
                    diffs.append(f"{k}: {a} -> {b} (>±{ctol})")
            else:
                denom = max(abs(a), 1e-12)
                if abs(a - b) / denom > rtol:
                    diffs.append(f"{k}: {a} -> {b} (rtol>{rtol})")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                diffs.append(f"{k}: len {len(a)} -> {len(b)}")
        else:
            if a != b:
                diffs.append(f"{k}: {a!r} -> {b!r}")
    return {"match": not diffs, "grade": grade, "diffs": diffs}


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def replay_session(out_dir: str, *, executor: Callable[[dict], dict] | None = None) -> dict:
    """Replay a session's run log deterministically (no LLM).

    ``executor(record) -> new_summary`` re-runs one logged tool with its recorded
    params and returns the new summary; the tool registry (C1) provides it. Without
    an executor this runs in **dry-run** mode: it validates/parses the log and lists
    what it would replay. Returns a report dict (JSON-serializable).
    """
    from scpilot.session import Session

    sess = Session.open(out_dir)
    runs = _read_jsonl(sess.run_log_path)
    decisions = _read_jsonl(sess.decisions_path)

    steps = []
    all_match = True
    for rec in runs:
        tool = rec.get("tool")
        grade = rec.get("determinism_grade") or "B"
        step = {"tool": tool, "grade": grade, "status": rec.get("status")}
        if executor is None:
            step["replay"] = "dry-run (no executor; tool registry pending — plan C1/A5)"
        else:
            try:
                new_summary = executor(rec)
                cmp = compare_summaries(rec.get("summary", {}), new_summary, grade=grade)
                step["diff"] = cmp
                all_match = all_match and cmp["match"]
            except Exception as exc:  # noqa: BLE001
                step["error"] = f"{type(exc).__name__}: {exc}"
                all_match = False
        steps.append(step)

    return {
        "session_id": sess.manifest.session_id,
        "out_dir": str(sess.out),
        "n_runs": len(runs),
        "n_decisions": len(decisions),
        "mode": "executed" if executor else "dry-run",
        "all_match": all_match if executor else None,
        "steps": steps,
    }
