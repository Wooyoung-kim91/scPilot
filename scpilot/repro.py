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

# A replay executor re-runs ONE logged tool from its run-log record (which carries
# the recorded `params`/`tool`/`determinism_grade`) and returns the NEW summary dict
# to diff against the recorded `summary`. The tool registry (plan C1/A5) provides one.
ReplayExecutor = Callable[[dict], dict]   # (run_log_record) -> new_summary

# Decision types that are FORCED LLM structured outputs (mode-2 emit_* tools): recorded
# as decision events + JSON artifacts but NOT re-derived by replay — they are
# non-deterministic LLM products (plan E1 / agent.py KNOWN BOUNDARY). Replay surfaces
# them so a green ``all_match`` is not misread as full-fidelity reproduction.
NON_REPLAYABLE_DECISIONS = ("annotation_strategy", "de_design")

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
def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _diff_value(path: str, key: str, a, b, *, rtol: float, ctol: int,
                cluster_keys) -> list[str]:
    """Recursively diff one (a, b) pair at dotted ``path`` (leaf name ``key``).

    Numbers use ``rtol`` (or ``ctol`` integer tolerance for ``cluster_keys``);
    dicts recurse key-by-key; equal-length lists compare elementwise (so nested
    summary structures like ``label_distribution`` get the grade tolerance instead
    of an all-or-nothing length check); everything else compares exactly.
    """
    diffs: list[str] = []
    if _is_number(a) and _is_number(b):
        if key in cluster_keys:
            if abs(a - b) > ctol:
                diffs.append(f"{path}: {a} -> {b} (>±{ctol})")
        else:
            denom = max(abs(a), 1e-12)
            if abs(a - b) / denom > rtol:
                diffs.append(f"{path}: {a} -> {b} (rtol>{rtol})")
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            sub = f"{path}.{k}"
            if k not in a:
                diffs.append(f"+{sub} (only in new)")
            elif k not in b:
                diffs.append(f"-{sub} (missing in new)")
            else:
                diffs += _diff_value(sub, k, a[k], b[k], rtol=rtol, ctol=ctol,
                                     cluster_keys=cluster_keys)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: len {len(a)} -> {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                # list elements inherit the parent key's cluster-ness
                diffs += _diff_value(f"{path}[{i}]", key, x, y, rtol=rtol, ctol=ctol,
                                     cluster_keys=cluster_keys)
    else:
        if a != b:
            diffs.append(f"{path}: {a!r} -> {b!r}")
    return diffs


def compare_summaries(ref: dict, new: dict, *, grade: str = "B",
                      cluster_keys=("n_clusters", "n_leiden", "leiden_n")) -> dict:
    """Compare two summary dicts under a determinism-grade tolerance.

    Returns {match: bool, grade, diffs: [...]}. Numbers compare within ``rtol``;
    keys in ``cluster_keys`` compare within ``cluster_tol`` (integer clusters drift
    a little for leiden/umap). Grades A/C are strict on numbers; B is tolerant.
    Nested dicts/lists are compared recursively (plan A4) so structural-invariant
    summaries with nested values are not silently passed by a top-level-only check.
    """
    tol = GRADE_TOLERANCE.get(grade, GRADE_TOLERANCE["B"])
    rtol, ctol = tol["rtol"], tol["cluster_tol"]
    diffs: list[str] = []

    for k in sorted(set(ref) | set(new)):
        if k not in ref:
            diffs.append(f"+{k} (only in new)")
        elif k not in new:
            diffs.append(f"-{k} (missing in new)")
        else:
            diffs += _diff_value(k, k, ref[k], new[k], rtol=rtol, ctol=ctol,
                                 cluster_keys=cluster_keys)
    return {"match": not diffs, "grade": grade, "diffs": diffs}


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def replay_session(out_dir: str, *, executor: "ReplayExecutor | None" = None,
                   verify_session=None) -> dict:
    """Replay a session's run log deterministically (no LLM).

    ``executor(record) -> new_summary`` re-runs one logged tool with its recorded
    params and returns the new summary; the tool registry (C1) provides it. Without
    an executor this runs in **dry-run** mode: it validates/parses the log and lists
    what it would replay. Returns a report dict (JSON-serializable).

    Plan A3 — replay does NOT trust summary numbers alone:
    - per mutating step the executor re-runs the tool through ``Session.checkpoint``,
      which enforces the AnnData invariants (B1); a counts/gene violation therefore
      raises during replay and is surfaced here as a step error (``all_match=False``).
    - when the re-execution ``Session`` is passed as ``verify_session``, its raw-counts
      fingerprint is cross-checked against the original so a counts layer that
      reproduced *differently* is caught even if every summary number matched.
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
            step["replay"] = "dry-run (pass an executor — tools.make_replay_executor — to re-execute)"
        elif rec.get("status") != "success":
            # only successful original runs have a summary to reproduce/diff
            step["replay"] = f"skipped (original status={rec.get('status')})"
        else:
            try:
                new_summary = executor(rec)
                cmp = compare_summaries(rec.get("summary", {}), new_summary, grade=grade)
                step["diff"] = cmp
                # invariants were re-checked inside checkpoint() during re-execution (B1)
                step["invariants_ok"] = True
                all_match = all_match and cmp["match"]
            except Exception as exc:  # noqa: BLE001
                step["error"] = f"{type(exc).__name__}: {exc}"
                step["invariants_ok"] = not isinstance(exc, AssertionError)
                all_match = False
        steps.append(step)

    report = {
        "session_id": sess.manifest.session_id,
        "out_dir": str(sess.out),
        "n_runs": len(runs),
        "n_decisions": len(decisions),
        "mode": "executed" if executor else "dry-run",
        "all_match": all_match if executor else None,
        "steps": steps,
    }

    # E1: surface forced LLM structured outputs that replay does NOT re-derive, so a green
    # all_match is not misread. The deterministic tool recipe (incl. apply_annotation's label
    # params, which ARE in the run log) replays; only the standalone emit_* JSON products do not.
    skipped = [d for d in decisions if d.get("decision_type") in NON_REPLAYABLE_DECISIONS]
    if skipped:
        report["structured_decisions_not_reexecuted"] = {
            "count": len(skipped),
            "types": sorted({d.get("decision_type") for d in skipped}),
            "note": "forced LLM structured outputs (annotation labels / DE design) are recorded "
                    "as decision events + JSON artifacts but NOT re-derived by replay; restore "
                    "them from the decision log. all_match reflects the deterministic tool recipe only.",
        }

    # A3 cross-check: did the raw counts layer reproduce identically?
    if executor and verify_session is not None:
        def _content_sha(s) -> str | None:
            fp = getattr(getattr(s, "manifest", None), "counts_fingerprint", None) or {}
            return fp.get("content_sha")

        orig_sha, new_sha = _content_sha(sess), _content_sha(verify_session)
        fp_match = (orig_sha == new_sha) if (orig_sha and new_sha) else None
        report["counts_fingerprint_match"] = fp_match
        if fp_match is False:
            report["all_match"] = False

    return report
