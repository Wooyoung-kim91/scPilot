"""On-disk analysis session — scpilot plan A3 (single-out_dir model).

A ``Session`` is a working directory that owns the analysis state on disk so a
crash/restart/re-entry can resume: a JSON manifest, an append-only run log, an
append-only decision log (schema frozen in A7), and one ``.h5ad`` checkpoint per
mutating stage. The in-memory AnnData is just a cache of the latest checkpoint.

MVP decision (2026-06-10): **single out_dir, single client** — multi-client file
locking / ownership is deferred (a no-op advisory ``.lock`` marker is written for
forward-compat, but not enforced). Reproducibility hashing/replay is layered on in
A7; here we provide the durable session scaffolding + provenance/invariant helpers.

Layout::

    <out_dir>/
      session.json         # manifest (session_id, x_state, checkpoints[], stage, ...)
      run_log.jsonl        # append-only: one record per tool run
      decisions.jsonl      # append-only: LLM decision events (A7 freezes the schema)
      checkpoints/NN_<stage>.h5ad
      artifacts/           # CSV/PNG outputs
      logs/                # per-tool logs
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from scpilot import __version__
from scpilot.vendor.harness import atomic_path, build_provenance, _fingerprint, init_runtime

UNS_KEY = "scpilot"
# Default on-disk session workspace (override via env SCPILOT_RUN_DIR or --workdir).
DEFAULT_RUN_DIR = os.environ.get("SCPILOT_RUN_DIR", str(Path.home() / "data" / "scpilot_run"))
# Semantic meaning of adata.X at a given point (recorded per step; counts layer is immutable).
XState = Literal["raw_counts", "normalized", "log1p", "scaled", "unknown"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def counts_fingerprint(adata) -> dict | None:
    """Fingerprint of ``layers['counts']`` to detect accidental mutation.

    Avoids hashing the whole 6GB: shape + dtype + nnz PLUS a cheap *content* hash
    over a deterministic sample of the data values (so value drift that preserves
    nnz/shape is still caught — Codex review 2.1).
    """
    if "counts" not in getattr(adata, "layers", {}):
        return None
    import hashlib
    import numpy as np

    m = adata.layers["counts"]
    fp = {"shape": [int(adata.n_obs), int(adata.n_vars)], "dtype": str(getattr(m, "dtype", "?"))}
    nnz = getattr(m, "nnz", None)
    fp["nnz"] = int(nnz) if nnz is not None else int(getattr(m, "size", 0))
    # content hash over a deterministic stride sample of the nonzero/data values
    data = getattr(m, "data", None)            # sparse: nonzero values
    if data is None:
        data = np.asarray(m).ravel()           # dense
    data = np.asarray(data)
    if data.size:
        stride = max(1, data.size // 20000)    # cap the sample at ~20k values
        sample = np.ascontiguousarray(data[::stride])
        h = hashlib.sha256()
        h.update(str(sample.dtype).encode())
        h.update(sample.tobytes())
        fp["content_sha"] = h.hexdigest()[:16]
        fp["content_n"] = int(sample.size)
    return fp


@dataclass
class Checkpoint:
    id: str
    stage: str
    path: str
    created_at: str
    fingerprint: dict | None = None
    x_state: str = "unknown"


@dataclass
class Manifest:
    session_id: str
    scpilot_version: str
    created_at: str
    updated_at: str
    out_dir: str
    input: dict = field(default_factory=dict)          # {path, fingerprint}
    x_state: str = "unknown"
    counts_fingerprint: dict | None = None
    stage: str | None = None                            # last completed stage
    checkpoints: list = field(default_factory=list)     # list[dict] (Checkpoint asdict)
    n_runs: int = 0


class Session:
    """A durable, single-client analysis session bound to ``out_dir``."""

    MANIFEST = "session.json"

    def __init__(self, out_dir: str | Path, manifest: Manifest):
        self.out = Path(out_dir).resolve()
        self.manifest = manifest
        self._adata = None  # in-memory cache (lazy)

    # ---------- directories ----------
    @property
    def checkpoints_dir(self) -> Path:
        return self.out / "checkpoints"

    @property
    def artifacts_dir(self) -> Path:
        return self.out / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.out / "logs"

    @property
    def run_log_path(self) -> Path:
        return self.out / "run_log.jsonl"

    @property
    def decisions_path(self) -> Path:
        return self.out / "decisions.jsonl"

    def _ensure_dirs(self) -> None:
        for d in (self.out, self.checkpoints_dir, self.artifacts_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---------- construction ----------
    @classmethod
    def create(cls, out_dir: str | Path, *, input_path: str | None = None,
               session_id: str | None = None, exist_ok: bool = True) -> "Session":
        out = Path(out_dir).resolve()
        existing = out / cls.MANIFEST
        if existing.exists() and not exist_ok:
            raise FileExistsError(f"session already exists: {existing}")
        if existing.exists() and exist_ok:
            return cls.open(out)
        init_runtime()
        now = _now()
        man = Manifest(
            session_id=session_id or uuid.uuid4().hex[:12],
            scpilot_version=__version__,
            created_at=now, updated_at=now, out_dir=str(out),
            input={"path": str(Path(input_path).resolve()), "fingerprint": _fingerprint(input_path)}
            if input_path else {},
        )
        sess = cls(out, man)
        sess._ensure_dirs()
        # forward-compat advisory marker (NOT enforced in single-client MVP)
        (out / ".lock").write_text(json.dumps({"pid": os.getpid(), "at": now}))
        sess.save()
        return sess

    @classmethod
    def open(cls, out_dir: str | Path) -> "Session":
        out = Path(out_dir).resolve()
        path = out / cls.MANIFEST
        if not path.exists():
            raise FileNotFoundError(f"no session manifest at {path}")
        data = json.loads(path.read_text())
        return cls(out, Manifest(**data))

    # ---------- manifest ----------
    def save(self) -> None:
        self.manifest.updated_at = _now()
        self._ensure_dirs()
        with atomic_path(self.out / self.MANIFEST) as tmp:
            tmp.write_text(json.dumps(self.manifest.__dict__, indent=2, default=str))

    # ---------- data cache ----------
    def load_input(self, path: str | None = None, *, backed: str | None = None):
        """Load the input h5ad into the in-memory cache (and set as working adata)."""
        import anndata as ad
        src = path or self.manifest.input.get("path")
        if not src:
            raise ValueError("no input path given and none recorded in manifest")
        self._adata = ad.read_h5ad(src, backed=backed) if backed else ad.read_h5ad(src)
        if not self.manifest.input:
            self.manifest.input = {"path": str(Path(src).resolve()), "fingerprint": _fingerprint(src)}
        self._refresh_counts_state()
        return self._adata

    @property
    def adata(self):
        """The cached working AnnData; lazily load the latest checkpoint if absent."""
        if self._adata is None:
            cp = self.latest_checkpoint()
            if cp is None:
                raise RuntimeError("no working AnnData and no checkpoint to load; call load_input first")
            import anndata as ad
            self._adata = ad.read_h5ad(cp["path"])
            self._refresh_counts_state()
        return self._adata

    def set_adata(self, adata) -> None:
        self._adata = adata

    def _refresh_counts_state(self) -> None:
        if self._adata is not None and self.manifest.counts_fingerprint is None:
            self.manifest.counts_fingerprint = counts_fingerprint(self._adata)

    # ---------- provenance & invariants ----------
    def stamp_provenance(self, adata, stage: str, *, params: dict | None = None,
                         x_state: str | None = None, decision_ref: str | None = None) -> dict:
        """Write the self-describing provenance block into ``adata.uns['scpilot']``."""
        block = {
            "stage": stage,
            "scpilot_version": __version__,
            "session_id": self.manifest.session_id,
            "params": params or {},
            "x_state": x_state or self.manifest.x_state,
            "decision_ref": decision_ref,
            "provenance": build_provenance(),
        }
        adata.uns[UNS_KEY] = block
        return block

    def assert_invariants(self, adata, *, require_counts: bool = True) -> None:
        """Enforce the AnnData invariants (plan §데이터 불변식)."""
        if require_counts and "counts" not in getattr(adata, "layers", {}):
            raise AssertionError("invariant violated: layers['counts'] missing")
        ref = self.manifest.counts_fingerprint
        if ref is not None:
            cur = counts_fingerprint(adata)
            if cur is not None:
                # genes must never change; cells may shrink via legitimate filtering
                if cur["shape"][1] != ref["shape"][1]:
                    raise AssertionError(f"invariant violated: gene count changed {ref['shape']} -> {cur['shape']}")
                # value drift on the SAME cell set (no filtering) → counts was mutated
                if cur["shape"][0] == ref["shape"][0] and ref.get("content_sha") \
                        and cur.get("content_sha") and cur["content_sha"] != ref["content_sha"]:
                    raise AssertionError("invariant violated: counts values changed (content hash drift)")

    # ---------- checkpoints ----------
    def checkpoint(self, stage: str, *, adata=None, x_state: str | None = None,
                   params: dict | None = None, compression: str = "lzf") -> Checkpoint:
        """Atomically write a post-stage .h5ad checkpoint and register it in the manifest."""
        a = adata if adata is not None else self.adata
        if x_state:
            self.manifest.x_state = x_state
        self.stamp_provenance(a, stage, params=params, x_state=x_state)
        self._ensure_dirs()
        idx = len(self.manifest.checkpoints)
        cid = f"{idx:02d}_{stage}"
        path = self.checkpoints_dir / f"{cid}.h5ad"
        with atomic_path(path) as tmp:
            a.write_h5ad(tmp, compression=compression)
        cp = Checkpoint(id=cid, stage=stage, path=str(path), created_at=_now(),
                        fingerprint=_fingerprint(path), x_state=self.manifest.x_state)
        self.manifest.checkpoints.append(cp.__dict__)
        self.manifest.stage = stage
        if self.manifest.counts_fingerprint is None:
            self.manifest.counts_fingerprint = counts_fingerprint(a)
        self._adata = a
        self.save()
        return cp

    def latest_checkpoint(self) -> dict | None:
        return self.manifest.checkpoints[-1] if self.manifest.checkpoints else None

    # ---------- append-only logs ----------
    def log_run(self, record: dict) -> None:
        """Append one tool-run record to run_log.jsonl (full schema frozen in A7)."""
        rec = {"ts": _now(), "session_id": self.manifest.session_id, **record}
        self._append_jsonl(self.run_log_path, rec)
        self.manifest.n_runs += 1
        self.save()

    def log_decision(self, record: dict, *, validate: bool = True) -> None:
        """Append one LLM decision event to decisions.jsonl (schema frozen in A7).

        Accepts a dict or a ``schemas.DecisionEvent``. Validates against the frozen
        decision schema by default (raises on missing required keys).
        """
        if hasattr(record, "to_dict"):
            record = record.to_dict()
        if validate:
            from scpilot.schemas import validate_decision
            problems = validate_decision(record)
            if problems:
                raise ValueError(f"invalid decision event: {problems}")
        rec = {"ts": _now(), "session_id": self.manifest.session_id, **record}
        self._append_jsonl(self.decisions_path, rec)

    @staticmethod
    def _append_jsonl(path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def __repr__(self) -> str:
        return (f"Session(id={self.manifest.session_id}, out={self.out}, "
                f"stage={self.manifest.stage}, checkpoints={len(self.manifest.checkpoints)})")
