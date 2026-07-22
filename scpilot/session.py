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

import hashlib
import json
import os
import pprint
import shutil
import uuid
import warnings
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


class InputMismatch(RuntimeError):
    """Raised (I-12) when a session dir is reused with a DIFFERENT input than it was created for —
    e.g. two shards sharing one --workdir. Prevents the silent reuse that caused invalid_state (I-2)."""


def default_workdir_for_input(input_path: str) -> str:
    """Per-input session dir (``<stem>_scpilot_session`` next to the input) — the single source used
    by BOTH the MCP server and mode-2 ``run`` so different inputs (shards) never collide on one dir."""
    p = Path(input_path).resolve()
    return str(p.parent / f"{p.stem}_scpilot_session")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pkg_dir() -> Path:
    import scpilot
    return Path(scpilot.__file__).resolve().parent


def scpilot_source_hash() -> str:
    """sha256 over all scpilot package .py source (stable, order-independent)."""
    h = hashlib.sha256()
    for p in sorted(_pkg_dir().rglob("*.py")):
        h.update(p.relative_to(_pkg_dir()).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _one_line(s: str) -> str:
    """Collapse a string to a single line so it can't break out of a generated comment
    (newline/CR → space). Used for the human-readable ``Input:`` comment in repro code;
    executable interpolations use ``repr()`` instead."""
    return " ".join(str(s).splitlines()).strip()


_PIPELINE_TEMPLATE = '''#!/usr/bin/env python
"""Full scpilot pipeline — AUTO-GENERATED; reproduces the ENTIRE analysis from the input.

Read top->bottom for the whole flow: each step is one tool call with its resolved
params, in execution order. The ACTUAL implementation of every tool used is inlined
(as comments) at the bottom so the full code is verifiable in one file; the exact
importable source is also pinned under {snapshot_rel}/.

Provenance:
{prov}

Input: {input_comment}
"""
import sys
from pathlib import Path

# pin imports to the exact scpilot source that produced this analysis
sys.path.insert(0, str(Path(__file__).resolve().parent / "{snapshot_rel}"))

from scpilot import tools
from scpilot.repro import set_global_seed
from scpilot.session import Session

INPUT = {input_lit}
SEED = {seed}


def main():
    set_global_seed(SEED)
    sess = Session.create(str(Path(__file__).resolve().parent.parent / "repro_pipeline"),
                          input_path=INPUT, exist_ok=True)
    sess.load_input()
    # ============================== FLOW ==============================
{flow}
    return sess


if __name__ == "__main__":
    s = main()
    print("pipeline reproduced through stage:", s.manifest.stage)


# ====================================================================
# TOOL IMPLEMENTATIONS — the actual code executed at each step above
# (inlined for inspection; the runnable copy is in {snapshot_rel}/)
# ====================================================================
{sources}
'''


# Cell-delimited (jupytext "percent") notebook — open directly in Jupyter/VSCode and run
# CELL BY CELL: each step prints its status/warnings/summary and renders its plot inline.
# MCP-free (uses the pinned scpilot package only). This is the human-facing, visual repro
# artifact. NO helper functions (no `def`) — every step's actions are written out flat in
# its own cell so you can watch the whole process and edit any step in place.
_NOTEBOOK_TEMPLATE = '''# %% [markdown]
# # scpilot pipeline — reproducible notebook (AUTO-GENERATED)
# Run top-to-bottom, **cell by cell**, to watch every step: each prints its status,
# warnings, result summary and renders its plot inline. No MCP server needed, no helper
# functions — each cell is flat, self-contained, and editable.
#
# Provenance:
# {prov_md}
#
# Input: `{input_comment}`

# %% [markdown]
# ## setup — pin the scpilot source, seed the RNG, open the session

# %%
import json, sys
from pathlib import Path
# RISK #21: resolve the pinned-source dir relative to THIS file (robust to the run CWD);
# fall back to CWD only in interactive sessions where __file__ is undefined.
try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()
sys.path.insert(0, str(_HERE / "{snapshot_rel}"))   # pin exact scpilot source
_RUN_DIR = _HERE.parent                              # code/ -> session root
from IPython.display import Image, display           # this notebook runs in a Jupyter kernel

from scpilot import tools
from scpilot.repro import set_global_seed
from scpilot.session import Session
from scpilot.core.autoplot import auto_plots

set_global_seed({seed})
sess = Session.create(str(_RUN_DIR / "repro_notebook"), input_path={input_lit}, exist_ok=True)
sess.load_input()
print("session:", sess.out)
{cells}
'''


def _notebook_cell(cid: str, stage: str, params: dict, seed: int = 0,
                   reasoning: str | None = None) -> str:
    """One markdown + one code cell for a pipeline step (percent format).

    TUTORIAL STYLE: every parameter chosen for this step is written as an EXPLICIT, editable
    keyword argument (one per line) AND listed in the markdown, so the reader sees — from file
    input through each step — exactly which settings were used and can tweak any value and re-run.

    The code is written out FLAT — no helper function — so the reader sees exactly what each step
    does: re-pin the recorded seed, run the stage, then print status / warnings / error / summary
    and render any plot inline. Re-pinning the step's seed makes the cell self-contained: running
    top-to-bottom — or re-running a single cell — reproduces the SAME result as the recorded run,
    independent of accumulated kernel RNG state (cell-by-cell determinism requirement).
    """
    if params:
        md_params = "\n".join(f"# - `{k}` = `{v!r}`" for k, v in params.items())
        kwargs = "".join(f"    {k}={v!r},\n" for k, v in params.items())
        call = f'_res = tools.run("{stage}", sess,\n{kwargs})\n'
    else:
        md_params = "# - _(no parameters — tool defaults)_"
        call = f'_res = tools.run("{stage}", sess)\n'
    # the WHY (rationale recorded for this step) as a markdown blockquote — turns the cell into a
    # tutorial ("we chose these params because …"); whitespace collapsed + capped for readability.
    why_md = ""
    if reasoning:
        _why = " ".join(str(reasoning).split())
        if len(_why) > 600:
            _why = _why[:600].rstrip() + "…"
        why_md = f"# > **why:** {_why}\n"
    return (
        f'\n# %% [markdown]\n# ## {cid} · {stage}\n'
        f'{why_md}'
        f'# **parameters** (edit any value, then re-run the cell to explore):\n{md_params}\n\n'
        f'# %%\n'
        f'set_global_seed({seed})\n'
        f'{call}'
        f'print("[" + _res.status + "]", "{stage}", "·", _res.determinism_grade or "")\n'
        f'for _w in (_res.warnings or []):\n'
        f'    print("  warn:", _w)\n'
        f'if _res.error:\n'
        f'    print("  ERROR[" + str(_res.error_code) + "]:", _res.error)\n'
        f'print(json.dumps(_res.summary, indent=2, default=str)[:2500])\n'
        f'for _a in (auto_plots(sess, "{stage}", _res.summary) or []):\n'
        f'    if getattr(_a, "kind", None) == "png":\n'
        f'        display(Image(filename=_a.path))\n'
    )


def _step_script(idx: int, cid: str, stage: str, params: dict, seed: int, snapshot_rel: str,
                 in_expr: str, reasoning: str | None = None, is_first: bool = False) -> str:
    """One numbered step script (``code/NN_<stage>.py``) in the per-step tutorial chain.

    Each step reads the previous step's ``standalone_data/NN_<stage>.h5ad`` and writes its own, so
    the files chain when run IN ORDER. Stages that have been transpiled (``scriptgen.EMITTERS``)
    emit STANDALONE plain-scanpy code (no scpilot import); stages not yet transpiled fall back to a
    thin scpilot-backed step that reads/writes the SAME h5ad chain — so converting a tool to
    standalone is a drop-in replacement and the chain never breaks mid-rollout."""
    from . import scriptgen

    standalone = scriptgen.build(idx, cid, stage, params, in_expr, reasoning=reasoning)
    if standalone is not None:
        return standalone

    # ---- fallback: not-yet-transpiled stage, still scpilot-backed (h5ad-chained) ----
    if params:
        md = "\n".join(f"  - {k} = {v!r}" for k, v in params.items())
        kw = "".join(f"    {k}={v!r},\n" for k, v in params.items())
        call = f'_res = tools.run("{stage}", sess,\n{kw})'
    else:
        md = "  - (no parameters — tool defaults)"
        call = f'_res = tools.run("{stage}", sess)'
    why = ("why: " + " ".join(str(reasoning).split())[:400]) if reasoning else \
        "scpilot deterministic step (re-pins its recorded seed → reproducible)."
    return (
        f'#!/usr/bin/env python\n"""\n'
        f'{cid}_{stage}.py — scpilot pipeline step {idx} (NOT YET TRANSPILED — still uses scpilot; '
        f'will become standalone scanpy). Run the NN_*.py files IN ORDER.\n'
        f'{why}\n'
        f'parameters:\n{md}\n'
        f'outputs: standalone_data/{cid}_{stage}.h5ad\n"""\n'
        f'import json, sys\n'
        f'from pathlib import Path\n'
        f'_HERE = Path(__file__).resolve().parent\n'
        f'sys.path.insert(0, str(_HERE / "{snapshot_rel}"))   # pinned scpilot source\n'
        f'_DATA = _HERE.parent / "standalone_data"\n'
        f'_DATA.mkdir(exist_ok=True)\n'
        f'from scpilot import tools\n'
        f'from scpilot.repro import set_global_seed\n'
        f'from scpilot.session import Session\n'
        f'from scpilot.core.autoplot import auto_plots\n\n'
        f'set_global_seed({seed})\n'
        f'IN  = {in_expr}\n'
        f'OUT = _DATA / "{cid}_{stage}.h5ad"\n'
        f'sess = Session.create(str(_DATA / "_sess_{cid}_{stage}"), input_path=str(IN), exist_ok=True)\n'
        # step 0 reads the raw session input (e.g. a profile YAML) — the ingest/load tool loads it
        # itself; later steps explicitly load the previous step's h5ad as the chain input.
        + ("" if is_first
           else f'sess.load_input(path=str(IN))            # read the previous step\'s h5ad (chain input)\n')
        + f'{call}\n'
        f'sess.adata.write_h5ad(OUT)               # hand off to the next step\n'
        f'print("[{cid}] {stage}:", _res.status, _res.determinism_grade or "", "->", OUT.name)\n'
        f'for _w in (_res.warnings or []):\n'
        f'    print("    warn:", _w)\n'
        f'if _res.error:\n'
        f'    print("    ERROR[" + str(_res.error_code) + "]:", _res.error)\n'
        f'print(json.dumps(_res.summary, indent=2, default=str)[:2000])\n'
        f'for _a in (auto_plots(sess, "{stage}", _res.summary) or []):\n'
        f'    print("    plot:", getattr(_a, "path", _a))\n'
    )


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


def _artifact_sha(path: str, *, cap_bytes: int = 100 * 1024 * 1024) -> str | None:
    """sha256 of an output artifact for integrity/audit (None if missing or > ``cap_bytes``).

    Capped so we never hash a multi-GB h5ad; figures/tables/JSON are small. This is a
    provenance fingerprint, NOT a replay-equality check (PNG bytes are non-deterministic).
    """
    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size > cap_bytes:
            return None
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:  # noqa: BLE001 — hashing must never break logging
        return None


@dataclass
class Checkpoint:
    id: str
    stage: str
    path: str
    created_at: str
    fingerprint: dict | None = None
    x_state: str = "unknown"
    params: dict = field(default_factory=dict)   # resolved params (for the full-pipeline script)


@dataclass
class Manifest:
    session_id: str
    scpilot_version: str
    created_at: str
    updated_at: str
    out_dir: str
    input: dict = field(default_factory=dict)          # {path, fingerprint}
    derived_from: dict | None = None                    # child sessions: parent provenance pointer
    x_state: str = "unknown"
    counts_fingerprint: dict | None = None
    stage: str | None = None                            # last completed stage
    checkpoints: list = field(default_factory=list)     # list[dict] (Checkpoint asdict)
    pipeline_script: str | None = None                  # code/pipeline.py (full flow)
    n_runs: int = 0
    n_outputs: int = 0                                   # outputs.jsonl records written (should track n_runs)
    log_inconsistencies: int = 0                         # times the outputs record failed to write after run_log
    llm_topology: dict | None = None                     # per-role LLM invocation config (configure_run)


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
    def code_dir(self) -> Path:
        return self.out / "code"

    @property
    def run_log_path(self) -> Path:
        return self.out / "run_log.jsonl"

    @property
    def decisions_path(self) -> Path:
        return self.out / "decisions.jsonl"

    @property
    def outputs_path(self) -> Path:
        return self.out / "outputs.jsonl"

    @property
    def reasoning_log_path(self) -> Path:
        return self.out / "reasoning_log.md"

    def _ensure_dirs(self) -> None:
        for d in (self.out, self.checkpoints_dir, self.artifacts_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---------- construction ----------
    @classmethod
    def create(cls, out_dir: str | Path, *, input_path: str | None = None,
               session_id: str | None = None, exist_ok: bool = True) -> "Session":
        out = Path(out_dir).resolve()
        init_runtime()   # I-6: set NUMBA_CACHE_DIR + njit-cache patch on EVERY entry, including the
                         # resume branch below (which returns before the old placement) and replay.
        existing = out / cls.MANIFEST
        if existing.exists() and not exist_ok:
            raise FileExistsError(f"session already exists: {existing}")
        if existing.exists() and exist_ok:
            sess = cls.open(out)
            # I-12 guard: reject a SILENT input swap — same workdir, different input file. This is the
            # root cause of I-2 (a shard's run reused another's session and read the wrong checkpoint).
            if input_path:
                recorded = (sess.manifest.input or {}).get("fingerprint")
                incoming = _fingerprint(input_path)
                if recorded and incoming and recorded != incoming:
                    rec_path = (sess.manifest.input or {}).get("path")
                    raise InputMismatch(
                        f"session at {out} was created for input '{rec_path}' but a DIFFERENT input "
                        f"'{Path(input_path).resolve()}' was given (fingerprint mismatch). Use a "
                        f"separate --workdir per input — shards must not share a session.")
            return sess
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

    def create_child(self, name: str, *, seed_adata, stage: str, x_state: str | None = None,
                     params: dict | None = None, derived_from: dict | None = None) -> "Session":
        """Spawn a nested session under ``<self.out>/compartments/<name>/`` seeded with
        ``seed_adata`` as its checkpoint 0. Used by compartment_subset so each Tier-2 compartment
        gets its OWN directory (checkpoints/artifacts/run-log) instead of overwriting the parent's
        working adata. The child is a full, independently replayable Session; ``derived_from``
        records the parent pointer (session id, checkpoint, groupby, compartment) for provenance.
        merge_fine_annotations later globs ``<self.out>/compartments`` to reassemble the parent."""
        child_out = self.out / "compartments" / name
        child = Session.create(child_out, input_path=(self.manifest.input or {}).get("path"),
                               exist_ok=True)
        child.manifest.derived_from = dict(derived_from or {})
        child.set_adata(seed_adata)
        # first checkpoint CREATES this session's counts fingerprint (like ingest/load), so it must
        # not require a pre-existing one — the subset carries the parent's immutable counts layer.
        child.checkpoint(stage, adata=seed_adata, x_state=x_state or self.manifest.x_state,
                         params=params or {}, require_counts=False)
        child.save()
        return child

    @classmethod
    def open(cls, out_dir: str | Path) -> "Session":
        out = Path(out_dir).resolve()
        init_runtime()   # I-6: replay/direct-open must also get NUMBA_CACHE_DIR + njit-cache patch
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
        # Entry-point symbol normalization (I-11): CELLxGENE stores Ensembl-ID var_names with
        # symbols in a var column; left as-is, MT-/RPS prefix matching silently no-ops. Remap to
        # the data's own symbols here so every downstream path (load tool AND lazy auto-load before
        # qc) sees symbols. No-op when var_names are already symbols; skipped on backed reads.
        if backed is None:
            from scpilot.core import _species
            ev = _species.normalize_var_symbols(self._adata)
            if ev.get("remapped") or ev.get("reason") == "ensembl_but_no_symbol_column":
                self._adata.uns["scpilot_var_symbol_normalization"] = ev
        if not self.manifest.input:
            self.manifest.input = {"path": str(Path(src).resolve()), "fingerprint": _fingerprint(src)}
        self._refresh_counts_state()
        return self._adata

    @property
    def adata(self):
        """The cached working AnnData. Lazily load the latest checkpoint, else the
        session input — so step-by-step CLI runs (each a fresh process) resume from
        the on-disk state without an explicit `load` call."""
        if self._adata is None:
            cp = self.latest_checkpoint()
            if cp is not None:
                import anndata as ad
                self._adata = ad.read_h5ad(cp["path"])
                self._refresh_counts_state()
            elif self.manifest.input.get("path"):
                self.load_input()                     # first mutating step: load the input
            else:
                raise RuntimeError("no working AnnData, no checkpoint, and no session input")
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
                   params: dict | None = None, compression: str = "gzip",
                   enforce_invariants: bool = True,
                   require_counts: bool | None = None) -> Checkpoint:
        """Atomically write a post-stage .h5ad checkpoint and register it in the manifest.

        Invariants are enforced HERE (the single mutating-write boundary) before the
        h5ad is persisted, so every mutating tool — current or future — is guarded
        without each having to remember to call ``assert_invariants`` (plan B1). The
        ``require_counts`` default is "require iff a counts fingerprint is already
        established" — so the first checkpoint that *creates* counts (ingest/load) is
        not tripped. ``enforce_invariants=False`` is an escape hatch for rare cases.
        """
        a = adata if adata is not None else self.adata
        if x_state:
            self.manifest.x_state = x_state
        self.stamp_provenance(a, stage, params=params, x_state=x_state)
        self._ensure_dirs()
        if enforce_invariants:
            rc = require_counts if require_counts is not None \
                else (self.manifest.counts_fingerprint is not None)
            self.assert_invariants(a, require_counts=rc)
        idx = len(self.manifest.checkpoints)
        cid = f"{idx:02d}_{stage}"
        path = self.checkpoints_dir / f"{cid}.h5ad"
        with atomic_path(path) as tmp:
            a.write_h5ad(tmp, compression=compression)
        cp = Checkpoint(id=cid, stage=stage, path=str(path), created_at=_now(),
                        fingerprint=_fingerprint(path), x_state=self.manifest.x_state,
                        params=params or {})
        self.manifest.checkpoints.append(cp.__dict__)
        # NOTE: the FULL pipeline script + notebook are (re)generated in record_run() from the
        # run log — covering EVERY step incl. non-mutating ones (plan P1) — not here, where we
        # would only see mutating/checkpointing tools.
        self.manifest.stage = stage
        if self.manifest.counts_fingerprint is None:
            self.manifest.counts_fingerprint = counts_fingerprint(a)
        self._adata = a
        self.save()
        return cp

    def latest_checkpoint(self) -> dict | None:
        return self.manifest.checkpoints[-1] if self.manifest.checkpoints else None

    # ---------- per-step executed code (source snapshot + standalone repro) ----------
    def _snapshot_source(self) -> str:
        """Copy the scpilot package source into code/scpilot-<ver>-<hash>/ (dedup by hash).

        Returns the snapshot dir name (relative to code_dir) so repro scripts can pin it.
        """
        snap_name = f"scpilot-{__version__}-{scpilot_source_hash()}"
        dst = self.code_dir / snap_name
        if not dst.exists():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(_pkg_dir(), dst / "scpilot",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        return snap_name

    def _read_runs(self) -> list[dict]:
        """Successful tool runs IN ORDER from the append-only run log.

        Pipeline/notebook generation is driven by the run log (every step — mutating
        AND non-mutating like annotation_review/benchmark/report), NOT by checkpoints
        (mutating-only), so the generated repro artifacts cover the WHOLE analysis (plan P1).
        """
        p = self.run_log_path
        if not p.exists():
            return []
        runs: list[dict] = []
        bad = 0
        for lineno, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                # F7: do NOT silently drop a corrupt record — a truncated/garbled line means the
                # generated pipeline would be missing a step. Surface it loudly + count it as a
                # log inconsistency so `log_consistency()` / the report flag the gap.
                bad += 1
                warnings.warn(f"run_log.jsonl: skipping malformed JSON at line {lineno} of {p}")
                continue
            if rec.get("status") == "success":
                runs.append(rec)
        if bad:
            self.manifest.log_inconsistencies += bad
        return runs

    def _write_pipeline(self) -> str:
        """(Re)write code/pipeline.py — the FULL flow (every logged step) as one runnable
        script, with each tool's actual source inlined for inspection."""
        import inspect

        from scpilot import tools as _tools

        self.code_dir.mkdir(parents=True, exist_ok=True)
        snapshot_rel = self._snapshot_source()
        prov = build_provenance()
        prov_comment = "\n".join(
            f"  {k}: {prov.get(k)}" for k in ("scpilot_version", "source_hash", "git_commit", "timestamp"))
        inp = self.manifest.input.get("path", "")

        flow_lines, src_blocks, seen = [], [], set()
        seed = 0
        for i, r in enumerate(self._read_runs()):
            stage, params = r.get("tool"), r.get("params", {}) or {}
            if r.get("seed") is not None:
                seed = int(r["seed"])
            elif "seed" in params:
                seed = int(params["seed"])
            flow_lines.append(f'    tools.run("{stage}", sess, **{params!r})  # {i:02d}_{stage}')
            if stage not in seen:
                seen.add(stage)
                try:
                    src = inspect.getsource(_tools.get(stage).fn)
                except Exception:  # noqa: BLE001
                    src = f"# (source unavailable for {stage})"
                commented = "\n".join("# " + ln for ln in src.splitlines())
                src_blocks.append(f"# --- {stage} " + "-" * (60 - len(stage)) + "\n" + commented)

        text = _PIPELINE_TEMPLATE.format(
            prov=prov_comment, input_lit=repr(inp), input_comment=_one_line(inp),
            snapshot_rel=snapshot_rel, seed=seed,
            flow="\n".join(flow_lines) if flow_lines else "    pass",
            sources="\n\n".join(src_blocks),
        )
        path = self.code_dir / "pipeline.py"
        with atomic_path(path) as tmp:
            tmp.write_text(text)
        return str(path)

    def _write_notebook(self) -> str:
        """(Re)write code/pipeline_notebook.py — a jupytext cell-delimited notebook that
        runs CELL BY CELL in Jupyter, printing each step's summary and rendering its plot
        inline. MCP-free (pinned scpilot package only). The human-facing visual repro."""
        self.code_dir.mkdir(parents=True, exist_ok=True)
        snapshot_rel = self._snapshot_source()
        prov = build_provenance()
        prov_md = "\n".join(
            f"# - {k}: {prov.get(k)}" for k in ("scpilot_version", "source_hash", "git_commit", "timestamp"))
        inp = self.manifest.input.get("path", "")
        runs = self._read_runs()
        # rationale (the WHY) per step from outputs.jsonl, matched by recipe_hash so retries or a
        # run_log↔outputs count mismatch can't misalign it (index zipping would be fragile).
        reason_by_hash: dict = {}
        if self.outputs_path.exists():
            for line in self.outputs_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:  # noqa: BLE001 — skip a malformed outputs line, keep generating
                    continue
                if o.get("reasoning"):
                    reason_by_hash[o.get("recipe_hash")] = o.get("reasoning")
        cells = []
        for i, r in enumerate(runs):
            stage, params = r.get("tool"), r.get("params", {}) or {}
            step_seed = r.get("seed")
            if step_seed is None:
                step_seed = int(params["seed"]) if "seed" in params else 0
            # each cell re-pins ITS recorded seed → cell-by-cell reproduction (not the
            # last step's seed for the whole notebook)
            cells.append(_notebook_cell(f"{i:02d}_{stage}", stage, params, int(step_seed),
                                        reasoning=reason_by_hash.get(r.get("recipe_hash"))))
        # top-of-notebook seed = first step's seed (initial/load state); each cell reseeds itself
        seed = int(runs[0].get("seed") or 0) if runs else 0
        text = _NOTEBOOK_TEMPLATE.format(
            prov_md=prov_md, input_lit=repr(inp), input_comment=_one_line(inp),
            snapshot_rel=snapshot_rel, seed=seed,
            cells="".join(cells) if cells else "")
        path = self.code_dir / "pipeline_notebook.py"
        with atomic_path(path) as tmp:
            tmp.write_text(text)
        return str(path)

    def _write_step_scripts(self) -> list[str]:
        """(Re)write per-step numbered standalone scripts ``code/NN_<stage>.py`` — the tutorial form
        of one .py file per step (run IN ORDER), complementing the single cell-by-cell notebook. Each
        re-pins its recorded seed, opens the repro session, runs ONE tool with explicit params, and
        prints status/summary/plot-paths. Driven by the run log (every step), like the notebook."""
        self.code_dir.mkdir(parents=True, exist_ok=True)
        snapshot_rel = self._snapshot_source()
        inp = self.manifest.input.get("path", "")
        runs = self._read_runs()
        reason_by_hash: dict = {}
        if self.outputs_path.exists():
            for line in self.outputs_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if o.get("reasoning"):
                    reason_by_hash[o.get("recipe_hash")] = o.get("reasoning")
        written: list[str] = []
        prev_cid_stage: str | None = None       # previous step's h5ad basename (chain input)
        for i, r in enumerate(runs):
            stage, params = r.get("tool"), r.get("params", {}) or {}
            step_seed = r.get("seed")
            if step_seed is None:
                step_seed = int(params["seed"]) if "seed" in params else 0
            cid = f"{i:02d}"
            # chain input: step 0 reads the raw session input; later steps read the prior step's h5ad
            in_expr = repr(inp) if prev_cid_stage is None else f'_DATA / "{prev_cid_stage}.h5ad"'
            text = _step_script(i, cid, stage, params, int(step_seed), snapshot_rel, in_expr,
                                reasoning=reason_by_hash.get(r.get("recipe_hash")), is_first=(i == 0))
            path = self.code_dir / f"{cid}_{stage}.py"
            prev_cid_stage = f"{cid}_{stage}"
            with atomic_path(path) as tmp:
                tmp.write_text(text)
            written.append(str(path))
        return written

    # ---------- append-only logs ----------
    def log_run(self, record: dict) -> None:
        """Append one tool-run record to run_log.jsonl (full schema frozen in A7)."""
        rec = {"ts": _now(), "session_id": self.manifest.session_id, **record}
        self._append_jsonl(self.run_log_path, rec)
        self.manifest.n_runs += 1
        self.save()

    @staticmethod
    def _artifact_records(artifacts) -> list[dict]:
        """Serialize result artifacts to dicts with a sha256 (integrity/audit) in meta."""
        out: list[dict] = []
        for a in (artifacts or []):
            if isinstance(a, dict):
                d = {"path": a.get("path"), "kind": a.get("kind", "other"),
                     "description": a.get("description", ""), "meta": dict(a.get("meta") or {})}
            else:
                d = {"path": getattr(a, "path", None), "kind": getattr(a, "kind", "other"),
                     "description": getattr(a, "description", ""), "meta": dict(getattr(a, "meta", {}) or {})}
            if d["path"]:
                sha = _artifact_sha(d["path"])
                if sha:
                    d["meta"]["sha256"] = sha
            out.append(d)
        return out

    def record_run(self, result, *, params: dict | None = None, seed: int | None = None,
                   input_checkpoint: str | None = None, lib_versions: dict | None = None,
                   stage: str | None = None, reasoning: str | None = None,
                   compute_recipe_hash: bool = True) -> str | None:
        """Build a FULLY-populated RunLogRecord from a ToolResult and append it, then bind
        the step's OUTPUTS + reasoning to ``outputs.jsonl`` and regenerate the repro pipeline.

        Returns the computed ``recipe_hash`` (or ``None``) so a caller can attach the SAME
        join key to a related DecisionEvent (Improvement ①) — the RunLogRecord/OutputRecord
        for this step already carry it.

        The single run-logging chokepoint shared by all four drivers (CLI ``step`` +
        ``run``-report, the MCP handler, the mode-2 agent) so their records can no
        longer drift (plan C1). Always fills ``seed``, ``lib_versions`` and ``recipe_hash``
        (plan A1/A2). Additionally writes one ``OutputRecord`` per step binding
        ``[step → params → artifacts(+sha256) → reasoning → provenance]`` (the artifacts/
        reasoning harness) and rewrites ``code/pipeline.py`` + the notebook from the run log
        so EVERY step (incl. non-mutating + report) is covered (plan P1).
        """
        from scpilot import repro
        from scpilot import schemas as S
        from scpilot.vendor.harness import _env_versions

        libs = lib_versions if lib_versions is not None else _env_versions()
        rh = None
        if compute_recipe_hash:
            try:
                fp = repro.dataset_fingerprint(self._adata) if self._adata is not None else None
                rh = repro.recipe_hash(params=params or {}, lib_versions=libs,
                                       input_checkpoint_id=input_checkpoint, fingerprint=fp)
            except Exception:  # noqa: BLE001 — a hashing hiccup must never break logging
                rh = None
        self.log_run(S.RunLogRecord(
            tool=result.tool, status=result.status, stage=stage or result.tool,
            params=params or {}, summary=result.summary, seed=seed,
            input_checkpoint=input_checkpoint, output_checkpoint=result.checkpoint,
            determinism_grade=result.determinism_grade, recipe_hash=rh,
            lib_versions=libs, duration_s=result.duration_s, error_code=result.error_code,
        ).to_dict())

        # outputs index: bind this step's artifacts + reasoning + provenance, COUPLED to the
        # run log by the run index n. A failure here must not break the result, but it is NOT
        # swallowed silently — it is counted (manifest.log_inconsistencies) and logged, so a
        # run_log↔outputs divergence is DETECTABLE via log_consistency() rather than hidden (C-2).
        try:
            out_rec = {"ts": _now(), "session_id": self.manifest.session_id,
                       **S.OutputRecord(
                           tool=result.tool, status=result.status, stage=stage or result.tool,
                           n=self.manifest.n_runs, recipe_hash=rh, seed=seed, params=params or {},
                           summary=result.summary, artifacts=self._artifact_records(result.artifacts),
                           reasoning=reasoning, warnings=list(result.warnings or []),
                       ).to_dict()}
            self._append_jsonl(self.outputs_path, out_rec)
            self.manifest.n_outputs += 1
        except Exception as exc:  # noqa: BLE001 — surface (count + log), never silently swallow
            self.manifest.log_inconsistencies += 1
            import logging
            logging.getLogger("scpilot.session").error(
                "outputs.jsonl write FAILED for '%s' — run_log↔outputs diverged (n_runs=%d): %s",
                result.tool, self.manifest.n_runs, exc)
        try:
            self.save()                      # persist the n_outputs / inconsistency counters
        except Exception:  # noqa: BLE001
            pass

        # regenerate the FULL repro pipeline + notebook from the run log (covers every step)
        try:
            self.manifest.pipeline_script = self._write_pipeline()
            self._write_notebook()
            self._write_step_scripts()      # per-step numbered tutorial scripts (code/NN_<stage>.py)
            self.save()
        except Exception:  # noqa: BLE001 — repro-artifact emission must never break the result
            pass
        return rh

    def record_tool_run(self, result, *, params: dict | None = None, seed: int | None = None,
                        input_checkpoint: str | None = None, lib_versions: dict | None = None,
                        reasoning: str | None = None, attach_plots: bool = True) -> str | None:
        """Full per-step record for the human-facing drivers (CLI ``step`` + MCP handler):
        attach a stage-appropriate auto-plot, append the run-log record, and write the
        Markdown reasoning entry — all in one place so ``step`` and MCP stay identical.

        All three orchestrated drivers (CLI ``step``, MCP handler, mode-2 agent) use this so
        auto-plots, the run-log + outputs records, and the reasoning narrative are produced
        identically in every mode. ``reasoning`` (the WHY) is bound to the OutputRecord AND
        written to the Markdown narrative.
        """
        if attach_plots and result.status == "success":
            try:
                from scpilot.core.autoplot import auto_plots
                extra = auto_plots(self, result.tool, result.summary)
                if extra:
                    result.artifacts = list(result.artifacts or []) + extra
            except Exception:  # noqa: BLE001 — a missing plot must never break the step
                pass
        # record_run binds artifacts(+sha) + reasoning to outputs.jsonl and regenerates the pipeline
        rh = self.record_run(result, params=params, seed=seed, input_checkpoint=input_checkpoint,
                             lib_versions=lib_versions, reasoning=reasoning)
        try:
            plot_paths = [a.path for a in (result.artifacts or [])
                          if getattr(a, "kind", None) == "png"]
            self.log_reasoning(tool=result.tool, params=params, summary=result.summary,
                               reasoning=reasoning, status=result.status,
                               checkpoint=result.checkpoint, plots=plot_paths)
        except Exception:  # noqa: BLE001 — logging must never break the result
            pass
        return rh   # Improvement ①: the recipe_hash join key for a related DecisionEvent

    def log_consistency(self) -> dict:
        """run_log ↔ outputs.jsonl coupling check (C-2): every logged run should have one
        outputs record. Returns counts + a ``consistent`` flag; a mismatch means an outputs
        write failed after the run-log append (recorded in ``log_inconsistencies``).

        I-14 also surfaces a PROVENANCE BYPASS: the "25 checkpoints vs 2 runs" case, where
        ``checkpoint()`` was called by ad-hoc/direct code that skipped the ``record_run`` chokepoint
        (so those steps are not replayable). A mutating tool run through the harness produces BOTH a
        checkpoint and a run, so far more checkpoints than runs ⇒ bypass. Non-mutating tools add runs
        without checkpoints, so a small/negative gap is normal; a large POSITIVE gap is the signal.
        """
        m = self.manifest
        n_ckpt = len(m.checkpoints or [])
        consistent = m.log_inconsistencies == 0 and m.n_outputs == m.n_runs
        gap = n_ckpt - m.n_runs
        return {"n_runs": m.n_runs, "n_outputs": m.n_outputs, "n_checkpoints": n_ckpt,
                "log_inconsistencies": m.log_inconsistencies, "consistent": bool(consistent),
                "checkpoint_run_gap": gap, "checkpoint_bypass_suspected": gap > 1}

    def artifact_path(self, name: str) -> Path:
        """A non-colliding path under ``artifacts_dir`` for an output file (P1-2).

        Returns ``artifacts_dir/name`` when free; if a prior run already wrote that name,
        inserts the current run index so the earlier evidence file is NOT silently overwritten
        (the returned path is what the tool writes + reports, so outputs.jsonl/report point at it).

        F8: this is a check-then-write allocation, safe under scpilot's single-writer session model
        (one process owns a session at a time — CLI step / MCP handler / agent each run serially).
        It is NOT safe for concurrent writers sharing one session directory; that mode is
        unsupported. The run-index suffix already disambiguates re-runs within a session.
        """
        self._ensure_dirs()
        p = self.artifacts_dir / name
        if not p.exists():
            return p
        stem, dot, ext = name.rpartition(".")
        base = stem if dot else name
        suffix = f".{ext}" if dot else ""
        return self.artifacts_dir / f"{base}.{self.manifest.n_runs:02d}{suffix}"

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

    def log_reasoning(self, *, tool: str, params: dict | None = None, summary: dict | None = None,
                      reasoning: str | None = None, status: str = "success",
                      checkpoint: str | None = None, plots: list | None = None) -> None:
        """Append a human-readable Markdown entry to reasoning_log.md — one section per
        tool run so the analysis can be read top-to-bottom as a narrative (plan: result
        plot + reasoning rule). ``reasoning`` is the caller/LLM's WHY (optional); the rest
        is auto-captured from the ToolResult so a log entry exists even without an LLM note.
        """
        n = self.manifest.n_runs
        lines = [f"\n## {n:02d} · {tool}  ({status})  {_now()}"]
        if reasoning:
            lines.append(f"\n**Reasoning:** {reasoning}")
        if params:
            kv = ", ".join(f"`{k}={v}`" for k, v in params.items() if k != "reasoning")
            if kv:
                lines.append(f"\n**Params:** {kv}")
        if summary:
            keys = ("n_cells", "n_genes", "n_clusters", "n_hvg", "n_samples_merged",
                    "label_distribution", "best", "ranking_by_total", "n_cells_after",
                    "scores", "out_key", "label_key")
            picked = {k: summary[k] for k in keys if k in summary}
            if picked:
                lines.append("\n**Result:** " + "; ".join(f"{k}={v}" for k, v in picked.items()))
        if plots:
            for p in plots:
                lines.append(f"\n![{tool}]({p})")
        self.reasoning_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.reasoning_log_path, "a") as fh:
            fh.write("\n".join(lines) + "\n")

    @staticmethod
    def _append_jsonl(path: Path, record: dict) -> None:
        """Append one record as a single line. F7: the whole line is written in ONE ``write()``
        under O_APPEND (POSIX serializes the append offset, so concurrent writers don't interleave),
        then flushed + fsync'd so a crash can't leave a half-written record that later breaks
        JSONL parsing / code generation."""
        import os as _os
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with open(path, "a") as fh:
            fh.write(line)
            fh.flush()
            _os.fsync(fh.fileno())

    def __repr__(self) -> str:
        return (f"Session(id={self.manifest.session_id}, out={self.out}, "
                f"stage={self.manifest.stage}, checkpoints={len(self.manifest.checkpoints)})")
