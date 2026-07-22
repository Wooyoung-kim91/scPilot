# =====================================================================
# VENDORED FROM scqc_pipeline @ source_hash debef308904633e1
#   source: /home/wykim/data/PDAC/scqc_pipeline/ (copied 2026-06-10)
# scpilot 베다링 정책: 독립 진화. import 경로·provenance 키·uns 키만
#   scpilot으로 적응했고 로직은 원본 유지. 재동기화 절차/원본 대비 diff는
#   scpilot/vendor/VENDORING.md 참조. scpilot 고유 코드는 여기 두지 말 것.
# =====================================================================
"""The uniform stage harness: runtime init, checkpointing, reporting, provenance.

Every stage runs through ``run_stage`` so behaviour, logging, freshness checks,
atomic writes and StageReport JSON are identical across the pipeline. The CLI and
the (minimal) LLM layer both drive stages the same way and read the same compact
JSON reports — no large h5ad is loaded just to make a decision.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from scpilot import __version__
from scpilot.vendor.config import PipelineConfig, SCHEMA_VERSION

# --------------------------------------------------------------------------- #
# Runtime init (ported from notebook cell 1 / scripts/*.py header convention)
# --------------------------------------------------------------------------- #
_RUNTIME_READY = False

# BLAS/OpenMP thread-count env vars we bound so a process pool cannot oversubscribe the box.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def bounded_thread_count() -> int:
    """A conservative per-pool BLAS/OpenMP thread bound: ``min(cpu_count, 8)``.

    Symmetric with ``cnv._resolve_n_jobs``' worker cap, so the worst case is bounded
    (≤8 workers × ≤8 threads) instead of the historical cpu_count × cpu_count runaway.
    """
    return min(os.cpu_count() or 1, 8)


def bound_thread_env(n: int | None = None) -> None:
    """``setdefault`` the BLAS/OpenMP thread-count env vars to a bounded value.

    Why: ``cnv_score`` → ``infercnvpy.tl.infercnv`` spawns a ``ProcessPoolExecutor``; with no
    thread cap each of N workers spawns ``cpu_count()`` BLAS/OpenMP threads (120 on this box),
    so N×cpu_count oversubscription pins the machine (the ~900-CPU-hour runaways). Bounding the
    thread count keeps every worker (and the main process) from oversubscribing.

    Ordering INVARIANT: this MUST run before numpy/BLAS is imported (BLAS reads these once, at
    import) AND before the ``forkserver`` daemon is warmed, because forkserver-spawned workers
    inherit the environment captured at warmup time — see ``scpilot/mcp_server.py:main()``. Only
    a var the user has NOT already set is touched (``setdefault``), so an explicit environment
    always wins (evidence-based / no clobbering).
    """
    val = str(n if n is not None else bounded_thread_count())
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, val)


def init_runtime() -> None:
    """Idempotent process setup: numba/matplotlib caches + numba njit patch.

    Numba-backed packages (scanpy/umap) can fail at import if their cache dir is
    unusable in a detached session, so we pin caches and drop ``cache=True``.
    """
    global _RUNTIME_READY
    if _RUNTIME_READY:
        return
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
    # Bound BLAS/OpenMP threads for CLI and any non-MCP entrypoint too. In the MCP server this has
    # already run earlier (main(), before forkserver warmup); here it is an idempotent setdefault.
    bound_thread_env()
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import numba

    _orig_njit = numba.njit

    def _njit_no_cache(*args, **kwargs):
        kwargs.pop("cache", None)
        return _orig_njit(*args, **kwargs)

    numba.njit = _njit_no_cache
    _RUNTIME_READY = True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Provenance & source snapshot
# --------------------------------------------------------------------------- #
def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def source_hash() -> str:
    """sha256 over all package .py source (stable, order-independent)."""
    h = hashlib.sha256()
    for p in sorted(_package_dir().rglob("*.py")):
        h.update(p.relative_to(_package_dir()).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(_package_dir()), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _env_versions() -> dict:
    from importlib.metadata import version, PackageNotFoundError

    out = {"python": sys.version.split()[0]}
    for mod in ("scanpy", "anndata", "numpy", "scipy", "seaborn", "matplotlib"):
        try:
            out[mod] = version(mod)
        except PackageNotFoundError:
            out[mod] = None
    return out


def build_provenance() -> dict:
    return {
        "scpilot_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "source_hash": source_hash(),
        "git_commit": _git_commit(),
        "command": ["python", "-m", "scpilot"] + sys.argv[1:],
        "env": _env_versions(),
        "timestamp": _now(),
    }


def snapshot_source(cfg: PipelineConfig) -> Path:
    """Copy package source into <out>/code/scpilot-<ver>-<hash>/ (dedup by hash)."""
    dst = cfg.code_dir / f"scpilot-{__version__}-{source_hash()}"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            _package_dir(), dst / "scpilot",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return dst


_REPRO_TEMPLATE = '''#!/usr/bin/env python
"""Standalone reproduction script for stage `{stage}`.

This file + the pinned source snapshot regenerate the exact artifact(s) below.
Provenance (how the original was produced):
{prov_comment}

Outputs:
{out_comment}
"""
import sys
from pathlib import Path

# Pin imports to the exact source snapshot used for the original artifact.
sys.path.insert(0, str(Path(__file__).resolve().parent / "{snapshot_rel}"))

# (scpilot 레지스트리는 A7에서 재구성 — 원본 scqc는 여기서 stages 등록)
from scpilot.vendor.config import PipelineConfig
from scpilot.vendor.harness import run_stage

# Resolved config inlined for full reproducibility (no external profile needed).
CONFIG = {config_literal}

if __name__ == "__main__":
    cfg = PipelineConfig(**CONFIG)
    rep = run_stage(cfg, "{stage}", force=True)
    print(rep.status, rep.outputs)
'''


def write_repro(cfg: PipelineConfig, stage: str, provenance: dict,
                outputs: list[str]) -> Path:
    """Emit <out>/code/<stage>.repro.py with inlined config pointing at the snapshot."""
    snap = snapshot_source(cfg)
    snapshot_rel = os.path.relpath(snap, cfg.code_dir)
    prov_comment = "\n".join(
        f"  {k}: {provenance.get(k)}"
        for k in ("scpilot_version", "source_hash", "git_commit", "command", "timestamp")
    )
    out_comment = "\n".join(f"  - {o}" for o in outputs)
    # Python literal (not JSON) so the inlined CONFIG is valid Python (True/False/None).
    import pprint
    config_literal = pprint.pformat(cfg.to_public_dict(), indent=4, width=100, sort_dicts=True)
    text = _REPRO_TEMPLATE.format(
        stage=stage,
        prov_comment=prov_comment,
        out_comment=out_comment,
        snapshot_rel=snapshot_rel,
        config_literal=config_literal,
    )
    cfg.code_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.code_dir / f"{stage}.repro.py"
    with atomic_path(path) as tmp:
        tmp.write_text(text)
    return path


# --------------------------------------------------------------------------- #
# Atomic write helper
# --------------------------------------------------------------------------- #
@contextmanager
def atomic_path(final: Path):
    """Yield a temp path; on success os.replace() into ``final``. Cleans up on error.

    Guarantees a reader never sees a half-written artifact, and a crash leaves no
    partial file at the final path (so is_fresh won't mistake it for complete).
    """
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(final.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        yield tmp
        if not tmp.exists():
            raise RuntimeError(f"atomic_path: writer did not create {tmp}")
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()


def _fingerprint(path: Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    st = p.stat()
    return {"mtime": round(st.st_mtime, 3), "size": st.st_size}


# --------------------------------------------------------------------------- #
# Stage report
# --------------------------------------------------------------------------- #
@dataclass
class StageReport:
    stage: str
    status: str = "pending"           # ok | skipped | failed | dirty
    params: dict = field(default_factory=dict)
    config_hash: str = ""
    schema_version: str = SCHEMA_VERSION
    input_fingerprints: dict = field(default_factory=dict)
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    error: str | None = None
    provenance: dict = field(default_factory=dict)
    started_at: str = ""
    duration_s: float = 0.0

    def write(self, cfg: PipelineConfig) -> Path:
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        path = cfg.reports_dir / f"{self.stage}.report.json"
        with atomic_path(path) as tmp:
            tmp.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path


def load_report(cfg: PipelineConfig, stage: str) -> dict | None:
    path = cfg.reports_dir / f"{stage}.report.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Stage registration & context
# --------------------------------------------------------------------------- #
@dataclass
class StageContext:
    cfg: PipelineConfig
    logger: logging.Logger
    report: StageReport


@dataclass
class StageSpec:
    name: str
    fn: Callable                       # fn(ctx) -> None ; fills ctx.report
    inputs: Callable                   # fn(cfg) -> list[Path]
    outputs: Callable                  # fn(cfg) -> list[Path]
    depends: list = field(default_factory=list)


STAGES: dict[str, StageSpec] = {}


def register(spec: StageSpec) -> None:
    STAGES[spec.name] = spec


def _logger(cfg: PipelineConfig, stage: str) -> logging.Logger:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"scqc.{stage}")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s")
    fh = logging.FileHandler(cfg.reports_dir / f"{stage}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    lg.addHandler(fh)
    lg.addHandler(sh)
    lg.propagate = False
    return lg


def is_fresh(cfg: PipelineConfig, stage: str) -> tuple[bool, str]:
    """A stage is fresh iff outputs all exist (atomically) AND fingerprints match.

    Returns (fresh, reason). ``reason`` explains why it is dirty.
    """
    spec = STAGES[stage]
    outs = spec.outputs(cfg)
    missing = [str(p) for p in outs if not Path(p).exists()]
    if missing:
        return False, f"missing outputs: {missing[:3]}"
    # any leftover .tmp means a previous run was interrupted mid-write
    for p in outs:
        if Path(str(p) + ".tmp").exists():
            return False, f"interrupted (.tmp present): {p}"
    prev = load_report(cfg, stage)
    if prev is None:
        return False, "no prior report"
    if prev.get("status") not in ("ok", "skipped"):
        return False, f"prior status={prev.get('status')}"
    if prev.get("schema_version") != SCHEMA_VERSION:
        return False, "schema_version changed"
    if prev.get("config_hash") != cfg.stage_config_hash(stage):
        return False, "config changed"
    cur_fp = {str(p): _fingerprint(p) for p in spec.inputs(cfg)}
    if prev.get("input_fingerprints") != cur_fp:
        return False, "inputs changed"
    return True, "fresh"


def run_stage(cfg: PipelineConfig, stage: str, *, force: bool = False,
              dry_run: bool = False) -> StageReport:
    """Run one stage through the uniform contract. Returns its StageReport."""
    init_runtime()
    spec = STAGES[stage]
    lg = _logger(cfg, stage)
    rep = StageReport(stage=stage)
    rep.config_hash = cfg.stage_config_hash(stage)
    rep.inputs = [str(p) for p in spec.inputs(cfg)]
    rep.outputs = [str(p) for p in spec.outputs(cfg)]
    rep.input_fingerprints = {str(p): _fingerprint(p) for p in spec.inputs(cfg)}
    rep.started_at = _now()
    rep.provenance = build_provenance()

    fresh, reason = is_fresh(cfg, stage)
    if dry_run:
        rep.status = "skipped" if (fresh and not force) else "dirty"
        rep.metrics["plan_reason"] = reason if not force else "force"
        lg.info("DRY-RUN %s → %s (%s)", stage, rep.status, rep.metrics["plan_reason"])
        return rep

    if fresh and not force:
        rep.status = "skipped"
        rep.metrics["skip_reason"] = reason
        lg.info("SKIP %s (%s)", stage, reason)
        rep.write(cfg)
        return rep

    ctx = StageContext(cfg=cfg, logger=lg, report=rep)
    t0 = datetime.now(timezone.utc)
    try:
        lg.info("RUN %s (dirty: %s, force=%s)", stage, reason, force)
        spec.fn(ctx)
        rep.status = "ok"
        # Save the exact code that produced these artifacts (snapshot + repro).
        repro = write_repro(cfg, stage, rep.provenance, rep.outputs)
        rep.outputs.append(str(repro))
    except Exception as exc:  # noqa: BLE001 — capture into report
        rep.status = "failed"
        rep.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        lg.error("FAILED %s: %s", stage, exc)
    finally:
        rep.duration_s = (datetime.now(timezone.utc) - t0).total_seconds()
        # refresh fingerprints to the just-written outputs
        if rep.status == "ok":
            rep.input_fingerprints = {str(p): _fingerprint(p) for p in spec.inputs(cfg)}
        rep.write(cfg)
    return rep


def provenance_uns(cfg: PipelineConfig, stage: str, config_hash: str,
                   provenance: dict) -> dict:
    """Block to embed in adata.uns["scpilot"] so an h5ad is self-describing."""
    return {
        "stage": stage,
        "version": __version__,
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash,
        "command": provenance.get("command"),
        "repro": f"code/{stage}.repro.py",
        "src": f"code/scpilot-{__version__}-{source_hash()}/",
    }


class Pipeline:
    """Ordered execution of stages with declared file-level dependencies."""

    ORDER = ["metadata", "qc", "merge", "qc-plots", "visualize", "report", "subset"]

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def select(self, frm: str | None, to: str | None) -> list[str]:
        order = self.ORDER
        i = order.index(frm) if frm else 0
        j = order.index(to) + 1 if to else len(order)
        return order[i:j]

    def run(self, stages: list[str], *, force: bool = False,
            dry_run: bool = False) -> dict:
        results = {}
        dirty: set[str] = set()
        for st in stages:
            rep = run_stage(self.cfg, st, force=force, dry_run=dry_run)
            if dry_run:
                # propagate: if an upstream dependency will re-run, so will this stage
                spec = STAGES[st]
                if rep.status == "dirty" or any(d in dirty for d in spec.depends):
                    rep.status = "dirty"
                    dirty.add(st)
            results[st] = asdict(rep)
            if rep.status == "failed":
                break
        summary = {"stages": results, "generated_at": _now()}
        if not dry_run:
            self.cfg.reports_dir.mkdir(parents=True, exist_ok=True)
            (self.cfg.reports_dir / "pipeline.summary.json").write_text(
                json.dumps(summary, indent=2, default=str)
            )
        return summary
