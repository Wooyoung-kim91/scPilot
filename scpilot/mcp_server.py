"""FastMCP (stdio) server — scpilot plan A6/C2.

A6 spike: expose ONE read-only tool (``inspect_h5ad``) over stdio so we can verify
the transport works from both Claude Code and Codex CLI (tool discovery, a short
call, stderr/stdout hygiene). The full registry (all tools + job model) lands in C2.

Hard rules (stdio MCP):
- **stdout carries ONLY protocol JSON** — every log/warning goes to stderr.
- Run ``init_runtime()`` before importing/using numba-backed code (scanpy/umap)
  so a detached session's numba cache does not break imports.
"""

from __future__ import annotations

import itertools
import logging
import os
import sys
import threading
import time
import warnings
from pathlib import Path


# --------------------------------------------------------------------------- #
# Job model for long_running tools (plan C1) — the durable fix for "cnv_score drops the
# stdio connection". A long tool runs as ONE synchronous MCP call today, so the client's
# per-call timeout tears the connection down regardless of protocol pings. Instead we start
# the body, return the normal result inline if it finishes within a short grace window, else
# hand back {"status":"running","job_id":...} and let it finish in the background — polled via
# get_job_status / get_job_result / cancel_job. Jobs are process-local runtime state (NOT part
# of the reproducible recipe), so an incrementing job id is fine.
# --------------------------------------------------------------------------- #
def _grace_seconds() -> float:
    """Grace window (s) before a long tool is handed back as a background job.

    Fast/small inputs finish inside the window and return inline (nothing changes for them).
    Env-configurable via ``SCPILOT_MCP_JOB_GRACE_SECONDS`` (default 25s)."""
    try:
        return max(0.0, float(os.environ.get("SCPILOT_MCP_JOB_GRACE_SECONDS", "25")))
    except ValueError:
        return 25.0


def _max_retained_jobs() -> int:
    """Cap on retained FINISHED (succeeded|failed|cancelled) jobs before LRU eviction (Bug C).

    Every ``long_running`` call — even ones that finish inline within the grace window — creates a
    ``_Job`` holding the full result dict, and nothing ever evicted them, so a long-lived server
    leaked memory unboundedly. We keep only the most-recent N finished jobs (insertion-order
    eviction); RUNNING jobs are never counted or evicted. Job state is process-local (not part of
    the reproducible recipe), so eviction is safe. Env-configurable via
    ``SCPILOT_MCP_MAX_RETAINED_JOBS`` (default 64)."""
    try:
        return max(1, int(os.environ.get("SCPILOT_MCP_MAX_RETAINED_JOBS", "64")))
    except ValueError:
        return 64


class _Job:
    """One long-running tool invocation. State: running → succeeded | failed | cancelled."""

    def __init__(self, job_id: str, tool: str) -> None:
        self.job_id = job_id
        self.tool = tool
        self.state = "running"          # running | succeeded | failed | cancelled
        self.started = time.monotonic()
        self.finished: float | None = None
        self.result: dict | None = None  # the ToolResult dict (success OR structured error)
        self.cancel_requested = False
        self.done = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def elapsed_s(self) -> float:
        return round((self.finished or time.monotonic()) - self.started, 3)


class _JobRegistry:
    """In-process registry of long-running jobs, keyed by job_id (thread-safe)."""

    def __init__(self, max_retained: int | None = None) -> None:
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        # Bug C: bound retained FINISHED jobs (LRU/insertion-order eviction); never a RUNNING job.
        self._max_retained = max_retained if max_retained is not None else _max_retained_jobs()

    def create(self, tool: str) -> _Job:
        with self._lock:
            job = _Job(f"job-{next(self._counter)}", tool)
            self._jobs[job.job_id] = job
            self._evict_locked()
            return job

    def _evict_locked(self) -> None:
        """Evict oldest FINISHED jobs beyond the retention cap (call under ``self._lock``).

        A job is 'finished' iff its ``done`` event is set (succeeded|failed|cancelled) — a RUNNING
        job never has ``done`` set, so it is never evicted (its body is still using its result slot
        and cancel_job must still find it). ``dict`` preserves insertion order, so slicing the
        finished ids from the front drops the least-recently-created ones first."""
        finished = [jid for jid, j in self._jobs.items() if j.done.is_set()]
        excess = len(finished) - self._max_retained
        for jid in finished[:excess] if excess > 0 else ():
            del self._jobs[jid]

    def get(self, job_id: str) -> _Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[_Job]:
        with self._lock:
            return list(self._jobs.values())


def default_workdir_for_input(input_path: str) -> str:
    """Per-input session dir. Thin re-export of the single source in ``scpilot.session`` (I-12) — kept
    at module level (public API) but imported lazily so importing this module does not pull the
    scientific stack before ``init_runtime()`` runs."""
    from scpilot.session import default_workdir_for_input as _f
    return _f(input_path)


def _select_specs(specs, lg):
    """F5: optionally restrict which registry tools the MCP server exposes.

    The server exposes EVERY registered tool by default (the primary integration is a trusted
    local host such as Claude Code). For tighter deployments, two env vars gate the surface by
    tool name (without the ``_tool`` suffix):
      - ``SCPILOT_MCP_ENABLE_TOOLS`` — comma-separated allowlist (only these are exposed).
      - ``SCPILOT_MCP_DISABLE_TOOLS`` — comma-separated denylist (these are removed).
    Allowlist is applied first, then denylist. Unknown names are ignored (logged)."""
    def _names(var):
        return {n.strip() for n in os.environ.get(var, "").split(",") if n.strip()}

    enable, disable = _names("SCPILOT_MCP_ENABLE_TOOLS"), _names("SCPILOT_MCP_DISABLE_TOOLS")
    selected = list(specs)
    if enable:
        selected = [s for s in selected if s.name in enable]
        lg.info("MCP allowlist active (SCPILOT_MCP_ENABLE_TOOLS): %s", ", ".join(sorted(enable)))
    if disable:
        selected = [s for s in selected if s.name not in disable]
        lg.info("MCP denylist active (SCPILOT_MCP_DISABLE_TOOLS): %s", ", ".join(sorted(disable)))
    return selected


def _configure_io() -> logging.Logger:
    """Keep stdout clean for the protocol; route logs + warnings to stderr."""
    # numba/matplotlib caches + njit patch (detached-session safety)
    from scpilot.vendor.harness import init_runtime
    init_runtime()

    # Python warnings -> stderr (never stdout)
    logging.captureWarnings(True)
    warnings.simplefilter("default")

    lg = logging.getLogger("scpilot.mcp")
    lg.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [scpilot.mcp] %(message)s"))
    lg.handlers.clear()
    lg.addHandler(handler)
    lg.propagate = False
    return lg


def build_server():
    """Create the FastMCP server, exposing EVERY tool in the registry dynamically.

    Tools register themselves in ``scpilot.tools.REGISTRY`` at import time; this
    function wraps each as an MCP tool that opens/creates a Session and dispatches
    through the registry. So adding a B-tool (``@register(...)``) auto-exposes it
    over MCP — no edits here. Session working dir comes from a ``workdir`` arg
    (defaults next to the input) so MCP callers get the same on-disk session model.
    """
    import anyio
    from mcp.server.fastmcp import FastMCP

    from scpilot import __version__, tools
    from scpilot.llm import prompts
    from scpilot.session import Session

    specs = tools.all_specs()  # triggers registration of all core tool modules

    lg = _configure_io()
    specs = _select_specs(specs, lg)   # F5: optional env-gated allow/deny filter

    # Why offload: FastMCP calls a *sync* tool handler directly on the asyncio event loop
    # (mcp/server/fastmcp func_metadata: `return fn(...)` when the handler is not a coroutine).
    # So a long tool (ingest 100k+ cells / train_scvi / benchmark / cnv) blocks the loop for
    # minutes, the server cannot answer the client's protocol PINGs, and the client tears the
    # stdio connection down mid-run ("works for a while then disconnects"). We instead run the
    # blocking body in a worker thread so the loop stays free to answer pings for the full
    # duration. `_tool_limiter` (capacity 1) keeps the NON-long inline path one-at-a-time, and
    # `_tool_lock` (acquired inside EVERY tool body — inline AND background job) is the single
    # cross-path serialization guarantee: the body pins the *process-global* RNG (set_global_seed)
    # and mutates scanpy's global settings, so two concurrent tools would corrupt each other's
    # determinism — the reproducibility invariant requires one-at-a-time (jobs included).
    _tool_limiter = anyio.CapacityLimiter(1)
    _tool_lock = threading.Lock()
    _registry = _JobRegistry()
    # Model-agnostic guidance: ship the orchestration brief in the MCP `initialize` handshake so
    # EVERY client's LLM (Claude Code, Codex, a local model) — not just Claude — sees how to drive
    # the pipeline. The full canonical flow is fetchable via prompt/resource/tool below.
    mcp = FastMCP("scpilot", instructions=prompts.MCP_INSTRUCTIONS)

    def _run_blocking(name: str, input: str, workdir: str, params: dict | None, seed: int,
                      job: "_Job | None" = None) -> dict:
        """The synchronous tool body — run in a worker thread (see offload comment above).

        Acquires `_tool_lock` so it never runs concurrently with any other tool body (inline or
        a background job), preserving the process-global RNG + scanpy-settings invariant. Always
        returns a dict (structured error on failure), never throws — MCP requires that.

        Bug B (pre-execution cancel): for a background job that was cancelled WHILE queued on
        `_tool_lock`, the body must not run at all — no tool call, no `record_tool_run`, so no
        checkpoint and no run_log entry are committed for a "cancelled" job. The check is the FIRST
        thing done after acquiring `_tool_lock`, and reads `job.cancel_requested` under `job.lock`,
        so it cannot race with `cancel_job` (which sets that flag under the same `job.lock`): either
        cancel wins here (body skipped) or it arrives after the body has started and is handled by
        the existing post-run cancel path in `_worker`. The inline (non-long) path passes job=None
        and is unaffected."""
        from scpilot import schemas as S
        from scpilot.repro import set_global_seed
        with _tool_lock:
            if job is not None:
                with job.lock:
                    if job.cancel_requested:
                        lg.info("job %s cancelled before execution (tool=%s); body NOT run "
                                "(no checkpoint, no run_log)", job.job_id, name)
                        return {"status": "cancelled", "job_id": job.job_id, "tool": name,
                                "message": ("job cancelled before execution; the tool body was "
                                            "not run (no checkpoint, no run_log entry).")}
            lg.info("tool=%s input=%s seed=%s", name, input, seed)
            params = dict(params or {})
            # optional LLM narration for reasoning_log.md (not a tool param)
            reasoning = params.pop("reasoning", None)
            try:
                # pin RNGs per call so mode-1 (MCP) is reproducible like the CLI (plan A1).
                set_global_seed(seed)
                wd = workdir or default_workdir_for_input(input)
                session = Session.create(wd, input_path=input)
                # Bug G: cache this step's recipe_hash BEFORE the tool runs so any in-tool
                # DecisionEvent shares the SAME join key its run-log/outputs record will carry.
                session.begin_step(params=params, seed=seed)
                result = tools.run(name, session, **params)
                # result-plot rule + run_log.jsonl + reasoning_log.md via the shared chokepoint
                # (plan C1): IDENTICAL to the CLI `step` path, so mode-1 runs are fully
                # replayable and the records cannot drift between drivers. Runs EXACTLY ONCE per
                # finished job (here, in the worker) whether it returns inline or via get_job_result.
                try:
                    session.record_tool_run(result, params=params, seed=seed,
                                            reasoning=reasoning)
                except Exception:  # noqa: BLE001 — logging must never break the tool result
                    lg.exception("run/reasoning logging failed for %s", name)
                return result.to_dict()
            except Exception as exc:  # noqa: BLE001 — MCP must return a structured error, not throw
                lg.exception("tool %s failed", name)
                return S.error(name, "internal", f"{type(exc).__name__}: {exc}").to_dict()

    def _make_handler(name: str, long_running: bool):
        if not long_running:
            async def handler(input: str, workdir: str = "", params: dict | None = None,
                              seed: int = 0) -> dict:
                # async handler => FastMCP awaits it, keeping the event loop responsive to protocol
                # pings while the blocking body runs off-loop in a worker thread (serialized by
                # _tool_limiter, then _tool_lock). Signature unchanged so the schema is identical.
                return await anyio.to_thread.run_sync(
                    _run_blocking, name, input, workdir, params, seed, limiter=_tool_limiter)
            return handler

        async def handler(input: str, workdir: str = "", params: dict | None = None,
                          seed: int = 0) -> dict:
            # long_running: start the body in a background job thread (it serializes on _tool_lock
            # like every other tool), then wait up to the grace window for it to finish. If it does,
            # return the normal result dict inline — identical to the old behavior for fast/small
            # inputs. If it exceeds the window, return promptly with a job_id and let it run on; the
            # client polls get_job_status / get_job_result / cancel_job instead of holding one
            # multi-minute MCP call open (which the per-call timeout would otherwise tear down).
            job = _registry.create(name)

            def _worker() -> None:
                try:
                    res = _run_blocking(name, input, workdir, params, seed, job)
                except BaseException as exc:  # noqa: BLE001 — a thread must not die silently
                    from scpilot import schemas as S
                    res = S.error(name, "internal", f"{type(exc).__name__}: {exc}").to_dict()
                with job.lock:
                    job.finished = time.monotonic()
                    job.result = res
                    if job.cancel_requested:
                        job.state = "cancelled"   # result computed but the caller asked to cancel
                    elif res.get("status") == "error":
                        job.state = "failed"
                    else:
                        job.state = "succeeded"
                job.done.set()

            job.thread = threading.Thread(target=_worker, name=f"scpilot-{job.job_id}", daemon=True)
            job.thread.start()
            # wait for completion OFF the event loop so protocol pings still flow during the window
            finished = await anyio.to_thread.run_sync(job.done.wait, _grace_seconds())
            if finished:
                return job.result
            lg.info("tool=%s exceeded grace window -> background job %s", name, job.job_id)
            return {"status": "running", "job_id": job.job_id, "tool": name,
                    "message": (f"'{name}' is long-running and exceeded the grace window; it "
                                "continues in the background. Poll get_job_status / fetch "
                                "get_job_result with this job_id, or cancel_job to stop it.")}
        return handler

    for spec in specs:
        handler = _make_handler(spec.name, spec.long_running)
        handler.__name__ = f"{spec.name}_tool"
        handler.__doc__ = (
            f"{spec.description}\n\n"
            "Args:\n"
            "    input: absolute path to a .h5ad on the server filesystem.\n"
            "    workdir: optional session directory (defaults next to input).\n"
            "    params: optional tool parameters (dict). May include a 'reasoning' "
            "string (the WHY for this step) — it is stripped from tool params and "
            "recorded in reasoning_log.md, not passed to the tool.\n"
            "    seed: global RNG seed pinned before the call (default 0) — recorded "
            "in run_log.jsonl so the run is reproducible/replayable."
        )
        mcp.tool(name=f"{spec.name}_tool")(handler)

    @mcp.tool()
    def scpilot_version() -> dict:
        """Return the scpilot version (cheap connectivity check)."""
        return {"scpilot_version": __version__}

    # --- job/poll interface for long_running tools (plan C1) — see the _Job/_JobRegistry note ---
    def _unknown_job(job_id: str) -> dict:
        return {"status": "error", "error_code": "missing_input",
                "error": f"unknown job_id '{job_id}'", "job_id": job_id}

    @mcp.tool()
    def get_job_status(job_id: str) -> dict:
        """Status of a background job started when a long_running tool exceeded the grace window.

        Returns state (running | succeeded | failed | cancelled) + elapsed seconds. Fetch the
        actual tool result with get_job_result once state != running."""
        job = _registry.get(job_id)
        if job is None:
            return _unknown_job(job_id)
        with job.lock:
            return {"status": "ok", "job_id": job.job_id, "tool": job.tool, "state": job.state,
                    "elapsed_s": job.elapsed_s(), "done": job.done.is_set(),
                    "cancel_requested": job.cancel_requested}

    @mcp.tool()
    def get_job_result(job_id: str) -> dict:
        """Fetch a finished job's tool result dict (identical to the inline return had it been
        fast). While still running, returns {"status":"running", ...}; for an unknown id, a
        structured missing_input error; for a cancelled job, a cancelled notice."""
        job = _registry.get(job_id)
        if job is None:
            return _unknown_job(job_id)
        with job.lock:
            if job.state == "cancelled":   # takes precedence even if the body is still finishing
                return {"status": "cancelled", "job_id": job.job_id, "tool": job.tool,
                        "message": "job was cancelled; its computed result (if any) is discarded."}
            if not job.done.is_set():
                return {"status": "running", "job_id": job.job_id, "tool": job.tool,
                        "message": "job still running — poll get_job_status, then retry get_job_result."}
            return job.result

    @mcp.tool()
    def cancel_job(job_id: str) -> dict:
        """Best-effort cancel of a background job.

        HONEST LIMITATION: this marks the job cancelled and WITHHOLDS its returned result payload,
        but it does NOT roll back persisted state and does NOT guarantee instant CPU reclamation.
        If the tool had ALREADY finished its body by the time cancel is processed (a mutating tool
        such as cnv_score/ingest), its checkpoint + run_log.jsonl entry were already committed via
        record_tool_run before cancel took effect — those persist, and only the in-memory result
        dict returned to the caller is withheld (so a naive re-run would DUPLICATE that checkpoint).
        Compute already handed to infercnvpy's INTERNAL ProcessPoolExecutor (or any C/BLAS section)
        cannot be interrupted from here — that work may run to completion; its CPU is fully reclaimed
        only when the pool/server shuts down (see the atexit/SIGTERM cleanup). A job that has not yet
        acquired the tool lock is stopped before it starts real work."""
        job = _registry.get(job_id)
        if job is None:
            return _unknown_job(job_id)
        with job.lock:
            if job.done.is_set():
                return {"status": job.state, "job_id": job.job_id, "tool": job.tool,
                        "message": f"job already finished ({job.state}); nothing to cancel."}
            job.cancel_requested = True
            job.state = "cancelled"
        lg.info("cancel requested for job %s (tool=%s)", job.job_id, job.tool)
        return {"status": "cancelled", "job_id": job.job_id, "tool": job.tool,
                "message": ("cancellation requested (best-effort). In-flight infercnvpy/BLAS "
                            "compute may run to completion; CPU is fully reclaimed on pool/server "
                            "shutdown. Only the returned result payload is withheld — if the tool "
                            "body had already finished, its checkpoint + run_log entry are ALREADY "
                            "committed and are NOT rolled back (a naive re-run would duplicate that "
                            "checkpoint).")}

    def _cleanup_jobs_and_pools() -> None:
        """Best-effort shutdown reaper: mark running jobs cancelled + terminate any lingering
        multiprocessing children (infercnvpy ProcessPoolExecutor workers + the forkserver daemon)
        so a client disconnect or server exit does not orphan CNV workers burning CPU. We have no
        handle to a single job's internal pool, so this reaps ALL children — correct at shutdown."""
        for job in _registry.all():
            if not job.done.is_set():
                with job.lock:
                    job.cancel_requested = True
                    job.state = "cancelled"
        try:
            import multiprocessing as mp
            for p in mp.active_children():
                try:
                    p.terminate()
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

    # attach (not auto-installed): main() wires these to atexit/SIGTERM; direct build_server()
    # callers (tests) get a server WITHOUT global signal/atexit side effects.
    mcp._scpilot_cleanup = _cleanup_jobs_and_pools   # type: ignore[attr-defined]
    mcp._scpilot_jobs = _registry                    # type: ignore[attr-defined]

    # --- model-agnostic workflow guidance, delivered over EVERY MCP channel so any client can
    # reach it regardless of which capabilities its LLM host supports (prompt / resource / tool) ---
    @mcp.tool()
    def scpilot_guidance() -> dict:
        """Return scpilot's full canonical analysis workflow (the same pipeline mode-2 follows):
        golden rules + step-by-step flow (QC -> preprocess -> cluster/markers -> marker-DB-free
        Tier-1 -> integration + per-embedding annotation -> harmonize/benchmark/best -> Tier-2
        subtype -> CNV/malignancy -> finalize/report) + annotation & malignancy hard rules.
        Call this once at the start of a run if your client did not surface the server instructions."""
        return {"workflow": prompts.full_workflow_guidance()}

    @mcp.prompt(name="scpilot_workflow",
                description="scpilot's full canonical scRNA-seq analysis pipeline + golden rules.")
    def scpilot_workflow() -> str:
        return prompts.full_workflow_guidance()

    @mcp.resource("scpilot://workflow", name="scpilot_workflow",
                  description="scpilot canonical analysis workflow (model-agnostic).",
                  mime_type="text/markdown")
    def scpilot_workflow_resource() -> str:
        return prompts.full_workflow_guidance()

    names = ([f"{s.name}_tool" for s in specs]
             + ["scpilot_version", "scpilot_guidance",
                "get_job_status", "get_job_result", "cancel_job"])
    lg.info("scpilot MCP server ready (v%s) — tools: %s", __version__, ", ".join(names))
    return mcp


def _forkserver_warmup_noop() -> None:
    """Picklable no-op used to force the forkserver daemon to spawn early (see
    ``_init_fork_safety``). Must be module-level so forkserver workers can import it."""
    return None


def _init_fork_safety(lg=None) -> None:
    """Make multiprocessing fork-safe in this long-lived, multi-threaded server.

    Some tools create a ``ProcessPoolExecutor`` via the DEFAULT mp context — notably
    ``cnv_score`` → infercnvpy ``process_map`` (``max_workers=cpu_count()``). The Linux
    default start method is ``fork``; forking a process that already holds threads/locks
    (the asyncio loop + the anyio tool worker thread + BLAS/numba pools spawned by earlier
    tools) lets each child inherit a LOCKED mutex, so the pool workers deadlock on a futex
    at 0% CPU — this was the historical cnv_score "stall" (N=cpu_count wedged children, the
    server's threads all in futex_wait). ``forkserver`` forks workers from a clean,
    single-threaded server process instead, so no held lock is inherited. We start the
    forkserver daemon NOW, at entry, while this process is still clean (single main thread),
    so even pools created much later (after many threads exist) fork from clean state."""
    import multiprocessing as mp
    if sys.platform == "win32":  # forkserver is POSIX-only; Windows already uses spawn
        return
    try:
        mp.set_start_method("forkserver", force=True)
        ctx = mp.get_context("forkserver")
        p = ctx.Process(target=_forkserver_warmup_noop)   # spawns the forkserver daemon while clean
        p.start()
        p.join()
        if lg:
            lg.info("multiprocessing start method = forkserver (fork-safe process pools)")
    except Exception as exc:  # noqa: BLE001 — never block server startup on this
        if lg:
            lg.warning("could not enable forkserver start method (%r); pools use default", exc)


def _install_cleanup_handlers(cleanup) -> None:
    """Wire the server's job/pool reaper to process exit (atexit) and SIGTERM.

    atexit covers a clean stdio disconnect (server.run returns → interpreter exit) and Ctrl-C
    (SIGINT → KeyboardInterrupt → normal exit). A SIGTERM handler is added too because SIGTERM's
    default action terminates WITHOUT running atexit — so a `kill <pid>` would otherwise orphan CNV
    workers. The handler reaps, then re-raises the signal with the default disposition so the
    process still terminates. Installed only from main() (not build_server) to avoid global
    side effects in tests. Best-effort: guarded so an unsupported platform never blocks startup."""
    import atexit
    import signal

    atexit.register(cleanup)

    def _on_sigterm(signum, frame):
        cleanup()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except Exception:  # noqa: BLE001 — e.g. not the main thread / unsupported platform
        pass


def main() -> None:
    """Entry point for ``scpilot mcp`` — run the stdio server."""
    # Bound BLAS/OpenMP threads FIRST — before _init_fork_safety warms the forkserver daemon — so
    # forkserver-spawned infercnvpy workers inherit the cap (the environment is captured at warmup/
    # fork time). Setting it later would leave the already-warmed daemon, and thus every CNV worker,
    # on cpu_count() threads and reintroduce the oversubscription runaway.
    from scpilot.vendor.harness import bound_thread_env
    bound_thread_env()
    _init_fork_safety()   # BEFORE build_server()/any thread: forkserver so tool pools are fork-safe
    server = build_server()
    _install_cleanup_handlers(server._scpilot_cleanup)   # reap jobs/pools on disconnect/exit
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
