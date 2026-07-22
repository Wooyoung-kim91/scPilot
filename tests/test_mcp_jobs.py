"""Unit tests for the long_running job/poll interface + parallelism cap (plan C1 / cnv fix).

These exercise the in-process pieces directly (no real infercnvpy/CNV compute, no MCP subprocess):
- Fix A: cnv._resolve_n_jobs caps an unset n_jobs to min(cpu,8) but preserves an explicit value,
  and harness.bound_thread_env sets the BLAS/OpenMP env vars without clobbering a user value.
- Fix B: the grace-window handler (fast tool returns inline; slow tool returns a job_id that
  becomes retrievable), the job registry (status transitions, unknown id), and best-effort cancel.

The slow tool is a MOCK that sleeps; nothing here launches real CNV work.
"""

import time

import anndata as ad
import anyio
import numpy as np
import pytest
from scipy import sparse

from scpilot import schemas as S
from scpilot import tools
from scpilot.vendor.harness import bound_thread_env, bounded_thread_count, _THREAD_ENV_VARS


# --------------------------------------------------------------------------- #
# Fix A — parallelism / thread-count caps
# --------------------------------------------------------------------------- #
def test_resolve_n_jobs_caps_none_and_preserves_explicit():
    from scpilot.core.cnv import _resolve_n_jobs
    import os

    cap = min(os.cpu_count() or 1, 8)
    assert _resolve_n_jobs(None) == cap        # unset => bounded default
    assert cap <= 8                            # never the historical cpu_count() runaway
    assert _resolve_n_jobs(1) == 1             # explicit small value preserved
    assert _resolve_n_jobs(16) == 16           # explicit large value honored (caller's choice)


def test_bound_thread_env_setdefault_only(monkeypatch):
    # a value the user already set must NOT be clobbered
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    for v in _THREAD_ENV_VARS:
        if v != "OMP_NUM_THREADS":
            monkeypatch.delenv(v, raising=False)
    bound_thread_env()
    import os
    assert os.environ["OMP_NUM_THREADS"] == "3"                      # user value preserved
    assert os.environ["OPENBLAS_NUM_THREADS"] == str(bounded_thread_count())  # unset -> bounded


# --------------------------------------------------------------------------- #
# Fix B — job registry + grace window
# --------------------------------------------------------------------------- #
def test_job_registry_create_get_unknown():
    from scpilot.mcp_server import _JobRegistry

    reg = _JobRegistry()
    j1 = reg.create("cnv_score")
    j2 = reg.create("cnv_score")
    assert j1.job_id != j2.job_id                 # unique ids
    assert reg.get(j1.job_id) is j1
    assert reg.get("job-does-not-exist") is None  # unknown id
    assert j1.state == "running" and not j1.done.is_set()


def _tiny_h5ad(path):
    a = ad.AnnData(sparse.csr_matrix(np.ones((4, 3), dtype="float32")))
    a.write_h5ad(path)
    return str(path)


def _fns(srv):
    """name -> callable for every tool on the server (handlers + job tools)."""
    return {t.name: t.fn for t in srv._tool_manager.list_tools()}


@pytest.fixture
def fake_tools():
    """Register a fast + slow MOCK long_running tool; unregister after the test."""
    calls = {"fast": 0, "slow": 0}

    @tools.register("fake_fast", long_running=True, description="mock fast long tool")
    def _fake_fast(session, **params):
        calls["fast"] += 1
        return S.success("fake_fast", summary={"marker": "fast_done"})

    @tools.register("fake_slow", long_running=True, description="mock slow long tool")
    def _fake_slow(session, sleep_s: float = 2.0, **params):
        calls["slow"] += 1
        time.sleep(sleep_s)
        return S.success("fake_slow", summary={"marker": "slow_done"})

    yield calls
    for n in ("fake_fast", "fake_slow"):
        tools.REGISTRY.pop(n, None)


def test_fast_long_tool_returns_inline(tmp_path, fake_tools, monkeypatch):
    monkeypatch.setenv("SCPILOT_MCP_JOB_GRACE_SECONDS", "10")   # generous: fast tool finishes inline
    from scpilot.mcp_server import build_server

    srv = build_server()
    fn = _fns(srv)["fake_fast_tool"]
    h5ad = _tiny_h5ad(tmp_path / "in.h5ad")
    res = anyio.run(fn, h5ad, str(tmp_path / "wd"), None, 0)
    # finished within grace -> the normal ToolResult dict inline (no job envelope), bookkeeping ran once
    assert res["status"] == "success"
    assert res["summary"]["marker"] == "fast_done"
    assert "job_id" not in res
    assert fake_tools["fast"] == 1


def test_slow_long_tool_becomes_a_job(tmp_path, fake_tools, monkeypatch):
    monkeypatch.setenv("SCPILOT_MCP_JOB_GRACE_SECONDS", "0.3")  # short: slow tool exceeds it
    from scpilot.mcp_server import build_server

    srv = build_server()
    fns = _fns(srv)
    h5ad = _tiny_h5ad(tmp_path / "in.h5ad")

    res = anyio.run(fns["fake_slow_tool"], h5ad, str(tmp_path / "wd"), {"sleep_s": 1.5}, 0)
    # exceeded grace -> prompt job envelope, work continues in background
    assert res["status"] == "running"
    job_id = res["job_id"]
    assert job_id and res["tool"] == "fake_slow"

    # still running right away
    st = fns["get_job_status"](job_id)
    assert st["state"] == "running" and st["done"] is False
    rr = fns["get_job_result"](job_id)
    assert rr["status"] == "running"

    # poll until it finishes, then the result is retrievable (identical to an inline return)
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if fns["get_job_status"](job_id)["state"] != "running":
            break
        time.sleep(0.1)
    st = fns["get_job_status"](job_id)
    assert st["state"] == "succeeded" and st["done"] is True and st["elapsed_s"] > 0
    done = fns["get_job_result"](job_id)
    assert done["status"] == "success"
    assert done["summary"]["marker"] == "slow_done"


def test_get_job_status_unknown_id(tmp_path, fake_tools):
    from scpilot.mcp_server import build_server

    fns = _fns(build_server())
    for tool in ("get_job_status", "get_job_result", "cancel_job"):
        r = fns[tool]("nope")
        assert r["status"] == "error" and r["error_code"] == "missing_input"


def test_tool_bodies_never_run_concurrently(tmp_path, monkeypatch):
    """Mutual exclusion: even with two long tools launched overlapping, `_tool_lock` serializes
    their BODIES — the process-global RNG/scanpy-settings invariant requires one-at-a-time."""
    import threading

    monkeypatch.setenv("SCPILOT_MCP_JOB_GRACE_SECONDS", "0.05")  # both exceed grace -> both go bg
    from scpilot.mcp_server import build_server

    state = {"cur": 0, "max": 0}
    slock = threading.Lock()

    def _overlap_body(session, **params):
        with slock:                       # detect any window where two bodies are simultaneously in
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(0.4)                   # hold the body open long enough to expose an overlap
        with slock:
            state["cur"] -= 1
        return S.success("overlap_probe", summary={"marker": "ok"})

    for n in ("fake_ov_a", "fake_ov_b"):
        tools.register(n, long_running=True, description="overlap probe")(_overlap_body)
    try:
        fns = _fns(build_server())
        h5ad = _tiny_h5ad(tmp_path / "in.h5ad")
        job_ids: list[str] = []
        jlock = threading.Lock()

        def _launch(tool: str) -> None:
            res = anyio.run(fns[tool], h5ad, str(tmp_path / f"wd_{tool}"), None, 0)
            with jlock:
                job_ids.append(res["job_id"])

        t1 = threading.Thread(target=_launch, args=("fake_ov_a_tool",))
        t2 = threading.Thread(target=_launch, args=("fake_ov_b_tool",))
        t1.start(); t2.start(); t1.join(); t2.join()
        assert len(job_ids) == 2                                  # both handed back as bg jobs

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if all(fns["get_job_status"](j)["state"] != "running" for j in job_ids):
                break
            time.sleep(0.1)
        for j in job_ids:
            assert fns["get_job_status"](j)["state"] == "succeeded"
        assert state["max"] == 1     # never two bodies at once -> _tool_lock serialized them
    finally:
        for n in ("fake_ov_a", "fake_ov_b"):
            tools.REGISTRY.pop(n, None)


def test_long_tool_exception_becomes_failed_job(tmp_path, monkeypatch):
    """A backgrounded long tool whose body raises ends in state 'failed', and get_job_result
    returns a STRUCTURED error (not a raw unhandled exception, not a hang)."""
    monkeypatch.setenv("SCPILOT_MCP_JOB_GRACE_SECONDS", "0.2")
    from scpilot.mcp_server import build_server

    @tools.register("fake_boom", long_running=True, description="mock long tool that raises")
    def _fake_boom(session, **params):
        time.sleep(0.5)                       # exceed the grace window before blowing up
        raise RuntimeError("boom in tool body")

    try:
        fns = _fns(build_server())
        h5ad = _tiny_h5ad(tmp_path / "in.h5ad")
        res = anyio.run(fns["fake_boom_tool"], h5ad, str(tmp_path / "wd"), None, 0)
        assert res["status"] == "running"
        job_id = res["job_id"]

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if fns["get_job_status"](job_id)["state"] != "running":
                break
            time.sleep(0.1)
        assert fns["get_job_status"](job_id)["state"] == "failed"

        rr = fns["get_job_result"](job_id)
        assert rr["status"] == "error"                 # structured error envelope, no raw traceback
        assert rr["error_code"] == "internal"
        assert "boom in tool body" in rr["error"]
    finally:
        tools.REGISTRY.pop("fake_boom", None)


def test_cancel_job_best_effort(tmp_path, fake_tools, monkeypatch):
    monkeypatch.setenv("SCPILOT_MCP_JOB_GRACE_SECONDS", "0.3")
    from scpilot.mcp_server import build_server

    fns = _fns(build_server())
    h5ad = _tiny_h5ad(tmp_path / "in.h5ad")
    res = anyio.run(fns["fake_slow_tool"], h5ad, str(tmp_path / "wd"), {"sleep_s": 1.5}, 0)
    job_id = res["job_id"]

    c = fns["cancel_job"](job_id)
    assert c["status"] == "cancelled"
    # immediate best-effort feedback: state flips to cancelled (compute may still be finishing)
    assert fns["get_job_status"](job_id)["state"] == "cancelled"
    assert fns["get_job_result"](job_id)["status"] == "cancelled"
    # cancelling an unknown/finished job is handled gracefully
    time.sleep(1.8)
    again = fns["cancel_job"](job_id)
    assert again["status"] == "cancelled"


# --------------------------------------------------------------------------- #
# Bug B — a job cancelled while queued on _tool_lock does NOT run its body
# --------------------------------------------------------------------------- #
def test_cancel_before_execution_skips_body_and_checkpoint(tmp_path, monkeypatch):
    """A job cancelled WHILE it is still queued on `_tool_lock` (never started its body) must not
    call the tool and must not record a checkpoint/run_log entry. We hold the lock with a blocker
    job, queue a victim behind it, cancel the victim, then release the blocker; the victim then
    acquires the lock, sees cancel_requested, and skips its body entirely."""
    import threading

    monkeypatch.setenv("SCPILOT_MCP_JOB_GRACE_SECONDS", "0.2")  # both exceed grace -> background
    from scpilot.mcp_server import build_server

    gate = threading.Event()
    calls = {"blocker": 0, "victim": 0}

    def _blocker(session, **params):
        calls["blocker"] += 1
        gate.wait(5)                      # hold _tool_lock until the test releases it
        return S.success("blocker", summary={"marker": "ok"})

    def _victim(session, **params):
        calls["victim"] += 1              # MUST NOT run: cancelled while queued behind blocker
        return S.success("victim", summary={"marker": "ran"})

    tools.register("blk", long_running=True, description="holds the tool lock")(_blocker)
    tools.register("vic", long_running=True, description="queued behind the blocker")(_victim)
    try:
        fns = _fns(build_server())
        h5ad = _tiny_h5ad(tmp_path / "in.h5ad")

        # 1) blocker starts and acquires _tool_lock (wait until its body actually entered)
        rb = anyio.run(fns["blk_tool"], h5ad, str(tmp_path / "wd_b"), None, 0)
        assert rb["status"] == "running"
        deadline = time.monotonic() + 5
        while calls["blocker"] == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert calls["blocker"] == 1                       # blocker holds the lock now

        # 2) victim launches; its worker blocks on _tool_lock (queued behind blocker)
        victim_wd = tmp_path / "wd_v"
        rv = anyio.run(fns["vic_tool"], h5ad, str(victim_wd), None, 0)
        assert rv["status"] == "running"
        victim_id = rv["job_id"]

        # 3) cancel the victim WHILE it is still queued (before its body ran)
        assert fns["cancel_job"](victim_id)["status"] == "cancelled"

        # 4) release the blocker; the victim now acquires the lock and must skip its body
        gate.set()

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if fns["get_job_status"](victim_id)["done"]:
                break
            time.sleep(0.05)

        st = fns["get_job_status"](victim_id)
        assert st["done"] is True and st["state"] == "cancelled"
        assert calls["victim"] == 0                        # body never ran
        assert calls["blocker"] == 1
        # no persisted state for the cancelled-before-start job: Session.create runs only AFTER the
        # pre-exec check, so the victim's workdir / run_log were never written
        assert not (victim_wd / "run_log.jsonl").exists()
        assert fns["get_job_result"](victim_id)["status"] == "cancelled"
    finally:
        for n in ("blk", "vic"):
            tools.REGISTRY.pop(n, None)


# --------------------------------------------------------------------------- #
# Bug C — bounded eviction of retained finished jobs (never a running one)
# --------------------------------------------------------------------------- #
def test_registry_evicts_finished_beyond_cap_but_never_running():
    from scpilot.mcp_server import _JobRegistry

    reg = _JobRegistry(max_retained=3)
    finished_ids = []
    for _ in range(6):
        j = reg.create("t")
        j.done.set()                      # mark finished so the next create can evict it
        finished_ids.append(j.job_id)

    running = reg.create("t")             # done NOT set -> must never be evicted
    remaining = {j.job_id for j in reg.all()}

    kept_finished = [i for i in finished_ids if i in remaining]
    assert len(kept_finished) <= 3                     # capped
    assert kept_finished == finished_ids[-3:]          # the most-recent finished are retained
    assert reg.get(finished_ids[0]) is None            # oldest were purged
    assert running.job_id in remaining                 # running job survived eviction
    assert reg.get(running.job_id) is running


def test_purged_job_id_returns_unknown(tmp_path, fake_tools, monkeypatch):
    """A purged (evicted) job id is indistinguishable from a never-seen id: the job tools return
    the normal structured unknown-id message. Uses the server's REAL registry (the one the job-tool
    closures actually consult), with a tiny cap so real fast-job creates force an eviction."""
    monkeypatch.setenv("SCPILOT_MCP_JOB_GRACE_SECONDS", "10")   # generous: fast tool finishes inline
    monkeypatch.setenv("SCPILOT_MCP_MAX_RETAINED_JOBS", "1")    # keep only the most-recent finished
    from scpilot.mcp_server import build_server

    srv = build_server()
    reg = srv._scpilot_jobs                              # the exact registry the handlers use
    fns = _fns(srv)
    h5ad = _tiny_h5ad(tmp_path / "in.h5ad")

    # first fast job finishes inline; capture its id before later creates can evict it
    assert anyio.run(fns["fake_fast_tool"], h5ad, str(tmp_path / "wd0"), None, 0)["status"] == "success"
    first_id = reg.all()[-1].job_id
    # two more fast jobs; the create sweeps evict the oldest finished job (cap 1)
    for i in (1, 2):
        anyio.run(fns["fake_fast_tool"], h5ad, str(tmp_path / f"wd{i}"), None, 0)

    assert reg.get(first_id) is None                    # the earliest finished job was purged
    for tool in ("get_job_status", "get_job_result", "cancel_job"):
        r = fns[tool](first_id)
        assert r["status"] == "error" and r["error_code"] == "missing_input"
