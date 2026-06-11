"""scpilot CLI entrypoint (Typer) — scpilot plan A5.

Four modes (plan §사용 모드):
  - ``scpilot mcp``                : MCP (stdio) server — primary integration (mode 1)
  - ``scpilot run <input>``        : self-driving CLI agent via Anthropic API (mode 2, optional)
  - ``scpilot step <stage> <input>``: single deterministic stage, no LLM (mode 3, debug/regression)
  - ``scpilot replay <session>``   : deterministic replay from run log (mode 4)
  - ``scpilot doctor``             : environment / capability preflight (plan A2)

A1 scope = skeleton: subcommands are declared and wired; bodies are stubs that
exit with a clear "not implemented" message. Real behaviour lands in A2/A5/A6/A7.
"""

from __future__ import annotations

import typer

from scpilot import __version__

app = typer.Typer(
    name="scpilot",
    help="LLM-driven scRNA-seq analysis (MCP server + CLI agent).",
    no_args_is_help=True,
    add_completion=False,
)


def _todo(stage: str) -> None:
    typer.secho(
        f"[scpilot] '{stage}' is not implemented yet (A1 skeleton). "
        f"See scpilot_plan.md for the build order.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=2)


@app.command()
def version() -> None:
    """Print the scpilot version."""
    typer.echo(f"scpilot {__version__}")


@app.command()
def doctor() -> None:
    """Environment / capability preflight (plan A2): deps + capability flags + smoke, as JSON."""
    import json

    from scpilot.doctor import run as run_doctor

    report = run_doctor()
    typer.echo(json.dumps(report, indent=2))
    raise typer.Exit(code=0 if report["ok"] else 1)


@app.command()
def mcp() -> None:
    """Start the MCP (stdio) server (plan A6/C2). stdout = protocol only."""
    from scpilot.mcp_server import main as run_server

    run_server()


@app.command()
def step(
    stage: str = typer.Argument(..., help="tool/stage name (e.g. load, qc_metrics)"),
    inp: str = typer.Argument(None, metavar="[INPUT]",
                              help="input .h5ad / profile (entry step only; later steps resume from the session)"),
    workdir: str = typer.Option(None, "--workdir", "-w", help="session working directory"),
    param: list[str] = typer.Option(None, "--param", "-p", help="k=v tool param (repeatable)"),
    seed: int = typer.Option(0, "--seed", help="global RNG seed"),
) -> None:
    """Run a single tool deterministically, no LLM (plan A5 / mode 3).

    Dispatches ``stage`` through the tool registry against an on-disk Session and
    prints the ToolResult as JSON. INPUT is given on the entry step (load/ingest, or
    the first step); later steps omit it and resume from the session's checkpoints.
    """
    import json
    from pathlib import Path

    from scpilot import schemas as S
    from scpilot import tools
    from scpilot.repro import set_global_seed
    from scpilot.session import DEFAULT_RUN_DIR, Session

    try:
        spec = tools.get(stage)
    except KeyError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    params = _parse_params(param or [])
    seed_rec = set_global_seed(seed)
    wd = workdir or DEFAULT_RUN_DIR
    if inp is None and not (Path(wd) / Session.MANIFEST).exists():
        typer.secho(f"no session at {wd} and no INPUT given — provide INPUT on the entry step",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    session = Session.create(wd, input_path=inp)   # inp None on resume → opens existing session

    result = spec.fn(session, **params)
    session.log_run(S.RunLogRecord(
        tool=stage, status=result.status, params=params, summary=result.summary, seed=seed,
        determinism_grade=result.determinism_grade,
        output_checkpoint=result.checkpoint, error_code=result.error_code,
        duration_s=result.duration_s, lib_versions={"seed_record": seed_rec},
    ).to_dict())

    typer.echo(json.dumps(result.to_dict(), indent=2))
    raise typer.Exit(code=0 if result.status == "success" else 1)


def _parse_params(items: list[str]) -> dict:
    """Parse repeated ``k=v`` options into a dict (int/float/bool/str inferred)."""
    out: dict = {}
    for it in items:
        if "=" not in it:
            raise typer.BadParameter(f"--param must be k=v, got: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = _coerce(v.strip())
    return out


def _coerce(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


@app.command()
def run(
    inp: str = typer.Argument(..., metavar="INPUT", help="input .h5ad path"),
    workdir: str = typer.Option(None, "--workdir", help="session working directory"),
    goal: str = typer.Option(None, "--goal", help="analysis goal for the agent"),
    effort: str = typer.Option("high", "--effort", help="LLM effort level (high|medium|low)"),
    backend: str = typer.Option(None, "--backend",
                                help="LLM backend: anthropic | openai (default: env SCPILOT_LLM_BACKEND)"),
    model: str = typer.Option(None, "--model", help="model name (default per backend; never hardcoded)"),
    base_url: str = typer.Option(None, "--base-url",
                                 help="OpenAI-compatible endpoint for a LOCAL LLM (e.g. http://localhost:11434/v1)"),
    seed: int = typer.Option(0, "--seed", help="global RNG seed (recorded for replay)"),
    max_iters: int = typer.Option(40, "--max-iters", help="max LLM tool-loop iterations"),
) -> None:
    """Self-driving full pipeline (plan D5 / mode 2).

    Builds the configured LLM provider (Anthropic by default, or a local/OpenAI-compatible
    endpoint via --backend openai --base-url ...), drives the deterministic tool registry
    autonomously, then writes a report (figures + interpretation). Every tool run + decision
    is logged so ``scpilot replay`` reproduces the session with NO LLM.
    """
    import json
    from pathlib import Path

    from scpilot import schemas as S
    from scpilot import tools
    from scpilot.llm.agent import run_agent
    from scpilot.llm import prompts
    from scpilot.llm.provider import ProviderConfig, ProviderUnavailable, build_provider
    from scpilot.repro import set_global_seed
    from scpilot.session import DEFAULT_RUN_DIR, Session

    if not Path(inp).exists():
        typer.secho(f"input not found: {inp}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    cfg = ProviderConfig.from_env(backend=backend, model=model, base_url=base_url,
                                  effort=effort, thinking={"type": "adaptive"})
    try:
        provider = build_provider(cfg)
    except ProviderUnavailable as exc:
        typer.secho(f"[scpilot run] LLM backend unavailable: {exc}", fg=typer.colors.RED, err=True)
        typer.secho("Run `scpilot doctor` and check the llm_provider section.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=2)

    seed_rec = set_global_seed(seed)
    wd = workdir or str(Path(DEFAULT_RUN_DIR) / "mode2")
    session = Session.create(wd, input_path=inp)

    typer.secho(f"[scpilot run] backend={provider.name} model={provider.model} -> {wd}",
                fg=typer.colors.CYAN, err=True)

    result = run_agent(session, provider, goal=goal, seed=seed, max_iters=max_iters)

    # final interpretation + report (LLM prose injected into the deterministic report tool)
    interp = ""
    try:
        sys_p = prompts.INTERPRETATION_PROMPT
        runs = session.run_log_path.read_text() if session.run_log_path.exists() else ""
        ctx = ("Write the final interpretation. Pipeline run log (JSONL):\n" + runs[-12000:]
               + "\n\nAgent final note:\n" + (result.final_text or ""))
        interp_resp = provider.complete([{"role": "user", "content": ctx}], system=sys_p)
        interp = interp_resp.text
        result.stats.add_usage(interp_resp.usage)
        result.stats.llm_turns += 1
    except Exception as exc:  # noqa: BLE001 — report still useful without prose
        typer.secho(f"[scpilot run] interpretation step failed: {exc}", fg=typer.colors.YELLOW, err=True)

    rep_params = {"interpretation": interp or result.final_text}
    rep = tools.run("report", session, **rep_params)
    # log the report run so `scpilot replay` reproduces it too (the interpretation prose is
    # recorded here → replay re-runs report with the SAME text, no LLM re-query needed).
    session.log_run(S.RunLogRecord(
        tool="report", status=rep.status, stage="report", params=rep_params,
        summary=rep.summary, seed=seed, output_checkpoint=rep.checkpoint,
        determinism_grade=rep.determinism_grade, error_code=rep.error_code,
        duration_s=rep.duration_s,
    ).to_dict())

    if result.stopped_reason == "max_iters":
        typer.secho(f"[scpilot run] WARNING: agent hit max_iters ({max_iters}) — analysis may be "
                    "INCOMPLETE; raise --max-iters or refine --goal.", fg=typer.colors.YELLOW, err=True)

    out = {
        "session": str(session.out),
        "stage_reached": session.manifest.stage,
        "stopped_reason": result.stopped_reason,
        "provider": {"backend": provider.name, "model": provider.model},
        "seed_record": seed_rec,
        "stats": result.stats.to_dict(),
        "report": rep.to_dict(),
        "final_text": result.final_text,
    }
    typer.echo(json.dumps(out, indent=2, default=str))
    incomplete = result.stopped_reason == "max_iters"
    raise typer.Exit(code=0 if (rep.status == "success" and not incomplete) else 1)


@app.command()
def replay(
    session: str = typer.Argument(..., help="session directory to replay"),
    to: str = typer.Option(None, "--to", help="replay workdir (default <session>/replay)"),
    seed: int = typer.Option(0, "--seed", help="global RNG seed for the replay"),
    dry_run: bool = typer.Option(False, "--dry-run", help="validate/list only; do not re-execute"),
) -> None:
    """Deterministic replay from the run log, no LLM (plan A7 / mode 4).

    Re-runs every recorded tool with its recorded params on a FRESH session, then diffs
    each summary against the original with per-determinism-grade tolerance (A=exact,
    B=structural±tol, C=bit-identical). ``--dry-run`` validates/lists the log without
    re-executing. Emits a JSON report; exit code is non-zero on any mismatch.
    """
    import json
    import shutil
    from pathlib import Path

    from scpilot.repro import replay_session, set_global_seed
    from scpilot.session import Session
    from scpilot import tools

    executor = None
    if not dry_run:
        orig = Session.open(session)
        inp = orig.manifest.input.get("path")
        if not inp:
            typer.secho("original session has no recorded input path — cannot re-execute (use --dry-run)",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)
        replay_dir = Path(to) if to else Path(session) / "replay"
        if replay_dir.exists():
            shutil.rmtree(replay_dir)           # derived artifact — regenerate fresh each replay
        set_global_seed(seed)
        replay_sess = Session.create(replay_dir, input_path=inp)
        executor = tools.make_replay_executor(replay_sess)

    report = replay_session(session, executor=executor)
    typer.echo(json.dumps(report, indent=2))
    raise typer.Exit(code=0 if (dry_run or report.get("all_match")) else 1)


if __name__ == "__main__":
    app()
