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
    effort: str = typer.Option("high", "--effort", help="LLM effort level"),
) -> None:
    """Self-driving full pipeline via Anthropic API (plan D5/mode 2). Stub."""
    _todo("run")


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
