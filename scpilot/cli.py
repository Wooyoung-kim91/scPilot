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
    stage: str = typer.Argument(..., help="tool/stage name (e.g. inspect)"),
    inp: str = typer.Argument(..., metavar="INPUT", help="input .h5ad path"),
    workdir: str = typer.Option(None, "--workdir", "-w", help="session working directory"),
    param: list[str] = typer.Option(None, "--param", "-p", help="k=v tool param (repeatable)"),
    seed: int = typer.Option(0, "--seed", help="global RNG seed"),
) -> None:
    """Run a single tool deterministically, no LLM (plan A5 / mode 3).

    Dispatches ``stage`` through the tool registry against an on-disk Session and
    prints the ToolResult as JSON. Used for debug + structural-invariant regression.
    """
    import json
    from pathlib import Path

    from scpilot import schemas as S
    from scpilot import tools
    from scpilot.repro import set_global_seed
    from scpilot.session import Session

    try:
        spec = tools.get(stage)
    except KeyError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    params = _parse_params(param or [])
    seed_rec = set_global_seed(seed)
    wd = workdir or str(Path.cwd() / "scpilot_session")
    session = Session.create(wd, input_path=inp)

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
) -> None:
    """Deterministic replay from the run log, no LLM (plan A7 / mode 4).

    Without a tool registry (plan C1/A5) this runs in dry-run mode: validates and
    reports the recorded run log + decisions. Emits a JSON report.
    """
    import json

    from scpilot.repro import replay_session

    report = replay_session(session)  # executor wired once the registry exists
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
