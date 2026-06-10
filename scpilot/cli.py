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
    stage: str = typer.Argument(..., help="core stage name, e.g. cluster"),
    inp: str = typer.Argument(..., metavar="INPUT", help="input .h5ad path"),
) -> None:
    """Run a single deterministic stage with no LLM (plan A5/mode 3). Stub."""
    _todo(f"step {stage}")


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
    """Deterministic replay from the run log (plan mode 4). Stub."""
    _todo("replay")


if __name__ == "__main__":
    app()
