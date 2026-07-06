"""Run-level LLM topology — which engine plays each pipeline role, selectable at MCP-call time.

scpilot invokes LLMs in THREE mechanisms, chosen PER ROLE (see ``configure_run``):

- ``api``          : scpilot calls an LLM API directly (Anthropic / OpenAI-compatible) — see
                     ``llm/provider.py``.
- ``cli``          : scpilot spawns a CLI tool (``claude`` / ``codex``) as a subprocess to get the
                     answer (the "CLI tool execution" invocation type).
- ``host_plugin``  : scpilot calls NOTHING; it emits a request the HOST fulfils by delegating to its
                     plugin (e.g. Claude Code → Codex plugin), then feeds the verdict back via the
                     normal tool path. Fits mode-1 (MCP) where the host drives.

Roles:
- ``analysis``    : the annotator / pipeline driver. In mode-1 (MCP) this IS the host CLI itself, so
                    it is a DECLARATION recorded for provenance + to pick a cross-engine reviewer;
                    scpilot never invokes it.
- ``reviewer``    : Tier-4 adversarial critique. Cross-engine (reviewer ≠ analysis) is preferred.
- ``annotator``   : re-annotation of refuted clusters (defaults to analysis when omitted).
- ``interpreter`` : final report prose (defaults to analysis when omitted).

No biology lives here — these are execution-mechanism names, not marker/tissue assumptions (the §1
no-hardcoding rule is untouched). Model names remain caller-supplied, never hardcoded into logic.
"""

from __future__ import annotations

ROLES = ("analysis", "reviewer", "annotator", "interpreter")
PROVIDER_TYPES = ("api", "cli", "host_plugin")
# execution mechanisms (CLI tools / plugins), NOT models
KNOWN_CLIS = ("claude-code", "claude", "codex")
_API_BACKENDS = ("anthropic", "openai")
# map a CLI/plugin identity to the executable we'd probe / spawn
_CLI_EXECUTABLE = {"claude-code": "claude", "claude": "claude", "codex": "codex"}


def _norm_spec(role: str, spec) -> tuple[dict | None, list[str]]:
    if not isinstance(spec, dict):
        return None, [f"{role}: spec must be an object, got {type(spec).__name__}"]
    t = str(spec.get("type", "")).strip().lower()
    if t not in PROVIDER_TYPES:
        return None, [f"{role}: type must be one of {PROVIDER_TYPES}, got {spec.get('type')!r}"]
    out: dict = {"type": t}
    problems: list[str] = []
    if t == "api":
        backend = str(spec.get("backend", "anthropic")).lower().strip()
        if backend not in _API_BACKENDS:
            problems.append(f"{role}: api backend must be one of {_API_BACKENDS}, got {backend!r}")
        out["backend"] = backend
        if spec.get("model"):
            out["model"] = str(spec["model"])
        if spec.get("base_url"):
            out["base_url"] = str(spec["base_url"])
    elif t == "cli":
        cli = str(spec.get("cli", "")).lower().strip()
        if cli not in KNOWN_CLIS:
            problems.append(f"{role}: cli must be one of {KNOWN_CLIS}, got {cli!r}")
        out["cli"] = cli
        if spec.get("model"):
            out["model"] = str(spec["model"])
        if spec.get("cmd"):                       # optional explicit argv template (advanced)
            out["cmd"] = [str(x) for x in spec["cmd"]]
    else:  # host_plugin
        plugin = str(spec.get("plugin", "")).lower().strip()
        if plugin not in KNOWN_CLIS:
            problems.append(f"{role}: plugin must be one of {KNOWN_CLIS}, got {plugin!r}")
        out["plugin"] = plugin
    return out, problems


def validate_topology(topo) -> tuple[dict, list[str]]:
    """Validate + normalize a topology dict. Returns ``(normalized, problems)``.

    ``analysis`` is a declaration (recorded, never invoked). ``annotator``/``interpreter`` default to
    ``analysis`` at consumption time when omitted here.
    """
    if not isinstance(topo, dict):
        return {}, ["topology must be an object mapping role -> spec"]
    norm: dict = {}
    problems: list[str] = []
    for role, spec in topo.items():
        if role not in ROLES:
            problems.append(f"unknown role {role!r} (valid: {ROLES})")
            continue
        s, probs = _norm_spec(role, spec)
        problems.extend(probs)
        if s is not None and not probs:
            norm[role] = s
    return norm, problems


def cli_executable(cli_or_plugin: str) -> str:
    """The executable name we probe/spawn for a CLI/plugin identity."""
    return _CLI_EXECUTABLE.get(cli_or_plugin, cli_or_plugin)


def probe_availability(norm: dict) -> dict:
    """Non-fatal per-role availability probe (surfaced at ``configure_run`` time). No network calls."""
    import shutil

    from scpilot.llm import provider as P

    out: dict = {}
    for role, spec in norm.items():
        t = spec["type"]
        if t == "api":
            pb = P.probe_backend(spec.get("backend"))
            out[role] = {"type": "api", "backend": spec.get("backend"),
                         "model": spec.get("model") or pb.get("model"),
                         "ready": bool(pb.get("ready")), "reason": pb.get("reason")}
        elif t == "cli":
            exe = cli_executable(spec["cli"])
            path = shutil.which(exe)
            out[role] = {"type": "cli", "cli": spec["cli"], "executable": exe,
                         "model": spec.get("model"), "ready": bool(path), "path": path,
                         "reason": None if path else f"'{exe}' not found on PATH"}
        else:  # host_plugin — fulfilled by the host; scpilot cannot probe it
            out[role] = {"type": "host_plugin", "plugin": spec["plugin"], "ready": None,
                         "reason": "fulfilled by the host — scpilot cannot probe the host's plugin"}
    return out
