"""Parameter catalog + user presets — surface the tunable knobs so a user can pre-select
them before an autonomous ``scpilot run`` (instead of letting the LLM choose everything).

The catalog is built from the SAME single sources the agent already uses — the per-tool
parameter hints (``llm.agent._PARAM_HINTS``: type + description) merged with each tool's
function-signature defaults — so it never drifts from the real tools. A user fills a YAML
preset of ``{tool: {param: value}}``; those values are FIXED at run time (the LLM uses them
and does not re-choose), while unset params stay dynamically chosen. This generalizes the
existing ``resolutions`` (human-set clustering resolution) override to every catalogued knob.
"""

from __future__ import annotations

import inspect
from typing import Any


def params_catalog(toolset: list[str] | None = None) -> dict:
    """Build ``{tool: {param: {type, default, description}}}`` for the catalogued tools.

    Merges ``_PARAM_HINTS`` (type + description) with the tool's signature defaults. Tools
    default to the autonomous ``DEFAULT_TOOLSET`` (the decision-relevant knobs).
    """
    from scpilot import tools
    from scpilot.llm.agent import _PARAM_HINTS, DEFAULT_TOOLSET

    names = toolset if toolset is not None else DEFAULT_TOOLSET
    catalog: dict = {}
    for name in names:
        hints = _PARAM_HINTS.get(name)
        if not hints:
            continue
        try:
            sig = inspect.signature(tools.get(name).fn)
        except Exception:  # noqa: BLE001
            sig = None
        entry: dict = {}
        for param, meta in hints.items():
            default = None
            if sig is not None and param in sig.parameters:
                d = sig.parameters[param].default
                default = None if d is inspect.Parameter.empty else d
            entry[param] = {
                "type": meta.get("type", "string"),
                "default": default,
                "description": meta.get("description", ""),
            }
        if entry:
            catalog[name] = entry
    return catalog


def render_catalog_table(catalog: dict) -> str:
    """Human-readable catalog (tool -> param: type (default) — description)."""
    lines: list[str] = ["scpilot tunable parameters (unset = chosen dynamically by the LLM)\n"]
    for tool, params in catalog.items():
        lines.append(f"{tool}")
        for p, m in params.items():
            d = m["default"]
            dflt = "" if d is None else f"  [default: {d}]"
            lines.append(f"    {p:<16} {m['type']:<8}{dflt}  {m['description']}")
        lines.append("")
    return "\n".join(lines)


def render_template(catalog: dict) -> str:
    """A fillable YAML preset: every knob is a COMMENTED line with its default; the user
    uncomments + sets only the params they want to FIX. Pass via ``run --param-file``."""
    out: list[str] = [
        "# scpilot parameter preset — uncomment + set ONLY the params you want to FIX.",
        "# Unset params stay dynamically chosen by the LLM.",
        "# Use with:  scpilot run <input.h5ad> --param-file <this file>",
        "",
    ]
    for tool, params in catalog.items():
        out.append(f"{tool}:")
        for p, m in params.items():
            d = m["default"]
            shown = "" if d is None else d
            out.append(f"  # {m['type']} — {m['description']}")
            out.append(f"  # {p}: {shown}")
        out.append("")
    return "\n".join(out)


def load_param_file(path: str) -> dict:
    """Load a ``{tool: {param: value}}`` preset YAML. Tool entries that are empty/None (all
    params left commented) are dropped. Returns a plain nested dict (values coerced by YAML)."""
    import yaml

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("param-file must be a mapping of {tool: {param: value}}")
    out: dict = {}
    for tool, params in data.items():
        if isinstance(params, dict) and params:
            out[str(tool)] = {str(k): v for k, v in params.items()}
    return out


def validate_overrides(overrides: dict, catalog: dict | None = None) -> list[str]:
    """Return problems for a user preset: unknown tools/params AND out-of-range / wrong-type values.

    A user-fixed preset is authoritative (it overrides the LLM), so it is validated STRICTLY — the
    CLI rejects the run if this returns anything, rather than silently feeding a bad value into a
    tool. Never raises; the caller decides how to surface the list."""
    from scpilot.validate import validate_params

    catalog = catalog if catalog is not None else params_catalog()
    problems: list[str] = []
    for tool, params in overrides.items():
        if tool not in catalog:
            problems.append(f"unknown tool in param-file: '{tool}'")
            continue
        for p in params:
            if p not in catalog[tool]:
                problems.append(f"unknown param '{p}' for tool '{tool}'")
        problems.extend(validate_params(tool, params))   # type / bound / enum guards
    return problems
