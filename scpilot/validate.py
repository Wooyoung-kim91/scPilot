"""Tool-parameter validation — the single guard shared by the LLM dispatch path
(``llm.agent._execute_registry_tool``) and the user-preset path (``params.validate_overrides``).

These are SANITY guards (correct type, no div-by-zero, no negative counts, valid enum) built
from the JSON-Schema constraint keywords already carried in ``llm.agent._PARAM_HINTS`` — they do
NOT encode analysis decisions (which stay evidence-driven / LLM-chosen). A failed check yields a
human-readable message; the caller decides whether to reject (presets) or hand it back to the
model as a recoverable error (autonomous run).

Only CATALOGUED knobs are checked. Tools also accept rich, un-catalogued params (label maps,
marker sets) — those pass through here and are validated by the tool function's own signature.
"""

from __future__ import annotations

from typing import Any

# Cross-field constraints that a per-property schema can't express: (tool, message, predicate).
_CROSS_FIELD = {
    "cluster_sweep": lambda a: (
        ["res_max must be > res_min"]
        if ("res_min" in a and "res_max" in a
            and _is_num(a["res_min"]) and _is_num(a["res_max"]) and a["res_max"] <= a["res_min"])
        else []
    ),
}


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_one(param: str, value: Any, spec: dict) -> list[str]:
    """Validate one value against its JSON-Schema-style constraint spec."""
    problems: list[str] = []
    if value is None and spec.get("nullable"):
        return problems
    t = spec.get("type")
    # type checks (bool is NOT an int here; reject ints where a bool is wanted and vice-versa)
    if t == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        return [f"{param}: expected integer, got {type(value).__name__}"]
    if t == "number" and not _is_num(value):
        return [f"{param}: expected number, got {type(value).__name__}"]
    if t == "boolean" and not isinstance(value, bool):
        return [f"{param}: expected boolean, got {type(value).__name__}"]
    if t == "string" and not isinstance(value, str):
        return [f"{param}: expected string, got {type(value).__name__}"]
    if t == "array" and not isinstance(value, (list, tuple)):
        return [f"{param}: expected array, got {type(value).__name__}"]
    if t == "object" and not isinstance(value, dict):
        return [f"{param}: expected object, got {type(value).__name__}"]
    # bound checks (only meaningful for numbers)
    if _is_num(value):
        if "minimum" in spec and value < spec["minimum"]:
            problems.append(f"{param}={value}: must be >= {spec['minimum']}")
        if "exclusiveMinimum" in spec and value <= spec["exclusiveMinimum"]:
            problems.append(f"{param}={value}: must be > {spec['exclusiveMinimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            problems.append(f"{param}={value}: must be <= {spec['maximum']}")
    if "enum" in spec and value not in spec["enum"]:
        problems.append(f"{param}={value!r}: must be one of {spec['enum']}")
    return problems


def validate_params(tool: str, args: dict, *, hints: dict | None = None) -> list[str]:
    """Return a list of constraint-violation messages for ``args`` against ``tool``'s catalogued
    knobs (empty list = OK). Un-catalogued params are ignored here (the tool's signature owns them).
    """
    if hints is None:
        from scpilot.llm.agent import _PARAM_HINTS
        hints = _PARAM_HINTS
    tool_hints = hints.get(tool, {})
    problems: list[str] = []
    for param, value in (args or {}).items():
        spec = tool_hints.get(param)
        if spec:
            problems.extend(_check_one(param, value, spec))
    cross = _CROSS_FIELD.get(tool)
    if cross:
        problems.extend(cross(args or {}))
    return problems
