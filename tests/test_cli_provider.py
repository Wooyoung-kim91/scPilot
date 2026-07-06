"""CLIProvider (subprocess claude/codex) + build_role_provider — I-24 Increment 2."""

import shutil

import pytest

from scpilot.llm import provider as P
from scpilot.llm.provider import (CLIProvider, ProviderConfig, ProviderUnavailable,
                                   _extract_json_object, build_role_provider)


def test_build_role_provider_routing():
    cli = build_role_provider({"type": "cli", "cli": "codex", "model": "gpt-5"})
    assert isinstance(cli, CLIProvider) and cli.cli == "codex" and cli.model == "gpt-5"
    assert build_role_provider({"type": "host_plugin", "plugin": "codex"}) is None
    with pytest.raises(ProviderUnavailable):
        build_role_provider({"type": "telepathy"})


def test_cli_argv_default_and_model():
    p = CLIProvider(ProviderConfig(backend="cli", model="gpt-5"), cli="codex")
    assert p.argv() == ["codex", "exec", "--model", "gpt-5"]
    q = CLIProvider(ProviderConfig(backend="cli"), cli="claude-code")
    assert q.argv() == ["claude", "-p"]                       # no model → CLI's own default
    r = CLIProvider(ProviderConfig(backend="cli", model="m"), cli="codex",
                    cmd=["codex", "exec", "--model", "{model}", "--json"])
    assert r.argv() == ["codex", "exec", "--model", "m", "--json"]


def test_cli_complete_plain_text(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _e: "/usr/bin/codex")
    p = CLIProvider(ProviderConfig(backend="cli"), cli="codex")
    monkeypatch.setattr(p, "_run_cli", lambda argv, prompt: "  the review says fine  ")
    resp = p.complete([{"role": "user", "content": "review this"}], system="you are a critic")
    assert resp.text == "the review says fine" and not resp.tool_calls


def test_cli_complete_forced_structured(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _e: "/usr/bin/codex")
    p = CLIProvider(ProviderConfig(backend="cli"), cli="codex")
    captured = {}

    def fake_run(argv, prompt):
        captured["prompt"] = prompt
        return 'Sure!\n```json\n{"verdicts": [{"cluster": "0", "verdict": "refuted"}]}\n```\n'

    monkeypatch.setattr(p, "_run_cli", fake_run)
    tools = [{"name": "emit_audit", "input_schema": {"type": "object"}}]
    resp = p.complete([{"role": "user", "content": "audit"}], tools=tools, tool_choice="emit_audit")
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "emit_audit"
    assert resp.tool_calls[0].arguments["verdicts"][0]["verdict"] == "refuted"
    # the forced-tool JSON-only instruction was appended to the prompt
    assert "ONLY a single JSON object" in captured["prompt"]


def test_cli_missing_executable_raises(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _e: None)
    p = CLIProvider(ProviderConfig(backend="cli"), cli="codex")
    with pytest.raises(ProviderUnavailable):
        p.complete([{"role": "user", "content": "x"}], tool_choice="t")


def test_extract_json_object():
    assert _extract_json_object('prefix {"a": 1} suffix') == {"a": 1}
    assert _extract_json_object('```json\n{"b": [1,2]}\n```') == {"b": [1, 2]}
    assert _extract_json_object('a string with { unbalanced') is None
    assert _extract_json_object("no json here") is None
    # nested + a brace inside a string must not confuse the balancer
    assert _extract_json_object('{"s": "a}b", "n": {"x": 1}}') == {"s": "a}b", "n": {"x": 1}}
