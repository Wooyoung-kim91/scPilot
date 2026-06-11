"""LLM provider abstraction — scpilot plan D2 (+ user scope: local/OpenAI-compatible).

Mode 2 (``scpilot run``) is the substitute for a host agent (Claude Code / Codex
via MCP) when the user has **no subscription** or **must use a local LLM**. So this
abstraction supports two backends behind one interface:

- ``anthropic``           : Anthropic API (default; model configurable, default
                            ``claude-opus-4-8``). Reads ``ANTHROPIC_API_KEY`` from env.
- ``openai`` / local      : any OpenAI-compatible chat-completions endpoint
                            (Ollama / vLLM / LM Studio / OpenAI) via ``base_url``
                            + optional ``api_key``. Model name configurable.

Backend selection is by config/env — **no hardcoded model names**. Both SDKs are
*optional*: if the chosen backend's SDK is missing we raise ``ProviderUnavailable``
with a doctor-style, actionable message (we never hard-require the dep at import).

The interface is intentionally small and stateless::

    provider.complete(messages, tools=..., system=..., tool_choice=...) -> LLMResponse

``LLMResponse`` normalizes the two backends' wire formats: ``text`` (assistant prose),
``tool_calls`` (list of ``ToolCall``), ``stop_reason``, and ``usage`` (token counts).
The agent loop (``llm/agent.py``) drives the call→result→repeat cycle on top of this —
identical control flow regardless of backend.

Config resolution (highest priority first):
  1. explicit kwargs to ``build_provider`` / env passed by the CLI ``run`` command
  2. environment variables (see ``ProviderConfig.from_env``)
  3. built-in defaults (backend=anthropic, model=claude-opus-4-8)

Env vars:
  SCPILOT_LLM_BACKEND   anthropic | openai          (default: anthropic)
  SCPILOT_LLM_MODEL     model name                  (default per backend)
  SCPILOT_LLM_BASE_URL  OpenAI-compatible base_url   (e.g. http://localhost:11434/v1)
  SCPILOT_LLM_API_KEY   api key for the chosen backend (openai backend; optional for local)
  ANTHROPIC_API_KEY     api key (anthropic backend; standard env name)
  SCPILOT_LLM_MAX_TOKENS / SCPILOT_LLM_TEMPERATURE  generation knobs (optional)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

# Built-in defaults (model name is config, NEVER hardcoded into the agent logic).
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"           # only a fallback default for the OAI backend
DEFAULT_MAX_TOKENS = 8192


class ProviderError(RuntimeError):
    """Base class for provider problems."""


class ProviderUnavailable(ProviderError):
    """The selected backend cannot run (SDK missing, no key, unreachable)."""


# --------------------------------------------------------------------------- #
# Normalized wire types
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """A single tool/function call the model wants run (backend-normalized)."""
    id: str                       # provider-assigned call id (echoed back in the result)
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """One assistant turn, normalized across backends."""
    text: str = ""                              # assistant prose (may be empty on a tool turn)
    tool_calls: list = field(default_factory=list)   # list[ToolCall]
    stop_reason: str | None = None              # "tool_use" | "end_turn" | "stop" | ...
    usage: dict = field(default_factory=dict)   # {input_tokens, output_tokens}
    raw: Any = None                             # provider-native object (debug only)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class ProviderConfig:
    backend: str = "anthropic"                  # "anthropic" | "openai"
    model: str | None = None                    # resolved to a backend default if None
    base_url: str | None = None                 # OpenAI-compatible endpoint (local LLM)
    api_key: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float | None = None
    # Anthropic-only extended controls (plan D5): adaptive thinking + effort.
    thinking: dict | None = None                # e.g. {"type": "adaptive"}
    effort: str | None = None                   # "high" -> output_config.effort

    @classmethod
    def from_env(cls, **overrides) -> "ProviderConfig":
        """Build config from env vars; ``overrides`` (CLI args) win over env."""
        def pick(key_env: str, default=None):
            v = os.environ.get(key_env)
            return v if v not in (None, "") else default

        backend = overrides.get("backend") or pick("SCPILOT_LLM_BACKEND", "anthropic")
        backend = backend.lower().strip()
        model = overrides.get("model") or pick("SCPILOT_LLM_MODEL")
        base_url = overrides.get("base_url") or pick("SCPILOT_LLM_BASE_URL")
        # api key: backend-specific env, falling back to the generic one
        if backend == "anthropic":
            api_key = overrides.get("api_key") or pick("ANTHROPIC_API_KEY") or pick("SCPILOT_LLM_API_KEY")
        else:
            api_key = overrides.get("api_key") or pick("SCPILOT_LLM_API_KEY") or pick("OPENAI_API_KEY")
        max_tokens = int(overrides.get("max_tokens") or pick("SCPILOT_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
        temp_raw = overrides.get("temperature")
        if temp_raw is None:
            temp_raw = pick("SCPILOT_LLM_TEMPERATURE")
        temperature = float(temp_raw) if temp_raw is not None else None
        return cls(
            backend=backend, model=model, base_url=base_url, api_key=api_key,
            max_tokens=max_tokens, temperature=temperature,
            thinking=overrides.get("thinking"), effort=overrides.get("effort"),
        )

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        return DEFAULT_ANTHROPIC_MODEL if self.backend == "anthropic" else DEFAULT_OPENAI_MODEL


# --------------------------------------------------------------------------- #
# Provider base
# --------------------------------------------------------------------------- #
class Provider:
    """Backend-agnostic LLM provider. Subclasses implement ``complete``."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @property
    def name(self) -> str:
        return self.config.backend

    @property
    def model(self) -> str:
        return self.config.resolved_model()

    def complete(self, messages: list[dict], *, tools: list[dict] | None = None,
                 system: str | None = None, tool_choice: str | dict | None = None,
                 max_tokens: int | None = None) -> LLMResponse:
        raise NotImplementedError

    # ---- message-shaping helpers (the agent builds backend-neutral messages) ----
    @staticmethod
    def tool_result_message(call: ToolCall, content: str) -> dict:
        """A backend-neutral 'tool result' message. Subclasses translate as needed.

        Neutral shape: {"role": "tool", "tool_call_id": id, "name": name, "content": str}.
        """
        return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}


# --------------------------------------------------------------------------- #
# Anthropic backend
# --------------------------------------------------------------------------- #
class AnthropicProvider(Provider):
    """Anthropic Messages API backend (default). Imports the SDK lazily."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        try:
            import anthropic  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise ProviderUnavailable(
                "anthropic backend selected but the 'anthropic' SDK is not importable "
                f"({exc}). Install it (pip install anthropic) or select a local backend "
                "(SCPILOT_LLM_BACKEND=openai SCPILOT_LLM_BASE_URL=http://localhost:11434/v1)."
            ) from exc
        if not config.api_key:
            raise ProviderUnavailable(
                "anthropic backend selected but no API key found. Set ANTHROPIC_API_KEY, "
                "or use a local backend (SCPILOT_LLM_BACKEND=openai with SCPILOT_LLM_BASE_URL)."
            )
        from anthropic import Anthropic
        self._client = Anthropic(api_key=config.api_key)

    def complete(self, messages, *, tools=None, system=None, tool_choice=None,
                 max_tokens=None) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": _to_anthropic_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = _to_anthropic_tool_choice(tool_choice)
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"anthropic completion failed: {type(exc).__name__}: {exc}") from exc
        return _from_anthropic_response(resp)


# --------------------------------------------------------------------------- #
# OpenAI-compatible backend (OpenAI / Ollama / vLLM / LM Studio)
# --------------------------------------------------------------------------- #
class OpenAICompatProvider(Provider):
    """Any OpenAI-compatible /chat/completions endpoint. Imports the SDK lazily."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        try:
            import openai  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise ProviderUnavailable(
                "openai (OpenAI-compatible/local) backend selected but the 'openai' SDK is "
                f"not importable ({exc}). Install it (pip install openai). For a local LLM, "
                "point SCPILOT_LLM_BASE_URL at e.g. http://localhost:11434/v1 (Ollama) or your "
                "vLLM/LM Studio server."
            ) from exc
        from openai import OpenAI
        # local servers usually ignore the key; OpenAI itself needs one. Provide a dummy
        # for keyless local endpoints so the SDK constructor doesn't refuse to build.
        key = config.api_key or "sk-no-key-required"
        client_kwargs: dict = {"api_key": key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._client = OpenAI(**client_kwargs)

    def complete(self, messages, *, tools=None, system=None, tool_choice=None,
                 max_tokens=None) -> LLMResponse:
        oai_messages = _to_openai_messages(messages, system=system)
        kwargs: dict = {
            "model": self.model,
            "messages": oai_messages,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = _to_openai_tool_choice(tool_choice)
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"openai-compatible completion failed: {type(exc).__name__}: {exc}") from exc
        return _from_openai_response(resp)


# --------------------------------------------------------------------------- #
# Factory + availability probe (used by doctor D1)
# --------------------------------------------------------------------------- #
_BACKENDS = {"anthropic": AnthropicProvider, "openai": OpenAICompatProvider}


def build_provider(config: ProviderConfig | None = None, **overrides) -> Provider:
    """Construct the configured provider (raises ProviderUnavailable on a hard gate)."""
    cfg = config or ProviderConfig.from_env(**overrides)
    if cfg.backend not in _BACKENDS:
        raise ProviderUnavailable(
            f"unknown LLM backend '{cfg.backend}'. valid: {sorted(_BACKENDS)} "
            "(set SCPILOT_LLM_BACKEND)."
        )
    return _BACKENDS[cfg.backend](cfg)


def probe_backend(backend: str | None = None) -> dict:
    """Non-fatal capability probe for ``scpilot doctor`` (D1). No network calls."""
    cfg = ProviderConfig.from_env(backend=backend) if backend else ProviderConfig.from_env()
    out: dict = {"backend": cfg.backend, "model": cfg.resolved_model(),
                 "base_url": cfg.base_url, "has_api_key": bool(cfg.api_key)}
    if cfg.backend == "anthropic":
        try:
            import anthropic  # noqa: F401
            out["sdk_present"] = True
        except Exception:  # noqa: BLE001
            out["sdk_present"] = False
        out["ready"] = out["sdk_present"] and out["has_api_key"]
        if not out["sdk_present"]:
            out["reason"] = "anthropic SDK not installed"
        elif not out["has_api_key"]:
            out["reason"] = "ANTHROPIC_API_KEY not set"
    else:  # openai-compatible / local
        try:
            import openai  # noqa: F401
            out["sdk_present"] = True
        except Exception:  # noqa: BLE001
            out["sdk_present"] = False
        # local endpoints don't need a key; OpenAI itself does (we can't know which) →
        # ready iff SDK present and a base_url is configured (local) OR a key is set (OpenAI)
        out["ready"] = out["sdk_present"] and (bool(cfg.base_url) or out["has_api_key"])
        if not out["sdk_present"]:
            out["reason"] = "openai SDK not installed"
        elif not (cfg.base_url or out["has_api_key"]):
            out["reason"] = "set SCPILOT_LLM_BASE_URL (local) or an API key"
    return out


# =========================================================================== #
# Translation helpers — keep the wire-format quirks isolated here.
# =========================================================================== #
# The agent emits backend-neutral messages with these roles:
#   {"role": "user"|"assistant"|"system", "content": str}
#   {"role": "assistant_tool_calls", "tool_calls": [ToolCall...]}   (echoed model turn)
#   {"role": "tool", "tool_call_id", "name", "content"}             (a tool result)

# ---- Anthropic ----
def _to_anthropic_tool(t: dict) -> dict:
    return {"name": t["name"], "description": t.get("description", ""),
            "input_schema": t.get("input_schema") or t.get("parameters")
            or {"type": "object", "properties": {}}}


def _to_anthropic_tool_choice(choice):
    if isinstance(choice, dict):
        return choice
    if choice == "auto":
        return {"type": "auto"}
    if choice == "any":
        return {"type": "any"}
    return {"type": "tool", "name": choice}     # force a specific tool by name


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    pending: list[dict] = []   # buffer consecutive tool results into ONE user message

    def _flush() -> None:
        if pending:
            out.append({"role": "user", "content": list(pending)})
            pending.clear()

    for m in messages:
        role = m["role"]
        if role == "tool":
            # Anthropic requires ALL tool_result blocks for one assistant tool-use turn
            # to live in a SINGLE following user message (multiple consecutive user
            # messages would break the alternation / drop results on multi-tool turns).
            pending.append({"type": "tool_result", "tool_use_id": m["tool_call_id"],
                            "content": m["content"]})
            continue
        _flush()
        if role in ("user", "assistant"):
            out.append({"role": role, "content": m["content"]})
        elif role == "assistant_tool_calls":
            blocks = []
            if m.get("text"):
                blocks.append({"type": "text", "text": m["text"]})
            for tc in m["tool_calls"]:
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name,
                               "input": tc.arguments})
            out.append({"role": "assistant", "content": blocks})
        # system handled separately by the caller
    _flush()
    return out


def _from_anthropic_response(resp) -> LLMResponse:
    text_parts, tool_calls = [], []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name,
                                       arguments=dict(block.input or {})))
    usage = {}
    if getattr(resp, "usage", None) is not None:
        usage = {"input_tokens": getattr(resp.usage, "input_tokens", 0),
                 "output_tokens": getattr(resp.usage, "output_tokens", 0)}
    return LLMResponse(text="".join(text_parts), tool_calls=tool_calls,
                       stop_reason=getattr(resp, "stop_reason", None), usage=usage, raw=resp)


# ---- OpenAI-compatible ----
def _to_openai_tool(t: dict) -> dict:
    return {"type": "function", "function": {
        "name": t["name"], "description": t.get("description", ""),
        "parameters": t.get("input_schema") or t.get("parameters")
        or {"type": "object", "properties": {}}}}


def _to_openai_tool_choice(choice):
    if isinstance(choice, dict):
        return choice
    if choice in ("auto", "none"):
        return choice
    if choice == "any":
        return "required"
    return {"type": "function", "function": {"name": choice}}


def _to_openai_messages(messages: list[dict], *, system: str | None) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        role = m["role"]
        if role in ("user", "assistant", "system"):
            out.append({"role": role, "content": m["content"]})
        elif role == "assistant_tool_calls":
            out.append({"role": "assistant", "content": m.get("text") or None,
                        "tool_calls": [{"id": tc.id, "type": "function",
                                        "function": {"name": tc.name,
                                                     "arguments": json.dumps(tc.arguments)}}
                                       for tc in m["tool_calls"]]})
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                        "content": m["content"]})
    return out


def _from_openai_response(resp) -> LLMResponse:
    choice = resp.choices[0]
    msg = choice.message
    text = msg.content or ""
    tool_calls = []
    for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except (json.JSONDecodeError, TypeError):
            args = {"_raw_arguments": tc.function.arguments}
        # local / OpenAI-compatible servers (Ollama, vLLM, LM Studio) sometimes omit or
        # mangle the tool_call id; synthesize a stable one so the result round-trips.
        tc_id = getattr(tc, "id", None) or f"call_{i}"
        tool_calls.append(ToolCall(id=tc_id, name=tc.function.name, arguments=args))
    usage = {}
    if getattr(resp, "usage", None) is not None:
        usage = {"input_tokens": getattr(resp.usage, "prompt_tokens", 0),
                 "output_tokens": getattr(resp.usage, "completion_tokens", 0)}
    return LLMResponse(text=text, tool_calls=tool_calls,
                       stop_reason=choice.finish_reason, usage=usage, raw=resp)
