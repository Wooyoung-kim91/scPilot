"""LLM layer (mode-2 self-driving CLI) — scpilot plan D1~D5.

Mode 1 (MCP) needs none of this — the host agent provides the LLM. This layer runs
only for ``scpilot run`` when the user has an API key OR a local/OpenAI-compatible
endpoint. Public surface:

- ``provider``: backend abstraction (Anthropic default + OpenAI-compatible/local).
- ``agent``: the tool-runner loop driving ``scpilot.tools`` + decision logging.
- ``prompts``: orchestration / annotation / interpretation / DE-design prompts +
  forced structured-output schemas.
"""

from scpilot.llm.provider import (  # noqa: F401
    LLMResponse, Provider, ProviderConfig, ProviderError, ProviderUnavailable,
    ToolCall, build_provider, probe_backend,
)
