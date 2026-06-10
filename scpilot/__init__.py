"""scpilot — LLM-driven scRNA-seq analysis (MCP server + CLI agent).

The package exposes pure analysis functions (``scpilot.core``) that take an
AnnData and return a *summary dict*; a tool registry (``scpilot.tools``) wraps
them; an MCP server (``scpilot.mcp_server``) and a Typer CLI (``scpilot.cli``)
both drive that single registry. The LLM never sees the matrix — only the
summaries tools return. See ``scpilot_plan.md`` (repo root) for the full plan.

Reproducibility/IO/figure primitives are vendored from scqc_pipeline under
``scpilot.vendor`` (see ``scpilot/vendor/VENDORING.md``).
"""

__version__ = "0.0.1"
