# scPilot

**The rulebook for working on this repo is [`AGENTS.md`](./AGENTS.md) — read it first.**
It is the single, model-neutral source of truth for how any agent (Claude Code, Codex, a
local LLM, or a human) works on scPilot: the evidence-based / no-hardcoding principle, the
reproducibility invariants, the tool contract, the environment/execution protocol, and the
diagnosis/verification procedures.

This file is intentionally thin so Claude Code and other harnesses read the *same* rules.
Do not duplicate rules here — add or change them in `AGENTS.md` (or the topic owner listed
in its §8 single-source-of-truth map).
