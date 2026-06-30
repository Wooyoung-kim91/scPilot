# scPilot — Agent Rulebook (model-agnostic)

**This file is the single, tool-neutral source of truth for *how* any agent works on
scPilot.** Read it first, whatever runs you: Claude Code, Codex, a local LLM, or a human.
It is plain Markdown with no harness-specific assumptions — a local model can paste it
straight into a system prompt.

Why this exists: the same task given to different models/harnesses was diverging mainly
because each one started from a *different* set of rules and read a *different* slice of
the repo. The divergence we want to keep is genuine reasoning difference; the divergence
we want to kill is "they were working from different facts." This document fixes the facts
and the procedure. Everything below is either an **INVARIANT** (always true, never break it
without an explicit decision) or a **PROTOCOL** (the procedure to follow so two agents
gather the same evidence).

> Entry points point here. `CLAUDE.md` (Claude Code) and this `AGENTS.md` (Codex and most
> other CLIs) should stay thin and defer to this file so all models read one rulebook.
> The deep domain docs listed in §9 each own one topic — update the owner, not a copy.

---

## 1. Core principle — evidence-based, NEVER hardcoded  *(INVARIANT)*

scPilot is meant to generalize across datasets, tissues, and organisms. The defining rule:

- **Deterministic tools emit EVIDENCE; the biological CALL is a judgment.** A tool returns
  DE results, CNV burden, reference contrasts, cross-tabs, distribution summaries — it does
  **not** assign a final cell type / `malignancy` / cutoff from a lone threshold. The call is
  made by the reasoning layer (LLM) or a clearly-justified, evidence-backed rule.
  *Example:* `cnv_score` returns per-cluster CNV burden + reference contrast but deliberately
  does **not** set `obs["malignancy"]`.
- **No hardcoded marker panels, tissue expectations, or fixed score thresholds.** Past commits
  explicitly removed these because they bias every call toward the developer's example tissue
  (PDAC) and defeat the LLM-judgment design. Marker sets used for plotting/annotation are
  **derived per dataset** from that dataset's own DE + recorded labels, not from a fixed list.
  Where a tool still needs a panel (the opt-in `annotate_broad`) or a gene pair (the opt-in
  mixed-lineage flag in `qc_metrics`), it is **caller/agent-supplied** (param defaults to None,
  no built-in biology); without it the tool errors and routes to the marker-DB-free path rather
  than silently applying a developer's panel. Any module-level example panel is named
  `EXAMPLE_*` and is never applied as a default.
- **Never assume human.** Organism is **detected from the data** (`scpilot/core/_species.py`:
  gene-symbol casing + mito-gene style), reported as evidence, and used to pick `MT-`/`mt-`
  and to resolve reference symbols to the data's own casing (`EPCAM`→`Epcam`). Cross-organism
  symbol lookups are case-insensitive *normalizations*, not species assumptions.

If you are tempted to write a literal gene name, tissue name, or magic threshold into a
deterministic tool: stop. Emit the evidence instead and let the caller decide.

---

## 2. Reproducibility — the harness  *(INVARIANT)*

Reproducibility is non-negotiable and is enforced at **two single chokepoints**. Extend the
harness *there*, never per-tool or per-driver:

1. **Run logging → `Session.record_run` / `record_tool_run`.** All four drivers (CLI `step`,
   CLI `run` reporting, `mcp_server.py` handler, `llm/agent.py`) go through these, so records
   cannot diverge. They write the append-only `run_log.jsonl` (+ `recipe_hash` + seed +
   library versions), and `record_tool_run` additionally emits auto-plots, reasoning, and the
   `outputs.jsonl` index (step→params→artifacts(+sha256)→reasoning→recipe_hash). New
   driver/field → change it *here only*.
2. **Invariants → `Session.checkpoint()`** calls `assert_invariants` right before every write
   (`enforce_invariants=True` by default). Every mutating tool is protected automatically — do
   not call it per-tool. Stages that first *create* counts (ingest/load) use the
   `enforce_invariants=False` escape hatch.

Further invariants:

- **Seed every call.** `set_global_seed(seed)` is pinned at the start of each tool invocation
  in *all* modes (CLI and MCP), so mode-1 is as reproducible as the CLI.
- **Checkpoints are content-addressed**, the decision-event schema is **frozen**
  (`RunLogRecord` is immutable), and `scpilot replay` re-runs deterministically, re-checking
  invariants each step and cross-verifying `counts` content hashes. Forced LLM structured
  outputs that replay can't re-execute are surfaced explicitly
  (`structured_decisions_not_reexecuted`) so a green "all match" is never misread.
- **Generated notebooks must reproduce cell-by-cell.** `code/pipeline_notebook.py` must give
  the *same* result run cell-by-cell in Jupyter, not just "Restart & Run All". Each step cell
  re-pins its own recorded seed on its first line (`set_global_seed(<step_seed>)`) — never a
  single top-level seed. Tools must take an explicit `random_state` and not depend on
  accumulated global-RNG order. Regression: `tests/test_harness_chokepoints.py::
  test_generated_notebook_reproduces_cell_by_cell`.

---

## 3. Tool & analysis contract  *(INVARIANT + PROTOCOL)*

- **Shape:** a core tool is AnnData-in / summary-dict-out. It returns a structured
  `ToolResult` (`scpilot/schemas.py`): `summary`, `tables`, `artifacts`, `warnings`,
  `checkpoint`, `determinism_grade`, structured `error`/`error_code` (recoverable flag).
  Mutating tools checkpoint; non-mutating tools (annotation_review, benchmark, report) still
  log via `record_run` and appear in the regenerated pipeline.
- **Dynamic parameters = evidence-out + judge** (never a hardcoded constant):
  - `n_pcs` ← variance-ratio elbow.
  - QC cutoffs ← `qc_metrics` returns per-batch + global `suggested_cutoffs`
    (median ± `n_mads`·MAD, default `n_mads=5`); the caller decides.
  - clustering `resolution` ← `cluster_sweep` sweeps 0.1–0.5 and suggests the knee (just
    before n_clusters jumps); the caller picks, then calls `cluster`.
  - markers ← cell-type marker DE is **fixed to Wilcoxon** (`rank_genes_groups`, `pts=True`); the
    method is NOT a parameter (Wilcoxon is the agreed standard for marker genes). Large-data
    tractability is handled by `max_genes_ranked` (output cap, default 5000), not by switching to
    a faster-but-weaker test. Pass `max_genes_ranked=None` only for a genuine full ranking; caps
    are recorded in `summary`/artifact meta (`csv_is_full_ranking`).
- **Capability gating:** optional/heavy deps (scVI, harmony, infercnvpy, celltypist, cellhint…)
  are probed via `doctor.check_capability` and return a structured `capability_unavailable`
  recoverable error — never a raw `ImportError`.
- **Warn, never silently change behavior.** Any auto-fallback (e.g. batch-aware HVG → global
  HVG on a singular per-batch loess, or a capped ranking) must be recorded in `warnings`.

---

## 4. Driving the pipeline — model-agnostic orchestration  *(INVARIANT)*

Any LLM that drives scpilot (Claude Code, Codex, a local model over MCP, or the autonomous
mode-2 loop) follows the **same canonical pipeline**, delivered from one single source
(`scpilot/llm/prompts.py`: `ORCHESTRATION_PROMPT` + `full_workflow_guidance()`), never a
per-model copy:

- The MCP server ships a concise orchestration brief (`MCP_INSTRUCTIONS`) in the `initialize`
  handshake — every compliant client surfaces it to its model, so guidance is not Claude-only.
- The full canonical flow is fetchable over **every** MCP channel so no client is left out:
  the `scpilot_workflow` **prompt**, the `scpilot://workflow` **resource**, and the
  `scpilot_guidance` **tool** (for clients whose LLM only calls tools). All three return the
  same single-source text.
- The pipeline spine: `detect_state` → `qc_metrics`/`qc_filter` → `preprocess` →
  `cluster_sweep`/`cluster`/`markers` → **marker-DB-free Tier-1** (`annotation_review` →
  `apply_annotation`) → integration + **per-embedding** annotation → `harmonize`/`benchmark` →
  pick best reduction → **Tier-2 subtype** (compartment subset) → **CNV/malignancy** (tumor
  only) → `finalize_annotation` → **Tier-4 consistency audit** (`annotation_audit` →
  independent critique → `apply_annotation_audit`) → `report`. Resolution is chosen per
  embedding from `cluster_sweep` (knee), not fixed. Each tool's `suggested_next_tools` nudges
  the next call.
- **Tier-4 = a bounded annotation+verification LOOP, not self-approval.** After annotation, the
  deterministic `annotation_audit` emits the seven inconsistency checks as evidence — including
  validating each claimed marker against the standard cell-type bar (pct ≥ `min_pct`, logFC ≥
  `min_lfc`, padj < `padj_max`) — plus profile collisions, hierarchy triples, single-patient/batch
  dominance, doublet/stress, and malignancy-without-CNV. An **independent** reviewer then
  *adversarially tries to refute* each label (verdict ∈ {confirmed, suspect, refuted}).
  - The reviewer is a **pure critic**: it flags WHETHER a label holds and **must give the REASON**
    (cited evidence), but it **never proposes the replacement cell type**. Refuted clusters are
    re-inferred **independently** by the annotator (told the rejection reason, not an answer).
  - The loop runs `audit → critique → re-annotate refuted → re-finalize → re-audit` until nothing
    is refuted (converged) or a **round cap** (default 3) — the cap guarantees termination and
    keeps the run reproducible. Still-refuted labels remain `review_required` for a human.
  - **Verification fires at EVERY annotation stage, not just the end.** In mode-2 `scpilot run`,
    `run_agent` runs an independent review right after each annotation-apply tool — broad
    (`apply_annotation`, auto-corrected by the bounded loop), fine (`apply_fine_annotation`), and
    final (`finalize_annotation`) (fine/final verify + flag). In mode-1 the host does the same per
    step 5/step 11 of the workflow. So a wrong broad call is caught before integration/subtype build
    on it, not only at the final consolidation.
  - Prefer a **different model** for the review (mode-2: `--reviewer-model` / `--review-max-rounds`;
    mode-1: the host delegates to a second agent) — annotator↔reviewer disagreement is exactly the
    signal a human should look at.
- Contract for the driver: summary-in → decision-out; state candidates + choice + rationale
  before every non-trivial choice (recorded as a decision event); one tool at a time.

## 5. Annotation & plotting conventions  *(pointer — deep source in §9)*

The biology (Tier scheme, marker strategy, label sets, FACS naming) lives in the annotation
single source — do **not** restate or override it here. The stable cross-cutting rules:

- **Tier scheme:** Tier 1 = broad, **Tier 2 = subtype/fine**; malignancy/CNV is a *de-tiered*
  "Malignancy (CNV) track" (tumor only). Use these numbers consistently.
- **FACS-like naming is primary** for Tier-2/subtype and the final annotation label.
- **Dotplot markers are derived, not a fixed panel:** scored per reduction from that
  reduction's own DE + recorded cluster→celltype labels (weight a gene by Σ over its clusters
  of `n_cells·(TOPN − DE_rank)` so the dominant cluster's canonical markers win), deduped
  across types, top-K per type. The y-axis follows the panel declaration order (staircase),
  and `dot_clipped` + `staircase` invariants are checked on every render.

---

## 6. Environment & execution  *(PROTOCOL — do this exactly)*

- **Run everything with the `scpilot` conda env interpreter, not `base`.** Base lacks pytest
  and the scientific stack.
  - Python / any scpilot import or run: `/home/wykim/miniforge3/envs/scpilot/bin/python ...`
  - Tests: `/home/wykim/miniforge3/envs/scpilot/bin/python -m pytest -q`
- **Set `NUMBA_CACHE_DIR` to a writable path** (e.g. `export NUMBA_CACHE_DIR=/tmp/numba-cache`)
  or scanpy import can fail on numba cache permissions.
- Do **not** use `conda run -n scpilot` with output capture — it was found to break the MCP
  stdio path. Use the direct env binary path.
- **Large `.h5ad` (multi-GB): never load fully for a quick check.** Read `var` with
  `backed='r'` for symbol-only inspection.
- One MCP session per input file: the working dir defaults to `<input_stem>_scpilot_session`
  so unrelated sessions are not silently reused.

---

## 7. Diagnosis protocol — BEFORE you conclude or change anything  *(PROTOCOL)*

This is the step that makes two agents reach the same evidence. When investigating an error,
regression, or "why did the pipeline do X":

1. **Read the run's own record first**, in this order: `run_log.jsonl` (what ran, with which
   params/seed) → `outputs.jsonl` (artifacts + hashes) → `reasoning_log.md` (why the agent
   chose what it chose) → the specific core tool in `scpilot/core/` that emitted the result.
2. **Reproduce with the env binary** (§6) before theorizing. State whether you actually ran it.
3. **Verify against current code, not memory.** Any remembered fact, file path, line number,
   or "this tool does X" is point-in-time and may be stale — confirm in the live source before
   asserting it.
4. **Separate root cause from symptom**, and name the layer: is it a deterministic-tool bug, a
   harness/record issue, a reasoning-layer call, an environment problem, or a data-quality
   issue in the input? Say which.

---

## 8. Verification protocol — BEFORE you call it done  *(PROTOCOL)*

- Run the affected tests with the env binary (§6) and report the real result; if something
  fails or was skipped, say so with the output.
- For changes touching large-data behavior, re-run on real data with caps/light methods (e.g.
  markers with `max_genes_ranked` set and `method="t-test_overestim_var"`) and confirm it
  completes — do not claim a large-data fix verified by a tiny fixture alone.
- Commit/push only when asked. Surface anything that contradicts how a thing was described
  (e.g. a file you were told is empty but isn't) instead of proceeding.

---

## 9. Single-source-of-truth map  *(where each topic is OWNED — update the owner, not a copy)*

| Topic | Owner document |
|---|---|
| How agents work (this rulebook) | `AGENTS.md` (this file) + thin `CLAUDE.md` pointer |
| Annotation biology: Tier design, marker strategy, label sets, FACS naming | `cancer_scrnaseq_annotation_strategy.md` (repo root) |
| Build plan / phase roadmap | `scpilot_plan.md` |
| Orchestration / canonical pipeline (which tool next, how to read summaries) | `scpilot/llm/prompts.py` (`ORCHESTRATION_PROMPT` + `full_workflow_guidance()`), mirrored from `.claude/agents/scrna-analyst.md`; delivered model-agnostically by the MCP server (instructions + `scpilot_workflow` prompt/resource + `scpilot_guidance` tool — see §4) |
| Tool contract & result schema | `scpilot/schemas.py`, `scpilot/tools.py` |
| Reproducibility harness | `scpilot/session.py`, `scpilot/repro.py`, `tests/test_harness_chokepoints.py` |
| Organism/symbol resolution | `scpilot/core/_species.py` |
| Current issues & resolution status | `scpilot_issue_resolution_plan.md`, `scpilot_codex_current_issues.md`, `error_log.md` |

When two agents disagree after both have followed §1–§8, the disagreement is a genuine
judgment split — that is the signal worth a human's attention, and it should be raised, not
silently resolved by whichever agent writes last.
