# scpilot

**LLM-driven single-cell RNA-seq analysis pipeline**, exposed as both an **MCP
(stdio) server** (primary) and a **self-driving CLI agent**.

Tools return *summary statistics*, never the data — the LLM reasons over those
compact JSON summaries to decide QC thresholds, integration method, clustering
resolution, and cell-type annotation, while the (multi-GB) AnnData stays
server-side. Every run is recorded to a durable on-disk session so a crash can
resume and the whole analysis can be **deterministically replayed without the LLM**.

- Plan: [`scpilot_plan.md`](scpilot_plan.md)
- Annotation strategy (single source): [`cancer_scrnaseq_annotation_strategy.md`](cancer_scrnaseq_annotation_strategy.md)
- Vendored reproducibility/IO/figure primitives: [`scpilot/vendor/`](scpilot/vendor/)

---

## Why scpilot

- **Summary-in → decision-out.** Tools emit small JSON (`ToolResult`); the LLM
  never loads the AnnData. This keeps the reasoning loop cheap and the data on
  the analysis host.
- **Reproducible by construction.** Seeds are pinned, every tool run is appended
  to `run_log.jsonl` with a `recipe_hash`, and `scpilot replay` re-runs the exact
  recipe with **no LLM in the loop**, diffing each result under a determinism grade.
- **Single chokepoints.** All four drivers (MCP, autonomous agent, deterministic
  `step`, `replay`) route tool-logging through one helper, and AnnData invariants
  (`layers["counts"]` immutable, genes never dropped) are enforced at the single
  checkpoint write boundary.
- **Capability-gated.** Optional tools (scVI, Harmony, inferCNV, scib) check their
  dependencies via `scpilot doctor` and return a recoverable error instead of
  crashing when a package is missing.

---

## Installation (dev)

The scientific stack is provided by the conda env **`scpilot`**. Install the
package editable **without touching env deps** (so pip does not re-resolve the
verified numpy-2.x stack):

```bash
conda run -n scpilot pip install -e . --no-deps
conda run -n scpilot scpilot version
```

Optional Tier-2/3 + trajectory extras (gated at runtime by `doctor`):

```bash
conda run -n scpilot pip install -e ".[extra]" --no-deps   # infercnvpy, gtfparse, pybiomart, celltypist
```

Requires Python ≥ 3.11. Core deps (documented in `pyproject.toml`): scanpy,
anndata, numpy ≥ 2, scvi-tools, harmonypy, leidenalg, scib-metrics, mcp, anthropic.

### Preflight

```bash
conda run -n scpilot scpilot doctor      # deps + capability flags + smoke test, as JSON
```

`doctor` reports per-tool **capability flags**; the orchestrator must not select a
tool whose capability is `false`.

---

## Usage

scpilot has four run modes plus the preflight:

| Command | Mode | What it does |
|---|---|---|
| `scpilot mcp` | 1 — MCP server | stdio server exposing every tool (primary integration) |
| `scpilot run <input.h5ad>` | 2 — autonomous agent | LLM drives the full pipeline end-to-end |
| `scpilot step <stage> <input.h5ad>` | 3 — deterministic | run one tool, no LLM (debug / regression) |
| `scpilot replay <session>` | 4 — replay | re-run a session's recipe with no LLM, diff results |
| `scpilot doctor` | preflight | environment / capability report |

### Mode 1 — MCP server (primary)

```bash
conda run -n scpilot scpilot mcp        # stdout carries ONLY MCP protocol JSON
```

Connect from an MCP host (Claude Code, Codex CLI, …). Each tool takes
`input` (absolute `.h5ad` path), an optional `workdir` (session directory), a
`params` dict, and a `seed` (pinned per call). A tool run returns its `ToolResult`
JSON and records the step to the session for replay.

### Mode 2 — autonomous agent

```bash
conda run -n scpilot scpilot run data.h5ad \
  --tissue "human pancreas, PDAC" \
  --goal "annotate major + fine cell types, flag malignant cells" \
  --resolution 0.25 \
  --backend anthropic         # or: --backend openai --base-url http://localhost:11434/v1
```

The agent reads each tool's JSON summary, chooses the next tool + params, and
writes a final report (figures + interpretation). Every tool run and consequential
decision is logged so the session replays with **no LLM**. Use `--seed`,
`--max-iters`, `--model`, `--effort` to control the run.

### Mode 3 — deterministic single step (no LLM)

```bash
# entry step gives the input; later steps resume from the session checkpoints
conda run -n scpilot scpilot step qc_metrics data.h5ad -w runs/demo -p run_scrublet=false
conda run -n scpilot scpilot step qc_filter   -w runs/demo -p min_genes=200 -p max_pct_mt=15
conda run -n scpilot scpilot step preprocess  -w runs/demo -p n_top_genes=2000 -p n_pcs=30
conda run -n scpilot scpilot step cluster     -w runs/demo -p resolution=0.25
```

Each `step` prints the `ToolResult` JSON and writes a checkpoint. `-p k=v`
(repeatable) passes tool params; `--seed` pins RNGs.

### Mode 4 — replay (reproduce a session, no LLM)

```bash
conda run -n scpilot scpilot replay runs/demo               # re-execute + diff
conda run -n scpilot scpilot replay runs/demo --dry-run     # validate/list only
```

Re-runs every recorded tool with its recorded params on a fresh session and diffs
each summary against the original under its determinism grade
(A = exact, B = structural ± tolerance, C = bit-identical). Exit code is non-zero
on any mismatch. Replay also re-checks the AnnData invariants per step and
cross-checks that the raw counts layer reproduced identically.

---

## Pipeline tools

QC & embedding: `ingest` · `load` · `detect_state` · `qc_metrics` · `qc_filter` ·
`preprocess` · `cluster` · `markers` · `plots`

Integration & benchmark: `integrate_scvi` · `train_scvi` · `integrate_harmony` ·
`benchmark`

Annotation (Tier 1–3): `annotation_review` · `apply_annotation` ·
`consensus_annotation` · `compartment_plan` · `compartment_subset` ·
`fine_annotation_review` · `apply_fine_annotation` · `merge_fine_annotations`

Malignancy (Tier 2, CNV): `annotate_genomic_positions` · `cnv_score` ·
`malignancy_evidence` · `apply_malignancy`

Reporting: `report`

---

## Session layout

A session is a working directory that owns the analysis state on disk:

```
<workdir>/
  session.json          # manifest (id, x_state, checkpoints[], stage, …)
  run_log.jsonl         # append-only: one record per tool run (params, summary, seed, recipe_hash)
  decisions.jsonl       # append-only: LLM decision events (frozen schema)
  reasoning_log.md      # human-readable narrative (one section per step + plots)
  checkpoints/NN_<stage>.h5ad
  artifacts/            # CSV / PNG outputs
  code/                 # auto-generated runnable pipeline.py + cell-by-cell notebook + pinned source snapshot
  logs/
```

The in-memory AnnData is just a cache of the latest checkpoint, so any `step`
(a fresh process) resumes from on-disk state.

---

## Reproducibility harness

- **Seed control** — `set_global_seed` pins numpy/random/torch/scvi; every driver
  (incl. the MCP server) pins per run and records the seed.
- **Run log** — each tool run is a `RunLogRecord` with `params`, structural-invariant
  `summary`, `seed`, `lib_versions`, and a `recipe_hash` (params + libs + input +
  data fingerprint) for drift detection.
- **Invariants** — enforced at `Session.checkpoint()`: `layers["counts"]` content
  never changes and genes are never dropped (cells may shrink via filtering).
- **Replay** — `scpilot replay` consumes the recorded recipe to reproduce a session
  with no LLM, surfacing any forced-LLM structured outputs it cannot re-derive.

---

## Testing

```bash
conda run -n scpilot python -m pytest tests/ -q
```

Each tool has unit + structural-invariant tests on a tiny fixture; the harness has
driver-parity, replay round-trip, and invariant-violation regression tests.
