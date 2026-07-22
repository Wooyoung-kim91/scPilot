# scPilot

**LLM-driven single-cell RNA-seq analysis pipeline** — exposed as both an **MCP
(stdio) server** (primary integration) and a **self-driving CLI agent**.

scPilot's core idea: **tools return summary statistics, never the data.** Each
analysis step (QC, integration, clustering, annotation, CNV, …) emits a compact
JSON summary; the LLM reasons over those summaries to *decide* the thresholds,
the integration method, the clustering resolution, and the cell-type labels —
while the multi-GB `AnnData` stays server-side. Every run is recorded to a
durable on-disk session, so a crash can resume and the whole analysis can be
**deterministically replayed without the LLM**.

- Plan: [`scpilot_plan.md`](scpilot_plan.md)
- Annotation strategy (single source): [`cancer_scrnaseq_annotation_strategy.md`](cancer_scrnaseq_annotation_strategy.md)
- Vendored reproducibility / IO / figure primitives: [`scpilot/vendor/`](scpilot/vendor/)

---

## Why scPilot

- **Summary-in → decision-out.** Tools emit small JSON (`ToolResult`); the LLM
  never loads the `AnnData`. The reasoning loop stays cheap and the data stays on
  the analysis host.
- **No hardcoded parameters.** Thresholds and resolutions are *suggested from the
  data* (e.g. MAD-based QC cutoffs, a resolution sweep with a knee detector) and
  the *LLM judges the final call* from the evidence — nothing is baked in. A user
  can still **pre-fix any knob** via a preset (see below).
- **Reproducible by construction.** Seeds are pinned per run, every tool run is
  appended to `run_log.jsonl` with a `recipe_hash`, and `scpilot replay` re-runs
  the exact recipe with **no LLM in the loop**, diffing each result under a
  determinism grade.
- **Single chokepoints.** All four drivers (MCP, autonomous agent, deterministic
  `step`, `replay`) route tool-logging through one helper, and AnnData invariants
  (`layers["counts"]` immutable, genes never dropped) are enforced at the single
  checkpoint-write boundary.
- **Capability-gated.** Optional tools (scVI, Harmony, inferCNV, scib) check their
  dependencies via `scpilot doctor` and return a recoverable error instead of
  crashing when a package is missing.
- **Exports as plain scanpy.** Every run is written out as standalone, scPilot-free
  tutorial scripts (`code/NN_<stage>.py`) — direct scanpy/pandas with the exact
  parameters used — so the analysis is readable and re-runnable without scPilot.

---

## Installation

Requires Python ≥ 3.11. Core deps (in `pyproject.toml`): scanpy, anndata,
numpy ≥ 2, scvi-tools, harmonypy, leidenalg, scib-metrics, mcp, anthropic.

### Option A — fresh environment (clone + editable)

```bash
git clone https://github.com/Wooyoung-kim91/scPilot.git
cd scPilot
conda create -n scpilot python=3.11 -y
conda activate scpilot
pip install -e .                  # installs the scientific stack from pyproject
scpilot version
```

Or install straight from GitHub without cloning (non-editable):

```bash
pip install "scpilot @ git+https://github.com/Wooyoung-kim91/scPilot.git"
```

### Option B — into an existing verified env (recommended)

If you already have the numpy-2.x-verified conda env, clone the repo and install
editable **without re-resolving deps** so pip does not upgrade the verified stack:

```bash
git clone https://github.com/Wooyoung-kim91/scPilot.git && cd scPilot
conda run -n scpilot pip install -e . --no-deps
conda run -n scpilot scpilot version
```

Optional CNV + annotation extras (gated at runtime by `doctor`) — CNV scoring
(`infercnvpy`), gene-position mapping (`gtfparse`, `pybiomart`), and CellTypist
reference annotation (`celltypist`):

```bash
pip install -e ".[extra]"         # infercnvpy, gtfparse, pybiomart, celltypist
```

> **Running an exported analysis needs no scPilot install.** Every run also writes
> plain-scanpy tutorial scripts to `<workdir>/code/NN_<stage>.py` that import only
> `scanpy` / `anndata` / `numpy` / `pandas` (+ `harmonypy` / `scib-metrics` for the
> integration & benchmark steps). You can re-run the whole pipeline from those
> scripts in any scientific-Python env — scPilot itself is not required. See
> [Reproducible standalone export](#reproducible-standalone-export).

### Preflight

```bash
conda run -n scpilot scpilot doctor      # deps + per-tool capability flags + smoke test, as JSON
```

`doctor` reports per-tool **capability flags**; the orchestrator must not select a
tool whose capability is `false`.

### LLM credentials (mode 2 only)

The autonomous agent needs an LLM backend. Anthropic by default
(`ANTHROPIC_API_KEY`), or any OpenAI-compatible endpoint — including a **local**
model — via `--backend openai --base-url`. Modes 1/3/4 need no LLM key.

---

## Quickstart example

A minimal end-to-end run on a single `.h5ad`. This assumes the `scpilot` env is
installed (see above) and, for the autonomous agent, that `ANTHROPIC_API_KEY` is
set in your shell.

```bash
# 0) Preflight — confirm deps + per-tool capabilities are green
conda run -n scpilot scpilot doctor

# 1) Autonomous run — the LLM drives QC → integration → clustering → annotation
export ANTHROPIC_API_KEY=sk-...          # your key; needed only for `scpilot run`
conda run -n scpilot scpilot run data.h5ad \
  --tissue "human pancreas, PDAC" \
  --goal  "annotate major + fine cell types, flag malignant cells" \
  --seed 0 \
  -w runs/demo                            # session/working directory

# 2) Inspect the results
ls runs/demo/artifacts/                   # figures (UMAP, dotplots, CNV heatmaps) + CSVs
cat runs/demo/reasoning_log.md            # human-readable narrative of every decision

# 3) Reproduce the exact run with NO LLM in the loop
conda run -n scpilot scpilot replay runs/demo
```

**No API key? Run it deterministically instead** — drive the pipeline step by
step yourself (modes 3/4 need no LLM):

```bash
conda run -n scpilot scpilot step qc_metrics data.h5ad -w runs/demo
conda run -n scpilot scpilot step qc_filter  -w runs/demo -p min_genes=200 -p max_pct_mt=15
conda run -n scpilot scpilot step preprocess -w runs/demo -p n_top_genes=2000 -p n_pcs=30
conda run -n scpilot scpilot step cluster    -w runs/demo -p resolution=0.25
```

**Prefer plain scanpy?** After any run, `runs/demo/code/` holds standalone,
scPilot-free scripts (`00_ingest.py`, `01_qc_metrics.py`, …) that reproduce the
whole pipeline with direct scanpy/pandas — run them in order in any scientific-
Python env (see [Reproducible standalone export](#reproducible-standalone-export)).

---

## The analysis method

The autonomous pipeline (`scpilot run`) follows the cancer-scRNAseq annotation
strategy in tiers. The LLM walks this sequence, reading each tool's JSON summary
to set the next step's parameters from the evidence:

| Stage | Tools | What the LLM decides from the summary |
|---|---|---|
| **Tier 0 — QC** | `qc_metrics` → `qc_filter` | MAD-suggested cutoffs (`n_mads`); keeps/relaxes per the distribution + doublet rate. Before/after violin + scatter plots justify the call. |
| **Embedding** | `preprocess` (HVG → PCA) | `n_top_genes`, `n_pcs` from the variance curve (HVG + PCA-variance plots). |
| **Clustering** | `cluster_sweep` → `cluster` | Sweeps resolution 0.1–0.5; picks the value **just before clusters suddenly multiply** (knee, `jump_ratio`). |
| **Integration** | `integrate_harmony` / `integrate_scvi` → `benchmark` | Runs candidate embeddings, then scIB-benchmarks them and picks the **best reduction** for downstream work. |
| **Tier 1 — broad** | `annotation_review` → `apply_annotation` → `harmonize_annotations` | Top-50 DE per cluster (expression + specificity) → LLM combines a ≥3-gene marker set per broad type → harmonizes labels across reductions. Broad-type UMAP + family-contiguous dotplot. |
| **Tier 2 — subtype** | `compartment_plan` → `compartment_subset` → `fine_annotation_review` → `apply_fine_annotation` → `merge_fine_annotations` | Re-clusters each compartment on the best reduction; fine subtypes with **FACS-like names** as the primary label. |
| **Malignancy (CNV)** | `annotate_genomic_positions` → `cnv_score` → `malignancy_evidence` → `apply_malignancy` | inferCNV with immune cells as reference; derives `cnv_status` (tumor vs normal) + CNV UMAP/heatmaps. |
| **Finalize + report** | `finalize_annotation` → `report` | Consolidated FACS-like `final_annotation` (Malignant-prefixed where applicable) + a figures-and-interpretation report. |

Every threshold above is **data-driven, not hardcoded** — but any of them can be
fixed in advance via a parameter preset (next section).

---

## Pre-selecting parameters (catalog + presets)

Before launching a run you can list every tunable knob and **fix only the ones
you care about**; the rest stay dynamically chosen by the LLM. This generalizes
the human-in-the-loop `--resolution` override to every catalogued parameter.

```bash
# 1) See the tunable knobs (+ defaults). Unset = chosen dynamically by the LLM.
conda run -n scpilot scpilot params
conda run -n scpilot scpilot params --json            # machine-readable

# 2) Generate a fillable preset; uncomment + set only what you want to FIX.
conda run -n scpilot scpilot params --template preset.yaml
```

`preset.yaml` — fix only chosen knobs:

```yaml
qc_metrics:
  n_mads: 3            # stricter QC than the lenient default (5)
preprocess:
  n_top_genes: 3000
cluster_sweep:
  res_max: 0.8         # widen the resolution sweep
```

```bash
# 3) Run with the preset — fixed values override the LLM for those knobs;
#    everything else stays dynamic. The fixed set is echoed + recorded for replay.
conda run -n scpilot scpilot run data.h5ad --param-file preset.yaml \
  --tissue "human pancreas, PDAC"
```

Fixed values are recorded in `run_log.jsonl` / `outputs.jsonl`, so a preset run is
just as reproducible as a fully autonomous one.

---

## Run modes

scPilot has four run modes plus the preflight:

| Command | Mode | What it does |
|---|---|---|
| `scpilot mcp` | 1 — MCP server | stdio server exposing every tool (primary integration) |
| `scpilot run <input.h5ad>` | 2 — autonomous agent | LLM drives the full pipeline end-to-end |
| `scpilot step <stage> <input.h5ad>` | 3 — deterministic | run one tool, no LLM (debug / regression) |
| `scpilot replay <session>` | 4 — replay | re-run a session's recipe with no LLM, diff results |
| `scpilot doctor` / `scpilot params` | preflight | environment / capability report · tunable-knob catalog |

### Mode 1 — MCP server (primary)

```bash
conda run -n scpilot scpilot mcp        # stdout carries ONLY MCP protocol JSON
```

Connect from an MCP host (Claude Code, Codex CLI, …). Each tool takes `input`
(absolute `.h5ad` path), an optional `workdir` (session directory), a `params`
dict, and a `seed` (pinned per call). A tool run returns its `ToolResult` JSON and
records the step to the session for replay.

By default every registered tool is exposed (the host is a trusted local process).
To restrict the surface, gate by tool name with env vars:

```bash
SCPILOT_MCP_ENABLE_TOOLS=qc_metrics,qc_filter,preprocess scpilot mcp   # allowlist
SCPILOT_MCP_DISABLE_TOOLS=train_scvi,cnv_score scpilot mcp             # denylist
```

### Mode 2 — autonomous agent

```bash
conda run -n scpilot scpilot run data.h5ad \
  --tissue "human pancreas, PDAC" \
  --goal "annotate major + fine cell types, flag malignant cells" \
  --param-file preset.yaml \              # optional: fix chosen knobs
  --backend anthropic                     # or: --backend openai --base-url http://localhost:11434/v1
```

The agent reads each tool's JSON summary, chooses the next tool + params, and
writes a final report. Every tool run and consequential decision is logged so the
session replays with **no LLM**. Control the run with `--seed`, `--max-iters`,
`--model`, `--effort`, `--resolution`.

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
conda run -n scpilot scpilot replay runs/demo --dry-run     # validate / list only
```

Re-runs every recorded tool with its recorded params on a fresh session and diffs
each summary against the original under its determinism grade
(A = exact, B = structural ± tolerance, C = bit-identical). Exit code is non-zero
on any mismatch. Replay also re-checks the AnnData invariants per step and
cross-checks that the raw counts layer reproduced identically.

---

## Pipeline tools

QC & embedding: `ingest` · `load` · `detect_state` · `qc_metrics` · `qc_filter` ·
`preprocess` · `cluster` · `cluster_sweep` · `markers` · `plots`

Integration & benchmark: `integrate_scvi` · `train_scvi` · `integrate_harmony` ·
`benchmark`

Annotation (Tier 1–2): `annotation_review` · `apply_annotation` ·
`consensus_annotation` · `harmonize_annotations` · `compartment_plan` ·
`compartment_subset` · `fine_annotation_review` · `apply_fine_annotation` ·
`merge_fine_annotations` · `finalize_annotation`

Malignancy (CNV): `annotate_genomic_positions` · `cnv_score` ·
`malignancy_evidence` · `apply_malignancy`

Reporting: `report` · `export_final`

---

## Session layout

A session is a working directory that owns the analysis state on disk:

```
<workdir>/
  session.json          # manifest (id, x_state, checkpoints[], stage, …)
  run_log.jsonl         # append-only: one record per tool run (params, summary, seed, recipe_hash)
  decisions.jsonl       # append-only: LLM decision events (frozen schema)
  outputs.jsonl         # append-only: every artifact (figure/CSV) + its provenance + reasoning
  reasoning_log.md      # human-readable narrative (one section per step + plots)
  checkpoints/NN_<stage>.h5ad
  artifacts/            # CSV / PNG outputs
  code/                 # auto-generated STANDALONE per-step scripts NN_<stage>.py (plain scanpy, no scpilot)
  standalone_data/      # the h5ad chain the code/ scripts read & write (created when you run them)
  logs/
```

The in-memory AnnData is just a cache of the latest checkpoint, so any `step`
(a fresh process) resumes from on-disk state.

---

## Reproducible standalone export

Every session writes its pipeline as **standalone, scPilot-free tutorial scripts**
under `<workdir>/code/` — one numbered file per step (`00_ingest.py`,
`01_qc_metrics.py`, `02_qc_filter.py`, …):

- **Plain scanpy/pandas.** Each script is the actual operation written out directly
  — no `tools.run`, no `Session`, no scPilot import — with the **exact parameter
  values used in the run** baked in (e.g. `keep = (adata.obs["n_genes_by_counts"] >= 300) & …`).
- **Chained via h5ad.** Step *N* reads `standalone_data/<N-1>_<stage>.h5ad` and writes
  `standalone_data/<N>_<stage>.h5ad`; run the `NN_*.py` files **in order** in any
  scientific-Python env. Non-mutating evidence steps (`annotation_review`,
  `annotation_audit`, `benchmark`, …) also drop a sidecar JSON/CSV next to the h5ad.
- **Verified equivalent.** `tests/test_scriptgen_equivalence.py` runs each generated
  script as a real subprocess (with scPilot import *blocked*) and asserts its result
  matches the scPilot tool — equivalence by regression test, not by trust.

```bash
cd <workdir>
python code/00_ingest.py        # reads the dataset profile → standalone_data/00_ingest.h5ad
python code/01_qc_metrics.py    # → 01_qc_metrics.h5ad (+ suggested MAD cutoffs)
python code/02_qc_filter.py     # → 02_qc_filter.h5ad … run the rest in order
```

This turns a run into a readable, auditable, ordinary scanpy tutorial — independent
of scPilot, and runnable in any env with just the scientific stack.

---

## Reproducibility harness

- **Seed control** — `set_global_seed` pins numpy/random/torch/scvi; every driver
  (incl. the MCP server) pins per run and records the seed.
- **Run log** — each tool run is a `RunLogRecord` with `params`, a structural
  `summary`, `seed`, `lib_versions`, and a `recipe_hash` (params + libs + input +
  data fingerprint) for drift detection.
- **Output provenance** — each figure/CSV is logged to `outputs.jsonl` with the
  tool, params, and per-output reasoning that produced it.
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
driver-parity, replay round-trip, parameter-catalog, and invariant-violation
regression tests.

---

## License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute
with attribution. © 2026 Wooyoung Kim.
