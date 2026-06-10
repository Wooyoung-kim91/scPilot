# scpilot

LLM-driven scRNA-seq analysis pipeline, exposed as both an **MCP (stdio) server**
(primary) and a **self-driving CLI agent**. Tools return *summary statistics*, not
data — the LLM reasons over those summaries to decide thresholds, integration
method, clustering resolution, and annotation, while the AnnData stays server-side.

- Plan: [`scpilot_plan.md`](scpilot_plan.md)
- Annotation strategy (single source): [`cancer_scrnaseq_annotation_strategy.md`](cancer_scrnaseq_annotation_strategy.md)
- Vendored reproducibility/IO/figure primitives: [`scpilot/vendor/`](scpilot/vendor/VENDORING.md)

## Install (dev)

The scientific stack is provided by the conda env `scpilot`. Install editable
**without touching env deps**:

```bash
conda run -n scpilot pip install -e . --no-deps
scpilot version
```

## Status

Phase A1 (scaffolding + scqc vendoring). Most tools are skeletons — see the
To-Do list in `scpilot_plan.md`.
