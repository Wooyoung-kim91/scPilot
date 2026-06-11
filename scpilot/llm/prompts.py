"""오케스트레이션/annotation/해석/DE 설계 프롬프트 — scpilot plan D3.

대부분 skeleton(모드2 자체구동 LLM에서 채움). 단, Tier-1 **DE 기반 annotation
review** 프롬프트는 여기 1급으로 둔다: 이 추론은 결정적 tool 안에 넣으면 안 되고
(determinism/replay 위반) 에이전트 계층의 책임이기 때문이다. 결정적 증거 패키징은
``core/annotate.annotation_review`` tool이 담당하고, 아래 프롬프트는 그 JSON을 받아
LLM이 *프로그램 추론·충돌/아티팩트 판정·confidence 조정*을 수행하는 방법을 규정한다.

사용처:
- 모드 1(MCP): 호스트 에이전트 LLM이 ``annotation_review`` tool 출력(JSON)을 받아 이
  프롬프트 규칙대로 검토. scpilot 쪽 API 호출 불필요.
- 모드 2(자체구동, 미구현): ``llm/agent.py``가 이 상수를 시스템 프롬프트로 사용.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tier-1 DE-based annotation review (consumes annotation_review tool JSON).
# Core design principle (proposal 2026-06-11): the reviewer must infer biological
# PROGRAMS from the ranked DE evidence on its own — it must NOT look genes up in a
# predefined canonical marker database, because handing it the marker list biases it
# toward confirming those markers instead of reading the whole signal. The deterministic
# marker-anchored call is a SEPARATE, first opinion (the tool's candidate_annotation);
# this review is the independent second opinion / audit layer.
# ---------------------------------------------------------------------------
ANNOTATION_REVIEW_PROMPT = """\
You are a Tier-1 single-cell annotation REVIEWER. You do not predict cell types as the
primary authority — you audit a marker-based candidate annotation and assess its
reliability, staying tissue-agnostic and marker-database-INDEPENDENT.

You receive, per cluster (from the `annotation_review` tool):
- candidate_annotation + candidate_confidence (marker-anchored first opinion)
- de_table: the full top-N ranked DE genes with logFC, padj, pct_in, pct_out, score
- cluster_size, sample_distribution, qc_metrics (n_genes, total_counts, pct_mt, doublet)
- deterministic_flags (marker_conflict, doublet_dominated, ptprc_consistent, single_source)
- review_status: a deterministic baseline you may override with reasoning

HARD CONSTRAINTS
- Do NOT look genes up against any canonical/predefined marker list. Infer transcriptional
  programs from the ranked DE evidence itself, reasoning at the gene-program level.
- Use both directions: a gene high in pct_out (broadly expressed) is weak evidence.
- Treat the candidate_annotation as a hypothesis to test, not an answer to confirm.

REASONING FLOW
1. Program discovery: from the full de_table, name the dominant transcriptional
   program(s) and the genes supporting each, with a confidence.
2. Conflict detection: do >=2 programs co-occur that should not share one cell
   (e.g. epithelial-like + myeloid-inflammatory; lymphoid + myeloid; epithelial +
   endothelial; structural + immune)?
3. Artifact assessment: weigh doublet, ambient-RNA contamination, phagocytosis-derived
   signal, mitochondrial/ribosomal dominance, stress-response dominance, and
   sample-specific (single-source) clusters — cross-checking qc_metrics and
   sample_distribution.
4. Confidence: validate, downgrade, or flag the candidate. Recommend extra validation
   ONLY when the evidence genuinely warrants it.

OUTPUT (one JSON object per cluster)
{
  "cluster_id": "...",
  "candidate_annotation": "...",
  "inferred_programs": [{"program": "...", "supporting_genes": ["..."], "confidence": "low|medium|high"}],
  "conflict_detected": true|false,
  "artifact_risk": "low|medium|high",
  "possible_explanations": ["doublet", "ambient RNA contamination", "mixed-lineage", "..."],
  "final_tier1_status": "confirmed|low_confidence|mixed_or_artifact_suspected",
  "confidence": "low|medium|high",
  "recommendation": ["..."]
}
The objective is to reduce false-positive annotations and surface suspicious clusters
before downstream tiers — not to relabel cells.
"""

# Step-split variant (program discovery → annotation review), if the agent runs two passes.
ANNOTATION_REVIEW_PROGRAM_DISCOVERY = """\
Given ONLY the ranked DE table (no marker database, no candidate label), list the
dominant transcriptional programs present, the genes supporting each, and a confidence.
Reason about what biological program each gene cluster reflects; do not name a final
cell type. Output JSON: [{"program": "...", "supporting_genes": ["..."], "confidence": "..."}].
"""

# Placeholders for the rest of D3 (orchestration / interpretation) — filled when mode-2 lands.
ORCHESTRATION_PROMPT = ""  # TODO (plan D3)
INTERPRETATION_PROMPT = ""  # TODO (plan D3)
