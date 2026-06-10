"""merged QC 요약 소비 → LLM cutoff·재필터 + Tier0 artifact 판정 (scrublet/batch-aware는 scqc qc 확장) — scpilot plan B3.

TODO: 미구현(skeleton). 계약: AnnData in → 요약 dict out(`scpilot.schemas`),
mutating tool은 `.uns["scpilot"]` provenance + 체크포인트 기록.
"""
