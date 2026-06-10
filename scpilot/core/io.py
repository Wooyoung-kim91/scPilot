"""load_h5ad/save (+ vendored 10x 리더 위임). load_10x/merge는 scqc 소유, 진입점=merged h5ad — scpilot plan B1.

TODO: 미구현(skeleton). 계약: AnnData in → 요약 dict out(`scpilot.schemas`),
mutating tool은 `.uns["scpilot"]` provenance + 체크포인트 기록.
"""
