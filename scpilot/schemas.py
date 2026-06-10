"""공통 구조화 결과 스키마 — scpilot plan A4.

모든 tool 반환: {status, summary, artifacts[], checkpoint, warnings[],
error_code?, recoverable?, suggested_next_tools?}. 표는 행수제한+미리보기,
전체는 artifact 경로. 잡 모델 결과 스키마(시도/경과/peak-mem/fallback) 포함.
TODO: 미구현(skeleton).
"""
