"""FastMCP 서버 — tools.py 레지스트리를 MCP tool로 등록 — scpilot plan A6/C2.

stdout엔 프로토콜 JSON만, 로그는 stderr/파일로. A6=읽기전용 inspect_h5ad 1개로
Claude Code+Codex stdio 스파이크. vendor.harness.init_runtime()을 서버 기동 시 호출
(detached 세션 numba 캐시 문제 회피).
TODO: 미구현(skeleton).
"""
