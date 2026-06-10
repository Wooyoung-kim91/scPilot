"""분석 세션 = 온디스크 작업디렉토리 + 상주 AnnData 캐시 + 이력 로그 — scpilot plan A3.

MVP: 단일 out_dir(scqc식). 멀티클라이언트(session_id/lock)는 연기(재검토 결정).
vendor.harness의 atomic_path/체크포인트/provenance 헬퍼 재사용.
TODO: 미구현(skeleton).
"""
