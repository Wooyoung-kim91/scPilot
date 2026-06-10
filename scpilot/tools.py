"""core 함수를 tool로 노출하는 단일 레지스트리 — scpilot plan C1.

⚠️ 베다링한 vendor.harness의 선형 Pipeline 대신, compartment 재귀 + 장시간
잡(start/get_job_status/get_job_result/cancel_job)을 표현하는 레지스트리 필요.
vendor.harness의 무상태 primitive(atomic_path/provenance/StageReport 등) 위에 구축.
TODO: 미구현(skeleton).
"""
