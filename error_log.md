# Obesity scpilot 분석 — 오류·경고 정리

작성일: 2026-06-24
대상: `/home/wykim/data/Obesity/sra/cellranger_count` (42 샘플, 마우스 scWAT/vWAT/SM)
저장 위치: `/home/wykim/data/Obesity_Claude`
중단 시점 마지막 체크포인트: `checkpoints/04_integrate_harmony.h5ad` (X_pca + X_harmony 보유)

---

## 진행 상황 (중단 시점)

| 단계 | 상태 | 결과 |
|------|------|------|
| Profile YAML | ✅ | `obesity_profile.yaml` |
| Ingest (42 샘플) | ✅ | 331,127 cells × 29,711 genes, 0 실패 |
| QC metrics | ✅ | doublet 1.06%, 분포 정상 |
| QC filter | ✅ | → 308,043 cells (7.0% 제거) |
| Preprocess (HVG+PCA) | ✅ | 3000 HVG, 50 PC (3회차 성공) |
| Harmony 통합 | ✅ | `X_harmony` (checkpoint 04) |
| scVI 학습 | ⏸️ 미실행 | 사용자 중단 (실행 전 거부) |
| Benchmark / cluster / annotation / report | ⏳ 대기 | — |

---

## 1. Preprocess (HVG + PCA) — 실제 오류, 2회 실패 후 해결 ⚠️→✅

```
ValueError: b'reciprocal condition number  7.6322e-15'
```

| 시도 | `hvg_batch_key` | 결과 | 원인 |
|------|----------------|------|------|
| 1차 | `library` (42개) | ❌ | 샘플별 loess 회귀가 특이행렬 |
| 2차 | `none` (→ 무시됨) | ❌ | `"none"`은 없는 컬럼 → auto-detect가 `sample_id`(42개) 재선택, 동일 분할 → 동일 오류 |
| 3차 | `tissue` (3개) | ✅ | 조직당 ~10만 cell, well-conditioned |

- **근본 원인**: seurat_v3 HVG가 batch별로 loess를 적합하는데, **132-cell짜리 `GSM5554974_TH1_scWAT_THFD`** 라이브러리에서 데이터가 너무 적어 설계행렬이 특이(singular)해짐.
- **함정**: scpilot `preprocess`는 `hvg_batch_key`가 None/없는 컬럼이면 자동으로 `sample_id`를 batch로 잡음. 즉 **batch-aware HVG를 끌 방법이 파라미터로 없음** → null이나 "none"을 넘겨도 우회 불가.
- **해결**: batch key를 cardinality 낮고 그룹이 큰 `tissue`로 지정 → 특이성 회피 + 조직 특이 유전자 포착. (샘플 단위 batch는 어차피 Harmony/scVI 임베딩에서 보정)

## 2. QC metrics — 경고 (오류 아님) ℹ️

```
mixed-lineage genes absent (('EPCAM', 'CD3D')); flag set False
```

- scpilot의 mixed-lineage(이중 계통) 검출이 **사람 유전자명 EPCAM/CD3D**를 하드코딩 → **마우스 데이터**라 부재. 해당 QC 항목만 스킵됐고 나머지(doublet, pct_mt 등)는 정상.
- 영향 없음. 단, 마우스에서 이 기능은 작동하지 않음을 기록.

## 3. 잠재적 함정으로 미리 막은 부분 (오류 아님) ✅

- **mito_prefix**: scpilot 기본값이 사람 기준 `MT-`. 마우스는 `mt-`라 profile에서 `mt-`로 명시 → pct_mt가 0으로 잘못 계산되는 사고 방지.

## 참고: 오류가 아닌 중단들

- Bash 호출 거부 2회, scVI 학습 거부 1회, scpilot 종료 — **모두 사용자 요청에 의한 중단**이며 도구 오류가 아님. scVI는 실행 전 거부되어 시작도 안 됨.

---

## 데이터 품질 주의 (재개 시 고려)

- **`GSM5555003_TH2_SM_THFD`**: QC 후에도 66,534 cells로 전체의 ~22% 차지. 원본 median 790 counts / 609 genes로 저복잡도(ambient 다수). n_genes 분포가 bimodal — cutoff 바로 위에 저품질 peak 잔존. 통합/클러스터링 후 저품질 클러스터 형성 여부 확인 필요.
- **초소형 라이브러리**: `GSM5554974_TH1_scWAT_THFD` (132 cells, loess 오류 원인), `GSM5554986_SH2_vWAT_SHFD` (824 cells). 재개 시 제외 여부 결정 권장.

## 요약

파이프라인이 막힌 실제 오류는 **preprocess의 loess 특이행렬 1건**뿐이며, 원인(132-cell 초소형 라이브러리 + batch-aware HVG 강제)과 해결(`tissue` batch key)까지 확인됨. 나머지는 경고 또는 사용자 중단.
