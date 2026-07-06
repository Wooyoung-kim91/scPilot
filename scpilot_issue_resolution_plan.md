# scPilot 실행 이슈 종합 — 원인 분석 및 해결 계획

작성일: 2026-06-25
근거 문서: `scpilot_codex_current_issues.md` (2026-06-24), `error_log.md` (2026-06-24)
대상 데이터: Obesity 마우스 scWAT/vWAT/SM, 42 샘플, 331,127 cells × ~33k genes

---

## 0. 요약

두 로그에서 보고된 문제는 8건이며, **코드 분석 결과 그중 2건(I-1 markers 과부하, I-2 세션 입력 오인)은 이미 작업트리에 수정이 적용되어 있고**(아직 미커밋), mito_prefix 관련 함정은 이미 커밋된 코드에서 대소문자 정규화로 해결되어 있다. 나머지 4건(HVG batch 강제, mixed-lineage 휴먼 하드코딩, 디스크 사용량, 환경/numba)은 미해결이다.

| # | 이슈 | 심각도 | 상태 |
|---|------|--------|------|
| I-1 | markers 도구 과부하(고정 Wilcoxon·전체 유전자) → 타임아웃 | 높음 | ✅ 작업트리 수정됨(미커밋) |
| I-2 | MCP 세션이 요청한 체크포인트 대신 원본 입력을 로드 → `invalid_state` | 높음 | ✅ 작업트리 수정됨(미커밋), 잔여 위험 1건 |
| I-3 | preprocess: batch-aware HVG 강제, 초소형 라이브러리에서 loess 특이행렬 | 높음 | ❌ 미해결 |
| I-4 | QC mixed-lineage가 휴먼 유전자(EPCAM/CD3D) 하드코딩 → 마우스 미작동 | 중간 | ❌ 미해결 |
| I-5 | 디스크 사용량 과다(체크포인트 11GB×N, 총 40GB+) | 중간 | ❌ 미해결 |
| I-6 | 환경: base에 anndata 부재, numba cache 권한, conda env 필요 | 낮음 | ⚠️ 부분(문서/doctor 보강 필요) |
| I-7 | 중단 시 부분 체크포인트 미보존(재개 불가) | 낮음 | ❌ 설계상 한계 |
| I-8 | mito_prefix 휴먼 기본값(`MT-`) → 마우스 pct_mt=0 우려 | 낮음 | ✅ 커밋된 코드에서 해결됨(대소문자 정규화) |

---

## 1. 이미 적용된 수정 (커밋 + 큰 데이터 검증만 필요)

### I-1. markers 도구 과부하 → 타임아웃
- **원인**: 구버전 `markers`는 Wilcoxon 고정 + `adata.n_vars`(33,696개) 전체를 랭킹 → 331k cells에서 MCP 타임아웃, 산출물 없음.
- **현재 코드** (`scpilot/core/markers.py`, 작업트리):
  - `method` 파라미터 추가(`wilcoxon` / `t-test` / `t-test_overestim_var`).
  - `max_genes_ranked` 파라미터 추가, **기본 5000개로 캡**(`DEFAULT_MAX_GENES_RANKED`). `None`이면 전체 랭킹 유지.
  - `invalid_params` 검증 추가. `_PARAM_HINTS`/`validate.py`에 nullable 처리 반영.
- **남은 작업**:
  1. 작업트리 변경 커밋.
  2. 331k cells에서 `method="t-test_overestim_var"`, `max_genes_ranked=200~500`으로 실제 실행해 **타임아웃 내 완료 검증**(`scpilot_codex_current_issues.md`의 권장 대안과 동일).
  3. 캡 적용 시 `csv_is_full_ranking=False`가 summary/artifact meta에 정확히 표기되는지 확인.

### I-2. MCP가 요청 체크포인트 대신 원본 입력 로드 → invalid_state
- **근본 원인**: 기존 MCP `run`은 `wd = workdir or DEFAULT_RUN_DIR`(고정 디렉터리)였다. `Session.create(out, input_path=..., exist_ok=True)`는 **이미 세션이 존재하면 `open()`으로 기존 세션을 반환하고 새 `input_path`를 무시**한다(`session.py:319-322`). 따라서 04_cluster.h5ad를 입력으로 줘도 DEFAULT_RUN_DIR의 기존 세션(원본 `obesity_merged_counts.h5ad` 기반)이 재사용되고, `adata` 프로퍼티가 그 세션의 입력/체크포인트를 lazy-load → `leiden` 부재 → `invalid_state`.
- **현재 코드** (`scpilot/mcp_server.py`, 작업트리): `default_workdir_for_input(input)` 추가 → 입력 파일별로 `<stem>_scpilot_session` 디렉터리를 분리. 서로 다른 입력은 서로 다른 세션 디렉터리를 갖게 되어 무관한 세션 재사용을 차단.
- **잔여 위험**: `Session.create(exist_ok=True)`는 여전히 **기존 세션의 매니페스트 input과 새로 넘긴 `input` 인자가 달라도 조용히 기존 것을 사용**한다. 같은 workdir에 다른 입력을 가리키면 silent mismatch.
- **남은 작업**:
  1. 작업트리 변경 커밋.
  2. `Session.create`(또는 MCP `run`)에 **입력 fingerprint 불일치 가드** 추가 — 기존 세션의 `manifest.input.fingerprint`와 넘어온 `input`의 fingerprint가 다르면 `invalid_state`로 명확히 거부하고 새 workdir 사용을 안내(silent reuse 금지).
  3. markers를 04_cluster.h5ad 입력으로 호출 시 `leiden`이 보이는지 재현 테스트.

### I-8. mito_prefix 휴먼 기본값
- **확인**: `scpilot/core/qc.py:120-121`이 `var_names`와 `mito_prefix`를 **양쪽 모두 `.upper()`로 정규화**(`up.str.startswith(mito_prefix.upper())`). 마우스 `mt-Nd1` → `MT-ND1`가 기본 `MT-`와 매칭됨. 즉 현재 커밋된 코드에서 기본값으로도 마우스 pct_mt가 정상 계산된다.
- **남은 작업**: 회귀 방지용 단위 테스트만 추가(마우스풍 `mt-` var_names 픽스처로 `pct_mt>0` 확인).

---

## 2. 미해결 — 우선순위 순 해결 계획

### I-3. preprocess HVG: batch-aware 강제 + 초소형 라이브러리 loess 특이행렬 (높음)
- **증상**: `ValueError: reciprocal condition number 7.6322e-15` (`hvg_batch_key=library`/auto 시). 132-cell `GSM5554974_TH1_scWAT_THFD`에서 seurat_v3 per-batch loess가 특이.
- **근본 원인** (`scpilot/core/preprocess.py:57-77`):
  - `hvg_batch_key=None`이면 `("sample_id","sample","batch","donor","patient")`를 자동 탐지해 **2≤cardinality≤200이면 강제로 batch-aware HVG**. 즉 batch-aware HVG를 **끌 방법이 파라미터로 없음**(`"none"`이나 없는 컬럼명을 줘도 auto-detect가 다시 sample_id를 잡음).
  - per-batch 셀 수가 너무 적으면 loess 설계행렬이 singular.
- **해결안**:
  1. **명시적 비활성화 토큰**: `hvg_batch_key`에 `"none"`/`""`/`False`를 받으면 auto-detect를 건너뛰고 `batch_key=None`(전역 HVG)로 확정. (현재는 없는 컬럼 경고 후 auto-detect로 되돌아감 → 이 fallthrough 제거.)
  2. **초소형 batch 가드**: auto-detect 또는 명시 batch 사용 시, `min_cells_per_batch`(기본 예: 200~500) 미만 그룹은 (a) HVG batch 계산에서 제외하거나 (b) 경고와 함께 batch-aware를 자동 비활성화. 임계값과 드롭된 그룹을 summary/warnings에 기록.
  3. **loess 실패 자동 복구**: `sc.pp.highly_variable_genes(...)`를 try/except로 감싸 특이행렬류 `ValueError` 발생 시 `batch_key=None`으로 1회 자동 재시도하고 그 사실을 warning으로 명시(silent 금지).
  4. 권장 기본 동작 문서화: 다수 초소형 라이브러리가 있으면 cardinality 낮고 그룹 큰 키(`tissue` 등) 권장 — 단, sample 단위 보정은 어차피 Harmony/scVI 임베딩이 담당하므로 전역 HVG도 안전한 기본임을 명시.
- **테스트**: 132-cell짜리 초소형 batch를 포함한 픽스처로 (1) 명시 비활성화 (2) 자동 가드 (3) 자동 복구 3경로 검증.

### I-4. QC mixed-lineage 휴먼 유전자 하드코딩 (중간)
- **증상**: `mixed-lineage genes absent (('EPCAM','CD3D')); flag set False` — 마우스에서 항상 스킵.
- **근본 원인** (`scpilot/core/qc.py:25, 137-148`): `_MIXED_LINEAGE_GENES=("EPCAM","CD3D")`를 `var_names`에 **대소문자 정규화 없이** 직접 매칭. 마우스 `Epcam`/`Cd3d` 미매칭.
- **해결안**:
  1. 매칭을 **대소문자 무시**로 변경(I-8과 동일하게 `var_names.str.upper()` 기준 인덱싱) → 마우스 `Epcam`/`Cd3d` 자동 포착.
  2. `mixed_lineage_genes` 파라미터를 노출(기본 `("EPCAM","CD3D")`)해 조직/종 특이 쌍을 호출자가 지정 가능하게.
  3. 일부만 존재하면 현행대로 스킵하되, 어떤 유전자가 없었는지 warning에 유지.
- **테스트**: 마우스풍 픽스처(`Epcam`,`Cd3d`)에서 flag가 계산되는지, 커스텀 유전자쌍 전달이 동작하는지.

### I-5. 디스크 사용량 과다 (중간)
- **증상**: 체크포인트 총 40GB+, `04_cluster.h5ad`/`00_load.h5ad` 각 ~11GB. markers 등 추가 단계마다 풀 h5ad 한 개씩 더 쌓임.
- **근본 원인**: 모든 mutating 단계가 풀 AnnData 체크포인트를 기록(`session.checkpoint`). markers는 X/레이어를 바꾸지 않고 `uns["rank_genes_groups"]`만 추가하는데도 ~11GB h5ad를 새로 씀.
- **해결안** (영향도/난이도 순):
  1. **체크포인트 gzip 압축**: `adata.write_h5ad(..., compression="gzip")`로 단계별 기록. CPU 약간 더 쓰고 용량 크게 절감(특히 sparse counts). 가장 빠른 win.
  2. **중간 체크포인트 프루닝**: 동일 `x_state` 계열에서 오래된 중간 체크포인트를 옵션으로 정리하는 `scpilot` 유틸(매니페스트의 최신/북마크는 보존). content-addressed 저장과 충돌하지 않게 manifest 참조 기준으로만 삭제.
  3. **경량 단계의 체크포인트 회피/delta**: markers/annotation_review 등 X 비변경 단계는 풀 h5ad 대신 산출물(CSV/uns)만 저장하거나, 직전 체크포인트를 재사용(포인터)하도록. (단, 재현 해시 체계와의 정합성 검토 필요 — `harness-chokepoints` 메모리 참조.)
  4. **운영 권고**: markers 실행 전 디스크 여유 확인, 불필요 세션(`scpilot_obesity_markers_from_cluster` 등 워크어라운드) 정리.
- **권장 시작점**: 1(압축) + 4(정리)부터. 2·3은 재현 해시 영향 검토 후.

### I-6. 환경 / numba (낮음)
- **증상**: base 파이썬에 anndata 없음; scanpy import 시 numba cache 권한 실패 → `NUMBA_CACHE_DIR=/tmp/numba-cache` 우회 필요.
- **해결안**:
  1. `scpilot doctor`에 **anndata/scanpy/scikit-misc(seurat_v3 의존) import 체크**와 conda env 안내 추가.
  2. numba cache 디렉터리 미설정/비쓰기 시 **`NUMBA_CACHE_DIR`을 세션 logs 하위 등 쓰기 가능 경로로 자동 설정**(`init_runtime`/`mcp_server` 부트스트랩 단계). 메모리 `scpilot-env-test-cmd`의 conda env 경로와 정합.
  3. README/실행 가이드에 "base가 아닌 scpilot conda env로 실행" 명시(기존 메모리와 일치).

### I-7. 중단 시 부분 체크포인트 미보존 (낮음, 설계 한계)
- **증상**: markers 중단 시 부분 산출물/체크포인트 없음 → 처음부터 재실행.
- **해결안**: 단일 scanpy 호출(`rank_genes_groups`)은 본질적으로 중간 저장이 어려움. 현실적 완화는 **I-1의 캡(빠른 완료)으로 중단 필요성 자체를 줄이는 것**. 추가로 장시간 단계는 cluster별 분할 실행 옵션을 검토하되 우선순위 낮음.

---

## 3. 실행 순서 (권장)

1. **즉시**: 작업트리 변경(I-1, I-2 수정) 커밋 → 331k 데이터에서 markers를 캡/경량 method로 재실행 검증.
2. **단기**: I-2 잔여 fingerprint 가드, I-3 HVG(비활성화 토큰 + 초소형 batch 가드 + 자동 복구), I-4 mixed-lineage 대소문자/파라미터화. 각 항목에 단위 테스트 동반.
3. **중기**: I-5 체크포인트 압축 + 워크어라운드 세션 정리, I-6 doctor/numba 자동화.
4. **데이터 품질 후속**(error_log §데이터 품질 주의): `GSM5555003_TH2_SM_THFD`(전체 ~22%, 저복잡도) 저품질 클러스터 형성 여부 클러스터링 후 확인, 초소형 라이브러리(`GSM5554974`, `GSM5554986`) 제외 여부 결정.

## 4. 검증용 안전 재개 지점 (기존 산출물 재사용)
- 클러스터 체크포인트: `scpilot_obesity_run/checkpoints/04_cluster.h5ad` (leiden, X_pca, X_umap, scale.data 보유)
- markers 재실행 시 위 파일을 입력으로, 캡 적용 + `t-test_overestim_var`로 우선 검증.

---

## 5. CELLxGENE 재주석 · 대규모 확장 감사 (2026-07-02)

근거: 코드 3영역(annotation 코어 / 결정론적 도구 / 오케스트레이션) 병렬 감사 + 현재 워크트리 코드 검증(§7.3).
목표: CELLxGENE **인간 췌장 완전 재주석**(저자 라벨 폐기, marker-DB-free) → **인간 primary 전체**로 확장(≤5M 세포 샤드, 샤드별 실행 → 샤드 간 라벨 통합).
하드웨어: 502GB RAM, 120스레드; 신규 96GB GPU + 8TB 저장 구매 예정(현재 torch CUDA 불가).

### 5.0 사용자 의도 결정 (수정 방향 확정)
- **CNV/malignancy: 전 샘플 수행** → 정상조직 robust화 필수(I-21).
- **통합: Harmony + scVI 둘 다 → benchmark 자동선택** → scVI GPU/epoch 수정 전제(I-10).
- **샤드 간 통합: cellhint 실제 배선** (I-9).
- **marker-DB-free: 하드 강제** — 오직 DE로 LLM 추론. `annotate_broad`/패널 경로 비활성, `harness_audit`가 패널 사용을 **fail**로 차단(현재 warn).
- **해상도: 분리도(separability) 기반** — 세포수 무관, 과분할/과소분할 방지(I-22, §3 계약 변경 — owner 결정).
- **Tier-3(trajectory/cell_state): 보류** — cell_state는 fine 단계 free-text 유지.
- **provenance: 강제하되 체크포인트 용량 억제** (I-14 + I-5).
- **토큰: 국면별 하이브리드** — 파일럿=역할별 티어링, 확장=결정론적 스캐폴딩; 공통으로 트랜스크립트 압축+캐시(I-23).

### 5.1 기존 이슈 상태 갱신
- **I-4 → ✅ 해결(코드+테스트)**: `qc.py` mixed-lineage opt-in + `_species.resolve` 대소문자 무시. 2026-07-06 마우스 케이싱 회귀 테스트 추가로 닫힘(§6).
- **I-3(HVG 강제)·I-5(디스크)·I-6(env): ✅ 2026-07-06 해결**(상세 §6). I-7(부분 체크포인트 미보존): 단일 scanpy 호출 특성상 설계 한계로 보류(I-1 캡으로 완화).

### 5.2 신규 이슈 (심각도 순)

**Blockers**
- **I-9 샤드 간 라벨 통합 부재.** `annotate.py:705-713` `_harmonize_with_cellhint`가 `raise NotImplementedError`; `harmonize_annotations`/`consensus_vote`(`recipes.py:273-316`)는 한 AnnData의 여러 obs 컬럼(같은 세포)끼리만 투표 → 샤드(다른 세포) 못 넘음. 결정: **cellhint 배선 + 크로스-데이터셋 라벨 어휘 정렬**.
- **I-10 scVI CPU 고정 + 저학습.** ✅(완료·검증) `train_scvi`에 `accelerator="auto"`(GPU 있으면 사용)+`devices` 파라미터, `max_epochs=None`일 때 `max(heuristic, MIN_SCVI_EPOCHS=50)` floor(대형 shard 저학습 방지, early_stopping 유지). `test_integrate.py` 7 pass(CPU 폴백 확인). GPU 실검증은 하드웨어 도착 후.
- **I-11 Ensembl-ID var_names 무증상 붕괴 (CELLxGENE 직격).** `_species.detect_organism`(`_species.py:30`)의 `isupper()`가 `ENSG…`를 True→"human" 오탐(마우스 `ENSMUSG`도), `qc.py:134`의 `MT-` 매칭 0 → `pct_counts_mt=0`, 미토 필터 무력화(경고 없음). 마커 DE·CNV 심볼매핑 동시 저하. 결정: **데이터의 `feature_name`/gene-symbol 컬럼에서 심볼로 재매핑하는 evidence-based 진입 단계**(하드코딩 아님) — load/ingest에서 Ensembl-ID var_names 감지 시 심볼 컬럼으로 스왑, 감지·스왑 사실을 warnings에 기록.
- **I-12 배치 러너 부재 + workdir 충돌.** ✅(부분 완료·검증) mode-2 `run` 기본 workdir를 `default_workdir_for_input`(입력별 고유 dir, session.py로 단일화 — MCP도 공유)로 변경 → 샤드 간 미충돌. `Session.create`에 **fingerprint 불일치 가드**(`InputMismatch`) 추가 → 같은 workdir에 다른 입력 시 silent 재사용 대신 명확히 거부(I-2 잔여작업 해결). `test_session_input_guard.py` 5 pass. **남음(확장)**: 외부 다중샤드 오케스트레이터(I-9와 함께).
- **I-13 LLM 재시도/백오프 없음.** ✅(완료·검증) base `Provider.complete`가 `_complete`(백엔드별)를 감싸 지수 백오프 재시도(chokepoint 한 곳 → agent 루프·force_structured·interpretation·run_review 전부 커버). `_is_retryable`(429/529/5xx/timeout/connection만; auth/bad-request/ProviderUnavailable 제외). agent 메인 루프는 재시도 소진/영구오류 시 graceful break(`stopped_reason="provider_error"`)로 report/interpretation 보존. `test_provider_retry.py` 5 pass.

**Major**
- **I-14 3중 행렬복사 + 체크포인트 비대 (I-5 확장).** ⚠️(부분 완료) ✅ `log_consistency`가 **checkpoint-vs-run 간극 감지**(`checkpoint_bypass_suspected`; 사용자가 본 "25 vs 2" ad-hoc bypass를 표면화). `test_provenance_gap.py` 3 pass. **남음(별도 집중 작업 — 위험)**: `scale.data` 중복 제거(markers/detect_state/benchmark/ingest 등 다수 파일 + 재현성 불변식 영향), 경량 단계 델타/포인터 체크포인트, `checkpoint()`→run_log 자동기록(§2 chokepoint 설계와 충돌 가능). 이들은 재현성 회귀 위험이 커 집중 세션 권장.
- **I-15 majority_vote 세포당 Python 루프** ✅(완료·검증) `recipes.py`에서 (n,k,k) 동치 텐서로 벡터화(n_keys 작음) → 최빈값·유일성·agreement 일괄 계산, 세포당 Python 루프 제거. `test_tier4_scale.py` 통과.
- **I-16 Tier-4 마커검증 특이성 누락** ✅(완료·검증) `audit.py` `_check_marker`에 pct_out≤max_pct_out AND spec≥min_specificity 게이트 추가(marker-DB-free 선택 게이트와 정합) → housekeeping/ambient 유전자가 marker_set_support_frac 부풀리는 것 방지. `marker_criteria`에 두 값 노출. `test_audit.py` 갱신 통과.
- **I-17 Tier-4 근거 20k 절단** ✅(완료·검증) `agent.py` `_cap_evidence()` 헬퍼로 캡 60k 상향 + 절단 시 **가시 마커 + 경고**(리뷰어가 부분 커버리지 인지, run_annotation_critique가 `evidence_truncated` 반환) → silent 누락 제거. `test_tier4_scale.py` 통과. (클러스터 단위 트리밍은 후속 개선 여지.)
- **I-18 detect_state가 원본 입력 읽음** (`state.py:84-87`) → 진행도 무관 `stage="raw"`, 자동 재개 깨짐(I-2/I-7 확장). 결정: 최신 체크포인트/manifest stage 기준으로 재진입 판정.
- **I-19 `--max-iters=40` 부족** ✅(완료) `cli.py` 기본값 40→**120**(전체 파이프라인 60-120 호출; ceiling 200). + **marker-DB-free 하드 강제**: `harness_audit`의 `marker_db_free` 체크를 warn→**fail**로 변경(annotate_broad 사용 시 governance 게이트 실패). `test_audit.py` 갱신 통과.
- **I-20 PDAC 특화 기본값** ✅(부분 완료·검증) `train_scvi`/`integrate_harmony`의 `batch_key` 기본을 `"GSM"`→**중립 auto-resolve**(`_resolve_batch_key`: 명시 우선, 없으면 sample_id/donor_id/dataset_id… 자동 탐지, 없으면 명확 에러). `test_integrate_defaults.py` 4 pass. (integrate_scvi 사전학습 applier는 모델-bound라 GSM 유지 — 데이터에 없으면 이미 명확히 게이트.)
- **I-21 reference-free CNV 정상조직 오탐** ✅(완료·검증) `cnv.py`: (a) advisory 경고에 acinar 등 pseudo-CNV 위험 명시, (b) **데이터 기반 내부 reference 후보 제안**(`suggested_reference_groups` — 최저 CNV burden 그룹, 하드코딩 아님), (c) malignancy_evidence의 clonal 신호에 `donor_confound_suspected` 플래그(단일 donor 우세 + CNV 비상승 → 정상 donor-특이 집단, 진짜 클론은 미플래그). 증거 방출만(판정은 LLM+하드룰 유지). `test_cnv.py` 13 pass(over-flag 방지 포함).

**설계 변경 / cost**
- **I-22 해상도 분리도 기반** ✅(완료·검증) `cluster_sweep`가 해상도별 **실루엣**(use_rep 임베딩, 시드 subsample)을 함께 산출; `suggest_resolution`이 ≥2클러스터 중 **실루엣 최대** 해상도 선택(실루엣 없으면 기존 n_clusters knee로 폴백). 과분할(희소/노이즈)·과소분할 모두 실루엣이 낮아 자동 회피 → 세포수 아닌 실제 분리도 기반. `test_resolution_separability.py` 3 pass + 기존 sweep 테스트 유지. (§3 계약 개정: knee→분리도.)
- **I-23 토큰 절감.** 원인: 메인 루프 트랜스크립트 O(n²) 재전송(`agent.py:500`), 전 역할 Opus 기본(`provider.py:48`), 샤드 간 캐시 미재사용, 20k 원시 JSON. 결정: 결정론적 스캐폴딩(고정 DAG, 판정 지점만 LLM) + 역할별 티어링 + 트랜스크립트 압축 + 지속 캐시 프리픽스 + per-shard 토큰 예산 가드.

**신규 기능 (사용자 요청, 2026-07-02)**
- **I-24 MCP 호출 시점 LLM topology 선택.** LLM 호출을 `api`(scpilot이 API 직접 호출) / `cli`(scpilot이 `claude`·`codex` 서브프로세스 실행) / `host_plugin`(호스트가 플러그인으로 위임)로 구분하고, 역할별(analysis/reviewer/annotator/interpreter)로 MCP 호출 시점에 선택. analysis는 mode-1에선 호스트 자신이므로 **선언/provenance 기록용**. 리뷰어는 세 방식 모두 config 선택. 노출은 전용 `configure_run` 툴.
  - **결정(사용자 확정)**: 리뷰어 실행 3방식 모두 지원 / `configure_run` 전용 툴 / analysis 선언·기록용.
  - **Increment 1 ✅(완료·검증)**: `llm/topology.py`(validate + availability probe), `core/configure.py`(`configure_run` 툴 — 세션 manifest 영속화, host_plugin 리뷰어 위임 directive, 미가용 경고), `session.Manifest.llm_topology` 필드. 테스트 `test_configure_run.py` 5 pass + 회귀 21 pass.
  - **Increment 2 ✅(완료·검증)**: `provider.py`에 `CLIProvider`(claude/codex subprocess; forced-structured JSON 파싱, `_run_cli` 격리로 테스트) + `build_role_provider(spec)`(api→API backend, cli→CLIProvider, host_plugin→None). 테스트 `test_cli_provider.py` 6 pass.
  - **Increment 3 ✅(완료·검증)**: `run_review` 툴(결정론적 — annotation_audit 증거 + topology 기반 라우팅 directive; LLM을 registry 툴 내부에서 안 돌려 replay 안전) + `review_routing()` 순수 헬퍼 + mode-2 `cli.py _role`이 세션 topology 소비(우선순위: CLI 플래그 > topology > analysis; host_plugin은 mode-2 비적용→analysis 폴백). 테스트 통과.
  - **전체 회귀**: `pytest` 230 passed / 1 skipped / 0 fail (459s). replay-safety 설계 결정: mode-1 registry 툴은 LLM 인라인 실행 안 함 — api/cli 리뷰어의 실제 실행은 replay-safe한 mode-2 `run_agent` 경로(verdicts가 apply_annotation_audit param으로 기록됨) 또는 호스트가 담당.

### 5.3 구현 우선순위
1. **파일럿 차단 해제(즉시)**: I-11(Ensembl 진입), I-13(백오프), I-12(workdir/fingerprint). — 췌장 파일럿을 안전히 돌리기 위한 최소 집합.
2. **품질·정확성**: I-16, I-17, I-15, I-21(CNV 정상조직), I-22(해상도), marker-DB-free 하드 강제.
3. **GPU·확장**: I-10(scVI GPU/epoch), I-20(중립 기본값), I-19(max-iters), I-14(용량/provenance).
4. **확장 전용**: I-9(cellhint 크로스샤드), 다중샤드 오케스트레이터, I-23(토큰).
각 항목 §8대로 env 바이너리로 테스트 후 완료 선언. 대규모 동작 변경은 실제 데이터 캡/경량 재현 포함.

---

## 6. 수정 패스 (2026-07-06, 커밋 `aa21f89`)

recon(코드 4영역 병렬 감사, 문서 아닌 현재 코드 기준 §7)으로 문서상 미완 항목의 실제 상태를 확정한 뒤,
main tree에서 순차 수정 + 이슈별 env 바이너리 타깃 테스트로 검증. **전체 스위트 260 passed / 1 skipped / 0 failed**
(직전 baseline은 253 passed / 1 failed).

### 6.1 해결 (코드 + 테스트)
- **I-18 (신규 발견, resume 깨짐)**: `state.py` `_detect_state_tool`이 `manifest.input`(원본 raw)이 아니라
  `session.latest_checkpoint()` 기준으로 재진입 판정. 진행된 세션이 항상 `stage="raw"`로 오보고되던 문제 해결.
  회귀 테스트 `test_io_state.py::test_detect_state_tool_keys_off_latest_checkpoint`.
- **I-6 (env)**: `init_runtime()`를 `Session.create` **및** `Session.open` 진입점 최상단으로 이동 →
  resume 분기·replay(fresh 프로세스)도 `NUMBA_CACHE_DIR`/njit-cache 패치를 받음.
  회귀 테스트(서브프로세스) `test_session_open_sets_numba_cache_dir`.
- **I-11 (CELLxGENE 직결)**: `ingest`가 `load_input`을 우회하던 문제 → 병합 직후
  `_species.normalize_var_symbols(merged)` 배선 + evidence를 warnings/uns 기록. `qc_metrics`에 Ensembl-ID
  var_names로 `pct_counts_mt=0`(mito 필터 무력)일 때 명시 경고 가드 추가(§3 warn-never-silent).
  테스트: `test_qc.py` 마우스 케이싱(I-4/I-8) + Ensembl 가드.
- **I-5 (디스크)**: 체크포인트/`save_h5ad` 기본 압축 `lzf`→`gzip`(비율↑; `compression` 파라미터로 오버라이드).
  replay 정합 확인.
- **I-3 (HVG)**: 코드는 기해결(`recipes.py:49-97`)이었고 회귀 테스트만 부재 → disable-token, tiny-batch 가드
  경로 테스트 추가(`test_preprocess_cluster_markers.py`). singular-loess 자동복구는 결정론적 트리거가 어려워
  코드 검토로 갈음.
- **I-9 (테스트만)**: cellhint 1.0.0 설치 반영해 낡은 `test_harmonize_annotations_consensus_fallback`를
  env-독립적으로 수정(설치 여부를 실제 반영; stub이 `NotImplementedError`라 여전히 consensus fallback + 경고).
  **실제 크로스샤드 배선은 새 다중샤드 입력 계약(§4 owner 결정)이 필요해 연기.**

### 6.2 API 충실도 정리 (검증된 라이브러리 API로 위임)
- `qc.py::_med_mad` → `scipy.stats.median_abs_deviation(scale="normal")`.
- Shannon entropy(`audit.py` batch_entropy, `compartment.py` `_norm_entropy`) → `scipy.stats.entropy`.
- (감사 결과 integrate=harmonypy/scvi-tools, cnv=infercnvpy, silhouette=sklearn 등 무거운 계산은 이미
  전부 라이브러리 위임 — 손수 재구현 없음. `audit` 마커 Jaccard는 소규모 정확 집합연산이라 의도적으로 유지.)

### 6.3 의도적 보류 (이유 명시)
- **I-9 실제 크로스샤드 cellhint 배선**: 다중샤드 입력 계약 = owner 결정(§4). GPU/8TB 확보 후 착수.
- **`integrate_scvi`의 `batch_key="GSM"` 기본값**: 사전학습 모델은 학습 시 batch_key에 바인딩되고 이미 미지
  카테고리 게이트가 있어 silent-wrong 아님 → 방어적으로 정당. 개선 시 모델 레지스트리에서 키를 읽도록.
- **cli.py**: 낡은 "A1 skeleton/stubs" docstring 정리(구현은 이미 완료).
