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
