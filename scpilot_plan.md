# Plan: `scpilot` — LLM-Driven scRNA-seq Analysis (MCP + CLI)

## Context

목표는 scRNA-seq 분석 파이프라인 전 과정을 LLM이 구동·판단하는 자동화 도구를 만드는 것.
사용자는 **공용 코어 라이브러리 위에 MCP 서버와 CLI 에이전트를 모두** 얹는 형태를 원하며,
[iS2C2](https://github.com/methodistsmab/iS2C2)(CLI + 다중 LLM provider + 다운스트림 CCC 해석)를 참고하되,
iS2C2가 다루지 않는 **upstream 전체 파이프라인 + 통합방법 벤치마크 자동선정**까지 포함한다.

LLM의 역할(전부): ① 파이프라인 오케스트레이션 + 파라미터 결정, ② 세포타입 annotation,
③ 조건비교 DE, ④ 결과 해석 + 리포트.

### 현재 환경 (확인 완료)
- 작업 env `scpilot` (Python 3.11): `scanpy 1.11.5`, `anndata 0.12.14`, `leidenalg`, `harmonypy 0.2.0`,
  `scvi-tools 1.4.2`, `scrublet`, `scib-metrics`(설치됨), numpy 2.4.
- **GPU 미탑재 (추후 추가 예정)** → scVI는 현 조건상 **CPU 모드**(`accelerator="cpu"`)로 진행.
  학습이 느리므로 검증 단계는 서브샘플 + epoch 축소. GPU 추가 시 `accelerator` 인자만 `"auto"`로 전환.
  Harmony(CPU)가 1차 통합 기본, scVI는 벤치마크 후보.
- **env 확정(2026-06-10)**: 작업 env = **`scpilot`** (`scRNAseq` 복제 → 검증된 numpy 2.x 호환 매트릭스 상속).
  누락분 설치 완료: `scikit-misc 0.5.2`, `mcp 1.27.2`, `anthropic 0.109.1`, `celltypist 1.7.1`.
  `infercnvpy 0.6.1`·`R 4.4.3`는 이미 존재. scrublet 0.2.3이 numpy 2.4.5에서 RUN 검증됨.
- **실데이터** `/home/wykim/data/PDAC/`: raw counts 포함 `PDAC_merged_qc.h5ad` (180,977 cells × 40,237 genes,
  GSE/GSM·disease·tissue·treatment 메타데이터, layers `counts`/`scale.data`; sample_id 35개·GSE 3개), 처리완료
  `PDAC_merged_qc_log1p_hvg_umap.h5ad` (**2000 HVG로 subset**, PCA/Leiden/UMAP 有, **annotation 없음** → 회귀검증 전용,
  upstream 재실행은 raw merged에서), 조건비교용 `PDAC_pancreas_primary_vs_normal.h5ad`.
  ⚠️ `var`에 **chromosome/start/end 좌표 없음**(n_cells만) → CNV(B12) 전 좌표 주석 단계 선행 필수.
- **raw upstream 테스트 데이터** `~/progenesis/Transcriptomics/scRNAseq/`(= scqc `input_root`): per-sample 10x 원본
  3개 GSE(`PDAC_GSE155698`·`GSE197177`·`GSE205013`, CellRanger v3 `barcodes/features/matrix.tsv.gz`) +
  `PDAC_GSE_metadata_combined.csv`(GSM/local_matrix_dir/GSE/condition…) + 원본 노트북. scqc가 이걸 소비해 merged 생성.
  → **scpilot 테스트 전략**: 단일 GSE/GSM(작은 단위)로 io_10x·scqc qc 확장(scrublet) 검증, **tiny fixture는 1개 GSM 서브샘플**로 고정.
- `claude` CLI 설치됨 (`~/.local/bin/claude`). proto는 **git 저장소로 초기화 완료**(branch `main`, 2026-06-10).
- **product 이름 = `scpilot`** (CLI/패키지/env/`uns["scpilot"]` 공통). 2026-06-10 전역 rename 완료.

### 핵심 설계 아이디어
LLM은 **데이터를 직접 보지 않고**, 서버가 보유한 AnnData에 대해 tool을 호출한다.
각 tool은 데이터가 아니라 **요약 통계/표(JSON)** 를 반환 → LLM이 그 수치를 보고 임계값·resolution·통합방법을
결정하고 다음 tool을 호출한다. 이 "tool은 요약만 반환, 데이터는 서버에 상주" 패턴이 토큰 효율과 재현성의 핵심.
동일한 tool 집합을 **MCP 서버(클라이언트 LLM 구동)** 와 **CLI 에이전트(Anthropic API 자체 구동)** 둘 다에서 재사용.

### ⚠️ 아키텍처 전환 (2026-06-10, 사용자 결정 A): scpilot = **raw 10x부터 end-to-end**
초기엔 "scqc=upstream 위임, scpilot=merged부터" 경계였으나, **사용자가 단일 도구(raw→분석)를 원해 scqc의 upstream을
scpilot에 흡수**. 구현: scqc primitive(io_10x·metaschema·config·harness)를 베다링하고, `core/ingest.py` `ingest` 도구가
profile(raw 10x+metadata)→harmonize→per-sample read+cell QC→merge→normalize(counts+scale.data)를 한 번에 수행
→ 세션에 merged 적재 → 이후 qc_metrics/preprocess/... 다운스트림. (이미 merged h5ad가 있으면 ingest 생략하고 그 파일로 세션 생성.)
검증: 합성 10x 유닛테스트 + 실 raw 2-GSM(5842셀) end-to-end. **아래 "scqc 연계" 절은 베다링 정책 부분만 유효**(경계 위임은 폐기).

### scqc_pipeline 베다링 (하네스/io/metaschema/plotting 재사용 — 위 흡수의 토대)
옆 디렉토리 `/home/wykim/data/PDAC/scqc_pipeline/`(~2600 LOC, 성숙한 재현 가능 QC→merge CLI, 계약 단일소스
`HARNESS.md`)가 **본 계획 Phase A의 상당 부분과 B1/B5·upstream QC/merge를 이미 구현**해 둠. 재구현은 명백한 중복이므로
아래 경계로 분업한다.

**① 책임 경계 (데이터 진입점)**
- **scqc_pipeline 담당(upstream)**: metadata harmonize → per-sample 10x read → cell QC → merge → normalize.
  산출물 `PDAC_merged_qc.h5ad`(counts+scale.data, 180977×40237)가 **scpilot의 진입점**. scpilot은 io/qc/merge를 재구현하지 않음.
- **scpilot 담당(downstream)**: preprocess/HVG/PCA → baseline cluster → Tier1 annotation → integrate → benchmark →
  final cluster → Tier2/3 annotation → CNV/trajectory → Tier5 review → DE → report.
- 따라서 아래 **B1(io)·B2(state 일부)는 "scqc 위임/소비"로 격하**, **merge 단계는 scqc 소유**. scpilot은 merged h5ad에서 시작.

**② 베다링(vendoring) — 코드 결합 방식 결정**
scqc의 다음 모듈을 scpilot으로 **복사 후 독립 진화**(라이브러리 import 아님): `harness.py`(run_stage 계약·StageReport·
`is_fresh` 5조건 체크포인트·`atomic_path`·provenance/소스 스냅샷/`repro.py`·`init_runtime` numba 캐시 패치),
`io_10x.py`(robust 10x 리더), `plotting.py`(auto-fit figure 하네스), `metaschema.py`(cross-dataset harmonize/filter/derive).
- 이로써 **Phase A(A3 session·A4 schemas·A7 재현성 하네스)는 "백지 설계"가 아니라 "scqc 1차자산 + scpilot 확장"으로 축소**.
- ⚠️ 베다링 리스크(분기·이중 유지보수) 완화: 각 vendored 파일 상단에 `VENDORED FROM scqc_pipeline@<source_hash>` 주석 +
  vendored primitive는 얇고 안정적으로 유지(과한 수정 자제). uns 키는 `scpilot`로 파라미터화.
- **`init_runtime` numba 패치는 그대로 필수**: stdio MCP 서브프로세스(detached 세션)에서 numba import가 깨지는 동일 문제 발생.

**③ scpilot이 scqc 위에 순수 신규로 추가하는 것** (scqc에 없음):
MCP 서버(1순위 인터페이스) · 장시간 tool **잡 모델**(scVI/Harmony/scib/CNV는 stdio 타임아웃 초과) ·
**분기/재귀·capability 게이트 가능한 tool 레지스트리**(scqc의 선형 `Pipeline.ORDER`로는 compartment 재귀·선택 도구 표현 불가) ·
**LLM `decision`-event 스키마** · **결정성 등급(A/B/C) tolerance replay** · **capability-flag doctor**(scqc doctor는 matrix-dir만 점검) ·
다운스트림 전 과학(integrate/benchmark/annotate Tier0-5/cnv/trajectory/de/report).

**④ QC 경계 결정**: scrublet per-sample + %ribo + stress/dissociation + mixed-lineage(EPCAM+CD3D) 플래그 + batch-aware 분포
요약 반환은 **scqc의 per-sample qc stage를 확장**해 처리(scrublet은 merge 前 per-sample이어야 하므로 upstream에 기여).
scpilot은 그 batch-aware QC 산출을 진입 데이터로 상속.

### 상태/세션 모델 (Codex 리뷰 반영 — **온디스크 우선**)
6GB AnnData를 stdio MCP 프로세스 메모리에만 상주시키면 호스트 재시작·크래시·타임아웃·2차 클라이언트에서
상태가 유실/포크된다. 따라서:
- **온디스크 세션 디렉토리를 1급 객체로**: `session_id` + manifest(JSON) + 이력 로그 + **각 mutating 단계 후
  `.h5ad` 체크포인트**. 인메모리 AnnData는 *캐시*일 뿐. → 크래시 복구·재진입·재현 가능.
- **멀티클라이언트**: 세션 디렉토리에 file lock + 소유 메타데이터. 읽기전용 inspect는 동시 허용, mutation은
  직렬화 또는 구조화 에러로 거부.
- **장시간 tool은 잡(job) 모델로**: scrublet/scVI(CPU)/Harmony/UMAP/scib는 stdio JSON-RPC 타임아웃을 넘김 →
  `start_<x>` → `get_job_status` → `get_job_result` → `cancel_job` + 진행로그/체크포인트 경로 반환.

### tool 계약 / 데이터 불변식 (Codex 리뷰 반영)
- **구조화 결과**: 모든 tool은 `{status: success|error, summary, artifacts[], checkpoint, warnings[],
  error_code?, recoverable?, suggested_next_tools?}` 형태. 표(marker/DE/benchmark)는 **행수 제한 + 미리보기**,
  전체는 CSV/PNG **artifact 경로(절대경로 + 메타)** 로. PNG는 호스트 파일시스템 가시성 차이 고려해 경로+메타 반환.
- **AnnData 불변식**: `layers["counts"]` 불변 / `.X`의 의미(정규화 여부)를 단계마다 기록 / 통합 임베딩은 `.obsm`에만 /
  모든 mutating tool은 `.uns["scpilot"]`에 provenance(파라미터·시드·버전) 기록.
- **레이어 규약(사용자 확정 2026-06-10)**: `layers["counts"]`=raw count(보존·불변), `layers["scale.data"]`=
  `normalize_total`+`log1p` 값(log-normalized; marker/annotation 계산용). scqc merged의 기존 규약과 일치.
- **차원축소 보존 규약(사용자 확정 2026-06-10)**: PCA/UMAP 등 **모든 reduction을 통합 전후·모델별로 전부 보존**(덮어쓰기 금지).
  obsm: `X_pca`·`X_umap`(baseline) / `X_harmony`·`X_umap_harmony` / `X_scVI`·`X_umap_scvi` …; obs: `leiden`·`leiden_<model>`;
  uns/obsp neighbors도 `neighbors_<model>`로 네임스페이스. (cluster 도구가 use_rep에서 suffix 자동 도출, baseline은 canonical.)
- **재현성**: 난수 시드 고정·기록. 회귀검증은 정확값이 아니라 **구조 불변식**(키 존재, shape 일치,
  클러스터 수 허용오차 내)으로.

### 재현성 하네스 (적극 사용 — 1급 기능)
LLM 주도 탐색은 비결정적이므로, **결정적으로 재현 가능한 "레시피"를 LLM 실행과 분리**해 기록·재생한다.
1. **시드/결정성 + 보장 등급**: 전역 시드 고정·기록(numpy, `sc.settings`, torch/scvi, random). UMAP/leiden·igraph/
   scVI/numba는 완전 비트동일이 어려움 → **per-tool 결정성 등급 명시**: (A) 파라미터·환경 동일, (B) 구조 동등(허용오차),
   (C) 가능시 비트동일. replay 비교는 **등급별 tolerance**로(정확값 X). R 단계 결정성도 등급으로 문서화.
2. **환경 캡처**: `doctor`가 전 의존성 버전 + `conda env export` + `pip freeze` 스냅샷. **R 도구 사용 시 `renv.lock` +
   `sessionInfo()` + replay 시 `renv::restore()`**.
3. **해싱 전략(6GB 회피)**: 기본은 **불변 입력파일 1회 해시 + 레시피(params)·lib버전·소스 체크포인트 ID·경량
   dataset fingerprint** 해시. 전체 h5ad content-hash는 **선택/백그라운드(아카이브용)**.
4. **Provenance / run log + 결정 이벤트**: 모든 mutating tool이 (params·seed·lib버전·입력/출력 체크포인트 ID)를
   append-only run log에 기록. **추가로 `decision` 이벤트를 1급으로**: LLM이 고른 통합방법·resolution·annotation 전략·
   compartment 분기·CNV fallback·trajectory 선택을 (후보·선택·근거·confidence·입력 요약 artifact ID·하위 params)로 기록.
   `.uns["scpilot"]`에는 **압축 포인터/현재상태만**(무한 증식 방지), 전체 로그·결정·대형 요약은 세션 파일에 artifact ID로.
5. **결정적 리플레이**: `scpilot replay <session>` — run log + **decision 이벤트를 소비**(LLM 재질의 X)해 그대로
   재실행 → 등급별 tolerance로 구조 diff. (= LLM 탐색 ↔ 재현 가능 레시피 분리.)
6. **테스트/회귀 하네스(pytest, 적극 사용)**: **각 tool 구현 시점마다** tiny fixture로 단위 + 구조불변식 테스트
   함께 작성(TDD식). CI 가능. step-by-step 빌드의 "검증" = 이 하네스.
7. **content-addressed 체크포인트**: 동일 입력+파라미터면 캐시 재사용·검증에 활용(위 경량 해시 기준).

---

## 아키텍처

```
scpilot/
  core/                # 순수 분석 함수 (AnnData in/out, LLM 비의존)
    io.py              # load_10x / load_h5ad / save; 입력형식 감지
    state.py           # AnnData 단계 감지 (어디까지 처리됐나) → 전체 파이프라인 재진입점 결정
    qc.py              # scrublet 더블릿, QC metric 계산, 필터 적용
    preprocess.py      # normalize_total, log1p, HVG, scale, PCA
    integrate.py       # harmony / scVI 실행 → obsm에 임베딩 저장
    benchmark.py       # scib-metrics로 통합 후보 비교 → 점수표
    cluster.py         # neighbors, leiden, umap
    markers.py         # rank_genes_groups → 클러스터별 pos/neg marker 표
    annotate.py        # Tier0-3 증거기반 annotation: major/fine/facs_style_label + malignancy + evidence/confidence/review
    review.py          # Tier 5 consistency/review: 라벨 모순·단일환자 지배·batch특이·CNV없는 malignancy → review_required
    compartment.py     # compartment subset → 재정규화·HVG 재선정·reclustering (재귀)
    cnv.py             # Tier 2 malignancy: CNV(infercnvpy)+tumor marker+normal-epi ref+patient expansion 통합 — 선택
    trajectory.py      # 궤적/분화 도구군 → obs["cell_state"] — 선택 (PAGA 기본, 나머지 가용성 게이트)
                       #   PAGA(coarse connectivity,scanpy) · Slingshot(R,cluster lineage) · Monocle3(R,pseudotime/graph)
                       #   scVelo(spliced/unspliced 필요,RNA velocity) · CellRank(velocity+fate prob)
                       #   Palantir(differentiation continuum) · CytoTRACE(분화 potential, malignant/epithelial 보조)
    de.py              # 조건비교 DE (pseudobulk sample단위 기본 + cell-level 탐색)
    plots.py           # umap/qc/dotplot/heatmap/pseudotime overlay → PNG 파일
    report.py          # 분석 산출물 + LLM 해석 → Markdown/HTML 리포트
  schemas.py           # 각 tool 입출력 JSON Schema (요약 반환 구조 정의)
  tools.py             # core 함수를 tool로 노출하는 단일 레지스트리 (이름/설명/스키마/핸들러)
  session.py           # 분석 세션 = 작업디렉토리 + 상주 AnnData + 이력 로그
  mcp_server.py        # FastMCP 서버: tools.py 레지스트리를 MCP tool로 등록
  llm/
    provider.py        # LLM provider 추상화 (기본=Claude/Anthropic; iS2C2식 확장지점)
    agent.py           # 에이전트 루프 (Anthropic tool-runner) + 단계별 system prompt
    prompts.py         # 오케스트레이션/annotation/해석/DE 설계용 프롬프트
  cli.py               # Typer CLI 엔트리포인트
  vendor/              # scqc_pipeline 베다링 (독립 진화) — VENDORING.md 참조
    harness.py         #   재현성 primitive: atomic_path/provenance/StageReport/init_runtime (+ scqc run_stage/Pipeline=참고용)
    config.py          #   PipelineConfig(profile+CLI 우선순위, per-stage config hash)
    io_10x.py          #   robust 10x 리더 (CellRanger v2/v3·gzip·h5 fallback)
    plotting.py        #   auto-fit 출판품질 figure 하네스
pyproject.toml         # 패키지/의존성/콘솔 스크립트 정의 (`scpilot`)
```

### LLM 계층 (Claude API) — **모드 2(선택적 자체구동) 전용**
- 1순위 통합은 MCP(모드 1)이고 거기선 호스트 에이전트가 LLM을 제공하므로 이 계층은 불필요.
  아래는 API 키를 가진 사용자가 `scpilot run`으로 자율 실행할 때만 동작.
- 기본 모델 **`claude-opus-4-8`**, `thinking={"type":"adaptive"}`, `output_config={"effort":"high"}`.
- 에이전트 루프는 Anthropic Python SDK **tool-runner**(`client.beta.messages.tool_runner`, `@beta_tool`) 사용 —
  tool 실행 → 결과 피드백 → 반복을 SDK가 처리. 핵심 단계(annotation 라벨 등)는 structured output 스키마로 강제.
- `provider.py`로 provider를 추상화해 iS2C2처럼 향후 Ollama/Gemini 추가 가능하게 두되, **1차 구현은 Claude만**.
- `ANTHROPIC_API_KEY`는 env에서 로드(코드에 하드코딩 금지). MCP 경로는 사용자의 Claude Code/Desktop이
  LLM이 되므로 API 키 불필요.

### Annotation 설계 — Tier 0–5 증거기반 (참고: 레포 내 `cancer_scrnaseq_annotation_strategy.md` — annotation 지식 카드 단일 원천)
**원칙**: `cell type + malignancy + cell state + trajectory + uncertainty = 최종 annotation 제안`.
**LLM은 증거 통합·감사 레이어이지 단독 annotation 권위가 아님** (각 tier 출력에 evidence_for/against·confounders·confidence·
review_required 동반). 흐름: QC/artifact → broad → malignant/non-malignant → compartment subcluster → fine(marker/ref) →
trajectory/state → **consistency review**.
- **Tier 0 QC/Artifact** — 저품질·doublet-like(예 EPCAM+CD3D 공발현)·ambient·dissociation/stress·mixed-lineage 클러스터 플래그.
- **Tier 1 Broad** — Epithelial/T·NK/B·Plasma/Myeloid/Stromal/Endothelial/Mast/Mixed-Artifact + marker 충돌 검출.
- **Tier 2 Malignancy** — epithelial marker만 의존 금지: **CNV burden + tumor marker + normal-epi reference 유사도 +
  patient-specific clonal expansion** 통합 → `malignancy ∈ {malignant, non_malignant, uncertain, not_applicable}`.
- **Tier 3 Compartment fine** — compartment별 세분(T/NK·Myeloid·Stromal·Epi/malignant·B/Plasma 패널). 무관 타입이
  같은 trajectory/label 공간에 섞이지 않게.
- **Tier 4 Trajectory/State** — **compartment 내에서만** 수행. pseudotime≠시간; tumor 궤적은 CNV clone·cell cycle·
  hypoxia·stress·batch 반영 가능 → 결과는 `cell_state`/`trajectory_state` 증거로만(타입 증명 X). 교란 overlay
  (patient·sample·batch·cell_cycle·stress·IFN·activation·doublet).
- **Tier 5 Consistency/Review** — 최종 테이블 감사: 동일 marker·다른 라벨 / 동일 라벨·모순 marker / 계층 모순
  (major=T·NK인데 fine=macrophage) / 단일 환자 지배 / batch 특이 / 고 doublet·stress / **CNV·tumor 증거 없는 malignancy** →
  `review_required` 플래그.

**메타데이터 스키마(obs, 분리 보관)**: `major_cell_type` / `fine_cell_type` / `facs_style_label`(예 `CD8+ PD-1+ T cells`) /
`malignancy` / `cell_state`(cycling·exhausted·EMT-like·hypoxia 등) / `trajectory_state` / `confidence` / `review_required`.
**evidence_for/against·confounders·cluster키·parent-child**는 `.uns["scpilot"]["annotation_tree"]`에. (FACS식=표시용,
biological 라벨=계산용 — 쌍으로.) 면역 워크플로(독립/ TME): CD45+ 선택 → broad lineage → lineage별 subcluster → subtype →
activation/exhaustion/cycling/IFN state 점수 → trajectory → review. **lineage와 state를 단일 비가역 라벨로 섞지 말 것.**

### 도메인 지식 / Skills 전략 (채택 — 단, 후반 추출 방식)
- **Tool ≠ Skill**: tool은 실행함수, skill은 LLM에게 *언제·어떻게 쓸지* 알려주는 도메인 지식 문서.
- **결정**: 강점(지식 외부화·편집성, 토큰 효율, MCP/CLI 일관성, 재사용)이 단점(전달경로 이원화,
  CLI 네이티브 런타임 부재, 유지보수 이중화, 초기 과설계)보다 큼 → **채택**. 단점은 **단일 소스+이중 전달**로 회피.
- **단일 소스**: `scpilot/knowledge/*.md` 에 지식 카드를 한 벌만 작성
  (예: `qc_heuristics.md`, `integration_metrics.md`, `de_design.md`, 그리고 **`annotation_strategy.md`·`cancer_markers.md`·
  `immune_markers.md`·`facs_labels.md`는 `cancer_scrnaseq_annotation_strategy.md`의 Tier 설계·marker 패널·FACS 매핑을
  그대로 카드화**).
- **이중 전달**: ① CLI 에이전트 = 해당 단계 카드를 system prompt에 주입(수동 progressive disclosure),
  ② Claude Code/MCP = 같은 카드를 `.claude/skills/`로 노출(네이티브 progressive disclosure).
- **시점**: MVP는 `prompts.py`만으로 시작(과설계 회피). 프롬프트 안정화 **후**(아래 Step 9)에 카드로 추출.

### 재사용할 기존 함수 (신규 구현 금지)
- QC/더블릿: `sc.pp.scrublet`, `sc.pp.calculate_qc_metrics`, `sc.pp.filter_cells/filter_genes`
- 전처리: `sc.pp.normalize_total`, `sc.pp.log1p`, `sc.pp.highly_variable_genes(flavor="seurat_v3")`,
  `sc.pp.scale`, `sc.pp.pca`
- 통합: `sc.external.pp.harmony_integrate` (harmonypy), `scvi.model.SCVI` (latent → `obsm["X_scVI"]`)
- 벤치마크: `scib_metrics.benchmark.Benchmarker` (batch correction + bio conservation 점수)
- 군집/임베딩: `sc.pp.neighbors`, `sc.tl.leiden`(igraph flavor), `sc.tl.umap`
- 마커/DE: `sc.tl.rank_genes_groups`, pseudobulk(메타데이터 groupby 합산) + 통계
- 그림: `sc.pl.umap/violin/dotplot` (matplotlib backend, 파일 저장)

---

## 분석 흐름 (LLM 오케스트레이션, 전체 파이프라인)

1. **load + state 감지** — 입력 h5ad/10x를 적재, `state.py`가 완료 단계 판단(raw/HVG/clustered/annotated).
2. **QC** — **scrublet은 per-sample 기본**(라이브러리별 doublet 분포 차이 → 병합본 단일 실행 금지; per-sample 분포
   반환·sample별 임계 허용, 결과를 병합본에 머지) + QC metric → **batch-aware 분포**(per-sample/per-batch quantile:
   `n_genes_by_counts`,`total_counts`,`pct_counts_mt`, doublet rate, 유지/제거 셀·유전자 수, outlier flag) 반환 →
   LLM 컷오프 결정 → 필터. (글로벌 컷오프가 sample/tissue 특이 생물학을 제거하지 않도록 배치 인지 필수.)
   (선택) **ambient RNA 평가/제거**(load 직후·정규화 전; raw droplet/background 정보 있을 때만, 없으면 "미수행" 기록
   + marker 해석 경고).
3. **전처리** — normalize/log1p/HVG(seurat_v3, **counts 레이어 필수·`scikit-misc` 의존**, batch-aware)/scale/PCA.
   분산설명비·HVG 개수 후보 요약 반환 → LLM이 HVG 개수·PC 수 결정. (count 정수성·레이어·의존성 preflight 게이트.)
4. **baseline 군집** — 통합 전 unintegrated PCA로 neighbors → leiden → umap (Tier 1 토대).
5. **Tier 1 annotation (= benchmark `label_key`)** — marker 기반 **major cell type** (broad). scib bio-conservation
   지표가 생물학적 라벨을 요구하므로 **반드시 benchmark 앞**. ⚠️ **순환참조·배치파편화 동시 회피**:
   unintegrated 단일 라벨에만 의존하면 강한 cross-GSE 배치로 같은 타입이 GSE별로 쪼개져 scib가 *배치 보존*을
   보상할 위험. → **consensus 라벨**: ①unintegrated marker 라벨 + ②celltypist advisory(가장 비순환적 anchor) +
   ③quick-Harmony marker 라벨의 일치도/confidence를 산출, **agreement 요약** 후 `obs["major_cell_type"]` 확정
   (라벨·confidence·unknown 허용). Harmony 유래 라벨을 쓰면 *Harmony 임베딩 평가시 label 기반 지표는 circular-risk로
   플래그/제외*. 이 major 라벨이 7단계 benchmark의 `label_key`.
6. **통합 후보 생성** — `sample_id`/`GSE`를 batch로 **unintegrated + Harmony + scVI(CPU)** 임베딩.
   scVI는 **진단용 기본**: counts-HVG, 20–50k stratified 서브샘플, 작은 latent, epoch 20–50, early stopping,
   runtime/peak-mem 기록. **fallback 정책 스키마**(시도 method·params·예외·경과·체크포인트·대체).
7. **통합 벤치마크** — `scib-metrics`(`label_key=major_cell_type`, `batch_key=sample_id/GSE`) **2-tier**,
   kNN 고비용 지표 기본 off, **batch contingency·조건별 조성·지표 분해 + overcorrection 경고** 반환 →
   LLM이 (집계점수 맹신 X, 생물학 보존 함께 보고) 최종 통합방법 선정.
8. **최종 군집/임베딩 (명시 단계)** — 선정 임베딩으로 neighbors → leiden(resolution 조정) → umap.
   compartment 재귀는 **이 최종 cluster 키 + 선정 임베딩 provenance를 필수 입력**으로 받는다.
9. **Tier 2/3 annotation — 컨텍스트 적응형 (LLM이 조직/질환에서 전략 결정)** — 핵심 확장.
   먼저 **compartment 계획 tool**: 실제 `obs` 카운트·sample/batch coverage·marker 증거를 반환 → LLM이 *실존하는*
   compartment만 분기(없는 compartment 환각 방지), 최소 셀/coverage 임계 미달 시 reclustering·HVG·세분 생략.
   공통 메커니즘 = **각 compartment subset → 재처리**. ⚠️ **배치 재유입 방지**: subset 재처리는 두 모드 분리 —
   (a) marker 발굴용 expression 재정규화·HVG, (b) **클러스터링은 integration-aware 임베딩**(전역 통합 보존). 각
   compartment마다 **batch-mixing 진단** 통과 후에야 fine 클러스터 채택. 작은 클러스터는 최소크기·merge·
   "insufficient evidence" 라벨 규칙.
   라벨은 **분리 저장(문서 스키마)**: `major_cell_type`·`fine_cell_type`·`facs_style_label`·`malignancy`·`cell_state`·
   `trajectory_state`·`confidence`·`review_required`. **권위 계층/증거는 `.uns["scpilot"]["annotation_tree"]`**
   (parent-child·cluster키·evidence_for/against·confounders·provenance); obs 컬럼은 셀 단위 접근용.
   - **면역(PBMC/림프절 등)**: T3 = subset 후 TF·function 기반 세분화 → FACS식(`CD8+ PD-1+ T cells`).
   - **암(예: PDAC)**: ① broad → ② **CNV 추론으로 malignancy 먼저 확정**(fine epithelial 라벨은 이 필드에서 파생,
     라벨 race 방지) → T/myeloid/fibroblast 등 compartment 분리 → ③ compartment별 (위 두 모드)재처리 →
     ④ **PAGA(기본)** 연결구조, 선택적으로 Slingshot/Monocle3(R) lineage → ⑤ 선택 방향성·분화:
     scVelo(**spliced/unspliced 있을 때만**)·CellRank(velocity 모드는 velocity 가용시만, non-velocity 모드 별도)·
     Palantir·CytoTRACE → ⑥ pseudotime에 marker·pathway·CNV·patient·treatment overlay →
     ⑦ **trajectory는 `cell_type_*` 아닌 `obs["cell_state"]`로만 저장**(type 먼저 기록 후 state; state가 type 오염 금지).
   - 궤적/분화/CNV 도구는 모두 **선택**: `doctor` 가용성 + **데이터/주석 조건 게이트**
     (scVelo→spliced/unspliced 없으면 `velocity_available=false`로 **하드 비활성**; CNV→`var`에 chromosome/start/end·
     genome build·정렬·중복 점검, 깨끗한 normal epithelial reference 없으면 immune/stromal·외부 reference·advisory-only·
     skip 중 택1). R 도구(Slingshot/Monocle3)는 subprocess+h5ad/CSV 교환 권장(rpy2보다), CytoTRACE는 구현(CytoTRACE2/
     Py·R) 하나로 고정 또는 미지원 표기. CNV는 epithelial/후보+reference로 **범위 한정** + 잡 모델 + runtime/mem 보고.
10. **Tier 5 consistency/review** — 최종 annotation 테이블 감사(동일marker·다른라벨 / 계층 모순 / 단일환자 지배 /
    batch특이 / 고 doublet·stress / CNV·tumor 증거 없는 malignancy) → `obs["review_required"]` + 리뷰 요약.
11. **조건비교 DE** — **DE 설계 점검 tool**(그룹크기·복제·교란) → **pseudobulk(sample 단위)** 기본,
    cell-level wilcoxon 탐색용. major_cell_type/fine_cell_type/compartment/cell_state 축으로 비교 가능.
12. **해석 + 리포트** — 그림(PNG) + 표 + LLM 해석(FACS식 라벨=표시, biological=계산) → Markdown/HTML 리포트.

---

## 사용 모드 (3종) & 인터페이스

핵심 요구: **Anthropic API 직접호출에 의존하지 않고, Claude Code·Codex 등 기존 CLI 에이전트가 호출 가능**해야 함.
→ **MCP 서버를 1순위 통합 지점**으로 삼는다. MCP는 표준 프로토콜이라 MCP 지원 에이전트
(Claude Code, Codex CLI, Cursor 등)면 모두 stdio로 우리 서버에 붙어 도구 호출 가능하고,
**LLM·크레덴셜은 호스트 에이전트가 제공** → 우리 쪽 API 키 불필요.
(도구가 "요약만 반환" 설계라 어떤 호스트 LLM이 구동해도 토큰 효율·재현성 유지.)

`pyproject.toml` 콘솔 스크립트 `scpilot`, 세 가지 모드:

- **모드 1 (1순위) — MCP 서버**: `scpilot mcp`
  표준 MCP(stdio) 서버 기동. **Claude Code·Codex·기타 MCP 클라이언트**가 붙어 대화형으로 도구 구동.
  우리 API 키 불필요(호스트가 LLM 제공). 향후 HTTP/SSE transport도 옵션으로 추가 가능.
- **모드 2 (선택) — 자체구동 CLI 에이전트**: `scpilot run <input> [--workdir] [--goal] [--effort high]`
  Anthropic API(`claude-opus-4-8`)로 8단계 자율 수행 + 리포트. **API 키 있을 때만**. 배치/재현 파이프라인용.
- **모드 3 — 결정론적 단일 단계**: `scpilot step <stage> <input>`
  특정 단계만 LLM 없이 실행. 디버그/회귀검증용.
- **모드 4 — 결정적 리플레이**: `scpilot replay <session>`
  기록된 run log(파라미터만, LLM 없이)를 그대로 재실행 → 원본과 구조불변식 diff. 재현성 하네스의 핵심.

### MCP 등록 설정 (호스트별)
> ⚠️ **A6 스파이크 실측(2026-06-10): `conda run -n scpilot scpilot mcp`는 stdio MCP에서 실패한다**
> ("Connection closed" — `conda run`이 기본적으로 자식 stdout을 캡처/버퍼링해 JSON-RPC 스트림을 끊음).
> → **직접 env 바이너리 경로 등록**(권장) 또는 `conda run --no-capture-output` 사용.

권장 서버 실행 커맨드(직접 경로, conda run 불필요): `/home/wykim/miniforge3/envs/scpilot/bin/scpilot mcp`

- **Claude Code**: `claude mcp add scpilot -- /home/wykim/miniforge3/envs/scpilot/bin/scpilot mcp`
  (또는 프로젝트 `.mcp.json`의 `mcpServers`에 `command`/`args` 등록).
- **Codex CLI** (codex-cli ≥0.137, `codex mcp add` 사용 — config.toml 자동 기록):
  ```bash
  codex mcp add scpilot -- /home/wykim/miniforge3/envs/scpilot/bin/scpilot mcp
  ```
  (`--env KEY=VALUE`는 stdio 서버에 추가 가능하나 scpilot MCP 경로는 API 키 불필요 → 보통 불필요.
  `--` 뒤는 **직접 바이너리**여야 함 — `conda run` 금지. 수동 편집 시 `[mcp_servers.scpilot] command/args` 동일.)
- 두 호스트 모두 동일 서버 바이너리를 stdio 서브프로세스로 띄우므로 **단일 구현으로 호환**.
- ✅ 프로토콜 호환 검증 완료(MCP SDK stdio 클라이언트=Claude Code/Codex와 동일 프로토콜): 직접 경로 2종 OK,
  conda run(캡처) 실패, `--no-capture-output` OK. **호스트 등록 후 실제 도구목록 인식 확인은 사용자 단계**(긴호출 취소·재연결 포함).

---

## To-Do List (step-by-step tool 빌드)

tool은 한 번에 하나씩 추가하고, **추가할 때마다 `scpilot step`(LLM 없이 결정론적)으로 검증** 후 다음으로 진행.
각 core 함수는 AnnData를 받아 처리하고 **요약 dict** 를 반환한다는 계약을 공통으로 따른다.
(체크 표기: `[ ]` 미착수 · `[~]` 부분 진행 · `[x]` 완료)

> **최우선 디리스크 5 (Codex)** — 현황(2026-06-10): ①scib `label_key`(Tier1 consensus) 유효성 ✅**PoC 검증 완료(2026-06-10)**: unintegrated 클러스터 27/35가 단일-GSE(>80%)
→ 순환참조 실재 확인. Harmony는 1/22로 해소. **→ Tier1은 per-cell marker score(배치무관 anchor) 기반, unint 클러스터 라벨 금지**
> ②MCP stdio 동작 ✅**프로토콜 검증 완료**(A6; `conda run` 캡처 함정 발견→직접경로 등록) / 잡모델·실호스트 등록은 추후 ③scVI CPU 서브샘플 실현성 ⏳**미검증(조기 PoC)**
> ④CNV preflight·reference 선택 ✅**설계+PoC 검증 완료**(B12-pre) ⑤run-log `decision` 스키마 완전성 ⏳**설계 미완(A7서 동결)**.
> → ④를 PoC로 깬 방식 그대로 ①·③도 조기 PoC로 검증.

### 진행 현황 + 재검토 (2026-06-10, 구현 착수 전 점검)

**✅ 완료된 사전작업 (코드 구현 전 토대)**
- env `scpilot` 생성(scRNAseq 복제) + 누락분 설치(scikit-misc/mcp/anthropic/celltypist/gtfparse/pybiomart). numpy 2.x 호환 검증.
- proto `git init`(branch main). 계획서 전역 rename(scpilot). 메모리 기록(프로젝트·scqc자산·블로커).
- **scqc_pipeline 연계 전략 확정**: upstream(io/qc/merge) 위임 + 하네스/io/plotting/metaschema **베다링**.
- **디리스크 ④ CNV 좌표 주석 설계+PoC 검증 완료**(B12-pre): protein-coding 커버리지 89.8% 실측.

**⏳ 남은 핵심 위험 (구현 중 조기 PoC/스파이크로 검증)**
- ① **Tier1 consensus 라벨**(과학적으로 가장 미묘): cross-GSE 배치 파편화↔scib 과보정 순환참조. 실데이터 PoC로 검증.
- ② **MCP 잡 모델**(Claude Code+Codex stdio start/poll/cancel): A6 스파이크로 조기 검증.
- ③ **scVI CPU 타이밍**(180k셀): 서브샘플 runtime/peak-mem 실측.

**🔧 열린 결정 2개 (착수 전/중 확정)**
1. ~~**scqc qc 확장 방식**~~ ✅**확정(2026-06-10, Codex 2차의견 반영): (b) scpilot 내부 소유.** scqc 원본 미수정(베다링=독립
   정합). B3에서 merged를 `obs[sample_id]`로 **그룹분리→그룹별 scrublet→doublet score를 merged에 기록**해 per-sample
   의미론 보존(merged 단일 scrublet 금지).
2. ~~**세션 모델**~~ ✅**확정(2026-06-10)**: 단일 `out_dir`(A3 구현 완료). 멀티클라이언트/lock 연기.

**📐 착수 전 보완 필요(미명세)**
- **재귀 레지스트리 ↔ 잡 모델 동거 설계**(C1 진짜 난점): 선형 `run_stage` 위에 compartment 재귀 + 장시간 잡을 어떻게 표현할지.
- **테스트 fixture 고정**: PDAC ~2k셀 서브샘플을 1회 생성·고정 → 전 B단계 회귀 기준.
- **annotation에 시간예산 집중**: B8(Tier1)·B13(Tier3)이 가치·난도 모두 최대. 프롬프트 추출(E1)은 그 후.

**🎯 권장 빌드 순서(MVP 임계경로)**: ✅Phase A 완료(A1~A7) → **B1~B7 순차** → ⛔**B7-B8 하드게이트: 디리스크① Tier1
consensus PoC**(cross-GSE 파편화↔scib 과보정) → B8 → ⛔**B8-B9 스파이크: 디리스크③ scVI CPU 타이밍** → B9~. **MVP 루프**
= merged 진입→전처리→cluster→Tier1→Harmony→benchmark→최종cluster→Tier3→report를 **MCP로 끝까지 1회**. CNV/trajectory/
scVI/DE·멀티클라이언트·E단계는 그 다음(과설계 회피).

**🧱 착수 직전 정비 완료(2026-06-10, Codex 2차의견 반영)**: ① `RunLogRecord.summary` 필드 추가 + `ReplayExecutor` 타입
명시(동결 스키마가 replay diff 지원). ② **MCP 서버를 `tools.REGISTRY` 기반 동적 빌드**로 — `@register` 한 줄이면 step·MCP·
replay 자동 노출(이중등록 제거). ③ **ToolSpec 잡 생명주기(start/status/cancel/result)는 B9 직전 하드게이트로 연기** —
B1~B7엔 장시간 도구 없음, scVI 실측 후 설계(과설계 회피). `long_running` 플래그만 자리표시.

### Phase A — 기반 + 위험 조기 검증 (LLM 무관)
- [x] **A1. 스캐폴딩** — ✅**완료(2026-06-10)**: `pyproject.toml`(console script `scpilot`, `pip install -e . --no-deps`로
      env 보호), 패키지 골격(core/ 16 stub + schemas/tools/session/mcp_server + llm/ + cli.py Typer 6 subcommand),
      **scqc primitive 베다링**(`scpilot/vendor/`: harness/config/io_10x/plotting, scqc@debef308 — `vendor/VENDORING.md` 참조).
      의존성 설치 완료. 검증: `scpilot version`·`--help`·vendor primitive import·전 stub import OK.
      **선택(Tier2/3·궤적)**: `celltypist`/`infercnvpy`/**`gtfparse`(infercnvpy GTF 좌표주석 필수 옵션의존 — 실측 확인)**/
      `scvelo`/`cellrank`/`palantir`/`cytotrace` + R(Slingshot/Monocle3) (있을 때만 도구 활성, `doctor`로 게이트).
      (현 scpilot env엔 celltypist·infercnvpy·gtfparse·pybiomart 설치 완료.)
- [x] **A2. 환경 preflight** — ✅**완료(2026-06-10, `scpilot/doctor.py`)**: `scpilot doctor`가 의존성 probe + 버전 +
      tiny smoke(normalize/log1p/**HVG seurat_v3**/pca) + capability 플래그를 **compact JSON(stdout 순수)**로 반환,
      `ok` 따라 exit 0/1. 실측: ok=true, cnv_available=true, velocity=false, r=true, smoke ok. **numpy 2.x 호환 조기 확인**,
      실패 시 actionable 가이드.
      **capability 플래그 산출**: `velocity_available`(spliced/unspliced 有),
      **`cnv_available` = infercnvpy import OK AND 좌표소스 가용(GTF 캐시 존재 OR `--gtf` 제공 OR biomart 도달+pybiomart)
      AND `var`에 매핑 가능 식별자(symbol/ensembl) 존재**(좌표 주석 단계 B12-pre 성공 가능성 게이트),
      `r_available`(R+renv: Slingshot/Monocle3), celltypist/cytotrace 가용성. LLM은 플래그 false면 해당 도구 선택 불가.
- [x] **A3. `session.py` (온디스크 1급)** — ✅**완료(2026-06-10, 단일 out_dir 확정)**: `Session`(create/open/save) +
      manifest(`session.json`: session_id·x_state·counts_fingerprint·checkpoints[]·stage) + 단계별 `.h5ad` 체크포인트
      (atomic) + append-only `run_log.jsonl`/`decisions.jsonl`(스키마는 A7서 동결) + provenance stamp(`.uns["scpilot"]`)
      + 불변식 헬퍼(counts 존재·fingerprint drift). 인메모리=캐시(lazy 체크포인트 로드). **열린결정#2 확정: 단일 out_dir,
      멀티클라이언트/lock 연기**(`.lock`은 forward-compat 마커만). 검증: `tests/test_session.py` 5 passed.
- [x] **A4. `schemas.py`** — ✅**완료(2026-06-10, 동결)**: `ToolResult`(status/summary/tables/artifacts/checkpoint/
      warnings/suggested_next_tools/**determinism_grade**/params/provenance/error_code/recoverable) + `Artifact`(절대경로+meta)
      + `TablePreview`(행수 캡+full CSV 포인터) + 잡 스키마(`JobStatus`/`FallbackAttempt`) + `success()`/`error()`/
      `artifact_csv/png()`/`table_preview()` 생성자 + `_sanitize`(numpy/NaN/Path→strict JSON). 표준 `ERROR_CODES`.
      검증: `tests/test_schemas.py` 7 passed (JSON 직렬화·numpy/NaN 정화·표 캡·artifact 절대경로·잡 스키마).
- [x] **A5. `cli.py` 골격 + `step`** — ✅**완료(2026-06-10)**: `tools.py` **최소 레지스트리**(`register`/`get`/`run`/
      `list_tools`, `ToolSpec` name/fn/mutating/long_running; 계약 `fn(session,**params)->ToolResult`; `inspect` 등록) +
      `cli step <stage> <input> [--workdir/-w][--param/-p k=v][--seed]` 디스패치(세션 생성→시드 핀→tool 실행→run_log 기록→
      ToolResult JSON, exit by status). 검증: `step inspect`로 실데이터 end-to-end(stdout 순수 JSON·run_log·manifest),
      미등록 stage 에러. `tests/test_tools.py` 5 passed. **B 도구는 이 레지스트리에 register하면 step/MCP/replay 자동 연결.**
- [x] **A6. MCP 최소 서버 조기 도입** — ✅**완료(2026-06-10)**: `mcp_server.py`(FastMCP)에 읽기전용 `inspect_h5ad_tool`
      + `scpilot_version`, `init_runtime()` 기동, stdout=프로토콜만·로그 stderr. `core/io.py inspect_h5ad`(backed='r', ToolResult).
      cli `mcp` + `__main__.py`. **MCP SDK stdio 클라이언트 end-to-end 검증**(`tests/test_mcp_server.py` 1 passed).
      **Codex(전역) + Claude Code(user scope) 실등록 + `✔ Connected` 확인.** 디리스크② 핵심 발견: `conda run`(캡처) 실패
      → 직접 env 바이너리 경로 등록(등록 섹션 갱신). (긴호출 취소·재연결은 잡 모델 C1에서.)
- [x] **A7. 재현성 하네스 토대** — ✅**완료(2026-06-10)**: `scpilot/repro.py`(전역 시드 제어 `set_global_seed`:
      numpy/random/torch/scvi; 경량 해시 `dataset_fingerprint`/`recipe_hash`; 등급별 tolerance 구조 diff `compare_summaries`
      A/B/C; `replay_session` 드라이버) + **`decision` 이벤트 스키마 + run-log 스키마 동결**(`schemas.DecisionEvent`/
      `RunLogRecord` + `DECISION_TYPES` + `validate_decision`) + `session.log_decision` 검증 연결 + `scpilot replay`(현재
      dry-run; executor는 tool 레지스트리 C1/A5서 연결) + pytest. **decision 스키마 동결 완료 → B11+ 진행 가능.**
      검증: `tests/test_repro.py` 7 passed(시드 결정성·해시·등급 tolerance·decision 검증·replay). provenance stamp는 A3(session).

### Phase B — core tool 단계별 구현 (annotation→benchmark 순서, 각 단계 = tool + step + MCP 동시 검증)
- [x] **B1. `core/io.py`** — ✅**완료(2026-06-10)**: `load_h5ad`/`save_h5ad` 헬퍼 + `inspect`(A6, ToolResult명 정렬) +
      `load` 도구(input→세션 캐시 적재, summary 반환) self-register. load_10x/merge는 scqc 소유. 검증: 실데이터 적재·tiny fixture.
- [x] **B2. `core/state.py`** — ✅**완료(2026-06-10)**: `detect_state` 도구 — backed='r'로 단계 감지
      (raw/normalized/hvg/pca/neighbors/clustered/umap/annotated 누적 플래그) → reentry_point. 검증: 처리완료본→umap/annotate,
      raw merged→normalized/preprocess(실데이터), clustered fixture. self-register.
- [x] **B3. `core/qc.py` (Tier 0 artifact)** — ✅**완료(2026-06-10)**: `qc_metrics`(counts에서 %MT/%ribo +
      **per-sample scrublet**(sample_key 그룹분리, <30셀 graceful skip) + **mixed-lineage EPCAM+CD3D 플래그** +
      **batch-aware 분포 요약**(global + per-sample 표)) + `qc_filter`(min_genes/max_pct_mt/max_doublet 컷오프·per-sample
      kept/removed). 둘 다 mutating·checkpoint. 검증: fixture 5 tests + 실데이터 서브샘플(35 samples). scqc 원본 미수정.
      (ambient RNA는 raw droplet 필요 — 추후.)
- [x] **B4. `core/preprocess.py`** — ✅**완료(2026-06-10)**: `preprocess` — counts에서 시작 normalize/log1p(+lognorm 레이어)
      /HVG(seurat_v3, batch-aware, counts·skmisc)/PCA(mask_var=HVG, 대용량 scale 회피) → variance_ratio·n_hvg·elbow 제안 요약.
      counts 불변·x_state=log1p 기록. invalid_state 게이트(counts 없으면). grade B(PCA). 검증: chain 테스트.
- [x] **B5. `core/plots.py` (scqc `plotting.py` 베다링)** — ✅**완료(2026-06-10)**: `plots` 도구
      (kind=umap/qc_violin/hvg/pca_variance/**dotplot**[sc.pl.dotplot에 마커패널 dict→세포타입 브라켓/라벨 자동]) —
      vendored `save_*`에 정책 config 적용(max 1.5×1.5 + **신규
      `square_limit_col=1.0`**으로 양쪽>1 금지), FitResult→Artifact(w/h in·dpi). 검증: 정책 테스트(1.5×1/1×1.5/1×1 허용,
      1.5×1.5 금지) + 실데이터 umap(180977셀→[0.75,0.5]col). scib/타패키지 동일 하네스 라우팅은 해당 도구 구현 시.
      아래 정책 **vendored `fit_and_save` auto-fit 하네스 위에** 구현: **plot 스타일 정책(사용자 확정 2026-06-10)**:
      ① 각 패키지 튜토리얼 스타일 그대로 — scanpy는 `sc.pl.*`, scib는 scib 자체 plotter, 나머지도 자기 plot 도구를 빌더로
      → 같은 `fit_and_save`에 태워 **동일 로직 공유**. ② 저장 크기 = 컬럼 단위 **min 0.5×0.5, max는 방향유연
      {1×1.5(세로)·1.5×1(가로)·1×1} — 양쪽 동시 >1 금지**(현 vendored 기본 h≤1.0을 이 제약으로 조정). ③ 잘림 발생 시
      축·텍스트·레전드·제목을 **능동 조절**(knob ladder)해 저장 크기 안에서 미리 layout한 figure처럼 구분 가능하게,
      고정 캔버스 저장(`bbox_inches='tight'` 미사용), 불만족 시 warning. 메타(가로/세로/dpi)는 `Artifact`로.
- [x] **B6. `core/cluster.py`** — ✅**완료(2026-06-10)**: `cluster` — neighbors→leiden(igraph)→umap on `use_rep`
      (X_pca 기본, 통합 임베딩도 동일 도구). cluster 수/크기 반환, invalid_state 게이트(임베딩 없으면). grade B. 검증: 구조 불변식.
- [x] **B7. `core/markers.py`** — ✅**완료(2026-06-10)**: `markers` — rank_genes_groups(Wilcoxon, lognorm 레이어) →
      클러스터별 top marker 표(미리보기) + 크기 + **sample 분포(단일환자 지배 클러스터 플래그)** + 전체 랭킹 CSV artifact. grade A.
- [x] **B8. `core/annotate.py` (Tier 1 broad)** — ✅**완료(2026-06-10, 사용자확정 로직)**: `annotate_broad`.
      **로직(사용자 확정)**: ① **leiden-cluster DE 기반**(`rank_genes_groups` Wilcoxon, pts=True) ② 마커 정의 =
      **pct≥0.25 AND LFC≥1** ③ 클러스터의 유의 마커를 broad 세포타입 패널(`BROAD_MARKERS`)과 **조합 매칭** ④ 세포타입 호출은
      **해당 패널 마커 ≥3개** 매칭 필수(미만=`Unknown`) ⑤ **샘플 출처 고려**(per-cluster sample/condition 조성, 단일샘플·
      단일배치 지배 플래그) ⑥ 결과 layout = **UMAP + dotplot**(`sc.pl.dotplot`에 패널을 dict로 줘 x축 상단 세포타입 브라켓/라벨
      자동). → `obs["major_cell_type","major_confidence"]`(confidence=매칭마커/패널크기) + evidence(matched_markers·candidates·
      provenance·flags)는 `.uns["scpilot_annotation"]["tier1"]`. grade A. 검증: fixture 6 tests(DE매칭·≥3규칙·single-source·
      dotplot 브라켓). **→ benchmark `label_key`=major_cell_type.** (구안의 per-cell score/celltypist consensus는 폐기.)
- [x] **B9. `core/integrate.py`** — ✅**완료(2026-06-10)**: `integrate_scvi`(**사전학습 모델 LOAD + get_latent, 학습 X →
      grade A·동기·잡모델 불필요**; 기본 `~/data/scpilot_run/models/scvi_GSM`, scvi 1.4.2 일치, n_latent 30) +
      `integrate_harmony`(harmonypy 직접호출 + `np.asarray(Z_corr).T` 우회). **카테고리 게이트**: 모델이 아는 batch(GSM 31개)
      외 샘플 있으면 data_gate_failed(비-PDAC 거부). 검증: fixture 5 tests + 실데이터(integrate_scvi→cluster(X_scVI)→
      X_scVI/X_umap_scvi/leiden_scvi 보존). 모델 scpilot_run/models로 vendoring. scVI>Harmony(벤치마크 확인).
      ⏳**B9b(후속, 별도게이트)**: 사전모델 없는 데이터용 scVI **학습** + ToolSpec 잡 생명주기 + 디리스크③ CPU 타이밍 —
      이 PDAC 데이터엔 학습 불필요라 디리스크③ 무의미.
      (구 항목) `harmony_integrate` + `scvi.model.SCVI(accelerator="cpu")` **잡 모델**. 검증(서브샘플): 임베딩·CPU 시간/peak-mem·fallback.
      `harmony_integrate` + `scvi.model.SCVI(accelerator="cpu")` **잡 모델**. 검증(서브샘플): 임베딩·CPU 시간/peak-mem·fallback.
      ⚠️**harmony API(PoC 발견)**: scanpy 1.11.5의 `sc.external.pp.harmony_integrate`는 harmonypy 0.2.0 **torch 출력과 비호환**
      (`Z_corr.T` shape 오류), `sc.pp.harmony_integrate`(native harmony2)는 1.11.5에 **없음** → **`harmonypy.run_harmony` 직접
      호출 + `np.asarray(ho.Z_corr).T`** (PoC2에서 shape 검증). scanpy 업그레이드 시 native harmony2로 전환 가능.
- [x] **B10. `core/benchmark.py`** — ✅**완료**: `benchmark`(scib-metrics, label_key=consensus/일관 라벨, batch_key,
      embeddings=[X_pca,X_harmony,X_scVI], drop_labels=sentinel+caller-set) + **교차-합의 `consensus_annotation`**
      (다중 통합법 라벨 majority vote→embedding-독립 label_key, 디리스크① 순환성 해소). overcorrection 경고. 검증: 통과.
- [ ] **B10.5. 최종 cluster (명시)** — 선정 임베딩으로 neighbors→leiden→umap, **final-cluster 키 + 임베딩 provenance**
      를 Tier2/3 입력으로 고정(B6 재사용).
- [ ] **B11. `core/compartment.py`** — **compartment 계획 tool**(실 카운트·coverage·marker, 임계 미달 분기 차단) +
      subset 재처리 **두 모드**(marker용 expression 재정규화·HVG / 클러스터링용 integration-aware) + batch-mixing 진단.
- [x] **B12-pre. `annotate_genomic_positions` (좌표 주석 — CNV의 필수 preflight 서브툴)** — ✅**완료(2026-06-12)**:
      tool+레지스트리+MCP 자동노출+검증 완료. content-addressed GTF 캐시(`SCPILOT_GTF_CACHE`, 기본 `~/data/scpilot_run/gtf_cache`)
      + 2-pass 매핑 + pc_coverage 게이트. **실데이터 재현(2026-06-12)**: pc_coverage 0.8982(17,981/20,020), make_unique 회수 63,
      25 chrom, gate_pass=True — PoC와 일치. 단위테스트 `tests/test_cnv.py` 3종(2-pass·저커버리지경고·missing-gtf). 설계(아래)는 그대로 구현:
      - **좌표 소스**: 기본 = **고정 릴리스 GENCODE GRCh38 GTF 1회 다운로드 → sha256 content-addressed 캐시·재사용**
        (결정성 등급 A; **GTF 경로는 `gtfparse` 필수 — 미설치 시 infercnvpy가 ImportError, 실측 확인**). 대체 = 사용자 `--gtf`
        (정렬 ref면 최선), offline 불가 시 biomart `hgnc_symbol`(등급 B, 네트워크·Ensembl 버전 의존으로 플래그).
        `--genome-build {GRCh38(기본),GRCh37}` provenance 기록.
      - **2-pass symbol 매핑(make_unique 안전 역연산)**: Pass1 = `var_names` 원본을 `gene_name`에 매칭(→ `HLA-A`/`NKX2-1`
        등 실제 하이픈 유전자 정상 매핑). Pass2 = Pass1 실패 + `^(.+)-\d+$` 패턴만 base symbol로 재매핑(중복 suffix 회수);
        그 외 미매핑은 `chromosome=NaN`(infercnvpy 자동 제외). **단순 trailing `-\d+` strip 금지**(실제 하이픈 유전자 훼손).
      - **chromosome 명명 `chr` prefix로 통일**(infercnvpy `exclude_chromosomes=('chrX','chrY')` 기본과 정합).
      - **커버리지 게이트 — 전체비율 아닌 protein-coding 커버리지로**(PoC로 정정): 전체 매핑률은 lncRNA/clone-contig
        미매핑에 눌려 낮게 보이므로 게이트 지표는 **`protein_coding_coverage` = (GTF protein_coding 중 데이터가 좌표 부여한 비율)**.
        `pc_coverage≥0.8` 정상 / `0.6~0.8` 경고(옛 심볼 drift·build 불일치 의심) / `<0.6` 강한 경고 + build 재확인·alias 해소·
        사용자 GTF 제안. ⚠️ **다중 GSE build/심볼버전 불일치**는 pc_coverage 저하로 감지.
      - **요약 반환**: `{n_genes_total, n_mapped, overall_fraction, protein_coding_coverage, make_unique_recovered,
        n_unmapped, unmapped_kind(noncoding/clone vs other), genome_build, source(type/name/sha256), reproducibility_grade,
        per_chromosome_gene_counts, unmapped_preview}`.
      - **(선택) 심볼 alias 해소 패스**: 미매핑 "other"의 상당수가 옛 HGNC 심볼(예 `AARS→AARS1`, `AAED1→PRXL2C`) — HGNC alias
        매핑으로 추가 회수 가능(필수 아님, 기본 PC 커버리지로 충분).
      - **✅ PoC 실측(2026-06-10, PDAC 40,237 symbol × GENCODE v44 basic GRCh38)**: pass1 53.4% + make_unique 회수 63 →
        전체 53.6%, 그러나 **protein_coding 커버리지 89.8%(17,981/20,020)**, 미매핑 18,688 중 91%가 noncoding/clone →
        **CNV 진행에 충분**. 25개 chromosome `chr` prefix 확인. GTF 다운로드 ~57s/29.6MB(sha 3e52f82c…).
      - **불변식**: `var` 컬럼 추가만(비파괴) — `layers["counts"]`·`.X` 의미 불변. provenance(GTF 해시·build·매핑률)는
        `.uns["scpilot"]`. replay는 등급별 tolerance.
- [x] **B12. `core/cnv.py` (Tier 2 malignancy, fine보다 선행)** — ✅**완료(2026-06-12)**. Tier-1과 동일한
      증거→LLM→적용 분리:
      ① `cnv_score`: infercnvpy `tl.infercnv`→cnv-space pca/neighbors/leiden→per-cell/per-cluster CNV burden +
         reference_key/reference_cat(None=advisory) + reference↔비reference contrast + 기존 라벨 교차표 (grade B).
      ② `malignancy_evidence`(read-only, grade A): group별 다축 증거 패키지 — reference 대비 CNV burden(ratio·
         frac_above_ref_q, **데이터 기반 상대값, 절대임계 없음**) + clonal expansion(top_sample_fraction) + 선택적
         caller marker score(하드코딩 패널 없음). 호출 없음.
      ③ `apply_malignancy`(grade A): LLM의 group→label 맵을 **고정 vocabulary** {malignant,non_malignant,uncertain,
         not_applicable}로 `obs["malignancy"]`+confidence+review_required에 기록. **HARD RULE 결정론적 강제**: CNV 증거
         없이 malignant 호출 시 review_required 강제. decision_type=`malignancy_call` 로깅.
      프롬프트 `MALIGNANCY_PROMPT`(agent system prompt에 배선) + 정식흐름 step 8. 검증: `tests/test_cnv.py` 11종
      (좌표·cnv_score·evidence 다축·vocab 거부·HARD RULE 강제 등). (잡 모델은 C1서, 현재 동기 long_running.)
- [ ] **B13. `core/annotate.py` (Tier 3 fine)** — compartment·malignancy에서 세분 → `obs["fine_cell_type"]` +
      `obs["facs_style_label"]`(예 `CD8+ PD-1+ T cells`) + evidence_for/against·confounders를 `.uns[...annotation_tree]`에.
      **LLM이 조직/질환 맥락으로 전략 결정**, 작은 클러스터 merge·insufficient-evidence 규칙.
- [ ] **B14. `core/trajectory.py` (Tier 4, 선택, MVP=PAGA만)** — **compartment 내에서만**. PAGA(scanpy) 기본.
      나머지(Slingshot/Monocle3=R subprocess, scVelo=spliced/unspliced 하드게이트, CellRank velocity/non-velocity 분리,
      Palantir, CytoTRACE=구현 고정)는 **experimental 플래그**. → `obs["cell_state"]`/`obs["trajectory_state"]`
      (type 오염 금지) + 교란 overlay(patient·cell_cycle·stress·IFN·doublet).
- [ ] **B14.5. `core/review.py` (Tier 5 consistency/review)** — 최종 annotation 테이블 감사: 동일marker·다른라벨 /
      계층 모순 / 단일환자 지배 / batch특이 / 고 doublet·stress / **CNV·tumor 증거 없는 malignancy** → `obs["review_required"]`.
- [ ] **B15. `core/de.py`** — **DE 설계 점검 tool**(그룹크기·복제·교란) + **pseudobulk(sample 단위)** 기본,
      cell-level wilcoxon 탐색용. major/fine/compartment/cell_state 축 비교.
- [ ] **B16. `core/report.py`** — PNG + 표 + 해석 텍스트 → Markdown/HTML.

### Phase C — tool 레지스트리 & 인터페이스 마감
- [ ] **C1. `tools.py`** — B1~B12를 단일 레지스트리로 노출 + **장시간 tool은 잡 인터페이스**
      (`start_*`/`get_job_status`/`get_job_result`/`cancel_job`).
- [ ] **C2. `mcp_server.py` 완성** — 전 tool 등록 + **QC/통합/annotation/DE 최소 tool-use 가이드(설명문) 동봉**
      (`qc_heuristics`·`integration_metrics` 핵심 기준은 Phase E를 기다리지 말고 여기서 최소판 포함).
      검증: Claude Code + Codex 양쪽 풀 워크플로 도구호출.
- [ ] **C3. `step` 완성** — 각 단계 결정론적 단독 실행(재현/디버그) 마무리.

### Phase D — LLM 에이전트 & 자율 실행 (모드 2, 선택)
> **📝 메모(2026-06-10 논의)**: 현재 **scpilot 자체가 LLM API를 호출하는 기능은 없음**(llm/ 전부 skeleton,
> `scpilot run` stub). 지금 "LLM 개입"은 **호스트(Claude Code/Codex)** 가 도구를 구동하는 경로(MCP=mode1 또는 Bash로
> `scpilot step` 구동)뿐. `scpilot step`(mode3)·`annotate_broad` 등은 **결정론적, LLM API 0**(ANTHROPIC_API_KEY 불필요).
> → **scpilot 단독으로(호스트 없이) API 키로 자율 실행**이 필요해지면 **이 Phase D를 구축**: `scpilot run`이 Anthropic API
> (`claude-opus-4-8`, tool_runner)를 직접 호출해 8단계 자율 수행. 빌드 시점은 MVP 루프(Tier1→통합→benchmark→report) 안정화 후.
- [ ] **D1. preflight** — `claude-opus-4-8`·`tool_runner` 가용성 확인, **모델명 설정화**(하드코딩 금지).
- [ ] **D2. `llm/provider.py`** — provider 추상화(기본 Claude/Anthropic).
- [ ] **D3. `llm/prompts.py`** — 단계별 system prompt(오케스트레이션/annotation/해석/DE 설계).
- [ ] **D4. `llm/agent.py`** — tool-runner 루프 + structured output(annotation 라벨, DE 설계 강제).
- [ ] **D5. `cli.py run`** — 전체 자율 실행 + 리포트. 검증: 서브샘플 풀런, 토큰·호출수 로깅.

### Phase E — 후속 확장 (선택)
- [ ] **E1. 지식 카드 추출(Skills)** — 안정화된 프롬프트를 `knowledge/*.md` 단일 소스로 추출,
      CLI는 프롬프트 주입 / Claude Code는 `.claude/skills/`로 이중 전달.
- [ ] **E2. 다운스트림 CCC** — iS2C2식 LIANA+ 기반 cell-cell communication 해석 모듈.

> **iS2C2 범위 구분**: 차용 = provider 추상화 + LLM 결과해석/리포트 패턴. **제외(E2까지 미룸)** = LIANA+/NicheNet
> 기반 CCC 자체. upstream 전체 파이프라인 + 통합 벤치마크는 iS2C2에 없는 우리 고유 확장.

---

## 검증 (End-to-End)

- **단계별(LLM 무관)**: 처리완료 파일로 회귀 — `scpilot step cluster PDAC_merged_qc_log1p_hvg_umap.h5ad`
  → **구조 불변식**(키 존재, shape 일치, 클러스터 수 허용오차, 시드 기록)으로 검증(정확값 비교 X — 결정성 등급 기준).
- **벤치마크 경로**: `PDAC_merged_qc.h5ad`(raw)에서 Harmony vs scVI vs unintegrated scib-metrics 점수표 산출.
  scVI는 CPU 모드라 느리므로 **서브샘플 + epoch 축소로 먼저 검증**, 시간/메모리 로깅 후 전량은 선택적.
  (GPU 추가 후 `accelerator="auto"`로 전환해 전량 재실행.)
- **MCP 경로(핵심)**: Claude Code **와 Codex CLI** 양쪽에 서버 등록 → 각 에이전트가 도구 목록을 인식하고
  대화로 QC→annotation까지 도구 호출이 도는지 확인(우리 API 키 없이 호스트 LLM으로).
- **CLI 자율 경로**: 소규모 서브샘플 h5ad로 `scpilot run` 풀 파이프라인 → 리포트(PNG+해석) 생성 확인.
  토큰 사용·tool 호출 횟수 로깅으로 비용 점검.
- **Annotation 타당성**: LLM 라벨을 알려진 PDAC marker(EPCAM/KRT, PTPRC, COL1A1 등)와 대조 검수.

## 미해결/기본값으로 진행한 결정 (구현 중 확정)
- LLM provider: 1차 **Claude만**, provider 추상화로 확장 여지 유지 (사용자가 iS2C2식 다중 provider 원하면 후속).
- scVI CPU 설정값(epoch/배치/서브샘플 크기)은 실측해 튜닝. GPU 추가 시 `accelerator="auto"`로 전환.
- Skills는 채택하되 Step E1(후반)에서 지식 카드로 추출 — 단일 소스 `knowledge/*.md`, CLI/Claude Code 이중 전달.
