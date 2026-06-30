"""Common structured-result contract — scpilot plan A4 (frozen interface).

Every core tool (``scpilot.core.*``, wrapped by ``scpilot.tools``) returns a
``ToolResult``. The LLM/orchestrator reads only this small JSON — never the
AnnData. Big tables/figures are written to disk and referenced as ``Artifact``s;
tables are returned as a capped ``TablePreview`` with the full data in a CSV
artifact. Long-running tools use the job schema (``JobStatus`` / ``FallbackAttempt``).

Conventions (kept stable so B-tools and the MCP server agree):
- ``status`` ∈ {"success","error"}; on error set ``error_code`` (+ ``recoverable``).
- ``summary`` = the decision-relevant numbers the LLM reasons over (must be small).
- ``artifacts`` = absolute paths + metadata (PNG/CSV/h5ad); host filesystem may
  differ, so always return path + meta, never bytes.
- ``checkpoint`` = absolute path to the post-tool .h5ad checkpoint (mutating tools).
- ``determinism_grade`` ∈ {"A","B","C"} per plan A7 (A=params/env equal,
  B=structural-equivalent within tolerance, C=bit-identical when possible).
- ``provenance``/``params`` are lightweight pointers; the full run log + frozen
  ``decision`` events live in the session files (plan A7), not inline here.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# ---- frozen vocabularies (Literal, not Enum, so dataclasses.asdict stays JSON-clean) ----
Status = Literal["success", "error"]
ArtifactKind = Literal["csv", "png", "svg", "h5ad", "json", "txt", "html", "other"]
DeterminismGrade = Literal["A", "B", "C"]
JobState = Literal["pending", "running", "succeeded", "failed", "cancelled"]

# Standard error codes (tools may use these; free-form str also allowed).
ERROR_CODES = (
    "missing_input",          # required input/arg absent
    "invalid_state",          # AnnData not at the expected pipeline stage
    "capability_unavailable", # doctor capability flag is false
    "data_gate_failed",       # data-level precondition unmet (e.g. no spliced layer)
    "dependency_missing",     # optional package not installed
    "convergence_failed",     # algorithm did not converge / produced empty result
    "cancelled",              # job cancelled by caller
    "timeout",                # job exceeded its budget
    "internal",               # unexpected exception
)

MAX_PREVIEW_ROWS = 20  # default cap for inline table previews


# --------------------------------------------------------------------------- #
# Artifacts & table previews
# --------------------------------------------------------------------------- #
@dataclass
class Artifact:
    """A file written to disk, referenced by absolute path + metadata (never bytes)."""
    path: str
    kind: ArtifactKind = "other"
    description: str = ""
    meta: dict = field(default_factory=dict)   # e.g. {n_rows,n_cols} for csv; {w,h,dpi} for png

    def __post_init__(self) -> None:
        self.path = str(Path(self.path).resolve())
        st = Path(self.path)
        if st.exists():
            self.meta.setdefault("bytes", st.stat().st_size)


@dataclass
class TablePreview:
    """A capped view of a table; full data lives in a CSV ``Artifact`` (``full``)."""
    columns: list[str]
    rows: list[list]                 # up to n_shown rows
    n_rows_total: int
    n_rows_shown: int
    truncated: bool
    full: str | None = None          # absolute path to the full CSV, if written


def table_preview(
    df, *, max_rows: int = MAX_PREVIEW_ROWS, full_csv: str | None = None
) -> TablePreview:
    """Build a ``TablePreview`` from a pandas DataFrame (rows capped at ``max_rows``)."""
    n_total = int(len(df))
    head = df.head(max_rows)
    rows = head.where(head.notna(), None).values.tolist()
    return TablePreview(
        columns=[str(c) for c in df.columns],
        rows=rows,
        n_rows_total=n_total,
        n_rows_shown=int(len(head)),
        truncated=n_total > max_rows,
        full=str(Path(full_csv).resolve()) if full_csv else None,
    )


# --------------------------------------------------------------------------- #
# Job model (long-running tools: start_* / get_job_status / get_job_result / cancel_job)
# --------------------------------------------------------------------------- #
@dataclass
class FallbackAttempt:
    """One attempt in a fallback chain (plan §fallback policy)."""
    method: str
    params: dict = field(default_factory=dict)
    status: Literal["succeeded", "failed", "skipped"] = "failed"
    error: str | None = None
    elapsed_s: float | None = None
    checkpoint: str | None = None


@dataclass
class JobStatus:
    """Status of a long-running job (polled via get_job_status)."""
    job_id: str
    tool: str
    state: JobState = "pending"
    progress: float | None = None        # 0..1 when known
    message: str = ""
    started_at: str = ""
    updated_at: str = ""
    elapsed_s: float = 0.0
    peak_mem_mb: float | None = None
    log_path: str | None = None
    checkpoint: str | None = None
    attempts: list = field(default_factory=list)   # list[FallbackAttempt]


# --------------------------------------------------------------------------- #
# The core tool result
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    """Uniform return value of every core tool. JSON-serializable via ``to_dict()``."""
    tool: str
    status: Status = "success"
    summary: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)        # name -> TablePreview
    artifacts: list = field(default_factory=list)     # list[Artifact]
    checkpoint: str | None = None
    warnings: list = field(default_factory=list)      # list[str]
    suggested_next_tools: list = field(default_factory=list)
    determinism_grade: DeterminismGrade | None = None
    params: dict = field(default_factory=dict)        # resolved params (lightweight)
    provenance: dict = field(default_factory=dict)    # version/seed/checkpoint-id pointers
    duration_s: float | None = None
    # error fields (set when status == "error")
    error_code: str | None = None
    error: str | None = None
    recoverable: bool | None = None

    def to_dict(self) -> dict:
        """Recursively convert to a JSON-serializable dict (handles numpy/Path/set)."""
        return _sanitize(dataclasses.asdict(self))


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #
def success(
    tool: str,
    *,
    summary: dict | None = None,
    tables: dict | None = None,
    artifacts: list | None = None,
    checkpoint: str | None = None,
    warnings: list | None = None,
    suggested_next_tools: list | None = None,
    determinism_grade: DeterminismGrade | None = None,
    params: dict | None = None,
    provenance: dict | None = None,
    duration_s: float | None = None,
) -> ToolResult:
    return ToolResult(
        tool=tool, status="success",
        summary=summary or {}, tables=tables or {}, artifacts=artifacts or [],
        checkpoint=checkpoint, warnings=warnings or [],
        suggested_next_tools=suggested_next_tools or [],
        determinism_grade=determinism_grade, params=params or {},
        provenance=provenance or {}, duration_s=duration_s,
    )


def error(
    tool: str,
    error_code: str,
    message: str,
    *,
    recoverable: bool = True,
    summary: dict | None = None,
    warnings: list | None = None,
    suggested_next_tools: list | None = None,
) -> ToolResult:
    return ToolResult(
        tool=tool, status="error", error_code=error_code, error=message,
        recoverable=recoverable, summary=summary or {}, warnings=warnings or [],
        suggested_next_tools=suggested_next_tools or [],
    )


def artifact_csv(path: str, *, n_rows: int | None = None, n_cols: int | None = None,
                 description: str = "", **meta_fields) -> Artifact:
    meta: dict = {}
    if n_rows is not None:
        meta["n_rows"] = int(n_rows)
    if n_cols is not None:
        meta["n_cols"] = int(n_cols)
    meta.update(meta_fields)
    return Artifact(path=path, kind="csv", description=description, meta=meta)


def artifact_png(path: str, *, width_in: float | None = None, height_in: float | None = None,
                 dpi: int | None = None, description: str = "") -> Artifact:
    meta: dict = {}
    if width_in is not None:
        meta["width_in"] = round(float(width_in), 3)
    if height_in is not None:
        meta["height_in"] = round(float(height_in), 3)
    if dpi is not None:
        meta["dpi"] = int(dpi)
    return Artifact(path=path, kind="png", description=description, meta=meta)


# --------------------------------------------------------------------------- #
# Reproducibility harness schemas — scpilot plan A7 (FROZEN before B11+)
# --------------------------------------------------------------------------- #
# Canonical decision types the LLM/orchestrator records. Free-form str is allowed,
# but new recurring kinds SHOULD be added here so replay/audit stay consistent.
DECISION_TYPES = (
    "qc_cutoff",                # QC thresholds (min_genes/max_pct_mt/...)
    "hvg_npcs",                 # n HVGs / n PCs
    "integration_method",       # unintegrated | harmony | scvi
    "clustering_resolution",    # leiden resolution
    "tier1_consensus_label",    # broad major_cell_type consensus (de-risk ①)
    "compartment_branch",       # which compartments to recurse into
    "malignancy_call",          # malignancy: malignant/non/uncertain
    "cnv_reference",            # reference cells chosen for inferCNV
    "cnv_fallback",             # CNV fallback path taken
    "trajectory_method",        # PAGA | slingshot | ... (gated)
    "annotation_strategy",      # tissue/disease-adaptive strategy choice
    "de_design",                # pseudobulk vs cell-level, grouping
)

# Required keys for a decision event (validated by validate_decision).
_DECISION_REQUIRED = ("decision_type", "choice", "candidates", "rationale")


@dataclass
class DecisionEvent:
    """A first-class LLM decision (plan A7). Append to the session decision log.

    FROZEN shape — recorded so ``scpilot replay`` can re-run a pipeline WITHOUT
    re-querying the LLM (it consumes the recorded ``choice``/``params``).
    """
    decision_type: str                       # ∈ DECISION_TYPES (or free str)
    choice: Any                              # selected option (str or dict)
    candidates: list = field(default_factory=list)   # options considered (name + metrics)
    rationale: str = ""                      # short reasoning
    confidence: float | None = None          # 0..1
    input_summary_ref: str | None = None     # artifact/checkpoint id the call was based on
    params: dict = field(default_factory=dict)       # downstream params implied by the choice
    stage: str | None = None                 # tool/stage that prompted it
    alternatives_rejected: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return _sanitize(dataclasses.asdict(self))


@dataclass
class RunLogRecord:
    """One mutating-tool run in the append-only run log (plan A7). FROZEN shape.

    ``summary`` carries the tool's structural-invariant numbers so ``replay`` can
    diff a re-run against the recorded values without re-loading checkpoints.
    """
    tool: str
    status: Status = "success"
    stage: str | None = None
    params: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)   # structural invariants for replay diff
    seed: int | None = None
    input_checkpoint: str | None = None
    output_checkpoint: str | None = None
    determinism_grade: DeterminismGrade | None = None
    recipe_hash: str | None = None
    lib_versions: dict = field(default_factory=dict)
    duration_s: float | None = None
    error_code: str | None = None

    def to_dict(self) -> dict:
        return _sanitize(dataclasses.asdict(self))


@dataclass
class OutputRecord:
    """One step's OUTPUTS bound to its WHY — appended to ``outputs.jsonl``.

    The single per-step record that binds ``[step → params → artifacts → reasoning →
    provenance]`` so the report, the generated pipeline, and audit can answer "which
    artifacts did this step produce, with what params, and why?". Kept SEPARATE from the
    frozen ``RunLogRecord`` (the run log stays a pure replayable recipe); this index adds
    output provenance without inflating that contract.

    ``artifacts`` are ``Artifact``-shaped dicts whose ``meta`` carries ``sha256`` + ``bytes``
    (integrity/audit, NOT replay equality — figure bytes are non-deterministic). ``reasoning``
    is the caller/LLM's prose WHY for this step (optional).
    """
    tool: str
    status: Status = "success"
    stage: str | None = None
    n: int | None = None                              # session run index when recorded
    recipe_hash: str | None = None
    seed: int | None = None
    params: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)     # [{path, kind, description, meta(+sha256,bytes)}]
    reasoning: str | None = None
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return _sanitize(dataclasses.asdict(self))


def validate_decision(d: dict) -> list[str]:
    """Return a list of problems (empty == valid) for a decision-event dict."""
    problems = [f"missing required key: {k}" for k in _DECISION_REQUIRED if k not in d]
    if "confidence" in d and d["confidence"] is not None:
        c = d["confidence"]
        if not isinstance(c, (int, float)) or not (0.0 <= float(c) <= 1.0):
            problems.append("confidence must be in [0,1]")
    if "candidates" in d and not isinstance(d["candidates"], list):
        problems.append("candidates must be a list")
    return problems


# --------------------------------------------------------------------------- #
# JSON sanitizer (numpy scalars/arrays, Path, set, non-finite floats)
# --------------------------------------------------------------------------- #
def _sanitize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None      # JSON has no NaN/Inf
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    # numpy without a hard import: 0-d scalar -> .item(), ndarray -> .tolist()
    if obj.__class__.__module__ == "numpy":
        ndim = getattr(obj, "ndim", None)
        if ndim == 0 and hasattr(obj, "item"):
            return _sanitize(obj.item())
        if hasattr(obj, "tolist"):
            return _sanitize(obj.tolist())
        return str(obj)
    return str(obj)  # last resort: stringify (keeps JSON valid)
