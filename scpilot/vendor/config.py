# =====================================================================
# VENDORED FROM scqc_pipeline @ source_hash debef308904633e1
#   source: /home/wykim/data/PDAC/scqc_pipeline/ (copied 2026-06-10)
# scpilot 베다링 정책: 독립 진화. import 경로·provenance 키·uns 키만
#   scpilot으로 적응했고 로직은 원본 유지. 재동기화 절차/원본 대비 diff는
#   scpilot/vendor/VENDORING.md 참조. scpilot 고유 코드는 여기 두지 말 것.
# =====================================================================
"""Pipeline configuration: dataclass defaults + profile(YAML) + CLI overrides.

Precedence (high → low): CLI flag  >  --config/--profile YAML  >  dataclass default.
All dataset-specific values (column names, filters, label/harmonize rules,
expected oracle numbers, plotting) live in a profile YAML — never hardcoded here.
The dataclass defaults reproduce the original PDAC notebook behaviour so the tool
still works with an empty profile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# Bump when the meaning/shape of stage outputs changes. Embedded in every output
# so a stale on-disk artifact produced by an older code version is detected as
# dirty (see harness.is_fresh) rather than silently reused.
SCHEMA_VERSION = "1.0"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins). Returns new dict."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class PipelineConfig:
    """Resolved configuration for one pipeline run.

    Scalar fields are the QC/HVG/clustering knobs (overridable by CLI flags).
    Dict fields carry the dataset profile (column mapping, filters, harmonize
    rules, derive rules, stratify, plotting, expected oracle, output names).
    """

    # ---- profile identity ----
    profile_name: str = "default"

    # ---- paths ----
    input_root: str = ""
    metadata_csv: str = ""
    out_dir: str = "."

    # ---- column mapping (generic; PDAC profile sets GSM/local_matrix_dir/GSE) ----
    sample_id_col: str = "sample_id"
    matrix_dir_col: str = "matrix_dir"
    batch_col: str = "batch"

    # ---- QC thresholds ----
    min_genes: int = 200
    max_pct_mt: float = 20.0
    min_cells: int = 3
    target_sum: float = 1e4
    mito_prefix: str = "MT-"
    normalized_layer: str = "scale.data"

    # ---- HVG / embedding ----
    n_top_genes: int = 2000
    hvg_flavor: str = "seurat_v3"
    hvg_batch_key: str = ""          # profile maps to e.g. integration_batch
    n_neighbors: int = 15
    n_pcs: int = 30
    leiden_resolution: float = 0.5
    random_state: int = 0
    umap_color: list = field(default_factory=lambda: ["leiden"])

    # ---- behaviour flags ----
    strict_harmonize: bool = False

    # ---- dict-shaped profile blocks ----
    filters: dict = field(default_factory=dict)        # {include: [...], exclude: [...]}
    harmonize: dict = field(default_factory=dict)       # {field: {canonical: [raw,...]}}
    harmonize_overrides: dict = field(default_factory=dict)  # {batch_value: {field: {...}}}
    derive: list = field(default_factory=list)          # ordered label-derivation rules
    stratify: dict = field(default_factory=lambda: {"candidates": None, "max_facet_levels": 12})
    plotting: dict = field(default_factory=dict)
    subset: dict = field(default_factory=dict)          # {column, values, out}
    outputs: dict = field(default_factory=dict)         # filenames per stage
    expected: dict = field(default_factory=dict)        # validate oracle

    # ---------- construction ----------
    @classmethod
    def from_profile(cls, profile_path: str | None, overrides: dict | None = None) -> "PipelineConfig":
        """Build config: dataclass defaults ← profile YAML ← CLI overrides (non-None)."""
        base = asdict(cls())
        if profile_path:
            with open(profile_path) as fh:
                prof = yaml.safe_load(fh) or {}
            prof.setdefault("profile_name", Path(profile_path).stem)
            base = _deep_merge(base, prof)
        if overrides:
            clean = {k: v for k, v in overrides.items() if v is not None}
            base = _deep_merge(base, clean)
        known = {f for f in asdict(cls())}
        unknown = set(base) - known
        if unknown:
            raise ValueError(f"Unknown profile/override keys: {sorted(unknown)}")
        return cls(**base)

    # ---------- derived paths ----------
    @property
    def out(self) -> Path:
        return Path(self.out_dir)

    @property
    def input_root_path(self) -> Path:
        """Absolute input root; relative profile values resolve against out_dir."""
        p = Path(self.input_root)
        return p if p.is_absolute() else (self.out / p)

    @property
    def per_sample_dir(self) -> Path:
        return self.out / "per_sample"

    @property
    def qc_dir(self) -> Path:
        return self.out / "qc"

    @property
    def reports_dir(self) -> Path:
        return self.qc_dir / "reports"

    @property
    def fig_dir(self) -> Path:
        return self.out / "figures"

    @property
    def code_dir(self) -> Path:
        return self.out / "code"

    def output_path(self, key: str, default: str) -> Path:
        """Resolve a named output filename (profile can override)."""
        return self.out / self.outputs.get(key, default)

    # ---------- hashing for checkpoint invalidation ----------
    # Which config keys each stage actually depends on lives in _STAGE_DEP_KEYS
    # below. Changing an unrelated key (e.g. plotting) must NOT invalidate qc/merge.
    def stage_config_hash(self, stage: str) -> str:
        relevant = _STAGE_DEP_KEYS.get(stage, [])
        d = asdict(self)
        subset = {k: d.get(k) for k in relevant}
        blob = json.dumps(subset, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_public_dict(self) -> dict:
        """Serializable view (for reports / repro scripts)."""
        return asdict(self)


# Per-stage config dependency keys (drives fingerprint invalidation).
_COMMON_META = [
    "input_root", "metadata_csv", "sample_id_col", "matrix_dir_col", "batch_col",
    "filters", "harmonize", "harmonize_overrides", "derive",
]
_STAGE_DEP_KEYS: dict[str, list[str]] = {
    "metadata": _COMMON_META,
    "qc": _COMMON_META + ["min_genes", "max_pct_mt", "mito_prefix"],
    "merge": _COMMON_META + ["min_genes", "max_pct_mt", "mito_prefix",
                             "min_cells", "target_sum", "normalized_layer"],
    "qc-plots": ["stratify", "plotting"],
    "visualize": ["n_top_genes", "hvg_flavor", "hvg_batch_key", "n_neighbors",
                  "n_pcs", "leiden_resolution", "random_state", "normalized_layer",
                  "umap_color", "plotting"],
    "report": _COMMON_META,
    "subset": _COMMON_META + ["target_sum", "normalized_layer", "subset"],
}
