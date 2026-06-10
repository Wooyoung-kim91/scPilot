# =====================================================================
# VENDORED FROM scqc_pipeline @ source_hash debef308904633e1 (2026-06-10)
# scpilot 베다링: import만 scpilot.vendor로 적응. (A) end-to-end 흡수용.
# =====================================================================
"""Metadata schema layer: load → column-map → harmonize → filter → derive.

All of this is profile-driven so it works for any per-sample metadata table, not
just PDAC. Order is fixed (harmonize before filter/derive) so canonicalized values
drive filtering and stratification — a `Tumor`-labelled sample is not dropped by a
`condition == PDAC` filter just because of surface wording.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from scpilot.vendor.config import PipelineConfig


# --------------------------------------------------------------------------- #
# value normalization (case / whitespace / punctuation insensitive matching)
# --------------------------------------------------------------------------- #
def normalize_value(v: Any) -> str:
    s = "" if v is None else str(v)
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)   # punctuation → space
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- #
# condition evaluation for filters / derive rules
# --------------------------------------------------------------------------- #
def _cond_mask(df: pd.DataFrame, cond: dict) -> pd.Series:
    col = cond["col"]
    op = cond.get("op", "eq")
    val = cond.get("value")
    if col not in df.columns:
        # missing column → no rows match (treated as False)
        return pd.Series(False, index=df.index)
    s = df[col].astype(str)
    if op == "eq":
        return s == str(val)
    if op == "ne":
        return s != str(val)
    if op == "eq_ci":
        return s.map(normalize_value) == normalize_value(val)
    if op == "ne_ci":
        return s.map(normalize_value) != normalize_value(val)
    if op == "in":
        return s.isin([str(x) for x in val])
    if op == "not_in":
        return ~s.isin([str(x) for x in val])
    if op == "in_ci":
        targets = {normalize_value(x) for x in val}
        return s.map(normalize_value).isin(targets)
    if op == "notna":
        return s.str.len() > 0
    raise ValueError(f"Unknown filter op: {op}")


def _all(df: pd.DataFrame, conds: list[dict]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for c in conds:
        mask &= _cond_mask(df, c)
    return mask


# --------------------------------------------------------------------------- #
# load + filter + harmonize + derive
# --------------------------------------------------------------------------- #
def load_metadata(cfg: PipelineConfig) -> pd.DataFrame:
    return pd.read_csv(cfg.metadata_csv, dtype=str).fillna("")


def apply_filters(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    info = {"n_in": len(df)}
    include = cfg.filters.get("include", [])
    exclude = cfg.filters.get("exclude", [])
    if include:
        df = df[_all(df, include)].copy()
    if exclude:
        # drop rows matching ANY exclude condition
        drop = pd.Series(False, index=df.index)
        for c in exclude:
            drop |= _cond_mask(df, c)
        df = df[~drop].copy()
    info["n_out"] = len(df)
    return df, info


def apply_harmonize(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    """Map raw values to canonical per profile `harmonize`. Preserves <field>__raw.

    Returns (df, report) where report flags unmapped values for human review.
    """
    report = {"applied": {}, "unmapped": {}}
    if not cfg.harmonize:
        return df, report

    batch_col = cfg.batch_col

    for field, mapping in cfg.harmonize.items():
        if field not in df.columns:
            continue
        # normalized synonym → canonical
        syn = {}
        for canonical, raws in mapping.items():
            for raw in raws:
                syn[normalize_value(raw)] = canonical
            syn[normalize_value(canonical)] = canonical  # canonical maps to itself

        raw_col = f"{field}__raw"
        if raw_col not in df.columns:
            df[raw_col] = df[field]

        applied: dict[str, str] = {}
        unmapped: dict[str, dict] = {}
        new_vals = []
        for idx, raw in df[field].items():
            batch = str(df.at[idx, batch_col]) if batch_col in df.columns else ""
            # per-dataset override takes precedence
            over = cfg.harmonize_overrides.get(batch, {}).get(field, {})
            osyn = {}
            for canonical, raws in over.items():
                for r in raws:
                    osyn[normalize_value(r)] = canonical
            n = normalize_value(raw)
            canon = osyn.get(n) or syn.get(n)
            if canon is None:
                u = unmapped.setdefault(str(raw), {"batches": set(), "n_samples": 0})
                u["batches"].add(batch)
                u["n_samples"] += 1
                new_vals.append(raw)  # pass raw through (flagged), do not silently drop
            else:
                applied[str(raw)] = canon
                new_vals.append(canon)
        df[field] = new_vals
        report["applied"][field] = applied
        if unmapped:
            report["unmapped"][field] = {
                k: {"batches": sorted(v["batches"]), "n_samples": v["n_samples"]}
                for k, v in unmapped.items()
            }
    return df, report


def apply_derive(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Ordered label-derivation rules (profile-driven). Empty list = pass-through."""
    for rule in cfg.derive:
        rtype = rule.get("type")
        if rtype == "relabel":
            mask = _all(df, rule.get("where", []))
            for col, val in rule["set"].items():
                df.loc[mask, col] = val
        elif rtype == "case":
            target = rule["target"]
            df[target] = rule.get("default", "")
            for case in rule.get("cases", []):
                mask = _all(df, case.get("when", []))
                df.loc[mask, target] = case["value"]
        elif rtype == "alias":
            src = rule["source"]
            if src == "__batch__":
                src = cfg.batch_col
            df[rule["target"]] = df[src]
        elif rtype == "const":
            df[rule["target"]] = rule["value"]
        elif rtype == "isin_flag":
            src = df[rule["source"]].astype(str)
            present = src.isin([str(x) for x in rule["values"]])
            df[rule["target"]] = present.map(
                {True: rule.get("true_value", "True"),
                 False: rule.get("false_value", "False")}
            )
        else:
            raise ValueError(f"Unknown derive rule type: {rtype}")
    return df


def build_metadata(cfg: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    """Full pipeline: load → harmonize → filter → derive. Returns (df, report)."""
    df = load_metadata(cfg)
    df, harm = apply_harmonize(df, cfg)
    df, filt = apply_filters(df, cfg)
    df = apply_derive(df, cfg)
    report = {
        "n_samples": int(len(df)),
        "filter": filt,
        "harmonize": harm,
        "unmapped_total": sum(len(v) for v in harm.get("unmapped", {}).values()),
    }
    return df, report


# --------------------------------------------------------------------------- #
# stratifier auto-detection + same-partition dedup
# --------------------------------------------------------------------------- #
# GEO/SRA bookkeeping tokens — columns whose name contains any of these are
# excluded from AUTO stratifier detection (profiles can still list them explicitly).
_NOISE_SUBSTR = (
    "accession", "biosample", "sra", "series_", "contact", "supplementary",
    "_file", "file_", "protocol", "date", "submission", "platform", "taxid",
    "data_row", "data_processing", "web_link", "characteristics", "source_name",
    "description", "relation", "consent", "assembly", "bytes", "bases",
    "experiment", "center", "datastore", "filetype", "version", "channel_count",
    "molecule", "extract", "growth", "library", "query", "geo", "study",
    "organism", "strategy", "selection", "layout", "sample name", "sample_name",
    "title", "case_id", "barcode", "matrix_dir",
)


def _is_noise_column(name: str) -> bool:
    low = name.lower()
    if low.endswith("_id") or low.endswith("_ids"):
        return True
    return any(tok in low for tok in _NOISE_SUBSTR)


def detect_stratifiers(obs: pd.DataFrame, cfg: PipelineConfig) -> dict:
    """Pick categorical columns suitable for stratified QC plots.

    Returns {"low": [(col, n_levels)], "high": [(col, n_levels)], "dropped": {...}}.
    Columns inducing an identical cell partition are de-duplicated (keep first).
    """
    max_levels = cfg.stratify.get("max_facet_levels", 12)
    candidates = cfg.stratify.get("candidates")
    # internal bookkeeping columns are never biologically meaningful stratifiers
    internal = {"barcode", "matrix_dir", "sample_id", "sample_id_from_concat"}
    explicit = candidates is not None
    if candidates is None:
        # auto-detect: drop GEO/SRA bookkeeping noise so messy public metadata
        # doesn't explode into dozens of meaningless grouping axes.
        candidates = [c for c in obs.columns
                      if (obs[c].dtype == object or str(obs[c].dtype) == "category")
                      and c not in internal
                      and not _is_noise_column(c)]
    sample_col = cfg.sample_id_col

    low, high, dropped = [], [], {}
    seen_partitions: dict[tuple, str] = {}
    # seed the per-sample partition so any low-card column identical to it
    # (e.g. matrix dir, sample title) is dropped as redundant with the sample id
    if sample_col in obs.columns:
        seen_partitions[tuple(pd.factorize(obs[sample_col].astype(str))[0].tolist())] = sample_col
    for col in candidates:
        if col not in obs.columns or col.endswith("__raw"):
            continue
        n = obs[col].nunique(dropna=False)
        if n < 2:
            dropped[col] = f"single level ({n})"
            continue
        if col == sample_col:
            high.append((col, int(n)))
            continue
        if n > max_levels:
            dropped[col] = f"too many levels ({n} > {max_levels})"
            continue
        # partition signature: factorized codes (order-independent of label text)
        codes = tuple(pd.factorize(obs[col].astype(str))[0].tolist())
        if codes in seen_partitions:
            dropped[col] = f"same partition as {seen_partitions[codes]}"
            continue
        seen_partitions[codes] = col
        low.append((col, int(n)))
    return {"low": low, "high": high, "dropped": dropped}
