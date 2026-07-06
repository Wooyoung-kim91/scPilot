"""Evidence-based organism + gene-symbol resolution — scpilot NEVER assumes human.

scPilot's design rule is no hardcoded species/tissue priors (see the no-hardcoding
principle): deterministic tools must read what the data actually IS and report that
evidence, leaving the biological call to the reasoning layer.

These helpers let the same deterministic code work for human (ALL-CAPS symbols,
``MT-`` mito genes) and mouse (Title-case symbols, ``mt-``) without branching on a
hardcoded organism. ``detect_organism`` INFERS the organism from the gene names in
the data; ``resolve`` maps any reference symbol to the data's actual casing
(``EPCAM`` -> ``Epcam`` in mouse) — orthologous symbols differ only by casing, so a
case-insensitive lookup is a normalization, not a species assumption.
"""

from __future__ import annotations

import re

# Ensembl gene-ID format across species (ENSG human, ENSMUSG mouse, ENSDARG zebrafish, …),
# optionally version-suffixed ("…​.3"). This is a FORMAT check on the identifiers themselves,
# not a species/biology assumption — the same spirit as reading symbol casing.
_ENSEMBL_GENE_RE = re.compile(r"^ENS[A-Z]{0,5}G\d{6,}(\.\d+)?$")

# var columns that, WHEN ALREADY PRESENT IN THE DATA, carry gene symbols. We only CHOOSE among
# columns the object already ships (CELLxGENE writes symbols to ``feature_name``); no gene
# biology is hardcoded — this is a column-name convention list, not a marker/panel.
_SYMBOL_COL_CANDIDATES = (
    "feature_name", "gene_symbols", "gene_symbol", "gene_name", "gene_names",
    "Symbol", "symbol", "symbols", "SYMBOL", "hgnc_symbol", "mgi_symbol",
)


def looks_like_ensembl(var_names, *, min_frac: float = 0.5) -> bool:
    """True if ``var_names`` are predominantly Ensembl gene IDs (a format check, not a species guess)."""
    import pandas as pd
    s = pd.Index([str(x) for x in var_names]).to_series()
    if s.empty:
        return False
    return float(s.str.match(_ENSEMBL_GENE_RE).mean()) >= min_frac


def _usable_symbols(col):
    """Boolean mask of var rows whose symbol value is a non-empty, non-'nan' string."""
    s = col.astype("string")
    return s.notna() & (s.str.len() > 0) & (s.str.upper() != "NAN")


def normalize_var_symbols(adata, *, min_coverage: float = 0.5) -> dict:
    """Remap Ensembl-ID ``var_names`` → gene symbols read from the data's OWN var column.

    Evidence-based, no hardcoded biology (§1): the symbol source is a column already present on
    the object; we only pick WHICH existing column to use. Original IDs are preserved in
    ``var['gene_ids']``. Idempotent and a no-op when var_names are already symbols or no usable
    symbol column exists. Returns an evidence dict; callers surface it in ``warnings``/``uns``.

    Motivation: CELLxGENE stores Ensembl IDs (``ENSG…``) as var_names with symbols in
    ``feature_name``. Left as-is, ``MT-``/``RPS`` prefix matching finds nothing → ``pct_counts_mt``
    silently 0 and marker/CNV symbol lookups degrade, with no error. This normalizes at entry.
    """
    import pandas as pd
    v = adata.var_names
    if not looks_like_ensembl(v):
        return {"remapped": False, "reason": "not_ensembl", "n_vars": int(len(v))}
    chosen = None
    for c in _SYMBOL_COL_CANDIDATES:
        if c in adata.var.columns and float(_usable_symbols(adata.var[c]).mean()) >= min_coverage:
            chosen = c
            break
    if chosen is None:
        return {"remapped": False, "reason": "ensembl_but_no_symbol_column",
                "var_columns": [str(c) for c in adata.var.columns], "n_vars": int(len(v))}
    ids = pd.Index([str(x) for x in v])
    if "gene_ids" not in adata.var.columns:
        adata.var["gene_ids"] = list(ids)
    ok = _usable_symbols(adata.var[chosen])
    filled = adata.var[chosen].astype("string").where(ok, pd.Series(list(ids), index=adata.var.index))
    adata.var_names = pd.Index([str(x) for x in filled])
    adata.var_names_make_unique()
    return {"remapped": True, "symbol_column": chosen, "n_vars": int(len(v)),
            "n_symbols_missing_kept_as_id": int((~ok).sum())}


def detect_organism(adata) -> dict:
    """Infer organism from gene-name casing + mito-gene style. Returns evidence, never
    a hardcoded guess. ``organism`` ∈ {"human", "mouse", "unknown"}.

    Human gene symbols are ALL-CAPS (``EPCAM``, ``MT-ND1``); mouse are Title-case
    (``Epcam``, ``mt-Nd1``). The mito prefix is the strongest single signal.
    """
    v = adata.var_names
    n = int(len(v))
    if n == 0:
        return {"organism": "unknown", "n_genes": 0, "frac_symbols_uppercase": 0.0,
                "mito_prefix": "MT-", "evidence": "no genes"}
    # Ensembl IDs are ALL-CAPS by format (ENSG…/ENSMUSG…) and would be misread as "human" by
    # casing while matching no ``MT-`` gene — defer instead of guessing (see normalize_var_symbols).
    if looks_like_ensembl(v):
        return {"organism": "unknown", "n_genes": n, "frac_symbols_uppercase": 1.0,
                "mito_prefix": "MT-",
                "evidence": "var_names are Ensembl IDs — call normalize_var_symbols first to map to symbols"}
    frac_upper = float(v.str.isupper().mean())
    has_mt_lower = bool(v.str.startswith("mt-").any())
    has_mt_upper = bool(v.str.startswith("MT-").any())

    if has_mt_lower and not has_mt_upper:
        organism, why = "mouse", "mito genes use 'mt-' (mouse style)"
    elif has_mt_upper and not has_mt_lower:
        organism, why = "human", "mito genes use 'MT-' (human style)"
    elif frac_upper >= 0.7:
        organism, why = "human", f"{frac_upper:.0%} of symbols are ALL-CAPS"
    elif frac_upper <= 0.2:
        organism, why = "mouse", f"only {frac_upper:.0%} of symbols are ALL-CAPS (Title-case dominant)"
    else:
        organism, why = "unknown", f"ambiguous casing ({frac_upper:.0%} ALL-CAPS), no clear mito style"

    return {"organism": organism, "n_genes": n,
            "frac_symbols_uppercase": round(frac_upper, 3),
            "mito_prefix": "mt-" if organism == "mouse" else "MT-",
            "evidence": why}


def _symbol_index(adata) -> dict:
    """Case-insensitive map UPPER(symbol) -> actual var_name (first occurrence wins)."""
    idx: dict[str, str] = {}
    for name in adata.var_names:
        key = str(name).upper()
        if key not in idx:
            idx[key] = name
    return idx


def resolve(adata, symbols, *, index: dict | None = None) -> dict:
    """Resolve reference gene symbols to the data's actual var_names, case-insensitively.

    Returns ``{requested_symbol: actual_var_name_or_None}`` so callers see exactly which
    references are present (and under what casing) without assuming an organism.
    """
    idx = index if index is not None else _symbol_index(adata)
    return {s: idx.get(str(s).upper()) for s in symbols}


def present(adata, symbols, *, index: dict | None = None) -> list:
    """The subset of ``symbols`` present in the data, returned as the data's actual names."""
    return [v for v in resolve(adata, symbols, index=index).values() if v is not None]
