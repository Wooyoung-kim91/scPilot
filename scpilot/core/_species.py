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
