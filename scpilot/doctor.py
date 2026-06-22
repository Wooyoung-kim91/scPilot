"""Environment / capability preflight — scpilot plan A2.

``scpilot doctor`` probes the dependency stack, runs a tiny functional smoke test,
and emits **capability flags** as compact JSON (the scqc-style "deterministic
intelligence -> small JSON" pattern). The LLM / orchestrator reads these flags and
must NOT select a tool whose capability is ``false``.

Design notes:
- Env-level flags only (no dataset needed). Some tools have an additional *data*
  gate evaluated at runtime (e.g. velocity needs spliced/unspliced layers, CNV
  needs mappable gene identifiers) — those are noted in ``data_gated`` and checked
  by the tools themselves, not here.
- No network calls (kept deterministic/fast). Biomart reachability is a runtime
  concern of the CNV coordinate step.
"""

from __future__ import annotations

import importlib
import importlib.metadata as _md
import importlib.util
import platform
import shutil
import sys

# Packages required for the core pipeline (missing -> ok=False + actionable warning).
_REQUIRED = [
    "scanpy", "anndata", "numpy", "pandas", "scipy",
    "leidenalg", "igraph", "harmonypy", "scvi", "scrublet",
    "scib_metrics", "skmisc", "matplotlib", "seaborn",
    "typer", "mcp", "anthropic",
]
# Optional packages — enable specific (gated) tools when present.
_OPTIONAL = [
    "celltypist", "infercnvpy", "gtfparse", "pybiomart",
    "scvelo", "cellrank", "palantir", "cytotrace", "cellhint",
]
# import name -> distribution name (for version lookup) when they differ.
_DIST = {"scvi": "scvi-tools", "skmisc": "scikit-misc", "igraph": "igraph"}

# Per-tool HARD dependency modules (import names). A tool with a missing entry here
# runs its capability gate (plan D1): a missing package becomes a recoverable
# ``capability_unavailable`` ToolResult instead of a raw ImportError mid-execution.
CAPABILITY_REQUIRES = {
    "integrate_scvi": ["scvi", "torch"],
    "train_scvi": ["scvi", "torch"],
    "integrate_harmony": ["harmonypy"],
    "annotate_genomic_positions": ["infercnvpy"],
    "cnv_score": ["infercnvpy"],
    "benchmark": ["scib_metrics"],
}


def _findable(modname: str) -> bool:
    """True if ``modname`` is importable — cheap, no side effects (no real import)."""
    try:
        return importlib.util.find_spec(modname) is not None
    except Exception:  # noqa: BLE001 — a broken parent package counts as not findable
        return False


def check_capability(tool: str) -> tuple[bool, list[str]]:
    """Lightweight presence probe for a tool's hard deps (no full ``run()``).

    Returns ``(ok, missing_modules)``; a tool with no entry in ``CAPABILITY_REQUIRES``
    is always ``(True, [])``.
    """
    missing = [m for m in CAPABILITY_REQUIRES.get(tool, []) if not _findable(m)]
    return (not missing, missing)


def _probe(modname: str) -> dict:
    """Import a module without raising; return {present, version, error}."""
    try:
        importlib.import_module(modname)
    except Exception as exc:  # noqa: BLE001
        return {"present": False, "version": None, "error": f"{type(exc).__name__}: {exc}"[:120]}
    try:
        version = _md.version(_DIST.get(modname, modname))
    except Exception:  # noqa: BLE001
        version = "?"
    return {"present": True, "version": version, "error": None}


def _smoke() -> dict:
    """Tiny functional check: the previously-fragile numpy-2.x / HVG-seurat_v3 path."""
    out = {}
    try:
        import numpy as np
        import anndata as ad
        import scanpy as sc

        rng = np.random.default_rng(0)
        X = rng.poisson(1.0, size=(120, 200)).astype("float32")
        a = ad.AnnData(X)
        a.layers["counts"] = a.X.copy()
        sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=50, layer="counts")
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.pca(a, n_comps=10)
        out["normalize_log1p_hvg_seurat_v3_pca"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["normalize_log1p_hvg_seurat_v3_pca"] = f"fail: {type(exc).__name__}: {exc}"[:160]
    return out


def run() -> dict:
    """Build the doctor report dict (JSON-serializable)."""
    from scpilot import __version__
    from scpilot.vendor.harness import init_runtime

    init_runtime()  # pin numba/matplotlib caches (detached-session safety)

    pkgs = {m: _probe(m) for m in _REQUIRED + _OPTIONAL}
    present = {m: v["present"] for m, v in pkgs.items()}

    # numpy 2.x check
    np_ver = pkgs["numpy"]["version"]
    np_major = int(np_ver.split(".")[0]) if np_ver and np_ver[0].isdigit() else None

    # GPU (scVI runs CPU on this host; flag for when GPU is added)
    torch_cuda = False
    try:
        import torch
        torch_cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass

    r_present = bool(shutil.which("R") and shutil.which("Rscript"))

    # ---- capability flags ----
    capabilities = {
        "preprocess_seurat_v3": present["skmisc"],            # HVG seurat_v3 (needs scikit-misc)
        "doublet_scrublet": present["scrublet"],
        "integrate_harmony": present["harmonypy"],
        "integrate_scvi": present["scvi"],
        "benchmark_scib": present["scib_metrics"],
        "cluster_leiden": present["leidenalg"] and present["igraph"],
        "annotate_celltypist": present["celltypist"],
        "harmonize_cellhint": present["cellhint"],            # optional label-vocab alignment (else consensus)
        # CNV (malignancy track): infercnvpy + at least one coordinate source (GTF via gtfparse, or biomart)
        "cnv_available": present["infercnvpy"] and (present["gtfparse"] or present["pybiomart"]),
        "velocity_available": present["scvelo"],              # + data gate: spliced/unspliced layers
        "trajectory_cellrank": present["cellrank"],
        "trajectory_palantir": present["palantir"],
        "trajectory_cytotrace": present["cytotrace"],
        "trajectory_paga": present["scanpy"],                 # PAGA ships with scanpy (MVP default)
        "r_available": r_present,                             # Slingshot/Monocle3 (R pkgs unchecked)
        "mode2_llm_agent": present["mcp"] and present["anthropic"],
    }
    # tools whose true availability ALSO depends on the dataset at runtime
    data_gated = {
        "velocity_available": "needs spliced/unspliced layers",
        "cnv_available": "needs mappable gene identifiers in var (symbol/ensembl)",
    }

    smoke = _smoke()

    # ---- mode-2 LLM provider preflight (plan D1) ----
    # Configurable backend, no hardcoded model name; non-fatal (mode 2 is optional).
    try:
        from scpilot.llm.provider import probe_backend
        llm_provider = probe_backend()
    except Exception as exc:  # noqa: BLE001
        llm_provider = {"ready": False, "reason": f"probe failed: {exc}"}
    capabilities["mode2_llm_ready"] = bool(llm_provider.get("ready"))

    # ---- warnings (actionable) ----
    warnings = []
    missing_req = [m for m in _REQUIRED if not present[m]]
    for m in missing_req:
        warnings.append(f"required package missing: {m} -> conda run -n scpilot pip install {_DIST.get(m, m)}")
    if np_major is not None and np_major < 2:
        warnings.append(f"numpy {np_ver} < 2.x — env was verified on numpy 2.x; mismatch risk")
    if not capabilities["cnv_available"]:
        warnings.append("cnv_available=false: need infercnvpy + (gtfparse or pybiomart) for CNV (malignancy track)")
    if not torch_cuda:
        warnings.append("no CUDA GPU: scVI runs CPU-only (subsample + reduced epochs); set accelerator='auto' after GPU")
    if "fail" in smoke.get("normalize_log1p_hvg_seurat_v3_pca", ""):
        warnings.append("core smoke test FAILED — see smoke.* ; the analysis stack is not functional")

    ok = (not missing_req) and (np_major is None or np_major >= 2) \
        and all("fail" not in v for v in smoke.values())

    return {
        "ok": ok,
        "scpilot_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np_ver,
        "packages": {m: pkgs[m]["version"] for m in pkgs},
        "missing_required": missing_req,
        "capabilities": capabilities,
        "data_gated": data_gated,
        "gpu": {"torch_cuda": torch_cuda},
        "smoke": smoke,
        "llm_provider": llm_provider,
        "warnings": warnings,
    }
