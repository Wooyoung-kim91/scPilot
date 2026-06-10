"""Batch integration → integration embeddings in obsm — scpilot plan B9.

Two methods, both **synchronous** for this dataset (no job model needed):

- ``integrate_scvi`` — LOADS a pretrained scVI model (scvi_version 1.4.2, matches
  env) and applies ``get_latent_representation`` (FAST; no CPU training). For PDAC
  the model lives at .../integration_benchmark/scvi_model_GSM (batch_key=GSM, 2000
  HVGs, n_latent=30). Training a fresh model (datasets without a pretrained one)
  is future B9b and is where the job model + de-risk ③ apply.
- ``integrate_harmony`` — runs harmonypy directly (``sc.external.pp.harmony_integrate``
  is broken with harmonypy 0.2.0 torch output; ``sc.pp.harmony_integrate`` absent
  in scanpy 1.11.5) and stores ``np.asarray(Z_corr).T``.

All embeddings are kept per-model (obsm ``X_scVI`` / ``X_harmony``; never overwrite
``X_pca``) per the reductions-preservation convention. scVI was found superior to
Harmony on this dataset (see integration benchmark).
"""

from __future__ import annotations

import time
from pathlib import Path

from scpilot import schemas as S
from scpilot.tools import register

# Pretrained scVI model for PDAC (batch_key=GSM), vendored into the run dir.
# Override via param. (Falls back to the benchmark source if not yet copied.)
import os as _os
_RUN = _os.environ.get("SCPILOT_RUN_DIR", _os.path.expanduser("~/data/scpilot_run"))
DEFAULT_SCVI_MODEL = _os.path.join(_RUN, "models", "scvi_GSM")


def _model_batch_categories(model_pt: Path) -> list[str]:
    """Read the batch categories the scVI model was trained on (registry)."""
    import torch
    ck = torch.load(str(model_pt), map_location="cpu", weights_only=False)
    sr = ck["attr_dict"]["registry_"]["field_registries"]["batch"]["state_registry"]
    return [str(x) for x in sr.get("categorical_mapping", [])]


@register("integrate_scvi", mutating=True,
          description="Apply a PRETRAINED scVI model (load + get_latent, no training) → obsm['X_scVI']. "
                      "Primary integration for PDAC (scVI > Harmony in benchmark) (plan B9).")
def integrate_scvi(session, *, model_dir: str = DEFAULT_SCVI_MODEL, out_key: str = "X_scVI",
                   batch_key: str = "GSM", **params) -> S.ToolResult:
    import torch
    import scvi

    t0 = time.time()
    adata = session.adata
    mp = Path(model_dir) / "model.pt"
    if not mp.exists():
        return S.error("integrate_scvi", "missing_input", f"scVI model not found: {mp}", recoverable=False)

    genes = list(torch.load(str(mp), map_location="cpu", weights_only=False)["var_names"])
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        return S.error("integrate_scvi", "data_gate_failed",
                       f"{len(missing)}/{len(genes)} model HVGs absent in data (e.g. {missing[:3]}) — "
                       "input must contain the model's training genes", recoverable=False)
    if batch_key not in adata.obs.columns:
        return S.error("integrate_scvi", "data_gate_failed",
                       f"batch_key '{batch_key}' absent in obs (model was trained on it)", recoverable=False)

    # a pretrained model can only score cells whose batch (e.g. GSM) it was trained on
    # (this PDAC model = 31 PDAC samples). Out-of-model samples (e.g. non-PDAC) can't be
    # integrated by this model — gate explicitly rather than failing cryptically in scvi.
    known = set(_model_batch_categories(mp))
    data_cats = set(adata.obs[batch_key].astype(str).unique())
    extra = sorted(data_cats - known)
    if extra:
        in_model = int(adata.obs[batch_key].astype(str).isin(known).sum())
        return S.error("integrate_scvi", "data_gate_failed",
                       f"{len(extra)} batch value(s) not in the model (e.g. {extra[:3]}); "
                       f"{in_model}/{adata.n_obs} cells are in-model. This model covers {len(known)} "
                       f"{batch_key} samples — subset the data to those (e.g. PDAC-only) or retrain (B9b).",
                       recoverable=True, summary={"out_of_model_samples": extra,
                                                  "n_in_model_cells": in_model, "n_cells": int(adata.n_obs)})

    # subset (genes only — same cells/order) so latent rows align back to the full adata
    sub = adata[:, genes].copy()
    if "counts" not in sub.layers:
        return S.error("integrate_scvi", "invalid_state", "no 'counts' layer — scVI needs raw counts",
                       recoverable=False)
    model = scvi.model.SCVI.load(str(model_dir), adata=sub)
    z = model.get_latent_representation()
    adata.obsm[out_key] = z

    summary = {
        "method": "scvi_pretrained", "out_key": out_key, "batch_key": batch_key,
        "model_dir": str(model_dir), "n_latent": int(z.shape[1]),
        "n_cells": int(adata.n_obs), "n_model_genes": len(genes),
        "embeddings_present": sorted(adata.obsm.keys()),
        "note": "scVI ranked above Harmony on this dataset's integration benchmark",
    }
    cp = session.checkpoint("integrate_scvi", x_state=session.manifest.x_state,
                            params={"model_dir": str(model_dir), "out_key": out_key, "batch_key": batch_key})
    return S.success("integrate_scvi", summary=summary, checkpoint=cp.path, determinism_grade="A",
                     duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["cluster", "benchmark"])


@register("integrate_harmony", mutating=True,
          description="Harmony integration via harmonypy (direct call, torch-output workaround) → obsm['X_harmony'] "
                      "(plan B9). Baseline/candidate; scVI is primary on this dataset.")
def integrate_harmony(session, *, batch_key: str = "GSM", use_rep: str = "X_pca",
                      out_key: str = "X_harmony", seed: int = 0, **params) -> S.ToolResult:
    import numpy as np
    import harmonypy

    t0 = time.time()
    adata = session.adata
    if use_rep not in adata.obsm:
        return S.error("integrate_harmony", "invalid_state",
                       f"'{use_rep}' absent — run preprocess (PCA) first", recoverable=True,
                       suggested_next_tools=["preprocess"])
    if batch_key not in adata.obs.columns:
        return S.error("integrate_harmony", "data_gate_failed", f"batch_key '{batch_key}' absent in obs",
                       recoverable=False)

    ho = harmonypy.run_harmony(adata.obsm[use_rep], adata.obs, [batch_key], random_state=seed)
    Z = np.asarray(ho.Z_corr)                      # harmonypy 0.2.0 torch → numpy
    Z = Z.T if Z.shape[0] == adata.obsm[use_rep].shape[1] else Z   # (cells × dims)
    adata.obsm[out_key] = Z

    summary = {
        "method": "harmony", "out_key": out_key, "batch_key": batch_key,
        "use_rep": use_rep, "n_dims": int(Z.shape[1]), "n_cells": int(adata.n_obs),
        "embeddings_present": sorted(adata.obsm.keys()),
    }
    cp = session.checkpoint("integrate_harmony", x_state=session.manifest.x_state,
                            params={"batch_key": batch_key, "use_rep": use_rep, "out_key": out_key, "seed": seed})
    return S.success("integrate_harmony", summary=summary, checkpoint=cp.path, determinism_grade="B",
                     duration_s=round(time.time() - t0, 3),
                     suggested_next_tools=["cluster", "benchmark"])
