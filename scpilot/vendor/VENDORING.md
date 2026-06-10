# Vendored from scqc_pipeline

These modules are **copied** from `/home/wykim/data/PDAC/scqc_pipeline/` and then
evolve independently inside scpilot (decision 2026-06-10: *vendoring*, not a
library dependency). This file records provenance and the re-sync procedure.

## Provenance
- Source: `/home/wykim/data/PDAC/scqc_pipeline/`
- scqc `source_hash`: **`debef308904633e1`**
- Copied: **2026-06-10**

## Files vendored
| file | upstream | adaptation applied |
|---|---|---|
| `harness.py` | scqc `harness.py` | imports → `scpilot` / `scpilot.vendor.config`; provenance key `scqc_pipeline_version`→`scpilot_version`; command `-m scqc_pipeline`→`-m scpilot`; snapshot dir `scqc_pipeline-*`→`scpilot-*`; `uns["scqc_pipeline"]`→`uns["scpilot"]`; repro-template stages-import neutralized |
| `config.py` | scqc `config.py` | none (no scqc imports) — scqc-specific profile fields kept as starting point; will diverge |
| `io_10x.py` | scqc `io_10x.py` | none (self-contained) |
| `plotting.py` | scqc `plotting.py` | config import → `scpilot.vendor.config`; **+`square_limit_col`** option (plotting_cfg setdefault + `_size_grid` filter) so scpilot can cap to orientation-flexible {1×1.5,1.5×1,1×1} per the user plot policy (B5) |

**Not vendored (yet):** `metaschema.py` — upstream raw-metadata harmonize/filter/derive
domain; scpilot enters at the merged h5ad, so vendor only if scpilot ever handles
raw per-sample metadata.

## Known adaptation TODOs (resolve in plan A7 when scpilot's own runtime lands)
- `harness.run_stage` / `Pipeline` are tied to scqc's linear stage model +
  `PipelineConfig`. scpilot needs a recursive/job-aware registry (plan C1); these
  are kept as a *reference foundation* — build scpilot's runtime on the stateless
  primitives (`atomic_path`, `source_hash`, `build_provenance`, `StageReport`,
  `init_runtime`, `_fingerprint`), not on `Pipeline`.
- `snapshot_source` snapshots `_package_dir()` = `scpilot/vendor`, so it currently
  captures only vendored code. When scpilot's runtime is built, point the source
  snapshot at the full scpilot package.

## Re-sync procedure
1. Recompute upstream hash: `python -c "from scqc_pipeline.harness import source_hash; print(source_hash())"`.
2. If it differs from `debef308904633e1`, diff upstream vs vendored, port wanted
   changes manually, update the hash + date above.
