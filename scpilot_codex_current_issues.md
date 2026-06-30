# scpilot Current Issues

Date: 2026-06-24

## Current State

- Main session: `scpilot_obesity_run`
- Main session completed through: `scpilot_obesity_run/checkpoints/04_cluster.h5ad`
- Safe resume checkpoint: `scpilot_obesity_markers_from_cluster/checkpoints/00_load.h5ad`
- Dataset at clustered stage: `331,127 cells x 33,696 genes`
- Leiden clusters: 13

## Problems Found

1. The main `scpilot_obesity_run` session completed clustering, but `markers` failed afterward.
   - Failure: `invalid_state`
   - Likely cause: the MCP tool appeared to read the original session input, `obesity_merged_counts.h5ad`, instead of the requested clustered checkpoint.
   - The original input does not contain `leiden`, so marker calculation was rejected as if clustering had not been run.

2. The built-in scpilot `markers` step is too heavy for this dataset.
   - It is fixed to Wilcoxon ranking.
   - It ranks all genes: `adata.n_vars`, currently 33,696 genes.
   - On 331k cells, this exceeded the MCP call timeout and produced no marker artifact.

3. A new session, `scpilot_obesity_markers_from_cluster`, was created as a workaround.
   - Input was the clustered checkpoint: `scpilot_obesity_run/checkpoints/04_cluster.h5ad`
   - Its `00_load.h5ad` contains `leiden`, `X_pca`, `X_umap`, and `scale.data`.
   - Marker calculation was started from this session but was stopped by user request.
   - No marker output or partial checkpoint remained after stopping.

4. Local Python environment issues were observed.
   - The default Python environment does not have `anndata`.
   - The scpilot conda environment can run Scanpy, but Scanpy import initially failed due to numba cache permissions.
   - Workaround: set `NUMBA_CACHE_DIR=/tmp/numba-cache`.

5. Disk usage is already substantial.
   - Existing scpilot checkpoints total more than 40 GB.
   - `04_cluster.h5ad` and the workaround `00_load.h5ad` are each about 11 GB.
   - Writing another full marker checkpoint may add another large file.

## Recommended Next Step

Do not rerun the built-in Wilcoxon `markers` tool on the full dataset as-is.

Recommended alternatives:

- Run a lighter marker calculation using `t-test_overestim_var` with limited top-N genes per cluster.
- Or compute marker evidence on a controlled subsample per cluster.
- Then generate annotation evidence from that marker table before applying Tier-1 labels.

## Paths

- Main clustered checkpoint:
  `scpilot_obesity_run/checkpoints/04_cluster.h5ad`
- Workaround resume checkpoint:
  `scpilot_obesity_markers_from_cluster/checkpoints/00_load.h5ad`
- Main run log:
  `scpilot_obesity_run/run_log.jsonl`
- Workaround run log:
  `scpilot_obesity_markers_from_cluster/run_log.jsonl`
