# Cancer Tissue scRNA-seq Annotation Strategy

## Overview

Cancer tissue scRNA-seq annotation should be handled as a hierarchical, evidence-based process rather than a single-pass cluster naming task. The key distinction from normal tissue or PBMC annotation is that malignant cells, tumor microenvironment cells, artifacts, cell states, and trajectory-like continua can overlap.

Recommended principle:

```text
cell type annotation
+ malignancy evidence
+ cell state evidence
+ trajectory evidence
+ uncertainty tracking
= final annotation proposal
```

LLM reasoning agents can be useful at each tier, but they should act as evidence integration and audit layers, not as the sole annotation authority.

## Core Annotation Workflow

```text
QC / doublet / ambient RNA assessment
-> broad cell type annotation
-> malignant vs non-malignant classification
-> compartment-specific subclustering
-> fine annotation using markers and references
-> trajectory / state interpretation
-> consistency review and uncertainty flagging
```

## Tiered LLM Reasoning Agent Design

### Tier 0: QC and Artifact Agent

Purpose:

- Flag low-quality clusters.
- Detect doublet-like populations.
- Identify ambient RNA contamination.
- Detect dissociation stress or stress-dominated clusters.

Inputs:

- QC metrics: `n_genes`, `total_counts`, `%MT`, `%ribo`.
- Doublet score.
- Ambient RNA scores.
- Stress gene signatures.
- Sample and patient composition.
- Mixed lineage marker expression.

Example output:

```json
{
  "cluster_id": "7",
  "qc_status": "suspect_doublet",
  "evidence_for": [
    "co-expression of EPCAM and CD3D",
    "high n_genes",
    "high doublet score"
  ],
  "confidence": 0.82,
  "recommended_action": "exclude_or_review"
}
```

### Tier 1: Broad Cell Type Agent

Purpose:

- Assign broad tissue compartments.
- Detect marker conflicts.
- Separate immune, stromal, endothelial, epithelial, and artifact populations.

Example broad compartments:

- Epithelial
- T/NK
- B/Plasma
- Myeloid
- Stromal
- Endothelial
- Mast
- Mixed/Artifact

Representative markers:

| Compartment | Example markers |
| --- | --- |
| Epithelial | `EPCAM`, `KRT8`, `KRT18`, `KRT19` |
| T cell | `CD3D`, `CD3E`, `TRAC` |
| NK cell | `NKG7`, `GNLY`, `KLRD1` |
| B cell | `MS4A1`, `CD79A` |
| Plasma cell | `MZB1`, `JCHAIN`, `XBP1` |
| Myeloid | `LYZ`, `LST1`, `S100A8`, `FCGR3A` |
| Endothelial | `PECAM1`, `VWF`, `KDR` |
| Fibroblast / CAF | `COL1A1`, `COL1A2`, `DCN`, `LUM`, `ACTA2`, `FAP` |
| Pericyte | `RGS5`, `PDGFRB`, `MCAM` |

### Tier 2: Malignancy Agent

Purpose:

- Distinguish malignant epithelial cells from non-malignant epithelial cells.
- Avoid relying only on epithelial markers.
- Integrate CNV, sample specificity, tumor markers, and reference similarity.

Recommended evidence:

- CNV inference: `inferCNV`, `copyKAT`, `HoneyBADGER`, or similar.
- Tumor marker expression.
- Normal epithelial reference similarity.
- Patient-specific clonal expansion.
- CNV burden.
- Doublet and artifact flags.

Example output:

```json
{
  "cluster_id": "Epi_3",
  "malignancy": "malignant",
  "confidence": 0.91,
  "evidence_for": [
    "high CNV burden",
    "patient-specific expansion",
    "EPCAM/KRT19 positive"
  ],
  "evidence_against": [
    "moderate stress signature"
  ],
  "warnings": [
    "trajectory may reflect CNV clone structure"
  ]
}
```

### Tier 3: Compartment-Specific Fine Annotation Agent

Purpose:

- Refine annotation within each major compartment.
- Prevent unrelated cell types from being forced into the same trajectory or label space.

Recommended compartment-specific analyses:

| Compartment | Fine annotation examples |
| --- | --- |
| T/NK | CD4 T, CD8 T, Treg, exhausted T, cycling T, NK |
| Myeloid | Monocyte, macrophage, TAM, cDC1, cDC2, pDC, neutrophil |
| Stromal | Fibroblast, myCAF, iCAF, apCAF, pericyte |
| Epithelial / malignant | Tumor cells, cycling tumor cells, EMT-like tumor cells, hypoxic tumor cells, IFN-high tumor cells |
| B/Plasma | B cell, memory B cell, plasma cell |

### Tier 4: Trajectory and Cell State Agent

Purpose:

- Interpret pseudotime or developmental-like continua.
- Distinguish biological differentiation from confounders.
- Use trajectory output as cell-state evidence, not as direct cell type proof.

Recommended tools:

- `PAGA`
- `Slingshot`
- `Monocle3`
- `Palantir`
- `scVelo`
- `CellRank`
- `CytoTRACE`

Important caveats:

- Pseudotime is not actual time.
- Tumor trajectories can reflect CNV clones, cell cycle, hypoxia, stress, or batch effects.
- Patient-specific malignant clones can dominate the inferred axis.
- Trajectory should usually be performed within a compartment, not across all cells.

Example output:

```json
{
  "trajectory": "Malignant lineage 1",
  "interpretation": "EMT-like progression",
  "confidence": 0.67,
  "supporting_gradients": [
    "VIM",
    "ZEB1",
    "FN1",
    "COL1A1"
  ],
  "confounders": [
    "patient-specific CNV",
    "hypoxia score"
  ],
  "use_as": "cell_state_not_lineage"
}
```

### Tier 5: Consistency and Review Agent

Purpose:

- Audit the final annotation table.
- Detect inconsistent labels.
- Flag low-confidence or artifact-prone clusters.

Checks:

- Same marker profile but different labels.
- Same label but inconsistent marker evidence.
- Contradictory hierarchy, such as `major_cell_type = T/NK` and `fine_cell_type = macrophage`.
- Single-patient cluster dominance.
- Batch-specific clusters.
- High doublet or stress score.
- Malignancy label without CNV or tumor evidence.

## Immune Cell Annotation Strategy

Immune cell annotation can be used either as part of cancer tissue TME analysis or as a standalone immune-focused scRNA-seq workflow. The same principle applies: separate lineage identity, subtype identity, activation state, and trajectory or differentiation state.

Recommended immune workflow:

```text
CD45+ immune compartment selection
-> broad immune lineage annotation
-> lineage-specific subclustering
-> subtype annotation
-> activation / exhaustion / proliferation / cytokine state scoring
-> trajectory or differentiation-state interpretation
-> consistency and marker conflict review
```

### Immune-Specific Annotation Principles

- Do not mix lineage and state into a single irreversible label.
- Annotate T/NK, B/plasma, myeloid, DC, mast, and granulocyte compartments separately.
- Treat `MKI67+`, interferon-high, heat-shock-high, hypoxia-high, and stress-high patterns as cell states unless lineage markers also support a distinct subtype.
- Use both positive and negative markers. For example, NK-like cytotoxic T cells can express `NKG7` and `GNLY`, but should still be separated from NK cells using `CD3D`, `CD3E`, and `TRAC`.
- In tumor tissue, activation and exhaustion gradients often dominate clustering. These should be stored in `cell_state`, not automatically promoted to `fine_cell_type`.
- For CITE-seq or FACS-validated datasets, prefer protein markers for final FACS-style labels and RNA markers for supporting evidence.

### Broad Immune Lineage Markers

| Lineage | Example markers | Notes |
| --- | --- | --- |
| Pan-immune | `PTPRC` | RNA expression can be lower than CD45 protein signal |
| T cell | `CD3D`, `CD3E`, `TRAC`, `TRBC1`, `TRBC2` | Separate from NK using TCR genes |
| CD4 T | `CD4`, `IL7R`, `CCR7`, `LTB` | `CD4` RNA can be sparse |
| CD8 T | `CD8A`, `CD8B`, `GZMK`, `GZMB` | Separate cytotoxic state from lineage |
| Treg | `FOXP3`, `IL2RA`, `CTLA4`, `IKZF2` | Use multiple markers, not `FOXP3` alone |
| NK | `NKG7`, `GNLY`, `KLRD1`, `KLRF1`, `FCGR3A` | Confirm absence of TCR markers |
| B cell | `MS4A1`, `CD79A`, `CD79B`, `BANK1` | Separate from plasma cells |
| Plasma cell | `MZB1`, `JCHAIN`, `XBP1`, `SDC1` | Often high immunoglobulin genes |
| Monocyte | `LYZ`, `LST1`, `S100A8`, `S100A9`, `FCN1` | Classical monocytes are often `FCN1/S100A8` high |
| Macrophage | `C1QA`, `C1QB`, `C1QC`, `APOE`, `CD68` | TAM states require context |
| cDC1 | `CLEC9A`, `XCR1`, `BATF3`, `IRF8` | Usually rare |
| cDC2 | `CD1C`, `FCER1A`, `CLEC10A` | Can overlap with monocytes |
| pDC | `LILRA4`, `GZMB`, `IRF7`, `TCF4` | Interferon-high state can confuse annotation |
| Mast cell | `TPSAB1`, `TPSB2`, `CPA3`, `KIT` | May show degranulation state |
| Neutrophil | `S100A8`, `S100A9`, `FCGR3B`, `CSF3R` | Often fragile in scRNA-seq |

### T and NK Cell Fine Annotation

| FACS-style label | Internal annotation | Supporting markers |
| --- | --- | --- |
| `CD3+ T cells` | T cell | `CD3D`, `CD3E`, `TRAC` |
| `CD4+ T cells` | CD4 T cell | `CD4`, `IL7R`, `CCR7`, `LTB` |
| `CD8+ T cells` | CD8 T cell | `CD8A`, `CD8B` |
| `CCR7+ naive T cells` | Naive T cell | `CCR7`, `LEF1`, `TCF7`, `SELL` |
| `GZMK+ memory T cells` | Memory-like T cell | `GZMK`, `IL7R`, `LTB` |
| `GZMB+ cytotoxic T cells` | Cytotoxic T cell | `GZMB`, `PRF1`, `NKG7`, `GNLY` |
| `PD-1+ exhausted CD8+ T cells` | Exhausted CD8 T | `PDCD1`, `TOX`, `LAG3`, `HAVCR2`, `TIGIT` |
| `FOXP3+ Tregs` | Regulatory T cell | `FOXP3`, `IL2RA`, `CTLA4`, `IKZF2` |
| `MKI67+ cycling T cells` | Cycling T cell | `MKI67`, `TOP2A`, `STMN1` |
| `CD3- NKG7+ NK cells` | NK cell | `NKG7`, `GNLY`, `KLRD1`, low `CD3D/TRAC` |

Recommended T/NK state fields:

| State | Example markers |
| --- | --- |
| naive / central memory | `CCR7`, `SELL`, `TCF7`, `LEF1` |
| effector memory | `GZMK`, `IL7R`, `CXCR3` |
| cytotoxic | `GZMB`, `PRF1`, `NKG7`, `GNLY` |
| exhausted | `PDCD1`, `TOX`, `LAG3`, `HAVCR2`, `TIGIT` |
| tissue-resident | `ITGAE`, `CXCR6`, `ZNF683` |
| proliferating | `MKI67`, `TOP2A`, `UBE2C` |
| interferon-high | `ISG15`, `IFIT1`, `IFIT3`, `MX1` |

### B and Plasma Cell Fine Annotation

| FACS-style label | Internal annotation | Supporting markers |
| --- | --- | --- |
| `CD19+ B cells` | B cell | `MS4A1`, `CD79A`, `CD79B` |
| `MS4A1+ naive B cells` | Naive B cell | `MS4A1`, `TCL1A`, `IGHD`, `IGHM` |
| `CD27+ memory B cells` | Memory B cell | `CD27`, `TNFRSF13B`, class-switched immunoglobulins |
| `CD38hi plasma cells` | Plasma cell | `MZB1`, `JCHAIN`, `XBP1`, `SDC1` |
| `MKI67+ cycling B cells` | Cycling B cell | `MKI67`, `TOP2A` |

Notes:

- Immunoglobulin genes can dominate marker tables. Use B lineage genes and plasma cell programs together.
- Plasma cells often cluster far from B cells because of antibody production and ER stress programs.

### Myeloid and DC Fine Annotation

| FACS-style label | Internal annotation | Supporting markers |
| --- | --- | --- |
| `CD14+ monocytes` | Classical monocyte | `FCN1`, `S100A8`, `S100A9`, `VCAN` |
| `CD16+ monocytes` | Non-classical monocyte | `FCGR3A`, `MS4A7`, `LST1` |
| `CD68+ macrophages` | Macrophage | `C1QA`, `C1QB`, `C1QC`, `APOE`, `CD68` |
| `CD163+ TAMs` | TAM, immunosuppressive-like | `CD163`, `MRC1`, `MSR1`, `MARCO` |
| `IL1B+ inflammatory myeloid cells` | Inflammatory myeloid cell | `IL1B`, `CXCL8`, `S100A8`, `S100A9` |
| `CLEC9A+ cDC1` | cDC1 | `CLEC9A`, `XCR1`, `BATF3` |
| `CD1C+ cDC2` | cDC2 | `CD1C`, `FCER1A`, `CLEC10A` |
| `LILRA4+ pDC` | pDC | `LILRA4`, `TCF4`, `GZMB`, `IRF7` |
| `S100A8+ neutrophils` | Neutrophil-like cell | `S100A8`, `S100A9`, `FCGR3B`, `CSF3R` |

Recommended myeloid state fields:

| State | Example markers |
| --- | --- |
| inflammatory | `IL1B`, `CXCL8`, `S100A8`, `S100A9` |
| antigen-presenting | `HLA-DRA`, `HLA-DPB1`, `CD74` |
| complement-high | `C1QA`, `C1QB`, `C1QC` |
| lipid-associated | `APOE`, `TREM2`, `LPL` |
| immunosuppressive / scavenger | `CD163`, `MRC1`, `MSR1`, `MARCO` |
| interferon-high | `ISG15`, `IFIT1`, `MX1` |
| cycling | `MKI67`, `TOP2A` |

### Immune Trajectory Interpretation

Trajectory can be useful for immune annotation, but the interpretation should be lineage-specific.

Recommended trajectory uses:

| Compartment | Reasonable trajectory interpretation |
| --- | --- |
| T cells | naive/memory to effector/exhausted gradient |
| CD8 T cells | cytotoxic activation to exhaustion-like state |
| Tregs | resting to activated/suppressive Treg state |
| Monocyte/macrophage | monocyte to macrophage/TAM-like differentiation |
| DCs | precursor-like to antigen-presenting or interferon-high states |
| B cells | naive/memory to plasmablast/plasma transition |

Confounders to overlay:

- `patient`
- `sample`
- `batch`
- `cell_cycle_score`
- `stress_score`
- `interferon_score`
- `activation_score`
- `doublet_score`

Recommended trajectory output:

```json
{
  "compartment": "CD8 T cells",
  "trajectory_state": "memory_to_exhausted_axis",
  "confidence": 0.74,
  "supporting_gradients": [
    "TCF7 down",
    "GZMB up",
    "PDCD1/TOX up"
  ],
  "confounders": [
    "activation and exhaustion markers overlap",
    "patient-specific expansion of one clone-like cluster"
  ],
  "use_as": "cell_state_axis"
}
```

### Immune-Specific Agent Output Example

```json
{
  "cluster_id": "T_5",
  "major_cell_type": "T/NK",
  "fine_cell_type": "Exhausted CD8 T",
  "facs_style_label": "CD8+ PD-1+ T cells",
  "malignancy": "non_malignant",
  "cell_state": "exhausted",
  "trajectory_state": "memory_to_exhausted_axis",
  "confidence": 0.88,
  "review_required": false,
  "evidence_for": [
    "CD3D/TRAC positive",
    "CD8A/CD8B positive",
    "PDCD1/TOX/LAG3 elevated",
    "located at terminal end of CD8 T pseudotime"
  ],
  "evidence_against": [
    "moderate cytotoxic marker expression"
  ],
  "confounders": [
    "activation markers can overlap with exhaustion markers"
  ]
}
```

## Recommended Metadata Schema

Keep cell type, malignancy, state, and trajectory fields separate.

| Column | Meaning |
| --- | --- |
| `cluster_id` | Cluster identifier |
| `major_cell_type` | Broad compartment |
| `fine_cell_type` | Biological cell type or subtype |
| `facs_style_label` | FACS-like display label |
| `malignancy` | `malignant`, `non_malignant`, `uncertain`, or `not_applicable` |
| `cell_state` | Functional state such as cycling, exhausted, EMT-like, hypoxia-high |
| `trajectory_state` | Pseudotime or branch interpretation |
| `confidence` | Numeric confidence score |
| `review_required` | Whether manual review is required |
| `evidence_for` | Supporting evidence |
| `evidence_against` | Conflicting evidence |
| `confounders` | Potential confounders |

## FACS-Style Annotation Labels

FACS-style labels are useful for figures and experimental interpretation because they resemble marker-gated populations. They should be paired with internal biological labels.

### Example Mapping

| FACS-style population | scRNA-seq interpretation |
| --- | --- |
| `CD45- EpCAM+ tumor cells` | Malignant epithelial cells |
| `CD45- EpCAM+ Ki67+ tumor cells` | Proliferating malignant epithelial cells |
| `CD45- EpCAM+ VIM+ tumor cells` | EMT-like malignant epithelial cells |
| `CD45- EpCAMlow VIM+ tumor cells` | Mesenchymal-like tumor cells |
| `CD45- EpCAM- COL1A1+ CAFs` | Cancer-associated fibroblasts |
| `CD45- EpCAM- ACTA2+ CAFs` | myCAFs |
| `CD45- EpCAM- IL6+ CAFs` | iCAFs |
| `CD45- CD31+ endothelial cells` | Vascular endothelial cells |
| `CD45- PDGFRB+ pericytes` | Pericytes |
| `CD45+ immune cells` | Total immune cells |
| `CD45+ CD3+ T cells` | Total T cells |
| `CD45+ CD3+ CD4+ T cells` | CD4 T cells |
| `CD45+ CD3+ CD8+ T cells` | CD8 T cells |
| `CD45+ CD3+ FOXP3+ Tregs` | Regulatory T cells |
| `CD45+ CD3+ PDCD1+ exhausted T cells` | Exhausted T cells |
| `CD45+ CD3+ MKI67+ cycling T cells` | Proliferating T cells |
| `CD45+ CD3- NCAM1+ NK cells` | NK cells |
| `CD45+ CD19+ B cells` | B cells |
| `CD45+ CD19lo CD38hi plasma cells` | Plasma cells |
| `CD45+ CD11B+ myeloid cells` | Total myeloid cells |
| `CD45+ CD11B+ CD14+ monocytes` | Classical monocytes |
| `CD45+ CD11B+ FCGR3A+ monocytes` | Non-classical monocytes |
| `CD45+ CD11B+ CD68+ macrophages` | Macrophages |
| `CD45+ CD11B+ CD68+ CD163+ TAMs` | Immunosuppressive TAM-like macrophages |
| `CD45+ CD11B+ S100A8+ neutrophils` | Neutrophil-like or granulocytic cells |
| `CD45+ HLA-DRA+ CD11C+ DCs` | Dendritic cells |
| `CD45+ CLEC9A+ cDC1` | cDC1 |
| `CD45+ CD1C+ cDC2` | cDC2 |
| `CD45+ LILRA4+ pDC` | Plasmacytoid DCs |

### Display Label Examples

For paper figures or summary plots:

```text
EpCAM+ tumor cells
EpCAM+ Ki67+ tumor cells
EpCAM+ EMT-like tumor cells
CD3+ T cells
CD4+ T cells
CD8+ T cells
PD-1+ exhausted CD8+ T cells
FOXP3+ Tregs
NKG7+ NK cells
CD19+ B cells
CD38hi plasma cells
CD11B+ myeloid cells
CD14+ monocytes
CD68+ macrophages
CD163+ TAMs
S100A8+ neutrophils
CD11C+ dendritic cells
CLEC9A+ cDC1
CD1C+ cDC2
LILRA4+ pDC
ACTA2+ myCAFs
IL6+ iCAFs
CD31+ endothelial cells
PDGFRB+ pericytes
```

## Final Annotation Table Example

| cluster_id | facs_style_label | major_cell_type | fine_cell_type | malignancy | cell_state | confidence | review_required |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `EpCAM+ tumor cells` | Epithelial | Malignant epithelial | malignant | tumor | 0.94 | false |
| 1 | `EpCAM+ Ki67+ tumor cells` | Epithelial | Malignant epithelial | malignant | cycling | 0.90 | false |
| 2 | `EpCAM+ EMT-like tumor cells` | Epithelial | Malignant epithelial | malignant | EMT-like | 0.69 | true |
| 3 | `CD8+ PD-1+ T cells` | T/NK | Exhausted CD8 T | non_malignant | exhausted | 0.88 | false |
| 4 | `FOXP3+ Tregs` | T/NK | Regulatory T cell | non_malignant | suppressive | 0.86 | false |
| 5 | `CD68+ CD163+ TAMs` | Myeloid | TAM | non_malignant | M2-like | 0.76 | true |
| 6 | `ACTA2+ myCAFs` | Stromal | myCAF | non_malignant | contractile | 0.83 | false |
| 7 | `IL6+ iCAFs` | Stromal | iCAF | non_malignant | inflammatory | 0.81 | false |
| 8 | `CD31+ endothelial cells` | Endothelial | Vascular endothelial | non_malignant | angiogenic | 0.86 | false |
| 9 | `Epithelial-T mixed cells` | Mixed/Artifact | Doublet-like | uncertain | epithelial-T mix | 0.72 | true |

## Example Structured Agent Output

```json
{
  "cluster_id": "2",
  "major_cell_type": "Epithelial",
  "fine_cell_type": "Malignant epithelial",
  "facs_style_label": "EpCAM+ EMT-like tumor cells",
  "malignancy": "malignant",
  "cell_state": "EMT-like",
  "trajectory_state": "tumor_EMT_branch",
  "confidence": 0.69,
  "review_required": true,
  "evidence_for": [
    "EPCAM/KRT19 positive",
    "high CNV burden",
    "VIM/FN1/ZEB1 elevated",
    "located on EMT-like pseudotime branch"
  ],
  "evidence_against": [
    "partial fibroblast marker expression"
  ],
  "confounders": [
    "possible epithelial-fibroblast doublet",
    "patient-specific CNV structure"
  ]
}
```

## Practical Recommendation

Use FACS-style names for figure labels and communication with experimental collaborators, but keep structured metadata columns for computation and reproducibility.

Recommended final split:

```text
facs_style_label   -> human-readable display label
major_cell_type    -> broad compartment
fine_cell_type     -> biological subtype
malignancy         -> malignant / non_malignant / uncertain
cell_state         -> functional state
trajectory_state   -> pseudotime or branch interpretation
confidence         -> confidence score
review_required    -> manual review flag
evidence fields    -> reasoning trace
```

This keeps the annotation interpretable for FACS-style biological reasoning while preserving the detail needed for scRNA-seq downstream analysis.
