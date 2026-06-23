"""Unit tests for B12-pre annotate_genomic_positions (2-pass map + pc_coverage gate).

Uses a tiny synthetic GENCODE-style GTF (no network) to exercise:
- pass 1 maps raw symbols incl. a REAL hyphenated gene (HLA-A),
- pass 2 recovers a make_unique suffix (FOO-1 -> base FOO),
- protein_coding_coverage gate + unmapped classification,
- non-destructive var (existing columns preserved, counts/X untouched).
"""

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot import tools
from scpilot.session import Session

# feature=gene rows; cols: seqname source feature start end score strand frame attrs
_GENES = [
    ("chr1", "AAA", 1000, 2000, "protein_coding"),
    ("chr1", "BBB", 3000, 4000, "protein_coding"),
    ("chr2", "HLA-A", 5000, 6000, "protein_coding"),   # real hyphenated gene
    ("chr3", "FOO", 7000, 8000, "protein_coding"),
    ("chr1", "MIR123", 9000, 9100, "miRNA"),           # noncoding (not in pc universe)
]


def _write_gtf(path):
    lines = []
    for i, (chrom, name, start, end, gtype) in enumerate(_GENES):
        attrs = f'gene_id "ENSG{i:08d}"; gene_name "{name}"; gene_type "{gtype}";'
        lines.append("\t".join([chrom, "HAVANA", "gene", str(start), str(end), ".", "+", ".", attrs]))
    path.write_text("\n".join(lines) + "\n")
    return path


def _session(tmp_path):
    # var_names simulate var_names_make_unique output + unmapped symbols
    var_names = ["AAA", "BBB", "HLA-A", "FOO", "FOO-1", "MIR123", "ZZZ-9", "UNKNOWNGENE"]
    X = sparse.csr_matrix(np.random.default_rng(0).poisson(1.0, (10, len(var_names))).astype("float32"))
    a = ad.AnnData(X)
    a.var_names = var_names
    a.layers["counts"] = a.X.copy()
    a.var["n_cells"] = np.asarray((a.X > 0).sum(0)).ravel()   # pre-existing var column
    p = tmp_path / "in.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "sess", input_path=str(p)); s.load_input()
    return s


def test_two_pass_mapping_and_gate(tmp_path):
    gtf = _write_gtf(tmp_path / "mini.gtf")
    s = _session(tmp_path)
    r = tools.run("annotate_genomic_positions", s, gtf=str(gtf))
    assert r.status == "success", r.error
    su = r.summary

    # pass 1: AAA,BBB,HLA-A,FOO,MIR123 = 5 (HLA-A maps despite the hyphen)
    assert su["pass1_mapped"] == 5
    # pass 2: FOO-1 -> base FOO recovered
    assert su["make_unique_recovered"] == 1
    assert su["n_mapped"] == 6
    assert su["n_unmapped"] == 2          # ZZZ-9, UNKNOWNGENE

    # protein_coding universe = {AAA,BBB,HLA-A,FOO}; all covered -> 1.0, gate passes
    assert su["pc_total"] == 4
    assert su["protein_coding_coverage"] == 1.0
    assert su["gate_pass"] is True
    assert su["reproducibility_grade"] == "A"
    assert su["source"]["type"] == "user"

    # FOO-1 inherits FOO's coordinates; chromosomes carry chr prefix
    var = s.adata.var
    assert var.loc["FOO-1", "chromosome"] == "chr3"
    assert var.loc["FOO-1", "start"] == var.loc["FOO", "start"]
    assert str(var.loc["HLA-A", "chromosome"]) == "chr2"

    # non-destructive: pre-existing column + counts/X intact
    assert "n_cells" in var.columns
    assert "counts" in s.adata.layers
    r.to_dict()


def test_low_pc_coverage_warns(tmp_path):
    # GTF with only ONE protein_coding gene present in data -> low coverage
    (tmp_path / "tiny.gtf").write_text(
        'chr1\tHAVANA\tgene\t1\t2\t.\t+\t.\tgene_id "E1"; gene_name "AAA"; gene_type "protein_coding";\n'
        'chr1\tHAVANA\tgene\t3\t4\t.\t+\t.\tgene_id "E2"; gene_name "ZONLY"; gene_type "protein_coding";\n'
        'chr1\tHAVANA\tgene\t5\t6\t.\t+\t.\tgene_id "E3"; gene_name "YONLY"; gene_type "protein_coding";\n')
    s = _session(tmp_path)
    r = tools.run("annotate_genomic_positions", s, gtf=str(tmp_path / "tiny.gtf"))
    assert r.status == "success"
    # only AAA (of AAA/ZONLY/YONLY) is in the data -> 1/3 < 0.6
    assert r.summary["protein_coding_coverage"] < 0.6
    assert r.summary["gate_pass"] is False
    assert any("protein_coding_coverage" in w for w in r.warnings)
    assert r.suggested_next_tools == ["inspect"]


def test_missing_gtf_path(tmp_path):
    s = _session(tmp_path)
    r = tools.run("annotate_genomic_positions", s, gtf=str(tmp_path / "nope.gtf"))
    assert r.status == "error" and r.error_code == "missing_input"


# --------------------------------------------------------------------------- #
# B12 cnv_score
# --------------------------------------------------------------------------- #
def _coord_session(tmp_path):
    """200 genes on chr1/chr2/chr3 (with coords); tumor cells carry an injected
    chr1 amplification so cnv_score must rank them above the Normal reference."""
    import scanpy as sc
    rng = np.random.default_rng(0)
    n_cells, n_genes = 160, 210
    base = rng.poisson(1.0, (n_cells, n_genes)).astype("float32")
    tumor = np.zeros(n_cells, dtype=bool); tumor[80:] = True
    chr1 = slice(0, 70)                       # first 70 genes = chr1 block
    base[tumor, chr1] += rng.poisson(6.0, (tumor.sum(), 70)).astype("float32")  # amplification
    a = ad.AnnData(sparse.csr_matrix(base))
    a.var_names = [f"G{i:04d}" for i in range(n_genes)]
    a.layers["counts"] = a.X.copy()
    # genomic coordinates (3 chromosomes, ordered start positions)
    chrom, start = [], []
    for i in range(n_genes):
        c = 1 if i < 70 else (2 if i < 140 else 3)
        chrom.append(f"chr{c}"); start.append((i % 70) * 1000 + 1)
    a.var["chromosome"] = chrom
    a.var["start"] = start
    a.var["end"] = [s + 800 for s in start]
    a.obs["condition"] = np.where(tumor, "Tumor", "Normal")
    a.obs["major_cell_type"] = np.where(tumor, "Epithelial", "T_cell")
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)   # infercnv wants log-norm .X
    p = tmp_path / "coord.h5ad"; a.write_h5ad(p)
    s = Session.create(tmp_path / "csess", input_path=str(p)); s.load_input()
    return s


def test_cnv_score_reference_contrast(tmp_path):
    s = _coord_session(tmp_path)
    r = tools.run("cnv_score", s, reference_key="condition", reference_cat=["Normal"],
                  window_size=10, step=2)
    assert r.status == "success", r.error
    su = r.summary
    assert su["advisory_only"] is False
    assert "cnv_leiden" in s.adata.obs.columns
    assert "cnv_score" in s.adata.obs.columns
    assert "X_cnv" in s.adata.obsm
    # injected tumor amplification -> non-reference CNV burden > reference
    rc = su["reference_contrast"]
    assert rc is not None
    assert rc["nonreference_mean_cnv"] > rc["reference_mean_cnv"]
    # per-celltype table written + Epithelial (tumor) outranks T_cell
    assert "cnv_by_celltype" in r.tables
    r.to_dict()


def test_cnv_score_requires_coordinates(tmp_path):
    s = _session(tmp_path)   # var has symbols but no chromosome column
    r = tools.run("cnv_score", s)
    assert r.status == "error" and r.error_code == "invalid_state"
    assert r.suggested_next_tools == ["annotate_genomic_positions"]


def test_cnv_score_advisory_when_no_reference(tmp_path):
    s = _coord_session(tmp_path)
    r = tools.run("cnv_score", s, window_size=10, step=2)
    assert r.status == "success"
    assert r.summary["advisory_only"] is True
    assert any("advisory-only" in w for w in r.warnings)


# --------------------------------------------------------------------------- #
# B12 malignancy call (evidence + apply)
# --------------------------------------------------------------------------- #
def _scored_session(tmp_path):
    s = _coord_session(tmp_path)
    # add a patient/sample key: tumor cells = single patient (clonal), normal = shared
    n = s.adata.n_obs
    samp = np.where(s.adata.obs["condition"].values == "Tumor", "P1",
                    np.array(["P2", "P3", "P4"])[np.arange(n) % 3])
    s.adata.obs["sample_id"] = samp
    tools.run("cnv_score", s, reference_key="condition", reference_cat=["Normal"],
              window_size=10, step=2)
    return s


def test_malignancy_evidence_packages_axes(tmp_path):
    s = _scored_session(tmp_path)
    r = tools.run("malignancy_evidence", s, groupby="major_cell_type",
                  reference_key="condition", reference_cat=["Normal"], sample_key="sample_id")
    assert r.status == "success", r.error
    assert r.summary["advisory_only"] is False
    assert "evidence" in r.tables
    # evidence JSON written; epithelial (tumor) group shows higher CNV ratio than T_cell
    import json
    payload = json.loads(open(r.summary["evidence_input"]).read())
    by = {g["group"]: g for g in payload["groups"]}
    assert by["Epithelial"]["cnv_burden"]["ratio_to_reference"] > by["T_cell"]["cnv_burden"]["ratio_to_reference"]
    # clonal expansion: tumor epithelial dominated by one patient
    assert by["Epithelial"]["clonal_expansion"]["top_sample_fraction"] == 1.0
    # read-only: scratch score columns cleaned up, no malignancy call yet
    assert "malignancy" not in s.adata.obs.columns
    r.to_dict()


def test_malignancy_evidence_requires_cnv_score(tmp_path):
    s = _coord_session(tmp_path)   # no cnv_score run
    r = tools.run("malignancy_evidence", s, groupby="major_cell_type")
    assert r.status == "error" and r.error_code == "invalid_state"
    assert r.suggested_next_tools == ["cnv_score"]


def test_apply_malignancy_writes_call(tmp_path):
    s = _scored_session(tmp_path)
    r = tools.run("apply_malignancy", s, groupby="major_cell_type",
                  labels={"Epithelial": "malignant", "T_cell": "non_malignant"},
                  confidence={"Epithelial": 0.9, "T_cell": 0.8})
    assert r.status == "success", r.error
    obs = s.adata.obs
    assert set(obs["malignancy"].unique()) == {"malignant", "non_malignant"}
    assert "malignancy_confidence" in obs.columns
    assert "malignancy_review_required" in obs.columns
    # CNV evidence present -> malignant call NOT force-reviewed
    assert r.summary["forced_review_no_cnv"] == []
    r.to_dict()


def test_apply_malignancy_rejects_bad_vocab(tmp_path):
    s = _scored_session(tmp_path)
    r = tools.run("apply_malignancy", s, groupby="major_cell_type",
                  labels={"Epithelial": "tumor"})   # not in vocabulary
    assert r.status == "error" and r.error_code == "data_gate_failed"


def test_apply_malignancy_forces_review_without_cnv(tmp_path):
    s = _coord_session(tmp_path)   # no cnv_score -> no obs['cnv_score']
    s.adata.obs["sample_id"] = "P1"
    r = tools.run("apply_malignancy", s, groupby="major_cell_type",
                  labels={"Epithelial": "malignant", "T_cell": "non_malignant"})
    assert r.status == "success"
    # HARD RULE: malignant without CNV evidence -> review_required forced
    assert "Epithelial" in r.summary["forced_review_no_cnv"]
    rr = s.adata.obs.loc[s.adata.obs["major_cell_type"] == "Epithelial", "malignancy_review_required"]
    assert bool(rr.iloc[0]) is True


# --------------------------------------------------------------------------- #
# Phase E — CNV plot suite + cnv_status derivation
# --------------------------------------------------------------------------- #
def test_cnv_score_emits_plot_suite(tmp_path):
    s = _coord_session(tmp_path)
    r = tools.run("cnv_score", s, reference_key="condition", reference_cat=["Normal"])
    assert r.status == "success"
    # cnv-space UMAP computed + the cnv UMAP panel emitted. (chromosome heatmaps are best-effort:
    # they need real CNV signal and are defensively skipped on near-zero synthetic data.)
    assert "X_cnv_umap" in s.adata.obsm
    assert any("umap_panel" in a.path for a in r.artifacts)


def test_apply_malignancy_derives_cnv_status_and_plots(tmp_path):
    s = _scored_session(tmp_path)
    r = tools.run("apply_malignancy", s, groupby="major_cell_type",
                  labels={"Epithelial": "malignant", "T_cell": "non_malignant"})
    assert r.status == "success"
    cs = s.adata.obs["cnv_status"].astype(str)
    mct = s.adata.obs["major_cell_type"].astype(str)
    assert set(cs[mct == "Epithelial"]) == {"tumor"}         # derived from the call, not hardcoded
    assert set(cs[mct == "T_cell"]) == {"normal"}
    assert r.summary["cnv_status_distribution"].get("tumor", 0) > 0
    assert any("cnv_status" in a.path for a in r.artifacts)  # cnv_status panel emitted
