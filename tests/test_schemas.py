"""Unit tests for the frozen ToolResult contract — scpilot plan A4."""

import json

import numpy as np
import pandas as pd
import pytest

from scpilot import schemas as S


def _assert_json_roundtrips(d: dict) -> str:
    """to_dict() output must be strict-JSON serializable (no NaN/Inf/numpy/Path)."""
    return json.dumps(d, allow_nan=False)


def test_success_result_is_json_serializable():
    r = S.success(
        "cluster",
        summary={"n_clusters": 12, "resolution": 0.5},
        determinism_grade="B",
        suggested_next_tools=["markers"],
    )
    d = r.to_dict()
    _assert_json_roundtrips(d)
    assert d["status"] == "success"
    assert d["tool"] == "cluster"
    assert d["summary"]["n_clusters"] == 12
    assert d["determinism_grade"] == "B"
    assert d["error_code"] is None


def test_error_result_fields():
    r = S.error("cnv", "capability_unavailable", "infercnvpy missing", recoverable=False)
    d = r.to_dict()
    _assert_json_roundtrips(d)
    assert d["status"] == "error"
    assert d["error_code"] == "capability_unavailable"
    assert d["recoverable"] is False
    assert "infercnvpy" in d["error"]


def test_numpy_and_nan_are_sanitized():
    # tools often drop numpy scalars / NaN into summary — must survive to strict JSON
    r = S.success(
        "qc",
        summary={
            "median_genes": np.float64(1234.5),
            "n_cells": np.int64(180977),
            "fraction": float("nan"),     # -> None
            "scores": np.array([1.0, 2.0, 3.0]),
        },
    )
    d = r.to_dict()
    _assert_json_roundtrips(d)
    assert d["summary"]["median_genes"] == pytest.approx(1234.5)
    assert d["summary"]["n_cells"] == 180977
    assert d["summary"]["fraction"] is None          # NaN -> None
    assert d["summary"]["scores"] == [1.0, 2.0, 3.0]  # ndarray -> list


def test_table_preview_caps_rows():
    df = pd.DataFrame({"gene": [f"G{i}" for i in range(100)], "score": range(100)})
    tp = S.table_preview(df, max_rows=20)
    assert tp.n_rows_total == 100
    assert tp.n_rows_shown == 20
    assert tp.truncated is True
    assert tp.columns == ["gene", "score"]
    # embed in a result and confirm JSON-serializable
    r = S.success("markers", tables={"top_markers": tp})
    d = r.to_dict()
    _assert_json_roundtrips(d)
    assert d["tables"]["top_markers"]["n_rows_total"] == 100
    assert len(d["tables"]["top_markers"]["rows"]) == 20


def test_table_preview_not_truncated_when_small():
    df = pd.DataFrame({"a": [1, 2, 3]})
    tp = S.table_preview(df, max_rows=20)
    assert tp.truncated is False
    assert tp.n_rows_shown == 3


def test_artifact_path_is_absolute(tmp_path):
    f = tmp_path / "markers.csv"
    f.write_text("gene,score\nA,1\n")
    art = S.artifact_csv(str(f), n_rows=1, n_cols=2, description="markers")
    assert art.path == str(f.resolve())
    assert art.kind == "csv"
    assert art.meta["n_rows"] == 1
    assert art.meta["bytes"] > 0           # size auto-filled when file exists
    d = S.success("markers", artifacts=[art]).to_dict()
    _assert_json_roundtrips(d)
    assert d["artifacts"][0]["path"].endswith("markers.csv")


def test_job_status_serializes_with_attempts():
    js = S.JobStatus(
        job_id="job-1", tool="integrate", state="running", progress=0.4,
        attempts=[S.FallbackAttempt(method="scvi", status="failed", error="OOM")],
    )
    import dataclasses
    d = S._sanitize(dataclasses.asdict(js))
    _assert_json_roundtrips(d)
    assert d["state"] == "running"
    assert d["attempts"][0]["method"] == "scvi"
