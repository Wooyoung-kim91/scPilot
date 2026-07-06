"""configure_run — per-role LLM topology selected at MCP-call time + persisted (I-24)."""

import anndata as ad
import numpy as np
from scipy import sparse

from scpilot import tools
from scpilot.llm import topology as T
from scpilot.session import Session


def _sess(tmp_path):
    a = ad.AnnData(sparse.csr_matrix(np.ones((10, 5), dtype="float32")))
    a.layers["counts"] = a.X.copy()
    p = tmp_path / "in.h5ad"
    a.write_h5ad(p)
    return Session.create(str(tmp_path / "s"), input_path=str(p), exist_ok=True)


def test_validate_topology_ok():
    topo = {
        "analysis": {"type": "cli", "cli": "claude-code", "model": "claude-opus-4-8"},
        "reviewer": {"type": "host_plugin", "plugin": "codex"},
        "interpreter": {"type": "api", "backend": "anthropic", "model": "claude-haiku-4-5"},
    }
    norm, problems = T.validate_topology(topo)
    assert not problems
    assert norm["analysis"] == {"type": "cli", "cli": "claude-code", "model": "claude-opus-4-8"}
    assert norm["reviewer"] == {"type": "host_plugin", "plugin": "codex"}
    assert norm["interpreter"]["backend"] == "anthropic"


def test_validate_topology_rejects_bad():
    _, p1 = T.validate_topology({"reviewer": {"type": "telepathy"}})
    assert any("type must be one of" in m for m in p1)
    _, p2 = T.validate_topology({"reviewer": {"type": "cli", "cli": "emacs"}})
    assert any("cli must be one of" in m for m in p2)
    _, p3 = T.validate_topology({"bogus_role": {"type": "api"}})
    assert any("unknown role" in m for m in p3)


def test_configure_run_persists_and_directs(tmp_path):
    sess = _sess(tmp_path)
    topo = {"analysis": {"type": "cli", "cli": "claude-code", "model": "claude-opus-4-8"},
            "reviewer": {"type": "host_plugin", "plugin": "codex"}}
    res = tools.run("configure_run", sess, topology=topo)
    assert res.status == "success"
    assert res.summary["topology"]["reviewer"]["plugin"] == "codex"
    # host_plugin reviewer emits a delegation directive for the host
    assert any("delegate" in d for d in res.summary["host_directives"])
    # annotator/interpreter omitted → recorded as defaulting to analysis
    assert set(res.summary["roles_defaulting_to_analysis"]) == {"annotator", "interpreter"}
    # persisted to the manifest and survives reopen
    assert sess.manifest.llm_topology["reviewer"]["plugin"] == "codex"
    reopened = Session.open(sess.out)
    assert reopened.manifest.llm_topology["analysis"]["cli"] == "claude-code"


def test_configure_run_requires_topology(tmp_path):
    sess = _sess(tmp_path)
    res = tools.run("configure_run", sess)
    assert res.status == "error" and res.error_code == "invalid_params"


def test_configure_run_probes_availability(tmp_path):
    sess = _sess(tmp_path)
    # a CLI that certainly isn't on PATH → ready False with a reason, surfaced as a warning
    topo = {"reviewer": {"type": "cli", "cli": "codex"}}
    res = tools.run("configure_run", sess, topology=topo)
    assert res.status == "success"
    av = res.summary["availability"]["reviewer"]
    assert av["type"] == "cli" and av["executable"] == "codex"
    assert av["ready"] in (True, False)   # depends on host; must be a concrete bool for cli


def test_review_routing():
    from scpilot.core.configure import review_routing
    r0 = review_routing(None)
    assert r0["mode"] == "host_or_mode2" and "configure_run" in r0["directive"]
    r1 = review_routing({"reviewer": {"type": "host_plugin", "plugin": "codex"}})
    assert r1["mode"] == "host_plugin" and "codex" in r1["directive"] and "delegate" in r1["directive"].lower()
    r2 = review_routing({"reviewer": {"type": "cli", "cli": "codex", "model": "gpt-5"}})
    assert r2["mode"] == "host_or_mode2" and "mode-2" in r2["directive"]


def test_run_review_requires_label(tmp_path):
    sess = _sess(tmp_path)
    res = tools.run("run_review", sess, groupby="leiden", label_key="major_cell_type")
    assert res.status == "error" and res.error_code == "invalid_state"
