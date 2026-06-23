"""Parameter catalog + user preset (param_overrides) — the pre-selectable tuning knobs."""

import yaml

from scpilot.params import (load_param_file, params_catalog, render_catalog_table,
                            render_template, validate_overrides)


def test_catalog_merges_hints_and_defaults():
    cat = params_catalog()
    # defaults come from the tool signatures, descriptions from _PARAM_HINTS
    assert cat["cluster_sweep"]["res_min"]["default"] == 0.1
    assert cat["qc_metrics"]["n_mads"]["default"] == 5.0
    assert cat["preprocess"]["n_top_genes"]["default"] == 2000
    entry = cat["cluster"]["resolution"]
    assert set(entry) == {"type", "default", "description"} and entry["description"]


def test_template_is_yaml_loadable_and_table_nonempty():
    cat = params_catalog()
    data = yaml.safe_load(render_template(cat))     # all knobs commented → tool keys map to None
    assert isinstance(data, dict) and "cluster_sweep" in data
    assert "cluster_sweep" in render_catalog_table(cat)


def test_load_param_file_and_validate(tmp_path):
    p = tmp_path / "preset.yaml"
    p.write_text("cluster:\n  resolution: 0.7\npreprocess:\n  n_top_genes: 3000\n"
                 "bogus_tool:\n  x: 1\nempty_tool:\n")
    ov = load_param_file(str(p))
    assert ov == {"cluster": {"resolution": 0.7}, "preprocess": {"n_top_genes": 3000},
                  "bogus_tool": {"x": 1}}                   # empty_tool (None) dropped
    warns = validate_overrides(ov)
    assert any("bogus_tool" in w for w in warns)            # unknown tool flagged (not fatal)
    assert not any("cluster" in w for w in warns)           # valid tool/param: no warning
