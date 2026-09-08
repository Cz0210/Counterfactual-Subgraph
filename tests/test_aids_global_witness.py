import json
from pathlib import Path

import numpy as np
import pytest
import sklearn
from sklearn.neighbors import NearestNeighbors

from src.baselines.comrecgc.aids_global_witness import scan, memory_plan
from src.baselines.comrecgc.rf_aligned_phase_owner import phase_argv


def run(tmp_path, values, *, anchors, failures, seeds, **kw):
    return scan(np.asarray(values, dtype=np.float32), anchors=anchors, failures=failures,
                seeds=seeds, eps=.02, min_samples=3, expected_version=sklearn.__version__,
                output=tmp_path, binding={"seed_failure_ledger_complete": False}, **kw)


def test_anchor_subgraph_insufficient_can_have_global_core(tmp_path):
    # The anchor at row3 has only itself in the anchor subgraph. Real global
    # rows4/5 provide the missing two neighbors; the old failure was inconclusive.
    values = [[0.], [.001], [.002], [1.], [1.001], [1.002]]
    r = run(tmp_path, values, anchors=[0,1,2,3], failures=[3], seeds=[0,1,2], block_rows=2)
    assert r["verified_core_anchors"] == 4
    assert not r["complete_dbscan_certificate"]  # no complete seed-failure authority
    cp = json.loads((tmp_path/"witness_checkpoint.json").read_text())
    assert [w["row_id"] for w in cp["witnesses"]["3"]] == [3,4,5]


def test_partial_scan_not_noncore_and_resume(tmp_path):
    values = [[0.],[.001],[.002],[1.],[1.001],[1.002]]
    a = run(tmp_path, values, anchors=[0,1,2,3], failures=[3], seeds=[0,1,2], block_rows=2, stop_after_blocks=1)
    assert a["scanned_rows"] == 2 and not a["proven_noncore"]
    b = run(tmp_path, values, anchors=[0,1,2,3], failures=[3], seeds=[0,1,2], block_rows=2)
    assert b["scanned_rows"] == 6 and b["new_rows_this_process"] == 4
    direct = run(tmp_path/"direct", values, anchors=[0,1,2,3], failures=[3], seeds=[0,1,2], block_rows=2)
    old = json.loads((tmp_path/"witness_checkpoint.json").read_text())
    new = json.loads((tmp_path/"direct/witness_checkpoint.json").read_text())
    assert old["witnesses"] == new["witnesses"]
    assert b["verified_core_anchors"] == direct["verified_core_anchors"]


def test_cross_partition_and_exact_boundary_neighbors(tmp_path):
    values=np.array([[0.],[.02],[np.nextafter(np.float32(.02),np.float32(1))],[.005]], dtype=np.float32)
    r=run(tmp_path,values,anchors=[0,1],failures=[1],seeds=[0],block_rows=1)
    model=NearestNeighbors(radius=.02,algorithm="brute",metric="euclidean").fit(values[[0,1]])
    ds, ids=model.radius_neighbors(values,return_distance=True)
    cp=json.loads((tmp_path/"witness_checkpoint.json").read_text())
    for a in [0,1]:
        expected=[(i,float(dist[list(neigh).index(a)])) for i,(neigh,dist) in enumerate(zip(ids,ds)) if a in neigh][:3]
        assert [(v["row_id"],v["distance"]) for v in cp["witnesses"][str(a)]] == expected
    assert r["exhaustive_source_scan"]


def test_exhaustive_noncore_only_after_all_rows(tmp_path):
    r=run(tmp_path,[[0],[1],[2]],anchors=[0,1],failures=[1],seeds=[0],block_rows=1)
    assert r["proven_noncore"] == [0,1]
    assert not r["complete_dbscan_certificate"]


def test_certificate_needs_all_core_and_connectivity_and_ledger(tmp_path):
    x=np.array([[0],[.001],[.002],[.003]],dtype=np.float32)
    common=dict(anchors=[0,1,2,3],failures=[3],seeds=[0,1,2],eps=.02,min_samples=3,
                expected_version=sklearn.__version__,block_rows=2)
    r=scan(x,output=tmp_path,binding={"seed_failure_ledger_complete":True},**common)
    assert r["complete_dbscan_certificate"]
    assert r["state"] == "GLOBAL_ALL_CORE_ONE_COMPONENT_CERTIFIED"


def test_resource_pause_keeps_cursor_without_fake_noise(tmp_path):
    r=run(tmp_path,[[0],[1],[2]],anchors=[0,1],failures=[1],seeds=[0],boundary=lambda:False)
    assert r["scanned_rows"] == 0 and r["proven_noncore"] == []


def test_scope_budget_and_real_inherited_writer_fd(tmp_path):
    p=memory_plan({"arrays":{"vectors":{"size":9559802240}}})
    assert p["component_sum_bytes"] <= p["charged_peak_bound_bytes"] == 2*1024**3
    assert p["maximum_new_rows_this_process"] == 1048576
    a=phase_argv(python="/python",worktree=tmp_path,manifest=tmp_path/"manifest",recourse_root=tmp_path,
        pool_root=tmp_path,evidence=tmp_path,phase="global-witness",fd=17)
    assert a[-2:] == ["--writer-fd","17"]
    with pytest.raises(ValueError,match="memory admission"):
        run(tmp_path,[[0]],anchors=[0],failures=[],seeds=[0],max_new_rows=1048577)


def test_no_implicit_global_dbscan_or_pair_regeneration():
    text=Path("src/baselines/comrecgc/aids_global_witness.py").read_text()
    assert "fit_external_memory_dbscan(" not in text
    assert "pair_rows_recomputed\": 0" in text
    assert "exact_fallback_max_samples" not in text
