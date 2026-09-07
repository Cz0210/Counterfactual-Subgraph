"""Focused AIDS phase-only safety and exact-engine compatibility tests."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.baselines.comrecgc import rf_aligned_cluster_phase as phase
from src.baselines.comrecgc import rf_aligned_phase_owner as owner
from src.baselines.comrecgc import external_memory_dbscan as engine


def binding():
    return {"rows": phase.PAIR_ROWS, "arrays": {"vectors": {"size": 9_559_802_240}, "pairs": {"size": 597_487_760}}}


def test_phase_bound_keeps_pagecache_and_reserve():
    for name in ("CERTIFIED_EXACT_DBSCAN", "NATIVE_SUMMARY", "RF_WNODE_RELEASE"):
        plan = phase.phase_memory_plan(binding(), phase=name)
        assert plan["component_sum_bytes"] <= plan["charged_peak_bound_bytes"] == 14*phase.GIB
        assert plan["experiment_total_ceiling_bytes"] == 64*phase.GIB
        assert plan["components_bytes"]["full_vector_file_pages"] == 9_559_802_240
        assert plan["workers"] == 1 and not plan["shared_cache_discounted"]
    with pytest.raises(ValueError, match="Unknown phase"):
        phase.phase_memory_plan(binding(), phase="UNKNOWN")
    huge = binding(); huge["arrays"]["vectors"]["size"] *= 10
    with pytest.raises(ValueError, match="exceed"):
        phase.phase_memory_plan(huge, phase="CERTIFIED_EXACT_DBSCAN")


def test_owner_command_cannot_generate_or_search():
    kw = dict(python=sys.executable, worktree="/repo", manifest="/fresh/config.json", recourse_root="/sealed",
              pool_root="/pool", evidence="/fresh/cluster", fd=17)
    command = owner.phase_argv(phase="cluster-existing", **kw)
    assert "--writer-fd" in command and command[-1] == "17"
    assert command[command.index("--recourse-root")+1] == "/sealed"
    for action in ("recourse", "screen-pool", "search", "repair-gaps"):
        with pytest.raises(ValueError): owner.phase_argv(phase=action, **kw)
    env = owner.child_env("/repo")
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"


def test_real_fd_competition_and_child_lifecycle(tmp_path):
    lock = tmp_path/"writer.lock"
    with lock.open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX|fcntl.LOCK_NB)
        owner.validate_writer_fd(held.fileno(), tmp_path)
        code = ("import src.baselines.comrecgc as p;p.__path__.insert(0,"+repr(str(Path(owner.__file__).parent))+");"
                "from src.baselines.comrecgc.rf_aligned_phase_owner import validate_writer_fd;import sys;validate_writer_fd(int(sys.argv[1]),sys.argv[2])")
        child = subprocess.run([sys.executable,"-c",code,str(held.fileno()),str(tmp_path)], pass_fds=(held.fileno(),), capture_output=True,text=True)
        assert child.returncode == 0, child.stderr
        wrong = tmp_path/"other.lock"
        with wrong.open("a+") as unrelated:
            with pytest.raises(RuntimeError, match="not the existing"):
                owner.validate_writer_fd(unrelated.fileno(),tmp_path)
        fcntl.flock(held,fcntl.LOCK_UN)
        with pytest.raises(RuntimeError, match="No exclusive"):
            owner.validate_writer_fd(held.fileno(),tmp_path)


def test_checkpoint_observer_preserves_return_and_always_restores(tmp_path):
    calls = []
    class Observer:
        def boundary(self): calls.append("boundary")
    from src.baselines.comrecgc import external_component_summary as component
    from src.baselines.comrecgc import external_memory_recourse as summary
    modules = ((engine,"_checkpoint"),(component,"_write_checkpoint"),(summary,"_summary_checkpoint"))
    originals = [getattr(m,n) for m,n in modules]
    with patch.object(engine,"_checkpoint",side_effect=lambda *a,**k: calls.append("committed") or 7):
        replacement = engine._checkpoint
        with phase.checkpoint_observer(Observer()):
            assert engine._checkpoint() == 7
        assert engine._checkpoint is replacement
    assert calls == ["committed","boundary"]
    assert [getattr(m,n) for m,n in modules] == originals


def test_small_float32_exact_labels_core_border_and_components(tmp_path):
    values = np.zeros((12,64), dtype=np.float32)
    values[:,0] = [0,0,.001,.019,.039,.060,.3,.3,.301,.319,.339,.8]
    source = tmp_path/"vectors.npy"; np.save(source,values)
    contract = engine.ExternalDBSCANContract(eps=.02,min_samples=3,query_block_size=2,
        checkpoint_interval_blocks=1,max_rss_bytes=2*phase.GIB,expected_sklearn_version="1.7.2",
        shortcut_mode=engine.ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
        shortcut_query_block_size=3,exact_fallback_max_samples=100)
    expected = DBSCAN(eps=.02,min_samples=3).fit(values)
    observed = engine.fit_external_memory_dbscan(vectors_path=source,work_dir=tmp_path/"exact",contract=contract)
    assert np.array_equal(np.load(observed.labels_path),expected.labels_)
    assert np.array_equal(np.flatnonzero(np.load(observed.core_mask_path)),expected.core_sample_indices_)
    assert observed.cluster_count == len(set(expected.labels_)-{-1})
    assert observed.noise_count == int((expected.labels_==-1).sum())


def test_sklearn_read_only_mmap_fit_has_no_vector_copy_and_rejects_nan(tmp_path):
    path = tmp_path/"mapped.npy"
    values = np.lib.format.open_memmap(path,mode="w+",dtype="float32",shape=(65536,64)); values[:]=0;values.flush();del values
    mm = np.load(path,mmap_mode="r",allow_pickle=False)
    fitted = NearestNeighbors(radius=.02,algorithm="auto",n_jobs=1).fit(mm)
    assert fitted._fit_method == "brute"  # adaptive certificate avoids full-N query
    assert np.shares_memory(fitted._fit_X,mm) and not fitted._fit_X.flags.writeable
    bad = np.zeros((4,64),dtype=np.float32); bad[0,0]=np.nan
    with pytest.raises(ValueError): NearestNeighbors(radius=.02,n_jobs=1).fit(bad)


def test_start_gate_rechecks_after_owner_and_refuses_science(tmp_path):
    with patch.object(phase,"resource_sample",return_value={"state":"WAITING_RESOURCE"}):
        with pytest.raises(RuntimeError,match="no science started"):
            phase.require_start_admission({}, {}, tmp_path)
    assert json.loads((tmp_path/"start_resource_admission.json").read_text())["state"] == "WAITING_RESOURCE"


def test_failed_or_uncertain_submission_never_restarts(tmp_path):
    args=dict(worktree=tmp_path,root=tmp_path,phase="cluster-existing",env={})
    (tmp_path/"cluster-existing_submission.json").write_text("{}")
    with pytest.raises(RuntimeError,match="Unreconciled"): owner.run_child([],**args)
    (tmp_path/"cluster-existing_terminal.json").write_text('{"returncode":1}')
    with pytest.raises(RuntimeError,match="no automatic retry"): owner.run_child([],**args)
