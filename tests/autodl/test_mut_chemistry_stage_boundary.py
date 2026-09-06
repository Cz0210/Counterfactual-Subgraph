import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from scripts.autodl import run_mut_comrecgc_parity_standardization as runner
from src.baselines.comrecgc.contracts import write_json
from src.utils import mut_chemistry_stage_boundary as b


def _closures(root):
    for name in ("chemistry/_RUN_COMPLETE.json", "chemistry/run_manifest.json",
                 "chemistry/final_artifact_audit.json", "generation_adoption_manifest.json",
                 "historical_adoption_manifest.json", "upstream_checkout_audit.json"):
        write_json(root / name, {"run_complete": True})


def test_sealed_phase_not_final_pass_and_exact_resume(tmp_path):
    _closures(tmp_path)
    contract = {"input": "frozen", "scientific_commit": "same"}
    receipt = b.seal_boundary(tmp_path, contract)
    assert not receipt["final_cell_pass"] and not receipt["test_evaluation_started"]
    assert not (tmp_path / "PASS").exists()
    assert b.validate_boundary(tmp_path, contract) == receipt
    with pytest.raises(ValueError, match="CONTRACT_CHANGED"):
        b.validate_boundary(tmp_path, {**contract, "input": "different"})
    write_json(tmp_path / "chemistry/run_manifest.json", {"run_complete": False})
    with pytest.raises(ValueError, match="ARTIFACT_CHANGED"):
        b.validate_boundary(tmp_path, contract)


def test_resume_cannot_duplicate_started_evaluation(tmp_path):
    _closures(tmp_path)
    b.seal_boundary(tmp_path, {})
    (tmp_path / "unified_eval").mkdir()
    with pytest.raises(ValueError, match="ALREADY_STARTED"):
        b.validate_boundary(tmp_path, {})


def _monitor(tmp_path, monkeypatch):
    policy = SimpleNamespace(load_stage_policy=lambda *a, **k: {},
        stage_file_admission=lambda *a, **k: {"admitted": True, "pause_requested": False})
    monkeypatch.setitem(sys.modules, "src.utils.stage_file_policy", policy)
    config = tmp_path / "resource.json"
    write_json(config, {"stage_file_policy": {"path": "/verified"}})
    state = dict(cgroup_limit_bytes=480*b.GIB, cgroup_usage_bytes=20*b.GIB,
        headroom_bytes=460*b.GIB, failcnt=0, free_bytes=400*b.GIB, free_inodes=80000)
    monkeypatch.setattr(b, "_resources", lambda *a: dict(state))
    monitor = b.ChemistryResourceMonitor(tmp_path, tmp_path, config, stage_id=b.CHEMISTRY_STAGE)
    return monitor, state, policy


def test_budget_is_not_measured_peak_and_pressure_stops(tmp_path, monkeypatch):
    monitor, state, policy = _monitor(tmp_path, monkeypatch)
    from scripts.autodl import run_t14_route_c_owner as existing
    monkeypatch.setattr(existing, "_process_tree_snapshot", lambda pid: [
        dict(pid=7, ppid=1, start_ticks=99, rss_bytes=8*b.GIB)])
    receipt = monitor.admission()
    assert receipt["required_start_headroom_bytes"] == 320*b.GIB
    assert receipt["limits"]["memory_limits_are_budgets_not_measured_peaks"]
    monitor.sample(7)
    assert json.loads((tmp_path / "mut_chemistry_resource_progress.json").read_text())["peak_rss_bytes"] == 8*b.GIB
    state["headroom_bytes"] = 191*b.GIB
    with pytest.raises(ValueError, match="PRESSURE"):
        monitor.sample(7)


@pytest.mark.parametrize("field,value", [("max_rss_gib",97), ("other_main_reserve_gib",191),
                                         ("transient_reserve_gib",31), ("min_free_gib",99)])
def test_budgets_cannot_be_weakened(tmp_path, monkeypatch, field, value):
    _monitor(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="BUDGET_WEAKENED"):
        b.ChemistryResourceMonitor(tmp_path, tmp_path, tmp_path / "resource.json",
            stage_id=b.CHEMISTRY_STAGE, **{field:value})


def test_unknown_stage_peak_blocks(tmp_path, monkeypatch):
    monitor, state, policy = _monitor(tmp_path, monkeypatch)
    policy.stage_file_admission = lambda *a, **k: {"admitted":False, "runtime_state":"PEAK_EVIDENCE_PENDING"}
    with pytest.raises(ValueError, match="ADMISSION_BLOCKED"):
        monitor.admission()


def test_monitor_reuses_tree_reader_not_signal_platform():
    source = Path(b.__file__).read_text()
    for forbidden in ("MemAvailable", "os.kill(", "os.killpg(", "SIGKILL", "cuda", "sqlite"):
        assert forbidden not in source
    assert "_process_tree_snapshot" in source
    assert b.CHEMISTRY_NEW_INODE_BOUND == 128


def test_observer_pressure_uses_existing_child_sigterm_cleanup(tmp_path, monkeypatch):
    from scripts.autodl import run_comrecgc_standardized_continuation as c
    signals = []
    class Process:
        pid = 4242
        stopped = False
        def poll(self): return 0 if self.stopped else None
        def wait(self, timeout=None): self.stopped = True; return -15
    process = Process()
    class Barrier:
        launcher_argv = ["python", "-S", "barrier"]
        def launch(self, **kwargs): return process
        def release(self): pass
        def abort(self): pass
    def arm(**kwargs):
        write_json(Path(kwargs["record_path"]), {"binding": "test"})
        return Barrier()
    monkeypatch.setattr(c, "arm_exec_startup_barrier", arm)
    monkeypatch.setattr(c, "_read_proc_start_ticks", lambda *a, **k: 99)
    monkeypatch.setattr(c, "_proc_argv", lambda *a, **k: Barrier.launcher_argv)
    monkeypatch.setattr(c, "_wait_for_process_group_quiescence", lambda *a, **k: None)
    monkeypatch.setattr(c.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    def observe(pid):
        assert pid == 4242
        raise ValueError("PRESSURE_TEST")
    with pytest.raises(ValueError, match="PRESSURE_TEST"):
        c._run_stage(stage="chemistry", argv=["python", "chemistry"],
                     marker=tmp_path / "absent-marker", required_field="run_complete",
                     environment={}, output_root=tmp_path, resource_monitor=observe)
    assert signals == [(4242, c.signal.SIGTERM)]
    assert json.loads((tmp_path / "stage_state.json").read_text())["status"] == "FAILED"


def test_real_loop_stops_and_does_not_repeat_chemistry(tmp_path, monkeypatch):
    from scripts.autodl.run_comrecgc_standardized_continuation import ContinuationInputs
    for name in ("source","upstream","dataset","common","molclr"):
        (tmp_path/name).mkdir()
    for name in ("lineage","adopt.json","teacher","distance","csv","checkpoint","thresholds",
                 "dataset/generation_source_graphs.pt","dataset/dataset_summary.json"):
        (tmp_path/name).write_text("{}")
    inputs = ContinuationInputs(dataset="mutagenicity", source_generation_root=tmp_path/"source",
        upstream_root=tmp_path/"upstream", dataset_dir=tmp_path/"dataset", source_csv=None,
        distance_checkpoint=tmp_path/"distance", dataset_csv=tmp_path/"csv", teacher_path=tmp_path/"teacher",
        molclr_root=tmp_path/"molclr", molclr_checkpoint=tmp_path/"checkpoint",
        thresholds_path=tmp_path/"thresholds", output_root=tmp_path/"out", device="cpu", theta_star=None, cost_cap=None)
    historical = dict(independent_scientific_adoption_authorized=True, common_root=str(tmp_path/"common"),
        source_lineage_path=str(tmp_path/"lineage"), common_recourse_count=100,
        generation_adoption={"counterfactual_candidate_count":runner.SOURCE_CANDIDATE_COUNT})
    monkeypatch.setattr(runner,"_validate_historical_adoption",lambda *a,**k:historical)
    monkeypatch.setattr(runner,"_verify_adopted_generation_integrity",lambda *a:{"status":"PASS"})
    monkeypatch.setattr(runner,"verify_checkout",lambda *a,**k:{"passed":True})
    monkeypatch.setattr(runner,"_git_head",lambda:"a"*40)
    class Monitor:
        def __init__(self,*a,**k): self.root_identity=None
        def admission(self): pass
        def sample(self,pid): pass
    monkeypatch.setattr(b,"ChemistryResourceMonitor",Monitor)
    seen=[]
    def stage(**k):
        seen.append(k["stage"])
        if k["stage"]=="chemistry":
            for n in ("_RUN_COMPLETE.json","run_manifest.json","final_artifact_audit.json"):
                write_json(inputs.output_root/"chemistry"/n,{"run_complete":True})
        else: raise RuntimeError("STOP_AT_FIRST_EVALUATION")
    monkeypatch.setattr(runner,"_run_stage",stage)
    kwargs=dict(common_adoption_path=None,historical_adoption_path=tmp_path/"adopt.json",resource_options={})
    first=runner.run(inputs,through_stage="chemistry",**kwargs)
    assert seen==["chemistry"] and not first["final_cell_pass"]
    with pytest.raises(RuntimeError,match="STOP_AT_FIRST"):
        runner.run(inputs,resume_after_chemistry=True,**kwargs)
    assert seen==["chemistry","unified_eval"]
