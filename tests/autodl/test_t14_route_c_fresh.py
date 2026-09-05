from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest

from scripts.autodl import run_t14_route_c_owner as route_c_owner
from src.baselines import tastemolnet_t14_route_c_fresh as route_c

from src.baselines.tastemolnet_t14_route_c_fresh import (
    EARLY_CHECKPOINT_STEPS,
    FIRST_CHECKPOINT_STEP,
    PROMOTABLE_CHECKPOINT_STEP,
    RELOAD_REPLAY_END_STEP,
    RouteCGraphMap,
    RouteCStateUpdater,
    T14RouteCFreshError,
    audit_no_live_t14_science_owner,
    append_step_state,
    build_spec,
    checkpoint_targets,
    compare_step_ledgers,
    file_sha256,
    recover_route_c_external_state,
    retire_failed_route_c_current,
    scientific_state_digest,
    stable_sha256,
    validate_spec,
    write_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _fake_graph_hash(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _simple_graph_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.baselines.comrecgc import live_graph_state

    monkeypatch.setattr(live_graph_state, "_entry_graph_sha256", _fake_graph_hash)


def _candidate(graph_hash: str, frequency: int = 2) -> dict[str, object]:
    return {
        "frequency": frequency,
        "graph_hash": graph_hash,
        "importance_parts": (0.75, 1.0),
        "input_graphs_covering_list": (1, 0, 1),
    }


def test_disk_backed_graph_store_is_append_only_and_lazy(tmp_path: Path) -> None:
    updater = RouteCStateUpdater(
        tmp_path / "state", candidate_capacity=4, record_capacity=12, lru_capacity=1
    )
    graph_map = RouteCGraphMap(
        updater.graph_store, {}, next_sequence=updater.next_sequence
    )
    graph_map["a"] = [{"nodes": [1]}]
    graph_map["b"] = [{"nodes": [2]}]
    first_id = updater.graph_store.graph_id("a")
    del graph_map["a"]
    assert "a" not in graph_map
    assert graph_map.contains_resolvable("a")
    assert graph_map["a"] == [{"nodes": [1]}]
    assert updater.graph_store.graph_id("a") == first_id
    assert len(updater.graph_store._lru) == 1  # noqa: SLF001
    assert updater.graph_store.checkpoint_state()["graph_objects_in_checkpoint"] == 0
    updater.close()


def test_route_c_graph_map_preserves_reference_eviction_state(tmp_path: Path) -> None:
    updater = RouteCStateUpdater(
        tmp_path / "state", candidate_capacity=4, record_capacity=12, lru_capacity=1
    )
    module = SimpleNamespace(counterfactual_candidates=[], graph_index_map={})
    graph_map = RouteCGraphMap(
        updater.graph_store,
        {},
        next_sequence=updater.next_sequence,
        module=module,
    )
    graph_map["a"] = [{"nodes": [1]}]
    graph_map.begin_move(["a"], current_step=1)
    with graph_map.pin_many(["a"]):
        del graph_map["a"]
        assert graph_map.deferred_deletions == {"a"}
    graph_map.end_move()
    assert graph_map.contains_resolvable("a")
    assert graph_map.deferred_deletions == set()
    assert graph_map.eviction_attempts == 1
    assert graph_map.eviction_committed == 1
    assert graph_map.eviction_deferred == 1
    assert graph_map.active_eviction_prevented == 1
    assert graph_map.deferred_flushed == 1
    assert graph_map.recent_evictions[-1]["was_pinned"] is True

    module.counterfactual_candidates = [{"graph_hash": "unmaterialized"}]
    del graph_map["unmaterialized"]
    assert graph_map.missing_unmaterialized_eviction_count == 1
    checkpoint = graph_map.export_checkpoint_state()
    restored = RouteCGraphMap(
        updater.graph_store,
        {},
        next_sequence=updater.next_sequence,
        module=module,
    )
    restored.restore_checkpoint_state(checkpoint)
    assert restored.runtime_diagnostics()["recent_evictions"] == list(
        graph_map.recent_evictions
    )
    assert restored.eviction_attempts == 2
    updater.close()


def _parity_fixture(*, graph_nodes: list[int] | None = None) -> tuple[object, ...]:
    graph_nodes = [1] if graph_nodes is None else graph_nodes
    candidate = _candidate("a", frequency=2)
    module = SimpleNamespace(
        counterfactual_candidates=[candidate],
        graph_index_map={"a": 0},
        input_graphs_covered=[1, 0],
        covering_graphs={"a"},
        start={"a": 0},
        is_sample=True,
        starting_step=0,
        traversed_hashes=["a"],
        sample_size=1,
        MAX_COUNTERFACTUAL_SIZE=100_000,
        graph_map={"a": [{"nodes": graph_nodes}]},
    )
    record = SimpleNamespace(
        graph_identity_sha256="a",
        canonical_graph="graph-a",
        probabilities=(0.25, 0.5, 0.25),
        prediction=1,
        score=0.5,
        candidate=False,
        valid_fullgraph=True,
        model_graph_sha256="b" * 64,
        model_graph_payload={"nodes": graph_nodes},
        embedding_sha256="c" * 64,
        embedding_dtype="float32",
    )
    bridge = SimpleNamespace(
        lineage_occurrences={"a": {"parent": 1}},
        records={"a": record},
        graph_collision_payloads={"a": {"nodes": graph_nodes}},
        call_count=1,
        evaluated_graph_count=1,
        calculate_hash_count=1,
        pending_hash_count=0,
    )
    entry = SimpleNamespace(
        target_hashes=("b",),
        actions=({"remove": 1},),
        importance_parts=((0.5, 1.0),),
        embeddings=((0.1, 0.2),),
    )
    transitions = SimpleNamespace(
        _entries={"a": entry},
        move_count=1,
        deferred_deletion_count=0,
        applied_deferred_deletion_count=0,
        cancelled_deferred_deletion_count=0,
        missing_lookup_count=0,
        captured_action_count=1,
    )
    loop = SimpleNamespace(
        completed_step=1,
        start_graph_hashes=("a",),
        current_graph_hashes=("b",),
        restart_indices=(0,),
    )
    diagnostics = {
        "eviction_attempts": 1,
        "eviction_committed": 1,
        "eviction_deferred": 0,
        "active_eviction_prevented": 0,
        "deferred_flushed": 0,
        "deferred_deletions": 0,
        "missing_unmaterialized_eviction_count": 0,
        "recent_evictions": [{"graph_hash": "z", "current_step": 1}],
    }
    live = SimpleNamespace(runtime_diagnostics=lambda: diagnostics)
    return module, bridge, loop, transitions, live, diagnostics


def test_scientific_digest_detects_all_route_c_scientific_mutations() -> None:
    def digest(values: tuple[object, ...]) -> dict[str, object]:
        module, bridge, loop, transitions, live, _diagnostics = values
        return scientific_state_digest(
            module=module,
            bridge=bridge,
            loop_state=loop,
            selected={"action": 1, "importance": 0.5},
            transition_map=transitions,
            live_graph_state=live,
        )

    baseline = digest(_parity_fixture())
    for mutate, field in (
        (lambda values: values[0].counterfactual_candidates[0].update(input_graphs_covering_list=(0, 1)), "candidate_records_sha256"),
        (lambda values: values[0].covering_graphs.add("b"), "module_scientific_state_sha256"),
        (lambda values: values[1].lineage_occurrences["a"].update(parent=2), "lineage_sha256"),
        (lambda values: values[0].graph_map["a"][0].update(nodes=[2]), "active_graph_state_sha256"),
        (lambda values: setattr(values[3]._entries["a"], "actions", ({"remove": 2},)), "transition_state_sha256"),
        (lambda values: values[5].update(eviction_committed=2), "lineage_collision_eviction_sha256"),
    ):
        values = _parity_fixture()
        mutate(values)
        assert digest(values)[field] != baseline[field]


def test_scientific_digest_accepts_official_sparse_covering_list() -> None:
    torch = pytest.importorskip("torch")
    values = _parity_fixture()
    values[0].counterfactual_candidates[0]["input_graphs_covering_list"] = (
        torch.tensor([1.0, 0.0]).to_sparse()
    )
    digest = scientific_state_digest(
        module=values[0],
        bridge=values[1],
        loop_state=values[2],
        selected={"action": 1},
        transition_map=values[3],
        live_graph_state=values[4],
    )
    assert len(digest["candidate_covering_lists_sha256"]) == 64


def test_external_state_recovery_rolls_back_uncommitted_suffix(tmp_path: Path) -> None:
    output_root = (tmp_path / "output").resolve()
    updater = RouteCStateUpdater(
        output_root / "route_c_state",
        candidate_capacity=4,
        record_capacity=12,
        lru_capacity=2,
    )
    updater.graph_store.put("g0", [{"nodes": [0]}], sequence_id=updater.next_sequence())
    updater.candidates.append(_candidate("g0", frequency=7))
    checkpoint = updater.checkpoint_state()
    snapshot = output_root / "checkpoint.sqlite3"
    sealed = sqlite3.connect(snapshot)
    updater.graph_store.checkpoint_connection.backup(sealed)
    sealed.close()
    updater.graph_store.put("g1", [{"nodes": [1]}], sequence_id=updater.next_sequence())
    updater.candidates.append(_candidate("g1", frequency=3))
    updater.close()

    validation = SimpleNamespace(completed_step=500, checkpoint_digest="d" * 64)
    loaded = SimpleNamespace(
        validation=validation,
        algorithm_state={
            "schema_version": "tastemolnet_t14_route_c_runtime_v1",
            "route_c_state": checkpoint,
            "live_graph_state": {"store": checkpoint["graph_store"]},
        },
        sqlite_snapshot_path=snapshot,
    )
    promotion = {
        "schema_version": "tastemolnet_t14_route_c_promotion_v1",
        "status": "PASS",
        "output_root": str(output_root),
        "completed_step": 500,
        "checkpoint_digest": "d" * 64,
        "payload_reload_pass": True,
        "latest_promoted": True,
    }
    promotion["receipt_sha256"] = stable_sha256(promotion)
    receipt = recover_route_c_external_state(
        output_root=output_root,
        loaded=loaded,
        promotion_receipt=promotion,
    )
    assert receipt["status"] == "PASS"
    assert receipt["removed_suffix_bytes"]["graph_blob"] > 0
    restored = RouteCStateUpdater(
        output_root / "route_c_state",
        candidate_capacity=4,
        record_capacity=12,
        lru_capacity=2,
        resume=True,
    )
    restored.restore_checkpoint_state(checkpoint)
    assert restored.graph_store.count() == 1
    assert [dict(row) for row in restored.candidates] == [_candidate("g0", frequency=7)]
    restored.close()


def test_watchdog_rss_includes_exact_descendant_tree(tmp_path: Path) -> None:
    proc = tmp_path / "proc"

    def process(pid: int, ppid: int, ticks: int, rss_kib: int, comm: str) -> None:
        root = proc / str(pid)
        root.mkdir(parents=True)
        fields = ["S", str(ppid), *("0" for _ in range(17)), str(ticks)]
        (root / "stat").write_text(
            f"{pid} ({comm}) " + " ".join(fields) + "\n", encoding="utf-8"
        )
        (root / "status").write_text(f"VmRSS:\t{rss_kib} kB\n", encoding="utf-8")

    process(100, 1, 1000, 10, "science root")
    process(101, 100, 1001, 20, "python child")
    process(102, 101, 1002, 30, "worker grandchild")
    process(200, 1, 2000, 999, "unrelated")
    rows = route_c_owner._process_tree_snapshot(100, proc_root=proc)  # noqa: SLF001
    assert [row["pid"] for row in rows] == [100, 101, 102]
    assert sum(row["rss_bytes"] for row in rows) == 60 * 1024


def test_route_c_convergence_requires_two_post_10k_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _path = _route_spec(tmp_path)
    checkpoints = Path(spec["output_root"]) / "checkpoints"
    for step in (5_000, 10_000, 12_500):
        (checkpoints / f"step-{step:012d}").mkdir(parents=True)

    ranking = [f"{index:064x}" for index in range(20)]

    def summary(*, spec: object, step: int) -> dict[str, object]:
        del spec
        return {
            "step": step,
            "checkpoint_digest": f"{step:064x}",
            "candidate_frequency": {key: 100 - index for index, key in enumerate(ranking)},
            "top100_candidate_hashes": ranking,
            "top20_provisional_rule_hashes": ranking,
            "valid_unique_rule_count": 20,
            "lineage_error_count": 0,
            "train_coverage": 0.5,
        }

    monkeypatch.setattr(route_c, "_route_c_checkpoint_summary", summary)
    audit = route_c.audit_route_c_convergence(spec)
    assert audit["converged"] is True
    assert audit["consecutive_passing_windows"] == 2
    assert [row["after_step"] for row in audit["windows"]] == [10_000, 12_500]


def test_memmap_frequency_metadata_and_checkpoint250_reload(tmp_path: Path) -> None:
    root = tmp_path / "state"
    updater = RouteCStateUpdater(
        root, candidate_capacity=4, record_capacity=12, lru_capacity=2
    )
    updater.graph_store.put("g0", [{"nodes": [6]}], sequence_id=updater.next_sequence())
    updater.candidates.append(_candidate("g0"))
    updater.candidates[0]["frequency"] = 7
    before = updater.checkpoint_state()
    assert before["candidates"]["mmap_frequency"] is True
    assert before["candidates"]["mmap_metadata"] is True
    assert before["candidates"]["full_python_candidate_list_saved"] is False
    updater.close()

    restored = RouteCStateUpdater(
        root,
        candidate_capacity=4,
        record_capacity=12,
        lru_capacity=2,
        resume=True,
    )
    restored.restore_checkpoint_state(before)
    assert restored.candidates[0]["graph_hash"] == "g0"
    assert restored.candidates[0]["frequency"] == 7
    assert int(restored.candidates.metadata[0]["graph_id"]) > 0
    restored.close()


def test_candidate_proxy_streams_as_exact_official_dict(tmp_path: Path) -> None:
    updater = RouteCStateUpdater(
        tmp_path / "state", candidate_capacity=4, record_capacity=12, lru_capacity=2
    )
    updater.graph_store.put("g0", [{"nodes": [6]}], sequence_id=updater.next_sequence())
    expected = _candidate("g0", frequency=7)
    updater.candidates.append(expected)
    materialized = dict(updater.candidates[0])
    assert type(materialized) is dict
    assert materialized == expected
    updater.close()


def test_candidate_proxy_swap_and_tail_replacement_keep_active_index_exact(
    tmp_path: Path,
) -> None:
    updater = RouteCStateUpdater(
        tmp_path / "state", candidate_capacity=2, record_capacity=8, lru_capacity=2
    )
    for key in ("g0", "g1", "g2"):
        updater.graph_store.put(
            key, [{"nodes": [key]}], sequence_id=updater.next_sequence()
        )
    updater.candidates.append(_candidate("g0"))
    updater.candidates.append(_candidate("g1"))
    updater.candidates[0], updater.candidates[1] = (
        updater.candidates[1],
        updater.candidates[0],
    )
    assert [row["graph_hash"] for row in updater.candidates] == ["g1", "g0"]
    updater.candidates[-1] = _candidate("g2")
    graph_ids = updater.candidates._active_record_by_graph_id  # noqa: SLF001
    assert updater.graph_store.graph_id("g0") not in graph_ids
    assert updater.graph_store.graph_id("g1") in graph_ids
    assert updater.graph_store.graph_id("g2") in graph_ids
    updater.close()


def test_single_state_updater_rejects_other_thread(tmp_path: Path) -> None:
    updater = RouteCStateUpdater(
        tmp_path / "state", candidate_capacity=2, record_capacity=4
    )
    errors: list[Exception] = []

    def mutate() -> None:
        try:
            updater.next_sequence()
        except Exception as exc:  # pragma: no branch - one required error
            errors.append(exc)

    thread = threading.Thread(target=mutate)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], T14RouteCFreshError)
    updater.close()


def test_compact_transition_and_checkpoint_schedule() -> None:
    assert FIRST_CHECKPOINT_STEP == 50
    assert EARLY_CHECKPOINT_STEPS == (50, 100, 250)
    assert PROMOTABLE_CHECKPOINT_STEP == 500
    assert RELOAD_REPLAY_END_STEP == 510
    assert checkpoint_targets(completed_step=0, stop_step=510, route_c=True) == (
        50,
        100,
        250,
        500,
    )
    assert checkpoint_targets(completed_step=500, stop_step=20_000, route_c=True) == (
        2_500,
        5_000,
        7_500,
        10_000,
        12_500,
        15_000,
        17_500,
        20_000,
    )
    assert checkpoint_targets(
        completed_step=20_000, stop_step=25_000, route_c=True
    ) == (22_500, 25_000)


def _ledger(path: Path, *, mutate_step: int | None = None) -> None:
    for step in range(1, 511):
        row = {
            "schema_version": "tastemolnet_t14_route_c_step_state_v1",
            "completed_step": step,
            "sequence_id": step,
            "rng_state_sha256": "a" * 64,
            "candidate_universe_sha256": (
                "b" * 64 if step != mutate_step else "c" * 64
            ),
        }
        append_step_state(path, row)


def test_reference_lowmemory_500_parity_and_501_510(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    lowmemory = tmp_path / "lowmemory.jsonl"
    _ledger(reference)
    _ledger(lowmemory)
    assert compare_step_ledgers(reference, lowmemory, end_step=500)["status"] == "PASS"
    assert compare_step_ledgers(
        reference, lowmemory, start_step=501, end_step=510
    )["status"] == "PASS"


def test_parity_fails_at_first_discrete_divergence(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    lowmemory = tmp_path / "lowmemory.jsonl"
    _ledger(reference)
    _ledger(lowmemory, mutate_step=507)
    receipt = compare_step_ledgers(
        reference, lowmemory, start_step=501, end_step=510
    )
    assert receipt["status"] == "FAILED"
    assert receipt["first_semantic_divergence_step"] == 507


def _route_spec(tmp_path: Path) -> tuple[dict[str, object], Path]:
    attempt = str(uuid4())
    wrapper = REPO_ROOT / "scripts/autodl/run_tastemolnet_t14_comrecgc_full.sh"
    owner = REPO_ROOT / "scripts/autodl/run_t14_route_c_owner.py"
    counters = tmp_path / "cgroup"
    counters.mkdir()
    for name, value in (("limit", "1000000"), ("current", "1"), ("failcnt", "0")):
        (counters / name).write_text(value, encoding="utf-8")
    spec = build_spec(
        attempt_uuid=attempt,
        execution_commit="1" * 40,
        python=Path(sys.executable).resolve(),
        science_wrapper=wrapper,
        owner_entrypoint=owner,
        output_root=tmp_path / f"science-{attempt}",
        owner_root=tmp_path / f"owner-{attempt}",
        cgroup_limit_path=counters / "limit",
        cgroup_current_path=counters / "current",
        cgroup_failcnt_path=counters / "failcnt",
        forbidden_legacy_root=tmp_path / "legacy-12500",
        science_environment={
            "RUN_TASTEMOLNET": "1",
            "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
            "TASTE_PAPER_RESULTS_ALLOWED": "1",
            "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
            "RUN_GNN_ABLATION": "0",
            "RUN_LLM_ABLATION": "0",
        },
        max_process_rss_bytes=100,
        launch_headroom_bytes=100,
        runtime_headroom_bytes=50,
    )
    path = tmp_path / "T14_ROUTE_C_TASK_SPEC.json"
    write_spec(path, spec)
    return spec, path


def test_canary_promotion_contract_and_old_12500_read_only(tmp_path: Path) -> None:
    spec, path = _route_spec(tmp_path)
    loaded = validate_spec(json.loads(path.read_text(encoding="utf-8")))
    assert loaded == spec
    assert loaded["promotable_checkpoint_step"] == 500
    assert loaded["forbidden_legacy_checkpoint_step"] == 12_500
    assert loaded["legacy_checkpoint_loaded"] is False
    assert loaded["route_c_state"]["atomic_index_checkpoint"] is True


def test_no_cal_test_generation_and_owner_is_t14_only(tmp_path: Path) -> None:
    spec, _path = _route_spec(tmp_path)
    encoded = json.dumps(spec, sort_keys=True)
    assert "calibration" not in encoded.lower()
    assert '"test"' not in encoded.lower()
    owner = (REPO_ROOT / "scripts/autodl/run_t14_route_c_owner.py").read_text(
        encoding="utf-8"
    )
    assert "run_mut" not in owner
    assert "run_t12" not in owner
    assert "forbidden_legacy_root" in owner


def test_20k_25k_policy_and_launcher_static_contract() -> None:
    source = (REPO_ROOT / "src/baselines/tastemolnet_comrecgc_full.py").read_text(
        encoding="utf-8"
    )
    launcher = (REPO_ROOT / "scripts/autodl/launch_t14_route_c_once.sh").read_text(
        encoding="utf-8"
    )
    assert "M_MAX = 20_000" in source
    assert "M_FALLBACK_MAX = 25_000" in source
    assert "ALLOW_T14_ROUTE_C" in launcher
    assert "RUN_GNN_ABLATION" in launcher
    assert "RUN_LLM_ABLATION" in launcher
    assert 'flock -n 9' in launcher
    assert 't14_route_c/launch.lock' in launcher
    assert "SIGKILL" not in launcher


def test_no_live_t14_owner_audit_uses_exact_entrypoints(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    control = tmp_path / "control"
    control.mkdir()
    (control / "old-t14-receipt.json").write_text("{}\n", encoding="utf-8")
    process = proc / "123"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python\0unrelated_t14_report.py\0")
    stat = ["0"] * 22
    stat[21] = "17"
    (process / "stat").write_text(" ".join(stat), encoding="utf-8")
    receipt = tmp_path / "no-live.json"
    result = audit_no_live_t14_science_owner(
        control_root=control, receipt_path=receipt, proc_root=proc
    )
    assert result["status"] == "PASS"
    assert result["process_signal_sent"] is False
    assert result["control_evidence_files"] == [
        str(control / "old-t14-receipt.json")
    ]

    (process / "cmdline").write_bytes(
        b"python\0/worktree/scripts/run_tastemolnet_comrecgc_full.py\0"
    )
    with pytest.raises(T14RouteCFreshError, match="live T14"):
        audit_no_live_t14_science_owner(
            control_root=control,
            receipt_path=tmp_path / "blocked.json",
            proc_root=proc,
        )


def test_failed_watchdog_attempt_is_preserved_and_pointer_retired(
    tmp_path: Path,
) -> None:
    attempt = str(uuid4())
    old_owner = tmp_path / "owners" / f"route-c-{attempt}"
    old_output = tmp_path / "science" / f"route-c-{attempt}"
    current = tmp_path / "control" / "t14_route_c" / "current"
    retired = tmp_path / "control" / "t14_route_c" / "retired"
    proc = tmp_path / "proc"
    for path in (old_owner, old_output, current, proc):
        path.mkdir(parents=True)
    spec_path = old_owner / "T14_ROUTE_C_TASK_SPEC.json"
    old_spec = {
        "attempt_uuid": attempt,
        "owner_root": str(old_owner),
        "output_root": str(old_output),
        "spec_sha256": "a" * 64,
    }
    spec_path.write_text(json.dumps(old_spec), encoding="utf-8")
    (old_owner / "owner.json").write_text(
        json.dumps(
            {
                "owner_pid": 321,
                "owner_start_ticks": 654,
                "task_spec": str(spec_path),
            }
        ),
        encoding="utf-8",
    )
    (old_owner / "terminal.json").write_text(
        json.dumps(
            {
                "status": "FAILED",
                "error": "Route C resource watchdog stopped full at a safe request",
            }
        ),
        encoding="utf-8",
    )
    (old_output / "cohort_manifest.json").write_text(
        json.dumps({"cohort_jsonl_sha256": "b" * 64}), encoding="utf-8"
    )
    with (old_output / "route_c_step_states.jsonl").open("w", encoding="utf-8") as stream:
        for step in (1, 50, 100, 161):
            stream.write(json.dumps({"completed_step": step}) + "\n")
    (current / "owner.pid").write_text("321\n", encoding="utf-8")
    (current / "owner.start_ticks").write_text("654\n", encoding="utf-8")
    (current / "task_spec.path").write_text(f"{spec_path}\n", encoding="utf-8")

    receipt = retire_failed_route_c_current(
        current_root=current, retired_root=retired, proc_root=proc
    )
    assert receipt["terminal_state"] == "TERMINAL_FAILED_RESOURCE_WATCHDOG"
    assert receipt["retirement_state"] == "SUPERSEDED_BY_FRESH_RETRY"
    assert receipt["observed_uncommitted_step"] == 161
    assert receipt["reuse_partial_step161"] is False
    assert receipt["old_root_deleted"] is False
    assert old_output.is_dir()
    assert not current.exists()
    assert Path(receipt["retired_pointer_root"]).is_dir()


def test_failed_watchdog_preserves_real_child_when_master_output_was_unused(
    tmp_path: Path,
) -> None:
    attempt = str(uuid4())
    child_attempt = str(uuid4())
    old_owner = tmp_path / "owners" / f"route-c-{attempt}"
    old_master_output = tmp_path / "science" / f"route-c-{attempt}"
    child_owner = old_owner / "canaries" / "reference_500" / child_attempt
    child_output = child_owner / f"science-{child_attempt}"
    current = tmp_path / "control" / "t14_route_c" / "current"
    retired = tmp_path / "control" / "t14_route_c" / "retired"
    proc = tmp_path / "proc"
    for path in (old_owner, child_output, current, proc):
        path.mkdir(parents=True)
    master_spec_path = old_owner / "T14_ROUTE_C_TASK_SPEC.json"
    master_spec_path.write_text(
        json.dumps(
            {
                "attempt_uuid": attempt,
                "owner_root": str(old_owner),
                "output_root": str(old_master_output),
                "spec_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    child_spec_path = child_owner / "route_c_spec.json"
    child_spec = {"output_root": str(child_output)}
    child_spec["spec_sha256"] = stable_sha256(child_spec)
    child_spec_path.write_text(
        json.dumps(child_spec), encoding="utf-8"
    )
    (old_owner / "owner_plan.json").write_text(
        json.dumps(
            {
                "children": {
                    "REFERENCE_500": {
                        "output_root": str(child_output),
                        "spec_path": str(child_spec_path),
                        "spec_sha256": child_spec["spec_sha256"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (old_owner / "owner.json").write_text(
        json.dumps(
            {
                "owner_pid": 321,
                "owner_start_ticks": 654,
                "task_spec": str(master_spec_path),
            }
        ),
        encoding="utf-8",
    )
    (old_owner / "terminal.json").write_text(
        json.dumps(
            {
                "status": "FAILED",
                "error": "Route C resource watchdog stopped reference at a safe request",
            }
        ),
        encoding="utf-8",
    )
    (child_output / "cohort_manifest.json").write_text(
        json.dumps({"cohort_jsonl_sha256": "b" * 64}), encoding="utf-8"
    )
    with (child_output / "route_c_step_states.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        stream.write(json.dumps({"completed_step": 161}) + "\n")
    (current / "owner.pid").write_text("321\n", encoding="utf-8")
    (current / "owner.start_ticks").write_text("654\n", encoding="utf-8")
    (current / "task_spec.path").write_text(
        f"{master_spec_path}\n", encoding="utf-8"
    )

    receipt = retire_failed_route_c_current(
        current_root=current, retired_root=retired, proc_root=proc
    )
    assert not old_master_output.exists()
    assert child_output.is_dir()
    assert receipt["preserved_science_root"] == str(child_output)
    assert receipt["preservation_source"] == "OWNER_PLAN_REFERENCE_500"
    assert receipt["declared_master_output_materialized"] is False


def test_fresh_retry_contract_has_strict_memory_and_checkpoint_policy(
    tmp_path: Path,
) -> None:
    old_attempt = str(uuid4())
    old_owner = tmp_path / f"old-owner-{old_attempt}"
    old_output = tmp_path / f"old-output-{old_attempt}"
    retired_pointer = tmp_path / f"retired-{old_attempt}"
    for path in (old_owner, old_output, retired_pointer):
        path.mkdir()
    terminal = old_owner / "terminal.json"
    terminal.write_text(
        json.dumps({"status": "FAILED", "error": "resource watchdog"}),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "tastemolnet_t14_route_c_failed_attempt_retirement_v1",
        "terminal_state": "TERMINAL_FAILED_RESOURCE_WATCHDOG",
        "retirement_state": "SUPERSEDED_BY_FRESH_RETRY",
        "retry_index": 1,
        "max_retries": 1,
        "reuse_partial_step161": False,
        "preserve_failed_attempt": True,
        "old_attempt_uuid": old_attempt,
        "old_owner_pid": 321,
        "old_owner_start_ticks": 654,
        "old_owner_root": str(old_owner),
        "old_output_root": str(old_output),
        "old_terminal_sha256": file_sha256(terminal),
        "observed_uncommitted_step": 161,
        "old_root_deleted": False,
        "process_signal_sent": False,
        "retired_pointer_root": str(retired_pointer),
        "retired_at": "2026-09-05T00:00:00+00:00",
    }
    receipt["receipt_sha256"] = stable_sha256(receipt)
    receipt_path = retired_pointer / "retirement_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    attempt = str(uuid4())
    counters = tmp_path / "new-cgroup"
    counters.mkdir()
    for name, value in (("limit", "1"), ("current", "0"), ("failcnt", "0")):
        (counters / name).write_text(value, encoding="utf-8")
    matrix_root = tmp_path / "matrix"
    matrix_root.mkdir()
    retry = {
        "schema_version": "tastemolnet_t14_route_c_fresh_retry_task_v1",
        "retry_index": 1,
        "max_retries": 1,
        "reuse_partial_step161": False,
        "preserve_failed_attempt": True,
        "fresh_uuid": attempt,
        "fresh_output_root": str(tmp_path / f"science-{attempt}"),
        "previous_attempt_uuid": old_attempt,
        "previous_output_root": str(old_output),
        "retirement_receipt": str(receipt_path),
        "retirement_receipt_sha256": file_sha256(receipt_path),
        "dataset_sha256": "1" * 64,
        "train_split_sha256": "2" * 64,
        "cohort_sha256": "3" * 64,
        "t3_gine_sha256": "4" * 64,
        "seed": 7,
        "config_sha256": "5" * 64,
        "candidate_capacity": 50_000,
        "m_configured_max": 20_000,
        "m_fallback_max": 25_000,
        "min_valid_unique": 10,
        "gpu_index": 2,
        "memory_policy": {
            "start_headroom_bytes": 384 * 1024**3,
            "runtime_reserve_bytes": 96 * 1024**3,
            "launch_samples_required": 3,
            "runtime_low_headroom_samples": 3,
            "sample_seconds": 30.0,
        },
        "checkpoint_policy": {
            "early_steps": [50, 100, 250, 500],
            "production_steps": [
                2_500,
                5_000,
                7_500,
                10_000,
                12_500,
                15_000,
                17_500,
                20_000,
            ],
            "fresh_process_reload_each_checkpoint": True,
            "route_c_500_promoted_to_full_without_replay": True,
        },
        "matrix_authority_root": str(matrix_root),
        "matrix_authority_state": str(matrix_root / "state.json"),
        "matrix_authority_lock": str(matrix_root / "publish.lock"),
    }
    spec = build_spec(
        attempt_uuid=attempt,
        execution_commit="1" * 40,
        python=Path(sys.executable).resolve(),
        science_wrapper=REPO_ROOT / "scripts/autodl/run_tastemolnet_t14_comrecgc_full.sh",
        owner_entrypoint=REPO_ROOT / "scripts/autodl/run_t14_route_c_owner.py",
        output_root=tmp_path / f"science-{attempt}",
        owner_root=tmp_path / f"owner-{attempt}",
        cgroup_limit_path=counters / "limit",
        cgroup_current_path=counters / "current",
        cgroup_failcnt_path=counters / "failcnt",
        forbidden_legacy_root=tmp_path / "legacy-12500",
        science_environment={
            "RUN_TASTEMOLNET": "1",
            "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
            "TASTE_PAPER_RESULTS_ALLOWED": "1",
            "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
            "RUN_GNN_ABLATION": "0",
            "RUN_LLM_ABLATION": "0",
        },
        max_process_rss_bytes=64 * 1024**3,
        launch_headroom_bytes=384 * 1024**3,
        runtime_headroom_bytes=96 * 1024**3,
        sample_seconds=30,
        launch_samples_required=3,
        runtime_low_headroom_samples=3,
        fresh_retry=retry,
    )
    assert spec["fresh_retry"]["reuse_partial_step161"] is False
    assert spec["memory"]["launch_headroom_bytes"] == 384 * 1024**3
    assert spec["memory"]["runtime_headroom_bytes"] == 96 * 1024**3
    assert checkpoint_targets(completed_step=0, stop_step=500, route_c=True) == (
        50,
        100,
        250,
        500,
    )


def _engineering_step50_failure(
    tmp_path: Path,
) -> tuple[Path, Path, Path, str]:
    """Build the sealed shape of the deployed retry-1 REFERENCE_500 failure."""

    original_attempt = str(uuid4())
    original_owner = tmp_path / "original-owner"
    original_output = tmp_path / "original-output"
    original_pointer = tmp_path / "original-retired"
    for path in (original_owner, original_output, original_pointer):
        path.mkdir()
    original_terminal = original_owner / "terminal.json"
    original_terminal.write_text(
        json.dumps({"status": "FAILED", "error": "resource watchdog"}),
        encoding="utf-8",
    )
    first_retirement = {
        "schema_version": route_c.FRESH_RETRY_RECEIPT_SCHEMA,
        "terminal_state": "TERMINAL_FAILED_RESOURCE_WATCHDOG",
        "retirement_state": "SUPERSEDED_BY_FRESH_RETRY",
        "retry_index": 1,
        "max_retries": 1,
        "reuse_partial_step161": False,
        "preserve_failed_attempt": True,
        "old_attempt_uuid": original_attempt,
        "old_owner_pid": 101,
        "old_owner_start_ticks": 102,
        "old_owner_root": str(original_owner),
        "old_output_root": str(original_output),
        "old_terminal_sha256": file_sha256(original_terminal),
        "observed_uncommitted_step": 161,
        "old_root_deleted": False,
        "process_signal_sent": False,
        "retired_pointer_root": str(original_pointer),
        "retired_at": "2026-09-05T00:00:00+00:00",
    }
    first_retirement["receipt_sha256"] = stable_sha256(first_retirement)
    first_receipt_path = original_pointer / "retirement_receipt.json"
    first_receipt_path.write_text(json.dumps(first_retirement), encoding="utf-8")

    failed_attempt = str(uuid4())
    failed_owner = tmp_path / "owners" / f"route-c-{failed_attempt}"
    failed_master = tmp_path / "science" / f"route-c-{failed_attempt}"
    child_attempt = str(uuid4())
    child_owner = failed_owner / "canaries" / "reference_500" / child_attempt
    child_science = child_owner / f"science-{child_attempt}"
    current = tmp_path / "control" / "t14_route_c" / "current"
    proc = tmp_path / "proc"
    for path in (failed_owner, child_science, current, proc):
        path.mkdir(parents=True)
    template_root = tmp_path / "template"
    template_root.mkdir()
    template_spec, _template_path = _route_spec(template_root)
    failed_spec_path = failed_owner / "T14_ROUTE_C_FRESH_RETRY_TASK_SPEC.json"
    failed_spec = {
        "attempt_uuid": failed_attempt,
        "execution_commit": route_c.ENGINEERING_FAILED_EXECUTION_COMMIT,
        "owner_root": str(failed_owner),
        "output_root": str(failed_master),
        "storage_mode": "lowmemory",
        "m_configured_max": 20_000,
        "m_fallback_max": 25_000,
        "science_environment": template_spec["science_environment"],
        "route_c_state": template_spec["route_c_state"],
        "fresh_retry": {
            "retry_index": 1,
            "max_retries": 1,
            "retirement_receipt": str(first_receipt_path),
            "retirement_receipt_sha256": file_sha256(first_receipt_path),
            "dataset_sha256": "1" * 64,
            "train_split_sha256": "2" * 64,
            "cohort_sha256": "3" * 64,
            "t3_gine_sha256": "4" * 64,
            "config_sha256": "5" * 64,
            "seed": 7,
            "candidate_capacity": 50_000,
            "m_configured_max": 20_000,
            "m_fallback_max": 25_000,
            "min_valid_unique": 10,
        },
    }
    failed_spec_path.write_text(json.dumps(failed_spec), encoding="utf-8")
    (failed_owner / "owner.json").write_text(
        json.dumps(
            {
                "owner_pid": 321,
                "owner_start_ticks": 654,
                "task_spec": str(failed_spec_path),
            }
        ),
        encoding="utf-8",
    )
    (failed_owner / "terminal.json").write_text(
        json.dumps(
            {
                "status": "FAILED",
                "error": "Route C science phase failed: reference-500, exit=1",
            }
        ),
        encoding="utf-8",
    )
    child_spec_path = child_owner / "route_c_spec.json"
    child_spec = {
        "output_root": str(child_science),
        "canary_role": "REFERENCE_500",
        "storage_mode": "reference",
    }
    child_spec["spec_sha256"] = stable_sha256(child_spec)
    child_spec_path.write_text(json.dumps(child_spec), encoding="utf-8")
    (failed_owner / "owner_plan.json").write_text(
        json.dumps(
            {
                "children": {
                    "REFERENCE_500": {
                        "output_root": str(child_science),
                        "spec_path": str(child_spec_path),
                        "spec_sha256": child_spec["spec_sha256"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cohort_jsonl = child_science / "cohort.jsonl"
    cohort_jsonl.write_text(json.dumps({"parent_id": "p0"}) + "\n", encoding="utf-8")
    (child_science / "cohort_manifest.json").write_text(
        json.dumps(
            {
                "cohort_jsonl_sha256": file_sha256(cohort_jsonl),
                "train_csv_sha256": "1" * 64,
                "checkpoint_id": "4" * 64,
            }
        ),
        encoding="utf-8",
    )
    (child_science / "route_c_step_states.jsonl").write_text(
        json.dumps({"completed_step": 50}) + "\n", encoding="utf-8"
    )
    (child_science / "checkpoints").mkdir()
    (current / "owner.pid").write_text("321\n", encoding="utf-8")
    (current / "owner.start_ticks").write_text("654\n", encoding="utf-8")
    (current / "task_spec.path").write_text(
        f"{failed_spec_path}\n", encoding="utf-8"
    )
    return current, proc, failed_spec_path, failed_attempt


def _formal_cadence_sources() -> dict[str, Path]:
    science = REPO_ROOT / "src/baselines/tastemolnet_comrecgc_full.py"
    return {
        "science_step": science,
        "checkpoint": science,
        "progress": science,
        "watchdog": REPO_ROOT / "scripts/autodl/run_t14_route_c_owner.py",
        "convergence_audit": REPO_ROOT
        / "scripts/autodl/run_t14_route_c_owner.py",
        "publisher": REPO_ROOT
        / "src/baselines/tastemolnet_t14_route_c_continuation.py",
    }


def test_second_engineering_retry_retires_step50_without_checkpoint_reuse(
    tmp_path: Path,
) -> None:
    current, proc, failed_spec_path, failed_attempt = _engineering_step50_failure(
        tmp_path
    )
    authorization_path = tmp_path / "authorizations" / "retry-2.json"
    corrected_commit = "2" * 40
    authorization = route_c.write_engineering_retry_authorization_receipt(
        path=authorization_path,
        current_root=current,
        corrected_execution_commit=corrected_commit,
    )
    receipt = retire_failed_route_c_current(
        current_root=current,
        retired_root=tmp_path / "control" / "t14_route_c" / "retired",
        proc_root=proc,
        retry_index=2,
        authorization_receipt=authorization_path,
    )

    assert authorization["retry_index"] == 2
    assert receipt["old_attempt_uuid"] == failed_attempt
    assert receipt["observed_uncommitted_step"] == 50
    assert receipt["committed_checkpoint_present"] is False
    assert receipt["reuse_failed_checkpoint"] is False
    assert receipt["reuse_partial_step161"] is False
    assert Path(receipt["preserved_science_root"]).is_dir()
    assert not current.exists()

    new_attempt = str(uuid4())
    new_output = tmp_path / "science" / f"route-c-{new_attempt}"
    new_owner = tmp_path / "owners" / f"route-c-{new_attempt}"
    scientific_diff_path = new_owner / "T14_ROUTE_C_SCIENTIFIC_CONFIG_DIFF.json"
    reference_spec = json.loads(failed_spec_path.read_text(encoding="utf-8"))
    route_c.write_scientific_config_diff_receipt(
        path=scientific_diff_path,
        reference_task_spec=failed_spec_path,
        corrected_scientific_config=route_c._retry_scientific_config(reference_spec),
        authorization_receipt=authorization_path,
    )
    cadence_path = new_owner / "T14_ROUTE_C_FORMAL_CADENCE_CONTRACT.json"
    route_c.write_formal_cadence_contract(
        path=cadence_path,
        attempt_uuid=new_attempt,
        execution_commit=corrected_commit,
        cadence_sources=_formal_cadence_sources(),
        authorization_receipt=authorization_path,
        scientific_config_diff_receipt=scientific_diff_path,
    )
    retirement_path = (
        Path(receipt["retired_pointer_root"]) / "retirement_receipt.json"
    )
    counters = tmp_path / "new-cgroup"
    counters.mkdir()
    for name, value in (("limit", "1"), ("current", "0"), ("failcnt", "0")):
        (counters / name).write_text(value, encoding="utf-8")
    matrix_root = tmp_path / "matrix"
    matrix_root.mkdir()
    retry = {
        "schema_version": route_c.ENGINEERING_RETRY_CONTRACT_SCHEMA,
        "retry_index": 2,
        "max_retries": 2,
        "max_attempts": 2,
        "reuse_partial_step161": False,
        "reuse_failed_checkpoint": False,
        "preserve_failed_attempt": True,
        "fresh_uuid": new_attempt,
        "fresh_output_root": str(new_output),
        "previous_attempt_uuid": failed_attempt,
        "previous_output_root": receipt["old_output_root"],
        "retirement_receipt": str(retirement_path),
        "retirement_receipt_sha256": file_sha256(retirement_path),
        "authorization_receipt": str(authorization_path),
        "authorization_receipt_sha256": file_sha256(authorization_path),
        "formal_cadence_contract": str(cadence_path),
        "formal_cadence_contract_sha256": file_sha256(cadence_path),
        "scientific_config_diff_receipt": str(scientific_diff_path),
        "scientific_config_diff_receipt_sha256": file_sha256(
            scientific_diff_path
        ),
        "previous_actual_cohort_jsonl": authorization[
            "previous_actual_cohort_jsonl"
        ],
        "previous_actual_cohort_jsonl_sha256": authorization[
            "previous_actual_cohort_jsonl_sha256"
        ],
        "dataset_sha256": "1" * 64,
        "train_split_sha256": "2" * 64,
        "cohort_sha256": "3" * 64,
        "t3_gine_sha256": "4" * 64,
        "seed": 7,
        "config_sha256": "5" * 64,
        "candidate_capacity": 50_000,
        "m_configured_max": 20_000,
        "m_fallback_max": 25_000,
        "min_valid_unique": 10,
        "gpu_index": 2,
        "memory_policy": {
            "start_headroom_bytes": 384 * 1024**3,
            "runtime_reserve_bytes": 96 * 1024**3,
            "launch_samples_required": 3,
            "runtime_low_headroom_samples": 3,
            "sample_seconds": 30.0,
        },
        "checkpoint_policy": {
            "early_steps": [50, 100, 250, 500],
            "production_steps": [
                2_500,
                5_000,
                7_500,
                10_000,
                12_500,
                15_000,
                17_500,
                20_000,
            ],
            "fresh_process_reload_each_checkpoint": True,
            "route_c_500_promoted_to_full_without_replay": True,
            "fallback_extension_steps": [22_500, 25_000],
        },
        "matrix_authority_root": str(matrix_root),
        "matrix_authority_state": str(matrix_root / "state.json"),
        "matrix_authority_lock": str(matrix_root / "publish.lock"),
    }
    spec = build_spec(
        attempt_uuid=new_attempt,
        execution_commit=corrected_commit,
        python=Path(sys.executable).resolve(),
        science_wrapper=REPO_ROOT
        / "scripts/autodl/run_tastemolnet_t14_comrecgc_full.sh",
        owner_entrypoint=REPO_ROOT / "scripts/autodl/run_t14_route_c_owner.py",
        output_root=new_output,
        owner_root=new_owner,
        cgroup_limit_path=counters / "limit",
        cgroup_current_path=counters / "current",
        cgroup_failcnt_path=counters / "failcnt",
        forbidden_legacy_root=tmp_path / "legacy-step161",
        science_environment={
            "RUN_TASTEMOLNET": "1",
            "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
            "TASTE_PAPER_RESULTS_ALLOWED": "1",
            "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
            "RUN_GNN_ABLATION": "0",
            "RUN_LLM_ABLATION": "0",
        },
        max_process_rss_bytes=64 * 1024**3,
        launch_headroom_bytes=384 * 1024**3,
        runtime_headroom_bytes=96 * 1024**3,
        sample_seconds=30,
        launch_samples_required=3,
        runtime_low_headroom_samples=3,
        fresh_retry=retry,
    )
    assert spec["fresh_retry"]["retry_index"] == 2
    assert spec["fresh_retry"]["max_attempts"] == 2
    assert spec["fresh_retry"]["reuse_failed_checkpoint"] is False


def test_second_engineering_retry_fails_closed_if_checkpoint_exists(
    tmp_path: Path,
) -> None:
    current, _proc, _failed_spec, _failed_attempt = _engineering_step50_failure(
        tmp_path
    )
    task_spec = Path((current / "task_spec.path").read_text(encoding="utf-8").strip())
    owner_root = task_spec.parent
    plan = json.loads((owner_root / "owner_plan.json").read_text(encoding="utf-8"))
    child_root = Path(plan["children"]["REFERENCE_500"]["output_root"])
    (child_root / "checkpoints" / "step-000000000050.pt").write_bytes(b"stale")

    with pytest.raises(T14RouteCFreshError, match="step-50 engineering failure"):
        route_c.write_engineering_retry_authorization_receipt(
            path=tmp_path / "retry-2.json",
            current_root=current,
            corrected_execution_commit="2" * 40,
        )


def test_second_engineering_retry_rejects_live_reference_child_writer(
    tmp_path: Path,
) -> None:
    current, proc, task_spec, _failed_attempt = _engineering_step50_failure(tmp_path)
    owner_root = task_spec.parent
    plan = json.loads((owner_root / "owner_plan.json").read_text(encoding="utf-8"))
    child_spec = plan["children"]["REFERENCE_500"]["spec_path"]
    process = proc / "444"
    process.mkdir()
    (process / "cmdline").write_bytes(
        b"python\0/worktree/scripts/autodl/run_tastemolnet_comrecgc_full.py\0"
        + str(child_spec).encode("utf-8")
        + b"\0"
    )
    authorization_path = tmp_path / "authorizations" / "retry-2.json"
    route_c.write_engineering_retry_authorization_receipt(
        path=authorization_path,
        current_root=current,
        corrected_execution_commit="2" * 40,
    )

    with pytest.raises(T14RouteCFreshError, match="exact writer"):
        retire_failed_route_c_current(
            current_root=current,
            retired_root=tmp_path / "control" / "t14_route_c" / "retired",
            proc_root=proc,
            retry_index=2,
            authorization_receipt=authorization_path,
        )


def test_second_retry_formal_cadence_binds_reference_and_lowmemory_step50(
    tmp_path: Path,
) -> None:
    current, _proc, failed_spec, _failed_attempt = _engineering_step50_failure(
        tmp_path
    )
    authorization_path = tmp_path / "authorizations" / "retry-2.json"
    route_c.write_engineering_retry_authorization_receipt(
        path=authorization_path,
        current_root=current,
        corrected_execution_commit="2" * 40,
    )
    reference = json.loads(failed_spec.read_text(encoding="utf-8"))
    diff_path = tmp_path / "new-owner" / "scientific-diff.json"
    route_c.write_scientific_config_diff_receipt(
        path=diff_path,
        reference_task_spec=failed_spec,
        corrected_scientific_config=route_c._retry_scientific_config(reference),
        authorization_receipt=authorization_path,
    )
    contract_path = tmp_path / "new-owner" / "cadence.json"
    contract = route_c.write_formal_cadence_contract(
        path=contract_path,
        attempt_uuid=str(uuid4()),
        execution_commit="2" * 40,
        cadence_sources=_formal_cadence_sources(),
        authorization_receipt=authorization_path,
        scientific_config_diff_receipt=diff_path,
    )

    assert contract["route_c_storage_modes"] == ["reference", "lowmemory"]
    assert contract["cadences"]["checkpoint"]["early_steps"][0] == 50
    assert contract["cadences"]["checkpoint"]["fallback_extension_steps"] == [
        22_500,
        25_000,
    ]
    assert contract["cadences"]["science_step"]["interval_steps"] == 1
    assert contract["cadences"]["watchdog"]["sample_seconds"] == 30.0
    assert contract["cadences"]["convergence_audit"]["check_interval_steps"] == 2_500
    assert contract["cadences"]["publisher"]["poll_seconds"] == 60
    assert contract["parameters_validated_without_cursor_substitution"] is True
    assert contract["failed_checkpoint_reused"] is False
    assert checkpoint_targets(completed_step=0, stop_step=500, route_c=True)[0] == 50


def test_second_retry_scientific_config_drift_fails_closed(tmp_path: Path) -> None:
    current, _proc, failed_spec, _failed_attempt = _engineering_step50_failure(
        tmp_path
    )
    authorization_path = tmp_path / "authorizations" / "retry-2.json"
    route_c.write_engineering_retry_authorization_receipt(
        path=authorization_path,
        current_root=current,
        corrected_execution_commit="2" * 40,
    )
    reference = json.loads(failed_spec.read_text(encoding="utf-8"))
    corrected = route_c._retry_scientific_config(reference)
    corrected["retry_scientific_inputs"]["candidate_capacity"] = 49_999

    with pytest.raises(T14RouteCFreshError, match="drift is non-empty"):
        route_c.write_scientific_config_diff_receipt(
            path=tmp_path / "scientific-diff.json",
            reference_task_spec=failed_spec,
            corrected_scientific_config=corrected,
            authorization_receipt=authorization_path,
        )


def test_second_retry_launcher_is_bounded_and_never_reuses_failed_checkpoint() -> None:
    launcher = (REPO_ROOT / "scripts/autodl/launch_t14_route_c_once.sh").read_text(
        encoding="utf-8"
    )
    assert "ALLOW_T14_ENGINEERING_CORRECTED_SECOND_FRESH_RETRY" in launcher
    assert 'T14_ROUTE_C_FRESH_RETRY_MAX_ATTEMPTS:-}" == "2"' in launcher
    assert 'REUSE_FAILED_ROUTE_C_CHECKPOINT:-}" == "0"' in launcher
    assert "superseded-by-fresh-retry-${RETRY_INDEX}" in launcher
    assert "--formal-cadence-contract-out" in launcher
    assert "--scientific-config-diff-out" in launcher
    assert "--poll-seconds 60" in launcher
    assert "requires a clean immutable checkout" in launcher
