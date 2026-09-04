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
    recover_route_c_external_state,
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
    assert FIRST_CHECKPOINT_STEP == 250
    assert PROMOTABLE_CHECKPOINT_STEP == 500
    assert RELOAD_REPLAY_END_STEP == 510
    assert checkpoint_targets(completed_step=0, stop_step=510, route_c=True) == (
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
