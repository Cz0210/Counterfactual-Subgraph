from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from src.baselines import bace_globalgce_k20_extension as k20
from src.baselines import globalgce_bace_native_rules as native_rules
from src.oracles.gnn_oracle import REQUIRED_CHECKPOINT_FILES
from src.baselines.globalgce_bace_native_rules import GlobalGCENativeRule
from src.utils.process_identity_v2 import ProcessSnapshotV2, stable_json_sha256


CHECKPOINT_HASH = "a" * 64


def _rule_row(
    candidate_id: str,
    *,
    native_rule_index: int = 0,
    rhs_value: float = 1.0,
) -> dict[str, object]:
    lhs = [[0.0, 1.0], [0.0, 1.0]]
    rhs = [[0.0, rhs_value], [0.0, rhs_value]]
    rule_payload = {
        "rule_id": candidate_id,
        "native_rule_index": native_rule_index,
        "lhs_feature": lhs,
        "lhs_adjacency": [[0.0, 1.0], [1.0, 0.0]],
        "lhs_edge_attr": [[0.0, 1.0]],
        "rhs_feature": rhs,
        "rhs_adjacency": [[0.0, 1.0], [1.0, 0.0]],
        "rhs_edge_attr": [[0.0, 1.0]],
        "atom_symbols": ["C"],
        "bond_names": ["no_edge", "single"],
    }
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "rank": native_rule_index + 1,
        "native_rank": native_rule_index + 1,
        "action_kind": "lhs_rhs_graph_transformation_rule",
        "action_semantics": "native_lhs_to_rhs_attachment_aware_v1",
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "oracle_checkpoint_hash": CHECKPOINT_HASH,
        "source_split": "train",
        "rule": rule_payload,
    }
    rule = GlobalGCENativeRule.from_payload(row)
    row["rule_content_hash"] = rule.content_hash()
    row["selector_chemistry"] = rule.selector_chemistry()
    return row


def _snapshot(pid: int, command: tuple[str, ...]) -> ProcessSnapshotV2:
    return ProcessSnapshotV2(
        pid=pid,
        ppid=1,
        pid_start_ticks=pid * 10,
        boot_id="boot",
        executable_realpath="/usr/bin/python3",
        command=command,
        command_hash=stable_json_sha256(list(command)),
        cwd_realpath="/run",
        cgroup_path="0::/test",
    )


def _publish_with_signal_mask(**kwargs: object) -> dict[str, object]:
    previous = k20.signal.pthread_sigmask(k20.signal.SIG_BLOCK, k20.RELEASE_SIGNALS)
    try:
        return k20._publish_release_candidate(**kwargs)
    finally:
        k20.signal.pthread_sigmask(k20.signal.SIG_SETMASK, previous)


def test_round_plan_is_exact_cumulative_80_200_500() -> None:
    k20.validate_round_plan()
    assert [item.cumulative_seeds for item in k20.ROUND_PLAN] == [
        (7,),
        (7, 17),
        (7, 17, 27),
    ]
    assert [item.incremental_raw_budget for item in k20.ROUND_PLAN] == [80, 120, 300]
    assert sum(item.incremental_raw_budget for item in k20.ROUND_PLAN) == 500
    assert k20.FINAL_K == 20


def test_round_command_is_internal_fresh_train_only_exact_topk(tmp_path: Path) -> None:
    command = k20.build_round_command(
        python=tmp_path / "python",
        project_root=tmp_path / "project",
        config=tmp_path / "project/configs/hpc.yaml",
        source_manifest=tmp_path / "train/source_manifest.jsonl",
        native_train_csv=tmp_path / "train/native.csv",
        official_root=tmp_path / "official",
        gnn_checkpoint=tmp_path / "gine",
        output_root=tmp_path / "round-1",
        spec=k20.ROUND_PLAN[0],
    )
    assert command[1:3] == ["-I", "-B"]
    assert "raw-round" in command
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--top-k-native") + 1] == "80"
    assert command[command.index("--epochs") + 1] == "100"
    assert command[command.index("--min-freq") + 1] == "7"
    assert "--gspan-exact-top-k-pruning" in command
    assert "--no-resume" in command
    option_names = [value.lower() for value in command if value.startswith("--")]
    assert not any("calibration" in value or "test" in value for value in option_names)


def test_semantic_dedup_ignores_candidate_id_and_native_index() -> None:
    first = _rule_row("rule-a", native_rule_index=0)
    duplicate_id_only = _rule_row("rule-b", native_rule_index=0)
    duplicate_index_only = _rule_row("rule-c", native_rule_index=9)
    genuinely_distinct = _rule_row("rule-d", native_rule_index=1, rhs_value=0.5)
    unique, audit = k20.merge_unique_rules(
        [
            (k20.ROUND_PLAN[0], [first]),
            (
                k20.ROUND_PLAN[1],
                [duplicate_id_only, duplicate_index_only, genuinely_distinct],
            ),
        ],
        expected_checkpoint_hash=CHECKPOINT_HASH,
    )
    assert [row["candidate_id"] for row in unique] == ["rule-a", "rule-d"]
    assert [row["accepted_unique"] for row in audit] == [True, False, False, True]
    assert audit[1]["duplicate_of_candidate_id"] == "rule-a"
    assert audit[2]["duplicate_of_candidate_id"] == "rule-a"


def test_candidate_id_collision_with_different_content_is_rejected() -> None:
    first = _rule_row("rule-a", native_rule_index=0)
    collision = _rule_row("rule-a", native_rule_index=1, rhs_value=0.5)
    with pytest.raises(k20.BACEGlobalGCEK20Error, match="multiple"):
        k20.merge_unique_rules(
            [(k20.ROUND_PLAN[0], [first, collision])],
            expected_checkpoint_hash=CHECKPOINT_HASH,
        )


def test_gpu2_lock_is_one_fixed_global_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "runtime/locks/gpu-2.lock"
    monkeypatch.setattr(k20, "GPU_LOCK_PATH", canonical)
    with k20.HeldGpuLease.acquire(canonical) as lease:
        lease.verify()
        with pytest.raises(BlockingIOError):
            k20.HeldGpuLease.acquire(canonical)
        with pytest.raises(k20.BACEGlobalGCEK20Error, match="globally canonical"):
            k20.HeldGpuLease.acquire(tmp_path / "other/locks/gpu-2.lock")


def test_protected_roles_bind_live_gpu_uuid_pid_and_task_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = {
        0: _snapshot(
            101,
            (
                "python",
                "/run/gcfexplainer/run_bace_vrrw.py",
                "--dataset-dir",
                "/data/gcfexplainer",
                "--gnn-checkpoint",
                "/gine",
            ),
        ),
        3: _snapshot(
            303,
            (
                "python",
                "/run/comrecgc/run_generation.py",
                "--route",
                "project",
                "--dataset",
                "bace",
                "--mode",
                "full",
            ),
        ),
    }
    inventory = {
        0: {"uuid": "GPU-0", "memory_used_mib": 1},
        3: {"uuid": "GPU-3", "memory_used_mib": 1},
    }
    monkeypatch.setattr(k20, "_snapshot_matches", lambda _snapshot: None)
    monkeypatch.setattr(k20, "_gpu_inventory", lambda: inventory)
    monkeypatch.setattr(
        k20,
        "_gpu_compute_processes",
        lambda: {"GPU-0": {101}, "GPU-3": {303}},
    )
    k20._verify_protected_gpu_roles(protected, gpu_inventory=inventory)

    wrong = dict(protected)
    wrong[3] = _snapshot(
        303,
        (
            "python",
            "/run/gcfexplainer/run_bace_vrrw.py",
            "--dataset-dir",
            "/data/gcfexplainer",
            "--gnn-checkpoint",
            "/gine",
        ),
    )
    with pytest.raises(k20.BACEGlobalGCEK20Error, match="wrong task role"):
        k20._verify_protected_gpu_roles(wrong, gpu_inventory=inventory)


def test_shortfall_requires_exact_structured_receipt_and_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "round"
    (root / "native").mkdir(parents=True)
    (root / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "training_summary.json").write_text("{}\n", encoding="utf-8")
    (root / "native/native_rule_catalog.jsonl").write_text("", encoding="utf-8")
    science = {"contract": "fixed"}
    proof = {"proof": "fixed"}
    monkeypatch.setattr(k20, "_validate_raw_manifest_binding", lambda *a, **kw: {})
    monkeypatch.setattr(k20, "_validate_raw_catalog", lambda *a, **kw: ([], proof))
    receipt = {
        "schema_version": k20.RAW_ROUND_SCHEMA,
        "status": "EXPECTED_SHORTFALL",
        "reason": k20.EXPECTED_SHORTFALL,
        "exit_code": k20.RAW_SHORTFALL_EXIT_CODE,
        "seed": 7,
        "raw_budget": 80,
        "valid_native_rule_count": 0,
        "science_contract": science,
        "gspan_exact_top_k_proof": proof,
        "artifacts": {
            "run_manifest": k20.file_identity(root / "run_manifest.json"),
            "training_summary": k20.file_identity(root / "training_summary.json"),
            "native_rule_catalog": k20.file_identity(
                root / "native/native_rule_catalog.jsonl"
            ),
        },
        "completed_at": "fixed",
    }
    receipt["receipt_payload_sha256"] = k20.stable_sha256(receipt)
    (root / k20.RAW_SHORTFALL_RECEIPT).write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    (root / "RAW_SHORTFALL").write_text(
        k20.RAW_SHORTFALL_MARKER + "\n", encoding="utf-8"
    )
    rows, observed = k20.validate_shortfall_raw_round(
        root, science_contract=science, seed=7, raw_budget=80
    )
    assert rows == []
    assert observed == receipt

    tampered = dict(receipt)
    tampered["exit_code"] = 1
    (root / k20.RAW_SHORTFALL_RECEIPT).write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    with pytest.raises(k20.BACEGlobalGCEK20Error, match="receipt/hash"):
        k20.validate_shortfall_raw_round(
            root, science_contract=science, seed=7, raw_budget=80
        )


def test_unexpected_raw_runtime_error_is_never_converted_to_shortfall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "runtime/outputs/bace_globalgce_k20"
    root = namespace / "controller/rounds/round-1"
    namespace.mkdir(parents=True)
    source = tmp_path / "source.jsonl"
    native = tmp_path / "native.csv"
    official = tmp_path / "official"
    checkpoint = tmp_path / "checkpoint"
    source.write_text("\n", encoding="utf-8")
    native.write_text("\n", encoding="utf-8")
    official.mkdir()
    checkpoint.mkdir()
    monkeypatch.setattr(k20, "OUTPUT_NAMESPACE", namespace)
    monkeypatch.setattr(
        k20,
        "_capture_science_contract",
        lambda **kwargs: ({"fixed": True}, checkpoint),
    )

    def fail(**_kwargs: object) -> None:
        raise RuntimeError("different scientific failure")

    monkeypatch.setattr(k20, "build_bace_frozen_gine_rule_pool", fail)
    with pytest.raises(RuntimeError, match="different scientific failure"):
        k20.run_raw_round(
            source_manifest=source,
            native_train_csv=native,
            official_root=official,
            gnn_checkpoint=checkpoint,
            output_dir=root,
            expected_parent_count=360,
            seed=7,
            min_freq=7,
            epochs=100,
            top_k_native=80,
            device="cuda:0",
            resume=False,
            gspan_exact_top_k_pruning=True,
        )


def test_real_rule_pool_shortfall_is_converted_to_hash_closed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "runtime/outputs/bace_globalgce_k20"
    root = namespace / "controller/rounds/round-1"
    namespace.mkdir(parents=True)
    source = tmp_path / "source.jsonl"
    native = tmp_path / "native.csv"
    official = tmp_path / "official"
    checkpoint = tmp_path / "checkpoint"
    source.write_text("\n", encoding="utf-8")
    native.write_text("\n", encoding="utf-8")
    official.mkdir()
    checkpoint.mkdir()
    monkeypatch.setattr(k20, "OUTPUT_NAMESPACE", namespace)
    monkeypatch.setattr(
        k20,
        "_capture_science_contract",
        lambda **kwargs: ({"fixed": True}, checkpoint),
    )
    monkeypatch.setattr(k20, "_require_science_contract_unchanged", lambda *a, **k: None)
    monkeypatch.setattr(k20, "_validate_raw_manifest_binding", lambda *a, **k: None)
    monkeypatch.setattr(k20, "_validate_raw_catalog", lambda *a, **k: ([], {"exact": True}))
    monkeypatch.setattr(
        k20,
        "file_identity",
        lambda path: {"path": str(path), "sha256": "a" * 64, "bytes": 1},
    )

    def shortfall(**_kwargs: object) -> None:
        root.mkdir(parents=True)
        raise RuntimeError(k20.EXPECTED_SHORTFALL)

    monkeypatch.setattr(k20, "build_bace_frozen_gine_rule_pool", shortfall)
    receipt = k20.run_raw_round(
        source_manifest=source,
        native_train_csv=native,
        official_root=official,
        gnn_checkpoint=checkpoint,
        output_dir=root,
        expected_parent_count=360,
        seed=7,
        min_freq=7,
        epochs=100,
        top_k_native=80,
        device="cuda:0",
        resume=False,
        gspan_exact_top_k_pruning=True,
    )

    assert receipt["status"] == "EXPECTED_SHORTFALL"
    assert receipt["reason"] == k20.EXPECTED_SHORTFALL
    assert receipt["receipt_payload_sha256"] == k20.stable_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_payload_sha256"}
    )
    assert json.loads((root / k20.RAW_SHORTFALL_RECEIPT).read_text(encoding="utf-8")) == receipt
    assert (root / "RAW_SHORTFALL").read_text(encoding="utf-8").strip() == k20.RAW_SHORTFALL_MARKER


def test_release_marker_is_after_two_live_revalidations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "release"
    root.mkdir()
    (root / "controller_receipt.json").write_text("{}\n", encoding="utf-8")
    selected = [
        {
            "candidate_id": f"rule-{index}",
            "rank": index,
            "extension_rank": index,
            "semantic_rule_content_hash": f"semantic-{index}",
        }
        for index in range(1, 21)
    ]

    def fake_validate(row: object, **_kwargs: object) -> tuple[dict, str, str]:
        payload = dict(row)  # type: ignore[arg-type]
        return payload, f"content-{payload['rank']}", payload["semantic_rule_content_hash"]

    monkeypatch.setattr(k20, "validate_catalog_row", fake_validate)
    calls: list[str] = []

    def revalidate() -> None:
        calls.append("live")
        if len(calls) == 1:
            assert not (root / "verification.json").exists()
        else:
            assert (root / "release_gate.json").is_file()
            assert not (root / "PASS").exists()

    controller = _snapshot(999, ("python", "controller"))
    original_marker = k20.atomic_marker

    def guarded_marker(path: Path, marker: str) -> None:
        current_mask = k20.signal.pthread_sigmask(k20.signal.SIG_BLOCK, set())
        assert k20.RELEASE_SIGNALS.issubset(current_mask)
        original_marker(path, marker)

    monkeypatch.setattr(k20, "atomic_marker", guarded_marker)
    result = _publish_with_signal_mask(
        root=root,
        controller_id="controller",
        controller=controller,
        selected=selected,
        audit=[{"accepted_unique": True} for _ in selected],
        provenance={"oracle_checkpoint_hash": CHECKPOINT_HASH},
        execution_contract={"fixed": True},
        round_receipts=[
            {
                "cumulative_raw_budget": 80,
                "cumulative_seeds": [7],
            }
        ],
        physical_gpu_uuid="GPU-2",
        sequence=1,
        revalidate=revalidate,
        stop_requested=threading.Event(),
    )
    assert calls == ["live", "live"]
    assert (root / "PASS").read_text(encoding="utf-8") == k20.PASS_MARKER + "\n"
    assert not (root / "_RUN_COMPLETE.json").exists()
    assert result["status"] == "SEALED_CANDIDATE"


def test_failed_final_revalidation_leaves_no_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "release"
    root.mkdir()
    (root / "controller_receipt.json").write_text("{}\n", encoding="utf-8")
    selected = [
        {
            "candidate_id": f"rule-{index}",
            "rank": index,
            "extension_rank": index,
            "semantic_rule_content_hash": f"semantic-{index}",
        }
        for index in range(1, 21)
    ]
    monkeypatch.setattr(
        k20,
        "validate_catalog_row",
        lambda row, **kwargs: (
            dict(row),
            "content",
            dict(row)["semantic_rule_content_hash"],
        ),
    )
    count = 0

    def revalidate() -> None:
        nonlocal count
        count += 1
        if count == 2:
            raise k20.BACEGlobalGCEK20Error("checkpoint drift")

    with pytest.raises(k20.BACEGlobalGCEK20Error, match="checkpoint drift"):
        _publish_with_signal_mask(
            root=root,
            controller_id="controller",
            controller=_snapshot(999, ("python", "controller")),
            selected=selected,
            audit=[{"accepted_unique": True} for _ in selected],
            provenance={"oracle_checkpoint_hash": CHECKPOINT_HASH},
            execution_contract={"fixed": True},
            round_receipts=[
                {"cumulative_raw_budget": 80, "cumulative_seeds": [7]}
            ],
            physical_gpu_uuid="GPU-2",
            sequence=1,
            revalidate=revalidate,
            stop_requested=threading.Event(),
        )
    assert not (root / "PASS").exists()
    assert json.loads((root / "summary.json").read_text())["status"] == "SEALED_CANDIDATE"


def test_pending_stop_at_final_commit_leaves_no_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "release"
    root.mkdir()
    (root / "controller_receipt.json").write_text("{}\n", encoding="utf-8")
    selected = [
        {
            "candidate_id": f"rule-{index}",
            "rank": index,
            "extension_rank": index,
            "semantic_rule_content_hash": f"semantic-{index}",
        }
        for index in range(1, 21)
    ]
    monkeypatch.setattr(
        k20,
        "validate_catalog_row",
        lambda row, **kwargs: (
            dict(row),
            "content",
            dict(row)["semantic_rule_content_hash"],
        ),
    )
    stop_requested = threading.Event()
    calls = 0

    def revalidate() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            stop_requested.set()

    with pytest.raises(k20.BACEGlobalGCEK20Error, match="stop requested"):
        _publish_with_signal_mask(
            root=root,
            controller_id="controller",
            controller=_snapshot(999, ("python", "controller")),
            selected=selected,
            audit=[{"accepted_unique": True} for _ in selected],
            provenance={"oracle_checkpoint_hash": CHECKPOINT_HASH},
            execution_contract={"fixed": True},
            round_receipts=[
                {"cumulative_raw_budget": 80, "cumulative_seeds": [7]}
            ],
            physical_gpu_uuid="GPU-2",
            sequence=1,
            revalidate=revalidate,
            stop_requested=stop_requested,
        )
    assert not (root / "PASS").exists()


def test_controller_signal_mask_rejects_an_existing_thread() -> None:
    release = threading.Event()
    ready = threading.Event()

    def wait() -> None:
        ready.set()
        release.wait()

    worker = threading.Thread(target=wait)
    worker.start()
    ready.wait()
    try:
        with pytest.raises(k20.BACEGlobalGCEK20Error, match="before.*thread|one OS thread"):
            k20._install_deferred_signal_mask()
    finally:
        release.set()
        worker.join()


def test_process_wide_signal_mask_survives_real_preimport_bootstrap() -> None:
    if not Path("/proc/self/task").is_dir():
        pytest.skip("Linux procfs is required for the process-wide mask proof")
    runner = Path("scripts/autodl/run_bace_globalgce_k20_extension.py").resolve()
    probe = f"""
import importlib.util
import sys

runner = {str(runner)!r}
sys.argv = [runner, "controller"]
spec = importlib.util.spec_from_file_location("k20_preimport_probe", runner)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from src.baselines import bace_globalgce_k20_extension as core
core._require_process_wide_deferred_signal_mask()
core._adopt_preinstalled_signal_mask(module._PREIMPORT_PREVIOUS_SIGNAL_MASK)
print("PROCESS_WIDE_PREIMPORT_MASK_PASS")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe],
        cwd=runner.parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "PROCESS_WIDE_PREIMPORT_MASK_PASS"


def test_wait_rejects_a_foreign_gpu2_compute_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Child:
        def __init__(self) -> None:
            self.polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            return None if self.polls == 1 else 0

        def wait(self) -> int:
            return 0

    class Lease:
        def verify(self) -> None:
            return None

    monkeypatch.setattr(k20, "_snapshot_matches", lambda snapshot: None)
    monkeypatch.setattr(k20, "_verify_protected_gpu_roles", lambda *a, **k: None)
    monkeypatch.setattr(
        k20, "_gpu_compute_processes", lambda: {"GPU-2": {101, 666}}
    )
    previous = k20.signal.pthread_sigmask(k20.signal.SIG_BLOCK, k20.RELEASE_SIGNALS)
    try:
        with pytest.raises(k20.BACEGlobalGCEK20Error, match="foreign.*666"):
            k20._wait_naturally(
                Child(),
                root=tmp_path,
                sequence=1,
                controller_id="controller",
                controller=_snapshot(999, ("python", "controller")),
                child_snapshot=_snapshot(101, ("python", "raw-round")),
                spec=k20.ROUND_PLAN[0],
                protected={},
                gpu_inventory={2: {"uuid": "GPU-2"}},
                lease=Lease(),
                stop_requested=threading.Event(),
                heartbeat_interval_seconds=60,
            )
    finally:
        k20.signal.pthread_sigmask(k20.signal.SIG_SETMASK, previous)


def test_science_contract_hash_closes_every_required_gine_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "gine"
    checkpoint.mkdir()
    model = checkpoint / "model.pt"
    model.write_bytes(b"frozen-model")
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    payloads: dict[str, bytes] = {
        "config.yaml": b"seed: 7\n",
        "model_card.json": (json.dumps(
            {
                "dataset": "bace",
                "backbone": "gine",
                "oracle_backend": "gnn",
                "rf_oracle_used": False,
                "num_classes": 2,
                "source_label": 1,
                "checkpoint_id": model_hash,
            }
        ) + "\n").encode(),
        "feature_schema.json": b"{}\n",
        "label_map.json": b"{}\n",
        "split_manifest.json": b'{"files": {}}\n',
        "training_metrics.json": b"{}\n",
        "validation_predictions.csv": b"molecule_id\n",
        "test_evaluation_status.json": (json.dumps(
            {
                "status": "NOT_EVALUATED",
                "test_loaded": False,
                "reason": "frozen before held-out test",
                "path": "/data/bace-test.csv",
                "sha256": "f" * 64,
            }
        ) + "\n").encode(),
        "temperature_scaling.json": b"{}\n",
        "environment.json": b"{}\n",
        "git_state.json": b"{}\n",
    }
    for name, payload in payloads.items():
        (checkpoint / name).write_bytes(payload)
    inventory = []
    for name in REQUIRED_CHECKPOINT_FILES:
        if name == "sha256sums.txt":
            continue
        path = checkpoint / name
        inventory.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
    (checkpoint / "sha256sums.txt").write_text(
        "\n".join(inventory) + "\n", encoding="utf-8"
    )
    card = json.loads((checkpoint / "model_card.json").read_text(encoding="utf-8"))
    train_contract = SimpleNamespace(
        audit={"calibration_loaded": False, "test_loaded": False},
        source_parents=[SimpleNamespace(parent_id=f"p-{index}") for index in range(360)],
    )
    monkeypatch.setattr(k20, "audit_bace_globalgce_train_contract", lambda **k: train_contract)
    monkeypatch.setattr(
        k20,
        "validate_official_globalgce_root",
        lambda root: {"clean_checkout": True, "runtime_source_authority": {"x.py": {}}},
    )
    monkeypatch.setattr(
        k20, "validate_bace_frozen_gine", lambda root: (checkpoint, card, object())
    )
    monkeypatch.setattr(
        k20,
        "oracle_provenance",
        lambda observed, root: {"oracle_checkpoint_hash": model_hash},
    )
    contract, _ = k20._capture_science_contract(
        source_manifest=tmp_path / "source.jsonl",
        native_train_csv=tmp_path / "native.csv",
        official_root=tmp_path / "official",
        gnn_checkpoint=checkpoint,
    )
    assert set(contract["checkpoint_bundle"]["files"]) == set(REQUIRED_CHECKPOINT_FILES)

    (checkpoint / "training_metrics.json").write_text('{"tampered": true}\n')
    with pytest.raises(ValueError, match="SHA mismatch: training_metrics.json"):
        k20._capture_science_contract(
            source_manifest=tmp_path / "source.jsonl",
            native_train_csv=tmp_path / "native.csv",
            official_root=tmp_path / "official",
            gnn_checkpoint=checkpoint,
        )


def test_official_runtime_import_closure_rejects_dirty_gtgnn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "official"
    required = set(native_rules.OFFICIAL_RUNTIME_FILES) | set(
        native_rules.OFFICIAL_SOURCE_SHA256
    )
    for relative in sorted(required):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=K20 Test",
            "-c",
            "user.email=k20@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    pinned = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in native_rules.OFFICIAL_SOURCE_SHA256
    }
    monkeypatch.setattr(native_rules, "OFFICIAL_SOURCE_SHA256", pinned)
    audit = native_rules.validate_official_globalgce_root(root, expected_commit=commit)
    assert "models/GTGNN.py" in audit["runtime_source_authority"]
    assert "models/gSpan/gSpan.py" in audit["runtime_source_authority"]
    assert audit["clean_checkout"] is True

    (root / "src/models/GTGNN.py").write_text("# dirty\n", encoding="utf-8")
    with pytest.raises(native_rules.GlobalGCENativeRuleError, match="not clean"):
        native_rules.validate_official_globalgce_root(root, expected_commit=commit)


def test_cli_has_separate_controller_and_raw_round_subcommands() -> None:
    runner = Path("scripts/autodl/run_bace_globalgce_k20_extension.py").resolve()
    spec = importlib.util.spec_from_file_location("k20_runner_test", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--config",
            "/repo/configs/hpc.yaml",
            "--set",
            "inference.fallback_to_heuristic=false",
            "controller",
            "--controller-id",
            "id",
            "--output-root",
            "/runtime/run",
            "--source-manifest",
            "/data/source.jsonl",
            "--native-train-csv",
            "/data/native.csv",
            "--official-root",
            "/official",
            "--gnn-checkpoint",
            "/checkpoint",
            "--protected-gpu0-process",
            "1:2",
            "--protected-gpu3-process",
            "3:4",
        ]
    )
    assert args.command == "controller"
    assert args.protected_gpu0_process == "1:2"
    assert args.protected_gpu3_process == "3:4"
    source = runner.read_text(encoding="utf-8")
    assert source.index("_install_mask_before_science_imports(sys.argv[1:])") < source.index(
        "from src.baselines.bace_globalgce_k20_extension import"
    )
    assert "preinstalled_signal_mask=_PREIMPORT_PREVIOUS_SIGNAL_MASK" in source
    assert source.index("unblock_deferred_signals_for_science_child()") < source.index(
        "result = run_raw_round("
    )


def test_cli_bootstrap_rejects_ambiguous_commands_and_ignored_hup() -> None:
    runner = Path("scripts/autodl/run_bace_globalgce_k20_extension.py").resolve()
    spec = importlib.util.spec_from_file_location("k20_runner_bootstrap_test", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match="unambiguous"):
        module._command_before_science_imports(["controller", "raw-round"])

    previous = module.signal.getsignal(module.signal.SIGHUP)
    module.signal.signal(module.signal.SIGHUP, module.signal.SIG_IGN)
    try:
        with pytest.raises(RuntimeError, match="ignored release signals"):
            module._install_mask_before_science_imports(["controller"])
    finally:
        module.signal.signal(module.signal.SIGHUP, previous)


def test_slurm_is_static_refusal_before_documentation_cli() -> None:
    script = Path("scripts/slurm/run_bace_globalgce_k20_extension.sh").read_text(
        encoding="utf-8"
    )
    refusal = script.index("BLOCKED_STATIC_REFUSAL")
    exit_75 = script.index("exit 75", refusal)
    command = script.index("python scripts/autodl/run_bace_globalgce_k20_extension.py")
    assert refusal < exit_75 < command
    assert '--config "$PWD/configs/hpc.yaml"' in script
    assert "--set inference.fallback_to_heuristic=false" in script


def test_controller_has_deferred_handlers_and_no_process_signal_api() -> None:
    source = Path(k20.__file__).read_text(encoding="utf-8")
    for forbidden in ("os.kill(", ".terminate(", ".kill(", "SIGKILL"):
        assert forbidden not in source
    assert "signal.SIGTERM" in source
    assert "signal.SIGHUP" in source
    assert "STOP_REQUESTED_WAITING_FOR_SCIENCE_CHILD" in source
    marker = source.index('atomic_marker(root / "PASS", PASS_MARKER)')
    final_revalidation = source.rindex("revalidate()", 0, marker)
    assert final_revalidation < marker
    publish_return = source.index("return manifest", marker)
    assert source[marker:publish_return].count("atomic_") == 1
