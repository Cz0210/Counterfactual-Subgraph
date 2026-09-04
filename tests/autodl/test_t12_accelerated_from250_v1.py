from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.baselines.tastemolnet_gcf_full import validate_cross_gpu_resume_identity
from src.utils.tastemolnet_t12_accelerated_from250 import (
    T12AcceleratedError,
    build_prebound_continuation,
    build_promotion_blocker,
    compare_checkpoint_payloads,
    fork_step250_prefix,
    validate_mut_gpu0_release_receipt,
)


SHA = "a" * 64
OTHER_SHA = "b" * 64


def _run_identity(uuid: str, *, physical: int, runtime_sha: str) -> dict:
    gpu = {
        "schema_version": "tastemolnet_t12_a800_runtime_v1",
        "visible_selector": str(physical),
        "physical_index": physical,
        "gpu_uuid": uuid,
        "gpu_name": "NVIDIA A800-SXM4-80GB",
        "nvidia_smi_total_memory_mib": 81920,
        "torch_device_name": "NVIDIA A800-SXM4-80GB",
        "torch_total_memory_bytes": 85_000_000_000,
        "torch_version": "2.1",
        "cuda_version": "11.8",
        "cudnn_version": 8900,
        "cuda_device_count": 1,
        "cuda_used": True,
    }
    return {
        "schema_version": "tastemolnet_t12_gcf_generation_run_v1",
        "identity_template": {
            "gpu_uuid": uuid,
            "runtime_identity_sha256": runtime_sha,
            "model_checkpoint_sha256": SHA,
            "checkpoint_cursor": 250,
        },
        "runtime": {"gpu": gpu, "determinism": {"exact": True}},
        "transition_contract_sha256": runtime_sha,
        "cohort_manifest": {"sha256": SHA},
        "production_parameters": {"M": 510},
    }


def _segment(name: str, body: bytes, *, transition: bool = False) -> dict:
    if transition:
        return {
            "segment_file": name,
            "segment_index": 0,
            "committed_bytes": len(body),
            "event_count": 4,
            "final_chain_sha256": SHA,
        }
    return {
        "segment_file": name,
        "header_sha256": SHA,
        "anchor_sequence": 0,
        "anchor_chain_head": "0" * 64,
        "record_count": 4,
        "terminal_sequence": 4,
        "terminal_chain_head": SHA,
        "committed_bytes": len(body),
        "committed_prefix_sha256": SHA,
    }


def _payload(root: Path, *, cursor: int) -> dict:
    history_body = b"history-prefix"
    first_body = b"first-seen-prefix"
    transition_body = b"transition-prefix"
    return {
        "identity": {"checkpoint_cursor": cursor, "model": SHA},
        "state": {
            "official": {
                "traversed_hashes": [SHA] * cursor,
                "transitions": {
                    "root": str(root / "transition_store"),
                    "segments": [
                        _segment("transitions-00.bin", transition_body, transition=True)
                    ],
                    "chain_sha256": SHA,
                    "event_count": 4,
                    "active_sources": [SHA],
                },
            },
            "bridge": {
                "history": {
                    "history_root": str(root / "bridge_history"),
                    "segments": [_segment("history-one.bin", history_body)],
                    "observation_count": 4,
                    "chain_head": SHA,
                    "first_seen_embedding_store": {
                        "store_root": str(
                            root / "bridge_history/first-seen-embeddings"
                        ),
                        "segments": [
                            _segment("embeddings-one.bin", first_body)
                        ],
                        "record_count": 4,
                        "chain_head": SHA,
                    },
                },
                "records": {SHA: {"prediction": 0, "probabilities": [0.6, 0.3, 0.1]}},
                "lineage_occurrences": {SHA: {OTHER_SHA: 1}},
            },
            "adapter": {"call_count": cursor, "decode_failures": {}},
            "action_counts": {"NA": cursor},
            "current_graph_identity": SHA,
        },
        "rng": {"python": [cursor], "torch": SHA},
        "state_sha256": OTHER_SHA,
        "rng_sha256": SHA,
        "generated_to_original_neurosed_assertion": True,
        "test_loaded": False,
    }


def _write_prefix_fixture(root: Path) -> None:
    (root / "bridge_history/first-seen-embeddings").mkdir(parents=True)
    (root / "transition_store").mkdir()
    (root / "run_identity.json").write_text("{}", encoding="utf-8")
    (root / "cohort_manifest.json").write_text("{}", encoding="utf-8")
    (root / "cohort.jsonl").write_text("fixture\n", encoding="utf-8")
    (root / "bridge_history/history-one.bin").write_bytes(b"history-prefix")
    (root / "bridge_history/first-seen-embeddings/embeddings-one.bin").write_bytes(
        b"first-seen-prefix"
    )
    (root / "transition_store/transitions-00.bin").write_bytes(
        b"transition-prefix"
    )


def test_existing_reference_not_restarted_and_gpu1_authorization_is_owner_bound() -> None:
    runner = Path("scripts/autodl/run_t12_accelerated_from250_v1.py").read_text()
    launcher = Path("scripts/autodl/launch_t12_accelerated_from250_v1.sh").read_text()
    builder = Path("scripts/autodl/build_t12_accelerated_from250_v1.py").read_text()
    assert "kill(" not in runner
    assert "SIGTERM" not in runner
    assert "pkill" not in launcher
    assert "validate_mut_gpu0_release_receipt" not in runner
    assert '"ALLOW_T12_ACCELERATED_FROM_CHECKPOINT250_NOW"' in builder
    assert 'args.gpu_index != 1' in builder
    assert "ACCELERATED_CHECKPOINT_CURSORS = (250, 500, 510)" in runner
    assert "SOURCE_CHECKPOINT_AUTHENTICATES_250_500_510_SCHEDULE" in builder
    assert "reference_signaled\": False" in runner


def test_mut_gpu0_release_receipt_is_physical_and_fail_closed(tmp_path: Path) -> None:
    receipt = (tmp_path / "mut-release.json").resolve()
    receipt.write_text(
        json.dumps({"status": "PASS", "gpu_index": 0, "gpu_released": True}),
        encoding="utf-8",
    )
    report = validate_mut_gpu0_release_receipt(receipt)
    assert report["gpu_index"] == 0
    assert report["gpu_released"] is True
    receipt.write_text(
        json.dumps({"status": "PASS", "gpu_index": 1, "gpu_released": True}),
        encoding="utf-8",
    )
    with pytest.raises(T12AcceleratedError, match="not released"):
        validate_mut_gpu0_release_receipt(receipt)
    target = (tmp_path / "target.json").resolve()
    target.write_text(
        json.dumps({"status": "PASS", "gpu_index": 0, "gpu_released": True}),
        encoding="utf-8",
    )
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(T12AcceleratedError, match="physical file"):
        validate_mut_gpu0_release_receipt(alias)


def test_checkpoint250_common_fork_and_firstseen_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (tmp_path / "reference").resolve()
    target = (tmp_path / "accelerated").resolve()
    source.mkdir()
    _write_prefix_fixture(source)
    source_manifest = source / "checkpoints/checkpoint-00000250.manifest.json"
    source_manifest.parent.mkdir()
    source_manifest.write_text("{}", encoding="utf-8")
    payload = _payload(source, cursor=250)

    import src.baselines.tastemolnet_gcf_full_resume as resume

    monkeypatch.setattr(
        resume,
        "reopen_checkpoint",
        lambda *_args, **_kwargs: copy.deepcopy(payload),
    )

    def fake_write(root: Path, observed: dict, *, torch: object) -> Path:
        del torch
        assert observed["state"]["bridge"]["history"]["history_root"] == str(
            target / "bridge_history"
        )
        assert observed["rng"] == payload["rng"]
        root.mkdir(parents=True)
        path = root / "checkpoint-00000250.manifest.json"
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(resume, "write_checkpoint", fake_write)
    result = fork_step250_prefix(
        source_root=source,
        target_root=target,
        source_checkpoint_manifest=source_manifest,
        expected_identity={"checkpoint_cursor": 250},
        torch=object(),
    )
    assert result["status"] == "PASS"
    assert result["scientific_state_mutated"] is False
    assert (target / "bridge_history/history-one.bin").read_bytes() == b"history-prefix"
    assert (
        target / "bridge_history/first-seen-embeddings/embeddings-one.bin"
    ).read_bytes() == b"first-seen-prefix"
    assert (source / "bridge_history/history-one.bin").read_bytes() == b"history-prefix"


def test_accelerated_from250_accepts_only_transport_gpu_change() -> None:
    reference = _run_identity("GPU-" + "1" * 32, physical=3, runtime_sha=SHA)
    accelerated = _run_identity("GPU-" + "2" * 32, physical=0, runtime_sha=OTHER_SHA)
    receipt = validate_cross_gpu_resume_identity(
        current=accelerated, authority=reference
    )
    assert receipt["checkpoint_identity_retained_from_authority"] is True
    changed = copy.deepcopy(accelerated)
    changed["identity_template"]["model_checkpoint_sha256"] = OTHER_SHA
    with pytest.raises(Exception, match="non-transport"):
        validate_cross_gpu_resume_identity(current=changed, authority=reference)


def test_251_500_step_parity_uses_complete_checkpoint_state(tmp_path: Path) -> None:
    reference = _payload((tmp_path / "reference").resolve(), cursor=500)
    accelerated = _payload((tmp_path / "accelerated").resolve(), cursor=500)
    report = compare_checkpoint_payloads(
        reference=reference, accelerated=accelerated
    )
    assert report["status"] == "ENDPOINT_STATE_MATCH"
    assert report["first_seen_authenticated_state_exact"] is True
    assert report["per_step_251_500_parity_proven"] is False
    assert report["promotion_allowed"] is False
    changed = copy.deepcopy(accelerated)
    changed["state"]["action_counts"]["NA"] += 1
    with pytest.raises(T12AcceleratedError, match="action_counts"):
        compare_checkpoint_payloads(reference=reference, accelerated=changed)


def test_reload501_510_requires_rng_and_complete_state(tmp_path: Path) -> None:
    reference = _payload((tmp_path / "reference").resolve(), cursor=510)
    accelerated = _payload((tmp_path / "accelerated").resolve(), cursor=510)
    assert compare_checkpoint_payloads(
        reference=reference, accelerated=accelerated
    )["rng_exact"] is True
    accelerated["rng"]["python"] = [509]
    with pytest.raises(T12AcceleratedError, match="rng"):
        compare_checkpoint_payloads(reference=reference, accelerated=accelerated)


def test_publisher_precreated_but_blocked_until_parity(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    plan = build_prebound_continuation(
        accelerated_spec_path=root / "spec.json",
        accelerated_root=root / "accelerated",
        full_root=root / "full",
        postprocess_root=root / "postprocess",
        publisher_root=root / "publisher",
        matrix_authority_root=root / "matrix",
    )
    assert plan["status"] == "BLOCKED_PENDING_PRODUCTION_IDENTITY_REFRAME"
    assert plan["dispatchable"] is False
    assert plan["full"]["state"] == "BLOCKED_PENDING_PRODUCTION_IDENTITY_REFRAME"
    assert plan["full"]["dispatchable"] is False
    assert plan["full"]["task_spec"] is None
    assert "resume_cursor" not in plan["full"]
    assert plan["full"]["diagnostic_source_cursor"] == 500
    assert plan["postprocess"]["dispatchable"] is False
    assert plan["publisher"]["state"] == "PREBOUND_NOT_DISPATCHED"
    assert plan["publisher"]["dispatchable"] is False
    assert plan["publisher"]["matrix_authority_root"] == str(root / "matrix")
    assert plan["authorized_parallel_gpu"] == 1
    assert plan["reference_gpu"] == 3
    assert plan["reference_must_continue"] is True


def test_current_schema_emits_exact_fail_closed_promotion_blocker() -> None:
    blocker = build_promotion_blocker()
    assert blocker["status"] == "BLOCKED_UNSUPPORTED_BY_CURRENT_STATE_SCHEMA"
    assert blocker["promotion_allowed"] is False
    assert blocker["per_step_parity"]["status"] == "BLOCKED_MISSING_REFERENCE_LEDGER"
    assert blocker["per_step_parity"]["endpoint_comparison_is_sufficient"] is False
    assert set(blocker["per_step_parity"]["missing_authenticated_per_step_fields"]) == {
        "selected_parent",
        "selected_action",
        "pre_softmax_gine_logits",
        "normalized_neurosed_distance",
    }
    reframe = blocker["production_identity_reframe"]
    assert reframe["diagnostic_total_steps"] == 510
    assert reframe["diagnostic_source_cursor"] == 500
    assert reframe["target_total_steps"] == 20_000
    assert reframe["target_checkpoint_cursors"] == list(range(2_500, 20_001, 2_500))
    assert reframe["direct_identity_relabel_allowed"] is False
    assert blocker["full_owner_spec_created"] is False
    assert blocker["gpu_task_started"] is False
