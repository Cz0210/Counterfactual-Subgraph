from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.bace_frozen_gnn_contracts import (
    BACEParent,
    assert_no_rf_provenance,
    file_identity,
    stable_sha256,
)
from src.eval.bace_frozen_gnn_pool import _parent_prompt, _validate_b7, merge_pool_shards
from src.eval.bace_frozen_gnn_prep import run_postfreeze_test_shard_manifest
from src.eval.full_candidate_pool import (
    CONNECTED_DELETION_PROMPT_MODE,
    render_generation_prompt,
)
from src.eval.bace_frozen_gnn_selection import (
    compute_frozen_test_metrics,
    run_b14_manifest_freeze,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _gnn_manifest(stage: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset": "bace",
        "stage": stage,
        "status": "PASS",
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "source_label": 1,
        "num_classes": 2,
        "cf_mode": "strict_flip",
        "calibration_loaded": False,
        "test_loaded": False,
    }
    payload.update(overrides)
    return payload


def test_bace_downstream_rf_guard_fails_closed() -> None:
    assert_no_rf_provenance({"rf_oracle_used": False, "oracle_backend": "gnn"})
    with pytest.raises(ValueError, match="RF guard"):
        assert_no_rf_provenance({"rf_oracle_used": True})
    with pytest.raises(ValueError, match="RF-contaminated"):
        assert_no_rf_provenance({"teacher_backend": "rf"})
    with pytest.raises(ValueError, match="Unclean"):
        assert_no_rf_provenance({"provenance_classification": "UNKNOWN"})


def test_b8_generation_prompt_is_complete_without_csv_prompt_column() -> None:
    parent = BACEParent(
        parent_id="BACE-1",
        smiles="CC(=O)N",
        label=1,
        source_row_index=0,
        prompt=None,
    )
    rendered = render_generation_prompt(
        _parent_prompt(parent), prompt_mode=CONNECTED_DELETION_PROMPT_MODE
    )
    assert "MOLECULE_SMILES: CC(=O)N" in rendered
    assert "ORIGINAL_LABEL: 1" in rendered
    assert "COUNTERFACTUAL_FRAGMENT_SMILES:" in rendered
    assert "Return fragment SMILES only" in rendered


def test_b7_adapter_identity_is_bound_to_current_disk_bytes(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    config.write_text("{}\n", encoding="utf-8")
    weights.write_bytes(b"real-adapter-bytes")
    config_identity = file_identity(config)
    weights_identity = file_identity(weights)
    hash_payload = {
        "schema_version": "bace_lora_checkpoint_identity_v1",
        "adapter_config_name": config.name,
        "adapter_config_sha256": config_identity["sha256"],
        "adapter_config_size": config_identity["size"],
        "adapter_weights_name": weights.name,
        "adapter_weights_sha256": weights_identity["sha256"],
        "adapter_weights_size": weights_identity["size"],
    }
    b7 = tmp_path / "b7"
    b7.mkdir()
    manifest = _gnn_manifest(
        "B7_PPO_FULL",
        ppo_training_performed=True,
        ppo_update_count=300,
        policy_checkpoint_hash_schema="bace_lora_checkpoint_identity_v1",
        policy_checkpoint_hash_payload=hash_payload,
        policy_checkpoint_hash=stable_sha256(hash_payload),
        final_adapter_config_identity=config_identity,
        final_adapter_weights_identity=weights_identity,
    )
    _write_json(b7 / "ppo_manifest.json", manifest)
    _write_json(b7 / "oracle_provenance.json", {"rf_oracle_used": False})
    _manifest, identity = _validate_b7(b7, policy_checkpoint=adapter)
    assert identity["declared_policy_hash"] == stable_sha256(hash_payload)
    weights.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="identity differs"):
        _validate_b7(b7, policy_checkpoint=adapter)


def test_b10_requires_eight_fixed_shards_and_merges_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.eval.bace_frozen_gnn_pool.canonicalize_smiles", lambda value: value or None
    )
    all_parent_ids = ["p0", "p1", "p2", "p3"]
    all_parent_hash = stable_sha256(all_parent_ids)
    shards: list[Path] = []
    for stage in ("B8_POOL_BASE", "B9_POOL_HIGHTEMP"):
        for shard_index, parent_id in enumerate(all_parent_ids):
            root = tmp_path / f"{stage}-{shard_index}"
            root.mkdir()
            rows = [
                {
                    "dataset": "bace",
                    "stage": stage,
                    "shard_index": shard_index,
                    "candidate_id": f"{stage}-{parent_id}-{index}",
                    "parent_id": parent_id,
                    "parent_smiles": "CC",
                    "final_fragment": f"fragment-{index:02d}",
                    "parse_ok": True,
                    "valid": True,
                    "connected": True,
                    "direct_substructure": True,
                    "oracle_ok": True,
                    "cf_flip": index % 2 == 0,
                    "cf_drop": index / 100.0,
                    "reward_total": index / 100.0,
                    "rf_oracle_used": False,
                }
                for index in range(20)
            ]
            _write_jsonl(root / "candidate_pool.jsonl", rows)
            _write_json(
                root / "pool_manifest.json",
                _gnn_manifest(
                    stage,
                    shard_index=shard_index,
                    num_shards=4,
                    all_parent_ids_sha256=all_parent_hash,
                    parent_ids=[parent_id],
                    candidate_row_count=len(rows),
                    candidate_pool_identity=file_identity(
                        root / "candidate_pool.jsonl"
                    ),
                    generation_config_hash=f"config-{stage}",
                    policy_checkpoint_hash="policy-hash",
                    oracle_checkpoint_hash="gnn-hash",
                ),
            )
            shards.append(root)
    output = tmp_path / "b10"
    result = merge_pool_shards(shard_dirs=shards, output_dir=output)
    assert result["status"] == "PASS"
    assert result["candidate_universe_count"] == 20
    assert (output / "PASS").read_text(encoding="utf-8") == "PASS\n"
    second = tmp_path / "b10-second"
    other = merge_pool_shards(shard_dirs=list(reversed(shards)), output_dir=second)
    assert result["candidate_pool_hash"] == other["candidate_pool_hash"]
    assert result["candidate_universe_hash"] == other["candidate_universe_hash"]


def test_frozen_test_metrics_use_only_ordered_prefixes_and_thresholds() -> None:
    ids = [f"rule-{index:02d}" for index in range(20)]
    frozen = {
        "ordered_rule_ids": ids,
        "thresholds": {
            "theta_star": 0.25,
            "merged_thresholds": [
                {"threshold_id": "q30", "threshold": 0.25, "weight": 1.0}
            ],
        },
    }
    rows: list[dict[str, object]] = []
    for parent in ("a", "b"):
        for index, candidate in enumerate(ids):
            strict = (parent == "a" and index == 0) or (parent == "b" and index == 1)
            rows.append(
                {
                    "parent_id": parent,
                    "candidate_id": candidate,
                    "pair_strict_flip": strict,
                    "wnode_distance": 0.1 if strict else None,
                }
            )
    metrics = compute_frozen_test_metrics(rows, frozen_selection=frozen)
    assert metrics["ccrcov_theta_star_by_k"][0] == 0.5
    assert metrics["ccrcov_theta_star_by_k"][1] == 1.0
    assert metrics["selector_refit_on_test"] is False


def test_b14_is_manifest_only_and_never_needs_raw_splits(tmp_path: Path) -> None:
    b12 = tmp_path / "b12"
    b13 = tmp_path / "b13"
    b12.mkdir()
    b13.mkdir()
    ids = [f"rule-{index:02d}" for index in range(20)]
    frozen = _gnn_manifest(
        "B12_SELECTOR",
        status="FROZEN",
        oracle_checkpoint_hash="gnn",
        molclr_checkpoint_hash="molclr",
        policy_checkpoint_hash="policy",
        selector_fitted_on_calibration=True,
        selection_frozen=True,
        K=20,
        ordered_rule_ids=ids,
        rule_hashes=[f"hash-{index}" for index in range(20)],
    )
    _write_json(b12 / "frozen_selection_manifest.json", frozen)
    verification = b13 / "matrix_manifest.json"
    pairs = b13 / "pair_matrix.jsonl"
    _write_json(verification, {"status": "PASS"})
    pairs.write_text("{}\n", encoding="utf-8")
    metrics = _gnn_manifest(
        "B13_FINAL_EVAL",
        oracle_checkpoint_hash="gnn",
        molclr_checkpoint_hash="molclr",
        policy_checkpoint_hash="policy",
        test_loaded=True,
    )
    _write_json(b13 / "final_metrics.json", metrics)
    test_manifest = _gnn_manifest(
        "B13_FINAL_EVAL",
        oracle_checkpoint_hash="gnn",
        molclr_checkpoint_hash="molclr",
        policy_checkpoint_hash="policy",
        ordered_rule_ids=ids,
        selector_fitted_on_calibration=True,
        test_used_only_after_freeze=True,
        test_loaded=True,
        frozen_selection_manifest_identity=file_identity(
            b12 / "frozen_selection_manifest.json"
        ),
        verification_manifest_identity=file_identity(verification),
        pair_matrix_identity=file_identity(pairs),
        final_metrics_identity=file_identity(b13 / "final_metrics.json"),
    )
    _write_json(b13 / "test_evaluation_manifest.json", test_manifest)
    result = run_b14_manifest_freeze(
        b12_output=b12,
        b13_output=b13,
        output_dir=tmp_path / "b14",
    )
    assert result["final_gate_pass"] is True
    assert result["manifest_only_gate"] is True
    assert result["raw_calibration_reopened"] is False
    assert result["raw_test_reopened"] is False


def test_test_parent_manifest_is_created_only_from_frozen_b12(tmp_path: Path) -> None:
    b12 = tmp_path / "b12"
    b12.mkdir()
    frozen = _gnn_manifest(
        "B12_SELECTOR",
        status="FROZEN",
        selector_fitted_on_calibration=True,
        selection_frozen=True,
        K=20,
    )
    _write_json(b12 / "frozen_selection_manifest.json", frozen)
    test_csv = tmp_path / "test.csv"
    test_csv.write_text(
        "molecule_id,smiles,label,split\n"
        "p3,CCC,1,test\n"
        "negative,CC,0,test\n"
        "p1,CN,1,test\n",
        encoding="utf-8",
    )
    output = tmp_path / "postfreeze-test-parents"
    result = run_postfreeze_test_shard_manifest(
        b12_output=b12,
        test_split=test_csv,
        output_dir=output,
    )
    document = json.loads(
        (output / "test_parent_ids.frozen.json").read_text(encoding="utf-8")
    )
    assert result["selection_frozen_before_test_load"] is True
    assert document["parent_ids"] == ["p1", "p3"]
    assert document["split"] == "test"
    assert (output / "PASS").read_text(encoding="utf-8") == "PASS\n"


def test_controller_contract_uses_actual_dependency_instance_outputs() -> None:
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "configs/autodl/bace_frozen_gnn_downstream_tasks.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rendered = json.dumps(contract, sort_keys=True)
    assert "attempt-0" not in rendered
    assert contract["execution"]["never_bind_attempt_zero_literal"] is True
    assert "numeric value 0..3" in contract["execution"]["shard_index_token"]
    assert contract["controller_task_order"][-4:] == [
        "bace_b13_test_parent_manifest",
        "bace_b13_verification_shards",
        "bace_b13_final_eval",
        "bace_b14_frozen",
    ]
    for stage in ("B8_POOL_BASE", "B9_POOL_HIGHTEMP"):
        command = contract[stage]["command"]
        assert command.count("{dep_bace_b7_ppo_full_output}") == 2
        assert "{shard_manifest}" in command
        assert "{shard_index}" in command
    b10_command = contract["B10_POOL_MERGED"]["command"]
    expected_b10 = {
        f"{{dep_bace_b8_pool_base_shard_00{index}_output}}"
        for index in range(4)
    } | {
        f"{{dep_bace_b9_pool_hightemp_shard_00{index}_output}}"
        for index in range(4)
    }
    assert expected_b10.issubset(set(b10_command))
    assert contract["B12_SELECTOR"]["command"][-2] == "--output-dir"
    b13 = contract["B13_FINAL_EVAL"]
    assert "B13_TEST_PARENT_MANIFEST" in b13["dependencies"]
    assert "{shard_manifest}" in b13["shard_command"]
    assert contract["B13_TEST_PARENT_MANIFEST"]["required_outputs"] == [
        "test_parent_ids.frozen.json",
        "test_shard_manifest.json",
        "PASS",
    ]
    assert (
        contract["B11_CROSS_PARENT_VERIFIED"]["required_merge_log_marker"]
        == "[BACE_B11_PASS]"
    )
    assert (
        contract["B13_FINAL_EVAL"]["required_merge_log_marker"]
        == "[BACE_B13_PASS]"
    )
    freeze = contract["B14_FROZEN"]
    assert freeze["manifest_only"] is True
    assert not any(
        token in {"--split-path", "--test-split", "--calibration-split"}
        for token in freeze["command"]
    )
