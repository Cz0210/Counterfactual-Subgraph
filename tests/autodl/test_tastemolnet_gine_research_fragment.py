from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.autodl.build_tastemolnet_gine_research_tasks import main as builder_main
from scripts.audit_tastemolnet_research_policy import (
    PENDING_MARKER,
    audit_research_policy,
    main as policy_audit_main,
)
from scripts.train_molecular_gnn import _taste_runtime_authority
from src.baselines.tastemolnet_gine_research_tasks import (
    PENDING_REASON,
    TASK_ID,
    build_tastemolnet_gine_research_fragment,
    validate_tastemolnet_gine_research_fragment,
)
from src.utils.env import load_and_merge_config_files
from src.utils.tastemolnet_research_policy import (
    ACTIVE_AUDIT_MARKER,
    ACTIVE_STATE,
    NO_REDISTRIBUTION_MARKER,
    PENDING_STATE,
    SOURCE_CSV_SHA256,
    UPSTREAM_COMMIT,
    TasteResearchPolicyError,
    stable_json_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY = PROJECT_ROOT / "configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml"
CONFIG = PROJECT_ROOT / "configs/autodl/tastemolnet_gine_research_v1.yaml"
SPLIT_ROWS = {"train": 9437, "validation": 1328, "calibration": 1328, "test": 1328}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _active_policy(tmp_path: Path, *, prepared: Path | None = None) -> Path:
    payload = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    payload["authorization_basis"] = "explicit_user_instruction"
    payload["authorization_state"] = ACTIVE_STATE
    payload["authorization_source"] = "user_project_owner_instruction"
    payload["research_compute_allowed"] = True
    payload["paper_result_reporting_allowed"] = True
    payload["aggregated_metrics_release_allowed"] = True
    payload["figure_release_allowed"] = True
    payload["permissions"]["research_execution"] = "ALLOWED"
    payload["permissions"]["paper_reporting"] = "ALLOWED"
    payload["permissions"]["aggregate_publication"] = (
        "ALLOWED_AFTER_PUBLIC_ARTIFACT_AUDIT"
    )
    payload["execution"]["run_tastemolnet"] = 1
    if prepared is not None:
        payload["dataset_identity"]["prepared_output_manifest_sha256"] = _sha(
            prepared / "output_manifest.json"
        )
        payload["dataset_identity"]["split_manifest_sha256"] = _sha(
            prepared / "splits/split_manifest.json"
        )
    path = tmp_path / "active-policy.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _pending_policy(tmp_path: Path, *, prepared: Path | None = None) -> Path:
    payload = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    payload["authorization_basis"] = "forwarded_user_instruction_pending_root_activation"
    payload["authorization_state"] = PENDING_STATE
    payload["authorization_source"] = "PENDING_ROOT_ACTIVATION"
    payload["research_compute_allowed"] = False
    payload["paper_result_reporting_allowed"] = False
    payload["aggregated_metrics_release_allowed"] = False
    payload["figure_release_allowed"] = False
    payload["trained_model_release_allowed"] = False
    payload["permissions"]["research_execution"] = "PENDING_ROOT_ACTIVATION"
    payload["permissions"]["paper_reporting"] = "PENDING_ROOT_ACTIVATION"
    payload["permissions"]["aggregate_publication"] = (
        "ALLOWED_ONLY_AFTER_ACTIVATION_AND_PUBLIC_ARTIFACT_AUDIT"
    )
    payload["execution"]["run_tastemolnet"] = 0
    if prepared is not None:
        payload["dataset_identity"]["prepared_output_manifest_sha256"] = _sha(
            prepared / "output_manifest.json"
        )
        payload["dataset_identity"]["split_manifest_sha256"] = _sha(
            prepared / "splits/split_manifest.json"
        )
    path = tmp_path / "pending-policy.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _private_authority(tmp_path: Path) -> tuple[Path, Path]:
    prepared = tmp_path / "prepared"
    split_root = prepared / "splits"
    split_root.mkdir(parents=True)
    for split in SPLIT_ROWS:
        (split_root / f"{split}.csv").write_text(
            "molecule_id,model_smiles,label,split\n", encoding="utf-8"
        )
    (split_root / "split_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "tastemolnet",
                "num_classes": 3,
                "label_map": {"0": "Bitter", "1": "Sweet", "2": "Tasteless"},
                "source_label": 1,
                "scaffold_overlap_gate_passed": True,
                "all_classes_present_per_split": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (split_root / "split_statistics.json").write_text(
        json.dumps(
            {
                "total_clean_rows": 13421,
                "splits": {
                    split: {"rows": count, "class_counts": {"0": 1, "1": 1, "2": 1}}
                    for split, count in SPLIT_ROWS.items()
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (prepared / "provenance_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "tastemolnet",
                "upstream_commit": UPSTREAM_COMMIT,
                "source_csv_sha256": SOURCE_CSV_SHA256,
                "download_performed": False,
                "raw_data_copied_into_output": False,
                "raw_data_commit_allowed": False,
                "license_status": "LICENSE_REVIEW_REQUIRED",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (prepared / "LICENSE_REVIEW_REQUIRED").write_text(
        "upstream terms remain not explicitly stated\n", encoding="utf-8"
    )
    identities: dict[str, dict[str, object]] = {}
    for path in sorted(prepared.rglob("*")):
        if path.is_file() and path.name != "output_manifest.json":
            relative = path.relative_to(prepared).as_posix()
            identities[relative] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
    (prepared / "output_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": identities,
                "manifest_digest": stable_json_sha256(identities),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    cache = tmp_path / "cache"
    cache.mkdir()
    cache_splits = {}
    for split, count in SPLIT_ROWS.items():
        path = cache / f"{split}.pt"
        path.write_bytes(f"private-cache-{split}".encode())
        cache_splits[split] = {
            "cache_file": path.name,
            "cache_sha256": _sha(path),
            "source_csv_sha256": _sha(split_root / f"{split}.csv"),
            "graph_count": count,
            "num_classes": 3,
            "safe_load_verified": True,
        }
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "molecular_graph_cache_manifest_v1",
                "dataset": "tastemolnet",
                "num_classes": 3,
                "split_order": ["train", "validation", "calibration", "test"],
                "total_graph_count": 13421,
                "splits": cache_splits,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return prepared, cache


def _active_authorities(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    prepared, cache = _private_authority(tmp_path)
    policy = _active_policy(tmp_path, prepared=prepared)
    audit_root = tmp_path / "policy-audit"
    audit_research_policy(
        policy_path=policy,
        prepared_root=prepared,
        graph_cache_root=cache,
        output_dir=audit_root,
        require_active=True,
    )
    return policy, prepared, cache, audit_root / "tastemolnet_policy_receipt.json"


def test_checked_config_and_fragment_are_active_and_authority_closed(tmp_path: Path) -> None:
    config = load_and_merge_config_files([CONFIG])
    autodl = config["autodl"]
    assert autodl["run_tastemolnet"] is True
    assert autodl["physical_gpu_index"] == 2
    assert autodl["gpu_lock_mode"] == "exclusive"
    assert autodl["num_classes"] == 3
    assert autodl["source_label"] == 1
    assert autodl["rf_oracle_used"] is False
    assert autodl["test_loaded"] is False
    assert autodl["hpc_execution_allowed"] is False
    config_training = config["training"]
    assert config_training["selection_metric"] == "macro_ovr_roc_auc"
    assert config_training["selection_tiebreak_metric"] == "macro_f1"
    assert config_training["health_gate"]["require_all_class_recall"] is True
    assert config["calibration"]["fit_on_validation"] is True
    assert autodl["prepared_output_manifest_sha256"] == (
        "36aaf17bf45e0a092a96a0379fab31d9e6bfcd719b87cb4ffa4e57a6642bb645"
    )
    assert autodl["split_manifest_sha256"] == (
        "841f3b911e5d353c1e00f010bafcc8a6f7b3433082dba8a8979fab1b558251af"
    )
    policy, prepared, cache, receipt = _active_authorities(tmp_path)
    fragment = build_tastemolnet_gine_research_fragment(
        policy_path=policy,
        prepared_root=prepared,
        graph_cache_root=cache,
        policy_receipt=receipt,
        expected_output_root=tmp_path / "fresh-science-root",
    )
    validate_tastemolnet_gine_research_fragment(fragment, require_active=True)
    task = fragment["tasks"][0]
    assert task["id"] == TASK_ID
    assert task["enabled"] is True
    assert task["blocked_reason"] is None
    assert task["run_tastemolnet"] == 1
    assert task["command"] == task["command_template"]
    assert task["physical_gpu_index"] == 2
    assert task["gpu_lock_mode"] == "exclusive"
    assert task["data_splits_loaded"] == ["train", "validation"]
    assert task["test_loaded"] is False
    assert task["required_log_marker"] == "[TASTE_GINE_THREE_CLASS_PASS]"
    assert "oracle_manifest.json" in task["required_output_files"]
    assert task["environment"]["TASTE_DATA_REDISTRIBUTION_ALLOWED"] == "0"
    assert task["environment"]["TASTE_UPSTREAM_LICENSE_STATUS"] == (
        "NOT_EXPLICITLY_STATED"
    )
    assert fragment["controller_contract"]["generic_four_gpu_controller_eligible"] is False


def test_active_fragment_requires_and_binds_policy_data_cache_and_receipt(tmp_path: Path) -> None:
    policy, prepared, cache, receipt = _active_authorities(tmp_path)
    fragment = build_tastemolnet_gine_research_fragment(
        policy_path=policy,
        expected_policy_sha256=_sha(policy),
        prepared_root=prepared,
        graph_cache_root=cache,
        policy_receipt=receipt,
        expected_output_root=tmp_path / "fresh-science-root",
    )
    validate_tastemolnet_gine_research_fragment(fragment, require_active=True)
    task = fragment["tasks"][0]
    assert task["enabled"] is True
    assert task["run_tastemolnet"] == 1
    assert task["command"] == [
        "bash",
        "scripts/autodl/run_tastemolnet_gine_controller.sh",
    ]
    assert task["environment"]["TASTEMOLNET_GNN_FULL_OUTPUT"] == task[
        "expected_output"
    ]
    assert task["environment"]["TASTEMOLNET_GINE_CONTROLLER_ROOT"] == task[
        "persistent_controller_root"
    ]
    assert task["policy_receipt"] == {"path": str(receipt), "sha256": _sha(receipt)}
    authority = task["data_contract"]["authority"]
    assert authority["prepared_rows"] == 13421
    assert authority["split_rows"] == SPLIT_ROWS
    assert authority["graph_cache_rows"] == 13421
    assert authority["data_reprepared"] is False
    assert authority["graph_cache_rebuilt"] is False
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["terminal_marker"] == ACTIVE_AUDIT_MARKER
    assert receipt_payload["no_redistribution_marker"] == NO_REDISTRIBUTION_MARKER
    assert (receipt.parent / ACTIVE_AUDIT_MARKER).is_file()
    assert (receipt.parent / NO_REDISTRIBUTION_MARKER).is_file()

    runtime = _taste_runtime_authority(
        SimpleNamespace(
            taste_policy_file=str(policy),
            taste_policy_sha256=_sha(policy),
            taste_policy_receipt=str(receipt),
            taste_prepared_root=str(prepared),
            graph_cache_root=str(cache),
        ),
        dataset_id="tastemolnet",
        profile="full",
        split_paths={
            split: prepared / "splits" / f"{split}.csv" for split in SPLIT_ROWS
        },
    )
    assert runtime is not None
    runtime_policy, runtime_authority, runtime_receipt = runtime
    assert runtime_policy.active is True
    assert runtime_authority.graph_cache_rows == 13421
    assert runtime_receipt.sha256 == _sha(receipt)


def test_pending_policy_audit_is_read_only_disabled_and_never_pass(tmp_path: Path) -> None:
    prepared, cache = _private_authority(tmp_path)
    pending_policy = _pending_policy(tmp_path, prepared=prepared)
    output = tmp_path / "policy-audit"
    receipt = audit_research_policy(
        policy_path=pending_policy,
        prepared_root=prepared,
        graph_cache_root=cache,
        output_dir=output,
    )
    assert receipt["run_tastemolnet"] == 0
    assert receipt["heavy_route_authorized"] is False
    assert receipt["paper_reporting_authorized"] is False
    assert receipt["dataset_redistribution_authorized"] is False
    assert receipt["upstream_terms_status"] == "NOT_EXPLICITLY_STATED"
    assert receipt["terminal_marker"] == PENDING_MARKER
    assert (output / PENDING_MARKER).is_file()
    assert not (output / "PASS").exists()
    assert "LICENSE_PASS" not in json.dumps(receipt, sort_keys=True)
    blocked_output = tmp_path / "require-active"
    assert policy_audit_main(
        [
            "--policy",
            str(pending_policy),
            "--prepared-root",
            str(prepared),
            "--graph-cache-root",
            str(cache),
            "--output-dir",
            str(blocked_output),
            "--require-active",
        ]
    ) == 65
    assert not blocked_output.exists()

    private_output = prepared / "audit"
    with pytest.raises(TasteResearchPolicyError, match="disjoint"):
        audit_research_policy(
            policy_path=pending_policy,
            prepared_root=prepared,
            graph_cache_root=cache,
            output_dir=private_output,
        )
    assert not private_output.exists()


def test_active_policy_without_receipt_or_cache_fails_closed(tmp_path: Path) -> None:
    policy = _active_policy(tmp_path)
    with pytest.raises(TasteResearchPolicyError, match="prepared_root is required"):
        build_tastemolnet_gine_research_fragment(
            policy_path=policy,
            expected_output_root=tmp_path / "out",
        )


def test_inactive_template_cannot_claim_live_authority(tmp_path: Path) -> None:
    pending = _pending_policy(tmp_path)
    with pytest.raises(TasteResearchPolicyError, match="inactive template"):
        build_tastemolnet_gine_research_fragment(
            policy_path=pending,
            prepared_root=tmp_path / "prepared",
            expected_output_root=tmp_path / "out",
        )


def test_fragment_requires_fresh_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "already-exists"
    output_root.mkdir()
    with pytest.raises(TasteResearchPolicyError, match="fresh and absent"):
        build_tastemolnet_gine_research_fragment(
            policy_path=POLICY,
            expected_output_root=output_root,
        )


def test_receipt_or_cache_tamper_is_rejected(tmp_path: Path) -> None:
    policy, prepared, cache, receipt = _active_authorities(tmp_path)
    (cache / "train.pt").write_bytes(b"tampered")
    with pytest.raises(TasteResearchPolicyError, match="graph-cache train hash changed"):
        build_tastemolnet_gine_research_fragment(
            policy_path=policy,
            prepared_root=prepared,
            graph_cache_root=cache,
            policy_receipt=receipt,
            expected_output_root=tmp_path / "out",
        )


def test_prepared_manifest_hash_is_bound_by_policy(tmp_path: Path) -> None:
    policy, prepared, cache, receipt = _active_authorities(tmp_path)
    manifest = prepared / "output_manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(TasteResearchPolicyError, match="manifest authority changed"):
        build_tastemolnet_gine_research_fragment(
            policy_path=policy,
            prepared_root=prepared,
            graph_cache_root=cache,
            policy_receipt=receipt,
            expected_output_root=tmp_path / "out",
        )


def test_active_output_root_must_be_disjoint_from_private_authority(
    tmp_path: Path,
) -> None:
    policy, prepared, cache, receipt = _active_authorities(tmp_path)
    with pytest.raises(TasteResearchPolicyError, match="disjoint"):
        build_tastemolnet_gine_research_fragment(
            policy_path=policy,
            prepared_root=prepared,
            graph_cache_root=cache,
            policy_receipt=receipt,
            expected_output_root=prepared / "future-output",
        )


def test_fragment_rejects_output_with_symlinked_parent(tmp_path: Path) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    with pytest.raises(TasteResearchPolicyError, match="symlink components"):
        build_tastemolnet_gine_research_fragment(
            policy_path=POLICY,
            expected_output_root=alias / "future-output",
        )


def test_builder_cli_writes_only_fresh_active_fragment(tmp_path: Path, capsys) -> None:
    policy, prepared, cache, receipt = _active_authorities(tmp_path)
    output = tmp_path / "fragment.json"
    root = tmp_path / "science"
    assert builder_main(
        [
            "--policy",
            str(policy),
            "--prepared-root",
            str(prepared),
            "--graph-cache-root",
            str(cache),
            "--policy-receipt",
            str(receipt),
            "--expected-output-root",
            str(root),
            "--output",
            str(output),
            "--require-active",
        ]
    ) == 0
    fragment = json.loads(output.read_text())
    assert fragment["tasks"][0]["enabled"] is True
    assert "[TASTEMOLNET_GINE_RESEARCH_FRAGMENT_ACTIVE]" in capsys.readouterr().out

    pending = _pending_policy(tmp_path)
    second = tmp_path / "pending-required.json"
    assert builder_main(
        [
            "--policy",
            str(pending),
            "--expected-output-root",
            str(root),
            "--output",
            str(second),
            "--require-active",
        ]
    ) == 65
    assert not second.exists()

    nested_output = root / "fragment.json"
    assert builder_main(
        [
            "--policy",
            str(policy),
            "--prepared-root",
            str(prepared),
            "--graph-cache-root",
            str(cache),
            "--policy-receipt",
            str(receipt),
            "--expected-output-root",
            str(root),
            "--output",
            str(nested_output),
        ]
    ) == 65
    assert not root.exists()


def test_autodl_wrapper_uses_scoped_policy_gpu2_cache_and_no_license_pass() -> None:
    wrapper = (
        PROJECT_ROOT / "scripts/autodl/run_tastemolnet_gnn_full.sh"
    ).read_text(encoding="utf-8")
    assert "TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW" not in wrapper
    assert "TASTE_RESEARCH_COMPUTE_ALLOWED" in wrapper
    assert "TASTE_PAPER_RESULTS_ALLOWED" in wrapper
    assert "TASTE_DATA_REDISTRIBUTION_ALLOWED" in wrapper
    assert "TASTE_UPSTREAM_LICENSE_STATUS" in wrapper
    assert "WAITING_FOR_PHYSICAL_GPU2" in wrapper
    assert "--graph-cache-root" in wrapper
    assert "--taste-policy-file" in wrapper
    assert "--taste-policy-sha256" in wrapper
    assert "--taste-policy-receipt" in wrapper
    assert "--taste-prepared-root" in wrapper
    assert "[TASTE_GINE_THREE_CLASS_PASS]" in wrapper
    assert "[TASTE_LICENSE_PASS]" not in wrapper
