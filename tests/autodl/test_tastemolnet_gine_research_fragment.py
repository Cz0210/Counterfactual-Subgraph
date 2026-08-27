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
from src.utils import autodl_tastemolnet_main_v1 as main_module
from src.utils.autodl_tastemolnet_main_v1 import (
    ADOPTION_SCHEMA,
    CURRENT_ROUTE_STATE,
    NAMESPACE_NAME,
    STAGES,
    SUPERSEDED_MARKER,
    SUPERSEDED_STATE,
    TasteMainControllerError,
    TasteMainSpec,
    prepare_tastemolnet_main,
    run_tastemolnet_main,
)
from src.utils.tastemolnet_research_policy import (
    ACTIVE_STATE,
    NO_REDISTRIBUTION_MARKER,
    PENDING_STATE,
    POLICY_V2_AUDIT_MARKER,
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
    assert autodl["schema_version"] == "tastemolnet_gine_research_autodl_v2"
    assert autodl["physical_gpu_index"] == 1
    assert autodl["gpu_lock_mode"] == "exclusive"
    assert autodl["min_free_after_reservations_gb"] == 100
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
    assert task["physical_gpu_index"] == 1
    assert task["gpu_lock_mode"] == "exclusive"
    assert task["data_splits_loaded"] == ["train", "validation"]
    assert task["test_loaded"] is False
    assert task["required_log_marker"] == "[TASTE_GINE_THREE_CLASS_PASS]"
    assert "oracle_manifest.json" in task["required_output_files"]
    assert "last.pt" in task["required_output_files"]
    assert "last_checkpoint.json" in task["required_output_files"]
    assert "checkpoint_reload.json" in task["required_output_files"]
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
    assert receipt_payload["schema_version"] == (
        "tastemolnet_research_reporting_policy_receipt_v2"
    )
    assert receipt_payload["policy"]["policy_version"] == 2
    assert receipt_payload["terminal_marker"] == POLICY_V2_AUDIT_MARKER
    assert receipt_payload["no_redistribution_marker"] == NO_REDISTRIBUTION_MARKER
    assert (receipt.parent / POLICY_V2_AUDIT_MARKER).is_file()
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("physical_gpu_index", 1.0),
        ("gpu_memory_reservation_mb", 0.0),
        ("run_tastemolnet", 1.0),
        ("classifier_contract.num_classes", 3.0),
        ("classifier_contract.source_label", 1.0),
    ],
)
def test_active_fragment_numeric_authority_requires_native_json_integers(
    tmp_path: Path, field: str, value: object
) -> None:
    policy, prepared, cache, receipt = _active_authorities(tmp_path)
    fragment = build_tastemolnet_gine_research_fragment(
        policy_path=policy,
        prepared_root=prepared,
        graph_cache_root=cache,
        policy_receipt=receipt,
        expected_output_root=tmp_path / "fresh-science-root",
    )
    target = fragment["tasks"][0]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(TasteResearchPolicyError, match="task contract|authority-closed"):
        validate_tastemolnet_gine_research_fragment(fragment, require_active=True)


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


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_policy_receipt_run_flag_requires_a_native_json_integer(
    tmp_path: Path, value: object
) -> None:
    policy, prepared, cache, receipt = _active_authorities(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["run_tastemolnet"] = value
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(TasteResearchPolicyError, match="typed Taste policy receipt"):
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


def test_autodl_wrapper_uses_scoped_policy_gpu1_storage_and_no_license_pass() -> None:
    wrapper = (
        PROJECT_ROOT / "scripts/autodl/run_tastemolnet_gnn_full.sh"
    ).read_text(encoding="utf-8")
    assert "TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW" not in wrapper
    assert "TASTE_RESEARCH_COMPUTE_ALLOWED" in wrapper
    assert "TASTE_PAPER_RESULTS_ALLOWED" in wrapper
    assert "TASTE_DATA_REDISTRIBUTION_ALLOWED" in wrapper
    assert "TASTE_UPSTREAM_LICENSE_STATUS" in wrapper
    assert "WAITING_FOR_PHYSICAL_GPU1_AND_DISK" in wrapper
    assert (
        'TASTEMOLNET_STORAGE_RESERVATION_GB="'
        '${TASTEMOLNET_STORAGE_RESERVATION_GB:-20}"'
    ) in wrapper
    assert 'MIN_FREE_AFTER_RESERVATIONS_GB="${MIN_FREE_AFTER_RESERVATIONS_GB:-100}"' in wrapper
    assert "MIN_FREE_AFTER_RESERVATIONS_GB + TASTEMOLNET_STORAGE_RESERVATION_GB" in wrapper
    assert "--graph-cache-root" in wrapper
    assert "--taste-policy-file" in wrapper
    assert "--taste-policy-sha256" in wrapper
    assert "--taste-policy-receipt" in wrapper
    assert "--taste-prepared-root" in wrapper
    assert "[TASTE_GINE_THREE_CLASS_PASS]" in wrapper
    assert "[TASTE_LICENSE_PASS]" not in wrapper


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _old_blocked_taste_authority(
    control_root: Path,
) -> tuple[Path, Path, tuple[Path, ...]]:
    source_manifest = control_root / "old-controller-source-manifest.json"
    _write_json(
        source_manifest,
        {
            "schema_version": 1,
            "controller_id": "four_methods_four_datasets_continuation_v1",
        },
    )
    task_root = (
        control_root
        / "four_methods_four_datasets_continuation"
        / "four_methods_four_datasets_continuation_v1"
        / "tasks"
        / "tastemolnet_foundation"
    )
    task_root.mkdir(parents=True)
    manifest = task_root / "manifest.json"
    gate = task_root / "gate.json"
    state = task_root / "state.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "task_id": "tastemolnet_foundation",
            "dataset": "tastemolnet",
            "status": "FROZEN",
            "blocked_reason": "TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW",
            "command": None,
            "expected_output": None,
            "adopt_existing_run_id": None,
            "adopt_gpu_index": None,
            "adopt_gpu_uuid": None,
            "controller_manifest_sha256": _sha(source_manifest),
        },
    )
    _write_json(
        gate,
        {
            "schema_version": 1,
            "task_id": "tastemolnet_foundation",
            "status": "BLOCKED",
            "reason": "TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW",
            "runs": [
                {
                    "state": "NOT_STARTED",
                    "run_id": None,
                    "gpu_index": None,
                    "gpu_uuid": None,
                    "expected_output": None,
                }
            ],
        },
    )
    _write_json(
        state,
        {
            "schema_version": 1,
            "task_id": "tastemolnet_foundation",
            "dataset": "tastemolnet",
            "stage": "TASTEMOLNET_FOUNDATION",
            "state": "BLOCKED",
            "reason": "TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW",
            "instances": {
                "main": {
                    "state": "NOT_STARTED",
                    "run_id": None,
                    "gpu_index": None,
                    "gpu_uuid": None,
                    "launcher_pid": None,
                    "worker_pid": None,
                    "child_pid": None,
                }
            },
        },
    )
    return source_manifest, task_root, (source_manifest, manifest, gate, state)


def _set_main_runtime_environment(
    monkeypatch: pytest.MonkeyPatch, *, matrix_path: Path
) -> None:
    values = {
        "RUN_TASTEMOLNET": "1",
        "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
        "TASTE_PAPER_RESULTS_ALLOWED": "1",
        "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
        "TASTE_UPSTREAM_LICENSE_STATUS": "NOT_EXPLICITLY_STATED",
        "PRIMARY_TASTE_SOURCE_LABEL": "1",
        "RUN_GNN_ABLATION": "0",
        "MAX_CONCURRENT_TASTE_FULL": "2",
        "MIN_FREE_AFTER_RESERVATIONS_GB": "100",
        "TASTEMOLNET_GPU_INDEX": "1",
        "TASTEMOLNET_STORAGE_RESERVATION_GB": "20",
        "TASTE_MATRIX_STATUS_PATH": str(matrix_path),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _main_spec(
    tmp_path: Path,
) -> tuple[TasteMainSpec, tuple[Path, ...], Path]:
    control_root = tmp_path / "control"
    runtime_root = tmp_path / "runtime"
    control_root.mkdir()
    runtime_root.mkdir()
    old_source, old_task_root, old_files = _old_blocked_taste_authority(control_root)
    policy, prepared, cache, receipt = _active_authorities(tmp_path)
    controller_id = "tastemolnet-main-v1-20260827T000000Z-deadbeef"
    namespace_root = control_root / NAMESPACE_NAME
    return (
        TasteMainSpec(
            controller_id=controller_id,
            control_root=control_root,
            runtime_root=runtime_root,
            controller_root=namespace_root / controller_id,
            old_source_manifest=old_source,
            old_task_root=old_task_root,
            policy_path=policy,
            policy_receipt=receipt,
            prepared_root=prepared,
            graph_cache_root=cache,
            project_root=PROJECT_ROOT,
            gine_controller_root=runtime_root / "fresh-gine-controller",
            gine_output_root=runtime_root / "fresh-gine-output",
            gine_training_state_root=runtime_root / "fresh-gine-training-state",
            reservation_gb=20,
            minimum_free_after_reservations_gb=100,
        ),
        old_files,
        namespace_root,
    )


def test_main_prepare_supersedes_only_old_block_and_materializes_t0_t16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, old_files, namespace_root = _main_spec(tmp_path)
    matrix = tmp_path / "main-matrix-status.json"
    methods = ["Ours", "GCFExplainer", "GlobalGCE", "ComRecGC"]
    datasets = ["AIDS", "Mutagenicity", "BACE", "TasteMolNet"]
    _write_json(
        matrix,
        {
            "schema_version": "four_methods_four_datasets_registry_v1",
            "matrix_complete_cells": 0,
            "matrix_total_cells": 16,
            "cells": [
                {"dataset": dataset, "method": method, "status": "MISSING"}
                for dataset in datasets
                for method in methods
            ],
        },
    )
    _set_main_runtime_environment(monkeypatch, matrix_path=matrix)
    monkeypatch.setattr(
        main_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=220 * 1024**3),
    )
    unchanged = {path: path.read_bytes() for path in (*old_files, matrix)}

    result = prepare_tastemolnet_main(spec)

    assert namespace_root.is_dir()
    assert spec.controller_root.parent == namespace_root
    assert spec.controller_root.is_dir()
    adoption = json.loads(
        (namespace_root / "policy_adoption.json").read_text(encoding="utf-8")
    )
    assert adoption["schema_version"] == ADOPTION_SCHEMA
    assert adoption["old_state"] == "BLOCKED_LICENSE_REVIEW"
    assert adoption["old_blocker_code"] == (
        "TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW"
    )
    assert adoption["old_state_superseded"] is True
    assert adoption["superseded_state"] == SUPERSEDED_STATE
    assert adoption["current_route_state"] == CURRENT_ROUTE_STATE
    assert adoption["old_science_adopted"] is False
    assert adoption["new_policy_version"] == 2
    assert adoption["research_compute_allowed"] is True
    assert adoption["paper_result_reporting_allowed"] is True
    assert adoption["data_redistribution_allowed"] is False
    assert adoption["license_pass_claimed"] is False

    for marker in (
        main_module.POLICY_MARKER,
        SUPERSEDED_MARKER,
        main_module.NO_REDISTRIBUTION_MARKER,
    ):
        assert (
            spec.controller_root / marker
        ).read_text(encoding="utf-8").strip() == marker

    expected_stage_files = {
        "manifest.json",
        "state.json",
        "gate.json",
        "input_hashes.json",
        "output_hashes.json",
    }
    assert {
        path.name for path in (spec.controller_root / "stages").iterdir()
    } == set(STAGES)
    for stage in STAGES:
        stage_root = spec.controller_root / "stages" / stage
        assert {path.name for path in stage_root.iterdir()} == expected_stage_files
    assert len(STAGES) == 17

    queue = result["queue"]
    stage_status = {row["stage"]: row["status"] for row in queue["stages"]}
    assert stage_status["T0_POLICY_MIGRATION"] == "PASS"
    assert stage_status["T1_DATA_READY"] == "PASS"
    assert stage_status["T2_GINE_FULL"] == "READY"
    assert stage_status["T5_CLEAN_POLICY_READY"] == (
        "RELEASE_DISABLED_PENDING_FINAL_T3_T4_SOURCE_EXECUTION_RECEIPT"
    )
    assert queue["resource_lanes"]["gpu1_taste_gine_full"] == {
        "gpu_index": 1,
        "status": "READY",
        "classifier_dependent": True,
    }
    gpu2 = queue["resource_lanes"]["gpu2_classifier_independent_precompute"]
    assert gpu2["gpu_index"] == 2
    assert gpu2["status"] == "READY_CLASSIFIER_INDEPENDENT_PRECOMPUTE"
    assert gpu2["science_started"] is False
    assert gpu2["classifier_dependent"] is False
    assert gpu2["allowed_splits"] == ["train"]
    assert gpu2["initializer_data_split_used"] == "none"
    assert gpu2["taste_split_access_max"] == "train_only"
    assert gpu2["t5_release_enabled"] is False
    assert gpu2["t5_release_state"] == (
        "RELEASE_DISABLED_PENDING_FINAL_T3_T4_SOURCE_EXECUTION_RECEIPT"
    )
    assert gpu2["test_loaded"] is False
    assert queue["gnn_ablation_started"] is False

    state = result["state"]
    assert state["phase"] == "READY_FOR_MAIN_ROUTE"
    assert state["current_stage"] == "T2_GINE_FULL"
    assert state["controller_pid"] is None
    assert state["gine_gpu_index"] == 1
    assert state["gpu2_stage"] == "READY_CLASSIFIER_INDEPENDENT_PRECOMPUTE"
    assert state["gpu2_science_started"] is False
    assert state["main_matrix_mutated"] is False
    assert state["main_matrix_evidence_at_start"]["sha256"] == _sha(matrix)
    assert state["main_matrix_evidence_at_start"]["matrix_complete_cells"] == 0
    assert state["main_matrix_evidence_at_start"]["matrix_total_cells"] == 16
    assert state["main_matrix_evidence_at_start"][
        "taste_gine_counts_as_method_cell"
    ] is False
    assert result["storage"]["requested_gb"] == 20
    assert result["storage"]["minimum_free_after_reservations_gb"] == 100
    assert result["storage"]["free_after_reservation_gb"] == 200

    for path, before in unchanged.items():
        assert path.read_bytes() == before
    assert not spec.gine_controller_root.exists()
    assert not spec.gine_output_root.exists()
    assert not spec.gine_training_state_root.exists()


def test_main_environment_drift_fails_before_fresh_namespace_or_old_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, old_files, namespace_root = _main_spec(tmp_path)
    matrix = tmp_path / "absent-main-matrix-status.json"
    _set_main_runtime_environment(monkeypatch, matrix_path=matrix)
    monkeypatch.setenv("RUN_GNN_ABLATION", "1")
    unchanged = {path: path.read_bytes() for path in old_files}

    with pytest.raises(TasteMainControllerError, match="environment changed"):
        prepare_tastemolnet_main(spec)

    assert not namespace_root.exists()
    assert not matrix.exists()
    assert not spec.gine_controller_root.exists()
    assert not spec.gine_output_root.exists()
    assert not spec.gine_training_state_root.exists()
    for path, before in unchanged.items():
        assert path.read_bytes() == before


def test_main_resume_rejects_controller_spec_drift_before_gine_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, old_files, _namespace_root = _main_spec(tmp_path)
    matrix = tmp_path / "main-matrix-status.json"
    methods = ["Ours", "GCFExplainer", "GlobalGCE", "ComRecGC"]
    datasets = ["AIDS", "Mutagenicity", "BACE", "TasteMolNet"]
    _write_json(
        matrix,
        {
            "schema_version": "four_methods_four_datasets_registry_v1",
            "matrix_complete_cells": 0,
            "matrix_total_cells": 16,
            "cells": [
                {"dataset": dataset, "method": method, "status": "MISSING"}
                for dataset in datasets
                for method in methods
            ],
        },
    )
    _set_main_runtime_environment(monkeypatch, matrix_path=matrix)
    monkeypatch.setattr(
        main_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=220 * 1024**3),
    )
    prepare_tastemolnet_main(spec)
    controller_spec_path = spec.controller_root / "controller_spec.json"
    controller_spec = json.loads(controller_spec_path.read_text(encoding="utf-8"))
    controller_spec["gine_gpu_index"] = 2
    _write_json(controller_spec_path, controller_spec)
    protected = {
        path: path.read_bytes()
        for path in (
            *old_files,
            spec.controller_root / "state.json",
            spec.controller_root / "queue.json",
            spec.controller_root / "stages" / "T2_GINE_FULL" / "state.json",
        )
    }
    delegated = False

    def fail_if_delegated(*_args: object, **_kwargs: object) -> int:
        nonlocal delegated
        delegated = True
        return 0

    monkeypatch.setattr(
        main_module, "run_tastemolnet_gine_controller", fail_if_delegated
    )

    with pytest.raises(TasteMainControllerError, match="controller spec changed"):
        run_tastemolnet_main(spec, resume=True)

    assert delegated is False
    assert not spec.gine_controller_root.exists()
    assert not spec.gine_output_root.exists()
    assert not spec.gine_training_state_root.exists()
    for path, before in protected.items():
        assert path.read_bytes() == before
