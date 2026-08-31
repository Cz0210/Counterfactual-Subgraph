#!/usr/bin/env python3
"""Run the real bounded TasteMolNet three-class Ours PPO smoke."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_ppo import (  # noqa: E402
    _infer_single_training_device,
    apply_config_overrides,
    apply_decoded_chem_generation_defaults,
    build_hf_dataset,
    build_quantized_base_model,
    build_tokenizer,
    build_value_model,
    collect_runtime_environment_debug,
    ensure_score_head_for_experimental_ppo,
    import_training_dependencies,
)
from scripts.train_ppo_stable import (  # noqa: E402
    build_parser as build_stable_parser,
    resolve_stable_config,
    run_stable_decoded_chem_ppo_loop,
)
from src.data.tastemolnet_ppo import load_tastemolnet_train_prompts  # noqa: E402
from src.oracles.gnn_oracle import sha256_file  # noqa: E402
from src.rewards.gnn_ppo_reward import (  # noqa: E402
    BatchedGNNPPORewardAdapter,
    GNNPPORewardConfig,
)
from src.train.bace_gnn_ppo import atomic_json, model_parameter_hash  # noqa: E402
from src.train.molecular_gnn_resume import assert_no_symlink_components  # noqa: E402
from src.train.tastemolnet_gnn_ppo import (  # noqa: E402
    TASTE_PPO_MARKER,
    TastePPOObserver,
    adapter_parameter_identity_from_model,
    build_taste_reward_manifest,
    build_taste_smoke_gate,
    validate_taste_adapter_checkpoint_reload,
    value_head_parameter_identity,
)
from src.utils.logging_utils import RunContext, configure_run_logger  # noqa: E402
from src.utils.retained_output_directory import (  # noqa: E402
    FreshOutputDirectory,
    prepare_terminal_output,
)
from src.utils.retained_readonly_file import hold_readonly_file  # noqa: E402


STAGE = "T6_OURS_SMOKE"
OUTPUT_SCHEMA = "tastemolnet_ours_ppo_smoke_manifest_v1"
T6_RELEASE_CONFIG_PATH = (
    REPO_ROOT / "configs/autodl/tastemolnet_t6_execution_release_v1.json"
)
T6_RELEASE_SCHEMA = "tastemolnet_t6_execution_release_v1"
T6_EXTERNAL_AUTHORITY_SCHEMA = "tastemolnet_t6_external_execution_authority_v1"
T6_RELEASE_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "release_enabled",
        "release_state",
        "implementation_commit",
        "implementation_tree",
        "external_authority_path",
        "external_authority_sha256",
        "t3_gate_sha256",
        "t4_gate_sha256",
        "t5_gate_sha256",
        "t5_output_inventory_sha256",
        "controller_receipt_sha256",
        "gpu_index",
        "output_parent",
    }
)


def build_parser():
    parser = build_stable_parser()
    parser.description = __doc__
    parser.add_argument("--stage", choices=(STAGE,), default=STAGE)
    parser.add_argument("--gnn-checkpoint", type=Path, required=True)
    parser.add_argument("--t5-output", type=Path, required=True)
    parser.add_argument("--downstream-policy", type=Path, required=True)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--parent-count", type=int, default=16)
    parser.add_argument("--oracle-batch-size", type=int, default=256)
    parser.add_argument("--gnn-device", default="cuda")
    return parser


def _configure(args: Any) -> None:
    if type(args.updates) is not int or not 5 <= args.updates <= 10:
        raise ValueError("Taste Ours smoke updates must be in [5, 10]")
    if type(args.parent_count) is not int or not 8 <= args.parent_count <= 16:
        raise ValueError("Taste Ours smoke parent count must be in [8, 16]")
    if type(args.oracle_batch_size) is not int or args.oracle_batch_size <= 0:
        raise ValueError("Taste Ours oracle batch size must be a positive native int")
    args.ppo_loop = "decoded_chem"
    args.require_chemistry_reward_path = True
    args.only_positive = True
    args.default_parent_label = 1
    args.disable_counterfactual_teacher = True
    args.require_teacher_sem = False
    args.log_unified_ppo_samples = True
    args.val_dataset_path = None
    args.eval_every_steps = 0
    args.enable_stable_early_stop = False
    args.save_best_checkpoint = False
    args.skip_tokenizer_checkpoint = True
    args.ppo_learning_rate = 1e-6
    args.ppo_clip_range = 0.05
    args.stable_ppo_epochs = 1
    args.max_grad_norm = 0.5
    args.target_kl = 0.3
    args.hard_kl = 0.8
    args.enable_adaptive_kl = True
    args.reward_clip_min = -5.0
    args.reward_clip_max = 5.0
    args.normalize_reward = False
    args.normalize_advantage = False
    args.max_steps = args.updates
    args.max_prompt_examples = args.parent_count
    args.save_steps = args.updates
    args.batch_size = min(max(1, int(args.batch_size)), 4)


def _bind_checkpoint_serialization_paths(
    *,
    tokenizer: Any,
    policy_model: Any,
    requested_model: Path,
) -> None:
    """Keep held load paths out of durable HF/PEFT checkpoint metadata."""

    lexical_model = str(requested_model)
    if "/proc/self/fd/" in lexical_model or not requested_model.is_absolute():
        raise RuntimeError("Taste T6 source-model serialization path is invalid")
    if not hasattr(tokenizer, "name_or_path") or type(tokenizer.init_kwargs) is not dict:
        raise RuntimeError("Taste T6 tokenizer cannot freeze its serialization identity")
    tokenizer.name_or_path = lexical_model
    tokenizer.init_kwargs["name_or_path"] = lexical_model
    peft_configs = getattr(policy_model, "peft_config", None)
    if not isinstance(peft_configs, Mapping) or not peft_configs:
        raise RuntimeError("Taste T6 policy has no PEFT serialization authority")
    for config in peft_configs.values():
        if not hasattr(config, "base_model_name_or_path"):
            raise RuntimeError("Taste T6 PEFT config lacks base-model identity")
        config.base_model_name_or_path = lexical_model
    peft_base = getattr(policy_model, "base_model", None)
    if peft_base is None or not hasattr(peft_base, "name_or_path"):
        raise RuntimeError("Taste T6 PEFT base model lacks serialization identity")
    peft_base.name_or_path = lexical_model
    rebound_model_configs = 0
    queue = [policy_model]
    visited: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        config = getattr(current, "config", None)
        if config is not None:
            for field in ("_name_or_path", "name_or_path"):
                if hasattr(config, field):
                    setattr(config, field, lexical_model)
                    rebound_model_configs += 1
        for field in ("base_model", "model", "pretrained_model"):
            child = getattr(current, field, None)
            if child is not None and child is not current:
                queue.append(child)
    if rebound_model_configs == 0:
        raise RuntimeError("Taste T6 policy has no base-model serialization config")


def _expected_saved_peft_config(
    *,
    policy_model: Any,
    requested_model: Path,
) -> dict[str, Any]:
    configs = getattr(policy_model, "peft_config", None)
    if not isinstance(configs, Mapping) or set(configs) != {"default"}:
        raise RuntimeError("Taste T6 requires exactly one default PEFT adapter")
    config = configs["default"]
    if not hasattr(config, "to_dict"):
        raise RuntimeError("Taste T6 PEFT config cannot be serialized")

    def native(value: Any) -> Any:
        if isinstance(value, set):
            return sorted(native(item) for item in value)
        if isinstance(value, Mapping):
            if not all(type(key) is str for key in value):
                raise RuntimeError("Taste T6 PEFT config has non-string keys")
            return {key: native(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [native(item) for item in value]
        return value

    payload = native(config.to_dict())
    if type(payload) is not dict:
        raise RuntimeError("Taste T6 PEFT config is malformed")
    payload["base_model_name_or_path"] = str(requested_model)
    payload["inference_mode"] = True
    try:
        return json.loads(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Taste T6 PEFT config is not native JSON") from exc


def _assert_terminal_layout(inventory: Mapping[str, Any], *, updates: int) -> None:
    periodic_root = f"checkpoint-{updates}"
    required_files = {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "decoded_chem_value_head.pt",
        f"{periodic_root}/README.md",
        f"{periodic_root}/adapter_config.json",
        f"{periodic_root}/adapter_model.safetensors",
        f"{periodic_root}/decoded_chem_value_head.pt",
        "candidate_pool.jsonl",
        "policy_provenance.json",
        "downstream_policy_binding.json",
        "parent_selection.json",
        "run_manifest.json",
        "observer_state.json",
        "oracle_provenance.json",
        "reward_manifest.json",
        "gate.json",
        "ppo_gate.json",
        "input_hashes.json",
        "manifest.json",
        "ppo_smoke_manifest.json",
        "state.json",
        f"logs/{STAGE}.log",
    }
    observed_files = set(inventory.get("files", {}))
    if observed_files != required_files:
        raise RuntimeError(
            "Taste T6 terminal output layout changed: "
            f"missing={sorted(required_files - observed_files)} "
            f"extra={sorted(observed_files - required_files)}"
        )
    if set(inventory.get("directories", {})) != {"logs", periodic_root}:
        raise RuntimeError("Taste T6 terminal output directory layout changed")


def _jsonl_document_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    ).encode("utf-8")


class _ReportedOutputPathFilter(logging.Filter):
    def __init__(self, runtime_path: Path, reported_path: Path) -> None:
        super().__init__()
        self._runtime = str(runtime_path)
        self._reported = str(reported_path)

    def reported(self, value: Any) -> Any:
        if isinstance(value, Path):
            value = str(value)
        if type(value) is str:
            return value.replace(self._runtime, self._reported)
        if type(value) is tuple:
            return tuple(self.reported(item) for item in value)
        if type(value) is list:
            return [self.reported(item) for item in value]
        if type(value) is dict:
            return {key: self.reported(item) for key, item in value.items()}
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.reported(record.msg)
        record.args = self.reported(record.args)
        return True


def _assert_execution_released() -> dict[str, Any]:
    release = _read_json(T6_RELEASE_CONFIG_PATH, label="T6 release config")
    if set(release) != T6_RELEASE_CONFIG_KEYS:
        raise RuntimeError("TASTE_T6_RELEASE_CONFIG_KEYS_CHANGED")
    if (
        release.get("schema_version") != T6_RELEASE_SCHEMA
        or type(release.get("release_enabled")) is not bool
        or type(release.get("release_state")) is not str
        or type(release.get("gpu_index")) is not int
        or release.get("gpu_index") != 0
    ):
        raise RuntimeError("TASTE_T6_RELEASE_CONFIG_INVALID")
    pinned_fields = (
        "implementation_commit",
        "implementation_tree",
        "external_authority_path",
        "external_authority_sha256",
        "t3_gate_sha256",
        "t4_gate_sha256",
        "t5_gate_sha256",
        "t5_output_inventory_sha256",
        "controller_receipt_sha256",
        "output_parent",
    )
    if release["release_enabled"] is not True:
        if (
            release["release_state"]
            != "RELEASE_DISABLED_PENDING_INTEGRATION_COMMIT_AND_EXTERNAL_AUTHORITY"
            or any(release.get(field) is not None for field in pinned_fields)
        ):
            raise RuntimeError("TASTE_T6_DISABLED_RELEASE_CONFIG_DRIFTED")
        raise RuntimeError("TASTE_T6_EXECUTION_NOT_RELEASED")
    if release["release_state"] != "RELEASED_BY_EXTERNAL_EXECUTION_AUTHORITY":
        raise RuntimeError("TASTE_T6_RELEASE_STATE_INVALID")
    for field in ("implementation_commit",):
        value = release.get(field)
        if (
            type(value) is not str
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"TASTE_T6_RELEASE_{field.upper()}_INVALID")
    implementation_tree = release.get("implementation_tree")
    if (
        type(implementation_tree) is not str
        or len(implementation_tree) != 40
        or any(
            character not in "0123456789abcdef"
            for character in implementation_tree
        )
    ):
        raise RuntimeError("TASTE_T6_RELEASE_IMPLEMENTATION_TREE_INVALID")
    for field in (
        "external_authority_sha256",
        "t3_gate_sha256",
        "t4_gate_sha256",
        "t5_gate_sha256",
        "t5_output_inventory_sha256",
        "controller_receipt_sha256",
    ):
        if not _is_sha256(release.get(field)):
            raise RuntimeError(f"TASTE_T6_RELEASE_{field.upper()}_INVALID")
    for field in ("external_authority_path", "output_parent"):
        _normalized_absolute(release.get(field), label=f"release {field}")
    return release


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_document_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            default=lambda value: str(value) if isinstance(value, Path) else value,
        )
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    assert_no_symlink_components(path, label=label)
    info = os.lstat(path)
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or info.st_size <= 0
    ):
        raise ValueError(f"Taste Ours {label} must be one nonempty physical file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError(f"Taste Ours {label} must contain one JSON object")
    return payload


def _json_from_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    if type(data) is not bytes or not data:
        raise ValueError(f"Taste Ours {label} bytes must be nonempty")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Taste Ours {label} is malformed") from exc
    if type(payload) is not dict:
        raise ValueError(f"Taste Ours {label} must contain one JSON object")
    return payload


def _normalized_absolute(path: str | Path, *, label: str) -> Path:
    requested = Path(path).expanduser()
    normalized = Path(os.path.abspath(requested))
    if not requested.is_absolute() or requested != normalized:
        raise ValueError(f"Taste Ours {label} must be one normalized absolute path")
    return normalized


def _is_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _git_output(*args: str) -> str:
    dot_git = REPO_ROOT / ".git"
    info = os.lstat(dot_git)
    if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
        git_directory = dot_git
    elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size < 4096:
        content = dot_git.read_text(encoding="utf-8").strip()
        prefix = "gitdir: "
        if not content.startswith(prefix) or "\n" in content:
            raise RuntimeError("Taste T6 worktree gitdir authority is malformed")
        candidate = Path(content[len(prefix) :])
        git_directory = (
            candidate
            if candidate.is_absolute()
            else Path(os.path.abspath(REPO_ROOT / candidate))
        )
        assert_no_symlink_components(
            git_directory, label="Taste T6 execution git directory"
        )
        if not stat.S_ISDIR(os.lstat(git_directory).st_mode):
            raise RuntimeError("Taste T6 execution git directory is not physical")
    else:
        raise RuntimeError("Taste T6 execution .git authority is not physical")
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    completed = subprocess.run(
        [
            "/usr/bin/git",
            f"--git-dir={git_directory}",
            f"--work-tree={REPO_ROOT}",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.attributesfile=/dev/null",
            "-c",
            "core.excludesfile=/dev/null",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed.stdout.strip()


def _verify_execution_checkout(release: Mapping[str, Any]) -> dict[str, str]:
    status = _git_output(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--ignored=matching",
    )
    if status:
        raise RuntimeError("Taste T6 execution tree is not completely clean")
    lineage = _git_output("rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(lineage) != 2 or lineage[1] != release["implementation_commit"]:
        raise RuntimeError("Taste T6 execution commit lineage changed")
    parent_tree = _git_output("rev-parse", f"{lineage[1]}^{{tree}}")
    if parent_tree != release["implementation_tree"]:
        raise RuntimeError("Taste T6 implementation tree changed")
    allowed_release_delta = {
        "configs/autodl/tastemolnet_t6_execution_release_v1.json",
        "scripts/autodl/run_tastemolnet_ours_ppo_smoke.sh",
    }
    changed = {
        line
        for line in _git_output(
            "diff",
            "--name-only",
            "--no-renames",
            lineage[1],
            "HEAD",
        ).splitlines()
        if line
    }
    if changed != allowed_release_delta:
        raise RuntimeError("Taste T6 execution commit changed non-release files")
    wrapper = REPO_ROOT / "scripts/autodl/run_tastemolnet_ours_ppo_smoke.sh"
    wrapper_assignments = [
        line.strip()
        for line in wrapper.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("TASTE_T6_WRAPPER_RELEASED=")
    ]
    if wrapper_assignments != ["TASTE_T6_WRAPPER_RELEASED=1"]:
        raise RuntimeError("Taste T6 execution wrapper is not released")
    return {
        "execution_commit": lineage[0],
        "execution_tree": _git_output("rev-parse", "HEAD^{tree}"),
        "implementation_commit": lineage[1],
        "implementation_tree": parent_tree,
    }


def _hold_external_release_authority(
    stack: ExitStack,
    release: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], Any]:
    authority_path = _normalized_absolute(
        release["external_authority_path"],
        label="external release authority",
    )
    authority = stack.enter_context(
        hold_readonly_file(
            authority_path,
            expected_sha256=str(release["external_authority_sha256"]),
        )
    )
    payload = _json_from_bytes(
        authority.read_bytes(),
        label="external release authority",
    )
    expected_keys = {
        "schema_version",
        "stage",
        "dataset",
        "implementation_commit",
        "implementation_tree",
        "t3_gate_sha256",
        "t4_gate_sha256",
        "t5_gate_sha256",
        "t5_output_inventory_sha256",
        "controller_receipt_path",
        "controller_receipt_sha256",
        "gpu_index",
        "gpu_uuid",
        "cuda_visible_devices",
        "output_parent",
        "minimum_persistent_free_gb",
        "minimum_free_after_reservations_gb",
        "max_concurrent_taste_full",
        "gpu0_gpu3_excluded",
        "gnn_ablation_enabled",
        "frozen_oracle_identity",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("Taste T6 external authority keys changed")
    expected_exact = {
        "schema_version": T6_EXTERNAL_AUTHORITY_SCHEMA,
        "stage": STAGE,
        "dataset": "tastemolnet",
        "implementation_commit": release["implementation_commit"],
        "implementation_tree": release["implementation_tree"],
        "t3_gate_sha256": release["t3_gate_sha256"],
        "t4_gate_sha256": release["t4_gate_sha256"],
        "t5_gate_sha256": release["t5_gate_sha256"],
        "t5_output_inventory_sha256": release[
            "t5_output_inventory_sha256"
        ],
        "controller_receipt_sha256": release["controller_receipt_sha256"],
        "gpu_index": 0,
        "cuda_visible_devices": "0",
        "output_parent": release["output_parent"],
        "minimum_persistent_free_gb": 20,
        "minimum_free_after_reservations_gb": 100,
        "max_concurrent_taste_full": 2,
        "gpu0_gpu3_excluded": False,
        "gnn_ablation_enabled": False,
    }
    if any(
        type(payload.get(key)) is not type(value) or payload.get(key) != value
        for key, value in expected_exact.items()
    ):
        raise RuntimeError("Taste T6 external authority values changed")
    gpu_uuid = payload.get("gpu_uuid")
    if type(gpu_uuid) is not str or not gpu_uuid.startswith("GPU-"):
        raise RuntimeError("Taste T6 external authority lacks GPU0 UUID")
    frozen = payload.get("frozen_oracle_identity")
    frozen_keys = {
        "dataset", "backbone", "num_classes", "label_map", "source_label",
        "strict_flip", "rf_oracle_used", "checkpoint_dir", "checkpoint_id",
        "checkpoint_sha256", "checkpoint_inventory_sha256",
        "checkpoint_stat_inventory_sha256", "checkpoint_sha256s_sha256",
        "feature_schema_sha256", "temperature_calibration_sha256",
        "downstream_policy_sha256", "t2_adoption_binding", "t3_output_root",
        "t3_gate_sha256", "t3_root_inventory_sha256", "t4_output_root",
        "t4_gate_sha256", "t4_root_inventory_sha256",
    }
    if type(frozen) is not dict or set(frozen) != frozen_keys:
        raise RuntimeError("Taste T6 frozen-oracle authority schema changed")
    frozen_exact = {
        "dataset": "tastemolnet",
        "backbone": "gine",
        "num_classes": 3,
        "label_map": {"0": "Bitter", "1": "Sweet", "2": "Tasteless"},
        "source_label": 1,
        "strict_flip": "pred_before == 1 and pred_after != 1",
        "rf_oracle_used": False,
        "t3_gate_sha256": release["t3_gate_sha256"],
        "t4_gate_sha256": release["t4_gate_sha256"],
    }
    if any(
        type(frozen.get(key)) is not type(value) or frozen.get(key) != value
        for key, value in frozen_exact.items()
    ):
        raise RuntimeError("Taste T6 frozen-oracle scientific identity changed")
    for key in (
        "checkpoint_id", "checkpoint_sha256", "checkpoint_inventory_sha256",
        "checkpoint_stat_inventory_sha256", "checkpoint_sha256s_sha256",
        "feature_schema_sha256", "temperature_calibration_sha256",
        "downstream_policy_sha256", "t3_root_inventory_sha256",
        "t4_root_inventory_sha256",
    ):
        if not _is_sha256(frozen.get(key)):
            raise RuntimeError(f"Taste T6 frozen-oracle {key} is malformed")
    if frozen["checkpoint_id"] != frozen["checkpoint_sha256"]:
        raise RuntimeError("Taste T6 frozen checkpoint identity changed")
    for key in ("checkpoint_dir", "t3_output_root", "t4_output_root"):
        _normalized_absolute(frozen.get(key), label=f"frozen oracle {key}")
    if type(frozen.get("t2_adoption_binding")) is not dict:
        raise RuntimeError("Taste T6 frozen oracle lacks T2 adoption binding")
    receipt_path = _normalized_absolute(
        payload.get("controller_receipt_path"),
        label="controller receipt",
    )
    receipt = stack.enter_context(
        hold_readonly_file(
            receipt_path,
            expected_sha256=str(payload["controller_receipt_sha256"]),
        )
    )
    if not receipt.read_bytes():
        raise RuntimeError("Taste T6 controller receipt is empty")
    authority.revalidate()
    receipt.revalidate()
    return authority, payload, receipt


def _assert_gpu_runtime(authority: Mapping[str, Any]) -> None:
    expected_uuid = authority["gpu_uuid"]
    if (
        os.environ.get("AUTODL_PHYSICAL_GPU_INDEX") != "0"
        or os.environ.get("AUTODL_PHYSICAL_GPU_UUID") != expected_uuid
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
    ):
        raise RuntimeError("Taste T6 GPU0 environment differs from authority")
    completed = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory: dict[int, str] = {}
    for line in completed.stdout.splitlines():
        index_text, separator, uuid = line.partition(",")
        if not separator:
            raise RuntimeError("Taste T6 GPU inventory is malformed")
        inventory[int(index_text.strip())] = uuid.strip()
    if inventory.get(0) != expected_uuid:
        raise RuntimeError("Taste T6 physical GPU0 UUID changed")


def _checkpoint_and_train_contract(
    checkpoint: Path,
    *,
    frozen_oracle: Mapping[str, Any],
    checkpoint_evidence: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], Path, str, int, dict[str, int]]:
    card = _json_from_bytes(payloads["model_card.json"], label="model card")
    exact = {
        "dataset": "tastemolnet",
        "oracle_backend": "gnn",
        "rf_oracle_used": False,
        "backbone": "gine",
        "num_classes": 3,
        "source_label": 1,
        "profile": "full",
    }
    mismatches = [
        key
        for key, expected in exact.items()
        if type(card.get(key)) is not type(expected) or card.get(key) != expected
    ]
    if mismatches:
        raise ValueError("Taste Ours GINE contract drift: " + ", ".join(mismatches))
    checkpoint_id = hashlib.sha256(payloads["model.pt"]).hexdigest()
    if card.get("checkpoint_id") != checkpoint_id:
        raise ValueError("Taste Ours selected GINE checkpoint identity changed")
    expected_files = {
        "checkpoint_sha256": "model.pt",
        "feature_schema_sha256": "feature_schema.json",
        "temperature_calibration_sha256": "temperature_scaling.json",
    }
    for field, name in expected_files.items():
        expected = frozen_oracle.get(field)
        if type(expected) is not str or hashlib.sha256(payloads[name]).hexdigest() != expected:
            raise ValueError(f"Taste Ours frozen oracle {field} changed")
        if field == "checkpoint_sha256" and checkpoint_id != expected:
            raise ValueError("Taste Ours frozen checkpoint hash changed")
    split = _json_from_bytes(payloads["split_manifest.json"], label="split manifest")
    roles = split.get("roles")
    files = split.get("files")
    if (
        set(split)
        != {
            "schema_version",
            "dataset",
            "roles",
            "files",
            "train_manifest",
            "validation_manifest",
            "calibration_loaded_for_training",
            "test_loaded_for_training",
            "test_evaluated_during_training",
            "test_used_for_checkpoint_selection",
        }
        or split.get("schema_version") != "molecular_gnn_split_manifest_v1"
        or split.get("dataset") != "tastemolnet"
        or type(roles) is not dict
        or roles
        != {
            "train": "model_fitting",
            "validation": "checkpoint_selection_and_temperature_calibration",
            "calibration": "reserved_for_threshold_and_selector_only",
            "test": "frozen_model_final_quality_evaluation",
        }
        or type(files) is not dict
        or set(files) != {"train", "validation", "calibration", "test"}
        or split.get("calibration_loaded_for_training") is not False
        or split.get("test_loaded_for_training") is not False
        or split.get("test_evaluated_during_training") is not False
        or split.get("test_used_for_checkpoint_selection") is not False
    ):
        raise ValueError("Taste Ours split-role authority changed")
    normalized_files: dict[str, tuple[Path, str]] = {}
    for role in ("train", "validation", "calibration", "test"):
        row = files[role]
        if type(row) is not dict or set(row) != {"path", "sha256"}:
            raise ValueError(f"Taste Ours {role} split authority is malformed")
        role_path = _normalized_absolute(row.get("path"), label=f"{role} split")
        role_sha256 = row.get("sha256")
        if not _is_sha256(role_sha256):
            raise ValueError(f"Taste Ours {role} split SHA-256 is malformed")
        normalized_files[role] = (role_path, role_sha256)
    train = files["train"]
    expected_train = _normalized_absolute(train.get("path"), label="train split")
    train_sha256 = train.get("sha256")
    train_manifest = split.get("train_manifest")
    if (
        type(train_manifest) is not dict
        or set(train_manifest)
        != {
            "schema_version",
            "num_records",
            "num_classes",
            "label_counts",
            "split_counts",
            "source_path",
            "source_sha256",
            "dataset_fingerprint",
            "feature_schema_sha256",
        }
        or train_manifest.get("schema_version") != "molecular_graph_dataset_v1"
        or type(train_manifest.get("num_records")) is not int
        or train_manifest["num_records"] <= 0
        or type(train_manifest.get("num_classes")) is not int
        or train_manifest["num_classes"] != 3
        or type(train_manifest.get("label_counts")) is not dict
        or set(train_manifest["label_counts"]) != {"0", "1", "2"}
        or any(
            type(train_manifest["label_counts"].get(label)) is not int
            or train_manifest["label_counts"][label] <= 0
            for label in ("0", "1", "2")
        )
        or sum(train_manifest["label_counts"].values())
        != train_manifest["num_records"]
        or train_manifest.get("split_counts")
        != {"train": train_manifest["num_records"]}
        or train_manifest.get("source_path") != str(expected_train)
        or train_manifest.get("source_sha256") != train_sha256
        or not _is_sha256(train_manifest.get("dataset_fingerprint"))
        or not _is_sha256(train_manifest.get("feature_schema_sha256"))
    ):
        raise ValueError("Taste Ours train manifest authority changed")
    feature_schema = _json_from_bytes(
        payloads["feature_schema.json"], label="feature schema"
    )
    if train_manifest["feature_schema_sha256"] != feature_schema.get("schema_sha256"):
        raise ValueError("Taste Ours train/GINE feature schema authority changed")
    temperature = _json_from_bytes(
        payloads["temperature_scaling.json"], label="temperature scaling"
    )
    if (
        type(temperature.get("temperature")) is not float
        or not math.isfinite(temperature["temperature"])
        or temperature["temperature"] <= 0.0
        or temperature.get("selection_split") != "validation"
        or temperature.get("test_used_for_fit") is not False
    ):
        raise ValueError("Taste Ours frozen temperature authority changed")
    label_map = _json_from_bytes(payloads["label_map.json"], label="label map")
    if label_map != {"0": "Bitter", "1": "Sweet", "2": "Tasteless"}:
        raise ValueError("Taste Ours three-class label map changed")
    test_status = _json_from_bytes(
        payloads["test_evaluation_status.json"], label="test status"
    )
    if (
        set(test_status)
        != {"schema_version", "status", "test_loaded", "reason", "path", "sha256"}
        or test_status.get("schema_version")
        != "molecular_gnn_test_evaluation_status_v1"
        or test_status.get("status") != "NOT_EVALUATED"
        or test_status.get("test_loaded") is not False
        or test_status.get("reason") != "held_out_until_frozen_final_evaluation"
        or test_status.get("path") != str(normalized_files["test"][0])
        or test_status.get("sha256") != normalized_files["test"][1]
    ):
        raise ValueError("Taste Ours frozen test authority changed")
    contract = {
        "schema_version": "tastemolnet_ours_train_contract_v1",
        "checkpoint_dir": str(checkpoint),
        "checkpoint_id": checkpoint_id,
        "checkpoint_inventory_sha256": checkpoint_evidence[
            "checkpoint_inventory_sha256"
        ],
        "checkpoint_stat_inventory_sha256": checkpoint_evidence[
            "checkpoint_stat_inventory_sha256"
        ],
        "checkpoint_sha256s_sha256": checkpoint_evidence[
            "checkpoint_sha256s_sha256"
        ],
        "split_manifest": str(checkpoint / "split_manifest.json"),
        "split_manifest_sha256": hashlib.sha256(
            payloads["split_manifest.json"]
        ).hexdigest(),
        "train_csv": str(expected_train),
        "train_csv_sha256": train_sha256,
        "train_num_records": train_manifest["num_records"],
        "train_label_counts": dict(train_manifest["label_counts"]),
        "train_loaded": True,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    return (
        contract,
        expected_train,
        train_sha256,
        train_manifest["num_records"],
        dict(train_manifest["label_counts"]),
    )


def _ensure_disjoint(output: Path, inputs: list[Path]) -> None:
    for source in inputs:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"Taste Ours output overlaps immutable input: {source}")


def _fresh_output(path: Path, *, inputs: list[Path]) -> FreshOutputDirectory:
    requested = path.expanduser()
    unresolved = Path(os.path.abspath(requested))
    if (
        not requested.is_absolute()
        or requested != unresolved
        or unresolved.name in {"", ".", ".."}
    ):
        raise ValueError("Taste Ours output path is not one normalized absolute child")
    _ensure_disjoint(unresolved, inputs)
    return FreshOutputDirectory.create(unresolved)


@dataclass(slots=True)
class _HeldT6CleanBase:
    """Direct T6 consumer for the independently verified managed T5 base."""

    managed: Any
    source: Any
    evidence: Mapping[str, Any]
    managed_evidence: Mapping[str, Any]

    def source_load_path(self) -> Path:
        self.revalidate()
        return self.source.stable_load_path()

    def revalidate(self) -> dict[str, Any]:
        if self.managed.revalidate() != dict(self.managed_evidence):
            raise RuntimeError("Taste T6 managed T5 final changed")
        source = self.source.revalidate()
        if (
            source.get("source_model_path")
            != self.evidence.get("source_model_path")
            or source.get("source_model_inventory_sha256")
            != self.evidence.get("source_model_inventory_sha256")
        ):
            raise RuntimeError("Taste T6 held clean-base source changed")
        return dict(self.evidence)

    def close(self) -> None:
        self.source.close()
        self.managed.close()

    def __enter__(self) -> "_HeldT6CleanBase":
        self.revalidate()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _hold_t6_clean_base(root: Path) -> _HeldT6CleanBase:
    """Adopt T5's clean generic base without claiming an adapter or training."""

    from src.train.tastemolnet_clean_policy import (
        hold_source_model_for_clean_policy,
    )
    from src.utils.managed_final_consumer_v2 import hold_verified_managed_final

    required = (
        "artifacts/clean_base_adoption_candidate.json",
        "artifacts/source_inventory.json",
    )
    managed = hold_verified_managed_final(root, required_relative_paths=required)
    source = None
    try:
        verification = managed.verification_payload.get("verification")
        if type(verification) is not dict:
            raise RuntimeError("Taste T6 managed T5 verification body is absent")
        exact = {
            "schema_version": "tastemolnet_t5_clean_base_adoption_v2",
            "status": "PASS",
            "stage": "T5_CLEAN_POLICY_READY",
            "task_id": "T5_CLEAN_BASE_ADOPTION",
            "marker": "[TASTE_T5_CLEAN_SFT_PASS]",
            "semantic_state": "ADOPTED_CLEAN_GENERIC_BASE",
            "independent_scientific_verifier": True,
            "downstream_clean_base_authority": True,
            "optimizer_steps": 0,
            "training_performed": False,
            "taste_splits_loaded": [],
            "validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "rf_reference_count": 0,
            "gnn_reward_used": False,
            "source_adapter_present": False,
            "peft_present": False,
            "source_weights_copied": False,
            "matrix_method_cell": False,
        }
        if any(
            type(verification.get(key)) is not type(value)
            or verification.get(key) != value
            for key, value in exact.items()
        ):
            raise RuntimeError("Taste T6 managed T5 scientific contract changed")
        source_path = _normalized_absolute(
            verification.get("source_model_path"), label="T5 clean source"
        )
        source_inventory = verification.get("source_model_inventory_sha256")
        if not _is_sha256(source_inventory):
            raise RuntimeError("Taste T6 managed T5 source inventory is malformed")
        candidate = _json_from_bytes(
            managed.file(required[0]).read_bytes(), label="T5 clean candidate"
        )
        source_receipt = _json_from_bytes(
            managed.file(required[1]).read_bytes(), label="T5 source inventory"
        )
        if (
            candidate.get("source_model_path") != str(source_path)
            or source_receipt.get("source_model_path") != str(source_path)
            or candidate.get("source_model_inventory_sha256") != source_inventory
            or source_receipt.get("source_model_inventory_sha256")
            != source_inventory
            or candidate.get("optimizer_steps") != 0
            or candidate.get("training_performed") is not False
            or candidate.get("source_adapter_present") is not False
        ):
            raise RuntimeError("Taste T6 managed T5 source/candidate binding changed")
        managed_evidence = dict(managed.revalidate())
        source = hold_source_model_for_clean_policy(source_path, source_inventory)
        evidence = {
            "schema_version": "tastemolnet_t6_clean_base_adoption_v1",
            "status": "PASS",
            "stage": "T5_CLEAN_POLICY_READY",
            "semantic_state": "ADOPTED_CLEAN_GENERIC_BASE",
            "output_root": str(root),
            "source_model_dir": str(source_path),
            "source_model_path": str(source_path),
            "source_model_inventory_sha256": source_inventory,
            "t5_gate_sha256": managed_evidence["gate_sha256"],
            "t5_output_inventory_sha256": managed_evidence[
                "published_inventory_sha256"
            ],
            "optimizer_step_count": 0,
            "training_performed": False,
            "taste_splits_loaded": [],
            "source_adapter_present": False,
            "adapter_materialization": "T6_RUNTIME_IN_MEMORY_ZERO_STEP_LORA",
        }
        held = _HeldT6CleanBase(
            managed=managed,
            source=source,
            evidence=evidence,
            managed_evidence=managed_evidence,
        )
        held.revalidate()
        return held
    except BaseException:
        if source is not None:
            source.close()
        managed.close()
        raise


def _build_zero_step_lora_model(
    deps: Mapping[str, Any],
    *,
    model_path: Path,
    lexical_model_path: Path,
    seed: int,
    is_trainable: bool,
) -> Any:
    """Materialize the reviewed zero-step LoRA in memory from managed T5."""

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - AutoDL dependency
        raise RuntimeError("Taste T6 zero-step LoRA requires PEFT") from exc
    deps["set_seed"](seed)
    base = build_quantized_base_model(
        dict(deps),
        model_path=model_path,
        trust_remote_code=True,
        local_files_only=True,
        prepare_for_training=is_trainable,
    )
    if getattr(base, "peft_config", None):
        raise RuntimeError("Taste T6 managed clean base unexpectedly contains PEFT")
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["wqkv", "wo", "w1", "w2", "w3"],
    )
    model = get_peft_model(base, config)
    for item in model.peft_config.values():
        item.base_model_name_or_path = str(lexical_model_path)
    if not is_trainable:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    return model


def _stage_authorities(
    stack: ExitStack,
    t5_root: Path,
    *,
    frozen: Mapping[str, Any],
    downstream_policy: Path,
    base_policy: Path,
) -> tuple[Any, Any, Any]:
    from src.train.tastemolnet_clean_policy_init import (
        hold_taste_managed_evidence_binding_v2,
    )
    from src.utils.tastemolnet_downstream_policy import (
        load_tastemolnet_downstream_policy,
    )

    t5_load = stack.enter_context(_hold_t6_clean_base(t5_root))
    managed_binding = hold_taste_managed_evidence_binding_v2(frozen)
    stack.callback(managed_binding.close)
    policy = stack.enter_context(
        load_tastemolnet_downstream_policy(
            downstream_policy, base_policy_path=base_policy
        )
    )
    stage_contract = policy.stage(STAGE)
    exact_stage = {
        "mode": "train_only_frozen_gine_reward_ppo_smoke",
        "device": "cuda:0",
        "physical_gpu_index": 0,
        "gpu_uuid_binding_required": True,
        "fresh_output_required": True,
        "frozen_gine_reward_required": True,
        "minimum_optimizer_steps": 5,
        "num_classes": 3,
        "source_label": 1,
        "rf_oracle_used": False,
        "run": 1,
        "split_payload_access": {
            "train": True,
            "validation": False,
            "calibration": False,
            "test": False,
        },
    }
    if any(
        type(stage_contract.get(key)) is not type(value)
        or stage_contract.get(key) != value
        for key, value in exact_stage.items()
    ):
        raise ValueError("Taste Ours downstream stage authority changed")
    allowed = stage_contract.get("allowed_input_files")
    if type(allowed) is not list or allowed != [
        "immutable_t2_bundle",
        "immutable_t3_stage_output",
        "immutable_t4_stage_output",
        "immutable_t5_clean_policy",
        "frozen_train_csv",
    ]:
        raise ValueError("Taste Ours allowed-input authority changed")
    policy_binding = policy.revalidate(stage=STAGE)
    held_frozen = managed_binding.revalidate().evidence()
    if held_frozen != dict(frozen):
        raise ValueError("Taste T6 managed-v2 frozen oracle authority changed")
    if held_frozen.get("downstream_policy_sha256") != policy.file_sha256:
        raise ValueError("Taste T6 frozen oracle/downstream policy binding changed")
    t5_load.revalidate()
    if policy.revalidate(stage=STAGE) != policy_binding:
        raise ValueError("Taste T6 downstream policy changed while held")
    managed_binding.revalidate()
    return managed_binding, t5_load, policy


def run(args: Any) -> int:
    release = _assert_execution_released()
    _configure(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    checkpoint = _normalized_absolute(args.gnn_checkpoint, label="GINE checkpoint")
    requested_train_csv = _normalized_absolute(
        args.dataset_path, label="train split argument"
    )
    requested_model = _normalized_absolute(args.model_path, label="source model")
    with ExitStack() as stack:
        execution_identity = _verify_execution_checkout(release)
        release_authority, execution_authority, controller_receipt = (
            _hold_external_release_authority(stack, release)
        )
        _assert_gpu_runtime(execution_authority)
        frozen = execution_authority["frozen_oracle_identity"]
        managed_oracle, t5_load, policy = _stage_authorities(
            stack,
            args.t5_output,
            frozen=frozen,
            downstream_policy=args.downstream_policy,
            base_policy=args.base_policy,
        )
        checkpoint_authority = managed_oracle.checkpoint
        t5_evidence = t5_load.revalidate()
        if requested_model != _normalized_absolute(
            t5_evidence["source_model_path"], label="T5 source model"
        ):
            raise ValueError("Taste Ours base model differs from T5 authority")
        expected_runtime = {
            "t3_gate_sha256": release["t3_gate_sha256"],
            "t4_gate_sha256": release["t4_gate_sha256"],
            "t5_gate_sha256": release["t5_gate_sha256"],
            "t5_output_inventory_sha256": release[
                "t5_output_inventory_sha256"
            ],
        }
        observed_runtime = {
            "t3_gate_sha256": frozen.get("t3_gate_sha256"),
            "t4_gate_sha256": frozen.get("t4_gate_sha256"),
            "t5_gate_sha256": t5_evidence.get("t5_gate_sha256"),
            "t5_output_inventory_sha256": t5_evidence.get(
                "t5_output_inventory_sha256"
            ),
        }
        if observed_runtime != expected_runtime:
            raise ValueError("Taste Ours predecessor release pins differ")
        checkpoint_payload_names = (
            "model.pt",
            "model_card.json",
            "feature_schema.json",
            "label_map.json",
            "split_manifest.json",
            "test_evaluation_status.json",
            "temperature_scaling.json",
        )
        checkpoint_payloads = {
            name: checkpoint_authority.read_frozen_gine_payload(name)
            for name in checkpoint_payload_names
        }
        checkpoint_evidence = checkpoint_authority.revalidate()
        (
            train_contract,
            train_csv,
            train_csv_sha256,
            train_num_records,
            train_label_counts,
        ) = _checkpoint_and_train_contract(
            checkpoint,
            frozen_oracle=frozen,
            checkpoint_evidence=checkpoint_evidence,
            payloads=checkpoint_payloads,
        )
        if requested_train_csv != train_csv:
            raise ValueError("Taste Ours train argument differs from frozen manifest")
        train_authority = stack.enter_context(
            hold_readonly_file(train_csv, expected_sha256=train_csv_sha256)
        )
        prompt_pool_bound = max(256, int(args.parent_count) * 16)
        source_pool, train_prompt_evidence = load_tastemolnet_train_prompts(
            train_authority.read_bytes(),
            expected_num_records=train_num_records,
            expected_label_counts=train_label_counts,
            max_prompt_examples=prompt_pool_bound,
        )
        checkpoint_authority.revalidate()
        train_authority.revalidate()
        input_paths = [
            Path(frozen["t2_adoption_binding"]["adoption_root"]),
            checkpoint,
            train_csv,
            Path(t5_evidence["output_root"]),
            Path(t5_evidence["source_model_path"]),
            Path(frozen["t3_output_root"]),
            Path(frozen["t4_output_root"]),
            release_authority.path,
            controller_receipt.path,
            _normalized_absolute(
                args.downstream_policy, label="downstream policy input"
            ),
            _normalized_absolute(args.base_policy, label="base policy input"),
        ]
        git_commit = execution_identity["execution_commit"]
        requested_output = _normalized_absolute(args.output_dir, label="output root")
        expected_output_parent = _normalized_absolute(
            execution_authority["output_parent"],
            label="released output parent",
        )
        if requested_output.parent != expected_output_parent:
            raise RuntimeError("Taste Ours output is outside the released private parent")
        output_authority = _fresh_output(requested_output, inputs=input_paths)
        stack.callback(output_authority.close)
        output = output_authority.stable_path
        # The shared stable loop resolves an explicitly supplied candidate-pool
        # path.  Leaving it unset makes the loop derive the file from the held
        # output_dir and preserves the descriptor-backed authority.
        args.candidate_pool_path = None
        logger = configure_run_logger(
            "train_tastemolnet_gnn_ppo",
            context=RunContext(
                run_name="tastemolnet_t6_ours_smoke",
                output_dir=output,
                stage=STAGE,
                seed=int(args.seed),
            ),
            log_dir=output / "logs",
        )
        reported_output = _ReportedOutputPathFilter(output, requested_output)
        logger.addFilter(reported_output)
        logger.info("Runtime environment: %s", collect_runtime_environment_debug())
        logger.info(
            "[TASTE_OURS_BOUNDARY] stage=%s stable_loop=run_stable_decoded_chem_ppo_loop "
            "dataset=tastemolnet num_classes=3 source_label=1 oracle_backend=gnn "
            "rf_oracle_used=false train_loaded=true validation_loaded=false "
            "calibration_loaded=false test_loaded=false",
            STAGE,
        )
        stable_config = resolve_stable_config(args)
        deps = import_training_dependencies()
        deps["set_seed"](int(args.seed))
        torch = deps["torch"]
        if (
            args.gnn_device != "cuda"
            or not bool(torch.cuda.is_available())
            or int(torch.cuda.device_count()) != 1
            or int(torch.cuda.current_device()) != 0
        ):
            raise RuntimeError(
                "Taste T6 requires exactly one visible logical CUDA device 0"
            )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        source_load_path = t5_load.source_load_path()
        tokenizer = build_tokenizer(
            deps,
            model_path=source_load_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )
        t5_load.revalidate()
        adapter = "T6_RUNTIME_IN_MEMORY_ZERO_STEP_LORA"
        policy_model = _build_zero_step_lora_model(
            deps,
            model_path=source_load_path,
            lexical_model_path=requested_model,
            seed=int(args.seed),
            is_trainable=True,
        )
        reference_model = _build_zero_step_lora_model(
            deps,
            model_path=source_load_path,
            lexical_model_path=requested_model,
            seed=int(args.seed),
            is_trainable=False,
        )
        initial_policy_adapter = adapter_parameter_identity_from_model(policy_model)[
            "parameter_sha256"
        ]
        initial_reference_adapter = adapter_parameter_identity_from_model(
            reference_model
        )["parameter_sha256"]
        if initial_policy_adapter != initial_reference_adapter:
            raise RuntimeError("Taste T6 zero-step policy/reference LoRA differ")
        policy_initializer_hash = _canonical_sha256(
            {
                "schema_version": "tastemolnet_t6_runtime_initializer_v1",
                "managed_t5_inventory_sha256": t5_evidence[
                    "t5_output_inventory_sha256"
                ],
                "source_model_inventory_sha256": t5_evidence[
                    "source_model_inventory_sha256"
                ],
                "adapter_parameter_sha256": initial_policy_adapter,
                "optimizer_step_count": 0,
                "taste_splits_loaded": [],
            }
        )
        reference_policy_hash = _canonical_sha256(
            {
                "schema_version": "tastemolnet_t6_reference_policy_v1",
                "source_model_inventory_sha256": t5_evidence[
                    "source_model_inventory_sha256"
                ],
                "adapter_parameter_sha256": initial_reference_adapter,
                "policy_initializer_hash": policy_initializer_hash,
            }
        )
        policy_t5_identity_before = reference_policy_hash
        reference_t5_identity_before = reference_policy_hash
        t5_evidence = {
            **t5_evidence,
            "frozen_oracle_identity": dict(frozen),
            "policy_initializer_hash": policy_initializer_hash,
            "reference_policy_hash": reference_policy_hash,
            "adapter_parameter_sha256": initial_policy_adapter,
        }
        value_model = build_value_model(
            deps,
            model_path=source_load_path,
            tokenizer=tokenizer,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )
        value_model = ensure_score_head_for_experimental_ppo(
            value_model, "tastemolnet_gnn_stable_value_model"
        )
        _bind_checkpoint_serialization_paths(
            tokenizer=tokenizer,
            policy_model=policy_model,
            requested_model=requested_model,
        )
        expected_adapter_config = _expected_saved_peft_config(
            policy_model=policy_model,
            requested_model=requested_model,
        )
        t5_load.revalidate()
        for authority in (
            managed_oracle,
            t5_load,
            checkpoint_authority,
            train_authority,
            release_authority,
            controller_receipt,
        ):
            authority.revalidate()
        policy_binding = policy.revalidate(stage=STAGE)
        rollout_device = _infer_single_training_device(
            logger=logger,
            torch=torch,
            policy_model=policy_model,
            reference_model=reference_model,
            value_model=value_model,
        )
        if str(rollout_device) not in {"cuda", "cuda:0"}:
            raise RuntimeError("Taste T6 models are not colocated on logical cuda:0")
        policy_before = adapter_parameter_identity_from_model(policy_model)[
            "parameter_sha256"
        ]
        reference_before = model_parameter_hash(reference_model)
        reward_adapter = BatchedGNNPPORewardAdapter.from_payloads(
            checkpoint_payloads,
            checkpoint_dir=checkpoint,
            device=rollout_device if args.gnn_device == "cuda" else args.gnn_device,
            policy_initializer_hash=t5_evidence["policy_initializer_hash"],
            reference_policy_hash=reference_policy_hash,
            config=GNNPPORewardConfig(
                dataset="tastemolnet",
                num_classes=3,
                source_label=1,
                oracle_batch_size=int(args.oracle_batch_size),
            ),
        )
        parent_records = reward_adapter.predict_parent_records(
            parent_smiles=[example.parent_smiles for example in source_pool],
            metas=[
                {
                    "molecule_id": example.molecule_id,
                    "index": example.index,
                }
                for example in source_pool
            ],
        )
        selected_examples = [
            example
            for example, record in zip(source_pool, parent_records, strict=True)
            if type(record.get("predicted_label")) is int
            and record.get("predicted_label") == 1
        ][: int(args.parent_count)]
        if len(selected_examples) != int(args.parent_count):
            raise ValueError(
                "Taste Ours train-only pool lacks the requested predicted-Sweet parents"
            )
        dataset = build_hf_dataset(deps, tokenizer, selected_examples)
        actual_batch_size = max(1, min(int(args.batch_size), len(dataset)))
        parent_selection = {
            "schema_version": "tastemolnet_ours_parent_selection_v1",
            "source_split": "train",
            "true_label": 1,
            "required_pred_before": 1,
            "pool_count": len(source_pool),
            "selected_count": len(selected_examples),
            "selected_parent_ids": [
                str(example.molecule_id or example.index) for example in selected_examples
            ],
            "prediction_counts": {
                str(label): sum(
                    1 for row in parent_records if row.get("predicted_label") == label
                )
                for label in (0, 1, 2)
            },
            "validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "train_prompt_evidence": train_prompt_evidence,
        }
        observer = TastePPOObserver()
        atomic_json(output / "policy_provenance.json", t5_evidence)
        atomic_json(output / "downstream_policy_binding.json", policy_binding)
        atomic_json(output / "parent_selection.json", parent_selection)
        run_manifest = {
            "schema_version": "tastemolnet_ours_ppo_run_v1",
            "stage": STAGE,
            "git_commit": git_commit,
            "stable_loop": "scripts.train_ppo_stable.run_stable_decoded_chem_ppo_loop",
            "shared_algorithm_reimplemented": False,
            "model_path": str(requested_model),
            "policy_initializer": str(adapter),
            "policy_initializer_hash": t5_evidence["policy_initializer_hash"],
            "adapter_config_authority": expected_adapter_config,
            "gnn_checkpoint": str(checkpoint),
            "checkpoint_id": train_contract["checkpoint_id"],
            "dataset_path": str(train_csv),
            "train_contract": train_contract,
            "parent_selection": parent_selection,
            "stable_config": asdict(stable_config),
            "num_classes": 3,
            "source_label": 1,
            "rf_oracle_used": False,
            "validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "tokenizer_checkpoint_saved": False,
        }
        atomic_json(output / "run_manifest.json", run_manifest)
        run_stable_decoded_chem_ppo_loop(
            deps=deps,
            args=args,
            stable_config=stable_config,
            actual_batch_size=actual_batch_size,
            policy_model=policy_model,
            reference_model=reference_model,
            value_model=value_model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            stable_reward_wrapper=reward_adapter,
            output_dir=output,
            logger=logger,
            run_observer=observer,
        )
        observer_state = reported_output.reported(observer.state_dict())
        if type(observer_state) is not dict:
            raise RuntimeError("Taste T6 observer state is not one JSON object")
        atomic_json(output / "observer_state.json", observer_state)
        policy_after = adapter_parameter_identity_from_model(policy_model)[
            "parameter_sha256"
        ]
        if not hasattr(value_model, "v_head"):
            raise RuntimeError("Taste T6 value model lost its reviewed value head")
        value_head_after = value_head_parameter_identity(
            value_model.v_head.state_dict()
        )["parameter_sha256"]
        reference_after = model_parameter_hash(reference_model)
        if (
            adapter_parameter_identity_from_model(reference_model)[
                "parameter_sha256"
            ]
            != initial_reference_adapter
        ):
            raise RuntimeError("Taste T6 frozen reference LoRA changed")
        reference_t5_identity_after = reference_policy_hash
        t5_load.revalidate()
        final_reload = validate_taste_adapter_checkpoint_reload(
            output,
            checkpoint_path_is_retained=True,
            checkpoint_display_path=requested_output,
            expected_base_model_path=requested_model,
            expected_adapter_config=expected_adapter_config,
        )
        periodic_reload = validate_taste_adapter_checkpoint_reload(
            output / f"checkpoint-{args.updates}",
            checkpoint_path_is_retained=True,
            checkpoint_display_path=requested_output / f"checkpoint-{args.updates}",
            expected_base_model_path=requested_model,
            expected_adapter_config=expected_adapter_config,
        )
        rows = observer.captured_candidate_rows()
        candidate_bytes = _jsonl_document_bytes(rows)
        if not rows or (output / "candidate_pool.jsonl").read_bytes() != candidate_bytes:
            raise RuntimeError(
                "Taste T6 candidate pool differs from real in-memory rollouts"
            )
        oracle = reward_adapter.provenance()
        reward = build_taste_reward_manifest(rows, oracle_provenance=oracle)
        t2_binding = frozen["t2_adoption_binding"]
        t2_binding_sha256 = _canonical_sha256(t2_binding)
        gate = build_taste_smoke_gate(
            policy_parameter_hash_before=policy_before,
            policy_parameter_hash_after=policy_after,
            reference_parameter_hash_before=reference_before,
            reference_parameter_hash_after=reference_after,
            policy_t5_identity_before=policy_t5_identity_before,
            reference_t5_identity_before=reference_t5_identity_before,
            reference_t5_identity_after=reference_t5_identity_after,
            expected_t5_reference_policy_hash=reference_policy_hash,
            observer=observer,
            checkpoint_reload=final_reload,
            periodic_checkpoint_reload=periodic_reload,
            reward_manifest=reward,
            oracle_provenance=oracle,
            policy_initializer_hash=t5_evidence["policy_initializer_hash"],
            t2_adoption_gate_sha256=t2_binding["gate_sha256"],
            t2_adoption_receipt_sha256=t2_binding["receipt_sha256"],
            t2_adoption_binding_sha256=t2_binding_sha256,
            t3_gate_sha256=frozen["t3_gate_sha256"],
            t4_gate_sha256=frozen["t4_gate_sha256"],
            value_head_parameter_sha256=value_head_after,
        )
        atomic_json(output / "oracle_provenance.json", oracle)
        atomic_json(output / "reward_manifest.json", reward)
        atomic_json(output / "gate.json", gate)
        atomic_json(output / "ppo_gate.json", gate)
        input_hashes = {
            "schema_version": "tastemolnet_ours_ppo_input_hashes_v1",
            "t2_adoption_binding": t2_binding,
            "t2_adoption_gate_sha256": t2_binding["gate_sha256"],
            "t2_adoption_receipt_sha256": t2_binding["receipt_sha256"],
            "t2_adoption_binding_sha256": t2_binding_sha256,
            "t3_gate_sha256": frozen["t3_gate_sha256"],
            "t4_gate_sha256": frozen["t4_gate_sha256"],
            "t5_gate_sha256": t5_evidence["t5_gate_sha256"],
            "t5_output_inventory_sha256": t5_evidence[
                "t5_output_inventory_sha256"
            ],
            "policy_initializer_hash": t5_evidence["policy_initializer_hash"],
            "source_model_inventory_sha256": t5_evidence[
                "source_model_inventory_sha256"
            ],
            "downstream_policy_sha256": policy.file_sha256,
            "base_policy_sha256": policy.base_policy.file_sha256,
            "checkpoint_id": train_contract["checkpoint_id"],
            "train_csv_sha256": train_contract["train_csv_sha256"],
            "parent_selection_sha256": _canonical_sha256(parent_selection),
        }
        atomic_json(output / "input_hashes.json", input_hashes)
        manifest = {
            **gate,
            "schema_version": OUTPUT_SCHEMA,
            "git_commit": git_commit,
            "gnn_checkpoint": str(checkpoint),
            "checkpoint_id": train_contract["checkpoint_id"],
            "policy_initializer": str(adapter),
            "policy_initializer_hash": t5_evidence["policy_initializer_hash"],
            "policy_checkpoint_hash": final_reload["policy_checkpoint_hash"],
            "candidate_pool": str(requested_output / "candidate_pool.jsonl"),
            "candidate_pool_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "reward_manifest": str(requested_output / "reward_manifest.json"),
            "reward_manifest_sha256": sha256_file(output / "reward_manifest.json"),
            "oracle_provenance": str(requested_output / "oracle_provenance.json"),
            "oracle_provenance_sha256": sha256_file(
                output / "oracle_provenance.json"
            ),
            "output_root": str(requested_output),
            "stable_loop": "scripts.train_ppo_stable.run_stable_decoded_chem_ppo_loop",
            "shared_algorithm_reimplemented": False,
            "no_dataset_redistribution": True,
        }
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "ppo_smoke_manifest.json", manifest)
        state = {
            "schema_version": "tastemolnet_main_stage_state_v1",
            "stage": STAGE,
            "state": gate["status"],
            "status": gate["status"],
            "science_started": True,
            "optimizer_step_count": gate["optimizer_step_count"],
            "checkpoint_id": train_contract["checkpoint_id"],
            "output_root": str(requested_output),
        }
        atomic_json(output / "state.json", state)
        for authority in (
            managed_oracle,
            t5_load,
            checkpoint_authority,
            train_authority,
            release_authority,
            controller_receipt,
        ):
            authority.revalidate()
        policy.revalidate(stage=STAGE)
        if gate["status"] != "PASS":
            atomic_json(output / "FAIL.json", manifest)
            return 2

        expected_documents = {
            "policy_provenance.json": t5_evidence,
            "downstream_policy_binding.json": policy_binding,
            "parent_selection.json": parent_selection,
            "run_manifest.json": run_manifest,
            "observer_state.json": observer_state,
            "oracle_provenance.json": oracle,
            "reward_manifest.json": reward,
            "gate.json": gate,
            "ppo_gate.json": gate,
            "input_hashes.json": input_hashes,
            "manifest.json": manifest,
            "ppo_smoke_manifest.json": manifest,
            "state.json": state,
        }

        for handler in list(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
        prepared = prepare_terminal_output(
            output_authority,
            marker_name="PASS",
            marker_payload=f"{TASTE_PPO_MARKER}\n".encode("utf-8"),
        )
        periodic_root = f"checkpoint-{args.updates}"
        _assert_terminal_layout(prepared.tree.inventory, updates=args.updates)
        prepared.tree.reject_byte_sequence(
            b"/proc/self/fd/",
            suffixes=("",),
        )

        def retained_input_closure() -> None:
            _assert_gpu_runtime(execution_authority)
            if _verify_execution_checkout(release) != execution_identity:
                raise RuntimeError("Taste T6 execution identity drifted at PASS boundary")
            for authority in (
                managed_oracle,
                t5_load,
                checkpoint_authority,
                train_authority,
                release_authority,
                controller_receipt,
            ):
                authority.revalidate()
            t5_load.revalidate()
            policy.revalidate(stage=STAGE)
            observed_final_reload = validate_taste_adapter_checkpoint_reload(
                output,
                checkpoint_path_is_retained=True,
                checkpoint_display_path=requested_output,
                expected_base_model_path=requested_model,
                expected_adapter_config=expected_adapter_config,
            )
            observed_periodic_reload = validate_taste_adapter_checkpoint_reload(
                output / f"checkpoint-{args.updates}",
                checkpoint_path_is_retained=True,
                checkpoint_display_path=requested_output
                / f"checkpoint-{args.updates}",
                expected_base_model_path=requested_model,
                expected_adapter_config=expected_adapter_config,
            )
            if (
                observed_final_reload != final_reload
                or observed_periodic_reload != periodic_reload
            ):
                raise RuntimeError("Taste T6 checkpoint closure drifted before PASS")
            if prepared.tree.read_bytes("candidate_pool.jsonl") != candidate_bytes:
                raise RuntimeError("Taste T6 candidate bytes drifted before PASS")
            if build_taste_reward_manifest(rows, oracle_provenance=oracle) != reward:
                raise RuntimeError("Taste T6 candidate/reward closure drifted before PASS")
            for name, payload in expected_documents.items():
                if (output / name).read_bytes() != _json_document_bytes(payload):
                    raise RuntimeError(
                        f"Taste T6 terminal document drifted before PASS: {name}"
                    )
            if (
                hashlib.sha256(candidate_bytes).hexdigest()
                != manifest["candidate_pool_sha256"]
                or sha256_file(output / "reward_manifest.json")
                != manifest["reward_manifest_sha256"]
                or sha256_file(output / "oracle_provenance.json")
                != manifest["oracle_provenance_sha256"]
            ):
                raise RuntimeError("Taste T6 terminal hash cross-links drifted before PASS")
            prepared.tree.reject_byte_sequence(
                b"/proc/self/fd/",
                suffixes=("",),
            )
            # PreparedTerminalOutput.commit() immediately revalidates the held
            # output root/tree after this callback and before marker rename.

        # Emit the external log condition before the terminal commit.  exp_run
        # also requires the descriptor-bound PASS and all output files, so a
        # later commit failure cannot be mistaken for a successful run.
        print(TASTE_PPO_MARKER, flush=True)
        retained_stack = stack.pop_all()
        try:
            prepared.commit(retained_input_closure=retained_input_closure)
        except Exception:
            try:
                prepared.close()
            finally:
                retained_stack.close()
            raise
        # Marker publication is the commit point.  Prepared/output close
        # failures are suppressed after commit; retained input FDs intentionally
        # remain open until process exit so no fallible cleanup follows PASS.
        prepared.close()
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Reject a checked-in candidate before reading any user-selected training
    # config.  run() reopens this gate at the execution boundary.  The
    # wrapper/registry owns failure reporting; this process never lexically
    # reopens a possibly displaced output root from an exception handler.
    _assert_execution_released()
    args = apply_config_overrides(args, parser, argv=argv)
    args = apply_decoded_chem_generation_defaults(args, argv=argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
