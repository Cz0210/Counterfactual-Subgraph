#!/usr/bin/env python3
"""Train the real train-only TasteMolNet T11 Ours PPO policy.

T6 is an integration prerequisite, not the full result.  This entrypoint
continues from its independently consumable adapter for 300 train-only PPO
updates, writes restartable state every 50 updates, and never opens validation,
calibration, or test data.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_ppo import (  # noqa: E402
    _infer_single_training_device,
    _safe_git_commit,
    apply_config_overrides,
    apply_decoded_chem_generation_defaults,
    build_hf_dataset,
    build_policy_model,
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
from src.rewards.gnn_ppo_reward import (  # noqa: E402
    BatchedGNNPPORewardAdapter,
    GNNPPORewardConfig,
)
from src.oracles.gnn_oracle import verify_checkpoint_bundle  # noqa: E402
from src.train.bace_gnn_ppo import (  # noqa: E402
    atomic_json,
    model_parameter_hash,
    validate_adapter_checkpoint_reload,
)
from src.train.stable_ppo_resume import (  # noqa: E402
    adopt_stable_ppo_checkpoint_prefix,
    read_stable_ppo_resume_manifest,
)
from src.train.bace_policy_init import atomic_text  # noqa: E402
from src.train.tastemolnet_gnn_ppo import (  # noqa: E402
    TastePPOObserver,
    build_taste_reward_manifest,
    validate_taste_ppo_output,
)
from src.utils.io import read_jsonl  # noqa: E402
from src.utils.logging_utils import RunContext, configure_run_logger  # noqa: E402


STAGE = "T11_OURS_PPO_FULL"
UPDATES = 300
CHECKPOINT_STEPS = (50, 100, 150, 200, 250, 300)
PASS_MARKER = "[TASTE_T11_OURS_PPO_FULL_PASS]"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return payload


def _fresh_output(path: Path) -> Path:
    path = path.expanduser().absolute()
    if not path.is_absolute():
        raise ValueError("T11 PPO output must be absolute")
    if path.exists():
        raise FileExistsError(f"T11 PPO output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def build_parser():
    parser = build_stable_parser()
    parser.description = __doc__
    parser.add_argument("--stage", choices=(STAGE,), default=STAGE)
    parser.add_argument("--t6-output", type=Path, required=True)
    parser.add_argument("--gnn-checkpoint", type=Path, required=True)
    parser.add_argument("--oracle-batch-size", type=int, default=256)
    parser.add_argument("--gnn-device", default="cuda")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    return parser


def _configure(args: Any) -> None:
    if type(args.oracle_batch_size) is not int or args.oracle_batch_size <= 0:
        raise ValueError("T11 oracle batch size must be a positive native int")
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
    args.max_steps = UPDATES
    args.max_prompt_examples = 0
    args.save_steps = 50
    args.batch_size = min(max(1, int(args.batch_size)), 8)


def _validate_inputs(
    *, t6_root: Path, checkpoint: Path, requested_dataset: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    t6_evidence = validate_taste_ppo_output(t6_root)
    verify_checkpoint_bundle(checkpoint, verify_hashes=True)
    t6_run = _read_json(t6_root / "run_manifest.json")
    t6_manifest = _read_json(t6_root / "manifest.json")
    card = _read_json(checkpoint / "model_card.json")
    required = {
        "dataset": "tastemolnet",
        "oracle_backend": "gnn",
        "rf_oracle_used": False,
        "backbone": "gine",
        "num_classes": 3,
        "source_label": 1,
        "profile": "full",
    }
    failures = [key for key, value in required.items() if card.get(key) != value]
    checkpoint_id = _sha256_file(checkpoint / "model.pt")
    if failures or card.get("checkpoint_id") != checkpoint_id:
        raise ValueError("T11 frozen Taste GINE contract failed: " + ", ".join(failures))
    train_contract = t6_run.get("train_contract")
    if not isinstance(train_contract, Mapping):
        raise ValueError("T6 run lacks its train contract")
    if (
        t6_run.get("gnn_checkpoint") != str(checkpoint)
        or t6_run.get("checkpoint_id") != checkpoint_id
        or t6_run.get("dataset_path") != str(requested_dataset)
        or train_contract.get("train_csv") != str(requested_dataset)
        or train_contract.get("train_csv_sha256") != _sha256_file(requested_dataset)
        or t6_manifest.get("policy_checkpoint_hash")
        != t6_evidence.get("policy_checkpoint_hash")
    ):
        raise ValueError("T11 inputs differ from the independently validated T6 lineage")
    split = _read_json(checkpoint / "split_manifest.json")
    if (
        split.get("calibration_loaded_for_training") is not False
        or split.get("test_loaded_for_training") is not False
        or split.get("test_used_for_checkpoint_selection") is not False
        or split.get("files", {}).get("train", {}).get("sha256")
        != _sha256_file(requested_dataset)
    ):
        raise ValueError("T11 split authority does not preserve train-only PPO")
    return t6_evidence, t6_run, card


def _resume_contract_base(
    *,
    args: Any,
    stable_config: Any,
    t6_evidence: Mapping[str, Any],
    t6_root: Path,
    checkpoint: Path,
    train_csv: Path,
    actual_batch_size: int,
    selected_parent_ids: list[str],
    git_commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": "tastemolnet_t11_ppo_resume_contract_v1",
        "stage": STAGE,
        "dataset": "tastemolnet",
        "git_commit": git_commit,
        "t6_root": str(t6_root),
        "t6_inventory_sha256": t6_evidence["output_inventory_sha256"],
        "t6_policy_checkpoint_hash": t6_evidence["policy_checkpoint_hash"],
        "gnn_checkpoint": str(checkpoint),
        "gnn_checkpoint_id": _sha256_file(checkpoint / "model.pt"),
        "train_csv": str(train_csv),
        "train_csv_sha256": _sha256_file(train_csv),
        "selected_parent_ids": selected_parent_ids,
        "max_steps": UPDATES,
        "save_steps": 50,
        "seed": int(args.seed),
        "actual_batch_size": int(actual_batch_size),
        "oracle_batch_size": int(args.oracle_batch_size),
        "stable_config": asdict(stable_config),
    }


def _resolve_resume(
    *,
    base: dict[str, Any],
    resume_checkpoint: Path | None,
    policy_hash_at_start: str,
    reference_hash: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if resume_checkpoint is None:
        return {
            **base,
            "policy_parameter_hash_before": policy_hash_at_start,
            "reference_policy_hash": reference_hash,
        }, None
    manifest = read_stable_ppo_resume_manifest(resume_checkpoint)
    stored = manifest.get("resume_contract")
    if not isinstance(stored, dict):
        raise ValueError("T11 resume checkpoint lacks a frozen resume contract")
    mismatches = [key for key, value in base.items() if stored.get(key) != value]
    if stored.get("reference_policy_hash") != reference_hash:
        mismatches.append("reference_policy_hash")
    original = stored.get("policy_parameter_hash_before")
    if not isinstance(original, str) or len(original) != 64:
        mismatches.append("policy_parameter_hash_before")
    if mismatches:
        raise ValueError("T11 resume contract drifted: " + ", ".join(sorted(set(mismatches))))
    return dict(stored), manifest


def _resume_artifacts(output: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for step in CHECKPOINT_STEPS:
        root = output / f"checkpoint-{step}"
        manifest = read_stable_ppo_resume_manifest(root)
        if int(manifest.get("completed_steps", -1)) != step or manifest.get("resume_contract") != contract:
            raise ValueError(f"T11 checkpoint-{step} resume state changed")
        result[str(step)] = {
            "root": str(root),
            "resume_manifest_sha256": _sha256_file(root / "stable_ppo_resume_manifest.json"),
            "training_state_sha256": _sha256_file(root / "stable_ppo_training_state.pt"),
            "candidate_pool_sha256": _sha256_file(root / "candidate_pool.jsonl"),
        }
    final = read_stable_ppo_resume_manifest(output)
    if int(final.get("completed_steps", -1)) != UPDATES or final.get("resume_contract") != contract:
        raise ValueError("T11 final resumable state changed")
    return {
        "schema_version": "tastemolnet_t11_resume_artifacts_v1",
        "checkpoints": result,
        "final": {
            "completed_steps": UPDATES,
            "resume_manifest_sha256": _sha256_file(output / "stable_ppo_resume_manifest.json"),
            "training_state_sha256": _sha256_file(output / "stable_ppo_training_state.pt"),
            "candidate_pool_sha256": _sha256_file(output / "candidate_pool.jsonl"),
        },
    }


def run(args: Any) -> int:
    _configure(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output = _fresh_output(Path(args.output_dir))
    args._failure_output_authorized = True
    args.output_dir = str(output)
    args.candidate_pool_path = str(output / "candidate_pool.jsonl")
    t6_root = args.t6_output.expanduser().resolve(strict=True)
    checkpoint = args.gnn_checkpoint.expanduser().resolve(strict=True)
    model_path = Path(args.model_path).expanduser().resolve(strict=True)
    train_csv = Path(args.dataset_path).expanduser().resolve(strict=True)
    t6_evidence, t6_run, card = _validate_inputs(
        t6_root=t6_root, checkpoint=checkpoint, requested_dataset=train_csv
    )
    if t6_run.get("model_path") != str(model_path):
        raise ValueError("T11 base model differs from T6")
    git_commit = _safe_git_commit()
    if not isinstance(git_commit, str) or len(git_commit) != 40:
        raise RuntimeError("T11 requires one exact execution commit")
    logger = configure_run_logger(
        "train_tastemolnet_ours_full",
        context=RunContext(run_name="tastemolnet_t11_ppo_full", output_dir=output, stage=STAGE, seed=int(args.seed)),
        log_dir=output / "logs",
    )
    logger.info("Runtime environment: %s", collect_runtime_environment_debug())
    stable_config = resolve_stable_config(args)
    deps = import_training_dependencies()
    deps["set_seed"](int(args.seed))
    torch = deps["torch"]
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tokenizer = build_tokenizer(deps, model_path=model_path, trust_remote_code=args.trust_remote_code, local_files_only=True)
    train_contract = dict(t6_run["train_contract"])
    sweet_count = int(train_contract["train_label_counts"]["1"])
    examples, prompt_evidence = load_tastemolnet_train_prompts(
        train_csv.read_bytes(),
        expected_num_records=int(train_contract["train_num_records"]),
        expected_label_counts=dict(train_contract["train_label_counts"]),
        max_prompt_examples=sweet_count,
    )
    resume_checkpoint = args.resume_from_checkpoint.expanduser().resolve(strict=True) if args.resume_from_checkpoint else None
    if resume_checkpoint is not None:
        validate_adapter_checkpoint_reload(resume_checkpoint)
    policy_adapter = resume_checkpoint or t6_root
    policy_model = build_policy_model(deps, model_path=model_path, adapter_path=policy_adapter, trust_remote_code=args.trust_remote_code, local_files_only=True, is_trainable=True)
    reference_model = build_policy_model(deps, model_path=model_path, adapter_path=t6_root, trust_remote_code=args.trust_remote_code, local_files_only=True, is_trainable=False)
    value_model = ensure_score_head_for_experimental_ppo(
        build_value_model(deps, model_path=model_path, tokenizer=tokenizer, trust_remote_code=args.trust_remote_code, local_files_only=True),
        "tastemolnet_t11_stable_value_model",
    )
    rollout_device = _infer_single_training_device(logger=logger, torch=torch, policy_model=policy_model, reference_model=reference_model, value_model=value_model)
    reward_adapter = BatchedGNNPPORewardAdapter.from_checkpoint(
        checkpoint,
        device=rollout_device if args.gnn_device == "cuda" else args.gnn_device,
        policy_initializer_hash=str(t6_evidence["policy_checkpoint_hash"]),
        reference_policy_hash=str(t6_evidence["policy_checkpoint_hash"]),
        config=GNNPPORewardConfig(dataset="tastemolnet", num_classes=3, source_label=1, oracle_batch_size=int(args.oracle_batch_size)),
    )
    parent_records = reward_adapter.predict_parent_records(
        parent_smiles=[row.parent_smiles for row in examples],
        metas=[{"molecule_id": row.molecule_id, "index": row.index} for row in examples],
    )
    selected = [row for row, pred in zip(examples, parent_records, strict=True) if pred.get("predicted_label") == 1]
    if not selected:
        raise ValueError("T11 has no train parent predicted as Sweet by the frozen GINE")
    selected_ids = [row.molecule_id for row in selected]
    dataset = build_hf_dataset(deps, tokenizer, selected)
    actual_batch_size = max(1, min(int(args.batch_size), len(dataset)))
    policy_start = model_parameter_hash(policy_model, trainable_only=True)
    reference_before = model_parameter_hash(reference_model, adapter_only=True)
    base = _resume_contract_base(
        args=args, stable_config=stable_config, t6_evidence=t6_evidence, t6_root=t6_root,
        checkpoint=checkpoint, train_csv=train_csv, actual_batch_size=actual_batch_size,
        selected_parent_ids=selected_ids, git_commit=git_commit,
    )
    resume_contract, resume_manifest = _resolve_resume(
        base=base, resume_checkpoint=resume_checkpoint, policy_hash_at_start=policy_start,
        reference_hash=reference_before,
    )
    policy_before = str(resume_contract["policy_parameter_hash_before"])
    adopted: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        completed = int(resume_manifest["completed_steps"])
        if completed not in CHECKPOINT_STEPS or completed >= UPDATES:
            raise ValueError("T11 resume source must be an incomplete periodic checkpoint")
        adopted = adopt_stable_ppo_checkpoint_prefix(
            resume_checkpoint=resume_checkpoint, output_dir=output, checkpoint_steps=CHECKPOINT_STEPS
        )
    atomic_json(output / "resume_provenance.json", {
        "schema_version": "tastemolnet_t11_resume_provenance_v1", "resumed": resume_checkpoint is not None,
        "source": str(resume_checkpoint) if resume_checkpoint else None, "source_manifest": resume_manifest,
        "resume_contract": resume_contract, "adopted_checkpoint_prefix": adopted,
    })
    atomic_json(output / "run_manifest.json", {
        "schema_version": "tastemolnet_t11_ppo_run_v1", "stage": STAGE, "status": "RUNNING",
        "git_commit": git_commit, "model_path": str(model_path), "t6_output": str(t6_root),
        "t6_evidence": t6_evidence, "gnn_checkpoint": str(checkpoint), "checkpoint_id": card["checkpoint_id"],
        "dataset_path": str(train_csv), "prompt_evidence": prompt_evidence, "selected_parent_ids": selected_ids,
        "stable_config": asdict(stable_config), "train_only": True, "validation_loaded": False,
        "calibration_loaded": False, "test_loaded": False, "rf_oracle_used": False,
    })
    observer = TastePPOObserver()
    run_stable_decoded_chem_ppo_loop(
        deps=deps, args=args, stable_config=stable_config, actual_batch_size=actual_batch_size,
        policy_model=policy_model, reference_model=reference_model, value_model=value_model,
        tokenizer=tokenizer, train_dataset=dataset, stable_reward_wrapper=reward_adapter,
        output_dir=output, logger=logger, run_observer=observer,
        resume_from_checkpoint=resume_checkpoint, resume_contract=resume_contract,
    )
    policy_after = model_parameter_hash(policy_model, trainable_only=True)
    reference_after = model_parameter_hash(reference_model, adapter_only=True)
    rows = read_jsonl(output / "candidate_pool.jsonl")
    oracle = reward_adapter.provenance()
    reward = build_taste_reward_manifest(rows, oracle_provenance=oracle)
    atomic_json(output / "oracle_provenance.json", oracle)
    atomic_json(output / "reward_manifest.json", reward)
    reload = validate_adapter_checkpoint_reload(output)
    artifacts = _resume_artifacts(output, resume_contract)
    failures: list[str] = []
    if policy_before == policy_after:
        failures.append("policy_unchanged")
    if reference_before != reference_after:
        failures.append("reference_changed")
    if len(observer.updates) != UPDATES:
        failures.append("optimizer_update_count")
    if not rows:
        failures.append("no_candidates")
    if int(reward.get("gnn_scored_deletion_count", 0)) < 1:
        failures.append("no_gnn_scored_deletion")
    if oracle.get("dataset") != "tastemolnet" or oracle.get("num_classes") != 3 or oracle.get("rf_oracle_used") is not False:
        failures.append("oracle_contract")
    gate = {
        "schema_version": "tastemolnet_t11_ppo_gate_v1", "stage": STAGE,
        "status": "PASS" if not failures else "FAIL", "failures": failures,
        "optimizer_step_count": len(observer.updates), "policy_parameter_hash_before": policy_before,
        "policy_parameter_hash_after": policy_after, "reference_parameter_hash_before": reference_before,
        "reference_parameter_hash_after": reference_after, "policy_checkpoint_hash": reload["policy_checkpoint_hash"],
        "checkpoint_reload_pass": reload["checkpoint_reload_pass"], "candidate_count": len(rows),
        "gnn_scored_deletion_count": int(reward.get("gnn_scored_deletion_count", 0)),
        "strict_flip_count": int(reward.get("strict_flip_count", 0)), "resume_supported": True,
        "train_only": True, "validation_loaded": False, "calibration_loaded": False,
        "test_loaded": False, "rf_oracle_used": False, "marker": PASS_MARKER,
    }
    atomic_json(output / "ppo_gate.json", gate)
    manifest = {
        **gate, "schema_version": "tastemolnet_t11_ppo_manifest_v1", "git_commit": git_commit,
        "output_root": str(output), "model_path": str(model_path), "policy_checkpoint": str(output),
        "gnn_checkpoint": str(checkpoint), "oracle_checkpoint_hash": card["checkpoint_id"],
        "train_split": str(train_csv), "train_split_sha256": _sha256_file(train_csv),
        "t6_output": str(t6_root), "t6_output_inventory_sha256": t6_evidence["output_inventory_sha256"],
        "resume_contract": resume_contract, "resume_artifacts": artifacts,
        "ppo_gate_sha256": _sha256_file(output / "ppo_gate.json"),
        "candidate_pool_sha256": _sha256_file(output / "candidate_pool.jsonl"),
        "reward_manifest_sha256": _sha256_file(output / "reward_manifest.json"),
        "oracle_manifest_sha256": _sha256_file(output / "oracle_provenance.json"),
    }
    atomic_json(output / "ppo_manifest.json", manifest)
    if failures:
        atomic_json(output / "FAIL.json", manifest)
        return 2
    completed_run = _read_json(output / "run_manifest.json")
    completed_run.update({
        "status": "PASS", "ppo_manifest_sha256": _sha256_file(output / "ppo_manifest.json"),
        "optimizer_step_count": UPDATES, "resume_supported": True,
    })
    atomic_json(output / "run_manifest.json", completed_run)
    atomic_text(output / "PASS", PASS_MARKER + "\n")
    print(PASS_MARKER, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args = apply_config_overrides(args, parser, argv=argv)
    args = apply_decoded_chem_generation_defaults(args, argv=argv)
    try:
        return run(args)
    except Exception as exc:
        if bool(getattr(args, "_failure_output_authorized", False)):
            output = Path(args.output_dir)
            if output.is_dir():
                atomic_json(output / "FAIL.json", {
                    "schema_version": "tastemolnet_t11_ppo_failure_v1", "stage": STAGE,
                    "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(), "calibration_loaded": False, "test_loaded": False,
                })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
