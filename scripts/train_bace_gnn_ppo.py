#!/usr/bin/env python3
"""Run real BACE B6-v2/B7 through the existing stable decoded-chem PPO loop."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any

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
    load_stable_prompt_examples,
    resolve_stable_config,
    run_stable_decoded_chem_ppo_loop,
)
from src.rewards.gnn_ppo_reward import (  # noqa: E402
    BatchedGNNPPORewardAdapter,
    GNNPPORewardConfig,
)
from src.train.bace_gnn_ppo import (  # noqa: E402
    BacePPOObserver,
    atomic_json,
    build_ppo_gate,
    build_reward_manifest,
    model_parameter_hash,
    validate_adapter_checkpoint_reload,
)
from src.train.bace_policy_init import (  # noqa: E402
    atomic_text,
    sha256_file,
    validate_frozen_train_contract,
    validate_policy_provenance_manifest,
)
from src.utils.io import read_jsonl  # noqa: E402
from src.utils.logging_utils import RunContext, configure_run_logger  # noqa: E402


STAGES = (
    "BACE_GNN_PPO_ADAPTER_CANARY",
    "B6_PPO_SMOKE_V2",
    "B7_PPO_FULL",
)
B7_CHECKPOINT_STEPS = (50, 100, 150, 200, 250, 300)


def build_parser():
    parser = build_stable_parser()
    parser.description = __doc__
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--gnn-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-initializer", type=Path, required=True)
    parser.add_argument("--policy-provenance-manifest", type=Path, required=True)
    parser.add_argument("--b6-v2-manifest", type=Path, default=None)
    parser.add_argument("--b6-updates", type=int, default=5)
    parser.add_argument("--b6-parent-count", type=int, default=16)
    parser.add_argument("--oracle-batch-size", type=int, default=256)
    parser.add_argument("--gnn-device", default="cuda")
    return parser


def _fresh(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"BACE PPO output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _validate_b6_predecessor(
    manifest_path: Path,
    *,
    checkpoint_id: str,
    policy_initializer_hash: str,
    git_commit: str,
) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "stage": "B6_PPO_SMOKE_V2",
        "status": "PASS",
        "oracle_backend": "gnn",
        "dataset": "bace",
        "classifier_type": "gnn",
        "source_label": 1,
        "num_classes": 2,
        "rf_oracle_used": False,
        "calibration_loaded": False,
        "calibration_dataset_loaded": False,
        "frozen_temperature_calibration_loaded": True,
        "test_loaded": False,
        "checkpoint_id": checkpoint_id,
        "policy_initializer_hash": policy_initializer_hash,
        "git_commit": git_commit,
    }
    failures = [
        f"{key}={payload.get(key)!r}"
        for key, expected in required.items()
        if payload.get(key) != expected
    ]
    if failures:
        raise ValueError("B7 rejected B6-v2 predecessor: " + ", ".join(failures))
    policy_checkpoint_hash = str(payload.get("policy_checkpoint_hash") or "")
    if len(policy_checkpoint_hash) != 64 or any(
        character not in "0123456789abcdef"
        for character in policy_checkpoint_hash.lower()
    ):
        raise ValueError("B7 rejected malformed B6-v2 policy checkpoint identity")
    return {"path": str(path), "sha256": sha256_file(path), "manifest": payload}


def _configure_stage(args: Any) -> None:
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
    if args.stage == "BACE_GNN_PPO_ADAPTER_CANARY":
        args.max_steps = 1
        args.max_prompt_examples = 8
        args.save_steps = 1
        args.batch_size = min(max(1, int(args.batch_size)), 2)
    elif args.stage == "B6_PPO_SMOKE_V2":
        if not 5 <= int(args.b6_updates) <= 10:
            raise ValueError("B6-v2 updates must be in [5, 10]")
        if not 8 <= int(args.b6_parent_count) <= 16:
            raise ValueError("B6-v2 parent count must be in [8, 16]")
        args.max_steps = int(args.b6_updates)
        args.max_prompt_examples = int(args.b6_parent_count)
        args.save_steps = int(args.b6_updates)
        args.batch_size = min(max(1, int(args.batch_size)), 4)
    else:
        args.max_steps = 300
        args.max_prompt_examples = 0
        args.save_steps = 50
        args.batch_size = min(max(1, int(args.batch_size)), 8)


def _gnn_checkpoint_contract(checkpoint: Path) -> dict[str, Any]:
    card = json.loads((checkpoint / "model_card.json").read_text(encoding="utf-8"))
    required = {
        "dataset": "bace",
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "backbone": "gine",
        "num_classes": 2,
        "source_label": 1,
    }
    failures = [
        f"{key}={card.get(key)!r}"
        for key, expected in required.items()
        if card.get(key) != expected
    ]
    if failures:
        raise ValueError("BACE GNN PPO checkpoint gate failed: " + ", ".join(failures))
    temperature = json.loads(
        (checkpoint / "temperature_scaling.json").read_text(encoding="utf-8")
    )
    if (
        temperature.get("status") != "fit"
        or temperature.get("selection_split") != "validation"
        or temperature.get("test_used_for_fit") is not False
        or temperature.get("argmax_invariant") is not True
    ):
        raise ValueError("BACE PPO requires the frozen validation-calibrated B4 GINE")
    return card


def run(args: Any) -> int:
    _configure_stage(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output = _fresh(Path(args.output_dir))
    # Only a directory proven fresh by this invocation may receive FAIL.json.
    # In particular, an old non-empty B6/B7 root must remain byte-for-byte
    # untouched when _fresh raises.
    args._failure_output_authorized = True
    args.output_dir = str(output)
    args.candidate_pool_path = str(output / "candidate_pool.jsonl")
    checkpoint = args.gnn_checkpoint.expanduser().resolve(strict=True)
    model_path = Path(args.model_path).expanduser().resolve(strict=True)
    initializer = args.policy_initializer.expanduser().resolve(strict=True)
    policy_provenance = validate_policy_provenance_manifest(
        initializer,
        args.policy_provenance_manifest,
    )
    card = _gnn_checkpoint_contract(checkpoint)
    git_commit = _safe_git_commit()
    if (
        not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit.lower())
    ):
        raise RuntimeError("BACE PPO requires one exact 40-hex execution commit")
    train_csv = Path(args.dataset_path).expanduser().resolve(strict=True)
    train_contract = validate_frozen_train_contract(checkpoint, train_csv)
    b6_predecessor = None
    if args.stage == "B7_PPO_FULL":
        if args.b6_v2_manifest is None:
            raise ValueError("B7 requires --b6-v2-manifest")
        b6_predecessor = _validate_b6_predecessor(
            args.b6_v2_manifest,
            checkpoint_id=str(card["checkpoint_id"]),
            policy_initializer_hash=str(policy_provenance["policy_initializer_hash"]),
            git_commit=git_commit,
        )

    logger = configure_run_logger(
        "train_bace_gnn_ppo",
        context=RunContext(
            run_name=f"bace_{args.stage.lower()}",
            output_dir=output,
            stage=args.stage,
            seed=int(args.seed),
        ),
        log_dir=output / "logs",
    )
    logger.info("Runtime environment: %s", collect_runtime_environment_debug())
    logger.info(
        "[BACE_GNN_PPO_BOUNDARY] stage=%s stable_loop=run_stable_decoded_chem_ppo_loop "
        "oracle_backend=gnn rf_oracle_used=false calibration_loaded=false test_loaded=false",
        args.stage,
    )
    stable_config = resolve_stable_config(args)
    atomic_json(
        output / "run_manifest.json",
        {
            "schema_version": "bace_gnn_ppo_run_v1",
            "stage": args.stage,
            "git_commit": git_commit,
            "stable_loop": "scripts.train_ppo_stable.run_stable_decoded_chem_ppo_loop",
            "shared_algorithm_reimplemented": False,
            "model_path": str(model_path),
            "policy_initializer": str(initializer),
            "policy_initializer_hash": policy_provenance["policy_initializer_hash"],
            "gnn_checkpoint": str(checkpoint),
            "checkpoint_id": card["checkpoint_id"],
            "dataset_path": str(train_csv),
            "train_contract": train_contract,
            "b6_predecessor": b6_predecessor,
            "stable_config": asdict(stable_config),
            "args": {
                key: value
                for key, value in vars(args).items()
                if not str(key).startswith("_")
            },
            "oracle_backend": "gnn",
            "rf_oracle_used": False,
            "calibration_loaded": False,
            "calibration_dataset_loaded": False,
            "frozen_temperature_calibration_loaded": True,
            "test_loaded": False,
        },
    )
    atomic_json(output / "policy_provenance.json", policy_provenance)

    examples = load_stable_prompt_examples(
        train_csv,
        default_parent_label=1,
        only_positive=True,
        max_prompt_examples=int(args.max_prompt_examples),
    )
    if args.stage in {"BACE_GNN_PPO_ADAPTER_CANARY", "B6_PPO_SMOKE_V2"}:
        if not 8 <= len(examples) <= 16:
            raise ValueError(f"{args.stage} requires 8-16 BACE train source parents")
    deps = import_training_dependencies()
    deps["set_seed"](int(args.seed))
    torch = deps["torch"]
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tokenizer = build_tokenizer(
        deps,
        model_path=model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    dataset = build_hf_dataset(deps, tokenizer, examples)
    actual_batch_size = max(1, min(int(args.batch_size), len(dataset)))
    policy_model = build_policy_model(
        deps,
        model_path=model_path,
        adapter_path=initializer,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
        is_trainable=True,
    )
    reference_model = build_policy_model(
        deps,
        model_path=model_path,
        adapter_path=initializer,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
        is_trainable=False,
    )
    value_model = build_value_model(
        deps,
        model_path=model_path,
        tokenizer=tokenizer,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    value_model = ensure_score_head_for_experimental_ppo(
        value_model, "bace_gnn_stable_value_model"
    )
    rollout_device = _infer_single_training_device(
        logger=logger,
        torch=torch,
        policy_model=policy_model,
        reference_model=reference_model,
        value_model=value_model,
    )
    policy_hash_before = model_parameter_hash(policy_model, trainable_only=True)
    reference_hash_before = model_parameter_hash(reference_model, adapter_only=True)
    reward_adapter = BatchedGNNPPORewardAdapter.from_checkpoint(
        checkpoint,
        device=rollout_device if args.gnn_device == "cuda" else args.gnn_device,
        policy_initializer_hash=str(policy_provenance["policy_initializer_hash"]),
        reference_policy_hash=reference_hash_before,
        config=GNNPPORewardConfig(
            source_label=1,
            oracle_batch_size=int(args.oracle_batch_size),
        ),
    )
    observer = BacePPOObserver()
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
    policy_hash_after = model_parameter_hash(policy_model, trainable_only=True)
    reference_hash_after = model_parameter_hash(reference_model, adapter_only=True)
    checkpoint_reload = validate_adapter_checkpoint_reload(output)
    last_periodic_step = int(args.max_steps)
    periodic_checkpoint_reload = validate_adapter_checkpoint_reload(
        output / f"checkpoint-{last_periodic_step}"
    )
    rows = read_jsonl(output / "candidate_pool.jsonl")
    oracle_provenance = reward_adapter.provenance()
    atomic_json(output / "oracle_provenance.json", oracle_provenance)
    reward_manifest = build_reward_manifest(rows, oracle_provenance=oracle_provenance)
    atomic_json(output / "reward_manifest.json", reward_manifest)
    expected = B7_CHECKPOINT_STEPS if args.stage == "B7_PPO_FULL" else (int(args.max_steps),)
    gate = build_ppo_gate(
        stage=args.stage,
        policy_parameter_hash_before=policy_hash_before,
        policy_parameter_hash_after=policy_hash_after,
        reference_parameter_hash_before=reference_hash_before,
        reference_parameter_hash_after=reference_hash_after,
        observer=observer,
        checkpoint_reload=checkpoint_reload,
        periodic_checkpoint_reload=periodic_checkpoint_reload,
        reward_manifest=reward_manifest,
        oracle_provenance=oracle_provenance,
        expected_checkpoints=expected,
        hard_kl=0.8,
    )
    atomic_json(output / "ppo_gate.json", gate)
    manifest_name = {
        "BACE_GNN_PPO_ADAPTER_CANARY": "canary_manifest.json",
        "B6_PPO_SMOKE_V2": "ppo_smoke_manifest.json",
        "B7_PPO_FULL": "ppo_manifest.json",
    }[args.stage]
    manifest = {
        **gate,
        "checkpoint_id": str(card["checkpoint_id"]),
        "git_commit": git_commit,
        "gnn_checkpoint": str(checkpoint),
        "policy_initializer": str(initializer),
        "policy_initializer_hash": str(policy_provenance["policy_initializer_hash"]),
        "reference_policy_hash": reference_hash_before,
        "policy_checkpoint_hash": checkpoint_reload["policy_checkpoint_hash"],
        "policy_checkpoint_hash_schema": checkpoint_reload[
            "policy_checkpoint_hash_schema"
        ],
        "policy_checkpoint_hash_payload": checkpoint_reload[
            "policy_checkpoint_hash_payload"
        ],
        "final_adapter_config_identity": checkpoint_reload["adapter_config"],
        "final_adapter_weights_identity": checkpoint_reload["adapter_weights"],
        "last_periodic_checkpoint_step": last_periodic_step,
        "last_periodic_policy_checkpoint_hash": periodic_checkpoint_reload[
            "policy_checkpoint_hash"
        ],
        "last_periodic_policy_checkpoint_hash_schema": periodic_checkpoint_reload[
            "policy_checkpoint_hash_schema"
        ],
        "last_periodic_policy_checkpoint_hash_payload": periodic_checkpoint_reload[
            "policy_checkpoint_hash_payload"
        ],
        "last_periodic_adapter_config_identity": periodic_checkpoint_reload[
            "adapter_config"
        ],
        "last_periodic_adapter_weights_identity": periodic_checkpoint_reload[
            "adapter_weights"
        ],
        "candidate_pool": str(output / "candidate_pool.jsonl"),
        "reward_manifest": str(output / "reward_manifest.json"),
        "oracle_provenance": str(output / "oracle_provenance.json"),
        "output_root": str(output),
        "stable_loop": "scripts.train_ppo_stable.run_stable_decoded_chem_ppo_loop",
        "shared_algorithm_reimplemented": False,
    }
    atomic_json(output / manifest_name, manifest)
    if gate["status"] != "PASS":
        atomic_json(output / "FAIL.json", manifest)
        logger.error("BACE %s gate failed: %s", args.stage, gate["failures"])
        return 2
    marker = {
        "BACE_GNN_PPO_ADAPTER_CANARY": "[BACE_GNN_PPO_ADAPTER_CANARY_PASS]",
        "B6_PPO_SMOKE_V2": "[BACE_B6_V2_PASS]",
        "B7_PPO_FULL": "[BACE_B7_PASS]",
    }[args.stage]
    atomic_text(output / "PASS", marker + "\n")
    if args.stage == "BACE_GNN_PPO_ADAPTER_CANARY":
        print(marker, flush=True)
    elif args.stage == "B6_PPO_SMOKE_V2":
        print("[BACE_CLEAN_POLICY_INITIALIZER_PASS]", flush=True)
        print("[BACE_GNN_PPO_ADAPTER_PASS]", flush=True)
        print(marker, flush=True)
    else:
        print(marker, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args = apply_config_overrides(args, parser, argv=argv)
    args = apply_decoded_chem_generation_defaults(args)
    try:
        return run(args)
    except Exception as exc:
        try:
            output = Path(args.output_dir).expanduser().resolve()
            if output.is_dir() and bool(
                getattr(args, "_failure_output_authorized", False)
            ):
                atomic_json(
                    output / "FAIL.json",
                    {
                        "schema_version": "bace_gnn_ppo_failure_v1",
                        "stage": getattr(args, "stage", None),
                        "status": "FAIL",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "calibration_loaded": False,
                        "test_loaded": False,
                    },
                )
        finally:
            raise


if __name__ == "__main__":
    raise SystemExit(main())
