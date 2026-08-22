#!/usr/bin/env python3
"""Build a fresh raw-base LoRA or bounded train-only oracle-neutral BACE SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.bace_policy_init import (  # noqa: E402
    POLICY_PROVENANCE_SCHEMA,
    atomic_text,
    audit_policy_initializer,
    build_oracle_neutral_sft_dataset,
    finalize_adapter_manifest,
    hash_path_inventory,
    source_model_hash_from_passed_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Accepted for paired Slurm compatibility; build inputs stay explicit.",
    )
    parser.add_argument(
        "--source-model-hash",
        default=None,
        help="Reuse the content hash frozen by the one-time provenance audit.",
    )
    parser.add_argument(
        "--audit-selection",
        type=Path,
        default=None,
        help="Reuse a PASS audit selection and avoid a second base-model scan.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("raw-base", "oracle-neutral-sft"):
        child = subparsers.add_parser(mode)
        child.add_argument("--model-path", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--seed", type=int, default=7)
        child.add_argument("--lora-rank", type=int, default=8)
        child.add_argument("--lora-alpha", type=int, default=16)
        child.add_argument("--lora-dropout", type=float, default=0.05)
    sft = subparsers.choices["oracle-neutral-sft"]
    sft.add_argument("--train-csv", type=Path, required=True)
    sft.add_argument("--gnn-checkpoint", type=Path, required=True)
    sft.add_argument("--max-steps", type=int, default=100)
    sft.add_argument("--max-parents", type=int, default=0)
    return parser


def _fresh(path: Path) -> Path:
    target = path.expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"BACE clean initializer output must be fresh: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _raw_base_adapter(args: argparse.Namespace, output: Path, source_hash: str) -> dict:
    audit = audit_policy_initializer(args.model_path, kind_hint="raw_base", content_hash=False)
    if audit.classification != "CLEAN_CHEMLLM_BASE":
        raise ValueError(f"Raw ChemLLM base provenance failed: {audit.to_dict()}")
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - AutoDL dependency
        raise RuntimeError("Raw-base LoRA initialization requires peft") from exc
    from scripts.train_ppo import (
        build_quantized_base_model,
        build_tokenizer,
        import_training_dependencies,
    )

    deps = import_training_dependencies()
    deps["set_seed"](int(args.seed))
    model_path = args.model_path.expanduser().resolve(strict=True)
    tokenizer = build_tokenizer(
        deps,
        model_path=model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    base = build_quantized_base_model(
        deps,
        model_path=model_path,
        trust_remote_code=True,
        local_files_only=True,
        prepare_for_training=False,
    )
    if getattr(base, "peft_config", None):
        raise ValueError("Raw ChemLLM base unexpectedly contains a PEFT adapter")
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        target_modules=["wqkv", "wo", "w1", "w2", "w3"],
    )
    model = get_peft_model(base, lora)
    adapter = output / "adapter"
    model.save_pretrained(str(adapter), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter))
    return {
        "schema_version": POLICY_PROVENANCE_SCHEMA,
        "policy_initialization_type": "chemllm_base_fresh_lora",
        "dataset": "bace",
        "data_split_used": "none",
        "adapter_initialized_from_scratch": True,
        "optimizer_step_count": 0,
        "rf_reference_count": 0,
        "gnn_reward_used": False,
        "formal_validation_loaded": False,
        "policy_internal_validation_loaded": False,
        "policy_internal_validation_source": None,
        "calibration_loaded": False,
        "test_loaded": False,
        "source_model_path": str(model_path),
        "source_model_hash": source_hash,
        "training_data_hash": None,
        "seed": int(args.seed),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "adapter_dir": str(adapter),
    }


def _oracle_neutral_sft(args: argparse.Namespace, output: Path, source_hash: str) -> dict:
    if not 1 <= int(args.max_steps) <= 200:
        raise ValueError("BACE oracle-neutral SFT max_steps must be in [1, 200]")
    if (
        int(args.lora_rank),
        int(args.lora_alpha),
        float(args.lora_dropout),
    ) != (8, 16, 0.05):
        raise ValueError(
            "The shared bounded SFT trainer fixes LoRA to r=8, alpha=16, dropout=0.05"
        )
    dataset_dir = output / "dataset"
    manifest = build_oracle_neutral_sft_dataset(
        train_csv=args.train_csv,
        checkpoint_dir=args.gnn_checkpoint,
        output_dir=dataset_dir,
        source_model_hash=source_hash,
        seed=int(args.seed),
        max_parents=int(args.max_parents),
    )
    adapter = output / "adapter"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_sft.py"),
        "--model-path",
        str(args.model_path.expanduser().resolve(strict=True)),
        "--train-file",
        str(manifest["train_jsonl"]),
        "--val-file",
        str(manifest["validation_jsonl"]),
        "--output-dir",
        str(adapter),
        "--seed",
        str(args.seed),
        "--max-steps",
        str(args.max_steps),
        "--logging-steps",
        "10",
        "--save-steps",
        str(min(50, args.max_steps)),
        "--eval-steps",
        str(min(50, args.max_steps)),
        "--save-total-limit",
        "2",
        "--report-to",
        "none",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return {
        **manifest,
        "schema_version": POLICY_PROVENANCE_SCHEMA,
        "policy_initialization_type": "oracle_neutral_sft",
        "bounded_lora_sft": True,
        "max_steps": int(args.max_steps),
        "adapter_initialized_from_scratch": True,
        "adapter_dir": str(adapter),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = _fresh(args.output_dir)
    model_path = args.model_path.expanduser().resolve(strict=True)
    if args.source_model_hash and args.audit_selection is not None:
        raise ValueError("Choose --source-model-hash or --audit-selection, not both")
    if args.audit_selection is not None:
        source_hash = source_model_hash_from_passed_audit(
            args.audit_selection,
            expected_model_path=model_path,
        )
    else:
        # Standalone builds hash once.  The AutoDL audit->build route always
        # supplies --audit-selection and therefore never scans the 7B tree twice.
        source_hash = str(args.source_model_hash or "").strip() or hash_path_inventory(
            model_path
        )
    manifest = (
        _raw_base_adapter(args, output, source_hash)
        if args.mode == "raw-base"
        else _oracle_neutral_sft(args, output, source_hash)
    )
    adapter = Path(str(manifest["adapter_dir"]))
    finalized = finalize_adapter_manifest(
        adapter_dir=adapter,
        manifest=manifest,
        output_path=output / "policy_provenance.json",
    )
    atomic_text(output / "PASS", "[BACE_CLEAN_POLICY_INITIALIZER_PASS]\n")
    print(json.dumps(finalized, sort_keys=True), flush=True)
    print("[BACE_CLEAN_POLICY_INITIALIZER_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
