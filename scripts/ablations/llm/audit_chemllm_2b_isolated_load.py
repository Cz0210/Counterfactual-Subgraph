#!/usr/bin/env python3
"""Audit and optionally CPU-load the exact local ChemLLM 2B snapshot.

The ordinary entrypoint creates a fresh evidence root and then re-executes
itself as ``python -I -B`` with offline caches.  ``--mode metadata`` proves the
isolated config/tokenizer/model-class import only.  ``--mode cpu-load`` also
loads every weight on CPU and emits the actual tensor parameter report needed
by the LLM scale-ablation runtime gate. ``--tiny-forward`` also performs one
native-chat greedy generation of at most four tokens. No mode acquires a GPU lock.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ablations.llm.contracts import (  # noqa: E402
    LLMAblationContractError,
    canonical_json_sha256,
)
from src.ablations.llm.isolated_chemllm_load import (  # noqa: E402
    atomic_json,
    audit_remote_code,
    build_isolated_child_command,
    build_isolated_child_environment,
    pin_chemllm_2b_snapshot,
    prepare_fresh_output_root,
    run_isolated_child_probe,
    sha256_file,
    validate_isolated_load_receipt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--snapshot-manifest-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("metadata", "cpu-load"), default="cpu-load")
    parser.add_argument("--tiny-forward", action="store_true")
    parser.add_argument(
        "--expected-code-inventory-sha256",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--_isolated-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def _validate_config(path: Path, overrides: Sequence[str]) -> Path:
    if not path.is_absolute():
        lexical = REPO_ROOT / path
    else:
        lexical = path
    if lexical.is_symlink():
        raise LLMAblationContractError("--config must be a physical file")
    path = lexical.resolve(strict=True)
    if not path.is_file():
        raise LLMAblationContractError("--config must be a physical file")
    if "inference.fallback_to_heuristic=false" not in set(overrides):
        raise LLMAblationContractError(
            "isolated model audit requires --set inference.fallback_to_heuristic=false"
        )
    return path


def _child(args: argparse.Namespace) -> int:
    _validate_config(args.config, args.overrides)
    if sys.flags.isolated != 1 or sys.dont_write_bytecode is not True:
        raise LLMAblationContractError("child must run under python -I -B")
    snapshot = pin_chemllm_2b_snapshot(
        args.snapshot_root,
        args.snapshot_manifest,
        args.snapshot_manifest_sha256,
    )
    audit = audit_remote_code(snapshot)
    if audit["code_inventory_sha256"] != args.expected_code_inventory_sha256:
        raise LLMAblationContractError("remote code changed between parent and child")
    atomic_json(args.output_root / "remote_code_static_audit.child.json", audit)
    run_isolated_child_probe(
        snapshot,
        audit,
        args.output_root,
        mode=args.mode,
        tiny_forward=bool(args.tiny_forward),
    )
    return 0


def _parent(args: argparse.Namespace) -> int:
    config = _validate_config(args.config, args.overrides)
    if args.expected_code_inventory_sha256 is not None:
        raise LLMAblationContractError(
            "--expected-code-inventory-sha256 is reserved for the isolated child"
        )
    if args.tiny_forward and args.mode != "cpu-load":
        raise LLMAblationContractError("--tiny-forward requires --mode cpu-load")
    snapshot = pin_chemllm_2b_snapshot(
        args.snapshot_root,
        args.snapshot_manifest,
        args.snapshot_manifest_sha256,
    )
    audit = audit_remote_code(snapshot)
    output_lexical = Path(args.output_root).expanduser()
    if not output_lexical.is_absolute():
        raise LLMAblationContractError("output_root must be a fresh absolute path")
    output_absolute = Path(os.path.abspath(output_lexical))
    snapshot_root = Path(snapshot.root)
    if output_absolute == snapshot_root or snapshot_root in output_absolute.parents:
        raise LLMAblationContractError("output_root may not be inside the snapshot")
    output_root = prepare_fresh_output_root(args.output_root)
    if snapshot_root == output_root or snapshot_root in output_root.parents:
        raise LLMAblationContractError("output_root may not be inside the snapshot")
    input_manifest: dict[str, Any] = {
        "schema_version": "chemllm_2b_isolated_load_input_v1",
        "status": "FROZEN",
        "snapshot": snapshot.to_dict(),
        "remote_code_audit": audit,
        "config": str(config),
        "config_sha256": sha256_file(config),
        "config_overrides": list(args.overrides),
        "mode": args.mode,
        "tiny_forward": bool(args.tiny_forward),
        "python_executable": sys.executable,
        "gpu_lock_requested": False,
        "main_output_root_requested": False,
    }
    input_manifest["input_manifest_sha256"] = canonical_json_sha256(input_manifest)
    atomic_json(output_root / "input_manifest.json", input_manifest)
    atomic_json(output_root / "remote_code_static_audit.parent.json", audit)

    environment = build_isolated_child_environment(os.environ, output_root)
    command = build_isolated_child_command(
        python=sys.executable,
        script=Path(__file__),
        snapshot=snapshot,
        output_root=output_root,
        mode=args.mode,
        tiny_forward=bool(args.tiny_forward),
        code_inventory_sha256=audit["code_inventory_sha256"],
        config=config,
        config_overrides=args.overrides,
    )
    launch = {
        "schema_version": "chemllm_2b_isolated_child_launch_v1",
        "command": command,
        "shell": False,
        "environment_contract": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "HF_HUB_OFFLINE",
                "HF_MODULES_CACHE",
                "PYTHONNOUSERSITE",
                "TRANSFORMERS_OFFLINE",
            )
        },
    }
    launch["launch_sha256"] = canonical_json_sha256(launch)
    atomic_json(output_root / "child_launch.json", launch)
    completed = subprocess.run(command, env=environment, shell=False, check=False)
    if completed.returncode != 0:
        failure = {
            "schema_version": "chemllm_2b_isolated_load_failure_v1",
            "status": "FAILED",
            "returncode": int(completed.returncode),
            "pass_receipt_written": False,
        }
        failure["failure_sha256"] = canonical_json_sha256(failure)
        atomic_json(output_root / "failure.json", failure)
        raise LLMAblationContractError(
            f"isolated child failed with exit code {completed.returncode}"
        )
    receipt_path = output_root / "isolated_load_receipt.json"
    receipt = validate_isolated_load_receipt(
        receipt_path, require_weights=args.mode == "cpu-load"
    )
    source_snapshot_manifest = json.loads(
        Path(snapshot.manifest_path).read_text(encoding="utf-8")
    )
    if not isinstance(source_snapshot_manifest, dict):
        raise LLMAblationContractError("source snapshot manifest is not an object")
    runtime_snapshot_manifest = dict(source_snapshot_manifest)
    runtime_snapshot_manifest.update(
        {
            "isolated_import_pass": True,
            "trust_remote_code_enabled": True,
            "source_snapshot_manifest": {
                "path": snapshot.manifest_path,
                "sha256": snapshot.manifest_sha256,
            },
            "snapshot_inventory_sha256": snapshot.snapshot_inventory_sha256,
            "isolated_load_receipt": {
                "path": str(receipt_path),
                "file_sha256": sha256_file(receipt_path),
                "self_sha256": receipt["receipt_sha256"],
            },
            "actual_parameter_report": (
                {
                    "path": receipt["actual_parameter_report"],
                    "sha256": receipt["actual_parameter_report_file_sha256"],
                }
                if args.mode == "cpu-load"
                else None
            ),
            "original_snapshot_modified": False,
        }
    )
    runtime_snapshot_manifest.pop("runtime_adoption_manifest_sha256", None)
    runtime_snapshot_manifest["runtime_adoption_manifest_sha256"] = (
        canonical_json_sha256(runtime_snapshot_manifest)
    )
    runtime_snapshot_path = output_root / "snapshot_runtime_adoption_manifest.json"
    atomic_json(runtime_snapshot_path, runtime_snapshot_manifest)
    terminal = {
        "schema_version": "chemllm_2b_isolated_load_terminal_v1",
        "status": "PASS",
        "mode": args.mode,
        "input_manifest_sha256": input_manifest["input_manifest_sha256"],
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": sha256_file(receipt_path),
        "receipt_self_sha256": receipt["receipt_sha256"],
        "actual_loaded_weight_evidence": args.mode == "cpu-load",
        "runtime_snapshot_manifest": str(runtime_snapshot_path),
        "runtime_snapshot_manifest_file_sha256": sha256_file(runtime_snapshot_path),
        "tiny_forward_pass": (
            receipt.get("tiny_forward", {}).get("status") == "PASS"
            if args.tiny_forward
            else None
        ),
    }
    terminal["terminal_sha256"] = canonical_json_sha256(terminal)
    atomic_json(output_root / "terminal.json", terminal)
    print("[CHEMLLM_2B_ISOLATED_LOAD_PASS]")
    if args.mode == "cpu-load":
        print("[CHEMLLM_2B_ACTUAL_PARAMETER_COUNT_PASS]")
    print(f"output_root={output_root}")
    print(f"isolated_load_receipt={receipt_path}")
    print(f"runtime_snapshot_manifest={runtime_snapshot_path}")
    print(f"runtime_snapshot_manifest_sha256={sha256_file(runtime_snapshot_path)}")
    print(f"mode={args.mode}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _child(args) if args._isolated_child else _parent(args)
    except (LLMAblationContractError, FileExistsError, OSError) as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
