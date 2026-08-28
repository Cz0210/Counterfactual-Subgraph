#!/usr/bin/env python3
"""Train one managed-worker TasteMolNet NeuroSED scientific bundle."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.tastemolnet_neurosed import (  # noqa: E402
    TasteNeuroSEDTrainConfig,
    train_tastemolnet_neurosed,
)
from src.utils.tastemolnet_neurosed_authority import (  # noqa: E402
    hold_tastemolnet_neurosed_data_authority,
)
from src.utils.retained_readonly_file import hold_readonly_file  # noqa: E402


def _hold_controller_authority(*args: Any, **kwargs: Any) -> Any:
    try:
        from src.utils.autodl_tastemolnet_main_v2 import (
            hold_taste_main_v2_controller_authority,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Taste main-v2 controller authority integration is required"
        ) from exc
    return hold_taste_main_v2_controller_authority(*args, **kwargs)


def _read_config(data: bytes) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - AutoDL dependency gate.
        raise RuntimeError("Taste NeuroSED config requires PyYAML") from exc
    payload = yaml.safe_load(data.decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("training"), dict):
        raise ValueError("Taste NeuroSED config lacks a training mapping")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument(
        "--neurosed-config",
        default="configs/autodl/tastemolnet_neurosed_v1.yaml",
    )
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--t2-receipt-root", type=Path, required=True)
    parser.add_argument("--t2-source-bundle-root", type=Path, required=True)
    parser.add_argument("--t3-final-root", type=Path, required=True)
    parser.add_argument("--controller-receipt", type=Path, required=True)
    parser.add_argument("--controller-heartbeat", type=Path, required=True)
    parser.add_argument("--expected-controller-id", required=True)
    parser.add_argument("--expected-controller-receipt-sha256", required=True)
    parser.add_argument("--expected-controller-heartbeat-sha256", required=True)
    parser.add_argument("--expected-controller-heartbeat-sequence", type=int, required=True)
    parser.add_argument("--expected-controller-heartbeat-uuid", required=True)
    parser.add_argument("--expected-neurosed-config-sha256", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execution-git-commit", required=True)
    parser.add_argument("--execution-git-tree", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root or os.environ.get("MANAGED_ARTIFACT_ROOT")
    if not output_root:
        raise ValueError("--output-root or MANAGED_ARTIFACT_ROOT is required")
    with ExitStack() as stack:
        config_file = stack.enter_context(
            hold_readonly_file(
                Path(args.neurosed_config),
                expected_sha256=args.expected_neurosed_config_sha256,
            )
        )
        raw = _read_config(config_file.read_bytes())
        training = dict(raw["training"])
        config = TasteNeuroSEDTrainConfig(
            seed=int(training["seed"]),
            train_pairs=int(training["train_pairs"]),
            validation_pairs=int(training["validation_pairs"]),
            batch_size=int(training["batch_size"]),
            max_epochs=int(training["max_epochs"]),
            early_stopping_patience=int(training["early_stopping_patience"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            cyclic_step_size_up=int(training["cyclic_step_size_up"]),
            cyclic_step_size_down=int(training["cyclic_step_size_down"]),
            max_grad_norm=float(training["max_grad_norm"]),
            num_workers=int(training["num_workers"]),
            require_cuda_health_gate=bool(
                training["require_cuda_health_gate"]
            ),
            pair_semantics=str(training["pair_semantics"]),
        )
        controller = stack.enter_context(
            _hold_controller_authority(
                args.controller_receipt,
                args.controller_heartbeat,
                expected_controller_id=args.expected_controller_id,
                expected_git_commit=args.execution_git_commit,
                expected_git_tree=args.execution_git_tree,
                expected_receipt_sha256=args.expected_controller_receipt_sha256,
                expected_heartbeat_sha256=args.expected_controller_heartbeat_sha256,
            )
        )
        data_authority = stack.enter_context(
            hold_tastemolnet_neurosed_data_authority(
                t2_receipt_root=args.t2_receipt_root,
                t2_source_bundle_root=args.t2_source_bundle_root,
                t3_final_root=args.t3_final_root,
                train_csv=args.train_csv,
                validation_csv=args.validation_csv,
            )
        )
        controller_evidence = {
            "schema_version": "tastemolnet_gcf_neurosed_controller_binding_v2",
            "worker_initial_heartbeat": {
                "receipt_sha256": args.expected_controller_receipt_sha256,
                "heartbeat_path": str(args.controller_heartbeat),
                "heartbeat_sha256": args.expected_controller_heartbeat_sha256,
                "heartbeat_uuid": args.expected_controller_heartbeat_uuid,
                "sequence": args.expected_controller_heartbeat_sequence,
            },
            "worker_latest": dict(controller.evidence),
        }
        result = train_tastemolnet_neurosed(
            train_csv=args.train_csv,
            validation_csv=args.validation_csv,
            train_csv_bytes=data_authority.train_bytes,
            validation_csv_bytes=data_authority.validation_bytes,
            output_root=Path(output_root),
            execution_git_commit=args.execution_git_commit,
            execution_git_tree=args.execution_git_tree,
            source_execution_config_sha256=config_file.sha256,
            authoritative_lineage=data_authority.evidence,
            controller_authority=controller_evidence,
            config=config,
            device=args.device,
        )
        data_authority.revalidate()
        controller.revalidate()
        config_file.revalidate()
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[TASTE_GCF_NEUROSED_WORKER_ARTIFACT_READY_FOR_SEAL]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
