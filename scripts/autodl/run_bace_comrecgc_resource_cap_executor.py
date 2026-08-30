#!/usr/bin/env python3
"""Run the exact BACE ComRecGC 20k/25k resource-cap executor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_bace_comrecgc_resource_cap_executor import (  # noqa: E402
    ProcessContract,
    ResourceCapExecutor,
    ResourceCapExecutorInputs,
    classify_liveness,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    actions = parser.add_subparsers(dest="action", required=True)

    liveness = actions.add_parser("classify-liveness")
    liveness.add_argument("--observations", type=Path, required=True)
    liveness.add_argument("--stalled-after-seconds", type=float, default=3600.0)
    liveness.add_argument("--output", type=Path)

    run = actions.add_parser("run")
    run.add_argument("--handover-request", required=True)
    run.add_argument("--checkpoint-dir", required=True)
    run.add_argument("--source-trace-root", required=True)
    run.add_argument("--source-resolved-config", required=True)
    run.add_argument("--dataset-dir", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--python", required=True)
    run.add_argument("--project-root", default=str(PROJECT_ROOT))
    run.add_argument("--gnn-checkpoint", required=True)
    run.add_argument("--calibration-split", required=True)
    run.add_argument("--test-split", required=True)
    run.add_argument("--molclr-root", required=True)
    run.add_argument("--molclr-checkpoint", required=True)
    run.add_argument("--neurosed-checkpoint", required=True)
    run.add_argument("--official-root", required=True)
    run.add_argument("--science-pid", type=int, required=True)
    run.add_argument("--science-start-ticks", type=int, required=True)
    run.add_argument("--science-cmdline-sha256", required=True)
    run.add_argument("--science-cwd", required=True)
    run.add_argument("--science-output-root", required=True)
    run.add_argument("--controller-id", required=True)
    run.add_argument("--controller-receipt", required=True)
    run.add_argument("--controller-receipt-sha256", required=True)
    run.add_argument("--expected-ppid", type=int)
    run.add_argument("--poll-seconds", type=int, default=60)
    run.add_argument("--exit-wait-seconds", type=int, default=300)
    run.add_argument("--once", action="store_true")
    return parser


def _load_observations(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("observations")
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("liveness observations must be a JSON list of objects")
    return [dict(row) for row in value]


def main() -> int:
    args = build_parser().parse_args()
    if args.action == "classify-liveness":
        result = classify_liveness(
            _load_observations(args.observations),
            stalled_after_seconds=args.stalled_after_seconds,
        )
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            destination = args.output.expanduser().resolve(strict=False)
            if destination.exists():
                raise FileExistsError(f"liveness output must be fresh: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        if result["state"] == "RUNNING_SLOW":
            print("[BACE_COMRECGC_RUNNING_SLOW]")
        return 0

    process = ProcessContract(
        pid=args.science_pid,
        start_ticks=args.science_start_ticks,
        cmdline_sha256=args.science_cmdline_sha256,
        cwd=args.science_cwd,
        output_root=args.science_output_root,
        controller_id=args.controller_id,
        controller_receipt=args.controller_receipt,
        controller_receipt_sha256=args.controller_receipt_sha256,
        expected_ppid=args.expected_ppid,
    )
    inputs = ResourceCapExecutorInputs(
        handover_request=args.handover_request,
        checkpoint_dir=args.checkpoint_dir,
        source_trace_root=args.source_trace_root,
        source_resolved_config=args.source_resolved_config,
        dataset_dir=args.dataset_dir,
        output_root=args.output_root,
        python=args.python,
        project_root=args.project_root,
        gnn_checkpoint=args.gnn_checkpoint,
        calibration_split=args.calibration_split,
        test_split=args.test_split,
        molclr_root=args.molclr_root,
        molclr_checkpoint=args.molclr_checkpoint,
        neurosed_checkpoint=args.neurosed_checkpoint,
        official_root=args.official_root,
        process=process,
        poll_seconds=args.poll_seconds,
        exit_wait_seconds=args.exit_wait_seconds,
    )
    return ResourceCapExecutor(inputs).run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
