"""Config-driven RL runtime entrypoint for local and HPC use."""

from __future__ import annotations

import argparse
from pathlib import Path

from _runtime_common import (
    add_common_arguments,
    maybe_load_local_artifacts,
    prepare_runtime,
    print_manifest_if_requested,
    write_named_manifest,
)

from src.train.interfaces import TrainStage, TrainingRunRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-steps", type=int, help="Override training.max_steps.")
    parser.add_argument("--resume-from", help="Override training.resume_from.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    prepared = prepare_runtime(args, stage_name="rl", stage_config_name="rl.yaml")

    max_steps = args.max_steps or int(prepared.config.get("training", {}).get("max_steps", 0))
    request = TrainingRunRequest(
        stage=TrainStage.COUNTERFACTUAL_RL,
        run_name=prepared.context.run_name,
        output_dir=prepared.runtime_paths.run_dir,
        max_steps=max_steps,
        resume_from=Path(args.resume_from).expanduser().resolve()
        if args.resume_from
        else None,
        extra_config={
            "config_files": prepared.config_files,
            "dry_run": prepared.config.get("runtime", {}).get("dry_run", True),
            "batch_size": prepared.config.get("training", {}).get("batch_size"),
            "learning_rate": prepared.config.get("training", {}).get("learning_rate"),
        },
    )
    write_named_manifest(
        prepared,
        "rl_plan.json",
        {
            "stage": request.stage.value,
            "run_name": request.run_name,
            "output_dir": str(request.output_dir),
            "max_steps": request.max_steps,
            "resume_from": str(request.resume_from) if request.resume_from else None,
            "extra_config": request.extra_config,
        },
    )
    maybe_load_local_artifacts(prepared)
    prepared.logger.info(
        "Prepared RL request: stage=%s max_steps=%s output_dir=%s",
        request.stage.value,
        request.max_steps,
        request.output_dir,
    )
    prepared.logger.warning("RL training loop is not implemented yet; runtime preparation only.")
    print_manifest_if_requested(prepared, enabled=args.print_config)


if __name__ == "__main__":
    main()
