"""Config-driven evaluation runtime entrypoint for local and HPC use."""

from __future__ import annotations

import argparse

from _runtime_common import (
    add_common_arguments,
    maybe_load_local_artifacts,
    prepare_runtime,
    print_manifest_if_requested,
    write_named_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--split", help="Override evaluation.split.")
    parser.add_argument("--max-examples", type=int, help="Override evaluation.max_examples.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.split:
        args.set.append(f"evaluation.split={args.split}")
    if args.max_examples is not None:
        args.set.append(f"evaluation.max_examples={args.max_examples}")

    prepared = prepare_runtime(args, stage_name="eval", stage_config_name="eval.yaml")
    maybe_load_local_artifacts(prepared)

    eval_plan = {
        "split": prepared.config.get("evaluation", {}).get("split"),
        "max_examples": prepared.config.get("evaluation", {}).get("max_examples"),
        "checkpoint_path": prepared.manifest.get("resolved_paths", {}).get("checkpoint_path"),
        "config_files": prepared.config_files,
    }
    write_named_manifest(prepared, "eval_plan.json", eval_plan)
    prepared.logger.info(
        "Prepared evaluation run: split=%s max_examples=%s",
        eval_plan["split"],
        eval_plan["max_examples"],
    )
    prepared.logger.warning("Evaluation loop is not implemented yet; runtime preparation only.")
    print_manifest_if_requested(prepared, enabled=args.print_config)


if __name__ == "__main__":
    main()
