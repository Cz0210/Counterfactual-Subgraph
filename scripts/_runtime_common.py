"""Shared CLI/runtime preparation for local and HPC scripts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import load_local_hf_artifacts
from src.utils import (
    RunContext,
    apply_dotlist_overrides,
    build_runtime_paths,
    configure_run_logger,
    detect_execution_environment,
    inject_runtime_paths,
    load_and_merge_config_files,
    set_global_seed,
    write_runtime_manifest,
)


@dataclass(frozen=True, slots=True)
class PreparedRuntime:
    """Resolved runtime state shared by run scripts."""

    config: dict[str, Any]
    manifest: dict[str, Any]
    runtime_paths: Any
    logger: Any
    context: RunContext
    config_files: tuple[str, ...]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared CLI arguments for local/HPC runtime preparation."""

    parser.add_argument(
        "--environment",
        choices=["auto", "local", "hpc"],
        default="auto",
        help="Select the runtime environment or detect it from environment variables.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Additional config file applied after base/environment/stage configs.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one config value using dotted keys, for example model.model_path=models/foo.",
    )
    parser.add_argument("--run-name", help="Override run.name.")
    parser.add_argument("--model-path", help="Override model.model_path with a local path.")
    parser.add_argument(
        "--tokenizer-path",
        help="Override model.tokenizer_path with a local path.",
    )
    parser.add_argument("--output-root", help="Override paths.output_root.")
    parser.add_argument("--seed", type=int, help="Override runtime.seed.")
    parser.add_argument("--device", help="Override runtime.device.")
    parser.add_argument("--num-workers", type=int, help="Override runtime.num_workers.")
    parser.add_argument("--train-file", help="Override data.train_file.")
    parser.add_argument("--valid-file", help="Override data.valid_file.")
    parser.add_argument("--test-file", help="Override data.test_file.")
    parser.add_argument("--checkpoint-path", help="Override evaluation.checkpoint_path.")
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Attempt to load model/tokenizer from local filesystem paths.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config manifest as JSON.",
    )
    dry_run_group = parser.add_mutually_exclusive_group()
    dry_run_group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=None,
        help="Force runtime.dry_run=true.",
    )
    dry_run_group.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Force runtime.dry_run=false.",
    )


def _resolve_environment(cli_value: str) -> str:
    if cli_value != "auto":
        return cli_value
    return detect_execution_environment().value


def _base_config_files(stage_config_name: str, environment: str) -> list[Path]:
    return [
        REPO_ROOT / "configs" / "base.yaml",
        REPO_ROOT / "configs" / f"{environment}.yaml",
        REPO_ROOT / "configs" / stage_config_name,
    ]


def _cli_overrides(args: argparse.Namespace) -> list[str]:
    overrides: list[str] = []
    if args.run_name:
        overrides.append(f"run.name={args.run_name}")
    if args.model_path:
        overrides.append(f"model.model_path={args.model_path}")
    if args.tokenizer_path:
        overrides.append(f"model.tokenizer_path={args.tokenizer_path}")
    if args.output_root:
        overrides.append(f"paths.output_root={args.output_root}")
    if args.seed is not None:
        overrides.append(f"runtime.seed={args.seed}")
    if args.device:
        overrides.append(f"runtime.device={args.device}")
    if args.num_workers is not None:
        overrides.append(f"runtime.num_workers={args.num_workers}")
    if args.train_file:
        overrides.append(f"data.train_file={args.train_file}")
    if args.valid_file:
        overrides.append(f"data.valid_file={args.valid_file}")
    if args.test_file:
        overrides.append(f"data.test_file={args.test_file}")
    if args.checkpoint_path:
        overrides.append(f"evaluation.checkpoint_path={args.checkpoint_path}")
    if args.dry_run is not None:
        overrides.append(f"runtime.dry_run={str(args.dry_run).lower()}")
    if args.load_model:
        overrides.append("runtime.load_model=true")
    overrides.extend(args.set)
    return overrides


def prepare_runtime(
    args: argparse.Namespace,
    *,
    stage_name: str,
    stage_config_name: str,
) -> PreparedRuntime:
    """Resolve config, paths, seed, logging, and manifest writing."""

    environment = _resolve_environment(args.environment)
    config_files = _base_config_files(stage_config_name, environment)
    config_files.extend((Path(path).expanduser() for path in args.config))

    merged_config = load_and_merge_config_files(config_files)
    merged_config = apply_dotlist_overrides(merged_config, _cli_overrides(args))
    merged_config = apply_dotlist_overrides(
        merged_config,
        [f"runtime.environment={environment}", f"run.stage={stage_name}"],
    )

    runtime_paths = build_runtime_paths(merged_config, stage_name=stage_name)
    config = inject_runtime_paths(merged_config, runtime_paths)

    seed = config.get("runtime", {}).get("seed")
    if isinstance(seed, int):
        set_global_seed(seed)

    context = RunContext(
        run_name=runtime_paths.run_dir.name,
        output_dir=runtime_paths.run_dir,
        stage=stage_name,
        seed=seed if isinstance(seed, int) else None,
        notes=(str(config.get("run", {}).get("notes", "")),),
    )
    logger = configure_run_logger(
        stage_name,
        context=context,
        log_dir=runtime_paths.log_dir,
        level=str(config.get("runtime", {}).get("log_level", "INFO")),
    )

    manifest = dict(config)
    manifest["config_files"] = [str(Path(path).resolve()) for path in config_files]
    write_runtime_manifest(runtime_paths.run_dir / "resolved_config.json", manifest)
    logger.info("Prepared %s runtime in %s", stage_name, runtime_paths.run_dir)
    return PreparedRuntime(
        config=config,
        manifest=manifest,
        runtime_paths=runtime_paths,
        logger=logger,
        context=context,
        config_files=tuple(str(Path(path).resolve()) for path in config_files),
    )


def maybe_load_local_artifacts(prepared: PreparedRuntime) -> None:
    """Optionally load model/tokenizer from local filesystem paths."""

    runtime_cfg = prepared.config.get("runtime", {})
    model_cfg = prepared.config.get("model", {})
    if not runtime_cfg.get("load_model", False):
        return

    model_path = model_cfg.get("model_path")
    tokenizer_path = model_cfg.get("tokenizer_path")
    if not model_path:
        raise ValueError("runtime.load_model=true but model.model_path is not configured.")

    artifacts = load_local_hf_artifacts(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        local_files_only=bool(runtime_cfg.get("local_files_only", True)),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    prepared.logger.info(
        "Loaded local model/tokenizer from %s and %s",
        artifacts.model_path,
        artifacts.tokenizer_path,
    )


def print_manifest_if_requested(prepared: PreparedRuntime, *, enabled: bool) -> None:
    """Print the resolved manifest as JSON when requested."""

    if enabled:
        print(json.dumps(prepared.manifest, indent=2, ensure_ascii=False, sort_keys=True))


def write_named_manifest(prepared: PreparedRuntime, name: str, payload: dict[str, Any]) -> None:
    """Write an additional JSON plan file inside the run directory."""

    write_runtime_manifest(prepared.runtime_paths.run_dir / name, payload)


def config_file_labels(paths: Sequence[str]) -> tuple[str, ...]:
    """Return filesystem-stable labels for config files."""

    return tuple(str(Path(path)) for path in paths)
