"""Shared implementation for read-only ablation status entrypoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from src.ablations.contracts import ContractError, sha256_file
from src.ablations.launch_gate import evaluate_launch_gate, load_json_object


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMMON_CONFIG = PROJECT_ROOT / "configs/ablations/common_v1.yaml"


def add_status_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--common-config", type=Path, default=DEFAULT_COMMON_CONFIG)
    parser.add_argument("--family", choices=("llm", "gnn"), required=True)
    parser.add_argument("--matrix-authority", type=Path, required=True)
    parser.add_argument("--final-audit", type=Path)
    parser.add_argument("--figure3-pass", type=Path)
    parser.add_argument("--figure4-pass", type=Path)
    parser.add_argument("--table2-pass", type=Path)
    parser.add_argument("--authorization-receipt", type=Path)
    parser.add_argument("--run-requested", action="store_true")
    parser.add_argument("--output", type=Path)


def _optional_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return load_json_object(path)


def _load_common_config(path: Path) -> tuple[Path, dict[str, Any]]:
    lexical = path.expanduser()
    if not lexical.is_absolute():
        lexical = (PROJECT_ROOT / lexical).resolve(strict=False)
    if lexical.is_symlink() or not lexical.is_file():
        raise ContractError(f"common ablation config is not physical: {lexical}")
    resolved = lexical.resolve(strict=True)
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"invalid common ablation config: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ContractError("common ablation config must be a mapping")
    if (
        payload.get("schema_version") != "ablation_common_config_v1"
        or payload.get("framework_build_only") is not True
        or payload.get("main_matrix_total_cells") != 16
        or payload.get("explicit_run_authorization") is not False
    ):
        raise ContractError("common ablation config safety defaults changed")
    runtime = payload.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("gpu_lock_allowed_during_framework_build") is not False
    ):
        raise ContractError("common config no-GPU-lock contract changed")
    for field in ("run_llm_ablation", "run_gnn_ablation"):
        if not isinstance(payload.get(field), bool):
            raise ContractError(f"common config {field} must be a boolean")
    return resolved, dict(payload)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_status(args: argparse.Namespace) -> dict[str, Any]:
    common_path, common = _load_common_config(
        Path(getattr(args, "common_config", DEFAULT_COMMON_CONFIG))
    )
    configured_authority = Path(str(common.get("matrix_authority") or "")).expanduser()
    requested_authority = Path(args.matrix_authority).expanduser()
    if (
        not configured_authority.is_absolute()
        or configured_authority.resolve(strict=False)
        != requested_authority.resolve(strict=False)
    ):
        raise ContractError("matrix authority differs from the common ablation config")
    authority = load_json_object(args.matrix_authority)
    configured_run_requested = bool(common[f"run_{args.family}_ablation"])
    cli_run_requested = bool(args.run_requested)
    decision = evaluate_launch_gate(
        family=args.family,
        matrix_authority=authority,
        final_audit=_optional_payload(args.final_audit),
        figure3=_optional_payload(args.figure3_pass),
        figure4=_optional_payload(args.figure4_pass),
        table2=_optional_payload(args.table2_pass),
        authorization_receipt=_optional_payload(
            getattr(args, "authorization_receipt", None)
        ),
        run_requested=configured_run_requested and cli_run_requested,
    )
    payload = {
        **decision.to_dict(),
        "family": args.family,
        "framework_build_only": True,
        "science_started": False,
        "gpu_lock_acquired": False,
        "matrix_authority_path": str(args.matrix_authority.resolve()),
        "matrix_authority_mutated": False,
        "common_config_path": str(common_path),
        "common_config_sha256": sha256_file(common_path),
        "configured_run_requested": configured_run_requested,
        "cli_run_requested": cli_run_requested,
    }
    if args.output is not None:
        _atomic_json(args.output, payload)
    return payload


def status_main(family: str, description: str) -> int:
    parser = argparse.ArgumentParser(description=description)
    add_status_arguments(parser)
    args = parser.parse_args()
    if args.family != family:
        parser.error(f"this entrypoint requires --family {family}")
    payload = run_status(args)
    print(json.dumps(payload, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_COMMON_CONFIG",
    "add_status_arguments",
    "run_status",
    "status_main",
]
