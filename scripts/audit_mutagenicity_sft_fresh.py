#!/usr/bin/env python3
"""Audit Fresh Mutagenicity SFT provenance without loading the 7B model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.mutagenicity_fresh_sft import write_json_atomic  # noqa: E402


DEFAULT_FORBIDDEN_ADAPTER = Path(
    "outputs/hpc/sft_checkpoints/"
    "sft_v3_hiv_20260508_resplit_lr2e4_seed7_fix_columns/checkpoint-500"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--forbidden-adapter-checkpoint",
        type=Path,
        default=DEFAULT_FORBIDDEN_ADAPTER,
    )
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--report-md", type=Path, default=None)
    return parser


def _resolve(path: Path) -> Path:
    value = path.expanduser()
    return (REPO_ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_file(root: Path) -> Path | None:
    return next(
        (
            candidate
            for candidate in (
                root / "adapter_model.safetensors",
                root / "adapter_model.bin",
            )
            if candidate.is_file()
        ),
        None,
    )


def audit_fresh_output(
    output_root: Path,
    *,
    forbidden_adapter_checkpoint: Path | None,
) -> dict[str, Any]:
    init_path = output_root / "fresh_initialization_audit.json"
    manifest_path = output_root / "checkpoint_manifest.json"
    if not init_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Fresh SFT output is missing initialization/manifest artifacts: "
            f"{init_path}, {manifest_path}"
        )
    initialization = json.loads(init_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    required_truths = {
        "adapter_initialized_from_scratch": True,
        "aids_adapter_weights_loaded": False,
        "single_active_adapter": True,
        "initialization_audit_passed": True,
    }
    for key, expected in required_truths.items():
        if initialization.get(key) is not expected:
            errors.append(f"{key} expected {expected!r}, got {initialization.get(key)!r}")
    if initialization.get("source_adapter_checkpoint") is not None:
        errors.append("source_adapter_checkpoint is not null")
    if initialization.get("adapter_names") != ["default"]:
        errors.append(f"adapter_names={initialization.get('adapter_names')!r}")
    if initialization.get("active_adapters") != ["default"]:
        errors.append(f"active_adapters={initialization.get('active_adapters')!r}")
    if int(initialization.get("base_parameter_trainable_count", -1)) != 0:
        errors.append("base_parameter_trainable_count is not zero")
    if int(initialization.get("adapter_trainable_parameter_count", 0)) <= 0:
        errors.append("adapter_trainable_parameter_count is not positive")
    if manifest.get("source_adapter_checkpoint") is not None:
        errors.append("checkpoint manifest names a source adapter")
    if bool(manifest.get("aids_adapter_weights_loaded")):
        errors.append("checkpoint manifest reports AIDS weights loaded")

    forbidden_hash = None
    compared: list[dict[str, Any]] = []
    manifest_rows = [
        *manifest.get("checkpoints", []),
        manifest.get("final_adapter", {}),
    ]
    if not manifest.get("final_adapter"):
        errors.append("checkpoint manifest is missing final_adapter")
    for row in manifest_rows:
        if not row:
            continue
        weight = Path(str(row.get("adapter_weights") or ""))
        if not weight.is_file():
            errors.append(f"checkpoint weight missing: {weight}")
    if forbidden_adapter_checkpoint and not forbidden_adapter_checkpoint.is_dir():
        errors.append(
            "forbidden AIDS adapter checkpoint is unavailable for hash audit: "
            f"{forbidden_adapter_checkpoint}"
        )
    if forbidden_adapter_checkpoint and forbidden_adapter_checkpoint.is_dir():
        forbidden_weight = _weight_file(forbidden_adapter_checkpoint)
        if forbidden_weight is None:
            errors.append(
                "forbidden AIDS adapter checkpoint has no adapter weights: "
                f"{forbidden_adapter_checkpoint}"
            )
        else:
            forbidden_hash = _sha256(forbidden_weight)
            for row in manifest_rows:
                if not row:
                    continue
                weight = Path(str(row.get("adapter_weights") or ""))
                if not weight.is_file():
                    continue
                checkpoint_hash = _sha256(weight)
                same = checkpoint_hash == forbidden_hash
                compared.append(
                    {
                        "checkpoint": row.get("checkpoint"),
                        "adapter_sha256": checkpoint_hash,
                        "same_as_forbidden_aids_adapter": same,
                    }
                )
                if same:
                    errors.append(
                        f"fresh checkpoint is byte-identical to forbidden AIDS adapter: {weight}"
                    )

    return {
        "output_root": str(output_root),
        "initialization_audit_path": str(init_path),
        "checkpoint_manifest_path": str(manifest_path),
        "forbidden_adapter_checkpoint": (
            str(forbidden_adapter_checkpoint)
            if forbidden_adapter_checkpoint
            else None
        ),
        "forbidden_adapter_sha256": forbidden_hash,
        "checkpoint_weight_comparisons": compared,
        "provenance_errors": errors,
        "fresh_sft_audit_passed": not errors,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = _resolve(args.output_root)
    forbidden = _resolve(args.forbidden_adapter_checkpoint)
    payload = audit_fresh_output(
        output_root,
        forbidden_adapter_checkpoint=forbidden,
    )
    audit_json = _resolve(args.audit_json) if args.audit_json else output_root / "fresh_sft_audit.json"
    report_md = _resolve(args.report_md) if args.report_md else output_root / "fresh_sft_audit.md"
    write_json_atomic(audit_json, payload)
    report = [
        "# Mutagenicity Fresh SFT Audit",
        "",
        f"- Output: `{output_root}`",
        f"- Pure-base/fresh-LoRA provenance passed: {str(payload['fresh_sft_audit_passed']).lower()}",
        f"- Compared checkpoint weights: {len(payload['checkpoint_weight_comparisons'])}",
        f"- Provenance errors: {payload['provenance_errors']}",
        "",
    ]
    report_md.write_text("\n".join(report), encoding="utf-8")
    if not payload["fresh_sft_audit_passed"]:
        raise SystemExit(2)
    print("[MUTAGENICITY_FRESH_SFT_INITIALIZATION_AUDIT_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
