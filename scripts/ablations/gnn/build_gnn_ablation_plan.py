#!/usr/bin/env python3
"""Validate and write one GNN-ablation plan without executing science."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.gnn import build_ablation_plan_from_config, stable_sha256  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan_document(
    *, ablation_config: Path, runtime_config: Path
) -> dict[str, Any]:
    ablation_path = ablation_config.expanduser().resolve(strict=True)
    runtime_path = runtime_config.expanduser().resolve(strict=True)
    plan = build_ablation_plan_from_config(
        ablation_path,
        project_root=PROJECT_ROOT,
    ).to_dict()
    payload: dict[str, Any] = {
        "schema_version": "gnn_ablation_config_only_plan_document_v1",
        "status": "PLANNED_NOT_RUN",
        "science_executed": False,
        "autodl_modified": False,
        "main_matrix_modified": False,
        "runtime_config_path": str(runtime_path),
        "runtime_config_sha256": _sha256_file(runtime_path),
        "ablation_config_path": str(ablation_path),
        "ablation_config_sha256": _sha256_file(ablation_path),
        "plan": plan,
    }
    payload["document_sha256"] = stable_sha256(payload)
    return payload


def write_plan_document(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser()
    if not target.is_absolute():
        raise ValueError("--output must be an absolute path")
    resolved = target.resolve(strict=False)
    protected_names = {
        "autodl-fs",
        "fast16_matrix_authority",
        "four_methods_four_datasets_v1",
        "paper_matrix",
    }
    if protected_names.intersection(resolved.parts):
        raise ValueError(
            "config-only plan output may not target AutoDL or main-matrix authority"
        )
    if target.exists():
        raise FileExistsError(f"plan output must be fresh: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--ablation-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.set:
        parser.error("--set is disabled: the checked-in ablation config is hash-closed")
    payload = build_plan_document(
        ablation_config=args.ablation_config,
        runtime_config=args.config,
    )
    write_plan_document(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "science_executed": False,
                "output": str(args.output),
                "document_sha256": payload["document_sha256"],
            },
            sort_keys=True,
        )
    )
    print("[GNN_ABLATION_CONFIG_ONLY_PLAN_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
