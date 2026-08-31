#!/usr/bin/env python3
"""Prepare a runnable controller for one completed BACE ComRecGC cap handover."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autodl.build_four_by_four_manifest import compose_manifest  # noqa: E402
from src.baselines.bace_gnn_baseline_generic_adapter import (  # noqa: E402
    adapt_bace_baseline_controller_fragment,
    atomic_write_generic_fragment,
)
from src.baselines.comrecgc.contracts import sha256_file  # noqa: E402
from src.utils.autodl_bace_comrecgc_resource_cap_executor import (  # noqa: E402
    POSTPROCESS_SCHEMA,
)


MARKER = "[BACE_COMRECGC_RESOURCE_CAP_POSTPROCESS_MANIFEST_PASS]"
NATIVE_SCHEMA = "bace_baseline_controller_fragment_v1"
GENERATION_TASK_ID = "bace_comrecgc_train_generation"
RECOURSE_TASK_ID = "bace_comrecgc_train_common_recourse"
STANDARDIZED_TASK_ID = "bace_comrecgc_standardized"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _flag_value(arguments: list[str], flag: str) -> str:
    matches = [index for index, value in enumerate(arguments) if value == flag]
    if len(matches) != 1 or matches[0] + 1 >= len(arguments):
        raise ValueError(f"Resource-cap postprocess has malformed {flag}")
    return arguments[matches[0] + 1]


def _replace_prefix(value: str, source: str, replacement: str) -> tuple[str, bool]:
    if value == source:
        return replacement, True
    prefix = source.rstrip("/") + "/"
    if value.startswith(prefix):
        return replacement.rstrip("/") + value[len(source.rstrip("/")) :], True
    return value, False


def prepare(
    *,
    source_fragment: Path,
    generic_fragment_output: Path,
    manifest_output: Path,
    controller_id: str,
) -> dict[str, Any]:
    source_path = source_fragment.expanduser().resolve(strict=True)
    source = _json(source_path)
    tasks = source.get("tasks")
    if (
        source.get("schema_version") != POSTPROCESS_SCHEMA
        or source.get("dataset") != "bace"
        or source.get("method") != "ComRecGC"
        or source.get("method_id") != "comrecgc"
        or source.get("generation_adopted_from_checkpoint") is not True
        or source.get("generation_task_omitted") != GENERATION_TASK_ID
        or source.get("test_decision_input") is not False
        or source.get("terminal_task_ids") != [STANDARDIZED_TASK_ID]
        or not isinstance(tasks, list)
        or not tasks
        or any(type(task) is not dict for task in tasks)
    ):
        raise ValueError("BACE ComRecGC resource-cap fragment contract changed")
    by_id = {str(task.get("task_id") or ""): task for task in tasks}
    if len(by_id) != len(tasks) or GENERATION_TASK_ID in by_id:
        raise ValueError("Resource-cap fragment task identity is invalid")
    if RECOURSE_TASK_ID not in by_id or STANDARDIZED_TASK_ID not in by_id:
        raise ValueError("Resource-cap fragment lacks post-generation terminals")

    receipt = Path(str(source.get("resource_cap_receipt") or "")).expanduser()
    if not receipt.is_absolute():
        raise ValueError("Resource-cap receipt path is not absolute")
    receipt = receipt.resolve(strict=True)
    if sha256_file(receipt) != source.get("resource_cap_receipt_sha256"):
        raise ValueError("Resource-cap receipt hash changed")
    receipt_payload = _json(receipt)
    if (
        int(receipt_payload.get("M_effective", -1)) < 10_000
        or int(receipt_payload.get("M_effective", -1)) > 25_000
        or receipt_payload.get("test_loaded") is True
    ):
        raise ValueError("Resource-cap receipt escaped the authorized train-only budget")

    recourse_argv = by_id[RECOURSE_TASK_ID].get("argv")
    if not isinstance(recourse_argv, list) or any(
        not isinstance(value, str) for value in recourse_argv
    ):
        raise ValueError("Resource-cap common-recourse argv is invalid")
    generation = Path(_flag_value(recourse_argv, "--generation-dir")).expanduser()
    if not generation.is_absolute():
        raise ValueError("Adopted resource-cap generation root is not absolute")
    generation = generation.resolve(strict=True)
    generation_complete = generation / "_RUN_COMPLETE.json"
    generation_manifest = generation / "run_manifest.json"
    for required in (generation_complete, generation_manifest):
        if not required.is_file() or required.is_symlink():
            raise ValueError(f"Adopted generation artifact is invalid: {required}")
    complete = _json(generation_complete)
    manifest = _json(generation_manifest)
    effective = int(receipt_payload["M_effective"])
    if (
        complete.get("run_complete") is not True
        or int(complete.get("M_effective", -1)) != effective
        or int(manifest.get("M_effective", -1)) != effective
        or int(manifest.get("M_configured_max", -1)) != 20_000
        or manifest.get("calibration_loaded") is True
        or manifest.get("test_loaded") is True
        or manifest.get("rf_oracle_used") is not False
    ):
        raise ValueError("Adopted generation manifest differs from the cap receipt")

    sentinel = f"/__bace_comrecgc_resource_cap_{sha256_file(receipt)[:20]}__"
    native = dict(source)
    native["schema_version"] = NATIVE_SCHEMA
    rewritten_tasks: list[dict[str, Any]] = []
    replacement_count = 0
    generation_text = str(generation)
    for original in tasks:
        task = dict(original)
        argv = task.get("argv")
        if not isinstance(argv, list):
            raise ValueError(f"Native task has no argv: {task.get('task_id')}")
        rewritten: list[str] = []
        for value in argv:
            updated, changed = _replace_prefix(str(value), generation_text, sentinel)
            replacement_count += int(changed)
            rewritten.append(updated)
        task["argv"] = rewritten
        rewritten_tasks.append(task)
    if replacement_count < 1:
        raise ValueError("Resource-cap generation root is not consumed by postprocess")
    native["tasks"] = rewritten_tasks

    # The generic adapter still relocates every mutable postprocess cache and
    # task output into its attempt root.  Only the hash-closed, train-only
    # generation path is protected with a temporary sentinel and restored.
    generic = adapt_bace_baseline_controller_fragment(
        native,
        output_root=generation.parent,
    )
    restored_count = 0
    for task in generic["tasks"]:
        command = task.get("command")
        if not isinstance(command, list):
            continue
        restored: list[str] = []
        for value in command:
            updated, changed = _replace_prefix(str(value), sentinel, generation_text)
            restored_count += int(changed)
            restored.append(updated)
        task["command"] = restored
    encoded = json.dumps(generic, sort_keys=True)
    if restored_count != replacement_count or sentinel in encoded:
        raise ValueError("Resource-cap generation sentinel was not exactly restored")
    recourse = next(
        task for task in generic["tasks"] if task["id"] == RECOURSE_TASK_ID
    )
    if generation_text not in recourse["command"]:
        raise ValueError("Generic postprocess lost the adopted generation root")

    generic.update(
        {
            "source_postprocess_fragment": str(source_path),
            "source_postprocess_fragment_sha256": sha256_file(source_path),
            "resource_cap_receipt": str(receipt),
            "resource_cap_receipt_sha256": sha256_file(receipt),
            "adopted_generation_root": generation_text,
            "adopted_generation_manifest_sha256": sha256_file(generation_manifest),
            "adopted_generation_complete_sha256": sha256_file(generation_complete),
            "M_effective": effective,
            "test_decision_input": False,
        }
    )
    generic_path = atomic_write_generic_fragment(generic_fragment_output, generic)
    composed = compose_manifest(
        controller_id=controller_id,
        fragments=[generic_path],
        output=manifest_output,
    )
    return {
        **composed,
        "status": "PASS",
        "generic_fragment": str(generic_path),
        "generic_fragment_sha256": sha256_file(generic_path),
        "source_fragment": str(source_path),
        "source_fragment_sha256": sha256_file(source_path),
        "resource_cap_receipt": str(receipt),
        "resource_cap_receipt_sha256": sha256_file(receipt),
        "adopted_generation_root": generation_text,
        "M_effective": effective,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--source-fragment", type=Path, required=True)
    parser.add_argument("--generic-fragment-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--controller-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.set and args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError("Unsupported inference override")
    result = prepare(
        source_fragment=args.source_fragment,
        generic_fragment_output=args.generic_fragment_output,
        manifest_output=args.manifest_output,
        controller_id=args.controller_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
