#!/usr/bin/env python3
"""Build one live, owner-bound snapshot for the early LLM launch gate.

The input runtime observation is written by the final16 controller.  This
entrypoint independently reopens the matrix pointer and canonical owner
registry, derives owner coverage, and emits no process or GPU mutation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.contracts import sha256_file  # noqa: E402
from src.ablations.launch_gate import (  # noqa: E402
    load_json_object,
    validate_matrix_authority_pointer,
)
from src.ablations.llm.early_launch_gate import EarlyLaunchSnapshot  # noqa: E402
from src.ablations.llm.final16_owner_evidence import (  # noqa: E402
    assert_snapshot_matches_owner_coverage,
    evaluate_final16_owner_coverage,
)


OBSERVATION_SCHEMA = "llm_early_runtime_observation_v1"


def _physical_file(path_like: Path, *, role: str) -> Path:
    lexical = path_like.expanduser()
    if not lexical.is_absolute() or lexical.is_symlink():
        raise ValueError(f"{role} must be an absolute physical file")
    path = lexical.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{role} is not a regular file")
    return path


def _list_of_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return tuple(value)


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if result < 0.0 or result != result or result in {float("inf"), -float("inf")}:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _runtime_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "main_ready_waiting_gpu",
        "main_publishers_waiting_gpu",
        "gpus",
        "persistent_free_gb",
        "minimum_persistent_free_gb",
        "memory_available_gb",
        "minimum_memory_available_gb",
        "checkpoint_resume_supported",
    }
    if set(payload) != required or payload.get("schema_version") != OBSERVATION_SCHEMA:
        raise ValueError("LLM early runtime observation schema/fields changed")
    gpus = payload.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise ValueError("runtime observation requires at least one GPU row")
    normalized_gpus: list[dict[str, Any]] = []
    seen: set[int] = set()
    gpu_fields = {
        "index",
        "idle_seconds",
        "compute_pids",
        "main_lease_held",
        "ablation_family",
    }
    for raw in gpus:
        if not isinstance(raw, Mapping) or set(raw) != gpu_fields:
            raise ValueError("runtime GPU row fields changed")
        index = raw.get("index")
        idle = raw.get("idle_seconds")
        pids = raw.get("compute_pids")
        main_lease = raw.get("main_lease_held")
        family = raw.get("ablation_family")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index <= 15
            or index in seen
            or not isinstance(idle, int)
            or isinstance(idle, bool)
            or idle < 0
            or not isinstance(pids, list)
            or any(
                not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
                for pid in pids
            )
            or main_lease not in (True, False)
            or family not in (None, "llm", "gnn")
        ):
            raise ValueError("runtime GPU row is invalid")
        seen.add(index)
        normalized_gpus.append(
            {
                "index": index,
                "idle_seconds": idle,
                "compute_pids": list(pids),
                "main_lease_held": main_lease,
                "ablation_family": family,
            }
        )
    return {
        "main_ready_waiting_gpu": _list_of_strings(
            payload["main_ready_waiting_gpu"], field="main_ready_waiting_gpu"
        ),
        "main_publishers_waiting_gpu": _list_of_strings(
            payload["main_publishers_waiting_gpu"],
            field="main_publishers_waiting_gpu",
        ),
        "gpus": normalized_gpus,
        "persistent_free_gb": _finite_nonnegative(
            payload["persistent_free_gb"], field="persistent_free_gb"
        ),
        "minimum_persistent_free_gb": _finite_nonnegative(
            payload["minimum_persistent_free_gb"],
            field="minimum_persistent_free_gb",
        ),
        "memory_available_gb": _finite_nonnegative(
            payload["memory_available_gb"], field="memory_available_gb"
        ),
        "minimum_memory_available_gb": _finite_nonnegative(
            payload["minimum_memory_available_gb"],
            field="minimum_memory_available_gb",
        ),
        "checkpoint_resume_supported": payload["checkpoint_resume_supported"] is True,
    }


def build_snapshot(
    *,
    matrix_path: Path,
    owner_registry_path: Path,
    runtime_observation: Mapping[str, Any],
    check_processes: bool = True,
) -> EarlyLaunchSnapshot:
    matrix_file = _physical_file(matrix_path, role="matrix authority pointer")
    owner_file = _physical_file(owner_registry_path, role="canonical owner registry")
    authority = validate_matrix_authority_pointer(load_json_object(matrix_file))
    authority["pointer_root"] = str(matrix_file.parent)
    coverage = evaluate_final16_owner_coverage(
        authority=authority,
        owner_registry=load_json_object(owner_file),
        check_processes=check_processes,
    )
    observation = _runtime_observation(runtime_observation)
    available = sorted(
        (
            row
            for row in observation["gpus"]
            if not row["compute_pids"]
            and row["main_lease_held"] is False
            and row["ablation_family"] is None
        ),
        key=lambda row: (-int(row["idle_seconds"]), int(row["index"])),
    )
    selected = available[0] if available else None
    active_llm = tuple(
        sorted(
            int(row["index"])
            for row in observation["gpus"]
            if row["ablation_family"] == "llm"
        )
    )

    def healthy(cell: str) -> bool:
        return cell in coverage.applied_cells or cell in coverage.healthy_owner_cells

    globalgce = "TasteMolNet/GlobalGCE"
    if globalgce in coverage.applied_cells or globalgce in coverage.pass_owner_cells:
        t8_state, t8_pid = "PASS", None
    elif globalgce in coverage.healthy_owner_cells:
        live = coverage.running_owner_pids_by_cell.get(globalgce, ())
        t8_state = "RUNNING"
        t8_pid = live[0] if live else None
    else:
        t8_state, t8_pid = "BLOCKED", None
    snapshot = EarlyLaunchSnapshot(
        matrix_complete_cells=int(authority["complete_cells"]),
        matrix_authority_path=str(matrix_file),
        matrix_authority_sha256=sha256_file(matrix_file),
        t8_t13_state=t8_state,
        t8_t13_science_pid=t8_pid,
        t12_healthy=healthy("TasteMolNet/GCFExplainer"),
        t14_healthy=healthy("TasteMolNet/ComRecGC"),
        # Despite the legacy field name, the final16 gate requires matrix PASS.
        mut_passed_or_gpu_released=(
            "Mutagenicity/ComRecGC" in coverage.applied_cells
        ),
        main_ready_waiting_gpu=observation["main_ready_waiting_gpu"],
        main_publishers_waiting_gpu=observation[
            "main_publishers_waiting_gpu"
        ],
        idle_gpu=int(selected["index"]) if selected is not None else None,
        idle_gpu_seconds=(int(selected["idle_seconds"]) if selected is not None else 0),
        persistent_free_gb=observation["persistent_free_gb"],
        minimum_persistent_free_gb=observation["minimum_persistent_free_gb"],
        memory_available_gb=observation["memory_available_gb"],
        minimum_memory_available_gb=observation["minimum_memory_available_gb"],
        checkpoint_resume_supported=observation["checkpoint_resume_supported"],
        requested_early_gpus=1,
        main_owner_registry_path=str(owner_file),
        main_owner_registry_sha256=sha256_file(owner_file),
        main_owner_registry_self_sha256=coverage.registry_self_sha256,
        all_incomplete_main_cells_owned=coverage.all_incomplete_cells_owned,
        unhealthy_or_unowned_main_cells=coverage.unhealthy_or_unowned_cells,
        missing_main_publisher_cells=coverage.missing_publisher_cells,
        active_early_llm_ablation_gpus=active_llm,
    )
    assert_snapshot_matches_owner_coverage(snapshot, coverage)
    return snapshot


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--matrix-authority", type=Path, required=True)
    parser.add_argument("--owner-registry", type=Path, required=True)
    parser.add_argument("--runtime-observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.config != "configs/hpc.yaml":
        raise ValueError("--config must be configs/hpc.yaml")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise ValueError("unsupported --set override")
    observation = load_json_object(
        _physical_file(args.runtime_observation, role="runtime observation")
    )
    snapshot = build_snapshot(
        matrix_path=args.matrix_authority,
        owner_registry_path=args.owner_registry,
        runtime_observation=observation,
    )
    _atomic_json(args.output, asdict(snapshot))
    print(json.dumps(asdict(snapshot), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
