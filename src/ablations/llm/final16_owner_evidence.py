"""Bind the early LLM gate to the canonical owners of unfinished main cells.

This module is intentionally read-only.  It reopens the canonical final16
owner registry, verifies its live process identities, and projects the exact
set of unfinished matrix cells that still have a healthy owner and a unique
publisher.  It never starts a process, takes a GPU lock, or writes the matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.ablations.launch_gate import EXPECTED_MAIN_CELL_NAMES
from src.eval.four_by_four_registry import DATASETS, METHODS
from src.utils.final16_owner_registry_v1 import (
    RUNNING_STATES,
    validate_owner_registry,
)

from .contracts import LLMAblationContractError, canonical_json_sha256


HEALTHY_NONRUNNING_OWNER_STATES = {"PASS"}
HEALTHY_OWNER_STATES = RUNNING_STATES | HEALTHY_NONRUNNING_OWNER_STATES


def _canonical_cell(dataset: object, method: object) -> str | None:
    dataset_name = next(
        (item for item in DATASETS if item.casefold() == str(dataset or "").casefold()),
        None,
    )
    method_name = next(
        (item for item in METHODS if item.casefold() == str(method or "").casefold()),
        None,
    )
    if dataset_name is None or method_name is None:
        return None
    return f"{dataset_name}/{method_name}"


@dataclass(frozen=True, slots=True)
class Final16OwnerCoverage:
    registry_self_sha256: str
    matrix_authority_root: str
    applied_cells: tuple[str, ...]
    incomplete_cells: tuple[str, ...]
    healthy_owner_cells: tuple[str, ...]
    unhealthy_or_unowned_cells: tuple[str, ...]
    missing_publisher_cells: tuple[str, ...]
    running_owner_pids_by_cell: Mapping[str, tuple[int, ...]]
    pass_owner_cells: tuple[str, ...]
    all_incomplete_cells_owned: bool
    schema_version: str = "final16_owner_coverage_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "applied_cells",
            "incomplete_cells",
            "healthy_owner_cells",
            "unhealthy_or_unowned_cells",
            "missing_publisher_cells",
            "pass_owner_cells",
        ):
            payload[key] = list(payload[key])
        payload["running_owner_pids_by_cell"] = {
            cell: list(pids)
            for cell, pids in sorted(self.running_owner_pids_by_cell.items())
        }
        payload["coverage_sha256"] = canonical_json_sha256(payload)
        return payload


def evaluate_final16_owner_coverage(
    *,
    authority: Mapping[str, Any],
    owner_registry: Mapping[str, Any],
    proc_root: str | Path = "/proc",
    check_processes: bool = True,
) -> Final16OwnerCoverage:
    """Return live owner coverage for every cell absent from the authority."""

    required_authority = {
        "root",
        "complete_cells",
        "applied_cells",
    }
    if not required_authority.issubset(authority):
        raise LLMAblationContractError("verified matrix authority projection is incomplete")
    applied = tuple(str(cell) for cell in authority["applied_cells"])
    if len(applied) != int(authority["complete_cells"]):
        raise LLMAblationContractError("matrix applied-cell count changed")
    if len(applied) != len(set(applied)) or any(
        cell not in EXPECTED_MAIN_CELL_NAMES for cell in applied
    ):
        raise LLMAblationContractError("matrix applied-cell inventory is invalid")

    registry = validate_owner_registry(
        owner_registry,
        proc_root=proc_root,
        check_processes=check_processes,
    )
    authority_root = Path(str(authority["root"])).resolve(strict=False)
    registry_root = Path(str(registry["matrix_authority_root"])).resolve(strict=False)
    allowed_authority_roots = {authority_root}
    pointer_root = authority.get("pointer_root")
    if isinstance(pointer_root, str) and pointer_root:
        allowed_authority_roots.add(Path(pointer_root).resolve(strict=False))
    if registry_root not in allowed_authority_roots:
        raise LLMAblationContractError(
            "canonical owner registry belongs to another matrix authority"
        )

    incomplete = tuple(
        cell for cell in EXPECTED_MAIN_CELL_NAMES if cell not in set(applied)
    )
    tasks_by_cell: dict[str, list[Mapping[str, Any]]] = {
        cell: [] for cell in incomplete
    }
    for task in registry["tasks"]:
        cell = _canonical_cell(task.get("dataset"), task.get("method"))
        if cell in tasks_by_cell:
            tasks_by_cell[cell].append(task)

    running_pids: dict[str, tuple[int, ...]] = {}
    pass_cells: list[str] = []
    healthy_cells: list[str] = []
    unhealthy: list[str] = []
    for cell in incomplete:
        owners = [
            task
            for task in tasks_by_cell[cell]
            if task.get("owner_state") in HEALTHY_OWNER_STATES
        ]
        live_pids = tuple(
            sorted(
                int(task["owner_pid"])
                for task in owners
                if task.get("owner_state") in RUNNING_STATES
                and isinstance(task.get("owner_pid"), int)
                and not isinstance(task.get("owner_pid"), bool)
            )
        )
        if live_pids:
            running_pids[cell] = live_pids
        if any(task.get("owner_state") == "PASS" for task in owners):
            pass_cells.append(cell)
        if owners:
            healthy_cells.append(cell)
        else:
            unhealthy.append(cell)

    claimed_cells = {
        str(row["cell_id"])
        for row in registry["publishers"]
        if row.get("claim_enabled") is True
    }
    missing_publishers = [cell for cell in incomplete if cell not in claimed_cells]
    all_owned = not unhealthy and not missing_publishers
    return Final16OwnerCoverage(
        registry_self_sha256=str(registry["self_sha256"]),
        matrix_authority_root=str(registry_root),
        applied_cells=applied,
        incomplete_cells=incomplete,
        healthy_owner_cells=tuple(healthy_cells),
        unhealthy_or_unowned_cells=tuple(unhealthy),
        missing_publisher_cells=tuple(missing_publishers),
        running_owner_pids_by_cell=running_pids,
        pass_owner_cells=tuple(pass_cells),
        all_incomplete_cells_owned=all_owned,
    )


def assert_snapshot_matches_owner_coverage(
    snapshot: Any,
    coverage: Final16OwnerCoverage,
) -> None:
    """Reject a stale/hand-authored snapshot that disagrees with live owners."""

    if int(snapshot.matrix_complete_cells) != len(coverage.applied_cells):
        raise LLMAblationContractError("LLM snapshot matrix count is stale")
    if snapshot.main_owner_registry_self_sha256 != coverage.registry_self_sha256:
        raise LLMAblationContractError("LLM snapshot owner registry self-hash changed")
    if bool(snapshot.all_incomplete_main_cells_owned) != bool(
        coverage.all_incomplete_cells_owned
    ):
        raise LLMAblationContractError("LLM snapshot owner-health projection changed")
    if tuple(snapshot.unhealthy_or_unowned_main_cells) != tuple(
        coverage.unhealthy_or_unowned_cells
    ):
        raise LLMAblationContractError("LLM snapshot unhealthy-owner cells changed")
    if tuple(snapshot.missing_main_publisher_cells) != tuple(
        coverage.missing_publisher_cells
    ):
        raise LLMAblationContractError("LLM snapshot publisher coverage changed")

    mut_pass = "Mutagenicity/ComRecGC" in coverage.applied_cells
    if bool(snapshot.mut_passed_or_gpu_released) != mut_pass:
        raise LLMAblationContractError(
            "early LLM gate requires registered Mut PASS, not GPU release alone"
        )

    cell_rules = (
        ("TasteMolNet/GCFExplainer", bool(snapshot.t12_healthy), None),
        ("TasteMolNet/ComRecGC", bool(snapshot.t14_healthy), None),
        (
            "TasteMolNet/GlobalGCE",
            snapshot.t8_t13_state in {"RUNNING", "PASS"},
            snapshot.t8_t13_science_pid,
        ),
    )
    for cell, claimed_healthy, claimed_pid in cell_rules:
        is_pass = cell in coverage.applied_cells
        is_owned = cell in coverage.healthy_owner_cells
        if claimed_healthy != (is_pass or is_owned):
            raise LLMAblationContractError(f"LLM snapshot health changed for {cell}")
        if cell == "TasteMolNet/GlobalGCE":
            expected_state = (
                "PASS"
                if is_pass or cell in coverage.pass_owner_cells
                else "RUNNING"
                if is_owned
                else "BLOCKED"
            )
            if snapshot.t8_t13_state != expected_state:
                raise LLMAblationContractError("LLM snapshot T8/T13 state changed")
            if expected_state == "RUNNING" and claimed_pid not in (
                coverage.running_owner_pids_by_cell.get(cell, ())
            ):
                raise LLMAblationContractError(
                    "LLM snapshot T8/T13 PID is not a live canonical owner"
                )
            if expected_state != "RUNNING" and claimed_pid is not None:
                raise LLMAblationContractError(
                    "LLM snapshot T8/T13 PASS/BLOCKED state claims a PID"
                )


__all__ = [
    "Final16OwnerCoverage",
    "HEALTHY_OWNER_STATES",
    "assert_snapshot_matches_owner_coverage",
    "evaluate_final16_owner_coverage",
]
