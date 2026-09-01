#!/usr/bin/env python3
"""Resume BACE GlobalGCE/ComRecGC strictly after frozen calibration selection.

This is a deliberately dataset-specific maintenance successor.  It adopts two
hash-closed, calibration-only selections and runs only held-out shards, merge,
final freeze, standardization, and the serialized fast16 matrix append.  It
never regenerates candidates, retrains a baseline, or reruns calibration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.bace_native_baseline_gnn import FINAL_STAGE, TEST_STAGE  # noqa: E402
from src.utils.autodl_runtime import (  # noqa: E402
    GPULockError,
    GPUFileLock,
    atomic_write_json,
    query_gpu_inventory,
    sha256_file,
    utc_now,
)


SCHEMA = "bace_heldout_closeout_successor_v1"
RECEIPT_SCHEMA = "bace_fresh_selection_test_adoption_v1"
EXPECTED_EMPTY_FIX_TOKEN = "UNEXPECTED_EMPTY_GRAPH_SEQUENCE"
EXPECTED_EMPTY_REASON = "NO_EVALUABLE_GRAPHS_AFTER_PRE_ORACLE_FILTERS"
SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
SHA256 = re.compile(r"[0-9a-f]{64}")
METHODS = (("GlobalGCE", "globalgce"), ("ComRecGC", "comrecgc"))


class BaceHeldoutCloseoutError(RuntimeError):
    """The frozen-selection successor cannot proceed safely."""


@dataclass(frozen=True)
class Config:
    project_root: Path
    python: Path
    runtime_root: Path
    controller_id: str
    control_dir: Path
    output_root: Path
    source_root: Path
    selection_receipt: Path
    expected_receipt_sha256: str
    gnn_checkpoint: Path
    test_split: Path
    molclr_root: Path
    molclr_checkpoint: Path
    matrix_authority_state: Path
    matrix_authority_lock: Path
    gpu_index: int
    min_free_memory_mb: int
    poll_seconds: float


@dataclass(frozen=True)
class Stage:
    task_id: str
    kind: str
    method: str
    output: Path
    command: tuple[str, ...]
    gpu: bool = False
    shard_index: int | None = None


def _absolute(value: str, *, existing: bool, kind: str = "path") -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute {kind} required: {value}")
    try:
        return path.resolve(strict=existing)
    except FileNotFoundError as exc:
        raise argparse.ArgumentTypeError(f"required {kind} is absent: {path}") from exc


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BaceHeldoutCloseoutError(f"required physical JSON is absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaceHeldoutCloseoutError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BaceHeldoutCloseoutError(f"JSON must contain one object: {path}")
    return value


def _regular_nonempty(path: Path) -> bool:
    return not path.is_symlink() and path.is_file() and path.stat().st_size > 0


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        return False
    return True


def _verify_file_identity(
    raw: Any, *, label: str, expected_path: Path | None = None
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise BaceHeldoutCloseoutError(f"{label} identity is absent")
    unresolved = Path(str(raw.get("path") or "")).expanduser()
    if not unresolved.is_absolute() or unresolved.is_symlink():
        raise BaceHeldoutCloseoutError(f"{label} identity path is not physical")
    path = unresolved.resolve(strict=True)
    if not path.is_file() or (expected_path is not None and path != expected_path):
        raise BaceHeldoutCloseoutError(f"{label} identity path changed")
    actual = {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
    if actual["size"] != raw.get("size") or actual["sha256"] != raw.get("sha256"):
        raise BaceHeldoutCloseoutError(f"{label} identity changed")
    return actual


def _validate_execution_inputs(
    config: Config, *, expected_oracle_hash: str
) -> dict[str, Any]:
    """Bind the runtime classifier and held-out split before any shard starts."""

    checkpoint = config.gnn_checkpoint.resolve(strict=True)
    if not checkpoint.is_dir():
        raise BaceHeldoutCloseoutError(
            "BACE GINE checkpoint must be one physical bundle directory"
        )
    model = checkpoint / "model.pt"
    model_card_path = checkpoint / "model_card.json"
    split_manifest_path = checkpoint / "split_manifest.json"
    if any(
        not _regular_nonempty(path)
        for path in (model, model_card_path, split_manifest_path)
    ):
        raise BaceHeldoutCloseoutError(
            "BACE GINE bundle lacks model.pt/model_card.json/split_manifest.json"
        )

    model_hash = sha256_file(model)
    card = _json(model_card_path)
    required_card = {
        "dataset": "bace",
        "backbone": "gine",
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "num_classes": 2,
        "source_label": 1,
        "checkpoint_id": expected_oracle_hash,
    }
    changed = [
        field
        for field, expected in required_card.items()
        if card.get(field) != expected
    ]
    if changed:
        raise BaceHeldoutCloseoutError(
            "runtime BACE GINE contract changed: " + ", ".join(changed)
        )
    if model_hash != expected_oracle_hash:
        raise BaceHeldoutCloseoutError(
            "runtime BACE GINE model.pt differs from the frozen selections"
        )

    split_manifest = _json(split_manifest_path)
    files = split_manifest.get("files")
    test_identity = files.get("test") if isinstance(files, Mapping) else None
    if not isinstance(test_identity, Mapping):
        raise BaceHeldoutCloseoutError(
            "BACE GINE split manifest lacks a held-out test identity"
        )
    declared_raw = Path(str(test_identity.get("path") or "")).expanduser()
    if not declared_raw.is_absolute() or declared_raw.is_symlink():
        raise BaceHeldoutCloseoutError(
            "BACE GINE declared test split is not one physical absolute path"
        )
    declared = declared_raw.resolve(strict=True)
    actual_test = config.test_split.resolve(strict=True)
    if not declared.is_file() or declared != actual_test:
        raise BaceHeldoutCloseoutError(
            "runtime held-out test split path differs from the frozen GINE bundle"
        )
    declared_hash = str(test_identity.get("sha256") or "").lower()
    actual_test_hash = sha256_file(actual_test)
    if not SHA256.fullmatch(declared_hash) or declared_hash != actual_test_hash:
        raise BaceHeldoutCloseoutError(
            "runtime held-out test split bytes differ from the frozen GINE bundle"
        )
    return {
        "gnn_checkpoint": str(checkpoint),
        "oracle_checkpoint_hash": model_hash,
        "model_card_sha256": sha256_file(model_card_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "test_split": str(actual_test),
        "test_split_sha256": actual_test_hash,
    }


def _worktree_commit(project: Path) -> str:
    commit = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
    ).strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise BaceHeldoutCloseoutError("execution commit is not a full SHA")
    dirty = subprocess.check_output(
        ["git", "-C", str(project), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise BaceHeldoutCloseoutError("execution worktree must be clean")
    evaluator = (project / "src/eval/bace_native_baseline_gnn.py").read_text(
        encoding="utf-8"
    )
    oracle = (project / "src/oracles/gnn_oracle.py").read_text(encoding="utf-8")
    if EXPECTED_EMPTY_FIX_TOKEN not in evaluator or any(
        token not in oracle for token in (EXPECTED_EMPTY_FIX_TOKEN, EXPECTED_EMPTY_REASON)
    ):
        raise BaceHeldoutCloseoutError("expected-empty BACE fix is absent")
    return commit


def _open_write_descriptors(
    root: Path, proc_root: Path = Path("/proc")
) -> list[dict[str, Any]]:
    if not proc_root.is_dir():
        return []
    writers: list[dict[str, Any]] = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            descriptors = list((proc / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = Path(os.readlink(descriptor))
                if not target.is_absolute() or not _inside(target, root):
                    continue
                flags_line = next(
                    line
                    for line in (proc / "fdinfo" / descriptor.name)
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.startswith("flags:")
                )
                flags = int(flags_line.split()[1], 8)
                if flags & os.O_ACCMODE in {os.O_WRONLY, os.O_RDWR}:
                    writers.append(
                        {
                            "pid": int(proc.name),
                            "fd": int(descriptor.name),
                            "path": str(target),
                        }
                    )
            except (OSError, StopIteration, ValueError):
                continue
    return writers


def validate_selection_adoption(config: Config) -> dict[str, Any]:
    """Reopen every byte frozen by the pre-maintenance adoption receipt."""

    receipt_path = config.selection_receipt.resolve(strict=True)
    receipt_sha = sha256_file(receipt_path)
    if receipt_sha != config.expected_receipt_sha256:
        raise BaceHeldoutCloseoutError(
            "selection adoption receipt SHA changed: "
            f"{receipt_sha} != {config.expected_receipt_sha256}"
        )
    receipt = _json(receipt_path)
    required_top = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS",
        "selection_frozen_before_test": True,
        "test_loaded": False,
        "source_root": str(config.source_root),
    }
    changed = [
        field
        for field, expected in required_top.items()
        if receipt.get(field) != expected
    ]
    if changed:
        raise BaceHeldoutCloseoutError(
            "selection adoption top-level contract changed: " + ", ".join(changed)
        )
    source_controllers = receipt.get("source_controller_manifests")
    if not isinstance(source_controllers, Mapping) or set(source_controllers) != {
        slug for _, slug in METHODS
    }:
        raise BaceHeldoutCloseoutError("selection source-controller set changed")
    controller_evidence: dict[str, dict[str, Any]] = {}
    for _method, slug in METHODS:
        identity = source_controllers[slug]
        if not isinstance(identity, Mapping):
            raise BaceHeldoutCloseoutError(f"{slug} source controller is absent")
        path = Path(str(identity.get("path") or "")).expanduser()
        if not path.is_absolute() or path.is_symlink():
            raise BaceHeldoutCloseoutError(
                f"{slug} source controller path is not physical"
            )
        path = path.resolve(strict=True)
        if not path.is_file() or sha256_file(path) != identity.get("sha256"):
            raise BaceHeldoutCloseoutError(
                f"{slug} source controller identity changed"
            )
        controller_evidence[slug] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    methods = receipt.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != {
        slug for _, slug in METHODS
    }:
        raise BaceHeldoutCloseoutError("selection adoption method set changed")
    oracle_hashes: set[str] = set()
    molclr_hashes: set[str] = set()
    selection_evidence: dict[str, Any] = {}
    for method, slug in METHODS:
        record = methods.get(slug)
        if not isinstance(record, Mapping):
            raise BaceHeldoutCloseoutError(f"missing {method} selection receipt")
        root = (config.source_root / slug / "selection-shared").resolve(strict=True)
        if str(root) != str(record.get("root")):
            raise BaceHeldoutCloseoutError(f"{method} selection root changed")
        writers = _open_write_descriptors(root)
        if writers:
            raise BaceHeldoutCloseoutError(
                f"{method} frozen selection has active writers: {writers}"
            )
        calibration_root = (
            config.source_root / slug / "calibration-merged"
        ).resolve(strict=True)
        calibration_pair = _verify_file_identity(
            record.get("calibration_pair_matrix"),
            label=f"{method} calibration pair matrix",
            expected_path=calibration_root / "pair_matrix.jsonl",
        )
        calibration_manifest = _verify_file_identity(
            record.get("calibration_run_manifest"),
            label=f"{method} calibration run manifest",
            expected_path=calibration_root / "run_manifest.json",
        )
        frozen_identity = _verify_file_identity(
            record.get("frozen_selection_manifest"),
            label=f"{method} frozen selection manifest",
            expected_path=root / "frozen_selection_manifest.json",
        )
        selected_identity = _verify_file_identity(
            record.get("selected_top20"),
            label=f"{method} selected top20",
            expected_path=root / "selected_top20.json",
        )
        inventory = record.get("source_inventory")
        if not isinstance(inventory, Mapping) or not inventory:
            raise BaceHeldoutCloseoutError(f"{method} source inventory is absent")
        verified_inventory: dict[str, dict[str, Any]] = {}
        for relative, raw_identity in inventory.items():
            if not isinstance(relative, str) or not isinstance(raw_identity, Mapping):
                raise BaceHeldoutCloseoutError(f"{method} inventory entry is malformed")
            path = root / relative
            if not _inside(path, root) or not _regular_nonempty(path):
                raise BaceHeldoutCloseoutError(
                    f"{method} frozen selection file is absent/nonphysical: {path}"
                )
            actual = {"size": path.stat().st_size, "sha256": sha256_file(path)}
            expected = {
                "size": raw_identity.get("size"),
                "sha256": raw_identity.get("sha256"),
            }
            if actual != expected:
                raise BaceHeldoutCloseoutError(
                    f"{method} frozen selection identity changed: {path}"
                )
            verified_inventory[relative] = actual
        if (root / "PASS").read_text(encoding="utf-8").strip() != "PASS":
            raise BaceHeldoutCloseoutError(f"{method} selection PASS marker changed")
        complete = _json(root / "_RUN_COMPLETE.json")
        frozen = _json(root / "frozen_selection_manifest.json")
        required_frozen = {
            "dataset": "bace",
            "method": method,
            "method_id": slug,
            "stage": "BASELINE_CALIBRATION_SELECTOR",
            "status": "FROZEN",
            "selection_frozen": True,
            "selector_fitted_on_calibration": True,
            "calibration_loaded": True,
            "test_loaded": False,
            "oracle_backend": "gnn",
            "rf_oracle_used": False,
            "effective_rule_count": 20,
        }
        bad = [
            field
            for field, expected in required_frozen.items()
            if frozen.get(field) != expected
        ]
        if bad or complete.get("test_loaded") is not False:
            raise BaceHeldoutCloseoutError(
                f"{method} frozen selection contract changed: {bad}"
            )
        oracle_hash = str(frozen.get("oracle_checkpoint_hash") or "")
        molclr_hash = str(frozen.get("molclr_checkpoint_hash") or "")
        if not SHA256.fullmatch(oracle_hash) or not SHA256.fullmatch(molclr_hash):
            raise BaceHeldoutCloseoutError(f"{method} model identities are malformed")
        if record.get("oracle_checkpoint_hash") != oracle_hash or record.get(
            "molclr_checkpoint_hash"
        ) != molclr_hash:
            raise BaceHeldoutCloseoutError(f"{method} receipt/model identity changed")
        if record.get("threshold_config_hash") != frozen.get(
            "threshold_config_hash"
        ):
            raise BaceHeldoutCloseoutError(f"{method} threshold identity changed")
        oracle_hashes.add(oracle_hash)
        molclr_hashes.add(molclr_hash)
        selection_evidence[slug] = {
            "root": str(root),
            "frozen_selection_manifest_sha256": sha256_file(
                root / "frozen_selection_manifest.json"
            ),
            "selected_top20_sha256": sha256_file(root / "selected_top20.json"),
            "inventory_count": len(verified_inventory),
            "active_write_descriptors": [],
            "calibration_pair_matrix": calibration_pair,
            "calibration_run_manifest": calibration_manifest,
            "frozen_selection_manifest": frozen_identity,
            "selected_top20": selected_identity,
        }
    if len(oracle_hashes) != 1 or len(molclr_hashes) != 1:
        raise BaceHeldoutCloseoutError("BACE selections do not share model identities")
    execution_inputs = _validate_execution_inputs(
        config, expected_oracle_hash=next(iter(oracle_hashes))
    )
    actual_molclr = sha256_file(config.molclr_checkpoint)
    if actual_molclr != next(iter(molclr_hashes)):
        raise BaceHeldoutCloseoutError("MolCLR checkpoint differs from frozen selections")
    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "receipt": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "source_root": str(config.source_root),
        "oracle_checkpoint_hash": next(iter(oracle_hashes)),
        "molclr_checkpoint_hash": actual_molclr,
        "execution_inputs": execution_inputs,
        "methods": selection_evidence,
        "source_controller_manifests": controller_evidence,
        "test_loaded": False,
        "generation_replayed": False,
        "calibration_replayed": False,
        "gnn_ablation_started": False,
    }


def _required_files(kind: str, method: str) -> tuple[str, ...]:
    common: dict[str, tuple[str, ...]] = {
        "shard": (
            "PASS",
            "pair_details.jsonl",
            "pair_details.csv",
            "oracle_provenance.json",
            "run_manifest.json",
        ),
        "merge": (
            "PASS",
            "pair_matrix.jsonl",
            "selected_candidate_universe.jsonl",
            "summary.json",
            "run_manifest.json",
        ),
        "final": (
            "PASS",
            "final_metrics.json",
            "prefix_metrics.csv",
            "FINAL_PASS.json",
            "run_manifest.json",
        ),
        "standardized": (
            "PASS",
            "figure3_coverage_vs_k.csv",
            "figure4_coverage_vs_threshold.csv",
            "prefix_metrics.csv",
            "prefix_metrics.json",
            "parent_best_distances.csv",
            "destination_distribution.csv",
            "summary.json",
            "run_manifest.json",
            "oracle_manifest.json",
            "evaluation_manifest.json",
            "artifact_manifest.json",
            "freeze_manifest.json",
            "_FINALIZED.json",
            "final_artifact_audit.json",
        ),
    }
    result = common[kind]
    if kind == "standardized":
        result += (f"table2_{method.lower()}_k10.csv",)
    return result


def terminal_valid(
    root: Path, *, kind: str, method: str, shard_index: int | None = None
) -> bool:
    try:
        if root.is_symlink() or not root.is_dir():
            return False
        if any(not _regular_nonempty(root / name) for name in _required_files(kind, method)):
            return False
        if (root / "PASS").read_text(encoding="utf-8").strip() != "PASS":
            return False
        manifest = _json(root / "run_manifest.json")
        expected_dataset = "BACE" if kind == "standardized" else "bace"
        if (
            manifest.get("status") != "PASS"
            or manifest.get("dataset") != expected_dataset
            or manifest.get("method") != method
            or manifest.get("rf_oracle_used") is not False
        ):
            return False
        if kind in {"shard", "merge"}:
            if (
                manifest.get("stage") != TEST_STAGE
                or manifest.get("test_loaded") is not True
                or manifest.get("selection_frozen_before_test") is not True
                or manifest.get("run_complete") is not True
            ):
                return False
            if kind == "shard" and manifest.get("shard_index") != shard_index:
                return False
        elif kind == "final":
            if (
                manifest.get("stage") != FINAL_STAGE
                or manifest.get("selection_frozen_before_test") is not True
                or manifest.get("test_used_only_after_freeze") is not True
                or manifest.get("run_complete") is not True
            ):
                return False
            if _json(root / "FINAL_PASS.json") != manifest:
                return False
        else:
            audit = _json(root / "final_artifact_audit.json")
            finalized = _json(root / "_FINALIZED.json")
            if (
                audit.get("passed") is not True
                or audit.get("final_artifact_audit_passed") is not True
                or audit.get("raw_test_opened") is not False
                or finalized.get("status") != "PASS"
                or finalized.get("raw_test_opened") is not False
            ):
                return False
        return True
    except (BaceHeldoutCloseoutError, OSError, ValueError, TypeError, KeyError):
        return False


def choose_attempt(
    base: Path, *, kind: str, method: str, shard_index: int | None = None
) -> tuple[Path, bool]:
    for attempt in range(1000):
        candidate = base / f"attempt-{attempt}"
        if terminal_valid(
            candidate, kind=kind, method=method, shard_index=shard_index
        ):
            return candidate, True
        if not candidate.exists() and not candidate.is_symlink():
            return candidate, False
    raise BaceHeldoutCloseoutError(f"no fresh attempt remains under {base}")


def build_method_stages(config: Config, method: str, slug: str) -> list[Stage]:
    route = config.project_root / "scripts/autodl/run_bace_baseline_gnn_route.py"
    standardize = (
        config.project_root / "scripts/autodl/standardize_bace_frozen_cell.py"
    )
    selection = config.source_root / slug / "selection-shared"
    shards: list[Path] = []
    stages: list[Stage] = []
    for shard in range(4):
        output, _ = choose_attempt(
            config.output_root / slug / "test" / f"shard-{shard}",
            kind="shard",
            method=method,
            shard_index=shard,
        )
        shards.append(output)
        command = (
            str(config.python),
            str(route),
            "verify-shard",
            "--method",
            method,
            "--gnn-checkpoint",
            str(config.gnn_checkpoint),
            "--output-dir",
            str(output),
            "--verification-stage",
            TEST_STAGE,
            "--split-path",
            str(config.test_split),
            "--predecessor-output",
            str(selection),
            "--molclr-root",
            str(config.molclr_root),
            "--molclr-checkpoint",
            str(config.molclr_checkpoint),
            "--shard-index",
            str(shard),
            "--wnode-cache-db",
            str(output / "_native_aux/test/cache" / f"shard-{shard}.sqlite3"),
            "--node-embedding-cache-dir",
            str(output / "_native_aux/test/cache" / f"node-emb-shard-{shard}"),
            "--device",
            "cuda:0",
        )
        stages.append(
            Stage(
                task_id=f"bace_{slug}_test_shard_{shard}",
                kind="shard",
                method=method,
                output=output,
                command=command,
                gpu=True,
                shard_index=shard,
            )
        )
    merged, _ = choose_attempt(
        config.output_root / slug / "test/merged", kind="merge", method=method
    )
    stages.append(
        Stage(
            task_id=f"bace_{slug}_test_merge",
            kind="merge",
            method=method,
            output=merged,
            command=(
                str(config.python),
                str(route),
                "merge",
                "--method",
                method,
                "--verification-stage",
                TEST_STAGE,
                "--predecessor-output",
                str(selection),
                "--output-dir",
                str(merged),
                *(value for shard in shards for value in ("--shard-dir", str(shard))),
            ),
        )
    )
    final, _ = choose_attempt(
        config.output_root / slug / "final", kind="final", method=method
    )
    stages.append(
        Stage(
            task_id=f"bace_{slug}_final_freeze",
            kind="final",
            method=method,
            output=final,
            command=(
                str(config.python),
                str(route),
                "freeze",
                "--method",
                method,
                "--selection-output",
                str(selection),
                "--test-output",
                str(merged),
                "--output-dir",
                str(final),
            ),
        )
    )
    standardized, _ = choose_attempt(
        config.output_root / slug / "standardized",
        kind="standardized",
        method=method,
    )
    stages.append(
        Stage(
            task_id=f"bace_{slug}_standardized",
            kind="standardized",
            method=method,
            output=standardized,
            command=(
                str(config.python),
                str(standardize),
                "--config",
                "configs/hpc.yaml",
                "--method",
                method,
                "--source-final-root",
                str(final),
                "--gnn-checkpoint",
                str(config.gnn_checkpoint),
                "--output-dir",
                str(standardized),
            ),
        )
    )
    return stages


class Runner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.execution_commit = ""
        self.active: subprocess.Popen[bytes] | None = None
        self.active_stage: Stage | None = None
        self.stop_requested = False
        self.gpu_lock: GPUFileLock | None = None
        self.owner_handle: TextIO | None = None

    def _heartbeat(
        self, state: str, *, stage: Stage | None = None, detail: str | None = None
    ) -> None:
        atomic_write_json(
            self.config.control_dir / "heartbeat.json",
            {
                "schema_version": SCHEMA,
                "controller_id": self.config.controller_id,
                "controller_pid": os.getpid(),
                "execution_commit": self.execution_commit,
                "state": state,
                "task": stage.task_id if stage else None,
                "child_pid": self.active.pid if self.active else None,
                "output": str(stage.output) if stage else None,
                "output_root": str(self.config.output_root),
                "gpu_index": self.config.gpu_index,
                "detail": detail,
                "updated_at": utc_now(),
            },
        )

    def _signal(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True
        if self.active is not None and self.active.poll() is None:
            self.active.terminate()

    def _acquire_owner(self) -> None:
        path = self.config.control_dir.parent / "successor.owner.lock"
        if path.is_symlink():
            raise BaceHeldoutCloseoutError(f"owner lock may not be a symlink: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise BaceHeldoutCloseoutError(
                "another BACE held-out successor is live"
            ) from exc
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "schema_version": SCHEMA,
                "controller_id": self.config.controller_id,
                "pid": os.getpid(),
                "acquired_at": utc_now(),
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        self.owner_handle = handle

    def _release_owner(self) -> None:
        if self.owner_handle is None:
            return
        self.owner_handle.seek(0)
        self.owner_handle.truncate()
        json.dump(
            {
                "schema_version": SCHEMA,
                "controller_id": self.config.controller_id,
                "pid": os.getpid(),
                "state": "RELEASED",
                "released_at": utc_now(),
            },
            self.owner_handle,
            sort_keys=True,
        )
        self.owner_handle.write("\n")
        self.owner_handle.flush()
        os.fsync(self.owner_handle.fileno())
        fcntl.flock(self.owner_handle.fileno(), fcntl.LOCK_UN)
        self.owner_handle.close()
        self.owner_handle = None

    def _acquire_gpu(self) -> None:
        lock_root = self.config.runtime_root / "locks"
        while not self.stop_requested:
            observations = {
                row.index: row for row in query_gpu_inventory()
            }
            gpu = observations.get(self.config.gpu_index)
            if gpu is None:
                raise BaceHeldoutCloseoutError(
                    f"physical GPU {self.config.gpu_index} is absent"
                )
            if gpu.is_idle(
                min_free_memory_mb=self.config.min_free_memory_mb,
                max_utilization_percent=10,
            ):
                lock = GPUFileLock(
                    lock_root,
                    gpu_index=gpu.index,
                    gpu_uuid=gpu.uuid,
                    owner={
                        "controller_id": self.config.controller_id,
                        "task": "BACE_HELDOUT_CLOSEOUT",
                    },
                )
                try:
                    lock.acquire()
                except GPULockError:
                    pass
                else:
                    reopened = {
                        row.index: row for row in query_gpu_inventory()
                    }[self.config.gpu_index]
                    if reopened.is_idle(
                        min_free_memory_mb=self.config.min_free_memory_mb,
                        max_utilization_percent=10,
                    ):
                        self.gpu_lock = lock
                        self._heartbeat("GPU_ACQUIRED", detail=gpu.uuid)
                        return
                    lock.release()
            self._heartbeat("WAITING_GPU", detail=f"gpu={self.config.gpu_index}")
            time.sleep(self.config.poll_seconds)
        raise BaceHeldoutCloseoutError("controller stop requested while waiting for GPU")

    def _run_stage(self, stage: Stage) -> None:
        if terminal_valid(
            stage.output,
            kind=stage.kind,
            method=stage.method,
            shard_index=stage.shard_index,
        ):
            self._heartbeat("ADOPTED", stage=stage)
            return
        if stage.output.exists() or stage.output.is_symlink():
            raise BaceHeldoutCloseoutError(
                f"selected stage output ceased to be fresh: {stage.output}"
            )
        stage.output.parent.mkdir(parents=True, exist_ok=True)
        log_path = self.config.control_dir / "logs" / f"{stage.task_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONPATH": str(self.config.project_root),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "RUN_GNN_ABLATION": "0",
                "CUDA_VISIBLE_DEVICES": str(self.config.gpu_index),
            }
        )
        with log_path.open("ab", buffering=0) as log:
            self.active_stage = stage
            self.active = subprocess.Popen(
                stage.command,
                cwd=self.config.project_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            self._heartbeat("RUNNING", stage=stage)
            while self.active.poll() is None:
                if self.stop_requested:
                    self.active.terminate()
                time.sleep(self.config.poll_seconds)
                self._heartbeat(
                    "STOPPING" if self.stop_requested else "RUNNING", stage=stage
                )
            returncode = self.active.wait()
        self.active = None
        self.active_stage = None
        if self.stop_requested:
            raise BaceHeldoutCloseoutError("controller stopped by TERM/INT")
        if returncode != 0 or not terminal_valid(
            stage.output,
            kind=stage.kind,
            method=stage.method,
            shard_index=stage.shard_index,
        ):
            raise BaceHeldoutCloseoutError(
                f"{stage.task_id} failed terminal verification; see {log_path}"
            )
        self._heartbeat("PASS", stage=stage)

    def _matrix_has_terminal(self, method: str, terminal: Path) -> bool:
        state = _json(self.config.matrix_authority_state)
        cell = f"BACE/{method}"
        if cell not in state.get("applied_cells", []):
            return False
        root = Path(str(state.get("latest_authority_root") or "")).resolve(strict=True)
        matrix = _json(root / "matrix_status.json")
        rows = [
            row
            for row in matrix.get("cells", [])
            if row.get("dataset") == "BACE" and row.get("method") == method
        ]
        return len(rows) == 1 and Path(
            str(rows[0].get("standardized_output_root") or "")
        ).resolve(strict=True) == terminal.resolve(strict=True)

    def _append_matrix(self, method: str, slug: str, terminal: Path) -> None:
        if self._matrix_has_terminal(method, terminal):
            self._heartbeat("MATRIX_ALREADY_APPLIED", detail=f"BACE/{method}")
            return
        append_root = (
            self.config.output_root
            / slug
            / "matrix-append"
            / f"attempt-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
        )
        command = (
            str(self.config.python),
            str(
                self.config.project_root
                / "scripts/autodl/append_non_taste_matrix_authority.py"
            ),
            "--config",
            "configs/hpc.yaml",
            "--set",
            "inference.fallback_to_heuristic=false",
            "--dataset",
            "BACE",
            "--method",
            method,
            "--cell-terminal-root",
            str(terminal),
            "--authority-state-path",
            str(self.config.matrix_authority_state),
            "--authority-lock-path",
            str(self.config.matrix_authority_lock),
            "--output-root",
            str(append_root),
        )
        log = self.config.control_dir / "logs" / f"bace_{slug}_matrix_append.log"
        self._heartbeat("MATRIX_APPEND_RUNNING", detail=f"BACE/{method}")
        with log.open("ab", buffering=0) as handle:
            result = subprocess.run(
                command,
                cwd=self.config.project_root,
                env={
                    **os.environ,
                    "PYTHONPATH": str(self.config.project_root),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "RUN_GNN_ABLATION": "0",
                },
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0 or not self._matrix_has_terminal(method, terminal):
            raise BaceHeldoutCloseoutError(
                f"BACE/{method} matrix append failed; see {log}"
            )
        self._heartbeat("MATRIX_APPLIED", detail=f"BACE/{method}")

    def run(self) -> dict[str, Any]:
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)
        self._acquire_owner()
        try:
            self.execution_commit = _worktree_commit(self.config.project_root)
            self.config.control_dir.mkdir(parents=True, exist_ok=True)
            self.config.output_root.mkdir(parents=True, exist_ok=True)
            evidence = validate_selection_adoption(self.config)
            manifest = {
                **evidence,
                "controller_id": self.config.controller_id,
                "controller_pid": os.getpid(),
                "execution_commit": self.execution_commit,
                "control_dir": str(self.config.control_dir),
                "output_root": str(self.config.output_root),
                "test_split": str(self.config.test_split),
                "test_split_sha256": sha256_file(self.config.test_split),
                "gpu_index": self.config.gpu_index,
                "method_order": [method for method, _ in METHODS],
                "stages": [
                    "test_shards_0_1_2_3",
                    "test_merge",
                    "final_freeze",
                    "standardize",
                    "matrix_append",
                ],
            }
            manifest_path = self.config.control_dir / "controller_manifest.json"
            if manifest_path.exists():
                previous = _json(manifest_path)
                for field in (
                    "controller_id",
                    "execution_commit",
                    "output_root",
                    "receipt_sha256",
                    "test_split_sha256",
                    "gpu_index",
                ):
                    if previous.get(field) != manifest.get(field):
                        raise BaceHeldoutCloseoutError(
                            f"controller restart identity changed: {field}"
                        )
            else:
                atomic_write_json(manifest_path, manifest)
            self._heartbeat("READY")
            self._acquire_gpu()
            for method, slug in METHODS:
                validate_selection_adoption(self.config)
                stages = build_method_stages(self.config, method, slug)
                for stage in stages:
                    self._run_stage(stage)
                self._append_matrix(method, slug, stages[-1].output)
            final_evidence = validate_selection_adoption(self.config)
            result = {
                "schema_version": SCHEMA,
                "status": "PASS",
                "controller_id": self.config.controller_id,
                "controller_pid": os.getpid(),
                "execution_commit": self.execution_commit,
                "output_root": str(self.config.output_root),
                "selection_adoption": final_evidence,
                "completed_cells": [f"BACE/{method}" for method, _ in METHODS],
                "generation_replayed": False,
                "calibration_replayed": False,
                "gnn_ablation_started": False,
                "completed_at": utc_now(),
            }
            atomic_write_json(self.config.control_dir / "terminal.json", result)
            self._heartbeat("COMPLETE")
            return result
        except Exception as exc:
            try:
                self._heartbeat("FAILED", stage=self.active_stage, detail=str(exc))
            except Exception:
                pass
            raise
        finally:
            if self.gpu_lock is not None:
                self.gpu_lock.release()
                self.gpu_lock = None
            self._release_owner()


def _config(args: argparse.Namespace) -> Config:
    controller_id = str(args.controller_id)
    if SAFE_ID.fullmatch(controller_id) is None:
        raise BaceHeldoutCloseoutError(f"unsafe controller_id: {controller_id!r}")
    if not 0 <= int(args.gpu_index) <= 3:
        raise BaceHeldoutCloseoutError("gpu-index must be in [0,3]")
    receipt_sha = str(args.expected_selection_receipt_sha256).lower()
    if SHA256.fullmatch(receipt_sha) is None:
        raise BaceHeldoutCloseoutError("expected receipt SHA must be lowercase SHA-256")
    if os.environ.get("RUN_GNN_ABLATION", "0") != "0":
        raise BaceHeldoutCloseoutError("RUN_GNN_ABLATION must remain 0")
    return Config(
        project_root=args.project_root,
        python=args.python,
        runtime_root=args.runtime_root,
        controller_id=controller_id,
        control_dir=args.control_dir,
        output_root=args.output_root,
        source_root=args.source_root,
        selection_receipt=args.selection_adoption_receipt,
        expected_receipt_sha256=receipt_sha,
        gnn_checkpoint=args.gnn_checkpoint,
        test_split=args.test_split,
        molclr_root=args.molclr_root,
        molclr_checkpoint=args.molclr_checkpoint,
        matrix_authority_state=args.matrix_authority_state,
        matrix_authority_lock=args.matrix_authority_lock,
        gpu_index=int(args.gpu_index),
        min_free_memory_mb=int(args.min_free_memory_mb),
        poll_seconds=float(args.poll_seconds),
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=lambda v: _absolute(v, existing=True), required=True)
    parser.add_argument("--python", type=lambda v: _absolute(v, existing=True), required=True)
    parser.add_argument("--runtime-root", type=lambda v: _absolute(v, existing=True), required=True)
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--control-dir", type=lambda v: _absolute(v, existing=False), required=True)
    parser.add_argument("--output-root", type=lambda v: _absolute(v, existing=False), required=True)
    parser.add_argument("--source-root", type=lambda v: _absolute(v, existing=True), required=True)
    parser.add_argument(
        "--selection-adoption-receipt",
        type=lambda v: _absolute(v, existing=True),
        required=True,
    )
    parser.add_argument("--expected-selection-receipt-sha256", required=True)
    parser.add_argument("--gnn-checkpoint", type=lambda v: _absolute(v, existing=True), required=True)
    parser.add_argument("--test-split", type=lambda v: _absolute(v, existing=True), required=True)
    parser.add_argument("--molclr-root", type=lambda v: _absolute(v, existing=True), required=True)
    parser.add_argument(
        "--molclr-checkpoint", type=lambda v: _absolute(v, existing=True), required=True
    )
    parser.add_argument(
        "--matrix-authority-state", type=lambda v: _absolute(v, existing=True), required=True
    )
    parser.add_argument(
        "--matrix-authority-lock", type=lambda v: _absolute(v, existing=False), required=True
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--min-free-memory-mb", type=int, default=16000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        _common(sub.add_parser(command))
    status = sub.add_parser("status")
    status.add_argument("--control-dir", type=lambda v: _absolute(v, existing=True), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config not in (None, "configs/hpc.yaml"):
        raise SystemExit("--config must be configs/hpc.yaml")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise SystemExit("unsupported --set override")
    if args.command == "status":
        payload = _json(args.control_dir / "heartbeat.json")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    config = _config(args)
    commit = _worktree_commit(config.project_root)
    evidence = validate_selection_adoption(config)
    if args.command == "preflight":
        result = {
            **evidence,
            "execution_commit": commit,
            "controller_id": config.controller_id,
            "control_dir": str(config.control_dir),
            "output_root": str(config.output_root),
            "gpu_index": config.gpu_index,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        print("[BACE_HELDOUT_CLOSEOUT_SUCCESSOR_PREFLIGHT_PASS]")
        return 0
    result = Runner(config).run()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("[BACE_GLOBALGCE_PASS]", flush=True)
    print("[BACE_COMRECGC_PASS]", flush=True)
    print("[BACE_HELDOUT_CLOSEOUT_SUCCESSOR_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
