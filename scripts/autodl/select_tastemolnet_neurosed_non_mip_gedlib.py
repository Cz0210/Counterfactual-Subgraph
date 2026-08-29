#!/usr/bin/env python3
"""Run the bounded deterministic non-MIP GEDLIB canary and fix the pair budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.tastemolnet_neurosed_non_mip import (  # noqa: E402
    NonMIPGEDLIBSelectionError,
    build_candidate_report,
    select_non_mip_backend,
    validate_non_mip_selection_manifest,
)
from src.eval.tastemolnet_neurosed_fixed_budget import (  # noqa: E402
    validate_benchmark_report,
)
from src.utils.tastemolnet_neurosed_gedlib_build import (  # noqa: E402
    BUILD_SCHEMA,
    GED_LABEL_BACKEND_VARIANT,
    NON_MIP_METHOD_CONFIGS,
    PINNED_GEDLIB_COMMIT,
    sha256_file,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise NonMIPGEDLIBSelectionError(f"{path} is not one JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if type(value) is not dict:
                raise NonMIPGEDLIBSelectionError(
                    f"{path}:{line_number} is not one JSON object"
                )
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _run_bounded_process(
    command: list[str],
    *,
    cwd: Path,
    deadline: float,
) -> dict[str, Any]:
    """Run one owned process group and reap it at the absolute deadline."""

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return {"returncode": 124, "stdout": "", "stderr": "deadline expired"}
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=remaining)
        return {
            "returncode": int(process.returncode),
            "stdout": stdout,
            "stderr": stderr,
        }
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate(timeout=5)
        return {
            "returncode": 124,
            "stdout": stdout,
            "stderr": stderr + "\nselector absolute deadline exceeded",
        }


def _validate_build(path: Path) -> dict[str, Any]:
    build = _load_json(path)
    smoke = build.get("smoke")
    dependencies = build.get("dependencies")
    if (
        build.get("schema_version") != BUILD_SCHEMA
        or build.get("status") != "PASS"
        or build.get("GED_LABEL_BACKEND_VARIANT") != GED_LABEL_BACKEND_VARIANT
        or build.get("F2_BLP_USED") is not False
        or build.get("GUROBI_USED") is not False
        or build.get("ged_label_backend_variant") != GED_LABEL_BACKEND_VARIANT
        or build.get("f2_blp_used") is not False
        or build.get("gurobi_used") is not False
        or build.get("candidate_ged_backends") != list(NON_MIP_METHOD_CONFIGS)
        or type(smoke) is not dict
        or smoke.get("candidate_methods") != list(NON_MIP_METHOD_CONFIGS)
        or type(dependencies) is not dict
        or dependencies.get("gedlib_commit") != PINNED_GEDLIB_COMMIT
    ):
        raise NonMIPGEDLIBSelectionError("non-MIP GEDLIB build is not PASS")
    module = Path(str(smoke.get("module_path") or ""))
    if not module.is_file() or sha256_file(module) != smoke.get("module_sha256"):
        raise NonMIPGEDLIBSelectionError("built pyged module changed")
    return build


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _normalized_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": row.get("pair_id"),
            "status": row.get("status"),
            "lower_bound": row.get("lower_bound"),
            "upper_bound": row.get("upper_bound"),
        }
        for row in rows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--pair-sampler-manifest", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--graph-inventory-jsonl", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--candidate-hard-wall-seconds", type=float, default=290.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.candidate_hard_wall_seconds <= 300:
        raise NonMIPGEDLIBSelectionError(
            "each replay hard wall must be in (0,300] seconds"
        )
    if args.output_dir.exists():
        raise NonMIPGEDLIBSelectionError("selector output directory already exists")
    args.output_dir.mkdir(parents=True, mode=0o700)
    build = _validate_build(args.build_manifest)
    pair_manifest = _load_json(args.pair_sampler_manifest)
    pair_rows = _load_jsonl(args.pairs_jsonl)
    pair_ids = [str(row.get("pair_id") or "") for row in pair_rows]
    if (
        pair_manifest.get("pair_sampling_seed") != 7
        or pair_manifest.get("split") != "train"
        or pair_manifest.get("pair_count") != 100
        or pair_manifest.get("independent_query_target_pairs") is not True
        or pair_manifest.get("parent_own_subgraph_shortcut") is not False
        or pair_manifest.get("calibration_loaded") is not False
        or pair_manifest.get("test_loaded") is not False
        or len(pair_ids) != 100
        or len(set(pair_ids)) != 100
    ):
        raise NonMIPGEDLIBSelectionError("real 100-pair train cohort changed")

    reports: dict[str, dict[str, Any]] = {}
    candidate_failures: dict[str, dict[str, Any]] = {}
    selection_started = time.monotonic()
    global_deadline = selection_started + 1800.0
    benchmark_script = REPO_ROOT / "scripts/autodl/benchmark_tastemolnet_neurosed_gedlib.py"
    for method in NON_MIP_METHOD_CONFIGS:
        candidate_started = time.monotonic()
        candidate_deadline = min(candidate_started + 600.0, global_deadline)
        observations_by_replay: list[list[dict[str, Any]]] = []
        replay_walls: list[float] = []
        replay_artifacts: list[dict[str, Any]] = []
        candidate_error: str | None = None
        for replay in (1, 2):
            remaining = min(
                args.candidate_hard_wall_seconds,
                candidate_deadline - time.monotonic(),
                global_deadline - time.monotonic(),
            )
            if remaining <= 10.0:
                candidate_error = "candidate/global absolute deadline exhausted"
                break
            replay_root = args.output_dir / "candidates" / method / f"replay-{replay}"
            inner_hard_wall = max(1.0, remaining - 10.0)
            command = [
                sys.executable,
                "-B",
                str(benchmark_script),
                "--build-manifest",
                str(args.build_manifest),
                "--pair-sampler-manifest",
                str(args.pair_sampler_manifest),
                "--pairs-jsonl",
                str(args.pairs_jsonl),
                "--graph-inventory-jsonl",
                str(args.graph_inventory_jsonl),
                "--benchmark-budget",
                "100",
                "--workers",
                str(args.workers),
                "--method",
                method,
                "--hard-wall-seconds",
                str(inner_hard_wall),
                "--output-dir",
                str(replay_root),
            ]
            replay_started = time.monotonic()
            completed = _run_bounded_process(
                command,
                cwd=REPO_ROOT,
                deadline=replay_started + remaining,
            )
            replay_elapsed = time.monotonic() - replay_started
            observations_path = replay_root / "gedlib_benchmark_100_observations.jsonl"
            report_path = replay_root / "gedlib_benchmark_100.json"
            if (
                completed["returncode"] not in (0, 78)
                or not observations_path.is_file()
                or not report_path.is_file()
            ):
                candidate_error = (
                    f"replay={replay} rc={completed['returncode']} "
                    f"stderr={completed['stderr'][-1000:]}"
                )
                break
            observations = _load_jsonl(observations_path)
            benchmark = validate_benchmark_report(
                _load_json(report_path), expected_budget=100
            )
            if (
                benchmark.get("ged_method") != method
                or benchmark.get("ged_method_args")
                != NON_MIP_METHOD_CONFIGS[method]
                or benchmark.get("pair_ids") != pair_ids
                or benchmark.get("pyged_module_sha256")
                != build["smoke"]["module_sha256"]
                or benchmark.get("gedlib_commit")
                != build["dependencies"]["gedlib_commit"]
            ):
                candidate_error = f"replay={replay} benchmark authority changed"
                break
            observations_by_replay.append(observations)
            replay_walls.append(replay_elapsed)
            replay_artifacts.append(
                {
                    "replay_index": replay,
                    "method": method,
                    "method_args": NON_MIP_METHOD_CONFIGS[method],
                    "observations_path": str(observations_path.resolve()),
                    "observations_sha256": sha256_file(observations_path),
                    "benchmark_report_path": str(report_path.resolve()),
                    "benchmark_report_sha256": sha256_file(report_path),
                    "benchmark_status": benchmark["status"],
                    "pair_ids_sha256": benchmark["pair_ids_sha256"],
                    "outcome_sha256": _stable_sha256(
                        _normalized_outcomes(observations)
                    ),
                    "successful_pair_count": benchmark[
                        "successful_pair_count"
                    ],
                    "selector_observed_wall_seconds": replay_elapsed,
                    "benchmark_wall_seconds": benchmark["wall_seconds"],
                    "pyged_module_sha256": benchmark["pyged_module_sha256"],
                    "gedlib_commit": benchmark["gedlib_commit"],
                }
            )
        if candidate_error is not None:
            candidate_failures[method] = {
                "status": "INELIGIBLE_BOUNDED_FAILURE",
                "error": candidate_error,
                "elapsed_seconds": time.monotonic() - candidate_started,
            }
            continue
        reports[method] = build_candidate_report(
            method=method,
            method_args=NON_MIP_METHOD_CONFIGS[method],
            pair_ids=pair_ids,
            replay_observations=observations_by_replay,
            replay_wall_seconds=replay_walls,
            replay_artifacts=replay_artifacts,
        )
    if time.monotonic() > global_deadline:
        raise NonMIPGEDLIBSelectionError("selection exceeded the thirty-minute budget")
    selection = select_non_mip_backend(
        reports,
        worker_count=args.workers,
        pyged_module_sha256=str(build["smoke"]["module_sha256"]),
        gedlib_commit=str(build["dependencies"]["gedlib_commit"]),
    )
    selection["bounded_candidate_failures"] = candidate_failures
    selection["build_manifest_path"] = str(args.build_manifest.resolve())
    selection["build_manifest_sha256"] = sha256_file(args.build_manifest)
    selection["pair_sampler_manifest_path"] = str(
        args.pair_sampler_manifest.resolve()
    )
    selection["pair_sampler_manifest_sha256"] = sha256_file(
        args.pair_sampler_manifest
    )
    selection["pairs_jsonl_sha256"] = sha256_file(args.pairs_jsonl)
    selection["pairs_jsonl_path"] = str(args.pairs_jsonl.resolve())
    selection["graph_inventory_sha256"] = sha256_file(
        args.graph_inventory_jsonl
    )
    selection["graph_inventory_path"] = str(
        args.graph_inventory_jsonl.resolve()
    )
    selection["selection_sha256"] = None
    selection_without_hash = dict(selection)
    selection_without_hash.pop("selection_sha256")
    selection["selection_sha256"] = _stable_sha256(selection_without_hash)
    validate_non_mip_selection_manifest(selection, reopen_artifacts=True)
    _atomic_json(args.output_dir / "non_mip_gedlib_selection.json", selection)
    print(selection["marker"])
    print(
        "selected_ged_backend=" + str(selection["selected_ged_backend"]),
        flush=True,
    )
    print(
        "selected_neurosed_pair_budget="
        f"{selection['selected_neurosed_train_pair_budget']}/"
        f"{selection['selected_neurosed_validation_pair_budget']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
